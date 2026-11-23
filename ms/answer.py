"""Writing the answer: what the LLM sees, the prompt (check 2), citation cleanup (check 3), multi-part questions.
The from-scratch notebook (Part 5.6 to 5.9) builds each step one cell at a time."""
import io, re, json, base64
from concurrent.futures import ThreadPoolExecutor

from .core import *
from .search import retrieve, gate_ok, transcript_of

ABSTAIN = ("I couldn't find that in your videos: nothing indexed looks related to the question "
           "(neither what's on screen nor what's said).")

SYSTEM = (
    "You answer a user's question about a video using the numbered moments "
    "provided as your evidence. Each moment has a timestamp and may include a "
    "video FRAME (what was shown on screen) and/or a TRANSCRIPT excerpt (what was "
    "said out loud). Use BOTH kinds of evidence: for a question about what someone "
    "SAID or talked about, read the transcript text; for a question about what is "
    "SHOWN, read the frame. A moment may also carry CONTEXT — what was said just "
    "before/after the matched excerpt, or while a frame was on screen. Treat context "
    "as part of that moment: use it, and cite the same [n].\n"
    "Rules:\n"
    "1. Read the question carefully and answer exactly what is asked. Start with a "
    "one-line direct answer, then explain in short paragraphs — ONE paragraph per "
    "distinct point. Keep it focused, don't pad. No preamble, don't restate the "
    "question.\n"
    "2. Cite a moment [n] ONLY when that moment's frame or transcript actually "
    "supports the specific sentence. NEVER attach a citation to a moment that "
    "doesn't contain what you're claiming — a wrong citation is worse than none. A "
    "general statement of what the video is broadly about (e.g. from its title, or "
    "the overall gist of the frames) may stay UNCITED; that's fine. When you cite, "
    "use square brackets, e.g. [1] or [2, 3]; when the question is about what was "
    "said, quote the transcript accurately — keep the wording and numbers.\n"
    "3. Group the relevant moments by the point they make:\n"
    "   - Moments that make the SAME point (especially several from the same "
    "video) belong TOGETHER in ONE paragraph, cited together, e.g. [1, 2]. Do not "
    "split one shared point across separate paragraphs.\n"
    "   - Moments that make DIFFERENT points, or come from different videos, go in "
    "SEPARATE paragraphs, each with its own citation.\n"
    "   Cover every distinct relevant point — don't merge unrelated ones and don't "
    "drop any.\n"
    "4. You MAY add a light framing from the video's title/obvious context, but the "
    "ANSWER ITSELF must come from the moments AND address the question. Never turn "
    "\"what the video is broadly about\" into the answer when the moments don't "
    "actually address what was asked (see rule 5). Do NOT invent SPECIFIC facts, "
    "quotes, numbers, opinions or claims that aren't in the moments — if a specific "
    "detail isn't shown or said in a moment, don't state it, and don't cite one.\n"
    "5. STAY TRUE TO THE QUESTION. First judge whether the moments actually ADDRESS "
    "WHAT THE USER ASKED — not merely whether they come from the video. If none of "
    "the moments address the question, reply with ONE sentence that you couldn't "
    "find it in the video(s) and STOP — do NOT substitute a general summary of the "
    "video instead. Answer only when at least one moment genuinely speaks to the "
    "question; a partial but on-topic match is still worth answering. (If the "
    "'question' is not a real question — e.g. a bare URL or gibberish — say you "
    "couldn't find an answer rather than summarizing.)\n"
    "6. SPEAKER ATTRIBUTION — only for moments tagged 'speaker: <name>'; attribute "
    "that moment's point to that exact person by name. NEVER invent a name or write "
    "a placeholder like \"Unnamed Speaker\"; don't attribute untagged moments to "
    "anyone. If (and only if) the instructions below the question ask for a "
    "\"Who said what\" table, add it as the very last thing in your answer.\n"
    "7. ANSWER COMPLETELY, AND USE THE FACTS THAT ARE THERE. Address EVERY part of "
    "the question — a multi-part question needs each part answered, each with its "
    "own citation. And do not leave specifics on the table: when a moment's "
    "transcript or frame holds an exact detail the question asks for — a number, "
    "name, definition, step, quote or example — state it explicitly and cite it. "
    "A vague or half answer while the precise fact is sitting right there in a "
    "moment is a failure, even if what you did say is correct."
)


def shrink(jpeg, max_px=LLM_IMAGE_MAX_PX):
    from PIL import Image
    img = Image.open(io.BytesIO(jpeg))
    if max(img.size) <= max_px:
        return jpeg
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=80)
    return buf.getvalue()


def context(chunks, c, pad=CONTEXT_PAD_S):
    """What was said 10 s around a moment, from the transcript copy in the bucket."""
    t0, t1 = c["t_start"], c["t_end"]
    parts = {"said just before": [], "said while this frame was on screen": [], "said just after": []}
    for ch in chunks:
        if ch["t_end"] <= t0 - pad or ch["t_start"] >= t1 + pad or ch["text"] == c.get("transcript"):
            continue
        where = ("said just before" if ch["t_end"] <= t0 else "said just after" if ch["t_start"] >= t1
                 else "said while this frame was on screen")
        parts[where].append(ch["text"])
    return [(k, " ".join(v)) for k, v in parts.items() if v]


def build_moments(citations):
    """For each citation: the frame, its transcript, and what was said around it."""
    def frame(c):
        return get_bytes(frame_key(c["video_id"], c["idx"])) if c["idx"] is not None else None
    vids = sorted({c["video_id"] for c in citations})
    with ThreadPoolExecutor(max_workers=16) as ex:
        images = list(ex.map(frame, citations))
        chunks = dict(zip(vids, ex.map(transcript_of, vids)))
    return [{"image": img, "transcript": c["transcript"], "speaker": c["speaker"], "timestamp": c["timestamp"],
             "title": c["title"], "context": context(chunks[c["video_id"]], c),
             **({"parts": c["parts"]} if c.get("parts") else {})}
            for img, c in zip(images, citations)]


def intro(question, moments, parts=None, missing=None):
    n = len(moments)
    text = (f"QUESTION: {question}\n\nAnswer this question using the {n} moments below (numbered 1 to {n}). "
            "Each has a timestamp and a video frame and/or a transcript excerpt. If the question is about what "
            "was said, use the transcript text. Give a direct answer grounded in the relevant moment(s), cited as "
            "[n]. FIRST check the moments actually ADDRESS the question — if none do, reply with ONE sentence that "
            "you couldn't find it and STOP; do NOT summarize the video instead.")
    if parts and len(parts) > 1:
        text += (f"\n\nThe question has {len(parts)} parts:\n" + "\n".join(f"  {i}) {p}" for i, p in enumerate(parts, 1))
                 + "\nEach moment below says which part(s) it was retrieved for. Answer EVERY part, in order, "
                   "one short paragraph each with its own citations.")
        gaps = "; ".join(f"{i + 1}) {parts[i]}" for i in (missing or []))
        if gaps:
            text += f"\nNo matching moments were found for: {gaps}. For each, write ONE sentence saying the videos don't cover it — do not guess."
    return text


def label(i, m):
    line = f"[{i}]" + (" (for part " + "/".join(map(str, m["parts"])) + ")" if m.get("parts") else "")
    line += f" @ {m['timestamp']} in \"{m['title']}\""
    if m.get("transcript"):
        line += f' transcript: "{m["transcript"]}"'
    if m.get("image") is None:
        line += " (transcript only, no frame)"
    for where, text in m.get("context") or []:
        line += f'\n    {where}: "{text}"'
    return line


def llm_messages(question, moments, **opts):
    content = [{"type": "text", "text": intro(question, moments, **opts)}]
    for i, m in enumerate(moments, 1):
        content.append({"type": "text", "text": label(i, m)})
        if m.get("image"):
            b64 = base64.b64encode(shrink(m["image"])).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]


CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def validate_citations(answer, n_moments):
    """Check 3: drop any [n] the model wasn't shown."""
    def fix(m):
        valid = [x for x in re.split(r"\s*,\s*", m.group(1)) if 1 <= int(x) <= n_moments]
        return f"[{', '.join(valid)}]" if valid else ""
    return CITE.sub(fix, answer)


SPLIT_SYSTEM = (
    "You prepare a user's question about video content for search.\n"
    "FIRST decide: could ONE short passage of a video (about 20 seconds) answer the whole question? That is the "
    "normal case, even for a question with two clauses: the clauses are about the SAME moment, so it is ONE search.\n"
    "Split ONLY when the parts need evidence from DIFFERENT places: different topics, different speakers, different "
    "videos, or moments far apart in time. Then each part must stand alone: replace pronouns with the names they "
    "refer to, keep the user's exact terms, and add nothing the user didn't ask.\n"
    'Reply with ONLY this JSON: {"same_passage": true|false, "parts": ["..."]}. When same_passage is true, parts '
    "holds the question unchanged as its only item.")
SPLIT_EXAMPLES = (
    'Q: What figures does Cleo cite about friendship decline, and how does Mark respond to them?\n'
    '{"same_passage": false, "parts": ["What figures does Cleo cite about friendship decline?", '
    '"How does Mark respond to the friendship decline figures?"]}\n'
    'Q: What is a Lagrange point and why does it matter?\n'
    '{"same_passage": true, "parts": ["What is a Lagrange point and why does it matter?"]}')
MULTI_SIGNAL = re.compile(r"\b(and|or|both|each|versus|vs|compare\w*|differen\w*|respectively)\b|[,;]|\?.*\S.*\?", re.I | re.S)


def split_question(question):
    if not MULTI_SIGNAL.search(question):
        return [question]
    try:
        raw = chat([{"role": "system", "content": SPLIT_SYSTEM},
                    {"role": "user", "content": f"Examples:\n{SPLIT_EXAMPLES}\n\nAt most {MULTI_QUERY_MAX_PARTS} parts.\nQ: {question}"}],
                   max_tokens=1500)
        data = json.loads(re.search(r"\{.*\}", raw, re.S).group(0))
        if data.get("same_passage") is True:
            return [question]
        parts = []
        for p in data.get("parts") or []:
            p = " ".join(str(p).split())
            if p and p.lower() not in (x.lower() for x in parts) and len(p) <= 2 * len(question) + 40:
                parts.append(p)
        return parts[:MULTI_QUERY_MAX_PARTS] if len(parts) >= 2 else [question]
    except Exception:
        return [question]


def merge_parts(results, found):
    out = []
    for pi in found:
        for c in results[pi]["citations"]:
            dup = next((o for o in out if o["video_id"] == c["video_id"] and abs(o["ms"] - c["ms"]) <= FUSION_WINDOW_S * 1000), None)
            if dup:
                dup["parts"] = sorted(set(dup["parts"] + [pi + 1]))
            else:
                out.append({**c, "parts": [pi + 1]})
    for i, c in enumerate(out, 1):
        c["n"] = i
    return out


def ask_engine(question, video_ids=None, use_frames=True, use_text=True):
    """The whole read path: search (and split, at the same time), check 1, LLM, check 3."""
    with ThreadPoolExecutor(max_workers=2) as ex:
        f_ret = ex.submit(retrieve, question, video_ids, TOP_K, use_frames, use_text)
        f_split = ex.submit(split_question, question)
        r, parts = f_ret.result(), f_split.result()
    base = {"question": question, "parts": parts, "best_visual": r["best_visual"], "best_text": r["best_text"]}
    opts = {}
    if len(parts) > 1:
        per_k = max(3, MULTI_QUERY_TOTAL_K // len(parts))
        with ThreadPoolExecutor(max_workers=len(parts)) as ex:
            results = list(ex.map(lambda p: retrieve(p, video_ids, per_k, use_frames, use_text), parts))
        found = [i for i, x in enumerate(results) if x["citations"] and gate_ok(x["best_visual"], x["best_text"])]
        if not found:
            return {**base, "answer": ABSTAIN, "citations": [], "llm_used": False}
        citations = merge_parts(results, found)
        opts = {"parts": parts, "missing": [i for i in range(len(parts)) if i not in found]}
    else:
        if not r["citations"]:
            return {**base, "citations": [], "llm_used": False,
                    "answer": ABSTAIN if r["candidates"] else "Nothing is indexed yet. Add a video first."}
        if not gate_ok(r["best_visual"], r["best_text"]):
            return {**base, "answer": ABSTAIN, "citations": [], "llm_used": False}
        citations = r["citations"]
    moments = build_moments(citations)
    answer = validate_citations(chat(llm_messages(question, moments, **opts)), len(moments))
    if not CITE.search(answer):
        citations = []
    return {**base, "answer": answer, "citations": citations, "llm_used": True}
