"""The same search as a small web app: ask, look inside, add a video. ms.app() starts it."""
import time, subprocess, sys
from pathlib import Path

from .core import *
from .answer import ask_engine
from .search import retrieve, gate_ok

CSS = """
.gradio-container {max-width: 1180px !important}
#answer {font-size: 15px; line-height: 1.55}
#title h1 {margin-bottom: 0}
.frames .grid-container {grid-auto-rows: auto !important}
.frames .thumbnail-item {aspect-ratio: 16 / 9 !important; height: auto !important}
.frames .grid-wrap {height: auto !important; max-height: none !important}
"""


def _video_path(vid):
    row = VIDEOS.get(vid, {})
    if vid in DEMO_VIDEOS:
        return REPO / "videos" / DEMO_VIDEOS[vid]["video"]
    return Path(row["path"]) if row.get("path") else None


def clip(vid, sec, before=3.0, length=15.0):
    """A short clip around a moment, cut once and kept."""
    src = _video_path(vid)
    if not src or not src.exists():
        return None
    start = max(0.0, sec - before)
    out = WORK / "clips" / f"{vid}_{int(start * 10)}.mp4"
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-ss", f"{start:.2f}", "-i", str(src),
                        "-t", f"{length:.2f}", "-vf", "scale=-2:480", "-c:v", "libx264", "-preset", "veryfast",
                        "-c:a", "aac", "-movflags", "+faststart", str(out)], check=True)
    return str(out)


def _choices():
    return [(f"{r['title']} ({v})", v) for v, r in VIDEOS.items()]


def _frame(vid, idx):
    return str(store() / frame_key(vid, idx))


def _gate_line(best_v, best_t):
    ok = gate_ok(best_v, best_t)
    mark = lambda v, need: f"**{v:.2f}** {'≥' if v >= need else '<'} {need}"
    return (f"Check 1 · best picture {mark(best_v, CONFIDENCE_THRESHOLD)} · best transcript {mark(best_t, TEXT_CONFIDENCE_THRESHOLD)} → "
            + ("**good enough, the LLM answers**" if ok else "**stop, no LLM call**"))


def _ask(question, video_ids, pictures, transcript):
    if not (question or "").strip():
        return "Type a question.", "", [], [], None
    t = time.time()
    res = ask_engine(question, video_ids or None, use_frames=pictures, use_text=transcript)
    cites = res["citations"]
    gallery = [(_frame(c["video_id"], c["preview_idx"]),
                f"[{c['n']}] {c['timestamp']} · {c['match']}% · {'+'.join(c['modalities'])}")
               for c in cites if c.get("preview_idx") is not None]
    status = (f"{_gate_line(res['best_visual'], res['best_text'])} · "
              f"{'LLM: ' + LLM_MODEL if res['llm_used'] else 'no LLM call'} · {time.time() - t:.1f} s")
    first = clip(cites[0]["video_id"], cites[0]["ms"] / 1000) if cites else None
    return res["answer"], status, gallery, cites, first


def _play(cites, evt):
    if not cites or evt is None:
        return None
    c = cites[evt.index]
    return clip(c["video_id"], c["ms"] / 1000)


def _inside(question, video_ids):
    import pandas as pd
    if not (question or "").strip():
        return [], None, None, None, "Type a question."
    r = retrieve(question, video_ids or None, trace=True)
    title = lambda v: short(VIDEOS.get(v, {}).get("title", v), 30)
    pics = [(_frame(h["video_id"], h["idx"]), f"{mmss(h['ms'] / 1000)} · {h['score']:.2f}")
            for h in r["pictures"][:8]]
    meaning = pd.DataFrame([[title(h["video_id"]), mmss(h["t_start"]), round(h["score"], 3), short(h["text"], 90)]
                            for h in r["meaning"][:5]], columns=["video", "time", "cosine", "transcript"])
    words = pd.DataFrame([[title(h["video_id"]), mmss(h["t_start"]), round(h["score"], 2), short(h["text"], 90)]
                          for h in r["words"][:5]], columns=["video", "time", "BM25", "transcript"])
    before = {(w["video_id"], w["t"]): i for i, w in enumerate(r["moments"], 1)}
    moments = pd.DataFrame([[i, before[(w["video_id"], w["t"])], title(w["video_id"]), mmss(w["t"]),
                             " + ".join(sorted(w["modalities"])), w.get("rerank", "–"),
                             short((w["text"] or {}).get("text", "(picture only)"), 70)]
                            for i, w in enumerate(r["reranked"][:TOP_K], 1)],
                           columns=["rank", "was", "video", "time", "found by", "reranker 0-1", "transcript"])
    return pics, meaning, words, moments, _gate_line(r["best_visual"], r["best_text"])


def _add(file, title):
    import gradio as gr
    from .ingest import add_video
    import io, contextlib
    if not file:
        return "Pick a video file first.", gr.update()
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            add_video(file, title=(title or "").strip() or None)
    except Exception as exc:
        buf.write(f"\nFAILED: {exc}")
    return buf.getvalue(), gr.update(choices=_choices())


def _style_kwargs(fn):
    import inspect
    import gradio as gr
    return {"css": CSS, "theme": gr.themes.Soft()} if "css" in inspect.signature(fn).parameters else {}


def build():
    import gradio as gr

    def play(c, evt: gr.SelectData):
        return _play(c, evt)

    with gr.Blocks(title="MomentSearch", **_style_kwargs(gr.Blocks.__init__)) as demo:
        gr.Markdown("# MomentSearch\nAsk a question about your videos. Get the moments that answer it: what was **said**, and what was **on screen**.",
                    elem_id="title")
        videos = gr.Dropdown(_choices(), multiselect=True, label="Search in (empty = all videos)")
        with gr.Tab("Ask"):
            with gr.Row():
                q = gr.Textbox(label="Question", value="Why does each turn of a long chat cost a little more than the last?", scale=5)
                go = gr.Button("Ask", variant="primary", scale=1)
            with gr.Row():
                pictures = gr.Checkbox(True, label="Picture search")
                transcript = gr.Checkbox(True, label="Transcript search")
            status = gr.Markdown()
            with gr.Row():
                with gr.Column(scale=3):
                    answer = gr.Markdown(elem_id="answer")
                with gr.Column(scale=2):
                    player = gr.Video(label="The moment (click a card to play another)", autoplay=True, height=260)
            gallery = gr.Gallery(label="Moments (click one to play it)", columns=3, height="auto", object_fit="contain", allow_preview=False,
                                 elem_classes="frames")
            cites = gr.State([])
            for trigger in (go.click, q.submit):
                trigger(_ask, [q, videos, pictures, transcript], [answer, status, gallery, cites, player])
            gallery.select(play, [cites], player)
            gr.Examples([["What three things turn a model into an agent?"],
                         ["Show me the receipt with lines for tokens in and tokens out."],
                         ["How long should I rest a steak after grilling it?"],
                         ["What is the context window, and what does Claude Code do before it deletes a file?"]], [q])
        with gr.Tab("Look inside"):
            gr.Markdown("The search side only, no LLM: three searches, grouped into moments, reranked, then check 1.")
            with gr.Row():
                q2 = gr.Textbox(label="Question", value="Why does each turn of a long chat cost a little more than the last?", scale=5)
                go2 = gr.Button("Search", variant="primary", scale=1)
            gate = gr.Markdown()
            pics = gr.Gallery(label="1. Pictures (CLIP): closest frames and their cosine", columns=4, height="auto", object_fit="contain",
                              allow_preview=False, elem_classes="frames")
            with gr.Row():
                meaning = gr.Dataframe(label="2. Meaning (OpenAI vectors)", wrap=True)
                words = gr.Dataframe(label="3. Exact words (BM25)", wrap=True)
            moments = gr.Dataframe(label="4. Moments, after the reranker", wrap=True)
            for trigger in (go2.click, q2.submit):
                trigger(_inside, [q2, videos], [pics, meaning, words, moments, gate])
        with gr.Tab("Add a video"):
            gr.Markdown("A video of 2 to 5 minutes with speech and something to see. On a T4 it takes under a minute.")
            with gr.Row():
                file = gr.File(label="Video file", file_types=["video"], type="filepath")
                with gr.Column():
                    vtitle = gr.Textbox(label="Title (optional)")
                    add = gr.Button("Index it", variant="primary")
            log = gr.Textbox(label="What happened", lines=8)
            add.click(_add, [file, vtitle], [log, videos])
    return demo


def app(share=None):
    """Start the app. In Colab it shows inline and also prints a public link you can open full screen."""
    try:
        import gradio  # noqa: F401
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio"], check=True)
    import gradio as gr
    demo = build()
    demo.queue().launch(share=IN_COLAB if share is None else share, allowed_paths=[str(WORK), str(REPO / "videos")],
                        **_style_kwargs(gr.Blocks.launch))
    return demo
