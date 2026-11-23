"""What you call from the notebook: ask(), look_inside(), library(), and the pictures and tables they draw."""
import time, base64

from .core import *
from .answer import ask_engine


def _display(obj):
    from IPython.display import display
    display(obj)


def img_tag(jpeg, width=150):
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(jpeg).decode()}" width="{width}">'


def strip(items, width=150):
    from IPython.display import HTML
    cells = "".join(f'<div style="margin:3px;text-align:center;font:12px monospace">{img_tag(j, width)}<br>{c}</div>'
                    for j, c in items)
    _display(HTML(f'<div style="display:flex;flex-wrap:wrap">{cells}</div>'))


def table(rows, columns=None):
    import pandas as pd
    pd.set_option("display.max_colwidth", 110)
    _display(pd.DataFrame(rows, columns=columns).style.hide(axis="index").format(precision=3))


def library():
    """Every indexed video: its id (use it in video_ids=[...]), frames, transcript chunks."""
    table([[v, r["title"], mmss(r.get("duration") or 0), r["frame_count"], r["chunks"]] for v, r in VIDEOS.items()],
          ["video_id", "title", "length", "frames", "transcript chunks"])


def show_answer(res):
    from IPython.display import HTML, Markdown
    if len(res.get("parts") or []) > 1:
        print("searched as:", " | ".join(res["parts"]))
    _display(Markdown(res["answer"]))
    if not res.get("llm_used"):
        print(f"(no LLM call: best picture {res['best_visual']:.2f}, needs {CONFIDENCE_THRESHOLD}; "
              f"best transcript {res['best_text']:.2f}, needs {TEXT_CONFIDENCE_THRESHOLD})")
    cards = []
    for c in res["citations"]:
        img = img_tag(get_bytes(frame_key(c["video_id"], c["preview_idx"])), 210) if c.get("preview_idx") is not None else ""
        when = f'<a href="{c["link"]}" target="_blank">{c["timestamp"]}</a>' if c.get("link") else c["timestamp"]
        said = f'<div style="font-size:11px;color:#666">{short(c["transcript"], 140)}</div>' if c.get("transcript") else ""
        cards.append(f'<div style="width:220px;margin:4px;font:12px sans-serif">{img}<br><b>[{c["n"]}]</b> {when} · '
                     f'{c["match"]}% · {"+".join(c["modalities"])}<br><i>{short(c["title"], 40)}</i>{said}</div>')
    if cards:
        _display(HTML(f'<div style="display:flex;flex-wrap:wrap">{"".join(cards)}</div>'))


def ask(question, video_ids=None, use_frames=True, use_text=True):
    """Ask across every indexed video, or only some: video_ids=["my_..."]. use_frames / use_text switch a search off."""
    if not str(question).strip():
        print("Write your question between the quotes first.")
        return None
    off = [n for n, on in (("pictures", use_frames), ("transcript", use_text)) if not on]
    print(f"Q: {question}   ({', '.join(video_ids) if video_ids else 'all videos'}"
          f"{'; off: ' + ', '.join(off) if off else ''})\n")
    t = time.time()
    res = ask_engine(question, video_ids, use_frames, use_text)
    show_answer(res)
    print(f"({time.time() - t:.1f} s)")
    return res


def look_inside(question, video_ids=None):
    """The search side, step by step: the three searches, the moments, the rerank, check 1. No LLM call."""
    from .search import retrieve, gate_ok
    t = time.time()
    r = retrieve(question, video_ids, trace=True)
    print(f"Q: {question}\nthree searches + merge + rerank in {time.time() - t:.2f} s\n")

    print("1. pictures (CLIP: the question and the frames in one space)")
    strip([(get_bytes(frame_key(h["video_id"], h["idx"])), f"{mmss(h['ms'] / 1000)}  {h['score']:.2f}")
           for h in r["pictures"][:5]], width=140)
    print("2. meaning (OpenAI vectors)")
    table([[short(VIDEOS.get(h["video_id"], {}).get("title", h["video_id"]), 28), mmss(h["t_start"]), round(h["score"], 3), short(h["text"], 80)]
           for h in r["meaning"][:4]], ["video", "time", "cosine", "transcript"])
    print("3. exact words (BM25)")
    table([[short(VIDEOS.get(h["video_id"], {}).get("title", h["video_id"]), 28), mmss(h["t_start"]), round(h["score"], 2), short(h["text"], 80)]
           for h in r["words"][:4]], ["video", "time", "bm25", "transcript"])

    print(f"4. grouped into moments ({len(r['pictures'])} picture hits + {len(r['merged_text'])} transcript hits "
          f"→ {len(r['moments'])} moments), then reranked")
    before = {(w["video_id"], w["t"]): i for i, w in enumerate(r["moments"], 1)}
    table([[i, before[(w["video_id"], w["t"])], short(VIDEOS.get(w["video_id"], {}).get("title", ""), 28), mmss(w["t"]),
            " + ".join(sorted(w["modalities"])), w.get("rerank", "–"), short((w["text"] or {}).get("text", "(picture only)"), 60)]
           for i, w in enumerate(r["reranked"][:TOP_K], 1)],
          ["rank", "before rerank", "video", "time", "found by", "reranker 0-1", "transcript"])

    ok = gate_ok(r["best_visual"], r["best_text"])
    print(f"5. check 1: best picture {r['best_visual']:.3f} (needs {CONFIDENCE_THRESHOLD}), "
          f"best transcript {r['best_text']:.3f} (needs {TEXT_CONFIDENCE_THRESHOLD})  →  "
          + ("good enough: these 6 moments go to the LLM" if ok else "STOP: answer 'couldn't find it', no LLM call"))
