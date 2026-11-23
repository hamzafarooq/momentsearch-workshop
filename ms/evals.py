"""An answer key for the three demo videos, and the three numbers: found@6, MRR, refusal accuracy.
No LLM call: just retrieve() and check 1. The from-scratch notebook (Part 6) explains each choice."""
from .core import mmss, short, VIDEOS
from .search import retrieve, gate_ok

WINDOW_SLACK_S = 15.0
BAR = 0.80
L, A, H = "demo_how-llms-work", "demo_llm-to-agent", "demo_harness"


def Q(question, kind, video=None, *windows):
    """kind: "said", "shown", "both" or "not in the videos". windows: (start_s, end_s) where the answer is."""
    return {"question": question, "kind": kind, "video": video, "windows": list(windows)}


KEY = [
    Q("In English, how much of a word is one token?", "said", L, (42.5, 50.7)),
    Q("What happens to anything that doesn't fit in the context window?", "said", L, (75.0, 81.8)),
    Q("When should you turn the temperature up?", "said", L, (148.1, 153.4)),
    Q("What did ChatGPT do in early 2023 when you asked what was on in London?", "said", A, (0.0, 9.4)),
    Q("What three things turn a model into an agent?", "said", A, (50.8, 58.6)),
    Q("What are the steps of the agent loop?", "said", A, (162.3, 168.9), (245.8, 252.0)),
    Q("Why would an order for 250 milkshakes go to a human?", "said", A, (182.6, 194.3)),
    Q("What should you type in Claude Code between tasks to start fresh?", "said", H, (114.8, 121.1)),
    Q("Which other harnesses besides Claude Code are named?", "said", H, (214.7, 222.5)),
    Q("Show me the receipt with lines for tokens in and tokens out.", "shown", L, (166.2, 183.7)),
    Q("Find the screen where the chat is searching the web and shows event cards.", "shown", A, (11.5, 21.2)),
    Q("Show me the permission prompt that asks before deleting the build folder.", "shown", H, (121.1, 138.6)),
    Q("Where is the tip splitter app with fields for the bill, tip and people?", "shown", H, (228.0, 243.0)),
    Q("What goes into one API call, and what comes back?", "both", L, (214.0, 236.7)),
    Q("On the one-screen diagram, what sits around the model?", "both", A, (194.3, 221.5)),
    Q("Which four stations light up around the model in Claude Code's harness?", "both", H, (188.9, 214.6)),
    Q("What's a good recipe for banana bread?", "not in the videos"),
    Q("Who won the 2022 World Cup final?", "not in the videos"),
    Q("How do you fine-tune a model with LoRA?", "not in the videos"),
    Q("How do you set up autoscaling in Kubernetes?", "not in the videos"),
]


def is_hit(c, item):
    return c["video_id"] == item["video"] and any(
        c["t_start"] <= w1 + WINDOW_SLACK_S and c["t_end"] >= w0 - WINDOW_SLACK_S for w0, w1 in item["windows"])


def run_key(key=KEY, video_ids=None, use_frames=True, use_text=True):
    rows = []
    for item in key:
        r = retrieve(item["question"], video_ids=video_ids, use_frames=use_frames, use_text=use_text)
        refused = not r["citations"] or not gate_ok(r["best_visual"], r["best_text"])
        rank = None if refused else next((c["n"] for c in r["citations"] if is_hit(c, item)), None)
        rows.append({**item, "refused": refused, "rank": rank})
    return rows


def scores(rows):
    answerable = [r for r in rows if r["kind"] != "not in the videos"]
    negatives = [r for r in rows if r["kind"] == "not in the videos"]
    out = {"found@6": sum(r["rank"] is not None for r in answerable) / max(1, len(answerable)),
           "MRR": sum(1 / r["rank"] for r in answerable if r["rank"]) / max(1, len(answerable)),
           "refusal accuracy": sum(r["refused"] for r in negatives) / max(1, len(negatives))}
    for kind in ("said", "shown", "both"):
        rs = [r for r in rows if r["kind"] == kind]
        if rs:
            out[f"found@6 {kind}"] = sum(r["rank"] is not None for r in rs) / len(rs)
    return out


def result(r):
    if r["kind"] == "not in the videos":
        return "refused (right)" if r["refused"] else "not refused (wrong)"
    return "refused (wrong)" if r["refused"] else (f"hit at rank {r['rank']}" if r["rank"] else "miss")


def report(key=KEY, video_ids=None, use_frames=True, use_text=True, rows=True):
    """Run the key and print the three numbers. rows=False prints only the numbers."""
    from .show import table
    ids = video_ids or [v for v in {q["video"] for q in key if q["video"]}]
    out = run_key(key, ids, use_frames, use_text)
    if rows:
        table([[r["kind"], short(r["question"], 70), short(VIDEOS.get(r["video"], {}).get("title", ""), 30),
                ", ".join(f"{mmss(a)}-{mmss(b)}" for a, b in r["windows"]), result(r)] for r in out],
              ["kind", "question", "video", "answer is at", "result"])
    s = scores(out)
    by_kind = ", ".join(f"{k} {s['found@6 ' + k]:.0%}" for k in ("said", "shown", "both") if f"found@6 {k}" in s)
    print(f"found@6 {s['found@6']:.0%} (bar {BAR:.0%})   MRR {s['MRR']:.2f}   "
          f"refusal accuracy {s['refusal accuracy']:.0%}   by kind: {by_kind}")
    return s


def which_branch(key=KEY):
    """The same key three ways: transcript only, pictures only, both."""
    from .show import table
    ids = [v for v in {q["video"] for q in key if q["video"]}]
    runs = {"transcript only": scores(run_key(key, ids, use_frames=False)),
            "pictures only": scores(run_key(key, ids, use_text=False)),
            "both": scores(run_key(key, ids))}
    table([[name] + [f"{s.get(k, 0):.0%}" for k in ("found@6", "found@6 said", "found@6 shown", "found@6 both")]
           + [round(s["MRR"], 2), f"{s['refusal accuracy']:.0%}"] for name, s in runs.items()],
          ["search", "found@6", "said", "shown", "both", "MRR", "refusal accuracy"])
    return runs
