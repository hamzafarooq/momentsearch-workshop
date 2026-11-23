"""Finding moments: three searches, merge the two transcript lists, group into moments, rerank, check 1.
The from-scratch notebook (Part 5.1 to 5.5) builds each step one cell at a time."""
import math, json
from concurrent.futures import ThreadPoolExecutor

from .core import *
from .store import owner_filter


def search_points(collection, query, video_ids=None, limit=BRANCH_TOP_K, using=None):
    from qdrant_client import models as qm
    try:
        hits = S.qdrant.query_points(
            collection_name=collection, query=query, using=using, limit=limit, with_payload=True,
            query_filter=owner_filter(video_ids),
            search_params=qm.SearchParams(quantization=qm.QuantizationSearchParams(rescore=True))
            if (using is None and QDRANT_QUANTIZATION) else None).points
    except Exception as exc:
        if "not found" in str(exc).lower() or "doesn't exist" in str(exc).lower():
            return []
        raise
    return [{"score": float(h.score), **(h.payload or {})} for h in hits]


def search_three(question, video_ids=None, use_frames=True, use_text=True):
    """Pictures (CLIP), meaning (OpenAI), exact words (BM25), in parallel. A switched-off branch returns []."""
    from qdrant_client import models as qm

    def pictures():
        return search_points(IMAGE_COLLECTION, embed_clip_text(question).tolist(), video_ids) if use_frames else []

    def meaning():
        return search_points(TEXT_COLLECTION, embed_texts([question])[0].tolist(), video_ids) if use_text else []

    def words():
        if not use_text:
            return []
        idx, val = sparse_query(question)
        return search_points(TEXT_COLLECTION, qm.SparseVector(indices=idx, values=val), video_ids, using=SPARSE_VECTOR)

    with ThreadPoolExecutor(max_workers=3) as ex:
        return [f.result() for f in [ex.submit(pictures), ex.submit(meaning), ex.submit(words)]]


def merge_text(dense, sparse):
    """Reciprocal Rank Fusion: each list gives a chunk 1 / (60 + rank) points."""
    if not sparse:
        return dense
    if not dense:
        return [{**h, "sparse_score": h["score"], "score": 0.0} for h in sparse][:BRANCH_TOP_K]
    acc = {}
    for rank, h in enumerate(dense):
        acc[(h["video_id"], h["ms"])] = {"hit": dict(h), "rrf": 1 / (RRF_K + rank)}
    for rank, h in enumerate(sparse):
        k = (h["video_id"], h["ms"])
        if k in acc:
            acc[k]["rrf"] += 1 / (RRF_K + rank)
            acc[k]["hit"]["sparse_score"] = h["score"]
        else:
            acc[k] = {"hit": {**h, "sparse_score": h["score"], "score": 0.0}, "rrf": 1 / (RRF_K + rank)}
    return [x["hit"] for x in sorted(acc.values(), key=lambda x: -x["rrf"])][:BRANCH_TOP_K]


def fuse(visual_hits, text_hits):
    """Hits from the same video within 15 s are one moment. Found by picture AND speech: x1.5."""
    ranked = [{**h, "modality": "frame", "rrf": 1 / (RRF_K + r), "t": h["ms"] / 1000} for r, h in enumerate(visual_hits)]
    ranked += [{**h, "modality": "text", "rrf": 1 / (RRF_K + r), "t": h["t_start"]} for r, h in enumerate(text_hits)]
    windows = []
    for h in sorted(ranked, key=lambda x: -x["rrf"]):
        w = next((w for w in windows if w["video_id"] == h["video_id"] and abs(w["t"] - h["t"]) <= FUSION_WINDOW_S), None)
        if w is None:
            w = {"video_id": h["video_id"], "t": h["t"], "modalities": set(), "frame": None, "text": None}
            windows.append(w)
        w["modalities"].add(h["modality"])
        w[h["modality"]] = w[h["modality"]] or h
    for w in windows:
        w["rrf"] = (w["frame"]["rrf"] if w["frame"] else 0) + (w["text"]["rrf"] if w["text"] else 0)
        if len(w["modalities"]) == 2:
            w["rrf"] *= CROSS_MODAL_BOOST
    return sorted(windows, key=lambda w: -w["rrf"])


def rerank(question, windows):
    """A cross-encoder reads question + transcript together. Final = 70% reranker + 30% fused rank."""
    windows = [dict(w) for w in windows]
    cand = [i for i, w in enumerate(windows) if (w.get("text") or {}).get("text")][:RERANK_TOP_K]
    if len(windows) < 2 or len(cand) < 2:
        for w in windows:
            w["blend"] = w["rrf"]
        return windows
    raw = list(S.reranker.rerank(question, [windows[i]["text"]["text"] for i in cand]))
    rel = {i: 1 / (1 + math.exp(-s)) for i, s in zip(cand, raw)}
    top_rrf = max(w["rrf"] for w in windows) or 1
    for i, w in enumerate(windows):
        rr = w["rrf"] / top_rrf
        if i in rel:
            w["rerank"] = round(rel[i], 4)
            w["blend"] = RERANK_WEIGHT * rel[i] + (1 - RERANK_WEIGHT) * rr
        else:
            vis = (w.get("frame") or {}).get("score", 0)
            w["blend"] = rr * max(0, min(1, (vis - CONFIDENCE_THRESHOLD) / (VISUAL_STRONG - CONFIDENCE_THRESHOLD)))
    return sorted(windows, key=lambda w: -w["blend"])


def gate_ok(best_visual, best_text):
    """Check 1: is the best match good enough to bother the LLM?"""
    return best_visual >= CONFIDENCE_THRESHOLD or best_text >= TEXT_CONFIDENCE_THRESHOLD


def frame_times(vid):
    pts, _ = S.qdrant.scroll(IMAGE_COLLECTION, scroll_filter=owner_filter([vid]), limit=10_000,
                             with_payload=["idx", "ms"], with_vectors=False)
    return [(p.payload["idx"], p.payload["ms"]) for p in pts]


def match_pct(w):
    clamp = lambda v, lo, hi: max(0.0, min(1.0, (v - lo) / (hi - lo)))
    s = []
    if w.get("frame"):
        s.append(clamp(w["frame"]["score"], CONFIDENCE_THRESHOLD, VISUAL_STRONG))
    if w.get("text"):
        s.append(w["rerank"] if "rerank" in w else clamp(w["text"]["score"], TEXT_CONFIDENCE_THRESHOLD, TEXT_STRONG))
    return int(round(50 + max(s or [0]) * 50))


def retrieve(question, video_ids=None, top_k=TOP_K, use_frames=True, use_text=True, trace=False):
    """The whole search side. Returns the top moments as numbered citations, plus the best raw scores for check 1.
    trace=True also returns every intermediate list, so you can look inside."""
    vhits, dhits, shits = search_three(question, video_ids, use_frames, use_text)
    thits = merge_text(dhits, shits)
    fused = fuse(vhits, thits)
    ranked = rerank(question, fused)
    citations, nearest = [], {}
    for n, w in enumerate(ranked[:top_k], 1):
        vid, fr, tx = w["video_id"], w["frame"], w["text"]
        ms = int(fr["ms"]) if fr else int(w["t"] * 1000)
        idx = fr["idx"] if fr else None
        preview = idx
        if idx is None:
            nearest.setdefault(vid, frame_times(vid))
            preview = min(nearest[vid], key=lambda t: abs(t[1] - ms))[0] if nearest[vid] else None
        row = VIDEOS.get(vid, {})
        citations.append({
            "n": n, "video_id": vid, "title": row.get("title") or vid, "ms": ms, "timestamp": mmss(ms / 1000),
            "idx": idx, "preview_idx": preview,
            "t_start": tx["t_start"] if tx else ms / 1000, "t_end": tx["t_end"] if tx else ms / 1000,
            "transcript": (tx or {}).get("text"), "speaker": (tx or {}).get("speaker"),
            "modalities": sorted(w["modalities"]), "match": match_pct(w),
            "link": f"{row['url']}&t={ms // 1000}" if row.get("source") == "youtube" else None})
    out = {"citations": citations, "candidates": len(fused),
           "best_visual": vhits[0]["score"] if vhits else 0.0,
           "best_text": dhits[0]["score"] if dhits else 0.0}
    if trace:
        out.update(pictures=vhits, meaning=dhits, words=shits, merged_text=thits, moments=fused, reranked=ranked)
    return out


def transcript_of(vid):
    try:
        return json.loads(get_bytes(transcript_key(vid)))
    except Exception:
        return []
