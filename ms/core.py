"""Setup, the models, the video table and the bucket. Everything else imports from here."""
import os, io, json, shutil, tempfile, threading, warnings
from pathlib import Path

import numpy as np

from .config import *

REPO = Path(__file__).resolve().parent.parent
PRELOADED = REPO / "data"                     # the three demo videos, already indexed
IN_COLAB = Path("/content").exists()
WORK = Path("/content/momentsearch_data") if IN_COLAB else Path(tempfile.gettempdir()) / "momentsearch_data"

os.environ.update({"HF_HUB_DISABLE_PROGRESS_BARS": "1", "HF_HUB_DISABLE_SYMLINKS_WARNING": "1",
                   "TRANSFORMERS_VERBOSITY": "error", "TOKENIZERS_PARALLELISM": "false"})
warnings.filterwarnings("ignore", message=".*Payload indexes have no effect.*")
warnings.filterwarnings("ignore", message=".*Local mode performs exact.*")
warnings.filterwarnings("ignore", message=".*Cannot enable progress bars.*")


class _State:
    qdrant = openai = clip = bm25 = reranker = None
    device = "cpu"
    text_dim = image_dim = None


S = _State()
VIDEOS = {}                                   # the video table (Postgres in the product)
_clip_lock, _bm25_lock = threading.Lock(), threading.Lock()


def secret(name):
    """A Colab secret, else an environment variable, else a line in .env at the repo root."""
    value = ""
    try:
        from google.colab import userdata
        value = userdata.get(name) or ""
    except Exception:
        pass
    value = value or os.environ.get(name, "")
    env = REPO / ".env"
    if not value and env.exists():
        for line in env.read_text().splitlines():
            k, _, v = line.partition("=")
            if k.strip() == name:
                value = v.strip().strip('"').strip("'")
    return value.strip()


def setup(fresh=False):
    """Copy the preloaded index into a work folder, connect Qdrant and OpenAI, load the free models."""
    from qdrant_client import QdrantClient
    from openai import OpenAI

    key = secret("OPENAI_API_KEY")
    assert key, ("Add OPENAI_API_KEY in Colab secrets (key icon, left sidebar) and switch on Notebook access. "
                 "On your own machine, put OPENAI_API_KEY=... in a .env file at the repo root.")

    if fresh and S.qdrant is not None:
        S.qdrant.close()
        S.qdrant = None
    if fresh or not (WORK / "qdrant").exists():
        shutil.rmtree(WORK, ignore_errors=True)
        if (PRELOADED / "qdrant").exists():
            shutil.copytree(PRELOADED, WORK)
        WORK.mkdir(parents=True, exist_ok=True)
    if S.qdrant is None:                      # local Qdrant allows one client per folder
        S.qdrant = QdrantClient(path=str(WORK / "qdrant"))
    S.openai = OpenAI(api_key=key)

    VIDEOS.clear()
    if (WORK / "library.json").exists():
        VIDEOS.update(json.loads((WORK / "library.json").read_text()))

    S.text_dim = embed_texts(["hello"]).shape[1]
    chat([{"role": "user", "content": "Reply with OK."}], max_tokens=16)
    load_models()

    print(f"OpenAI key works ({TEXT_EMBED_MODEL}, {LLM_MODEL}).")
    print(f"CLIP on {S.device}." + ("" if S.device == "cuda" else
          " No GPU: asking works fine, adding your own video takes a few minutes. "
          "Runtime > Change runtime type > T4 GPU makes it seconds."))
    print(f"Index: {WORK}")
    for vid, row in VIDEOS.items():
        print(f"  {vid:22s} {row['title']}: {row['frame_count']} frames, {row['chunks']} transcript chunks")


def load_models():
    if S.clip is None:
        import torch, logging
        from sentence_transformers import SentenceTransformer
        try:
            import transformers
            transformers.utils.logging.disable_progress_bar()
            logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        except Exception:
            pass
        S.device = "cuda" if torch.cuda.is_available() else "cpu"
        S.clip = SentenceTransformer(CLIP_MODEL, device=S.device)
        S.image_dim = len(embed_clip_text("hello"))
    if S.bm25 is None:
        from fastembed import SparseTextEmbedding
        S.bm25 = SparseTextEmbedding(SPARSE_MODEL)
    if S.reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder
        S.reranker = TextCrossEncoder(model_name=RERANK_MODEL)


def save_library():
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "library.json").write_text(json.dumps(VIDEOS, indent=1))


# ── vectors ──────────────────────────────────────────────────────────────────
def normalize(vecs):
    vecs = np.asarray(vecs, dtype=np.float32)
    return vecs / np.maximum(np.linalg.norm(vecs, axis=-1, keepdims=True), 1e-12)


def embed_texts(texts):
    out = []
    for i in range(0, len(texts), 64):
        resp = S.openai.embeddings.create(model=TEXT_EMBED_MODEL, input=texts[i:i + 64])
        out += [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
    return normalize(out)


def embed_images(jpegs):
    from PIL import Image
    images = [Image.open(io.BytesIO(j)).convert("RGB") for j in jpegs]
    with _clip_lock:
        return normalize(S.clip.encode(images, batch_size=32, convert_to_numpy=True, show_progress_bar=False))


def embed_clip_text(text):
    with _clip_lock:
        return normalize(S.clip.encode([text], convert_to_numpy=True, show_progress_bar=False))[0]


def sparse_docs(texts):
    with _bm25_lock:
        return [(e.indices.tolist(), e.values.tolist()) for e in S.bm25.embed(texts)]


def sparse_query(text):
    with _bm25_lock:
        e = next(iter(S.bm25.query_embed([text])))
    return e.indices.tolist(), e.values.tolist()


def chat(messages, max_tokens=LLM_MAX_TOKENS):
    kwargs = {"model": LLM_MODEL, "messages": messages, "temperature": 0.2, "max_tokens": max_tokens}
    for _ in range(3):
        try:
            resp = S.openai.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            msg = str(exc).lower()
            if "max_completion_tokens" in msg and "max_tokens" in kwargs:
                kwargs["max_completion_tokens"] = kwargs.pop("max_tokens")
            elif "temperature" in msg and "temperature" in kwargs:
                kwargs.pop("temperature")
            else:
                raise
    raise RuntimeError("chat call kept failing")


# ── the bucket (GCS / S3 / R2 in the product) ───────────────────────────────
def store():                  return WORK / "objects"
def video_prefix(vid):        return f"{USER_ID}/{vid}/"
def frame_key(vid, i):        return f"{video_prefix(vid)}frames/{i:06d}.jpg"
def transcript_key(vid):      return f"{video_prefix(vid)}transcript.json"
def get_bytes(key):           return (store() / key).read_bytes()
def delete_prefix(prefix):    shutil.rmtree(store() / prefix, ignore_errors=True)


def put_bytes(key, data):
    p = store() / key
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)


def mmss(sec):
    s = int(sec)
    return f"{s // 60:02d}:{s % 60:02d}"


def short(text, n=100):
    text = " ".join(str(text or "").split())
    return text if len(text) <= n else text[:n - 1] + "…"
