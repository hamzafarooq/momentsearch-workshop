"""Rebuild data/: the three demo videos, indexed, so the quickstart starts with a full library.
Run from the repo root with OPENAI_API_KEY set: python tools/build_index.py"""
import shutil, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ms.core as core
from ms.config import DEMO_VIDEOS
from ms.ingest import add_video
from qdrant_client import QdrantClient
from openai import OpenAI

core.WORK = core.PRELOADED
shutil.rmtree(core.WORK, ignore_errors=True)
core.WORK.mkdir(parents=True)
core.S.qdrant = QdrantClient(path=str(core.WORK / "qdrant"))
core.S.openai = OpenAI(api_key=core.secret("OPENAI_API_KEY"))
core.S.text_dim = core.embed_texts(["hello"]).shape[1]
core.load_models()
for vid in DEMO_VIDEOS:
    add_video(vid)
core.S.qdrant.close()
