"""Every setting in one place. These are the product's defaults; the from-scratch notebook explains each one."""

USER_ID = "notebook"

CLIP_MODEL       = "clip-ViT-L-14"                   # frames + questions in one space, 768 numbers
TEXT_EMBED_MODEL = "text-embedding-3-small"          # meaning of the transcript, 1536 numbers
SPARSE_MODEL     = "Qdrant/bm25"                     # exact words
RERANK_MODEL     = "Xenova/ms-marco-MiniLM-L-6-v2"   # reads question + transcript together
LLM_MODEL        = "gpt-4.1-mini"                    # writes the answer, sees the frames
ASR_MODEL        = "whisper-1"                       # speech to text for your own files

IMAGE_COLLECTION    = "moments_l14"
TEXT_COLLECTION     = "moments_text_openai"
SPARSE_VECTOR       = "bm25"
QDRANT_ON_DISK      = True
QDRANT_QUANTIZATION = True
QDRANT_HNSW_ON_DISK = True

FRAME_INTERVAL_SEC = 2.0
MAX_FRAMES         = 400
THUMB_WIDTH        = 480
THUMB_QUALITY      = 3
DEDUP_MAX_DISTANCE = 4
CLIP_BATCH         = 128
TRANSCRIPT_CHUNK_SECONDS = 20.0
TRANSCRIPT_LANGS   = ["en", "en-US", "en-GB"]

BRANCH_TOP_K       = 20
TOP_K              = 6
RRF_K              = 60
FUSION_WINDOW_S    = 15.0
CROSS_MODAL_BOOST  = 1.5
RERANK_TOP_K       = 30
RERANK_WEIGHT      = 0.7
CONFIDENCE_THRESHOLD      = 0.2
TEXT_CONFIDENCE_THRESHOLD = 0.35
VISUAL_STRONG      = 0.45
TEXT_STRONG        = 0.65
CONTEXT_PAD_S      = 10.0
LLM_IMAGE_MAX_PX   = 512
LLM_MAX_TOKENS     = 1024
MULTI_QUERY_MAX_PARTS, MULTI_QUERY_TOTAL_K = 3, 12

DEMO_VIDEOS = {
    "demo_how-llms-work": {"title": "How LLMs work",
                           "video": "how-llms-work.mp4", "transcript": "how-llms-work.transcript.json"},
    "demo_llm-to-agent":  {"title": "What turns an LLM into an agent",
                           "video": "llm-to-agent.mp4", "transcript": "llm-to-agent.transcript.json"},
    "demo_harness":       {"title": "The harness and Claude Code",
                           "video": "harness.mp4", "transcript": "harness.transcript.json"},
}
