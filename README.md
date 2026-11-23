# MomentSearch workshop: video RAG, hands on

Ask a question about a video and get the exact moments that answer it: what was **said**, and what was **on screen**.

There are two notebooks:

| | Notebook | What it is |
|---|---|---|
| **In class** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamzafarooq/momentsearch-workshop/blob/main/quickstart.ipynb) **Quickstart** | Three videos already indexed. Ask, look inside one search, measure it with an answer key, then add a video of your own. About 30 minutes. |
| **Homework** | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamzafarooq/momentsearch-workshop/blob/main/momentsearch_workshop.ipynb) **From scratch** | Every step built one cell at a time, plus the Prefect pipeline and how the storage holds up at 100,000 videos. |

## Before you run either one

1. Open it in Colab, then **File → Save a copy in Drive**, so your changes are yours.
2. **Runtime → Change runtime type → T4 GPU.** On a CPU the picture step takes minutes instead of seconds.
3. Click the key icon on the left, add `OPENAI_API_KEY`, and switch on notebook access. That's the only key you need.
4. Run the first cell. It clones this repo into `/content/momentsearch-workshop`.

## Run it on your own machine (optional)

Colab needs nothing installed, so use it in class. To run it on your laptop instead:

1. **Python 3.10 to 3.12** and **ffmpeg** (macOS: `brew install ffmpeg`; Ubuntu: `sudo apt install ffmpeg`; Windows: `winget install ffmpeg`).
2. Get the code and install (about 3 GB, mostly PyTorch and the CLIP model on first run):
   ```
   git clone https://github.com/hamzafarooq/momentsearch-workshop
   cd momentsearch-workshop
   python -m venv .venv
   source .venv/bin/activate        # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. Put your key in a file named `.env` in this folder: `OPENAI_API_KEY=sk-...` (git ignores it).
4. Run the app with `python app.py` and open the address it prints, or open `quickstart.ipynb` in VS Code or Jupyter and run the cells.

Without a GPU, asking is fast; adding a video of a few minutes takes a minute or two.

## Quickstart

| Part | What happens |
|---|---|
| 1 | Ask the three course videos anything |
| 2 | Look inside: three searches, moments, rerank, the check that refuses off-topic questions |
| 3 | Measure: an answer key, three numbers, transcript only vs pictures only vs both |
| 4 | Your turn: upload a short video and ask about something said, something shown, and something not in it |

## From scratch

| Part | What happens |
|---|---|
| 1 | Setup: keys, models, settings |
| 2 | Index a video: sample frames, drop repeats, turn pictures and speech into vectors |
| 3 | How the storage holds up at 100,000 videos |
| 4 | The same steps as a Prefect flow, with retries |
| 5 | Answer a question: three searches, merged into moments, checks so it doesn't make things up |
| 6 | Prove it: an answer key, three numbers, speech only vs frames only vs both |
| 7 | Your turn: add your own video and ask it anything |

## What's in here

- `quickstart.ipynb` and `momentsearch_workshop.ipynb`: the two notebooks
- `ms/`: the code the quickstart calls, one file per step (`ingest`, `search`, `answer`, `evals`, ...)
- `data/`: the three demo videos, already indexed (Qdrant, thumbnails, transcripts). Rebuild with `python tools/build_index.py`
- `videos/`: three videos of about four minutes each, with timed transcripts. *How LLMs work*, *What turns an LLM into an agent* and *The harness and Claude Code*, from the Forward Deployed Engineering Bootcamp by Traversaal.ai

The full product (API, web app, worker) is at [traversaal-ai/momentsearch](https://github.com/traversaal-ai/momentsearch).
