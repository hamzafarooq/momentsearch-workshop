# MomentSearch workshop: video RAG, hands on

Ask a question about a video and get the exact moments that answer it: what was **said**, and what was **on screen**. In this notebook you build that search yourself, one step at a time, on three short videos.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hamzafarooq/momentsearch-workshop/blob/main/momentsearch_workshop.ipynb)

## Before you run it

1. Click **Open in Colab** above, then **File → Save a copy in Drive**, so your changes are yours.
2. **Runtime → Change runtime type → T4 GPU.** On a CPU the picture step takes minutes instead of seconds.
3. Click the key icon on the left, add `OPENAI_API_KEY`, and switch on notebook access. That's the only key you need.
4. Run the first cell. The notebook clones this repo, so the three videos land in `/content/momentsearch-workshop/videos` on their own.

## What you'll build

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

- `momentsearch_workshop.ipynb`: the notebook
- `videos/`: three videos of about four minutes each, with timed transcripts. *How LLMs work*, *What turns an LLM into an agent* and *The harness and Claude Code*, from the Forward Deployed Engineering Bootcamp by Traversaal.ai

The full product (API, web app, worker) is at [traversaal-ai/momentsearch](https://github.com/traversaal-ai/momentsearch).
