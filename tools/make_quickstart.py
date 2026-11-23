"""Writes quickstart.ipynb. Edit the cells here, then run from the repo root: python tools/make_quickstart.py"""
import nbformat as nbf
M, C = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
SCRATCH = "https://colab.research.google.com/github/hamzafarooq/momentsearch-workshop/blob/main/momentsearch_workshop.ipynb"
cells = [
M(f"""# MomentSearch quickstart

Ask a question about a video and get back the exact moments that answer it: what was **said**, and what was **on screen**.

Three course videos are already indexed for you, so you can ask right away. Then you look inside one search, measure it with an answer key, and add a video of your own.

**Before you run anything**
1. **Runtime → Change runtime type → T4 GPU.**
2. Click the key icon on the left, add `OPENAI_API_KEY`, and switch on **Notebook access**.
3. Run the cells in order.

The code lives in the `ms/` folder of the repo, one file per step. The [from-scratch notebook]({SCRATCH}) builds every one of those steps in front of you. That one is your homework."""),
C("""# On Colab: get the code and the preloaded index, and install (about a minute).
# On your own machine you're already in the repo with everything installed: this does nothing.
import os, subprocess
if os.path.isdir("/content"):
    REPO, BRANCH = "/content/momentsearch-workshop", "main"
    if not os.path.isdir(REPO):
        subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, "https://github.com/hamzafarooq/momentsearch-workshop", REPO], check=True)
    os.chdir(REPO)
    subprocess.run(["pip", "install", "-q", "-r", "requirements.txt"], check=True)"""),
C("""import ms

ms.setup()        # checks your key, loads the models, opens the preloaded index
ms.library()"""),

M("""## 1. Ask

`ms.ask()` searches all three videos and answers with numbered citations. Each card under the answer is one moment: the frame, the time, how well it matched, and whether the **frame**, the **text**, or both found it."""),
C("""ms.ask("Why does each turn of a long chat cost a little more than the last?")"""),
C("""# Your question about the three videos. Try one about something said, and one about something shown.
ms.ask("What does a harness add around the model?")"""),

M("""## 2. Look inside

Before any LLM sees your question, five things happen:
1. **Pictures**: CLIP puts your question and every frame in one space and finds the closest frames.
2. **Meaning**: an OpenAI vector of your question against every transcript chunk.
3. **Exact words**: BM25, for names, numbers and terms the meaning search blurs.
4. **Moments**: hits within 15 s of each other become one moment; found by a picture **and** the speech counts 1.5×. Then a small reranker reads the question and each moment's text together.
5. **Check 1**: if the best picture and the best transcript match are both weak, stop and say "couldn't find it". No LLM call.

`ms.look_inside()` shows each of these for one question."""),
C("""ms.look_inside("Why does each turn of a long chat cost a little more than the last?")"""),
M("""Now a question the videos can't answer. Watch check 1 stop it before the LLM is called."""),
C("""ms.ask("How long should I rest a steak after grilling it?")"""),
M("""**Switch a search off.** This question is about something on screen. Ask it with only the picture search, then with only the transcript search. Which one finds the moment? Why might that be, for videos where the speaker describes what's on screen?"""),
C("""q = "Show me the permission prompt that asks before deleting the build folder."
ms.ask(q, use_text=False)     # pictures only
ms.ask(q, use_frames=False)   # transcript only"""),

M("""## 3. Measure it

One good answer proves little. The repo has an **answer key**: 20 questions about the three videos where we already know the answer and where it is.

* **said**: the answer is in the speech; **shown**: on screen; **both**: needs the two; **not in the videos**: the right move is to refuse.
* **found@6**: share of answerable questions with the right moment in the top 6.
* **MRR**: 1 / the rank of the first right moment, averaged. 1.0 means it's always first.
* **refusal accuracy**: share of off-topic questions that check 1 stops.

No LLM call here, so it takes seconds. The bar was set before anyone looked: found@6 of at least 80%."""),
C("""ms.report()"""),
C("""# The same key three ways. Which kind of question does each search carry?
ms.which_branch()"""),

M("""## 4. Your turn

Add a video of your own and find out what this search is good at and where it breaks.

1. **Pick a video of 2 to 5 minutes** with speech and something to see: a talk, a demo, a lecture clip. Shorter means faster.
2. **Upload it** with the next cell (a file picker opens). A YouTube link also works sometimes, but YouTube often blocks Colab, so a file is the safe choice.
3. **Index it.** On a T4 that takes under a minute: frames, CLIP vectors, then Whisper writes the transcript.
4. **Ask three questions**, only about your video:
   * one about something **said**,
   * one about something **only on screen** (a slide, a diagram, an object),
   * one about something that is **not in it at all**.
5. **Write down** which ones worked, and why you think so. Use `ms.look_inside()` on any that surprised you."""),
C("""path = ms.upload()            # or: path = "/content/my_video.mp4", or a YouTube link"""),
C("""MY = ms.add_video(path)       # sample frames, drop repeats, CLIP, Whisper, chunks, into Qdrant
ms.library()"""),
C("""SAID      = ""    # something said in your video
SHOWN     = ""    # something only on screen
NOT_IN_IT = ""    # something your video never covers

for q in (SAID, SHOWN, NOT_IN_IT):
    ms.ask(q, video_ids=[MY])"""),
C("""# Look inside any question that surprised you
ms.look_inside(SHOWN, video_ids=[MY])"""),
M("""**Stretch: an answer key for your video.** Write down where each answer really is (start and end in seconds) and measure, like Part 3. Add a few more questions of each kind."""),
C("""MY_KEY = [
    ms.Q(SAID,      "said",  MY, (0, 20)),     # (start, end) in seconds: where the answer is
    ms.Q(SHOWN,     "shown", MY, (0, 20)),
    ms.Q(NOT_IN_IT, "not in the videos"),
]
ms.report(MY_KEY)"""),
M("""## The same search as an app

Everything above, in one screen: ask, play the moment, look inside, add a video. On Colab it shows below and also prints a public link (`…gradio.live`) you can open full screen; it works while this notebook is running. On your own machine, `python app.py` does the same."""),
C("""ms.app()"""),
M(f"""## Discuss

* Which of your three questions worked, and which search found it: pictures, transcript, or both?
* Did the off-topic question get refused at check 1, or did it reach the LLM?
* If you shipped this for your own videos, what's the first thing you would change?

**Homework:** the [from-scratch notebook]({SCRATCH}) builds every step you just used, one cell at a time, plus the Prefect pipeline and how the storage holds up at 100,000 videos."""),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
     "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}})
nbf.write(nb, "quickstart.ipynb")
print(len(cells), "cells")
