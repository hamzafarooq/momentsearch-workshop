"""Adding a video: fetch, sample frames, drop repeats, CLIP vectors, transcript, chunks, two text vectors.
The from-scratch notebook (Part 2) builds each of these steps one cell at a time."""
import io, re, json, time, shutil, hashlib, tempfile, subprocess
from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .core import *
from .store import ensure_collection, delete_video, point_id, chunk_point_id


@dataclass
class Frame:
    ms: int
    jpeg: bytes


# ── fetch ────────────────────────────────────────────────────────────────────
YT_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})")


def youtube_id(source):
    m = YT_ID.search(source) if "youtu" in source else None
    return m.group(1) if m else None


def ffprobe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                          "-show_entries", "stream=width,height,avg_frame_rate:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True)
    info = json.loads(out.stdout or "{}")
    st = (info.get("streams") or [{}])[0]
    num, den = (st.get("avg_frame_rate") or "0/1").split("/")
    return {"duration": float((info.get("format") or {}).get("duration") or 0),
            "width": st.get("width"), "height": st.get("height"),
            "fps": float(num) / float(den) if float(den) else 0.0}


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def download_youtube(url, dst_dir):
    import yt_dlp
    opts = {"format": "bestvideo[height<=480][ext=mp4]+bestaudio[ext=m4a]/best[height<=480][ext=mp4]/best[height<=480]/best",
            "outtmpl": str(Path(dst_dir) / "%(id)s.%(ext)s"), "quiet": True, "no_warnings": True,
            "noprogress": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
            return Path(ydl.prepare_filename(info)), info.get("title")
    except Exception as exc:
        raise RuntimeError("YouTube didn't let this Colab download the video (it often blocks cloud servers). "
                           "Download it yourself, upload the file, and use its path.") from exc


def resolve_path(source):
    p = Path(str(source).strip())
    for cand in (p, Path("/content") / p, REPO / "videos" / p):
        if cand.is_file():
            return cand
    raise FileNotFoundError(f"{source} not found. Upload it with ms.upload() or the Files panel, then use its path.")


# ── sample + drop repeats ────────────────────────────────────────────────────
def split_mjpeg(stream):
    frames, pos = [], 0
    while (start := stream.find(b"\xff\xd8", pos)) >= 0:
        end = stream.find(b"\xff\xd9", start + 2)
        if end < 0:
            break
        frames.append(stream[start:end + 2])
        pos = end + 2
    return frames


def run_ffmpeg(path, vf):
    err = ""
    for sync in (["-vsync", "vfr"], ["-fps_mode", "vfr"]):
        proc = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(path), "-vf", vf, *sync,
                               "-f", "image2pipe", "-vcodec", "mjpeg", "-q:v", str(THUMB_QUALITY), "-"],
                              capture_output=True)
        if proc.stdout:
            return proc.stdout
        err = proc.stderr.decode(errors="replace")
        if "nrecognized option" not in err and "not found" not in err:
            break
    last = [l for l in err.strip().splitlines() if l.strip()][-1:] or ["no output"]
    raise RuntimeError(f"ffmpeg couldn't read this file: {last[0][:200]}")


def sample_frames(path):
    """One frame every FRAME_INTERVAL_SEC (wider for long videos), shrunk, as JPEG bytes. Returns (frames, interval)."""
    interval = FRAME_INTERVAL_SEC
    duration = ffprobe(path)["duration"]
    if MAX_FRAMES and duration / interval > MAX_FRAMES:
        interval = duration / MAX_FRAMES
    jpegs = split_mjpeg(run_ffmpeg(path, f"fps=1/{interval:.6f},scale='min({THUMB_WIDTH},iw)':-2"))
    return [Frame(ms=int(round(i * interval * 1000)), jpeg=j) for i, j in enumerate(jpegs)], interval


def dhash(jpeg):
    from PIL import Image
    img = Image.open(io.BytesIO(jpeg)).convert("L").resize((9, 8), Image.LANCZOS)
    px = np.asarray(img, dtype=np.int16)
    bits = (px[:, 1:] > px[:, :-1]).flatten()
    return int(np.packbits(bits).tobytes().hex(), 16), float(px.mean())


def same(a, b):
    return (a[0] ^ b[0]).bit_count() <= DEDUP_MAX_DISTANCE and abs(a[1] - b[1]) <= 10


def dedup(frames):
    """Keep a frame only if it differs from the last kept one."""
    if len(frames) < 2:
        return frames
    kept, last = [frames[0]], dhash(frames[0].jpeg)
    for f in frames[1:]:
        h = dhash(f.jpeg)
        if not same(h, last):
            kept.append(f)
            last = h
    return kept


# ── transcript ───────────────────────────────────────────────────────────────
def file_cues(path):
    return [{"text": c["text"].strip(), "t_start": float(c["t_start"]), "t_end": float(c["t_end"])}
            for c in json.loads(Path(path).read_text(encoding="utf-8")) if (c.get("text") or "").strip()]


def whisper_segments(audio_path, offset=0.0):
    with open(audio_path, "rb") as fh:
        resp = S.openai.audio.transcriptions.create(file=fh, model=ASR_MODEL, response_format="verbose_json",
                                                    timestamp_granularities=["segment"])
    return [{"text": s.text.strip(), "t_start": s.start + offset, "t_end": s.end + offset}
            for s in resp.segments or [] if (s.text or "").strip()]


def asr_cues(path):
    """Your own file: audio out with ffmpeg, then Whisper, in 20-minute pieces. [] if there's no audio."""
    tmp = Path(tempfile.mkdtemp())

    def audio(dst, start=None, dur=None):
        cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error"]
        cmd += ["-ss", f"{start:.3f}"] if start else []
        cmd += ["-t", f"{dur:.3f}"] if dur else []
        subprocess.run(cmd + ["-i", str(path), "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k", str(dst)],
                       check=True, capture_output=True)
        return dst
    try:
        full = audio(tmp / "audio.mp3")
    except subprocess.CalledProcessError:
        return []
    try:
        if full.stat().st_size <= 24 * 1024 * 1024:
            return whisper_segments(full)
        cues, start, dur = [], 0.0, ffprobe(path)["duration"]
        while start < dur:
            cues += whisper_segments(audio(tmp / f"w{int(start)}.mp3", start, min(1200, dur - start)), offset=start)
            start += 1200
        return cues
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def chunk_cues(cues, seconds=TRANSCRIPT_CHUNK_SECONDS):
    chunks, buf, start = [], [], None
    last_end = cues[0]["t_end"] if cues else 0.0
    for c in cues:
        if start is None:
            start = c["t_start"]
        buf.append(c["text"])
        last_end = c["t_end"]
        if last_end - start >= seconds:
            chunks.append({"text": " ".join(buf), "t_start": start, "t_end": last_end})
            buf, start = [], None
    if buf:
        chunks.append({"text": " ".join(buf), "t_start": start, "t_end": last_end})
    return chunks


# ── into Qdrant ──────────────────────────────────────────────────────────────
def index_frames(vid, frames):
    from qdrant_client import models as qm
    ensure_collection(IMAGE_COLLECTION, S.image_dim)
    for start in range(0, len(frames), CLIP_BATCH):
        batch = frames[start:start + CLIP_BATCH]
        vecs = embed_images([f.jpeg for f in batch])
        S.qdrant.upsert(IMAGE_COLLECTION, wait=True, points=[
            qm.PointStruct(id=point_id(vid, start + i), vector=v.tolist(), payload={
                "user_id": USER_ID, "video_id": vid, "ms": f.ms, "idx": start + i, "modality": "frame",
                "t_start": f.ms / 1000, "t_end": f.ms / 1000, "embed_version": f"{CLIP_MODEL}-v1"})
            for i, (f, v) in enumerate(zip(batch, vecs))])
    return len(frames)


def index_transcript(vid, chunks):
    from qdrant_client import models as qm
    put_bytes(transcript_key(vid), json.dumps(chunks, ensure_ascii=False).encode())
    ensure_collection(TEXT_COLLECTION, S.text_dim, sparse=True)
    texts = [c["text"] for c in chunks]
    dense, sparse = embed_texts(texts), sparse_docs(texts)
    S.qdrant.upsert(TEXT_COLLECTION, wait=True, points=[
        qm.PointStruct(id=chunk_point_id(vid, i),
                       vector={"": d.tolist(), SPARSE_VECTOR: qm.SparseVector(indices=s[0], values=s[1])},
                       payload={"user_id": USER_ID, "video_id": vid, "modality": "text",
                                "t_start": c["t_start"], "t_end": c["t_end"], "ms": int(c["t_start"] * 1000),
                                "text": c["text"], "embed_version": f"{TEXT_EMBED_MODEL}-v1"})
        for i, (c, d, s) in enumerate(zip(chunks, dense, sparse))])
    return len(chunks)


def slug(text):
    return re.sub(r"[^a-z0-9]+", "-", str(text).lower()).strip("-")[:40] or "video"


def add_video(source, title=None, transcript=None):
    """Index one video: a file path, an uploaded file name, a YouTube link, or a demo id.
    `transcript` is an optional JSON file of [{text, t_start, t_end}]; without it, Whisper listens to the audio."""
    t0 = time.time()
    source = str(source).strip()
    tmp = Path(tempfile.mkdtemp())
    try:
        if source in DEMO_VIDEOS:
            d = DEMO_VIDEOS[source]
            vid, path, title = source, REPO / "videos" / d["video"], title or d["title"]
            transcript = transcript or REPO / "videos" / d["transcript"]
            kind, url, content_id = "demo", None, source
        elif youtube_id(source):
            yt = youtube_id(source)
            vid, kind, url, content_id = f"yt_{yt}", "youtube", f"https://www.youtube.com/watch?v={yt}", yt
            print("downloading from YouTube…")
            path, yt_title = download_youtube(url, tmp)
            title = title or yt_title or vid
        else:
            path = resolve_path(source)
            title = title or path.stem
            vid, kind, url, content_id = f"my_{slug(path.stem)}", "upload", None, sha256_file(path)

        dup = next((r for r in VIDEOS.values() if r.get("content_id") == content_id and r["status"] == "indexed"), None)
        if dup:
            print(f"already indexed as {dup['id']}: nothing to do")
            return dup["id"]

        info = ffprobe(path)
        print(f"{title}: {mmss(info['duration'])} long, {info['width']}x{info['height']}")
        delete_video(vid)
        delete_prefix(video_prefix(vid))

        t = time.time()
        sampled, interval = sample_frames(path)
        kept = dedup(sampled)
        for i, f in enumerate(kept):
            put_bytes(frame_key(vid, i), f.jpeg)
        print(f"  frames     : {int(info['duration'] * info['fps']):,} in the video → {len(sampled)} sampled "
              f"(one every {interval:.1f} s) → {len(kept)} after dropping repeats   [{time.time() - t:.0f} s]")

        t = time.time()
        n_frames = index_frames(vid, kept)
        print(f"  CLIP       : {n_frames} frame vectors into '{IMAGE_COLLECTION}' on {S.device}   [{time.time() - t:.0f} s]")

        t = time.time()
        cues, origin = (file_cues(transcript), "transcript file") if transcript else (asr_cues(path), "Whisper")
        chunks = chunk_cues(cues)
        n_chunks = index_transcript(vid, chunks) if chunks else 0
        print(f"  transcript : {len(cues)} lines ({origin}) → {n_chunks} chunks of ~{TRANSCRIPT_CHUNK_SECONDS:.0f} s, "
              f"2 vectors each (OpenAI + BM25)   [{time.time() - t:.0f} s]"
              + ("" if n_chunks else "   no speech: searchable by pictures only"))

        VIDEOS[vid] = {"id": vid, "title": title, "source": kind, "url": url, "content_id": content_id,
                       "status": "indexed", "duration": info["duration"], "sampled": len(sampled),
                       "frame_count": n_frames, "chunks": n_chunks,
                       "path": str(path) if kind == "upload" else None}
        save_library()
        print(f"done in {time.time() - t0:.0f} s. video id: {vid}")
        return vid
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def upload():
    """Colab's upload button. Returns the path of the file you picked."""
    from google.colab import files
    up = files.upload()
    if not up:
        raise RuntimeError("No file picked.")
    name = next(iter(up))
    path = Path("/content") / name
    if not path.exists():
        Path(name).replace(path)
    print(f"uploaded: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    return str(path)
