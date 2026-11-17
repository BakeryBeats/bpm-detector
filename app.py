from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import librosa
import tempfile
from pathlib import Path
import numpy as np
from starlette.concurrency import run_in_threadpool

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Homepage
@app.get("/")
def home():
    return FileResponse(Path(__file__).with_name("index.html"))


def detect_bpm(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True, duration=120)
    if not np.isfinite(y).all() or len(y) < sr * 4:
        raise ValueError("Use an audio file with at least 4 seconds of music.")
    if np.max(np.abs(y)) < 1e-5:
        raise ValueError("This file is silent or too quiet to analyze.")

    y, _ = librosa.effects.trim(y, top_db=40)
    if len(y) < sr * 4:
        raise ValueError("Use at least 4 seconds of audible music.")
    y = librosa.util.normalize(y)
    hop_length = 256
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    if np.count_nonzero(onset > 0.1 * np.max(onset)) < 4:
        raise ValueError("No clear rhythm found. Try a section with drums.")
    tempo, beats = librosa.beat.beat_track(
        onset_envelope=onset, sr=sr, hop_length=hop_length, trim=False
    )
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    if not np.isfinite(bpm) or bpm <= 0 or len(beats) < 4:
        raise ValueError("No stable tempo found. Try a section with drums.")

    # Estimate the beat grid across many beats to reduce frame-rounding error.
    gaps = np.diff(beats)
    median_gap = np.median(gaps)
    regular_gaps = np.abs(gaps - median_gap) <= median_gap * 0.15
    if np.mean(regular_gaps) >= 0.9:
        frames_per_beat = np.polyfit(np.arange(len(beats)), beats, 1)[0]
        refined = 60 * sr / (hop_length * frames_per_beat)
        if abs(refined - bpm) / bpm < 0.05:
            bpm = refined
    return round(bpm, 1)


def analyze_upload(source, suffix):
    # Each request gets its own file, including simultaneous identical names.
    with tempfile.TemporaryDirectory(prefix="bakery-bpm-") as directory:
        filepath = Path(directory) / ("audio" + suffix)
        with filepath.open("wb") as destination:
            size = 0
            while chunk := source.read(1024 * 1024):
                size += len(chunk)
                if size > 100 * 1024 * 1024:
                    raise HTTPException(413, "Use a file smaller than 100 MB.")
                destination.write(chunk)
        if size == 0:
            raise HTTPException(400, "The uploaded file is empty.")
        try:
            return detect_bpm(filepath)
        except ValueError as error:
            raise HTTPException(422, str(error)) from error
        except Exception as error:
            raise HTTPException(422, "Could not read this audio. Try a WAV or MP3 file.") from error


@app.post("/upload/")
async def upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "").suffix.lower()
    try:
        if suffix not in {".mp3", ".wav"}:
            raise HTTPException(415, "Choose an MP3 or WAV file.")
        bpm = await run_in_threadpool(analyze_upload, file.file, suffix)
        return {
            "bpm": bpm,
            "file": file.filename,
            "alternatives": [round(bpm / 2, 1), round(bpm * 2, 1)],
        }
    finally:
        await file.close()
