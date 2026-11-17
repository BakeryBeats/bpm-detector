from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import math
import soundfile as sf
from scipy import signal
from scipy.ndimage import uniform_filter1d
import tempfile
import threading
from pathlib import Path
import numpy as np
from starlette.concurrency import run_in_threadpool

app = FastAPI()
analysis_lock = threading.Lock()

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
    # Decode and analyze short blocks, keeping memory bounded for long masters.
    target_rate = 11025
    hop_length = 128
    onset_chunks = []
    peak = 0.0
    with sf.SoundFile(file_path) as audio:
        if len(audio) < audio.samplerate * 4:
            raise ValueError("Use at least 4 seconds of music.")
        remaining = min(len(audio), audio.samplerate * 120)
        divisor = math.gcd(audio.samplerate, target_rate)
        while remaining > 0:
            block = audio.read(min(remaining, audio.samplerate * 10), dtype="float32", always_2d=True)
            if not len(block):
                break
            remaining -= len(block)
            if not np.isfinite(block).all():
                raise ValueError("This file contains invalid audio samples.")
            # Select the strongest channel to preserve out-of-phase stereo beats.
            channel = np.argmax(np.mean(block ** 2, axis=0))
            mono = block[:, channel]
            peak = max(peak, float(np.max(np.abs(mono))))
            mono = signal.resample_poly(
                mono, target_rate // divisor, audio.samplerate // divisor
            )
            _, _, spectrum = signal.stft(
                mono, fs=target_rate, nperseg=512, noverlap=384
            )
            # Positive spectral changes emphasize drum attacks across all bands.
            magnitude = np.log1p(100 * np.abs(spectrum))
            flux = np.maximum(np.diff(magnitude, axis=1), 0).mean(axis=0)
            onset_chunks.append(flux)
    if peak < 1e-5 or not onset_chunks:
        raise ValueError("This file is silent or too quiet to analyze.")
    onset = np.concatenate(onset_chunks)
    local_average = uniform_filter1d(onset, size=43)
    onset = np.maximum(onset - local_average, 0)
    if np.max(onset) < 1e-8:
        raise ValueError("No clear rhythm found. Try a section with drums.")
    onset /= np.max(onset)
    frame_rate = target_rate / hop_length
    minimum_lag = int(np.ceil(60 * frame_rate / 220))
    maximum_lag = min(int(60 * frame_rate / 40), len(onset) - 2)
    correlation = signal.fftconvolve(onset, onset[::-1], mode="full")[len(onset) - 1:]
    correlation /= np.arange(len(onset), 0, -1)
    candidates, _ = signal.find_peaks(correlation)
    candidates = candidates[(candidates >= minimum_lag) & (candidates <= maximum_lag)]
    if not len(candidates):
        raise ValueError("No stable tempo found. Try a section with drums.")
    tempos = 60 * frame_rate / candidates
    # A gentle preference resolves nearly equal half/double-time candidates.
    preference = np.exp(-0.5 * (np.log2(tempos / 100) / 0.8) ** 2)
    scores = correlation[candidates] * (1 + 0.08 * preference)
    lag = int(candidates[np.argmax(scores)])
    if correlation[lag] < correlation[0] * 0.08:
        raise ValueError("No clear rhythm found. Try a section with drums.")
    # Interpolate the correlation peak for fractional BPM precision.
    left, center, right = correlation[lag - 1:lag + 2]
    curvature = left - 2 * center + right
    offset = 0.0
    if abs(curvature) > 1e-12:
        offset = float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))
    return round(60 * frame_rate / (lag + offset), 1)


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
            # Concurrent FFT/JIT workloads can exhaust a small Render instance.
            with analysis_lock:
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
