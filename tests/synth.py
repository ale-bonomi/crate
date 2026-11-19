"""Synthetic audio for tests. Never commit copyrighted samples."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def write_wav(path: Path, samples: np.ndarray, sr: int) -> None:
    pcm = np.clip(samples, -1.0, 1.0)
    ints = (pcm * 32767.0).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sr)
        handle.writeframes(ints.tobytes())


def kick(sr: int = 22050) -> np.ndarray:
    n = int(sr * 0.4)
    t = np.arange(n) / sr
    freq = np.linspace(160, 42, n)
    phase = 2 * np.pi * np.cumsum(freq) / sr
    body = np.sin(phase) * np.exp(-t * 10)
    click = np.exp(-t * 400) * ((np.arange(n) % 2) * 2 - 1)
    return (0.85 * body + 0.35 * click).astype(np.float32)


def snare(sr: int = 22050, decay: float = 30.0) -> np.ndarray:
    n = int(sr * (0.15 if decay > 20 else 0.45))
    rng = np.random.default_rng(1 if decay > 20 else 2)
    noise = rng.uniform(-1, 1, n).astype(np.float32)
    t = np.arange(n) / sr
    env = np.exp(-t * decay)
    # Tilt toward mids/highs so it is not a kick.
    colored = np.diff(noise, prepend=noise[0])
    return (colored * env).astype(np.float32)


def clicks(sr: int, duration: float, bpm: int, swing: float) -> np.ndarray:
    n = int(sr * duration)
    sig = np.zeros(n, dtype=np.float32)
    step = 60.0 / bpm / 4.0
    i = 0
    t = 0.0
    while t < duration - 0.01:
        pos = t + (step * swing if i % 2 else 0.0)
        start = int(pos * sr)
        end = min(n, start + 12)
        if 0 <= start < n:
            sig[start:end] = np.hanning(end - start).astype(np.float32)
        i += 1
        t += step
    return sig


def chord(sr: int, duration: float, bpm: int = 120) -> np.ndarray:
    """C major pulses so the loop has onsets and a clear key."""
    n = int(sr * duration)
    t = np.arange(n) / sr
    tone = (
        np.sin(2 * np.pi * 261.63 * t)
        + 0.8 * np.sin(2 * np.pi * 329.63 * t)
        + 0.8 * np.sin(2 * np.pi * 392.00 * t)
    )
    beat = 60.0 / bpm
    env = np.exp(-(np.mod(t, beat)) * 6.0)
    return (0.3 * tone * env).astype(np.float32)
