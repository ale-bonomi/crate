"""Decode audio, hash the PCM, and build a coarse fingerprint for near-duplicates."""

from __future__ import annotations

import hashlib
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FEATURE_SECONDS = 30


@dataclass
class AudioData:
    samples: np.ndarray  # mono float32, trimmed to the analysis window
    duration: float  # full-file seconds
    sample_rate: int
    content_hash: str
    fingerprint: str
    raw_hash: str


def raw_hash(path: Path) -> str:
    h = hashlib.blake2b(digest_size=20)
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def pcm_hash(samples: np.ndarray) -> str:
    pcm = np.clip(samples, -1.0, 1.0)
    ints = (pcm * 32767.0).astype(np.int16)
    return hashlib.blake2b(ints.tobytes(), digest_size=20).hexdigest()


def _resample(samples: np.ndarray, sr: int, target: int = 8000) -> np.ndarray:
    if sr == target or len(samples) == 0:
        return samples
    n = max(1, int(round(len(samples) * target / sr)))
    src_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=n, endpoint=False)
    return np.asarray(np.interp(dst_x, src_x, samples), dtype=np.float32)


def fingerprint(samples: np.ndarray, sr: int) -> str:
    """Coarse loudness shape. Survives a gain change or a sample-rate conversion.

    Exact copies are `content_hash`. This is only for the near-duplicate report.
    """
    y = _resample(samples, sr)
    y = y[:8000]
    if len(y) < 160:
        return hashlib.blake2b(y.tobytes(), digest_size=16).hexdigest()
    block = 160  # 20 ms at 8 kHz
    n = len(y) // block
    env = y[: n * block].reshape(n, block).mean(axis=1)
    peak = float(np.max(np.abs(env))) or 1.0
    q = np.clip(np.round(env / peak * 2.0), -2, 2).astype(np.int8)
    return hashlib.blake2b(q.tobytes(), digest_size=16).hexdigest()


def _decode_wav(path: Path) -> tuple[np.ndarray, int] | None:
    try:
        with wave.open(str(path), "rb") as w:
            sr = w.getframerate()
            ch = w.getnchannels()
            sw = w.getsampwidth()
            n = w.getnframes()
            raw = w.readframes(n)
    except (wave.Error, EOFError, OSError):
        return None
    if sr <= 0 or n <= 0:
        return None
    if sw == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif sw == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sw == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        return None
    if ch > 1:
        usable = (len(data) // ch) * ch
        data = data[:usable].reshape(-1, ch).mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sr


def _decode_soundfile(path: Path) -> tuple[np.ndarray, int] | None:
    try:
        import soundfile as sf  # type: ignore[import-untyped]
    except ImportError:
        return None
    try:
        data, sr = sf.read(str(path), always_2d=False, dtype="float32")
    except Exception:
        return None
    arr = np.asarray(data, dtype=np.float32)
    if arr.size == 0 or sr <= 0:
        return None
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    return np.ascontiguousarray(arr, dtype=np.float32), int(sr)


def decode(path: Path) -> tuple[np.ndarray, int] | None:
    if path.suffix.lower() == ".wav":
        wav = _decode_wav(path)
        if wav is not None:
            return wav
    return _decode_soundfile(path)


def load(path: Path) -> AudioData | None:
    """Decode a file. Features see at most the first 30 seconds; hashes see the whole file."""
    decoded = decode(path)
    if decoded is None:
        return None
    full, sr = decoded
    if full.size == 0:
        return None
    window = full[: int(sr * FEATURE_SECONDS)]
    return AudioData(
        samples=window,
        duration=float(len(full) / sr),
        sample_rate=sr,
        content_hash=pcm_hash(full),
        fingerprint=fingerprint(full, sr),
        raw_hash=raw_hash(path),
    )
