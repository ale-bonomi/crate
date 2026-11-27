"""DSP features from mono float audio. No librosa: numpy is enough and stays installable."""

from __future__ import annotations

import numpy as np

from .types import Features

_HOP = 128
_FRAME = 512
# Krumhansl-Kessler major / minor profiles, used only as a correlation template.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _envelope(samples: np.ndarray) -> np.ndarray:
    if len(samples) < _FRAME:
        rms = float(np.sqrt(np.mean(samples**2))) if len(samples) else 0.0
        return np.array([rms], dtype=np.float32)
    frames = 1 + (len(samples) - _FRAME) // _HOP
    env = np.empty(frames, dtype=np.float32)
    for i in range(frames):
        start = i * _HOP
        frame = samples[start : start + _FRAME]
        env[i] = float(np.sqrt(np.mean(frame**2)))
    return env


def _onsets(env: np.ndarray, sr: int) -> list[float]:
    if len(env) == 0 or float(np.max(env)) <= 1e-8:
        return []
    peak = float(np.max(env))
    thresh = max(peak * 0.28, float(np.median(env)) * 2.0)
    times: list[float] = []
    last = -1.0
    for i in range(1, len(env) - 1):
        if env[i] >= thresh and env[i] >= env[i - 1] and env[i] > env[i + 1]:
            t = i * _HOP / sr
            if t - last >= 0.04:
                times.append(t)
                last = t
    if not times and peak > 0:
        times.append(float(np.argmax(env)) * _HOP / sr)
    return times


def _spectral(samples: np.ndarray, sr: int) -> tuple[float, float, float, float, float, float]:
    n = 1 << max(11, min(14, (max(len(samples), 2) - 1).bit_length()))
    n = min(max(n, 2048), 16384)
    chunk = np.zeros(n, dtype=np.float32)
    take = min(len(samples), n)
    chunk[:take] = samples[:take]
    mag = np.abs(np.fft.rfft(chunk * np.hanning(n).astype(np.float32))).astype(np.float64)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag[0] = 0.0
    total = float(mag.sum()) or 1.0
    centroid = float((freqs * mag).sum() / total)
    cumulative = np.cumsum(mag) / total
    rolloff_i = int(np.searchsorted(cumulative, 0.85))
    rolloff = float(freqs[min(rolloff_i, len(freqs) - 1)])
    power = mag**2
    ptot = float(power.sum()) or 1.0
    low = float(power[freqs < 200].sum() / ptot)
    mid = float(power[(freqs >= 200) & (freqs < 2000)].sum() / ptot)
    high = float(power[freqs >= 2000].sum() / ptot)
    positive = mag[1:] + 1e-12
    flatness = float(np.exp(np.mean(np.log(positive))) / np.mean(positive))
    tonalness = float(np.clip(1.0 - flatness, 0.0, 1.0))
    return centroid, rolloff, low, mid, high, tonalness


def _chroma(samples: np.ndarray, sr: int) -> np.ndarray:
    n = 8192 if len(samples) >= 8192 else 1 << max(10, (len(samples) - 1).bit_length())
    hop = n // 2
    chroma = np.zeros(12, dtype=np.float64)
    if len(samples) < 256:
        return chroma
    start = 0
    while start + 256 < len(samples):
        chunk = np.zeros(n, dtype=np.float32)
        take = min(n, len(samples) - start)
        chunk[:take] = samples[start : start + take]
        mag = np.abs(np.fft.rfft(chunk * np.hanning(n).astype(np.float32)))
        freqs = np.fft.rfftfreq(n, d=1.0 / sr)
        for k in range(1, len(freqs)):
            f = float(freqs[k])
            if f < 60 or f > 5000:
                continue
            midi = 69.0 + 12.0 * np.log2(f / 440.0)
            pc = int(np.round(midi)) % 12
            chroma[pc] += float(mag[k])
        start += hop
        if start > sr * 8:
            break
    total = chroma.sum()
    if total > 0:
        chroma /= total
    return chroma


def _key(samples: np.ndarray, sr: int, tonalness: float) -> tuple[str | None, float]:
    if tonalness < 0.55:
        return None, 0.0
    chroma = _chroma(samples, sr)
    if float(chroma.sum()) <= 0:
        return None, 0.0
    order = np.argsort(chroma)[::-1]
    triad = {int(i) for i in order[:3]}
    triad_energy = float(chroma[order[:3]].sum())
    rest_energy = float(chroma[order[3:]].sum())
    confidence = float(
        np.clip((triad_energy - rest_energy) / (triad_energy + rest_energy + 1e-6), 0.0, 1.0)
    )
    for root, note in enumerate(_NOTES):
        major = {root, (root + 4) % 12, (root + 7) % 12}
        minor = {root, (root + 3) % 12, (root + 7) % 12}
        if triad == major and confidence >= 0.2:
            return f"{note}maj", confidence
        if triad == minor and confidence >= 0.2:
            return f"{note}min", confidence
    return None, confidence


def _tempo(env: np.ndarray, sr: int) -> float | None:
    if len(env) < 8:
        return None
    odf = np.maximum(np.diff(env, prepend=env[0]), 0.0)
    odf = odf - odf.mean()
    if float(np.max(np.abs(odf))) < 1e-8:
        return None
    ac = np.correlate(odf, odf, mode="full")[len(odf) - 1 :]
    hop_s = _HOP / sr
    lag_min = max(1, int(60 / 180 / hop_s))
    lag_max = min(len(ac) - 1, int(60 / 70 / hop_s))
    if lag_max <= lag_min:
        return None
    window = ac[lag_min : lag_max + 1]
    lag = lag_min + int(np.argmax(window))
    if ac[lag] <= 0:
        return None
    bpm = 60.0 / (lag * hop_s)
    return float(bpm) if 70 <= bpm <= 180 else None


def _swing(onsets: list[float]) -> float | None:
    if len(onsets) < 5:
        return None
    gaps = np.diff(np.array(onsets, dtype=np.float64))
    gaps = gaps[gaps > 0.03]
    if len(gaps) < 4:
        return None
    a = float(np.median(gaps[0::2]))
    b = float(np.median(gaps[1::2]))
    small = min(a, b)
    if small <= 1e-4:
        return None
    return float(max(a, b) / small)


def _attack_decay(env: np.ndarray, sr: int) -> tuple[float, float]:
    if len(env) == 0 or float(np.max(env)) <= 1e-8:
        return 0.0, 0.0
    peak_i = int(np.argmax(env))
    peak = float(env[peak_i]) or 1e-8
    attack = peak_i * _HOP / sr
    for i, value in enumerate(env):
        if value >= 0.5 * peak:
            attack = i * _HOP / sr
            break
    decay = (len(env) - 1 - peak_i) * _HOP / sr
    for i in range(peak_i, len(env)):
        if env[i] <= 0.1 * peak:
            decay = (i - peak_i) * _HOP / sr
            break
    return float(attack), float(decay)


def extract(samples: np.ndarray, sr: int, duration: float | None = None) -> Features:
    x = np.asarray(samples, dtype=np.float32)
    if duration is None:
        duration = float(len(x) / sr) if sr else 0.0
    if len(x) == 0 or sr <= 0:
        return Features(
            0.0, sr, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, None, None, 0.0, None
        )
    peak = float(np.max(np.abs(x)))
    rms = float(np.sqrt(np.mean(x**2))) or 1e-8
    env = _envelope(x)
    onsets = _onsets(env, sr)
    attack, decay = _attack_decay(env, sr)
    centroid, rolloff, low, mid, high, tonalness = _spectral(x, sr)
    tempo = _tempo(env, sr)
    key, key_conf = _key(x, sr, tonalness)
    return Features(
        duration=float(duration),
        sample_rate=sr,
        onset_count=len(onsets),
        crest=peak / rms,
        attack=attack,
        decay=decay,
        centroid=centroid,
        rolloff=rolloff,
        low=low,
        mid=mid,
        high=high,
        tonalness=tonalness,
        tempo=tempo,
        key=key,
        key_confidence=key_conf,
        swing=_swing(onsets),
    )


def loop_guess(features: Features) -> bool | None:
    if features.duration >= 1.5 and features.onset_count >= 4:
        return True
    if features.duration < 0.9 and features.onset_count <= 3:
        return False
    if features.tempo and features.duration >= 1.2 and features.onset_count >= 4:
        bar = 4 * 60.0 / features.tempo
        ratio = features.duration / bar
        nearest = round(ratio)
        if nearest >= 1 and abs(ratio - nearest) / nearest <= 0.12:
            return True
    return None


# The only types type_guess can return. For any other type, the audio has no opinion.
AUDIO_TYPES = frozenset({"kick", "808", "hat", "snare", "melodic", "drum"})


def type_guess(features: Features) -> tuple[str | None, float]:
    """A coarse type from the spectrum. CLAP and filenames can overrule this."""
    short = features.duration < 1.15 and features.onset_count <= 4
    # Impulsive loops (click grids) look tonal in a long FFT. The crest gives them away.
    if features.duration >= 1.2 and features.onset_count >= 4 and features.crest >= 8:
        return "drum", 0.7
    if short and features.centroid < 700 and features.low >= 0.35:
        return "kick", 0.74
    if short and features.low >= 0.55 and features.centroid < 400:
        return "808", 0.66
    if (
        short
        and features.centroid >= 6000
        and features.duration < 0.08
        and features.tonalness < 0.45
    ):
        return "hat", 0.66
    if short and features.tonalness < 0.5 and features.centroid >= 1200:
        return "snare", 0.7
    if features.tonalness >= 0.58 and features.centroid < 4500 and features.low < 0.55:
        return "melodic", 0.7
    if features.onset_count >= 4 and features.tonalness < 0.5 and features.duration >= 1.2:
        return "drum", 0.7
    if short and features.tonalness >= 0.58:
        return "melodic", 0.62
    return None, 0.0


def characteristics(
    features: Features, sound_type: str | None, loop: bool | None
) -> tuple[str | None, str | None, str | None, str | None]:
    """punch, snare body, feel, wet/dry. Each is None when it doesn't apply."""
    punch = snare = feel = wet = None
    oneshot = loop is False or (loop is None and features.duration < 1.15)
    if oneshot:
        punch = "punchy" if features.crest >= 4.0 and features.attack <= 0.03 else "soft"
    if sound_type in {"snare", "rim", "clap"}:
        snare = "tight" if features.decay <= 0.2 and features.centroid >= 2000 else "body"
    if loop is True and features.swing is not None:
        feel = "swung" if features.swing >= 1.35 else "straight"
    tail = features.decay
    wet = "wet" if tail >= 0.45 and features.duration > 0.3 else "dry"
    return punch, snare, feel, wet
