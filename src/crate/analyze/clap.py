"""Optional CLAP model: zero-shot prompt scores, and audio embeddings for `crate train`.

The model is not a required dependency. `uv sync --extra ml` installs torch and
transformers; the first call downloads `laion/clap-htsat-unfused`. Until then,
fusion uses metadata and the DSP features only.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

MODEL_ID = "laion/clap-htsat-unfused"
RATE = 48000
CLIP_SECONDS = 10


def installed() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


class ClapTagger:
    def __init__(self, prompts: dict[str, list[str]]) -> None:
        self.prompts = {k: v for k, v in prompts.items() if v}
        self._ready = False
        self._failed = False
        self._model: Any = None
        self._processor: Any = None
        self._device = "cpu"
        self._labels: list[str] = []
        self._texts: list[str] = []
        self.error: str | None = None  # first failure, reported once instead of hidden

    def available(self) -> bool:
        """True when prompt scoring can run (embeddings need no prompts)."""
        return installed() and not self._failed and bool(self.prompts)

    def can_embed(self) -> bool:
        return installed() and not self._failed

    def _load(self) -> bool:
        if self._ready:
            return True
        if self._failed or not installed():
            return False
        try:
            # Untyped on purpose: both are optional and not installed in CI.
            torch: Any = importlib.import_module("torch")
            transformers: Any = importlib.import_module("transformers")
            ClapModel, ClapProcessor = transformers.ClapModel, transformers.ClapProcessor

            device = "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
            processor = ClapProcessor.from_pretrained(MODEL_ID)
            model = ClapModel.from_pretrained(MODEL_ID)
            model.to(device)
            model.eval()
        except Exception as exc:
            self._failed = True
            self.error = f"CLAP model did not load: {type(exc).__name__}: {exc}"
            return False
        labels: list[str] = []
        texts: list[str] = []
        for label, options in self.prompts.items():
            for text in options:
                labels.append(label)
                texts.append(text)
        self._labels = labels
        self._texts = texts
        self._processor = processor
        self._model = model
        self._device = device
        self._ready = True
        return True

    def score(self, samples: np.ndarray, sr: int) -> dict[str, float] | None:
        """Map each prompt label to its best similarity against this clip."""
        if not self._load():
            return None
        torch: Any = importlib.import_module("torch")
        audio = clip(samples, sr)
        try:
            processor = self._processor
            model = self._model
            assert processor is not None and model is not None
            inputs = _encode(processor, self._texts, audio, RATE)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                probs = torch.softmax(out.logits_per_audio, dim=-1)[0].detach().cpu().numpy()
        except Exception as exc:
            if self.error is None:
                self.error = f"CLAP scoring failed: {type(exc).__name__}: {exc}"
            return None
        best: dict[str, float] = {}
        for label, score in zip(self._labels, probs, strict=True):
            value = float(score)
            if value > best.get(label, 0.0):
                best[label] = value
        return best or None

    def embed(self, samples: np.ndarray, sr: int) -> np.ndarray | None:
        """CLAP's audio embedding: a vector where similar sounds land close together."""
        if not self._load():
            return None
        torch: Any = importlib.import_module("torch")
        try:
            inputs = _encode_audio(self._processor, clip(samples, sr), RATE)
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self._model.get_audio_features(**inputs)
            out = getattr(out, "pooler_output", out)  # transformers 5 returns an output object
            vector: np.ndarray = out[0].detach().cpu().numpy().astype(np.float32)
        except Exception as exc:
            if self.error is None:
                self.error = f"CLAP embedding failed: {type(exc).__name__}: {exc}"
            return None
        return vector


def clip(samples: np.ndarray, sr: int) -> np.ndarray:
    """The first CLIP_SECONDS, resampled to the model's rate."""
    if sr != RATE and len(samples):
        n = max(1, int(round(len(samples) * RATE / sr)))
        src_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=n, endpoint=False)
        audio = np.interp(dst_x, src_x, samples).astype(np.float32)
    else:
        audio = np.asarray(samples, dtype=np.float32)
    head: np.ndarray = audio[: RATE * CLIP_SECONDS]
    return head


def _encode_audio(processor: Any, audio: np.ndarray, sr: int) -> Any:
    try:
        return processor(audio=audio, sampling_rate=sr, return_tensors="pt")
    except (TypeError, ValueError):
        return processor(audios=audio, sampling_rate=sr, return_tensors="pt")


def _encode(processor: Any, texts: list[str], audio: np.ndarray, sr: int) -> Any:
    """transformers 5 takes `audio=`; releases before it took `audios=`."""
    common = {"text": texts, "sampling_rate": sr, "return_tensors": "pt", "padding": True}
    try:
        return processor(audio=audio, **common)
    except (TypeError, ValueError):
        return processor(audios=audio, **common)


def best_label(scores: dict[str, float] | None) -> tuple[str, float] | None:
    if not scores:
        return None
    label, score = max(scores.items(), key=lambda kv: kv[1])
    return label, score
