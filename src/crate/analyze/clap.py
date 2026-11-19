"""Optional CLAP zero-shot tagger.

The model is not a required dependency. `uv sync --extra ml` installs torch and
transformers; the first call downloads `laion/clap-htsat-unfused`. Until then,
fusion uses metadata and the DSP features only.
"""

from __future__ import annotations

import importlib.util
from typing import Any

import numpy as np

MODEL_ID = "laion/clap-htsat-unfused"


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

    def available(self) -> bool:
        return installed() and not self._failed and bool(self.prompts)

    def _load(self) -> bool:
        if self._ready:
            return True
        if self._failed or not installed():
            return False
        try:
            import torch  # type: ignore[import-not-found]
            from transformers import ClapModel, ClapProcessor  # type: ignore[import-not-found]

            device = "cpu"
            if torch.backends.mps.is_available():
                device = "mps"
            processor = ClapProcessor.from_pretrained(MODEL_ID)
            model = ClapModel.from_pretrained(MODEL_ID)
            model.to(device)
            model.eval()
        except Exception:
            self._failed = True
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
        import torch

        target = 48000
        if sr != target and len(samples):
            n = max(1, int(round(len(samples) * target / sr)))
            src_x = np.linspace(0.0, 1.0, num=len(samples), endpoint=False)
            dst_x = np.linspace(0.0, 1.0, num=n, endpoint=False)
            audio = np.interp(dst_x, src_x, samples).astype(np.float32)
        else:
            audio = np.asarray(samples, dtype=np.float32)
        audio = audio[: target * 10]
        try:
            processor = self._processor
            model = self._model
            assert processor is not None and model is not None
            inputs = processor(
                text=self._texts,
                audios=audio,
                sampling_rate=target,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(self._device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                probs = torch.softmax(out.logits_per_audio, dim=-1)[0].detach().cpu().numpy()
        except Exception:
            return None
        best: dict[str, float] = {}
        for label, score in zip(self._labels, probs, strict=True):
            value = float(score)
            if value > best.get(label, 0.0):
                best[label] = value
        return best or None


def best_label(scores: dict[str, float] | None) -> tuple[str, float] | None:
    if not scores:
        return None
    label, score = max(scores.items(), key=lambda kv: kv[1])
    return label, score
