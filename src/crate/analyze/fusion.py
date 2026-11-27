"""Fuse filename metadata, DSP, and optional CLAP scores. Manual corrections win."""

from __future__ import annotations

from .clap import best_label
from .features import AUDIO_TYPES, characteristics, loop_guess, type_guess
from .metadata import Meta
from .types import Correction, Features, Verdict

_EXPLICIT = 0.8
_CLAP_TYPES = {
    "kick": ("kick", None),
    "snare": ("snare", None),
    "snare_tight": ("snare", "tight"),
    "snare_body": ("snare", "body"),
    "clap": ("clap", None),
    "hat": ("hat", None),
    "cymbal": ("cymbal", None),
    "perc": ("perc", None),
    "tom": ("tom", None),
    "rim": ("rim", None),
    "bass": ("bass", None),
    "808": ("808", None),
    "melodic": ("melodic", None),
    "vocal": ("vocal", None),
    "fx": ("fx", None),
    "loop_drums": ("drum", None),
    "loop_perc": ("perc", None),
    "loop_bass": ("bass", None),
    "loop_melodic": ("melodic", None),
    "loop_vocal": ("vocal", None),
    "loop_fx": ("fx", None),
}
_LOOP_LABELS = {
    "loop_drums",
    "loop_perc",
    "loop_bass",
    "loop_melodic",
    "loop_vocal",
    "loop_fx",
}

VALID_TYPES = {
    "kick",
    "snare",
    "clap",
    "rim",
    "hat",
    "cymbal",
    "tom",
    "perc",
    "808",
    "bass",
    "melodic",
    "vocal",
    "fx",
    "drum",
    "top",
}
_LOOP_ALIASES = {
    "drums": "drum",
    "drum": "drum",
    "tops": "top",
    "top": "top",
    "percs": "perc",
    "perc": "perc",
    "percussion": "perc",
    "bass": "bass",
    "melodic": "melodic",
    "vocal": "vocal",
    "vocals": "vocal",
    "fx": "fx",
    "textures": "fx",
}


def parse_label(text: str) -> Correction:
    """Labels look like `kick`, `snare:tight`, `loop:drums`, or `oneshot:hat`."""
    raw = text.strip().lower().replace(" ", "")
    if not raw:
        raise ValueError("empty label")
    loop: bool | None = None
    if raw.startswith("loop:"):
        loop = True
        raw = raw.split(":", 1)[1]
        raw = _LOOP_ALIASES.get(raw, raw)
    elif raw.startswith("oneshot:"):
        loop = False
        raw = raw.split(":", 1)[1]
    punch = snare = feel = wet = None
    if ":" in raw:
        raw, char = raw.split(":", 1)
        if char in {"punchy", "soft"}:
            punch = char
        elif char in {"tight", "body"}:
            snare = char
        elif char in {"straight", "swung"}:
            feel = char
        elif char in {"wet", "dry"}:
            wet = char
        else:
            raise ValueError(f"unknown characteristic: {char}")
    if raw not in VALID_TYPES:
        raise ValueError(f"unknown type: {raw}")
    if loop is None:
        loop = raw in {"drum", "top"}
    return Correction(raw, loop, punch, snare, feel, wet)


def _from_meta(meta: Meta, source: str = "metadata") -> Verdict:
    return Verdict(
        type=meta.type,
        loop=meta.loop,
        confidence=meta.confidence,
        reason=meta.reason or source,
        bpm=meta.bpm,
        key=meta.key,
        punch=meta.punch,
        snare_body=meta.snare_body,
        feel=meta.feel,
        wet=meta.wet,
        source=source,
    )


def apply_correction(base: Verdict, correction: Correction) -> Verdict:
    return Verdict(
        type=correction.type or base.type,
        loop=correction.loop if correction.loop is not None else base.loop,
        confidence=1.0,
        reason="manual",
        bpm=base.bpm,
        key=base.key,
        punch=correction.punch or base.punch,
        snare_body=correction.snare_body or base.snare_body,
        feel=correction.feel or base.feel,
        wet=correction.wet or base.wet,
        source="manual",
        content_hash=base.content_hash,
        fingerprint=base.fingerprint,
        raw_hash=base.raw_hash,
        features=base.features,
    )


def fuse(
    meta: Meta,
    features: Features | None,
    clap: dict[str, float] | None,
    correction: Correction | None,
) -> Verdict:
    if features is None and not clap:
        verdict = _from_meta(meta)
        if correction:
            verdict = apply_correction(verdict, correction)
        return verdict

    audio_type: str | None = None
    audio_conf = 0.0
    audio_loop: bool | None = None
    clap_body: str | None = None
    from_clap = False
    if features is not None:
        audio_type, audio_conf = type_guess(features)
        audio_loop = loop_guess(features)
    clap_best = best_label(clap)
    if clap_best is not None:
        label, score = clap_best
        mapped = _CLAP_TYPES.get(label)
        # Softmax over many prompts rarely exceeds 0.5; twice uniform is a real winner.
        n = max(len(clap or {}), 1)
        ahead = score >= 0.25 and score * n >= 1.5 and score >= audio_conf
        if mapped and ahead:
            audio_type, clap_body = mapped
            from_clap = True
            audio_conf = min(0.9, 0.55 + score)
            if label in _LOOP_LABELS:
                audio_loop = True
            elif audio_loop is None:
                audio_loop = False

    chosen: str | None
    conf: float
    reason: str
    if meta.type and audio_type:
        if meta.type == audio_type:
            chosen, conf = audio_type, min(1.0, max(meta.confidence, audio_conf) + 0.05)
            reason = f"agree:{chosen}"
        elif meta.confidence >= _EXPLICIT:
            chosen, conf = meta.type, 0.85
            reason = f"metadata:{meta.type} over audio:{audio_type}"
        elif from_clap or meta.type not in AUDIO_TYPES:
            # The spectrum can't recognise a vocal, an FX or a clap, so its guess is no
            # evidence against the name. CLAP can, but measured on a real pack it was
            # wrong 78% of the times it disagreed with a name, so it only fills gaps.
            chosen, conf = meta.type, meta.confidence
            reason = f"metadata:{meta.type} (audio guessed {audio_type})"
        else:
            chosen, conf = audio_type, round(min(audio_conf, 0.75), 2)
            reason = f"audio:{audio_type} over metadata:{meta.type}"
    elif audio_type:
        chosen, conf, reason = audio_type, audio_conf, f"audio:{audio_type}"
    else:
        chosen, conf, reason = meta.type, meta.confidence, meta.reason or "metadata"

    loop: bool | None
    if meta.loop is not None and (audio_loop is None or meta.loop == audio_loop):
        loop = meta.loop
    elif audio_loop is not None and meta.loop is None:
        loop = audio_loop
    elif audio_loop is not None and meta.loop is not None and meta.loop != audio_loop:
        # A CLAP "loop" prompt winning is no evidence against a "one shot" folder.
        trust_meta = meta.confidence >= _EXPLICIT or from_clap
        loop = meta.loop if trust_meta else audio_loop
        conf = min(conf, 0.7)
        reason += f" loop:{'meta' if loop == meta.loop else 'audio'}"
    else:
        loop = meta.loop

    bpm = meta.bpm
    key = meta.key
    punch, snare, feel, wet = meta.punch, meta.snare_body, meta.feel, meta.wet
    if features is not None:
        if bpm is None and features.tempo:
            bpm = int(round(features.tempo))
        if key is None and features.key and features.key_confidence >= 0.2:
            key = features.key
        dp, ds, df, dw = characteristics(features, chosen, loop)
        punch = dp or punch
        snare = ds or snare
        feel = df or feel
        wet = dw or wet
    if clap_body and chosen == "snare":
        snare = clap_body

    verdict = Verdict(
        type=chosen,
        loop=loop,
        confidence=round(float(conf), 2) if conf else 0.0,
        reason=reason,
        bpm=bpm,
        key=key,
        punch=punch,
        snare_body=snare,
        feel=feel,
        wet=wet,
        source="fusion" if features is not None or clap else "metadata",
        features=features,
    )
    if correction:
        verdict = apply_correction(verdict, correction)
    return verdict
