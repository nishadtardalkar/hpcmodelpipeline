"""Registered model backends."""

from __future__ import annotations

from pathlib import Path

from models.ace_step import AceStepBackend
from models.base import ModelBackend
from models.flux2_dev import Flux2DevBackend
from models.hunyuan3d_paint import Hunyuan3DPaintBackend
from models.qwen_abliterated import QwenAbliteratedBackend
from models.qwen_rapid_nsfw import QwenRapidNsfwBackend
from models.stable_audio import StableAudioOpenBackend
from models.trellis2_4b import Trellis24BBackend

DEFAULT_MODELS_BASE = Path("./models")

BACKENDS: dict[str, ModelBackend] = {
    AceStepBackend.id: AceStepBackend(),
    Flux2DevBackend.id: Flux2DevBackend(),
    Hunyuan3DPaintBackend.id: Hunyuan3DPaintBackend(),
    QwenAbliteratedBackend.id: QwenAbliteratedBackend(),
    QwenRapidNsfwBackend.id: QwenRapidNsfwBackend(),
    StableAudioOpenBackend.id: StableAudioOpenBackend(),
    Trellis24BBackend.id: Trellis24BBackend(),
}

DEFAULT_MODEL_ID = QwenAbliteratedBackend.id


def weights_dir(model_id: str, base: Path | None = None) -> Path:
    """Default local weights path: ``<base>/<model_id>``."""
    return (base or DEFAULT_MODELS_BASE) / model_id


def get_backend(model_id: str) -> ModelBackend:
    if model_id not in BACKENDS:
        known = ", ".join(sorted(BACKENDS))
        raise ValueError(f"Unknown model {model_id!r}. Choose from: {known}")
    return BACKENDS[model_id]


def list_models(models_base: Path | None = None) -> list[dict]:
    base = models_base or DEFAULT_MODELS_BASE
    return [BACKENDS[k].public_spec(base) for k in sorted(BACKENDS)]
