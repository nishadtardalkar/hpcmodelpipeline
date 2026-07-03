"""Registered model backends."""

from __future__ import annotations

from pathlib import Path

from models.base import ModelBackend
from models.florence2_flux import Florence2FluxBackend
from models.flux_uncensored import FluxUncensoredBackend
from models.qwen import QwenBackend

DEFAULT_MODELS_BASE = Path("./models")

BACKENDS: dict[str, ModelBackend] = {
    Florence2FluxBackend.id: Florence2FluxBackend(),
    FluxUncensoredBackend.id: FluxUncensoredBackend(),
    QwenBackend.id: QwenBackend(),
}

DEFAULT_MODEL_ID = QwenBackend.id


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
