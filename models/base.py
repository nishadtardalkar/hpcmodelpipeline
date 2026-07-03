"""Model backend interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image


class ModelBackend(ABC):
    """One backend: download, load on GPU, run inference from job payload."""

    id: str
    display_name: str
    hf_id: str
    requires_images: bool
    output_kind: str = "image"  # "image" or "text"

    @abstractmethod
    def download(self, dest: Path) -> None:
        """Fetch weights to *dest* (CPU only, no CUDA)."""

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Load pipeline on CUDA from local *path*."""

    @abstractmethod
    def infer(self, pipeline: Any, payload: dict[str, Any]) -> Image.Image | str:
        """Run one job; *payload* is decrypted JSON from the browser."""

    def public_spec(self, models_base: Path) -> dict[str, Any]:
        from models.registry import weights_dir

        return {
            "id": self.id,
            "display_name": self.display_name,
            "requires_images": self.requires_images,
            "output_kind": self.output_kind,
            "default_path": str(weights_dir(self.id, models_base)),
            "hf_id": self.hf_id,
        }
