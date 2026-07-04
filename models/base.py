"""Model backend interface."""

from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from PIL import Image

# UI field presets per category (browser reads via public_spec).
CATEGORY_UI: dict[str, dict[str, Any]] = {
    "image_edit": {
        "hint": "Upload 1–2 images and describe how to edit them.",
        "show_negative_prompt": True,
        "show_true_cfg": True,
        "show_audio_duration": False,
        "show_resolution": False,
        "show_mesh": False,
        "defaults": {
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "true_cfg_scale": 1.0,
        },
    },
    "text_to_image": {
        "hint": "Text-to-image, or upload an image for editing / img2img.",
        "show_negative_prompt": True,
        "show_true_cfg": False,
        "show_audio_duration": False,
        "show_resolution": False,
        "show_mesh": False,
        "defaults": {
            "num_inference_steps": 28,
            "guidance_scale": 4.0,
            "true_cfg_scale": 1.0,
        },
    },
    "image_to_3d": {
        "hint": "Upload a single reference image to generate a textured 3D mesh (GLB).",
        "show_negative_prompt": False,
        "show_true_cfg": False,
        "show_audio_duration": False,
        "show_resolution": True,
        "show_mesh": False,
        "defaults": {
            "num_inference_steps": 50,
            "guidance_scale": 1.0,
            "true_cfg_scale": 1.0,
            "resolution": 1024,
        },
    },
    "text_to_audio": {
        "hint": "Describe the music or sound effect to generate.",
        "show_negative_prompt": True,
        "show_true_cfg": False,
        "show_audio_duration": True,
        "show_resolution": False,
        "show_mesh": False,
        "defaults": {
            "num_inference_steps": 50,
            "guidance_scale": 7.0,
            "true_cfg_scale": 1.0,
            "audio_duration_s": 30,
        },
    },
    "mesh_texture": {
        "hint": "Upload a 3D mesh (GLB/OBJ) and a reference image to paint PBR textures.",
        "show_negative_prompt": False,
        "show_true_cfg": False,
        "show_audio_duration": False,
        "show_resolution": False,
        "show_mesh": True,
        "defaults": {
            "num_inference_steps": 50,
            "guidance_scale": 1.0,
            "true_cfg_scale": 1.0,
            "texture_size": 4096,
        },
    },
}


class ModelBackend(ABC):
    """One backend: download, load on GPU, run inference from job payload."""

    id: str
    display_name: str
    hf_id: str
    category: str
    image_mode: str = "none"  # "required", "optional", or "none"
    requires_mesh: bool = False
    output_kind: str = "image"  # "image", "text", "audio", "model3d"
    output_mime: str = "image/png"

    @property
    def requires_images(self) -> bool:
        return self.image_mode == "required"

    @abstractmethod
    def download(self, dest: Path) -> None:
        """Fetch weights to *dest* (CPU only, no CUDA)."""

    @abstractmethod
    def load(self, path: Path) -> Any:
        """Load pipeline on CUDA from local *path*."""

    @abstractmethod
    def infer(
        self, pipeline: Any, payload: dict[str, Any]
    ) -> Image.Image | str | bytes:
        """Run one job; *payload* is decrypted JSON from the browser."""

    def ui_config(self) -> dict[str, Any]:
        base = copy.deepcopy(CATEGORY_UI.get(self.category, CATEGORY_UI["text_to_image"]))
        base["image_mode"] = self.image_mode
        base["requires_mesh"] = self.requires_mesh
        return base

    def public_spec(self, models_base: Path) -> dict[str, Any]:
        from models.registry import weights_dir

        return {
            "id": self.id,
            "display_name": self.display_name,
            "category": self.category,
            "requires_images": self.requires_images,
            "image_mode": self.image_mode,
            "requires_mesh": self.requires_mesh,
            "output_kind": self.output_kind,
            "output_mime": self.output_mime,
            "ui": self.ui_config(),
            "default_path": str(weights_dir(self.id, models_base)),
            "hf_id": self.hf_id,
        }
