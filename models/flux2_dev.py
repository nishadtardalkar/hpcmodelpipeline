"""FLUX.2 Dev — text-to-image and image editing.

https://huggingface.co/black-forest-labs/FLUX.2-dev
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from job_crypto import payload_to_images
from models.base import ModelBackend
from models.hf_utils import snapshot_download_repo

HF_ID = "black-forest-labs/FLUX.2-dev"


class Flux2DevBackend(ModelBackend):
    id = "flux2_dev"
    display_name = "FLUX.2 Dev"
    hf_id = HF_ID
    category = "text_to_image"
    image_mode = "optional"
    output_kind = "image"
    output_mime = "image/png"

    def download(self, dest: Path) -> None:
        snapshot_download_repo(dest, HF_ID)

    def load(self, path: Path) -> Any:
        from diffusers import Flux2Pipeline

        pipe = Flux2Pipeline.from_pretrained(
            str(path.resolve()),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=None)
        return pipe

    def infer(self, pipeline: Any, payload: dict[str, Any]) -> Image.Image:
        images = payload_to_images(payload)
        device = "cuda"
        seed = int(payload.get("seed", 0))
        generator = torch.Generator(device=device).manual_seed(seed)
        kwargs: dict[str, Any] = {
            "prompt": payload["prompt"],
            "generator": generator,
            "num_inference_steps": int(payload.get("num_inference_steps", 28)),
            "guidance_scale": float(payload.get("guidance_scale", 4.0)),
        }
        if images:
            kwargs["image"] = images[0].convert("RGB")
        with torch.inference_mode():
            out = pipeline(**kwargs)
        return out.images[0]
