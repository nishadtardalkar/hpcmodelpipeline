"""Qwen image-edit (requires input image(s))."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers import DiffusionPipeline
from PIL import Image

from job_crypto import payload_to_images
from models.base import ModelBackend


class QwenBackend(ModelBackend):
    id = "qwen"
    display_name = "Qwen Image Edit"
    hf_id = "Qwen/Qwen-Image-Edit-2511"
    requires_images = True

    def download(self, dest: Path) -> None:
        dest = dest.resolve()
        dest.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {self.hf_id} -> {dest} (CPU only)")
        pipe = DiffusionPipeline.from_pretrained(
            self.hf_id,
            torch_dtype=torch.float32,
            local_files_only=False,
        )
        pipe.save_pretrained(dest)
        print(f"Saved to {dest}")

    def load(self, path: Path) -> Any:
        pipe = DiffusionPipeline.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.to("cuda", dtype=torch.bfloat16)
        pipe.set_progress_bar_config(disable=None)
        return pipe

    def infer(self, pipeline: Any, payload: dict[str, Any]) -> Image.Image:
        images = payload_to_images(payload)
        if not images:
            raise ValueError("Qwen jobs require at least one input image")
        input_image = images[0].convert("RGB")
        generator = torch.manual_seed(int(payload.get("seed", 0)))
        inputs = {
            "image": input_image,
            "prompt": payload["prompt"],
            "generator": generator,
            "true_cfg_scale": float(payload.get("true_cfg_scale", 4.0)),
            "negative_prompt": payload.get("negative_prompt", " "),
            "num_inference_steps": int(payload.get("num_inference_steps", 40)),
            "guidance_scale": float(payload.get("guidance_scale", 1.0)),
            "num_images_per_prompt": int(payload.get("num_images_per_prompt", 1)),
        }
        with torch.inference_mode():
            out = pipeline(**inputs)
        return out.images[0]
