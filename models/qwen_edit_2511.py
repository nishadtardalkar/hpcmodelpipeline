"""Official Qwen Image Edit 2511.

https://huggingface.co/Qwen/Qwen-Image-Edit-2511
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from PIL import Image

from job_crypto import payload_to_images
from models.base import ModelBackend
from models.hf_utils import snapshot_download_repo

HF_ID = "Qwen/Qwen-Image-Edit-2511"


class QwenEdit2511Backend(ModelBackend):
    id = "qwen_edit_2511"
    display_name = "Qwen Image Edit 2511"
    hf_id = HF_ID
    category = "image_edit"
    image_mode = "required"

    def download(self, dest: Path) -> None:
        snapshot_download_repo(dest, HF_ID)

    def load(self, path: Path) -> Any:
        from diffusers import QwenImageEditPlusPipeline

        pipe = QwenImageEditPlusPipeline.from_pretrained(
            str(path.resolve()),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=None)
        return pipe

    def ui_config(self) -> dict[str, Any]:
        cfg = super().ui_config()
        cfg["defaults"] = {
            "num_inference_steps": 40,
            "guidance_scale": 1.0,
            "true_cfg_scale": 4.0,
        }
        return cfg

    def infer(self, pipeline: Any, payload: dict[str, Any]) -> Image.Image:
        images = [img.convert("RGB") for img in payload_to_images(payload)]
        if not images:
            raise ValueError("Qwen Edit jobs require at least one input image")
        generator = torch.manual_seed(int(payload.get("seed", 0)))
        inputs: dict[str, Any] = {
            "image": images if len(images) > 1 else images[0],
            "prompt": payload["prompt"],
            "generator": generator,
            "true_cfg_scale": float(payload.get("true_cfg_scale", 4.0)),
            "negative_prompt": payload.get("negative_prompt") or " ",
            "num_inference_steps": int(payload.get("num_inference_steps", 40)),
            "guidance_scale": float(payload.get("guidance_scale", 1.0)),
            "num_images_per_prompt": int(payload.get("num_images_per_prompt", 1)),
        }
        with torch.inference_mode():
            out = pipeline(**inputs)
        return out.images[0]
