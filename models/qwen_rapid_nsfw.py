"""Qwen Image Edit Rapid AIO NSFW v23 (community 4-step merge).

Checkpoint: https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO (v23 NSFW)
Base:       https://huggingface.co/Qwen/Qwen-Image-Edit-2511
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image

from models.base import ModelBackend
from models.qwen_edit_ckpt import (
    download_base_and_checkpoint,
    infer_qwen_edit,
    load_qwen_edit_pipeline,
)

BASE_HF_ID = "Qwen/Qwen-Image-Edit-2511"
CKPT_REPO = "Phr00t/Qwen-Image-Edit-Rapid-AIO"
CKPT_FILES = [
    "v23/Qwen-Rapid-AIO-NSFW-v23.safetensors",
]


class QwenRapidNsfwBackend(ModelBackend):
    id = "qwen_rapid_nsfw"
    display_name = "Qwen Rapid AIO NSFW v23"
    hf_id = CKPT_REPO
    requires_images = True

    def download(self, dest: Path) -> None:
        download_base_and_checkpoint(
            dest,
            base_hf_id=BASE_HF_ID,
            ckpt_repo=CKPT_REPO,
            ckpt_files=CKPT_FILES,
        )

    def load(self, path: Path) -> Any:
        return load_qwen_edit_pipeline(path)

    def infer(self, pipeline: Any, payload: dict[str, Any]) -> Image.Image:
        return infer_qwen_edit(
            pipeline,
            payload,
            default_steps=4,
            default_guidance=1.0,
            default_true_cfg=1.0,
        )
