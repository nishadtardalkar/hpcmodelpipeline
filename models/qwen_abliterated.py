"""Qwen-Edit-2509-Abliterated (community NSFW merge, 4-step).

Checkpoint: https://huggingface.co/jiangchengchengNLP/Qwen-Edit-2509-abliterated
Base:       https://huggingface.co/Qwen/Qwen-Image-Edit-2509
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

BASE_HF_ID = "Qwen/Qwen-Image-Edit-2509"
CKPT_REPO = "jiangchengchengNLP/Qwen-Edit-2509-abliterated"
# Prefer v1.2 FP8 diffusion weights (ComfyUI UNETLoader name), then older root file.
CKPT_FILES = [
    "v1.2/qwen-image-edit-abliterated-v1.2-fp8-4steps.safetensors",
    "qwen-image-edit-abliterated-v1.2-fp8-4steps.safetensors",
    "v1.2/Qwen-Edit-abliterated-4step-v1.safetensors",
    "Qwen-Edit-abliterated-4step-v1.safetensors",
]


class QwenAbliteratedBackend(ModelBackend):
    id = "qwen_abliterated"
    display_name = "Qwen Edit 2509 Abliterated"
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
