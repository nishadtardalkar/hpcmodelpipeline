"""FLUX.1-dev + whrw/Flux-Uncensored-V2 LoRA (image-to-image).

Base: https://huggingface.co/black-forest-labs/FLUX.1-dev
LoRA: https://huggingface.co/whrw/Flux-Uncensored-V2
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from diffusers import FluxImg2ImgPipeline
from PIL import Image

from job_crypto import payload_to_images
from models.base import ModelBackend

BASE_HF_ID = "black-forest-labs/FLUX.1-dev"
LORA_HF_ID = "whrw/Flux-Uncensored-V2"
LORA_WEIGHT_NAME = "lora.safetensors"


class FluxUncensoredBackend(ModelBackend):
    id = "flux_uncensored"
    display_name = "Flux Uncensored V2"
    hf_id = LORA_HF_ID
    requires_images = True

    def download(self, dest: Path) -> None:
        dest = dest.resolve()
        base_dir = dest / "base"
        lora_dir = dest / "lora"
        base_dir.mkdir(parents=True, exist_ok=True)
        lora_dir.mkdir(parents=True, exist_ok=True)

        print(f"Downloading base {BASE_HF_ID} -> {base_dir} (CPU only, large download)")
        print("Note: accept the FLUX.1-dev license on Hugging Face and set HF_TOKEN if needed.")
        pipe = FluxImg2ImgPipeline.from_pretrained(
            BASE_HF_ID,
            torch_dtype=torch.float32,
            local_files_only=False,
        )
        pipe.save_pretrained(base_dir)
        print(f"Base saved to {base_dir}")

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError("pip install huggingface_hub") from exc

        print(f"Downloading LoRA {LORA_HF_ID} -> {lora_dir}")
        snapshot_download(repo_id=LORA_HF_ID, local_dir=str(lora_dir))
        print(f"LoRA saved to {lora_dir}")

    def load(self, path: Path) -> Any:
        base_dir = path / "base"
        lora_dir = path / "lora"
        lora_file = lora_dir / LORA_WEIGHT_NAME
        if not base_dir.is_dir():
            raise FileNotFoundError(
                f"Missing {base_dir}. Run: python inference.py --download-only --model {self.id}"
            )
        if not lora_file.is_file():
            raise FileNotFoundError(
                f"Missing {lora_file}. Run: python inference.py --download-only --model {self.id}"
            )

        print(f"Loading FLUX img2img from {base_dir}")
        pipe = FluxImg2ImgPipeline.from_pretrained(
            str(base_dir),
            torch_dtype=torch.bfloat16,
            local_files_only=True,
        )
        pipe.load_lora_weights(str(lora_dir), weight_name=LORA_WEIGHT_NAME)
        pipe.to("cuda", dtype=torch.bfloat16)
        pipe.set_progress_bar_config(disable=None)
        return pipe

    def infer(self, pipeline: Any, payload: dict[str, Any]) -> Image.Image:
        images = payload_to_images(payload)
        if not images:
            raise ValueError("Flux jobs require at least one input image")
        input_image = images[0].convert("RGB")
        generator = torch.manual_seed(int(payload.get("seed", 0)))
        inputs = {
            "image": input_image,
            "prompt": payload["prompt"],
            "generator": generator,
            "num_inference_steps": int(payload.get("num_inference_steps", 28)),
            "guidance_scale": float(payload.get("guidance_scale", 3.5)),
            "strength": float(payload.get("strength", 0.85)),
        }

        with torch.inference_mode():
            out = pipeline(**inputs)
        return out.images[0]
