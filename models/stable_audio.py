"""Stable Audio Open 1.0 — text-to-audio / sound effects.

https://huggingface.co/stabilityai/stable-audio-open-1.0
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import torch

from models.base import ModelBackend
from models.hf_utils import snapshot_download_repo

HF_ID = "stabilityai/stable-audio-open-1.0"


class StableAudioOpenBackend(ModelBackend):
    id = "stable_audio_open"
    display_name = "Stable Audio Open 1.0"
    hf_id = HF_ID
    category = "text_to_audio"
    image_mode = "none"
    output_kind = "audio"
    output_mime = "audio/wav"

    def ui_config(self) -> dict[str, Any]:
        cfg = super().ui_config()
        cfg["defaults"]["num_inference_steps"] = 100
        cfg["defaults"]["guidance_scale"] = 7.0
        cfg["defaults"]["audio_duration_s"] = 30
        return cfg

    def download(self, dest: Path) -> None:
        snapshot_download_repo(dest, HF_ID)

    def load(self, path: Path) -> Any:
        from diffusers import StableAudioPipeline

        pipe = StableAudioPipeline.from_pretrained(
            str(path.resolve()),
            torch_dtype=torch.float16,
            local_files_only=True,
        )
        pipe.to("cuda")
        pipe.set_progress_bar_config(disable=None)
        return pipe

    def infer(self, pipeline: Any, payload: dict[str, Any]) -> bytes:
        import soundfile as sf

        device = "cuda"
        seed = int(payload.get("seed", 0))
        generator = torch.Generator(device=device).manual_seed(seed)
        duration = float(payload.get("audio_duration_s", 30))
        duration = max(1.0, min(duration, 47.0))

        with torch.inference_mode():
            out = pipeline(
                payload["prompt"],
                negative_prompt=payload.get("negative_prompt") or "Low quality.",
                num_inference_steps=int(payload.get("num_inference_steps", 100)),
                guidance_scale=float(payload.get("guidance_scale", 7.0)),
                audio_end_in_s=duration,
                generator=generator,
            )

        audio = out.audios[0].T.float().cpu().numpy()
        buf = io.BytesIO()
        sf.write(buf, audio, pipeline.vae.sampling_rate, format="WAV")
        return buf.getvalue()
