"""ACE-Step 1.5 — text-to-music generation.

https://huggingface.co/ACE-Step/Ace-Step1.5
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import torch

from models.base import ModelBackend
from models.hf_utils import snapshot_download_repo

HF_ID = "ACE-Step/Ace-Step1.5"


def _audio_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    import soundfile as sf

    buf = io.BytesIO()
    if audio.ndim == 1:
        sf.write(buf, audio, sample_rate, format="WAV")
    else:
        sf.write(buf, audio.T, sample_rate, format="WAV")
    return buf.getvalue()


class AceStepBackend(ModelBackend):
    id = "ace_step_15"
    display_name = "ACE-Step 1.5"
    hf_id = HF_ID
    category = "text_to_audio"
    image_mode = "none"
    output_kind = "audio"
    output_mime = "audio/wav"

    def ui_config(self) -> dict[str, Any]:
        cfg = super().ui_config()
        cfg["defaults"]["num_inference_steps"] = 8
        cfg["defaults"]["guidance_scale"] = 1.0
        cfg["defaults"]["audio_duration_s"] = 60
        return cfg

    def download(self, dest: Path) -> None:
        snapshot_download_repo(dest, HF_ID)

    def load(self, path: Path) -> Any:
        from transformers import pipeline

        pipe = pipeline(
            "text-to-audio",
            model=str(path.resolve()),
            trust_remote_code=True,
            device="cuda",
        )
        return pipe

    def infer(self, pipeline: Any, payload: dict[str, Any]) -> bytes:
        duration = float(payload.get("audio_duration_s", 60))
        seed = int(payload.get("seed", 0))
        generator = torch.Generator(device="cuda").manual_seed(seed)
        kwargs: dict[str, Any] = {
            "text_inputs": payload["prompt"],
            "generate_kwargs": {
                "max_new_tokens": None,
            },
        }
        steps = int(payload.get("num_inference_steps", 8))
        if steps:
            kwargs["forward_params"] = {"num_inference_steps": steps}

        with torch.inference_mode():
            out = pipeline(**kwargs)

        audio = out["audio"]
        if isinstance(audio, torch.Tensor):
            audio = audio.squeeze().float().cpu().numpy()
        elif isinstance(audio, list):
            audio = np.asarray(audio[0], dtype=np.float32)
        sample_rate = int(out.get("sampling_rate", 44100))
        return _audio_to_wav_bytes(audio, sample_rate)
