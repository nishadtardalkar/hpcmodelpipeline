"""Florence-2-Flux-Large: image → text (caption / description for Flux workflows).

https://huggingface.co/gokaygokay/Florence-2-Flux-Large
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

from job_crypto import payload_to_images
from models.base import ModelBackend

HF_MODEL_ID = "gokaygokay/Florence-2-Flux-Large"
# Pin remote-code revision so HF does not silently refresh custom python files.
HF_REVISION = "ed3af3df6d23d9f25d1dd4ce05ba95bb43c37209"
DEFAULT_TASK = "<DESCRIPTION>"
DEFAULT_INSTRUCTION = "Describe this image in great detail."


def _answer_text(parsed: Any, task: str) -> str:
    if isinstance(parsed, dict):
        if task in parsed:
            return str(parsed[task])
        if len(parsed) == 1:
            return str(next(iter(parsed.values())))
        return json.dumps(parsed, ensure_ascii=False, indent=2)
    return str(parsed)


def _patch_florence2_remote_code(repo_dir: Path) -> None:
    """Vendor Florence-2 custom code breaks on transformers 4.50+ / 5.x."""
    patched: list[str] = []

    config_py = repo_dir / "configuration_florence2.py"
    if config_py.is_file():
        text = config_py.read_text(encoding="utf-8")
        needle = 'getattr(self, "forced_bos_token_id", None) is None'
        if needle not in text and "self.forced_bos_token_id is None" in text:
            text = text.replace(
                "self.forced_bos_token_id is None",
                needle,
                1,
            )
            config_py.write_text(text, encoding="utf-8")
            patched.append("configuration_florence2.py")

    model_py = repo_dir / "modeling_florence2.py"
    if model_py.is_file():
        text = model_py.read_text(encoding="utf-8")
        changed = False

        old_dpr = "[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths)*2)]"
        new_dpr = (
            "[drop_path_rate * i / max(sum(depths) * 2 - 1, 1) "
            "for i in range(sum(depths) * 2)]"
        )
        if old_dpr in text:
            text = text.replace(old_dpr, new_dpr, 1)
            changed = True

        if "return self.language_model._supports_sdpa" in text:
            text, count = re.subn(
                r"[ \t]*@property\n"
                r"[ \t]*def _supports_flash_attn_2\(self\):.*?"
                r"return self\.language_model\._supports_flash_attn_2\n\n"
                r"[ \t]*@property\n"
                r"[ \t]*def _supports_sdpa\(self\):.*?"
                r"return self\.language_model\._supports_sdpa",
                " _supports_flash_attn_2 = False\n _supports_sdpa = True",
                text,
                count=1,
                flags=re.DOTALL,
            )
            changed = changed or count > 0

        if changed:
            model_py.write_text(text, encoding="utf-8")
            patched.append("modeling_florence2.py")

    if patched:
        print(
            "florence2_flux: patched vendor code for newer transformers: "
            + ", ".join(patched)
        )


def _fetch_florence2_repo(dest: Path) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub") from exc

    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=HF_MODEL_ID,
        revision=HF_REVISION,
        local_dir=str(dest),
    )
    _patch_florence2_remote_code(dest)
    return dest


def _load_florence2(path: Path) -> tuple[Any, Any]:
    _patch_florence2_remote_code(path)
    kwargs = {
        "trust_remote_code": True,
        "local_files_only": True,
    }
    model = AutoModelForCausalLM.from_pretrained(str(path), **kwargs)
    processor = AutoProcessor.from_pretrained(str(path), **kwargs)
    return model, processor


@dataclass
class Florence2FluxRuntime:
    model: Any
    processor: Any


class Florence2FluxBackend(ModelBackend):
    id = "florence2_flux"
    display_name = "Florence-2 Flux Large"
    hf_id = HF_MODEL_ID
    requires_images = True
    output_kind = "text"

    def download(self, dest: Path) -> None:
        print(
            f"Downloading {self.hf_id} (revision {HF_REVISION[:8]}) -> {dest} "
            "(CPU only, trust_remote_code=True)"
        )
        repo = _fetch_florence2_repo(dest)
        print(f"Saved to {repo}")

    def load(self, path: Path) -> Florence2FluxRuntime:
        if not torch.cuda.is_available():
            raise RuntimeError("Florence-2 inference requires CUDA.")
        print(f"Loading Florence-2 from {path}")
        model, processor = _load_florence2(path.resolve())
        model = model.to("cuda").eval()
        return Florence2FluxRuntime(model=model, processor=processor)

    def infer(self, runtime: Florence2FluxRuntime, payload: dict[str, Any]) -> str:
        images = payload_to_images(payload)
        if not images:
            raise ValueError("Florence-2 jobs require at least one input image")
        image = images[0].convert("RGB")

        task = str(payload.get("florence_task", DEFAULT_TASK))
        instruction = payload.get("prompt", DEFAULT_INSTRUCTION)
        prompt = task + instruction

        inputs = runtime.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to("cuda")

        max_new_tokens = int(payload.get("max_new_tokens", 1024))
        num_beams = int(payload.get("num_beams", 3))
        repetition_penalty = float(payload.get("repetition_penalty", 1.10))

        with torch.inference_mode():
            generated_ids = runtime.model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
            )

        generated_text = runtime.processor.batch_decode(
            generated_ids,
            skip_special_tokens=False,
        )[0]
        parsed = runtime.processor.post_process_generation(
            generated_text,
            task=task,
            image_size=(image.width, image.height),
        )
        return _answer_text(parsed, task)
