"""Shared helpers for Qwen Image Edit community checkpoints (single-file / AIO)."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from job_crypto import payload_to_images

CHECKPOINT_NAME = "checkpoint.safetensors"

# Common ComfyUI / AIO prefixes for the diffusion transformer.
_TRANSFORMER_PREFIXES = (
    "model.diffusion_model.",
    "diffusion_model.",
    "model.",
    "transformer.",
)


def download_base_and_checkpoint(
    dest: Path,
    *,
    base_hf_id: str,
    ckpt_repo: str,
    ckpt_files: list[str],
) -> None:
    """Download official base (configs + VAE/text encoder) and one community checkpoint."""
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub") from exc

    dest = dest.resolve()
    base_dir = dest / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading base {base_hf_id} -> {base_dir}")
    snapshot_download(repo_id=base_hf_id, local_dir=str(base_dir))

    last_error: Exception | None = None
    ckpt_path = dest / CHECKPOINT_NAME
    dl_root = dest / "_hf_dl"
    for filename in ckpt_files:
        print(f"Trying checkpoint {ckpt_repo}/{filename}")
        try:
            downloaded = Path(
                hf_hub_download(
                    repo_id=ckpt_repo,
                    filename=filename,
                    local_dir=str(dl_root),
                )
            )
            if ckpt_path.exists():
                ckpt_path.unlink()
            shutil.move(str(downloaded), str(ckpt_path))
            if dl_root.is_dir():
                shutil.rmtree(dl_root, ignore_errors=True)
            print(f"Checkpoint saved to {ckpt_path}")
            return
        except Exception as exc:  # pragma: no cover
            last_error = exc
            print(f"  not found / failed: {exc}")

    raise FileNotFoundError(
        f"Could not download any of {ckpt_files!r} from {ckpt_repo}: {last_error}"
    )


def _strip_prefix(state_dict: dict[str, Any]) -> dict[str, Any]:
    """If all keys share a known ComfyUI prefix, strip it."""
    keys = list(state_dict)
    if not keys:
        return state_dict
    for prefix in _TRANSFORMER_PREFIXES:
        if all(k.startswith(prefix) for k in keys):
            return {k[len(prefix) :]: v for k, v in state_dict.items()}
    # AIO: keep only diffusion_model.* keys and strip that prefix.
    for prefix in ("model.diffusion_model.", "diffusion_model."):
        subset = {k[len(prefix) :]: v for k, v in state_dict.items() if k.startswith(prefix)}
        if subset:
            return subset
    return state_dict


def _load_transformer(ckpt_path: Path, base_dir: Path) -> Any:
    from diffusers import QwenImageTransformer2DModel
    from safetensors.torch import load_file

    errors: list[str] = []

    for kwargs in (
        {
            "config": str(base_dir),
            "subfolder": "transformer",
            "torch_dtype": torch.bfloat16,
        },
        {
            "config": str(base_dir / "transformer"),
            "torch_dtype": torch.bfloat16,
        },
        {"torch_dtype": torch.bfloat16},
    ):
        try:
            return QwenImageTransformer2DModel.from_single_file(str(ckpt_path), **kwargs)
        except Exception as exc:  # pragma: no cover
            errors.append(f"from_single_file({kwargs}): {exc}")

    # Fallback: load state dict, strip ComfyUI prefixes, load into config from base.
    try:
        config_dir = base_dir / "transformer"
        transformer = QwenImageTransformer2DModel.from_config(str(config_dir))
        state = load_file(str(ckpt_path))
        state = _strip_prefix(state)
        # Drop non-tensor / unexpected keys quietly where possible.
        missing, unexpected = transformer.load_state_dict(state, strict=False)
        if missing:
            print(f"qwen_edit: missing {len(missing)} keys after load (showing up to 5): {missing[:5]}")
        if unexpected:
            print(
                f"qwen_edit: unexpected {len(unexpected)} keys after load "
                f"(showing up to 5): {unexpected[:5]}"
            )
        return transformer.to(dtype=torch.bfloat16)
    except Exception as exc:  # pragma: no cover
        errors.append(f"state_dict fallback: {exc}")

    raise RuntimeError(
        "Failed to load Qwen transformer from checkpoint. "
        "This file may be ComfyUI-only or need a newer diffusers.\n"
        + "\n".join(errors)
    )


def load_qwen_edit_pipeline(path: Path) -> Any:
    """Load base pipeline components and swap in the community transformer weights."""
    from diffusers import DiffusionPipeline

    path = path.resolve()
    base_dir = path / "base"
    ckpt_path = path / CHECKPOINT_NAME
    if not base_dir.is_dir():
        raise FileNotFoundError(f"Missing base weights at {base_dir}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")

    print(f"Loading transformer from {ckpt_path}")
    transformer = _load_transformer(ckpt_path, base_dir)
    print(f"Loading pipeline shell from {base_dir}")
    pipe = DiffusionPipeline.from_pretrained(
        str(base_dir),
        transformer=transformer,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    pipe.to("cuda", dtype=torch.bfloat16)
    pipe.set_progress_bar_config(disable=None)
    return pipe


def infer_qwen_edit(
    pipeline: Any,
    payload: dict[str, Any],
    *,
    default_steps: int = 4,
    default_guidance: float = 1.0,
    default_true_cfg: float = 1.0,
) -> Image.Image:
    images = payload_to_images(payload)
    if not images:
        raise ValueError("Qwen Edit jobs require at least one input image")
    input_image = images[0].convert("RGB")
    generator = torch.manual_seed(int(payload.get("seed", 0)))
    inputs: dict[str, Any] = {
        "image": input_image,
        "prompt": payload["prompt"],
        "generator": generator,
        "true_cfg_scale": float(payload.get("true_cfg_scale", default_true_cfg)),
        "negative_prompt": payload.get("negative_prompt") or " ",
        "num_inference_steps": int(payload.get("num_inference_steps", default_steps)),
        "guidance_scale": float(payload.get("guidance_scale", default_guidance)),
        "num_images_per_prompt": int(payload.get("num_images_per_prompt", 1)),
    }
    with torch.inference_mode():
        out = pipeline(**inputs)
    return out.images[0]
