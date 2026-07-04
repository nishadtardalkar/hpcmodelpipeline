"""Hunyuan3D-2.1 Paint PBR — texture painting for 3D meshes.

https://huggingface.co/tencent/Hunyuan3D-2.1/tree/main/hunyuan3d-paintpbr-v2-1
Requires: Hunyuan3D-2 package (https://github.com/Tencent/Hunyuan3D-2)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image

from job_crypto import payload_to_images, payload_to_mesh_bytes
from models.base import ModelBackend
from models.hf_utils import snapshot_download_repo

HF_ID = "tencent/Hunyuan3D-2.1"
PAINT_SUBFOLDER = "hunyuan3d-paintpbr-v2-1"


class Hunyuan3DPaintBackend(ModelBackend):
    id = "hunyuan3d_paintpbr"
    display_name = "Hunyuan3D Paint PBR"
    hf_id = f"{HF_ID}/{PAINT_SUBFOLDER}"
    category = "mesh_texture"
    image_mode = "required"
    requires_mesh = True
    output_kind = "model3d"
    output_mime = "model/gltf-binary"

    def download(self, dest: Path) -> None:
        snapshot_download_repo(
            dest,
            HF_ID,
            allow_patterns=[f"{PAINT_SUBFOLDER}/**", "README.md"],
        )

    def load(self, path: Path) -> Any:
        try:
            from hy3dgen.texgen import Hunyuan3DPaintPipeline
        except ImportError as exc:
            raise ImportError(
                "Hunyuan3D Paint requires hy3dgen. "
                "See https://github.com/Tencent/Hunyuan3D-2 for install steps."
            ) from exc

        paint_path = path.resolve() / PAINT_SUBFOLDER
        if not paint_path.is_dir():
            raise FileNotFoundError(
                f"Missing paint weights at {paint_path}. Re-run --download-only."
            )
        pipeline = Hunyuan3DPaintPipeline.from_pretrained(str(paint_path))
        pipeline.to("cuda")
        return pipeline

    def infer(self, pipeline: Any, payload: dict[str, Any]) -> bytes:
        images = payload_to_images(payload)
        mesh_bytes = payload_to_mesh_bytes(payload)
        if not images:
            raise ValueError("Hunyuan3D Paint requires a reference image")
        if not mesh_bytes:
            raise ValueError("Hunyuan3D Paint requires a mesh file (GLB or OBJ)")

        ref_image = images[0].convert("RGB")
        texture_size = int(payload.get("texture_size", 4096))

        with io.BytesIO(mesh_bytes) as mesh_buf:
            result = pipeline(
                mesh=mesh_buf,
                image=ref_image,
                prompt=payload.get("prompt") or "",
                texture_size=texture_size,
            )

        if isinstance(result, bytes):
            return result
        if hasattr(result, "export"):
            buf = io.BytesIO()
            result.export(buf)
            return buf.getvalue()
        if isinstance(result, (str, Path)):
            return Path(result).read_bytes()
        raise TypeError(f"Unexpected paint pipeline output: {type(result)}")
