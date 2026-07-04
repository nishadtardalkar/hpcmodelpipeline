"""TRELLIS.2 4B — image-to-3D with PBR materials.

https://huggingface.co/microsoft/TRELLIS.2-4B
Requires: pip install the TRELLIS.2 repo (https://github.com/microsoft/TRELLIS.2)
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from PIL import Image

from job_crypto import payload_to_images
from models.base import ModelBackend
from models.hf_utils import snapshot_download_repo

HF_ID = "microsoft/TRELLIS.2-4B"


class Trellis24BBackend(ModelBackend):
    id = "trellis2_4b"
    display_name = "TRELLIS.2 4B"
    hf_id = HF_ID
    category = "image_to_3d"
    image_mode = "required"
    output_kind = "model3d"
    output_mime = "model/gltf-binary"

    def download(self, dest: Path) -> None:
        snapshot_download_repo(dest, HF_ID)

    def load(self, path: Path) -> Any:
        try:
            from trellis2.pipelines import Trellis2ImageTo3DPipeline
        except ImportError as exc:
            raise ImportError(
                "TRELLIS.2 requires the official package. "
                "See https://github.com/microsoft/TRELLIS.2 for install steps."
            ) from exc

        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(str(path.resolve()))
        pipeline.cuda()
        return pipeline

    def infer(self, pipeline: Any, payload: dict[str, Any]) -> bytes:
        try:
            import o_voxel
        except ImportError as exc:
            raise ImportError(
                "TRELLIS.2 export needs o_voxel. "
                "Install from https://github.com/microsoft/TRELLIS.2"
            ) from exc

        images = payload_to_images(payload)
        if not images:
            raise ValueError("TRELLIS.2 requires one reference image")
        image = images[0].convert("RGB")
        resolution = int(payload.get("resolution", 1024))

        run_kwargs: dict[str, Any] = {}
        if resolution in (512, 1024, 1536):
            run_kwargs["resolution"] = resolution
        meshes = pipeline.run(image, **run_kwargs)
        mesh = meshes[0] if isinstance(meshes, (list, tuple)) else meshes
        mesh.simplify(16_777_216)

        glb = o_voxel.postprocess.to_glb(
            vertices=mesh.vertices,
            faces=mesh.faces,
            attr_volume=mesh.attrs,
            coords=mesh.coords,
            attr_layout=mesh.layout,
            voxel_size=mesh.voxel_size,
            aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
            decimation_target=1_000_000,
            texture_size=4096,
            remesh=True,
            remesh_band=1,
            remesh_project=0,
            verbose=False,
        )
        buf = io.BytesIO()
        glb.export(buf, extension_webp=True)
        return buf.getvalue()
