"""GPU worker: decrypt jobs, run selected model backend, write encrypted outputs."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import sys
import time
import traceback
import uuid
from pathlib import Path
from typing import Any

from job_crypto import pack_output_png, parse_session_key, unpack_job_envelope
from models import DEFAULT_MODEL_ID, DEFAULT_MODELS_BASE, get_backend, list_models, weights_dir
from models.base import ModelBackend

DEFAULT_PIPELINE_ROOT = Path("./jobs/")

SHARED_ROOT: Path
JOBS_PENDING: Path
JOBS_PROCESSING: Path
JOBS_DONE: Path
JOBS_FAILED: Path
ARTIFACTS: Path
POLL_SECONDS: float


def _configure_paths(pipeline_root: Path, poll_seconds: float) -> None:
    global SHARED_ROOT, JOBS_PENDING, JOBS_PROCESSING, JOBS_DONE, JOBS_FAILED
    global ARTIFACTS, POLL_SECONDS

    SHARED_ROOT = pipeline_root.resolve()
    JOBS_PENDING = SHARED_ROOT / "jobs" / "pending"
    JOBS_PROCESSING = SHARED_ROOT / "jobs" / "processing"
    JOBS_DONE = SHARED_ROOT / "jobs" / "done"
    JOBS_FAILED = SHARED_ROOT / "jobs" / "failed"
    ARTIFACTS = SHARED_ROOT / "artifacts"
    POLL_SECONDS = poll_seconds


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def ensure_dirs() -> None:
    for path in [JOBS_PENDING, JOBS_PROCESSING, JOBS_DONE, JOBS_FAILED, ARTIFACTS]:
        _private_dir(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Encrypted GPU worker."
    )
    parser.add_argument(
        "--model",
        choices=sorted({m["id"] for m in list_models()}),
        default=DEFAULT_MODEL_ID,
        help="Model backend to run (must match job payload model field).",
    )
    parser.add_argument(
        "--models-base",
        type=Path,
        default=DEFAULT_MODELS_BASE,
        help="Base directory for model weights (default: <base>/<model_id>).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Local weights directory (default: <models-base>/<model>).",
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=DEFAULT_PIPELINE_ROOT,
        help="Shared job directory (must match login node).",
    )
    parser.add_argument(
        "--session-key",
        default=None,
        help="Base64 session key (required for worker; omit with --download-only).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Download model weights and exit (CPU only, no CUDA).",
    )
    parser.add_argument(
        "--download-path",
        type=Path,
        default=None,
        help="Directory to save weights (default: <models-base>/<model>).",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=2.0,
        help="Seconds between pending-job checks.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="Print available models and exit.",
    )
    return parser.parse_args()


def write_status(
    job_id: str,
    status: str,
    *,
    error: str | None = None,
    output_kind: str | None = None,
) -> None:
    job_dir = ARTIFACTS / job_id
    _private_dir(job_dir)
    status_path = job_dir / "status.json"
    body: dict[str, Any] = {"job_id": job_id, "status": status}
    if error:
        body["error"] = error[:2000]
    if output_kind:
        body["output_kind"] = output_kind
    status_path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    status_path.chmod(0o600)


def process_job(
    backend: ModelBackend,
    expected_model: str,
    pipeline: Any,
    session_key: bytes,
    job_path: Path,
) -> None:
    processing_path = JOBS_PROCESSING / job_path.name
    try:
        job_path.rename(processing_path)
    except FileNotFoundError:
        return

    job_id = job_path.stem.removesuffix(".job")
    write_status(job_id, "processing")

    try:
        payload = unpack_job_envelope(session_key, processing_path.read_bytes())
        job_model = payload.get("model", DEFAULT_MODEL_ID)
        if job_model != expected_model:
            raise ValueError(
                f"Job model {job_model!r} does not match worker --model {expected_model!r}"
            )

        out = backend.infer(pipeline, payload)

        if isinstance(out, str):
            output_bytes = out.encode("utf-8")
        else:
            buf = io.BytesIO()
            out.save(buf, format="PNG")
            output_bytes = buf.getvalue()
        output_enc = pack_output_png(session_key, output_bytes)

        job_dir = ARTIFACTS / job_id
        _private_dir(job_dir)
        output_path = job_dir / "output.enc"
        output_path.write_bytes(output_enc)
        output_path.chmod(0o600)

        write_status(job_id, "done", output_kind=backend.output_kind)
        shutil.move(str(processing_path), str(JOBS_DONE / processing_path.name))
        print(f"completed {job_id}")
    except Exception as exc:  # pragma: no cover
        err = f"{type(exc).__name__}: {exc}"
        write_status(job_id, "failed", error=err)
        shutil.move(str(processing_path), str(JOBS_FAILED / processing_path.name))
        print(f"failed {job_id}: {err}", file=sys.stderr)
        traceback.print_exc()


def next_pending_job() -> Path | None:
    jobs = sorted(JOBS_PENDING.glob("*.job.enc"), key=lambda p: p.stat().st_mtime)
    return jobs[0] if jobs else None


def main() -> None:
    args = parse_args()

    if args.list_models:
        for spec in list_models(args.models_base):
            print(json.dumps(spec, indent=2))
        return

    backend = get_backend(args.model)
    weights_path = (
        args.model_path.resolve()
        if args.model_path
        else weights_dir(args.model, args.models_base).resolve()
    )

    if args.download_only:
        dest = (
            args.download_path.resolve()
            if args.download_path
            else weights_dir(args.model, args.models_base).resolve()
        )
        backend.download(dest)
        return

    if not args.session_key:
        raise SystemExit("error: --session-key is required unless --download-only")

    _configure_paths(args.pipeline_root, args.poll_seconds)
    session_key = parse_session_key(args.session_key)
    worker_id = uuid.uuid4().hex[:8]
    print(f"model={backend.id} path={weights_path}")
    print(f"pipeline_root={SHARED_ROOT}")
    pipeline = backend.load(weights_path)

    ensure_dirs()
    print(f"gpu worker {worker_id} watching {JOBS_PENDING}")

    while True:
        job = next_pending_job()
        if not job:
            time.sleep(POLL_SECONDS)
            continue
        process_job(backend, args.model, pipeline, session_key, job)


if __name__ == "__main__":
    main()
