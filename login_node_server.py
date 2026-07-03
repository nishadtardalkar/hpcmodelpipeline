"""Login node gateway: stores encrypted job blobs only (never decrypts)."""

from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, render_template, request

from job_crypto import generate_session_key, print_session_key_banner
from models import DEFAULT_MODELS_BASE, list_models

APP_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = APP_DIR / "templates"

DEFAULT_PIPELINE_ROOT = Path(
    "./jobs/"
)

SHARED_ROOT: Path
JOBS_PENDING: Path
ARTIFACTS: Path

app = Flask(__name__, template_folder=str(TEMPLATES_DIR))
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB


def _configure(pipeline_root: Path) -> None:
    global SHARED_ROOT, JOBS_PENDING, ARTIFACTS

    SHARED_ROOT = pipeline_root.resolve()
    JOBS_PENDING = SHARED_ROOT / "jobs" / "pending"
    ARTIFACTS = SHARED_ROOT / "artifacts"


def _private_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    path.chmod(0o700)


def ensure_dirs() -> None:
    _private_dir(JOBS_PENDING)
    _private_dir(ARTIFACTS)


def write_status(job_id: str, status: str) -> None:
    job_dir = ARTIFACTS / job_id
    _private_dir(job_dir)
    status_path = job_dir / "status.json"
    status_path.write_text(
        json.dumps({"job_id": job_id, "status": status}, indent=2),
        encoding="utf-8",
    )
    status_path.chmod(0o600)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/models")
def models():
    return jsonify({"models": list_models(DEFAULT_MODELS_BASE)})


@app.post("/api/submit")
def submit():
    blob = request.get_data()
    if not blob:
        return jsonify({"error": "Empty encrypted payload."}), 400

    job_id = uuid.uuid4().hex
    job_path = JOBS_PENDING / f"{job_id}.job.enc"
    job_path.write_bytes(blob)
    job_path.chmod(0o600)
    write_status(job_id, "queued")

    return jsonify({"job_id": job_id, "status": "queued"})


@app.get("/api/status/<job_id>")
def status(job_id: str):
    status_path = ARTIFACTS / job_id / "status.json"
    if not status_path.exists():
        return jsonify({"error": "Job not found."}), 404
    return jsonify(json.loads(status_path.read_text(encoding="utf-8")))


@app.get("/api/result/<job_id>")
def result(job_id: str):
    output_path = ARTIFACTS / job_id / "output.enc"
    if not output_path.exists():
        return jsonify({"error": "Result not ready."}), 404
    return Response(
        output_path.read_bytes(),
        mimetype="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.output.enc"'},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Login node gateway for encrypted GPU jobs."
    )
    parser.add_argument(
        "--pipeline-root",
        type=Path,
        default=DEFAULT_PIPELINE_ROOT,
        help="Shared job directory (must match GPU worker).",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (use 127.0.0.1 with SSH tunnel).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="HTTP port.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    _configure(args.pipeline_root)

    index_template = TEMPLATES_DIR / "index.html"
    if not index_template.is_file():
        raise FileNotFoundError(f"Missing {index_template}")

    session_key = generate_session_key()
    print_session_key_banner(session_key)
    del session_key

    ensure_dirs()
    app.run(host=args.host, port=args.port)
