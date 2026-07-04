"""Shared Hugging Face download helpers."""

from __future__ import annotations

from pathlib import Path


def snapshot_download_repo(
    dest: Path,
    repo_id: str,
    *,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ImportError("pip install huggingface_hub") from exc

    dest = dest.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {repo_id} -> {dest}")
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
    )
