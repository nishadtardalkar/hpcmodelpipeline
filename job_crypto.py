"""Encrypted job/output envelopes using a shared ephemeral session key."""

from __future__ import annotations

import base64
import io
import json
import os
import secrets
from typing import Any

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from PIL import Image

JOB_MAGIC = b"HPCENC02"
OUT_MAGIC = b"HPCOUT01"
NONCE_LEN = 12
KEY_LEN = 32


def generate_session_key() -> str:
    """Return a copy-pasteable base64 session key (32 random bytes)."""
    return base64.b64encode(secrets.token_bytes(KEY_LEN)).decode("ascii")


def parse_session_key(key_b64: str) -> bytes:
    """Decode a session key from base64; raises ValueError if invalid."""
    try:
        raw = base64.b64decode(key_b64.strip(), validate=True)
    except Exception as exc:
        raise ValueError("Invalid session key (expected base64)") from exc
    if len(raw) != KEY_LEN:
        raise ValueError(f"Session key must be {KEY_LEN} bytes after decoding")
    return raw


def print_session_key_banner(session_key_b64: str) -> None:
    border = "=" * 60
    print(border)
    print("SESSION KEY — paste into browser UI and GPU worker --session-key")
    print()
    print(session_key_b64)
    print()
    print(border, flush=True)


def _aes_encrypt(key: bytes, plaintext: bytes) -> tuple[bytes, bytes]:
    nonce = os.urandom(NONCE_LEN)
    ciphertext = AESGCM(key).encrypt(nonce, plaintext, None)
    return nonce, ciphertext


def _aes_decrypt(key: bytes, nonce: bytes, ciphertext: bytes) -> bytes:
    return AESGCM(key).decrypt(nonce, ciphertext, None)


def pack_job_envelope(session_key: bytes, payload: dict[str, Any]) -> bytes:
    plaintext = json.dumps(payload).encode("utf-8")
    nonce, ciphertext = _aes_encrypt(session_key, plaintext)
    return JOB_MAGIC + nonce + ciphertext


def unpack_job_envelope(session_key: bytes, blob: bytes) -> dict[str, Any]:
    if len(blob) < len(JOB_MAGIC) + NONCE_LEN:
        raise ValueError("Job blob too short")
    if blob[: len(JOB_MAGIC)] != JOB_MAGIC:
        raise ValueError("Invalid job magic")
    nonce = blob[len(JOB_MAGIC) : len(JOB_MAGIC) + NONCE_LEN]
    ciphertext = blob[len(JOB_MAGIC) + NONCE_LEN :]
    return json.loads(_aes_decrypt(session_key, nonce, ciphertext).decode("utf-8"))


def pack_output_png(session_key: bytes, png_bytes: bytes) -> bytes:
    nonce, ciphertext = _aes_encrypt(session_key, png_bytes)
    return OUT_MAGIC + nonce + ciphertext


def unpack_output_png(session_key: bytes, blob: bytes) -> bytes:
    if len(blob) < len(OUT_MAGIC) + NONCE_LEN:
        raise ValueError("Output blob too short")
    if blob[: len(OUT_MAGIC)] != OUT_MAGIC:
        raise ValueError("Invalid output magic")
    nonce = blob[len(OUT_MAGIC) : len(OUT_MAGIC) + NONCE_LEN]
    ciphertext = blob[len(OUT_MAGIC) + NONCE_LEN :]
    return _aes_decrypt(session_key, nonce, ciphertext)


def payload_to_images(payload: dict[str, Any]) -> list[Image.Image]:
    images: list[Image.Image] = []
    for entry in payload.get("images", []):
        raw = base64.b64decode(entry["data_b64"])
        images.append(Image.open(io.BytesIO(raw)))
    return images


def payload_to_mesh_bytes(payload: dict[str, Any]) -> bytes | None:
    mesh = payload.get("mesh")
    if not mesh or not mesh.get("data_b64"):
        return None
    return base64.b64decode(mesh["data_b64"])
