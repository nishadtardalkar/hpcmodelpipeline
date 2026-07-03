# HPC Encrypted Model Pipeline

Encrypted Qwen image-edit inference: browser encrypts jobs; login node stores ciphertext; GPU worker decrypts, runs the selected model, encrypts results.

## Architecture

```
models/
  base.py              # ModelBackend interface
  registry.py          # register backends, list_models()
  qwen_edit_ckpt.py    # shared single-file checkpoint loaders
  qwen_abliterated.py  # Qwen-Edit-2509-Abliterated
  qwen_rapid_nsfw.py   # Qwen Rapid AIO NSFW v23

inference.py           # GPU worker CLI (--model, --session-key, …)
login_node_server.py   # blind gateway + /api/models
job_crypto.py          # AES-GCM envelopes
templates/index.html   # model picker + encrypt in browser
```

| Component | Role |
|-----------|------|
| **Browser** | Pick model, encrypt payload (includes `model` id) |
| **Login node** | Store `.job.enc`, serve model list |
| **GPU worker** | `--model` must match job; one backend loaded per process |

## Models

| ID | Name | Images | Default path |
|----|------|--------|--------------|
| `qwen_abliterated` | Qwen Edit 2509 Abliterated | Required (1–2) | `./models/qwen_abliterated` |
| `qwen_rapid_nsfw` | Qwen Rapid AIO NSFW v23 | Required (1–2) | `./models/qwen_rapid_nsfw` |

Weights:

- [jiangchengchengNLP/Qwen-Edit-2509-abliterated](https://huggingface.co/jiangchengchengNLP/Qwen-Edit-2509-abliterated) on base [Qwen/Qwen-Image-Edit-2509](https://huggingface.co/Qwen/Qwen-Image-Edit-2509)
- [Phr00t/Qwen-Image-Edit-Rapid-AIO](https://huggingface.co/Phr00t/Qwen-Image-Edit-Rapid-AIO) NSFW v23 (`v23/Qwen-Rapid-AIO-NSFW-v23.safetensors`) on base [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)

Each download stores:

```
./models/<model_id>/
  base/                 # official Qwen-Image-Edit pipeline (VAE, text encoder, configs)
  checkpoint.safetensors  # community transformer / AIO weights
```

Both models are tuned for **4-step** sampling with **CFG ≈ 1**. Defaults in the UI match that.

Paths are `<models-base>/<model_id>` (default base: `./models`). Override with `--models-base`, `--model-path`, or `--download-path`.

List models: `python inference.py --list-models`

## Quick start

### 1) Login node

```bash
python login_node_server.py --pipeline-root ./jobs/
```

Copy the printed **session key**.

SSH tunnel from laptop:

```bash
ssh -L 8080:localhost:8080 <user>@<login-node>
```

Open `http://localhost:8080`, paste session key, choose a model, submit a job.

### 2) Download weights (internet node, no CUDA)

**Abliterated:**

```bash
python inference.py --download-only --model qwen_abliterated
# saves to ./models/qwen_abliterated
```

**Rapid AIO NSFW v23:**

```bash
python inference.py --download-only --model qwen_rapid_nsfw
# saves to ./models/qwen_rapid_nsfw
```

Custom base directory:

```bash
python inference.py --download-only --model qwen_abliterated \
  --models-base /path/to/weights
```

### 3) GPU worker

Start one worker per model (same session key, matching `--model` in UI):

```bash
python inference.py --model qwen_abliterated --session-key "<key>" --pipeline-root ./jobs/
```

```bash
python inference.py --model qwen_rapid_nsfw --session-key "<key>" --pipeline-root ./jobs/
```

## Adding a model

1. Subclass `ModelBackend` in `models/your_model.py`
2. Register in `models/registry.py`
3. UI picks it up via `/api/models` automatically

## Security

- Prompts and images encrypted before upload (`HPCENC02` envelope)
- Login node never decrypts
- GPU sees plaintext only in RAM during inference
- Session key: copy from login console to browser + GPU `--session-key`
