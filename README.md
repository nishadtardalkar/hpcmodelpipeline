# HPC Encrypted Model Pipeline

Encrypted image-edit inference: browser encrypts jobs; login node stores ciphertext; GPU worker decrypts, runs the selected model, encrypts results.

## Architecture

```
models/
  base.py            # ModelBackend interface
  registry.py        # register backends, list_models()
  qwen.py            # Qwen image-edit
  flux_uncensored.py # FLUX.1-dev + Flux-Uncensored-V2 LoRA
  florence2_flux.py  # Florence-2-Flux-Large (image → text)

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

| ID | Name | Output | Images | Default path |
|----|------|--------|--------|--------------|
| `qwen` | Qwen Image Edit | Image | Required (1–2) | `./models/qwen` |
| `flux_uncensored` | Flux Uncensored V2 | Image | Required (1) | `./models/flux_uncensored` |
| `florence2_flux` | Florence-2 Flux Large | Text | Required (1) | `./models/florence2_flux` |

Weights:

- [Qwen/Qwen-Image-Edit-2511](https://huggingface.co/Qwen/Qwen-Image-Edit-2511)
- [whrw/Flux-Uncensored-V2](https://huggingface.co/whrw/Flux-Uncensored-V2) LoRA on [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- [gokaygokay/Florence-2-Flux-Large](https://huggingface.co/gokaygokay/Florence-2-Flux-Large) (caption / describe image → text; remote code revision pinned)

Florence-2 uses `trust_remote_code=True`; the backend patches vendor files for transformers 4.50+ automatically after download.

Flux download saves `base/` (FLUX.1-dev) and `lora/` (`lora.safetensors`) under the model directory. Accept the FLUX.1-dev license on Hugging Face before downloading.

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

**Qwen:**

```bash
python inference.py --download-only --model qwen
# saves to ./models/qwen
```

**Flux Uncensored V2** (large base model + LoRA):

```bash
python inference.py --download-only --model flux_uncensored
# saves to ./models/flux_uncensored/base and .../lora
```

**Florence-2 Flux Large** (image captioning, ~0.8B):

```bash
python inference.py --download-only --model florence2_flux
# saves to ./models/florence2_flux
```

Custom base directory:

```bash
python inference.py --download-only --model flux_uncensored \
  --models-base /path/to/weights
# saves to /path/to/weights/flux_uncensored
```

### 3) GPU worker

Start one worker per model (same session key, matching `--model` in UI):

```bash
python inference.py --model qwen --session-key "<key>" --pipeline-root ./jobs/
```

```bash
python inference.py --model flux_uncensored --session-key "<key>" --pipeline-root ./jobs/
```

```bash
python inference.py --model florence2_flux --session-key "<key>" --pipeline-root ./jobs/
```

Florence-2 returns encrypted **text** (description/caption) instead of a PNG; the UI shows it in the status panel when the job completes.

## Adding a model

1. Subclass `ModelBackend` in `models/your_model.py`
2. Register in `models/registry.py`
3. UI picks it up via `/api/models` automatically

## Security

- Prompts and images encrypted before upload (`HPCENC02` envelope)
- Login node never decrypts
- GPU sees plaintext only in RAM during inference
- Session key: copy from login console to browser + GPU `--session-key`
