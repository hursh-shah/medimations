## Medical Diffusion (Hackathon Skeleton)

This is a runnable backend skeleton for:

`Veo video generator → validator agent loop → (medical + physics) verifiers → reroll`

### Quick start

Requirements:
- Python 3.10+ (3.11 recommended)
- `ffmpeg` on PATH (required for extracting frames + encoding `.mp4`)

API keys (only needed for real APIs):
```bash
export GOOGLE_API_KEY=...      # Veo (google-genai)
export TWELVELABS_API_KEY=...  # captions
export DEEPGRAM_API_KEY=...    # TTS (template)
```

Install deps:
```bash
cd backend
pip install -r requirements-web.txt
```

Optional (captioning + TTS templates):
```bash
pip install -r requirements-postprocess.txt
```

### Veo 3.1 backend

This backend expects the `google-genai` SDK and `ffmpeg` to extract frames.

```bash
export GOOGLE_API_KEY=...
cd backend
python3 -m medical_diffusion run --prompt "liver ct axial slice" --no-video
```

By default `run` uses Gemini-based prompt rewriting (same `GOOGLE_API_KEY`). You can disable it with `--prompt-rewrite rule` or `--prompt-rewrite none`.

### Demo backend (no Veo, no API keys)

Use this to test the full workflow (jobs → polling → video export) without consuming Veo/Gemini quota:
```bash
cd backend
uvicorn medical_diffusion.server:app --reload --port 8000
```
Then call `POST /api/generate` with `"backend": "demo"` (or use the frontend “Test (no Veo)” button).

### BiomedCLIP medical verifier (optional)

BiomedCLIP can score sampled frames against text labels (guardrail-style). Install:
```bash
pip install open_clip_torch==2.23.0 transformers==4.35.2 torch torchvision pillow
```

Notes:
- If you see errors about “NumPy 2.x” / `_ARRAY_API not found`, pin `numpy<2` (the Railway Docker build already does this).
- BiomedCLIP is large; on Railway you likely need a higher-memory instance. If the process gets `Killed`, increase memory.

Run with the validator enabled:
```bash
cd backend
python3 -m medical_diffusion run \
  --prompt "red blood cells moving through a capillary" \
  --biomedclip \
  --biomedclip-target "capillary" \
  --no-video
```

### Run as an API server (for the web frontend)

Install deps:
```bash
cd backend
pip install -r requirements-web.txt
```

Run:
```bash
export GOOGLE_API_KEY=...
uvicorn medical_diffusion.server:app --reload --port 8000
```

API endpoints:
- `POST /api/images/generate` → returns `{image_data_url, ...}` (optional helper for AI image generation)
- `POST /api/generate` → returns `{job_id, status_url}` (supports optional `input_image` for image+text → video)
- `GET /api/jobs/{job_id}` → status + results
- `GET /api/videos/{job_id}.mp4` → final video
- `GET /api/library` → saved videos

### Deploy backend to Railway

Railway is easiest with the included Dockerfile:

1) Create a new Railway project → New Service → Deploy from GitHub repo.
2) Set the service **Root Directory** to `backend/` (so it finds `backend/Dockerfile`).
3) Add env vars:
   - `GOOGLE_API_KEY` (required)
   - optional: `TWELVELABS_API_KEY`, `DEEPGRAM_API_KEY`
   - Note: `requirements-web.txt` is pinned to CPU-only PyTorch wheels on Linux to avoid multi-GB CUDA installs/timeouts.
4) (Recommended) Add a Railway Volume mounted at `/app/runs` so `runs/library` persists across deploys.
5) Keep the service at **1 replica** unless you add shared storage for job state.
6) Deploy, then check `https://<your-railway-domain>/api/health`.

### Where to plug in real stuff

- Generator backends:
  - `backend/medical_diffusion/generation/veo_genai.py` (Veo via google-genai)
  - Add a `diffusers` backend when you’re ready
- Validators:
  - `backend/medical_diffusion/validation/biomedclip.py` (BiomedCLIP frame verifier)
  - `backend/medical_diffusion/validation/medical.py` (sanity/flicker check)
  - `backend/medical_diffusion/validation/physics.py` (toy gravity tracker + PyBullet stub)
- Agent loop:
  - `backend/medical_diffusion/agent.py`
- Post-processing (templates):
  - `backend/medical_diffusion/postprocess/twelvelabs.py` (captions)
  - `backend/medical_diffusion/postprocess/deepgram_tts.py` (narration)
  - Examples: `backend/examples/twelvelabs_captioning.py`, `backend/examples/deepgram_tts.py`

### Suggested deps (optional)

Put these in your environment when you’re ready to integrate real models/tools:
- `diffusers`, `transformers`, `torch`, `accelerate`
- `pybullet`
- `opencv-python` (for tracking/optical flow)
- `pillow` (image IO)
