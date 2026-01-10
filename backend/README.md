## Medical Diffusion (Hackathon Skeleton)

This is a runnable backend skeleton for:

`diffusion/video generator → validator agent loop → (medical + physics) verifiers → reroll`

### Quick start (no GPUs, no APIs)

Requirements:
- Python 3.9+
- Optional: `ffmpeg` on PATH (to encode `.mp4`)

API keys (only needed for real APIs):
```bash
export GOOGLE_API_KEY=...      # Veo (google-genai)
export TWELVELABS_API_KEY=...  # captions
export DEEPGRAM_API_KEY=...    # TTS (template)
```

Run the mock backend:
```bash
cd backend
python3 -m medical_diffusion run \
  --backend mock \
  --prompt "red blood cells moving through a capillary" \
  --out runs/demo.mp4
```

If you don’t have `ffmpeg`, add `--no-video` and you’ll still get a frames folder.

### Veo 3.1 backend (optional)

This backend expects the `google-genai` SDK and `ffmpeg` to extract frames.

```bash
pip install google-genai
export GOOGLE_API_KEY=...
cd backend
python3 -m medical_diffusion run --backend veo --prompt "liver ct axial slice" --no-video
```

By default `--backend veo` uses Gemini-based prompt rewriting (same `GOOGLE_API_KEY`). You can disable it with `--prompt-rewrite rule` or `--prompt-rewrite none`.

### BiomedCLIP medical verifier (optional)

BiomedCLIP can score sampled frames against text labels (guardrail-style). Install:
```bash
pip install open_clip_torch==2.23.0 transformers==4.35.2 torch pillow
```

Run with the validator enabled:
```bash
cd backend
python3 -m medical_diffusion run \
  --backend veo \
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
- `POST /api/generate` → returns `{job_id, status_url}`
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
4) (Recommended) Add a Railway Volume mounted at `/app/runs` so `runs/library` persists across deploys.
5) Deploy, then check `https://<your-railway-domain>/api/health`.

### MedMNIST OrganMNIST3D “visual memory” (reference GIFs + prototypes)

This uses MedMNIST’s `OrganMNIST3D` volumes to build per-organ prototypes and (optionally) export a few reference slice GIFs.

```bash
cd backend
python3 -m medical_diffusion run \
  --backend mock \
  --prompt "liver ct axial slice" \
  --no-video \
  --medmnist-organ3d \
  --medmnist-download \
  --medmnist-export-gifs runs/organmnist3d_gifs
```

By default it caches downloads + prototypes under `backend/.cache/medmnist`.

### Where to plug in real stuff

- Generator backends:
  - `backend/medical_diffusion/generation/mock.py` (works now)
  - `backend/medical_diffusion/generation/veo_stub.py` (API stub)
  - Add a `diffusers` backend when you’re ready
- Validators:
  - `backend/medical_diffusion/validation/medical.py` (sanity + library keyword stub)
  - `backend/medical_diffusion/validation/physics.py` (toy gravity tracker + PyBullet stub)
  - `backend/medical_diffusion/validation/organmnist3d.py` (MedMNIST OrganMNIST3D visual reference)
- Agent loop:
  - `backend/medical_diffusion/agent.py`
- Post-processing (templates):
  - `backend/medical_diffusion/postprocess/twelvelabs.py` (captions)
  - `backend/medical_diffusion/postprocess/deepgram_tts.py` (narration)
  - Examples: `backend/examples/twelvelabs_captioning.py`, `backend/examples/deepgram_tts.py`

### Reference library format (starter)

Pass `--reference-dir backend/examples/reference_library` and edit:
- `backend/examples/reference_library/manifest.json`

### Suggested deps (optional)

Put these in your environment when you’re ready to integrate real models/tools:
- `diffusers`, `transformers`, `torch`, `accelerate`
- `pybullet`
- `opencv-python` (for tracking/optical flow)
- `pillow` (image IO)
