## Medimations

This repo contains:
- `backend/`: Python generation + agent loop (Veo 3.1), BiomedCLIP image verifier, and an API server.
- `frontend/`: Vercel-ready Next.js UI for **image + text → video** (upload/generate an image, then animate it).

Optional post-processing (backend): TwelveLabs captions + Deepgram TTS narration for generated videos.

### Local dev

1) Backend (API server)
```bash
brew install ffmpeg
cd backend
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
pip install -r requirements-web.txt
export GOOGLE_API_KEY=...
uvicorn medical_diffusion.server:app --reload --port 8000
```

2) Frontend (Next.js)
```bash
cd frontend
cp .env.example .env.local
npm install
npm run dev
```
Open `http://localhost:3000`.

### Deployment notes

- Deploy `backend/` somewhere that can run Python + `ffmpeg` (Railway/VM). Vercel serverless is not a good fit for the BiomedCLIP `torch` dependency.

**Railway (backend)**
- Service root directory: `backend/` (so it picks up `backend/Dockerfile`)
- Also keep in mind you gotta build from Dockerfile
- Env vars (Railway): `GOOGLE_API_KEY` (required)
- Domain: use a **public** Railway domain (not `.railway.internal`) and route to the service port (`PORT`, usually `8080`)
- Replicas: keep **1 replica** (jobs are stored locally; multi-replica needs shared storage)
- BiomedCLIP: requires more RAM; if you see the process get `Killed`, increase Railway memory.
- Health check: `https://<your-railway-domain>/api/health`

**Vercel (frontend)**
- Project root directory: `frontend/`
- Framework preset: Next.js (leave “Output Directory” empty; don’t set it to `public`)
- Env vars (Vercel): `NEXT_PUBLIC_BACKEND_URL=https://<your-railway-domain>`
