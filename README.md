## Medical Diffusion (Hackathon)

This repo contains:
- `backend/`: Python generation + agent loop (Veo 3.1), BiomedCLIP verifier, and an API server.
- `frontend/`: Vercel-ready Next.js UI that calls the backend and displays the generated video.

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

- Deploy `frontend/` to Vercel and set `NEXT_PUBLIC_BACKEND_URL` to your backend URL.
- Deploy `backend/` somewhere that can run Python + ffmpeg (Render/Railway/VM). Vercel serverless is not a good fit for the BiomedCLIP torch dependency.

