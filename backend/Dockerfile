FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: ffmpeg is required for extracting frames from Veo videos.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
  && rm -rf /var/lib/apt/lists/*

COPY requirements-web.txt /app/requirements-web.txt
RUN pip install --no-cache-dir -r /app/requirements-web.txt

COPY . /app

# Where videos + metadata are stored (mount a Railway Volume here for persistence).
ENV MEDICAL_DIFFUSION_RUNS_DIR=/app/runs \
    HF_HOME=/app/runs/hf_cache

CMD ["sh", "-c", "uvicorn medical_diffusion.server:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]

