from __future__ import annotations

import base64
import datetime as _dt
import io
import json
import os
import re
import threading
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .agent import AgentConfig, GeminiPromptAdjuster, ValidatorAgent
from .generation.demo import DemoBackend
from .generation.veo_genai import VeoGenaiBackend
from .io.export import export_final_video
from .prompt_processor import PromptProcessor
from .types import AgentResult, AnimationSpec, GenerationResult, ValidationScore
from .validation.biomedclip import BiomedCLIPMedicalValidator
from .validation.medical import FrameSanityMedicalValidator
from .validation.physics import PyBulletPhysicsValidator, RedDotGravityValidator


RUNS_DIR = Path(os.environ.get("MEDICAL_DIFFUSION_RUNS_DIR", "runs"))
JOBS_DIR = RUNS_DIR / "jobs"
LIBRARY_DIR = RUNS_DIR / "library"
IMAGES_DIR = RUNS_DIR / "images"

_biomedclip_lock = threading.Lock()
_biomedclip_image_validator: Optional[BiomedCLIPMedicalValidator] = None


@dataclass
class JobState:
    job_id: str
    status: Literal["queued", "running", "done", "error"] = "queued"
    created_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")
    updated_at: str = field(default_factory=lambda: _dt.datetime.utcnow().isoformat() + "Z")

    prompt: str = ""
    backend: str = "veo"
    prompt_rewrite: str = "gemini"
    gemini_model: str = "gemini-2.0-flash"
    biomedclip_target: Optional[str] = None
    use_biomedclip: bool = True
    input_image_provided: bool = False

    accepted: Optional[bool] = None
    report_summary: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    out_path: Optional[str] = None
    error: Optional[str] = None


_jobs_lock = threading.Lock()
_jobs: Dict[str, JobState] = {}
_job_fields = {f.name for f in fields(JobState)}


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="User prompt, e.g. 'heart surgery video'")
    backend: Literal["veo", "demo"] = "veo"
    input_image: Optional[str] = Field(
        default=None,
        description="Optional reference image for image+text→video (data URL or base64)",
    )

    prompt_rewrite: Literal["gemini", "rule", "none"] = "gemini"
    gemini_model: str = "gemini-2.0-flash"

    use_biomedclip: bool = True
    biomedclip_target: Optional[str] = None

    max_rounds: int = 2
    candidates: int = 1
    medical_threshold: float = 0.85
    physics_threshold: float = 0.85

    # Optional generation overrides
    fps: Optional[int] = None
    duration_s: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None

    # Veo backend knobs
    veo_model: str = "veo-3.1-generate-preview"
    veo_aspect_ratio: str = "9:16"
    veo_resolution: str = "720p"
    veo_poll_seconds: int = 20


class GenerateResponse(BaseModel):
    job_id: str
    status_url: str


class GenerateImageRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Prompt for a medical image, e.g. 'axial CT slice of liver'")
    model: str = Field(default="imagen-3.0-generate-001", description="Image model id (google-genai)")
    aspect_ratio: str = Field(default="1:1", description="Aspect ratio, e.g. 1:1 or 3:4")
    negative_prompt: Optional[str] = Field(default=None, description="Optional negative prompt")
    prompt_rewrite: Literal["gemini", "rule", "none"] = "gemini"
    gemini_model: str = "gemini-2.0-flash"
    use_biomedclip: bool = True
    biomedclip_target: Optional[str] = None
    biomedclip_threshold: float = 0.85
    max_rounds: int = 2


class GenerateImageResponse(BaseModel):
    image_data_url: str
    mime_type: str
    model: str
    accepted: Optional[bool] = None
    report_summary: Optional[str] = None
    report: Optional[Dict[str, Any]] = None
    final_prompt: Optional[str] = None
    rounds: int = 1


class ValidateImageRequest(BaseModel):
    input_image: str = Field(..., min_length=1, description="Image data URL or base64")
    prompt: Optional[str] = Field(default=None, description="Optional text context for target inference")
    biomedclip_target: Optional[str] = None
    biomedclip_threshold: float = 0.85


class ValidateImageResponse(BaseModel):
    accepted: Optional[bool] = None
    report_summary: str
    report: Dict[str, Any]


class JobResponse(BaseModel):
    job: Dict[str, Any]


class LibraryItem(BaseModel):
    job_id: str
    created_at: str
    prompt: str
    video_url: str
    accepted: Optional[bool] = None
    report_summary: Optional[str] = None


app = FastAPI(title="Medical Diffusion API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> Dict[str, str]:
    return {"ok": "true", "health": "/api/health", "docs": "/docs"}


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"ok": "true"}


@app.post("/api/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest) -> GenerateResponse:
    if req.backend == "veo" and not (req.input_image or "").strip():
        raise HTTPException(status_code=400, detail="input_image is required for backend=veo (image + text → video)")

    job_id = str(uuid.uuid4())
    job = JobState(
        job_id=job_id,
        prompt=req.prompt,
        backend=req.backend,
        prompt_rewrite=req.prompt_rewrite,
        gemini_model=req.gemini_model,
        biomedclip_target=req.biomedclip_target,
        use_biomedclip=bool(req.use_biomedclip),
        input_image_provided=bool(req.input_image),
    )
    with _jobs_lock:
        _jobs[job_id] = job
    _persist_job(job)

    thread = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    thread.start()

    return GenerateResponse(job_id=job_id, status_url=f"/api/jobs/{job_id}")


@app.post("/api/images/generate", response_model=GenerateImageResponse)
def generate_image(req: GenerateImageRequest) -> GenerateImageResponse:
    try:
        from google import genai
        from google.genai import types
    except Exception as e:
        raise HTTPException(status_code=500, detail="google-genai is required for image generation") from e

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="GOOGLE_API_KEY is not set (required for image generation)")

    if int(req.max_rounds or 2) < 1:
        raise HTTPException(status_code=400, detail="max_rounds must be >= 1")

    client = genai.Client(api_key=api_key)
    max_rounds = min(2, int(req.max_rounds or 2))
    biomedclip_threshold = float(req.biomedclip_threshold or 0.85)

    original_user_prompt = req.prompt.strip()
    prompt = original_user_prompt
    negative_prompt = (req.negative_prompt or "").strip() or _default_image_negative_prompt()

    run_id = str(uuid.uuid4())
    out_dir = IMAGES_DIR / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted: Optional[bool] = None
    report: Optional[Dict[str, Any]] = None
    report_summary: Optional[str] = None
    final_prompt: Optional[str] = None
    rounds = 0

    image_bytes: Optional[bytes] = None
    mime_type: str = "image/png"

    for round_index in range(max_rounds):
        rounds = round_index + 1
        image_bytes, mime_type = _genai_generate_image_bytes(
            client=client,
            types=types,
            model=req.model,
            prompt=prompt,
            negative_prompt=negative_prompt,
            aspect_ratio=req.aspect_ratio,
        )
        image_path = out_dir / f"round_{round_index:02d}.png"
        image_path.write_bytes(image_bytes)

        if not req.use_biomedclip:
            accepted = None
            final_prompt = prompt
            break

        score = _score_image_with_biomedclip(
            image_path=image_path,
            prompt=prompt,
            target=req.biomedclip_target,
        )
        accepted = bool(score.score >= biomedclip_threshold) if not score.skipped else True
        report = asdict(score)
        report_summary = f"biomedclip={score.score:.3f}" + (" (skipped)" if score.skipped else "")
        final_prompt = prompt

        if accepted:
            break

        if round_index >= max_rounds - 1:
            break

        prompt, negative_prompt = _reprompt_image_prompt(
            original_user_prompt=original_user_prompt,
            previous_prompt=prompt,
            previous_negative_prompt=negative_prompt,
            biomedclip_score=score,
            mode=req.prompt_rewrite,
            gemini_model=req.gemini_model,
        )

    if not image_bytes:
        raise HTTPException(status_code=500, detail="Image generation failed")

    b64 = base64.b64encode(image_bytes).decode("ascii")
    return GenerateImageResponse(
        image_data_url=f"data:{mime_type};base64,{b64}",
        mime_type=mime_type,
        model=req.model,
        accepted=accepted,
        report_summary=report_summary,
        report=report,
        final_prompt=final_prompt,
        rounds=rounds,
    )


@app.post("/api/images/validate", response_model=ValidateImageResponse)
def validate_image(req: ValidateImageRequest) -> ValidateImageResponse:
    out_dir = IMAGES_DIR / f"validate_{uuid.uuid4()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = _save_input_image(data=req.input_image, output_path=out_dir / "input_image.png")

    prompt = (req.prompt or "").strip() or (req.biomedclip_target or "")
    score = _score_image_with_biomedclip(image_path=image_path, prompt=prompt, target=req.biomedclip_target)

    threshold = float(req.biomedclip_threshold or 0.85)
    accepted = None if score.skipped else bool(score.score >= threshold)
    report_summary = f"biomedclip={score.score:.3f}" + (" (skipped)" if score.skipped else "")
    return ValidateImageResponse(
        accepted=accepted,
        report_summary=report_summary,
        report=asdict(score),
    )


@app.get("/api/jobs/{job_id}", response_model=JobResponse)
def job_status(job_id: str) -> JobResponse:
    _validate_job_id(job_id)
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        job = _load_job(job_id)
        if job:
            with _jobs_lock:
                _jobs[job_id] = job
        else:
            raise HTTPException(status_code=404, detail="Job not found")
    return JobResponse(job=asdict(job))


@app.get("/api/videos/{job_id}.mp4")
def get_video(job_id: str) -> FileResponse:
    _validate_job_id(job_id)
    path = LIBRARY_DIR / f"{job_id}.mp4"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)


@app.get("/api/library", response_model=List[LibraryItem])
def library(request: Request) -> List[LibraryItem]:
    base = str(request.base_url).rstrip("/")
    items = []
    if LIBRARY_DIR.exists():
        for meta_path in sorted(LIBRARY_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                meta = json.loads(meta_path.read_text())
            except Exception:
                continue
            job_id = str(meta.get("job_id", "")).strip()
            if not job_id:
                continue
            items.append(
                LibraryItem(
                    job_id=job_id,
                    created_at=str(meta.get("created_at", "")),
                    prompt=str(meta.get("prompt", "")),
                    video_url=f"{base}/api/videos/{job_id}.mp4",
                    accepted=meta.get("accepted"),
                    report_summary=meta.get("report_summary"),
                )
            )
    return items


def _run_job(job_id: str, req: GenerateRequest) -> None:
    _update_job(job_id, status="running")
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
        run_root = JOBS_DIR / job_id
        run_root.mkdir(parents=True, exist_ok=True)

        processor = PromptProcessor()
        spec = processor.parse(req.prompt)
        spec = _apply_overrides(spec, req)
        if req.input_image:
            image_path = _save_input_image(data=req.input_image, output_path=run_root / "input_image.png")
            from dataclasses import replace

            spec = replace(spec, input_image_path=image_path)

        if req.backend == "veo":
            if req.prompt_rewrite == "none":
                pass
            elif req.prompt_rewrite == "rule":
                spec = processor.rewrite_for_veo(spec)
            elif req.prompt_rewrite == "gemini":
                spec = processor.rewrite_for_veo_gemini(spec, model=req.gemini_model)
            else:
                raise ValueError(f"Unsupported prompt_rewrite: {req.prompt_rewrite}")

        backend = _make_backend(req)

        medical_validators = [FrameSanityMedicalValidator()]

        physics_validators = [RedDotGravityValidator(), PyBulletPhysicsValidator()]

        prompt_adjuster = GeminiPromptAdjuster(model=req.gemini_model) if req.prompt_rewrite == "gemini" else None
        agent = ValidatorAgent(
            generator=backend,
            medical_validators=medical_validators,
            physics_validators=physics_validators,
            config=AgentConfig(
                max_rounds=min(2, int(req.max_rounds)),
                candidates_per_round=int(req.candidates),
                medical_threshold=float(req.medical_threshold),
                physics_threshold=float(req.physics_threshold),
            ),
            prompt_adjuster=prompt_adjuster,
            run_root=run_root,
        )

        result: AgentResult = agent.run(spec=spec)

        out_path = LIBRARY_DIR / f"{job_id}.mp4"
        export_final_video(generation=result.final.generation, output_path=out_path)

        meta = {
            "job_id": job_id,
            "created_at": _dt.datetime.utcnow().isoformat() + "Z",
            "prompt": req.prompt,
            "backend": req.backend,
            "accepted": bool(result.accepted),
            "report_summary": result.final.report.summary(),
            "final_prompt": result.final.prompt,
            "generation": {
                "backend": result.final.generation.backend,
                "frames_dir": str(result.final.generation.frames_dir),
                "metadata": result.final.generation.metadata,
            },
            "report": {
                "medical": {
                    "score": result.final.report.medical.score,
                    "name": result.final.report.medical.name,
                    "feedback": result.final.report.medical.feedback,
                    "details": result.final.report.medical.details,
                    "skipped": result.final.report.medical.skipped,
                },
                "physics": {
                    "score": result.final.report.physics.score,
                    "name": result.final.report.physics.name,
                    "feedback": result.final.report.physics.feedback,
                    "details": result.final.report.physics.details,
                    "skipped": result.final.report.physics.skipped,
                },
            },
        }
        (LIBRARY_DIR / f"{job_id}.json").write_text(json.dumps(meta, indent=2))

        _update_job(
            job_id,
            status="done",
            accepted=bool(result.accepted),
            report_summary=result.final.report.summary(),
            report=meta["report"],
            out_path=str(out_path),
        )
    except Exception as e:
        _update_job(job_id, status="error", error=str(e))


def _apply_overrides(spec: AnimationSpec, req: GenerateRequest) -> AnimationSpec:
    from dataclasses import replace

    if req.fps is not None:
        spec = replace(spec, fps=int(req.fps))
    if req.duration_s is not None:
        spec = replace(spec, duration_s=float(req.duration_s))
    if (req.width is None) != (req.height is None):
        raise ValueError("--width and --height must be provided together")
    if req.width is not None and req.height is not None:
        spec = replace(spec, width=int(req.width), height=int(req.height))

    metadata = dict(spec.metadata or {})
    metadata.setdefault("user_prompt", spec.prompt)
    if req.fps is not None:
        metadata["user_set_fps"] = True
    if req.duration_s is not None:
        metadata["user_set_duration"] = True
    if req.width is not None and req.height is not None:
        metadata["user_set_size"] = True
    return replace(spec, metadata=metadata)


def _make_backend(req: GenerateRequest):
    if req.backend == "veo":
        return VeoGenaiBackend(
            model=req.veo_model,
            aspect_ratio=req.veo_aspect_ratio,
            resolution=req.veo_resolution,
            poll_seconds=int(req.veo_poll_seconds),
        )
    if req.backend == "demo":
        return DemoBackend()
    raise ValueError(f"Unsupported backend: {req.backend}")


def _default_image_negative_prompt() -> str:
    return "text, watermark, labels, low quality, blurry, cartoon, anime, unrealistic anatomy"


def _genai_generate_image_bytes(
    *,
    client: Any,
    types: Any,
    model: str,
    prompt: str,
    negative_prompt: Optional[str],
    aspect_ratio: str,
) -> tuple[bytes, str]:
    resp = client.models.generate_images(
        model=model,
        prompt=prompt,
        config=types.GenerateImagesConfig(
            negative_prompt=(negative_prompt or "").strip() or None,
            number_of_images=1,
            aspect_ratio=aspect_ratio,
            add_watermark=False,
            output_mime_type="image/png",
        ),
    )

    generated = (getattr(resp, "generated_images", None) or [])[:1]
    if not generated:
        raise HTTPException(status_code=500, detail="Image model returned no images")

    image = getattr(generated[0], "image", None)
    image_bytes = getattr(image, "image_bytes", None)
    mime_type = (getattr(image, "mime_type", None) or "image/png").strip()
    if not image_bytes:
        raise HTTPException(status_code=500, detail="Image model returned empty image bytes")

    return bytes(image_bytes), mime_type


def _get_biomedclip_image_validator() -> BiomedCLIPMedicalValidator:
    global _biomedclip_image_validator
    if _biomedclip_image_validator is None:
        _biomedclip_image_validator = BiomedCLIPMedicalValidator(n_frames=1)
    return _biomedclip_image_validator


def _score_image_with_biomedclip(*, image_path: Path, prompt: str, target: Optional[str]) -> ValidationScore:
    spec = AnimationSpec(prompt=prompt, metadata={"biomedclip_target": target} if target else {})
    generation = GenerationResult(
        spec=spec,
        frames=[image_path],
        frames_dir=image_path.parent,
        backend="image",
        metadata={"image_path": str(image_path)},
    )
    with _biomedclip_lock:
        validator = _get_biomedclip_image_validator()
        return validator.score(generation)


def _reprompt_image_prompt(
    *,
    original_user_prompt: str,
    previous_prompt: str,
    previous_negative_prompt: str,
    biomedclip_score: ValidationScore,
    mode: str,
    gemini_model: str,
) -> tuple[str, str]:
    mode = (mode or "gemini").strip().lower()
    if mode == "none":
        return previous_prompt, previous_negative_prompt

    suggested: List[str] = []
    top_guess = None
    target_label = None
    if isinstance(getattr(biomedclip_score, "details", None), dict):
        target_label = biomedclip_score.details.get("target_label")
        top_guess = biomedclip_score.details.get("top_guess")
        maybe = biomedclip_score.details.get("suggested_keywords")
        if isinstance(maybe, list):
            suggested.extend([str(x).strip() for x in maybe if str(x).strip()])
    suggested = suggested[:12]

    if mode == "rule":
        next_prompt = previous_prompt.rstrip()
        if suggested:
            next_prompt = next_prompt + " " + " ".join(suggested)
        else:
            next_prompt = next_prompt + " medically accurate, anatomically correct, clinical, no text, no watermark"
        return next_prompt, previous_negative_prompt

    if mode != "gemini":
        return previous_prompt, previous_negative_prompt

    system = """You are a prompt engineer for biomedical image generation (e.g. Imagen).

You will be given:
- the original user request for a biomedical image
- the previous image prompt
- BiomedCLIP score + feedback + top labels (a guardrail signal, not an oracle)

Goal: produce a medically plausible, anatomically correct, realistic biomedical image that matches the intended anatomy/view/modality.

Output STRICT JSON (no markdown):
{
  "image_prompt": "string",
  "negative_prompt": "string"
}
Keep the prompt one concise paragraph. Include: anatomy, modality (if relevant), viewpoint, style, constraints (no text/labels/watermark).
"""

    user = f"""original_user_request: {original_user_prompt}
previous_image_prompt: {previous_prompt}
biomedclip_score: {biomedclip_score.score:.3f}
biomedclip_feedback: {biomedclip_score.feedback}
biomedclip_target_label: {target_label}
biomedclip_top_guess: {top_guess}
suggested_keywords: {", ".join(suggested)}
"""

    try:
        from .gemini import generate_json, get_optional_str, load_gemini_config

        cfg = load_gemini_config(model=gemini_model)
        data = generate_json(system=system, user=user, config=cfg)
        next_prompt = get_optional_str(data, "image_prompt")
        next_negative = get_optional_str(data, "negative_prompt")
        if not next_prompt:
            raise RuntimeError("Gemini returned no image_prompt")
        return next_prompt, next_negative or previous_negative_prompt
    except Exception:
        # Fallback to rule-based keyword injection.
        next_prompt = previous_prompt.rstrip()
        if suggested:
            next_prompt = next_prompt + " " + " ".join(suggested)
        else:
            next_prompt = next_prompt + " medically accurate, anatomically correct, clinical, no text, no watermark"
        return next_prompt, previous_negative_prompt


def _save_input_image(*, data: str, output_path: Path) -> Path:
    """
    Accepts either:
    - data URL: data:image/png;base64,....
    - raw base64: iVBORw0KGgo....

    Writes a normalized PNG to `output_path` (suffix forced to .png).
    """
    raw = (data or "").strip()
    if not raw:
        raise ValueError("input_image is empty")

    mime_type = None
    b64 = raw
    if raw.lower().startswith("data:"):
        header, comma, payload = raw.partition(",")
        if not comma:
            raise ValueError("input_image data URL is missing a comma separator")
        header = header[5:]  # strip leading "data:"
        mime_part, _, params_part = header.partition(";")
        mime_type = mime_part.strip() or None
        params = [p.strip() for p in params_part.split(";") if p.strip()]
        if not any(p.lower() == "base64" for p in params):
            raise ValueError("input_image data URL must be base64-encoded")
        b64 = payload

    # remove all whitespace correctly
    b64_clean = re.sub(r"\s+", "", b64)
    # fix missing base64 padding
    pad = (-len(b64_clean)) % 4
    if pad:
        b64_clean = b64_clean + ("=" * pad)

    try:
        decoded = base64.b64decode(b64_clean, validate=False)
    except Exception as e:
        hint = f" ({mime_type})" if mime_type else ""
        raise ValueError(f"input_image is not valid base64{hint}") from e

    # magic bytes for debugging
    magic = decoded[:16]
    magic_hex = " ".join(f"{b:02x}" for b in magic)

    try:
        # HEIC/HEIF support (Apple tax)
        try:
            import pillow_heif  # type: ignore
            pillow_heif.register_heif_opener()
        except Exception:
            # If it's HEIC and this isn't installed, Pillow will fail below.
            pass

        from PIL import Image

        img = Image.open(io.BytesIO(decoded))
        img.load()  # force decode now, so we fail here if it's unsupported

        # normalize
        img = img.convert("RGB")

        output_path = output_path.with_suffix(".png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, format="PNG")
        return output_path

    except Exception as e:
        hint = f" ({mime_type})" if mime_type else ""
        raise ValueError(
            f"input_image could not be decoded as an image{hint}. "
            f"First bytes: {magic_hex}"
        ) from e
def _validate_job_id(job_id: str) -> None:
    try:
        uuid.UUID(job_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail="Invalid job id") from e


def _update_job(job_id: str, **kwargs: Any) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            job = _load_job(job_id) or JobState(job_id=job_id)
            _jobs[job_id] = job
        for k, v in kwargs.items():
            setattr(job, k, v)
        job.updated_at = _dt.datetime.utcnow().isoformat() + "Z"
    _persist_job(job)


def _job_state_path(job_id: str) -> Path:
    return JOBS_DIR / f"{job_id}.json"


def _persist_job(job: JobState) -> None:
    try:
        JOBS_DIR.mkdir(parents=True, exist_ok=True)
        path = _job_state_path(job.job_id)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(asdict(job), ensure_ascii=False))
        tmp.replace(path)
    except Exception:
        # Best-effort persistence; do not fail requests/jobs due to IO issues.
        return


def _load_job(job_id: str) -> Optional[JobState]:
    path = _job_state_path(job_id)
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    if not data.get("job_id"):
        return None
    kwargs = {k: data.get(k) for k in _job_fields if k in data}
    try:
        return JobState(**kwargs)
    except Exception:
        return None
