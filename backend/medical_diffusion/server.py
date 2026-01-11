from __future__ import annotations

import datetime as _dt
import json
import os
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
from .types import AnimationSpec, AgentResult
from .validation.biomedclip import BiomedCLIPMedicalValidator
from .validation.medical import FrameSanityMedicalValidator
from .validation.physics import PyBulletPhysicsValidator, RedDotGravityValidator


RUNS_DIR = Path(os.environ.get("MEDICAL_DIFFUSION_RUNS_DIR", "runs"))
JOBS_DIR = RUNS_DIR / "jobs"
LIBRARY_DIR = RUNS_DIR / "library"


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
    job_id = str(uuid.uuid4())
    job = JobState(
        job_id=job_id,
        prompt=req.prompt,
        backend=req.backend,
        prompt_rewrite=req.prompt_rewrite,
        gemini_model=req.gemini_model,
        biomedclip_target=req.biomedclip_target,
        use_biomedclip=bool(req.use_biomedclip),
    )
    with _jobs_lock:
        _jobs[job_id] = job
    _persist_job(job)

    thread = threading.Thread(target=_run_job, args=(job_id, req), daemon=True)
    thread.start()

    return GenerateResponse(job_id=job_id, status_url=f"/api/jobs/{job_id}")


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

        processor = PromptProcessor()
        spec = processor.parse(req.prompt)
        spec = _apply_overrides(spec, req)

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
        biomed_target = req.biomedclip_target or str(spec.metadata.get("target_label") or "").strip() or None
        if req.use_biomedclip:
            medical_validators.append(BiomedCLIPMedicalValidator(target_label=biomed_target))

        physics_validators = [RedDotGravityValidator(), PyBulletPhysicsValidator()]

        prompt_adjuster = (
            GeminiPromptAdjuster(model=req.gemini_model) if (req.backend == "veo" and req.prompt_rewrite == "gemini") else None
        )
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
            run_root=JOBS_DIR / job_id,
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
