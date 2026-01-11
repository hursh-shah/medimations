from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Optional

from .captions import CaptionResult, parse_caption_result


DEFAULT_TWELVELABS_CAPTION_PROMPT = """You are analyzing a short biomedical animation video generated for a hackathon.

Task:
1) Describe what is happening in medically accurate terms (structures, processes, motion).
2) Produce time-coded captions suitable for narration.

Output format (STRICT JSON, no markdown):
{
  "summary": "1-2 sentence summary of the animation",
  "captions": [
    {"start_s": 0.0, "end_s": 1.2, "text": "Caption text..."},
    ...
  ],
  "narration": "A single narration script that matches the captions",
  "medical_uncertainties": ["Any parts that look medically questionable or ambiguous", ...]
}

Guidelines:
- Use short, clear sentences for captions.
- Prefer anatomical terms a medical student would understand.
- If the video is stylized or unclear, say so in medical_uncertainties rather than guessing.
"""


@dataclass(frozen=True)
class TwelveLabsConfig:
    api_key: str
    index_id: Optional[str] = None
    language: str = "en"
    model_name: str = "pegasus1.2"
    enable_visual: bool = True
    enable_audio: bool = True


def _get_attr(obj: Any, *names: str) -> Any:
    for name in names:
        try:
            value = getattr(obj, name)
        except Exception:
            value = None
        if value is not None:
            return value
    return None


def _get_id(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    if isinstance(obj, dict):
        for k in ("id", "index_id", "task_id", "video_id"):
            v = obj.get(k)
            if v:
                return str(v)
        return None
    for k in ("id", "index_id", "task_id", "video_id"):
        v = getattr(obj, k, None)
        if v:
            return str(v)
    return None


def generate_captions_with_twelvelabs(
    *,
    video_path: Path,
    config: TwelveLabsConfig,
    prompt: str = DEFAULT_TWELVELABS_CAPTION_PROMPT,
) -> CaptionResult:
    """
    Template integration with TwelveLabs:
    - creates an index if index_id is not provided
    - uploads + waits for indexing
    - runs analyze() with a captioning prompt

    Requires network access and:
      pip install twelvelabs
    """
    try:
        from twelvelabs import TwelveLabs
    except Exception as e:
        raise RuntimeError("Missing dependency: pip install twelvelabs") from e

    # TwelveLabs SDK uses keyword-only args (api_key=...).
    client = TwelveLabs(api_key=config.api_key)

    index_id = config.index_id
    if not index_id:
        index_client = _get_attr(client, "index", "indexes")
        create_index = _get_attr(index_client, "create") if index_client is not None else None
        if create_index is None:
            raise RuntimeError("TWELVELABS_INDEX_ID must be set (this TwelveLabs SDK has no index create API)")

        engines = [
            {
                "name": config.model_name,
                "options": [opt for opt, on in [("visual", config.enable_visual), ("audio", config.enable_audio)] if on],
            }
        ]
        try:
            index = create_index(name="Medical Diffusion Hackathon Index", models=engines)
        except TypeError:
            try:
                index = create_index(name="Medical Diffusion Hackathon Index", engines=engines)
            except TypeError:
                index = create_index(name="Medical Diffusion Hackathon Index")
        index_id = _get_id(index)

    if not index_id:
        raise RuntimeError("TwelveLabs index_id is missing")

    task_client = _get_attr(client, "task", "tasks")
    create_task = _get_attr(task_client, "create") if task_client is not None else None
    if create_task is None:
        raise RuntimeError("This TwelveLabs SDK has no task create API")

    task = None
    create_attempts = [
        {"index_id": index_id, "file": str(video_path), "language": config.language},
        {"index_id": index_id, "file_path": str(video_path), "language": config.language},
        {"index_id": index_id, "video_file": str(video_path), "language": config.language},
        {"indexId": index_id, "file": str(video_path), "language": config.language},
        {"indexId": index_id, "file_path": str(video_path), "language": config.language},
        {"indexId": index_id, "video_file": str(video_path), "language": config.language},
    ]
    for kwargs in create_attempts:
        try:
            task = create_task(**kwargs)
            break
        except TypeError:
            continue
    if task is None:
        with open(video_path, "rb") as f:
            file_attempts = [
                {"index_id": index_id, "file": f, "language": config.language},
                {"index_id": index_id, "video_file": f, "language": config.language},
                {"indexId": index_id, "file": f, "language": config.language},
                {"indexId": index_id, "video_file": f, "language": config.language},
            ]
            for kwargs in file_attempts:
                try:
                    task = create_task(**kwargs)
                    break
                except TypeError:
                    continue
    if task is None:
        raise RuntimeError("Unsupported TwelveLabs task.create signature for file upload")

    # Wait for indexing to finish.
    wait_for_done = _get_attr(task, "wait_for_done")
    if callable(wait_for_done):
        wait_for_done()
    else:
        wait_task = _get_attr(task_client, "wait_for_done") if task_client is not None else None
        task_id = _get_id(task)
        if callable(wait_task) and task_id:
            wait_task(task_id)
        else:
            retrieve_task = _get_attr(task_client, "retrieve", "get") if task_client is not None else None
            if not callable(retrieve_task) or not task_id:
                raise RuntimeError("TwelveLabs task created but no wait mechanism is available in this SDK")
            for _ in range(180):  # up to ~15 minutes at 5s intervals
                current = retrieve_task(task_id)
                status = _get_attr(current, "status", "state")
                status = str(status).strip().lower() if status else ""
                if status in {"done", "completed", "ready", "succeeded"}:
                    task = current
                    break
                if status in {"failed", "error", "canceled", "cancelled"}:
                    raise RuntimeError(f"TwelveLabs indexing failed (status={status})")
                time.sleep(5)

    video_id = _get_attr(task, "video_id") or _get_attr(task, "videoId") or _get_attr(task, "video")
    if isinstance(video_id, dict):
        video_id = video_id.get("id") or video_id.get("video_id")
    video_id = str(video_id).strip() if video_id else ""

    if not video_id:
        raise RuntimeError("TwelveLabs task completed but no video_id was returned")

    # SDK uses keyword-only params for analyze(): video_id=..., prompt=...
    res = client.analyze(video_id=video_id, prompt=prompt)
    raw = getattr(res, "data", None) or ""

    parsed = parse_caption_result(str(raw))
    if parsed:
        return parsed

    # Fallback: treat the whole response as a narration blob.
    return CaptionResult(summary="", segments=[], narration=str(raw).strip(), medical_uncertainties=[])
