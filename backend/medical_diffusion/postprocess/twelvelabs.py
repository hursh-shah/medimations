from __future__ import annotations

import inspect
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from ..io.video import get_video_duration_seconds
from .captions import CaptionResult, CaptionSegment, normalize_segments, parse_caption_result


DEFAULT_TWELVELABS_CAPTION_PROMPT = """You are TwelveLabs Pegasus analyzing a short biomedical animation video.

Task:
1) Write a medically accurate voiceover script that matches what is visible in the video.
2) Produce time-coded captions that match the voiceover exactly.

Output format (STRICT JSON, no markdown):
{
  "summary": "1-2 sentence summary of the animation",
  "captions": [
    {"start_s": 0.0, "end_s": 1.2, "text": "Caption text..."},
    ...
  ],
  "narration": "A single voiceover script for TTS. MUST match the captions (same words, same order).",
  "medical_uncertainties": ["Any parts that look medically questionable or ambiguous", ...]
}

Hard requirements:
- The narration should last the FULL video duration (do not end early).
- Target speaking rate: ~150 words/minute (~2.5 words/second).
- Captions must cover the entire timeline: first start_s=0.0, last end_s should be close to the end of the video.
- Captions must be exactly what the narrator says during that time window.
- Do not invent events not visible; if unclear, list in medical_uncertainties instead of guessing.

Style:
- Use concise, professional medical language (anatomy + process).
- No filler words, no emojis, no disclaimers.
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
        index_client = _get_attr(client, "indexes", "index")
        create_index = _get_attr(index_client, "create") if index_client is not None else None
        if create_index is None:
            raise RuntimeError("TWELVELABS_INDEX_ID must be set (this TwelveLabs SDK has no index create API)")

        engines: list[dict[str, Any]] = [
            {
                "name": config.model_name,
                "options": [opt for opt, on in [("visual", config.enable_visual), ("audio", config.enable_audio)] if on],
            }
        ]

        sig = inspect.signature(create_index)
        kwargs: dict[str, Any] = {}
        if "index_name" in sig.parameters:
            kwargs["index_name"] = "Medical Diffusion Hackathon Index"
        elif "indexName" in sig.parameters:
            kwargs["indexName"] = "Medical Diffusion Hackathon Index"
        elif "name" in sig.parameters:
            kwargs["name"] = "Medical Diffusion Hackathon Index"
        else:
            raise RuntimeError("Unsupported TwelveLabs indexes.create signature (missing index_name/name)")

        if "models" in sig.parameters:
            kwargs["models"] = engines
        elif "engines" in sig.parameters:
            kwargs["engines"] = engines

        index = create_index(**kwargs)
        index_id = _get_id(index)

    if not index_id:
        raise RuntimeError("TwelveLabs index_id is missing")

    task_client = _get_attr(client, "tasks", "task")
    create_task = _get_attr(task_client, "create") if task_client is not None else None
    if create_task is None:
        raise RuntimeError("This TwelveLabs SDK has no task create API")

    task_sig = inspect.signature(create_task)
    task_kwargs: dict[str, Any] = {}
    if "index_id" in task_sig.parameters:
        task_kwargs["index_id"] = index_id
    elif "indexId" in task_sig.parameters:
        task_kwargs["indexId"] = index_id
    else:
        raise RuntimeError("Unsupported TwelveLabs task.create signature (missing index_id)")

    video_param = None
    for candidate in ("video_file", "videoFile", "file", "video"):
        if candidate in task_sig.parameters:
            video_param = candidate
            break
    if not video_param:
        raise RuntimeError("Unsupported TwelveLabs task.create signature (missing video file param)")

    if "language" in task_sig.parameters:
        task_kwargs["language"] = config.language

    mime_type = mimetypes.guess_type(str(video_path))[0] or "video/mp4"

    task = None
    # Preferred for v1.1+: video_file=(filename, fileobj, content_type)
    with open(video_path, "rb") as f:
        attempts: list[Any] = []
        if video_param in {"video_file", "videoFile"}:
            attempts = [(video_path.name, f, mime_type), f, str(video_path)]
        else:
            attempts = [str(video_path), f]

        for value in attempts:
            try:
                task = create_task(**{**task_kwargs, video_param: value})
                break
            except TypeError:
                continue

    if task is None:
        raise RuntimeError("Unsupported TwelveLabs task.create signature for file upload")

    # Wait for indexing to finish.
    task_id = _get_id(task)
    video_id = str(_get_attr(task, "video_id") or "").strip() or None

    wait_task = _get_attr(task_client, "wait_for_done") if task_client is not None else None
    if callable(wait_task) and task_id:
        task = wait_task(task_id)
        video_id = str(_get_attr(task, "video_id") or "").strip() or video_id
    else:
        wait_for_done = _get_attr(task, "wait_for_done")
        if callable(wait_for_done):
            wait_for_done()

    video_id = (video_id or "").strip()

    if not video_id:
        raise RuntimeError("TwelveLabs task completed but no video_id was returned")

    duration_s = get_video_duration_seconds(video_path)
    prompt_with_duration = prompt
    if duration_s:
        target_words = int(round(float(duration_s) * 2.5))
        prompt_with_duration = (
            f"video_duration_s: {duration_s:.2f}\n"
            f"voiceover_target_words: ~{target_words} (at ~2.5 words/second)\n\n"
            f"{prompt}"
        )

    # SDK uses keyword-only params for analyze(): video_id=..., prompt=...
    res = client.analyze(video_id=video_id, prompt=prompt_with_duration)
    raw = getattr(res, "data", None) or ""

    parsed = parse_caption_result(str(raw))
    if parsed:
        segs = normalize_segments(parsed.segments)
        if duration_s and segs:
            segs = [
                CaptionSegment(start_s=s.start_s, end_s=min(float(duration_s), s.end_s), text=s.text)
                for s in segs
                if s.start_s < float(duration_s)
            ]
        return CaptionResult(
            summary=parsed.summary,
            segments=segs,
            narration=parsed.narration,
            medical_uncertainties=parsed.medical_uncertainties,
        )

    # Fallback: treat the whole response as a narration blob.
    narration = str(raw).strip()
    end_s = float(duration_s) if duration_s else 1.0
    segments = [CaptionSegment(start_s=0.0, end_s=end_s, text=narration[:240] + ("…" if len(narration) > 240 else ""))] if narration else []
    return CaptionResult(summary="", segments=segments, narration=narration, medical_uncertainties=[])
