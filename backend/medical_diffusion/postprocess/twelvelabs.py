from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

    client = TwelveLabs(config.api_key)

    index_id = config.index_id
    if not index_id:
        index = client.index.create(
            name="Medical Diffusion Hackathon Index",
            models=[
                {
                    "name": config.model_name,
                    "options": [opt for opt, on in [("visual", config.enable_visual), ("audio", config.enable_audio)] if on],
                }
            ],
        )
        index_id = index.id

    task = client.task.create(index_id=index_id, file=str(video_path), language=config.language)
    task.wait_for_done()

    res = client.analyze(task.video_id, prompt)
    raw = getattr(res, "data", None) or ""

    parsed = parse_caption_result(str(raw))
    if parsed:
        return parsed

    # Fallback: treat the whole response as a narration blob.
    return CaptionResult(summary="", segments=[], narration=str(raw).strip(), medical_uncertainties=[])

