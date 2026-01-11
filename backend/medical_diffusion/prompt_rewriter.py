from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .gemini import GeminiConfig, GeminiError, generate_json, get_optional_str, load_gemini_config


@dataclass(frozen=True)
class PromptRewriteResult:
    veo_prompt: str
    negative_prompt: Optional[str]
    target_label: Optional[str]
    notes: Optional[str]


_SYSTEM = """You are a prompt engineer for Veo 3.1 generating short biomedical animations.

Goal: turn a user's vague request into a concrete, medically oriented Veo prompt that is:
- anatomically and clinically plausible (no fantasy anatomy)
- visually consistent over time (avoid flicker / random cuts)
- suitable for hackathon demos (clean background, no on-screen text)

Output STRICT JSON with these keys:
{
  "veo_prompt": "string",
  "negative_prompt": "string",
  "target_label": "string (single short noun phrase like 'heart' or 'capillary')",
  "notes": "string (optional, short)"
}
No markdown. No extra keys.
"""


def rewrite_user_prompt_for_veo(
    *,
    user_prompt: str,
    duration_s: float,
    fps: int,
    reference_image_provided: bool = False,
    model: str = "gemini-3.0-flash",
    gemini_config: Optional[GeminiConfig] = None,
) -> PromptRewriteResult:
    user_prompt = user_prompt.strip()
    if not user_prompt:
        raise ValueError("user_prompt is empty")

    cfg = gemini_config or load_gemini_config(model=model)

    ref = "yes (image-to-video; preserve the provided image)" if reference_image_provided else "no (text-only)"
    user = f"""User request: {user_prompt}
Reference image provided: {ref}

Video constraints:
- duration_seconds: {duration_s:g}
- fps: {int(fps)}

Write the Veo prompt as a single concise paragraph.
Include: subject, action, setting, camera framing, style, constraints (no text/watermark).
If a reference image is provided, assume it is the first frame and do not contradict its anatomy/view.
"""

    data = generate_json(system=_SYSTEM, user=user, config=cfg)
    veo_prompt = get_optional_str(data, "veo_prompt")
    if not veo_prompt:
        raise GeminiError("Gemini did not return a veo_prompt")

    return PromptRewriteResult(
        veo_prompt=veo_prompt,
        negative_prompt=get_optional_str(data, "negative_prompt"),
        target_label=get_optional_str(data, "target_label"),
        notes=get_optional_str(data, "notes"),
    )
