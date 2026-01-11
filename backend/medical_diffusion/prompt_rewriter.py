from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .gemini import GeminiConfig, GeminiError, generate_json, get_optional_str, load_gemini_config
from .veo_guidelines import get_default_negative_prompt, get_rewriter_system_prompt


@dataclass(frozen=True)
class PromptComponents:
    """Structured breakdown of the Veo prompt elements."""
    subject: Optional[str] = None
    action: Optional[str] = None
    scene: Optional[str] = None
    camera: Optional[str] = None
    style: Optional[str] = None


@dataclass(frozen=True)
class PromptRewriteResult:
    veo_prompt: str
    negative_prompt: Optional[str]
    target_label: Optional[str]
    notes: Optional[str]
    components: Optional[PromptComponents] = None


def _parse_components(data: Dict[str, Any]) -> Optional[PromptComponents]:
    """Extract prompt components from Gemini response."""
    components_data = data.get("components")
    if not isinstance(components_data, dict):
        return None
    return PromptComponents(
        subject=get_optional_str(components_data, "subject"),
        action=get_optional_str(components_data, "action"),
        scene=get_optional_str(components_data, "scene"),
        camera=get_optional_str(components_data, "camera"),
        style=get_optional_str(components_data, "style"),
    )


def rewrite_user_prompt_for_veo(
    *,
    user_prompt: str,
    duration_s: float,
    fps: int,
    reference_image_provided: bool = False,
    model: str = "gemini-3.0-flash",
    gemini_config: Optional[GeminiConfig] = None,
) -> PromptRewriteResult:
    """
    Rewrite a user's prompt into an optimized Veo prompt following best practices.
    
    Uses Gemini to transform a potentially vague user request into a detailed,
    structured Veo prompt that includes:
    - Specific subject description (anatomical structures, cells, etc.)
    - Clear action/movement description
    - Scene/context setting
    - Camera angle and movement direction
    - Visual style and lighting
    - Medical accuracy constraints
    
    Args:
        user_prompt: The original user request
        duration_s: Target video duration in seconds
        fps: Target frames per second
        reference_image_provided: Whether a reference image is provided for image-to-video
        model: Gemini model to use
        gemini_config: Optional pre-configured Gemini config
        
    Returns:
        PromptRewriteResult with the optimized prompt and metadata
    """
    user_prompt = user_prompt.strip()
    if not user_prompt:
        raise ValueError("user_prompt is empty")

    cfg = gemini_config or load_gemini_config(model=model)

    # Build reference image context
    if reference_image_provided:
        ref_context = (
            "A reference image IS provided (image-to-video mode). "
            "The reference image should be treated as the starting frame. "
            "Preserve its anatomy, viewpoint, and visual style. "
            "Only animate realistic motion consistent with the reference. "
            "Do not contradict or alter the anatomical structures shown."
        )
    else:
        ref_context = "No reference image is provided (text-only generation)."

    user = f"""User request: {user_prompt}

Reference image: {ref_context}

Video specifications:
- Duration: {duration_s:g} seconds
- Frame rate: {int(fps)} fps
- Total frames: approximately {int(duration_s * fps)} frames

Instructions:
1. Analyze the user's request and identify the medical/anatomical subject
2. Determine appropriate actions, camera work, and visual style
3. Construct a detailed Veo prompt following the guidelines
4. Ensure the prompt emphasizes medical accuracy and temporal consistency
5. Include constraints: no on-screen text, no watermarks, clean presentation

Output the complete JSON response with all fields.
"""

    # Use the comprehensive system prompt from veo_guidelines
    system = get_rewriter_system_prompt()
    
    data = generate_json(system=system, user=user, config=cfg)
    veo_prompt = get_optional_str(data, "veo_prompt")
    if not veo_prompt:
        raise GeminiError("Gemini did not return a veo_prompt")

    # Get negative prompt, falling back to default if not provided
    negative_prompt = get_optional_str(data, "negative_prompt")
    if not negative_prompt:
        negative_prompt = get_default_negative_prompt()

    # Parse structured components
    components = _parse_components(data)

    return PromptRewriteResult(
        veo_prompt=veo_prompt,
        negative_prompt=negative_prompt,
        target_label=get_optional_str(data, "target_label"),
        notes=get_optional_str(data, "notes"),
        components=components,
    )


@dataclass(frozen=True)
class ExtensionPromptResult:
    """Result of generating a video extension prompt."""
    extension_prompt: str
    negative_prompt: Optional[str]
    notes: Optional[str] = None


_EXTENSION_SYSTEM = """You are an expert prompt engineer for Veo 3.1 video extension. You are also a physician with extensive anatomical and clinical knowledge.

Your task: Generate a continuation prompt that will seamlessly extend an existing medical animation video.

== VIDEO EXTENSION PRINCIPLES ==

When extending a video:
1. CONTINUITY: The continuation must flow naturally from where the original ends
2. CONSISTENCY: Maintain the same visual style, lighting, camera work, and anatomical accuracy
3. PROGRESSION: The action should logically continue or progress the narrative
4. MEDICAL ACCURACY: All anatomical structures and processes must remain scientifically correct

== OUTPUT FORMAT ==

Output STRICT JSON with these keys:
{
  "extension_prompt": "string - the prompt for the video continuation",
  "negative_prompt": "string - elements to avoid",
  "notes": "string - optional brief notes about the continuation"
}

The extension_prompt should:
- Describe what happens NEXT in the video
- Reference the ongoing action/process naturally
- Maintain camera and style consistency
- Include temporal consistency cues
- Keep medical accuracy constraints

No markdown. No extra keys.
"""


def generate_extension_prompt(
    *,
    original_prompt: str,
    user_extension_hint: Optional[str] = None,
    model: str = "gemini-3.0-flash",
    gemini_config: Optional[GeminiConfig] = None,
) -> ExtensionPromptResult:
    """
    Generate a Gemini-supervised continuation prompt for video extension.
    
    Uses the original video's prompt and an optional user hint to generate
    a semantically coherent continuation prompt that maintains visual and
    medical accuracy consistency.
    
    Args:
        original_prompt: The prompt used to generate the original video
        user_extension_hint: Optional user-provided hint about what should happen next
        model: Gemini model to use
        gemini_config: Optional pre-configured Gemini config
        
    Returns:
        ExtensionPromptResult with the extension prompt and metadata
    """
    original_prompt = (original_prompt or "").strip()
    if not original_prompt:
        raise ValueError("original_prompt is empty")

    cfg = gemini_config or load_gemini_config(model=model)

    hint_section = ""
    if user_extension_hint and user_extension_hint.strip():
        hint_section = f"""
User's hint for what should happen next:
{user_extension_hint.strip()}

Incorporate this hint into the continuation while maintaining consistency with the original.
"""
    else:
        hint_section = """
No specific hint provided. Generate a natural, logical continuation of the medical animation.
Consider what would naturally happen next in this anatomical/physiological process.
"""

    user = f"""Original video prompt:
{original_prompt}

{hint_section}

Generate a continuation prompt that:
1. Flows seamlessly from where the original video would end
2. Maintains the same visual style, camera work, and lighting
3. Continues or progresses the medical/anatomical narrative
4. Preserves anatomical accuracy and temporal consistency
5. Includes constraints: no on-screen text, no watermarks

Output the complete JSON response.
"""

    data = generate_json(system=_EXTENSION_SYSTEM, user=user, config=cfg)
    extension_prompt = get_optional_str(data, "extension_prompt")
    if not extension_prompt:
        raise GeminiError("Gemini did not return an extension_prompt")

    negative_prompt = get_optional_str(data, "negative_prompt")
    if not negative_prompt:
        negative_prompt = get_default_negative_prompt()

    return ExtensionPromptResult(
        extension_prompt=extension_prompt,
        negative_prompt=negative_prompt,
        notes=get_optional_str(data, "notes"),
    )
