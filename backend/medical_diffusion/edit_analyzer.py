"""
Gemini-based edit analyzer for smart video corrections.

This module uses Gemini to analyze validation feedback and determine
the best correction strategy: targeted edits (insert/remove) vs full regeneration.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from .gemini import GeminiConfig, GeminiError, generate_json, get_optional_str, load_gemini_config
from .types import ValidationReport


class CorrectionAction(str, Enum):
    """Type of correction action to take."""
    REGENERATE = "regenerate"  # Full video regeneration with improved prompt
    INSERT = "insert"  # Insert missing anatomical structure
    REMOVE = "remove"  # Remove incorrect/extra structure
    NO_ACTION = "no_action"  # Video is acceptable


@dataclass(frozen=True)
class EditInstruction:
    """Instruction for a single edit operation."""
    action: CorrectionAction
    target_object: Optional[str] = None  # Object to insert or remove
    region_description: Optional[str] = None  # Where in the frame
    insertion_prompt: Optional[str] = None  # Full prompt for INSERT
    reason: Optional[str] = None  # Why this edit is needed


@dataclass(frozen=True)
class CorrectionPlan:
    """Plan for correcting a medically inaccurate video."""
    primary_action: CorrectionAction
    edits: List[EditInstruction]
    regeneration_prompt: Optional[str] = None  # Only if primary_action is REGENERATE
    confidence: float = 0.0  # 0-1 confidence in the plan
    reasoning: Optional[str] = None


_ANALYZER_SYSTEM = """You are a medical imaging expert and AI video editing specialist.

Your task: Analyze validation feedback about a medical animation video and determine the most efficient way to fix the issues.

== CORRECTION STRATEGIES ==

1. **INSERT** - Add a missing anatomical structure to the video
   - Use when: A required organ/structure is missing but the rest is correct
   - Example: "The animation shows the heart but is missing the aorta"
   - Requires: Precise description of what to add and where

2. **REMOVE** - Remove an incorrect or extra structure from the video
   - Use when: There's an incorrect structure that shouldn't be there
   - Example: "There's an extra vessel that doesn't exist anatomically"
   - Requires: Description of what to remove and where

3. **REGENERATE** - Create a completely new video with an improved prompt
   - Use when: The issues are too fundamental to fix with edits
   - Examples: Wrong anatomy entirely, wrong viewpoint, wrong scale, multiple severe issues
   - Use as a last resort if targeted edits won't work

4. **NO_ACTION** - The video is acceptable
   - Use when: Validation scores are passing or issues are minor/acceptable

== DECISION GUIDELINES ==

- Prefer INSERT/REMOVE for 1-2 specific anatomical errors
- Use REGENERATE if there are 3+ issues or fundamental problems
- Consider the validation scores: 
  - Score >= 0.85: Usually acceptable, consider NO_ACTION
  - Score 0.6-0.85: Likely fixable with INSERT/REMOVE
  - Score < 0.6: Likely needs REGENERATE

== OUTPUT FORMAT ==

Output STRICT JSON:
{
  "primary_action": "insert" | "remove" | "regenerate" | "no_action",
  "edits": [
    {
      "action": "insert" | "remove",
      "target_object": "string - the anatomical structure",
      "region_description": "string - where in the frame (e.g., 'upper left quadrant', 'center of frame')",
      "insertion_prompt": "string - for INSERT: describe what to add (e.g., 'a realistic aorta connected to the left ventricle')",
      "reason": "string - why this edit is needed"
    }
  ],
  "regeneration_prompt": "string - only if primary_action is 'regenerate', the improved full prompt",
  "confidence": 0.0-1.0,
  "reasoning": "string - brief explanation of the correction strategy"
}

For NO_ACTION, edits should be empty array.
For INSERT/REMOVE, edits contains the specific operations.
For REGENERATE, edits should be empty and regeneration_prompt must be provided.

No markdown. No extra keys.
"""


def analyze_for_corrections(
    *,
    original_prompt: str,
    report: ValidationReport,
    model: str = "gemini-3.0-flash",
    gemini_config: Optional[GeminiConfig] = None,
) -> CorrectionPlan:
    """
    Analyze validation feedback and determine the best correction strategy.
    
    Uses Gemini to analyze the medical and physics validation results
    and decide whether to use targeted edits (insert/remove) or full regeneration.
    
    Args:
        original_prompt: The prompt used to generate the video
        report: The validation report with scores and feedback
        model: Gemini model to use
        gemini_config: Optional pre-configured Gemini config
        
    Returns:
        CorrectionPlan with the recommended correction strategy
    """
    cfg = gemini_config or load_gemini_config(model=model)

    # Collect detailed feedback
    medical_details = report.medical.details if isinstance(report.medical.details, dict) else {}
    physics_details = report.physics.details if isinstance(report.physics.details, dict) else {}
    
    suggested_keywords = []
    suggested_keywords.extend(medical_details.get("suggested_keywords", []) or [])
    suggested_keywords.extend(physics_details.get("suggested_keywords", []) or [])
    
    # Get component-level details if available
    components = medical_details.get("components", []) or []
    component_feedback = []
    for comp in components:
        if isinstance(comp, dict) and not comp.get("skipped"):
            component_feedback.append(f"- {comp.get('name', 'Unknown')}: score={comp.get('score', 0):.3f}, details={comp.get('details', {})}")

    user = f"""ANALYZE VIDEO FOR CORRECTIONS

Original generation prompt:
{original_prompt}

== VALIDATION RESULTS ==

Medical Accuracy:
- Score: {report.medical.score:.3f} (threshold: 0.85)
- Feedback: {report.medical.feedback or "No specific feedback"}
- Status: {"PASSING" if report.medical.score >= 0.85 else "FAILING"}

Physics/Motion:
- Score: {report.physics.score:.3f} (threshold: 0.85)
- Feedback: {report.physics.feedback or "No specific feedback"}
- Status: {"PASSING" if report.physics.score >= 0.85 else "FAILING"}

Suggested keywords to address issues:
{", ".join(suggested_keywords) if suggested_keywords else "None"}

Component-level feedback:
{chr(10).join(component_feedback) if component_feedback else "No component details available"}

== TASK ==

Analyze these results and determine:
1. Is the video acceptable as-is? (NO_ACTION)
2. Can specific issues be fixed with INSERT or REMOVE edits?
3. Or does the video need full REGENERATION?

If recommending INSERT/REMOVE, be specific about:
- What anatomical structure to add/remove
- Where in the frame it should be edited
- For INSERT: what the inserted object should look like

Output your correction plan as JSON.
"""

    try:
        data = generate_json(system=_ANALYZER_SYSTEM, user=user, config=cfg)
    except Exception as e:
        # Fallback to regeneration if analysis fails
        return CorrectionPlan(
            primary_action=CorrectionAction.REGENERATE,
            edits=[],
            regeneration_prompt=None,
            confidence=0.0,
            reasoning=f"Analysis failed: {e}. Falling back to regeneration.",
        )

    # Parse primary action
    primary_action_str = get_optional_str(data, "primary_action") or "regenerate"
    try:
        primary_action = CorrectionAction(primary_action_str.lower())
    except ValueError:
        primary_action = CorrectionAction.REGENERATE

    # Parse edits
    edits: List[EditInstruction] = []
    raw_edits = data.get("edits", [])
    if isinstance(raw_edits, list):
        for edit_data in raw_edits:
            if not isinstance(edit_data, dict):
                continue
            action_str = get_optional_str(edit_data, "action") or ""
            try:
                action = CorrectionAction(action_str.lower())
            except ValueError:
                continue
            
            if action in (CorrectionAction.INSERT, CorrectionAction.REMOVE):
                edits.append(EditInstruction(
                    action=action,
                    target_object=get_optional_str(edit_data, "target_object"),
                    region_description=get_optional_str(edit_data, "region_description"),
                    insertion_prompt=get_optional_str(edit_data, "insertion_prompt"),
                    reason=get_optional_str(edit_data, "reason"),
                ))

    # Parse other fields
    regeneration_prompt = get_optional_str(data, "regeneration_prompt")
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))
    reasoning = get_optional_str(data, "reasoning")

    return CorrectionPlan(
        primary_action=primary_action,
        edits=edits,
        regeneration_prompt=regeneration_prompt,
        confidence=confidence,
        reasoning=reasoning,
    )


def generate_mask_prompt(
    *,
    edit: EditInstruction,
    frame_description: Optional[str] = None,
    model: str = "gemini-3.0-flash",
    gemini_config: Optional[GeminiConfig] = None,
) -> str:
    """
    Generate a description for mask creation based on the edit instruction.
    
    This returns a text description of where the mask should be placed,
    which can be used to guide manual mask creation or a future
    automated mask generation system.
    
    Args:
        edit: The edit instruction
        frame_description: Optional description of what's in the frame
        model: Gemini model to use
        gemini_config: Optional pre-configured Gemini config
        
    Returns:
        Text description of where to place the mask
    """
    cfg = gemini_config or load_gemini_config(model=model)
    
    action = "INSERT (add object)" if edit.action == CorrectionAction.INSERT else "REMOVE (remove object)"
    
    user = f"""Generate a clear mask placement description for video editing.

Edit operation: {action}
Target object: {edit.target_object or "unspecified"}
Region hint: {edit.region_description or "unspecified"}
{f"Frame context: {frame_description}" if frame_description else ""}

Describe precisely where a mask should be placed in the video frame for this edit.
The mask will be used to: {"add" if edit.action == CorrectionAction.INSERT else "remove"} the target object.

Output a single paragraph describing the mask placement in clear, actionable terms.
Include: approximate position (top/bottom/left/right/center), size relative to frame, shape if relevant.
"""

    system = "You are a video editing assistant. Generate clear, concise mask placement descriptions."
    
    try:
        from .gemini import generate_text
        return generate_text(system=system, user=user, config=cfg)
    except Exception:
        # Fallback
        return f"Place mask at {edit.region_description or 'center'} to {edit.action.value} {edit.target_object or 'target object'}"
