from __future__ import annotations

import datetime as _dt
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence

from .types import AgentResult, AgentRound, AnimationSpec, GenerationResult, ValidationReport, ValidationScore
from .validation.base import Validator
from .veo_guidelines import get_adjuster_system_prompt, get_default_negative_prompt


@dataclass(frozen=True)
class AgentConfig:
    # "max 1 reprompt" => 2 rounds total (initial + 1 retry).
    max_rounds: int = 2
    candidates_per_round: int = 1
    medical_threshold: float = 0.85
    physics_threshold: float = 0.85


class PromptAdjuster:
    def adjust(self, *, spec: AnimationSpec, report: ValidationReport, round_index: int) -> AnimationSpec:
        raise NotImplementedError


class RuleBasedPromptAdjuster(PromptAdjuster):
    def adjust(self, *, spec: AnimationSpec, report: ValidationReport, round_index: int) -> AnimationSpec:
        suggested = []
        suggested.extend(report.medical.details.get("suggested_keywords", []))
        suggested.extend(report.physics.details.get("suggested_keywords", []))
        suggested = [str(s).strip() for s in suggested if str(s).strip()]
        # De-dupe while preserving order.
        seen = set()
        suggested = [s for s in suggested if not (s in seen or seen.add(s))]
        suggested = suggested[:12]

        new_prompt = spec.prompt.rstrip()
        if suggested:
            new_prompt = new_prompt + " " + " ".join(suggested)
        else:
            new_prompt = new_prompt + " anatomically accurate, temporally consistent, physically plausible motion"
            if spec.input_image_path is not None:
                new_prompt = new_prompt + "; preserve the reference image anatomy/viewpoint"
        return replace(spec, prompt=new_prompt)


class GeminiPromptAdjuster(PromptAdjuster):
    """
    Uses Gemini to rewrite the Veo prompt based on validator feedback.

    Falls back to RuleBasedPromptAdjuster if Gemini is unavailable.
    """

    def __init__(self, *, model: str = "gemini-3.0-flash") -> None:
        self._model = model
        self._fallback = RuleBasedPromptAdjuster()

    def adjust(self, *, spec: AnimationSpec, report: ValidationReport, round_index: int) -> AnimationSpec:
        """
        Adjust the Veo prompt based on validation feedback using Gemini.
        
        Uses the comprehensive Veo guidelines to generate an improved prompt
        that addresses the specific issues identified by validators.
        """
        metadata = dict(spec.metadata or {})
        user_prompt = str(metadata.get("user_prompt") or "").strip() or spec.prompt

        # Collect suggested keywords from validators
        suggested = []
        if isinstance(report.medical.details, dict):
            suggested.extend(report.medical.details.get("suggested_keywords", []) or [])
        if isinstance(report.physics.details, dict):
            suggested.extend(report.physics.details.get("suggested_keywords", []) or [])
        suggested = [str(s).strip() for s in suggested if str(s).strip()]
        # De-dupe while preserving order.
        seen = set()
        suggested = [s for s in suggested if not (s in seen or seen.add(s))]
        suggested = suggested[:12]

        # Build reference image context
        if spec.input_image_path is not None:
            ref_context = (
                "A reference image IS provided (image-to-video mode). "
                "The reference image is the starting frame. "
                "You MUST preserve its anatomy, viewpoint, and visual style. "
                "Only animate realistic motion consistent with the reference."
            )
        else:
            ref_context = "No reference image is provided (text-only generation)."

        # Build validation feedback summary
        validation_summary = f"""
VALIDATION RESULTS (scores are 0.0 to 1.0, higher is better):

Medical Accuracy:
- Score: {report.medical.score:.3f}
- Feedback: {report.medical.feedback or "No specific feedback"}
- Issues to address: {"Passing" if report.medical.score >= 0.85 else "Needs improvement - strengthen anatomical accuracy"}

Physics/Motion:
- Score: {report.physics.score:.3f}
- Feedback: {report.physics.feedback or "No specific feedback"}
- Issues to address: {"Passing" if report.physics.score >= 0.85 else "Needs improvement - fix temporal consistency and motion physics"}

Suggested keywords to incorporate: {", ".join(suggested) if suggested else "None provided"}
"""

        # Use comprehensive system prompt from veo_guidelines
        system = get_adjuster_system_prompt()

        user = f"""REPROMPTING TASK

Original user request: {user_prompt}

Previous Veo prompt (to improve):
{spec.prompt}

Reference image: {ref_context}

Round: {round_index + 1} (attempting to fix validation issues)

{validation_summary}

Instructions:
1. Analyze the validation feedback carefully
2. Identify which Veo prompt elements need strengthening
3. Incorporate suggested keywords naturally into the prompt
4. Strengthen anatomical accuracy and temporal consistency language
5. Maintain successful elements from the previous prompt
6. Follow Veo best practices for camera, lighting, and style
7. Ensure constraints: no on-screen text, no watermarks

Output the improved JSON response.
"""

        try:
            from .gemini import generate_json, get_optional_str, load_gemini_config

            cfg = load_gemini_config(model=self._model)
            data = generate_json(system=system, user=user, config=cfg)
            veo_prompt = get_optional_str(data, "veo_prompt")
            negative_prompt = get_optional_str(data, "negative_prompt")
            if not veo_prompt:
                raise RuntimeError("Gemini returned no veo_prompt")
            
            # Use default negative prompt if none provided
            if not negative_prompt:
                negative_prompt = get_default_negative_prompt()
                
        except Exception as e:
            metadata["reprompt_mode"] = "rule_fallback"
            metadata["reprompt_error"] = str(e)
            return self._fallback.adjust(spec=replace(spec, metadata=metadata), report=report, round_index=round_index)

        metadata["reprompt_mode"] = "gemini"
        metadata["reprompt_model"] = self._model
        metadata["reprompt_round_index"] = round_index

        return replace(
            spec,
            prompt=veo_prompt,
            negative_prompt=spec.negative_prompt or negative_prompt,
            metadata=metadata,
        )


def build_report(*, medical: ValidationScore, physics: ValidationScore) -> ValidationReport:
    return ValidationReport(medical=medical, physics=physics)


class ValidatorAgent:
    def __init__(
        self,
        *,
        generator,
        medical_validators: Sequence[Validator],
        physics_validators: Sequence[Validator],
        config: AgentConfig,
        prompt_adjuster: Optional[PromptAdjuster] = None,
        run_root: Optional[Path] = None,
    ) -> None:
        self._generator = generator
        self._medical_validators = list(medical_validators)
        self._physics_validators = list(physics_validators)
        self._config = config
        self._prompt_adjuster = prompt_adjuster or RuleBasedPromptAdjuster()
        self._run_root = run_root

    def run(self, *, spec: AnimationSpec) -> AgentResult:
        run_root = self._run_root or Path("runs") / _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root.mkdir(parents=True, exist_ok=True)

        history: List[AgentRound] = []
        current_spec = spec

        for round_index in range(self._config.max_rounds):
            best_round: Optional[AgentRound] = None
            best_score = float("-inf")

            for candidate_index in range(self._config.candidates_per_round):
                seed = (
                    current_spec.seed
                    if current_spec.seed is not None
                    else random.randint(0, 2**31 - 1)
                )
                candidate_spec = replace(current_spec, seed=seed)
                out_dir = run_root / f"round_{round_index:02d}" / f"cand_{candidate_index:02d}"
                generation: GenerationResult = self._generator.generate(spec=candidate_spec, output_dir=out_dir)

                medical_score = _aggregate_scores(self._medical_validators, generation)
                physics_score = _aggregate_scores(self._physics_validators, generation)
                report = build_report(medical=medical_score, physics=physics_score)

                agent_round = AgentRound(
                    round_index=round_index,
                    prompt=candidate_spec.prompt,
                    candidate_index=candidate_index,
                    generation=generation,
                    report=report,
                )
                history.append(agent_round)

                combined = _combined_score(
                    report,
                    medical_threshold=self._config.medical_threshold,
                    physics_threshold=self._config.physics_threshold,
                )
                if combined > best_score:
                    best_score = combined
                    best_round = agent_round

                if _accepted(report, self._config):
                    return AgentResult(accepted=True, final=agent_round, history=history)

            if best_round is None:
                raise RuntimeError("Agent produced no candidates")

            current_spec = self._prompt_adjuster.adjust(
                spec=replace(current_spec, prompt=best_round.prompt),
                report=best_round.report,
                round_index=round_index,
            )

        assert history
        return AgentResult(accepted=False, final=history[-1], history=history)


def _aggregate_scores(validators: Sequence[Validator], generation: GenerationResult) -> ValidationScore:
    if not validators:
        return ValidationScore(name="none", score=1.0, skipped=True)

    scores = [v.score(generation) for v in validators]
    active = [s for s in scores if not s.skipped]
    if not active:
        return ValidationScore(
            name="none",
            score=1.0,
            skipped=True,
            details={"components": [{"name": s.name, "score": s.score, "skipped": s.skipped, "details": s.details} for s in scores]},
            feedback="All validators skipped",
        )

    avg = sum(s.score for s in active) / len(active)
    feedback = " ".join([s.feedback for s in active if s.feedback]).strip()

    suggested_keywords: List[str] = []
    for s in scores:
        if not isinstance(s.details, dict):
            continue
        maybe = s.details.get("suggested_keywords")
        if isinstance(maybe, list):
            suggested_keywords.extend([str(x) for x in maybe if x])
    # De-dupe while preserving order.
    seen = set()
    suggested_keywords = [k for k in suggested_keywords if not (k in seen or seen.add(k))]

    details = {
        "components": [
            {"name": s.name, "score": s.score, "skipped": s.skipped, "details": s.details} for s in scores
        ],
        "suggested_keywords": suggested_keywords,
    }
    return ValidationScore(name="aggregate", score=float(avg), details=details, feedback=feedback)


def _accepted(report: ValidationReport, config: AgentConfig) -> bool:
    return (report.medical.score >= config.medical_threshold) and (report.physics.score >= config.physics_threshold)


def _combined_score(report: ValidationReport, *, medical_threshold: float, physics_threshold: float) -> float:
    # Prefer candidates that exceed thresholds, otherwise prefer closer-to-threshold.
    m = report.medical.score / max(1e-6, medical_threshold)
    p = report.physics.score / max(1e-6, physics_threshold)
    return m + p


@dataclass(frozen=True)
class SmartAgentConfig(AgentConfig):
    """Configuration for SmartValidatorAgent with edit capabilities."""
    # Enable smart editing (INSERT/REMOVE) instead of always regenerating
    enable_smart_edits: bool = True
    # Minimum confidence required to attempt a targeted edit vs regeneration
    edit_confidence_threshold: float = 0.6
    # Maximum number of targeted edits to attempt before falling back to regeneration
    max_edit_attempts: int = 2


class SmartValidatorAgent:
    """
    Enhanced validator agent that can perform targeted edits.
    
    Instead of always regenerating the entire video when validation fails,
    this agent can analyze the specific issues and attempt targeted edits
    (INSERT missing structures or REMOVE incorrect ones) using Veo's
    mask-based editing capabilities.
    
    Falls back to full regeneration when:
    - Smart edits are disabled
    - Too many edits would be required
    - Confidence in edit strategy is too low
    - Edit attempts fail
    """

    def __init__(
        self,
        *,
        generator,  # VeoGenaiBackend
        medical_validators: Sequence[Validator],
        physics_validators: Sequence[Validator],
        config: SmartAgentConfig,
        prompt_adjuster: Optional[PromptAdjuster] = None,
        gemini_model: str = "gemini-3.0-flash",
        run_root: Optional[Path] = None,
    ) -> None:
        self._generator = generator
        self._medical_validators = list(medical_validators)
        self._physics_validators = list(physics_validators)
        self._config = config
        self._prompt_adjuster = prompt_adjuster or GeminiPromptAdjuster(model=gemini_model)
        self._gemini_model = gemini_model
        self._run_root = run_root

    def run(self, *, spec: AnimationSpec) -> AgentResult:
        """
        Run the smart validation loop.
        
        1. Generate initial video
        2. Validate
        3. If failing, analyze whether to edit or regenerate
        4. Attempt edits if appropriate, otherwise regenerate with improved prompt
        5. Repeat until passing or max rounds reached
        """
        run_root = self._run_root or Path("runs") / _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_root.mkdir(parents=True, exist_ok=True)

        history: List[AgentRound] = []
        current_spec = spec
        edit_attempts_this_round = 0
        last_generation: Optional[GenerationResult] = None
        last_video_path: Optional[Path] = None

        for round_index in range(self._config.max_rounds):
            best_round: Optional[AgentRound] = None
            best_score = float("-inf")

            for candidate_index in range(self._config.candidates_per_round):
                seed = (
                    current_spec.seed
                    if current_spec.seed is not None
                    else random.randint(0, 2**31 - 1)
                )
                candidate_spec = replace(current_spec, seed=seed)
                out_dir = run_root / f"round_{round_index:02d}" / f"cand_{candidate_index:02d}"
                
                generation: GenerationResult = self._generator.generate(spec=candidate_spec, output_dir=out_dir)
                last_generation = generation
                
                # Track video path for potential edits
                video_path = generation.metadata.get("video_path")
                if video_path:
                    last_video_path = Path(video_path)

                medical_score = _aggregate_scores(self._medical_validators, generation)
                physics_score = _aggregate_scores(self._physics_validators, generation)
                report = build_report(medical=medical_score, physics=physics_score)

                agent_round = AgentRound(
                    round_index=round_index,
                    prompt=candidate_spec.prompt,
                    candidate_index=candidate_index,
                    generation=generation,
                    report=report,
                )
                history.append(agent_round)

                combined = _combined_score(
                    report,
                    medical_threshold=self._config.medical_threshold,
                    physics_threshold=self._config.physics_threshold,
                )
                if combined > best_score:
                    best_score = combined
                    best_round = agent_round

                if _accepted(report, self._config):
                    return AgentResult(accepted=True, final=agent_round, history=history)

            if best_round is None:
                raise RuntimeError("Agent produced no candidates")

            # Decide correction strategy
            use_smart_edit = False
            correction_plan = None
            
            if (
                self._config.enable_smart_edits
                and last_video_path is not None
                and last_video_path.exists()
                and edit_attempts_this_round < self._config.max_edit_attempts
            ):
                # Analyze for potential targeted edits
                try:
                    from .edit_analyzer import CorrectionAction, analyze_for_corrections
                    
                    correction_plan = analyze_for_corrections(
                        original_prompt=current_spec.prompt,
                        report=best_round.report,
                        model=self._gemini_model,
                    )
                    
                    if (
                        correction_plan.primary_action in (CorrectionAction.INSERT, CorrectionAction.REMOVE)
                        and correction_plan.confidence >= self._config.edit_confidence_threshold
                        and len(correction_plan.edits) <= 2
                    ):
                        use_smart_edit = True
                        
                except Exception as e:
                    # Fall back to regeneration on analysis failure
                    metadata = dict(current_spec.metadata or {})
                    metadata["edit_analysis_error"] = str(e)
                    current_spec = replace(current_spec, metadata=metadata)

            if use_smart_edit and correction_plan is not None and correction_plan.edits:
                # Attempt targeted edit
                try:
                    edited_generation = self._apply_edit(
                        video_path=last_video_path,
                        edit=correction_plan.edits[0],
                        out_dir=run_root / f"round_{round_index:02d}" / "edit_{:02d}".format(edit_attempts_this_round),
                    )
                    
                    if edited_generation is not None:
                        # Validate the edited video
                        medical_score = _aggregate_scores(self._medical_validators, edited_generation)
                        physics_score = _aggregate_scores(self._physics_validators, edited_generation)
                        edit_report = build_report(medical=medical_score, physics=physics_score)
                        
                        edit_round = AgentRound(
                            round_index=round_index,
                            prompt=f"[EDIT:{correction_plan.edits[0].action.value}] {correction_plan.edits[0].target_object}",
                            candidate_index=candidate_index + 1,
                            generation=edited_generation,
                            report=edit_report,
                        )
                        history.append(edit_round)
                        
                        if _accepted(edit_report, self._config):
                            return AgentResult(accepted=True, final=edit_round, history=history)
                        
                        # Update for next round
                        last_generation = edited_generation
                        video_path = edited_generation.metadata.get("video_path")
                        if video_path:
                            last_video_path = Path(video_path)
                        
                        edit_attempts_this_round += 1
                        continue  # Try another edit or regenerate
                        
                except Exception as e:
                    metadata = dict(current_spec.metadata or {})
                    metadata["edit_apply_error"] = str(e)
                    current_spec = replace(current_spec, metadata=metadata)

            # Fall back to regeneration with improved prompt
            edit_attempts_this_round = 0  # Reset for new round
            
            # Use regeneration prompt from correction plan if available
            if correction_plan is not None and correction_plan.regeneration_prompt:
                metadata = dict(current_spec.metadata or {})
                metadata["correction_reasoning"] = correction_plan.reasoning
                current_spec = replace(
                    current_spec,
                    prompt=correction_plan.regeneration_prompt,
                    metadata=metadata,
                )
            else:
                current_spec = self._prompt_adjuster.adjust(
                    spec=replace(current_spec, prompt=best_round.prompt),
                    report=best_round.report,
                    round_index=round_index,
                )

        assert history
        return AgentResult(accepted=False, final=history[-1], history=history)

    def _apply_edit(
        self,
        *,
        video_path: Path,
        edit,  # EditInstruction
        out_dir: Path,
    ) -> Optional[GenerationResult]:
        """
        Apply a targeted edit (INSERT or REMOVE) to a video.
        
        Returns a GenerationResult if successful, None if edit cannot be applied.
        """
        from .edit_analyzer import CorrectionAction
        from .generation.veo_genai import VideoEditMode
        from .mask_generator import generate_mask_from_description, get_video_dimensions
        
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Get video dimensions
        try:
            width, height = get_video_dimensions(video_path)
        except Exception:
            width, height = 720, 1280
        
        # Generate mask
        mask_path = out_dir / "edit_mask.png"
        region_desc = edit.region_description or "center"
        
        generate_mask_from_description(
            description=region_desc,
            output_path=mask_path,
            width=width,
            height=height,
            shape="ellipse",  # Better for organic structures
            feather=15,
        )
        
        # Determine edit mode
        if edit.action == CorrectionAction.INSERT:
            mode = VideoEditMode.INSERT
            prompt = edit.insertion_prompt or edit.target_object or ""
        else:
            mode = VideoEditMode.REMOVE
            prompt = None
        
        # Apply the edit
        result = self._generator.edit_video(
            source_video_path=video_path,
            mask_path=mask_path,
            mode=mode,
            prompt=prompt,
            output_dir=out_dir,
        )
        
        # Convert VideoEditResult to GenerationResult for validation
        spec = AnimationSpec(
            prompt=f"[{mode.value}] {edit.target_object or 'edit'}",
            negative_prompt=None,
        )
        
        return GenerationResult(
            spec=spec,
            frames=result.frames,
            frames_dir=result.frames_dir,
            backend="veo_edit",
            metadata={
                "video_path": str(result.video_path),
                "edit_mode": result.edit_mode.value,
                "source_video_path": str(result.source_video_path),
                "mask_path": str(result.mask_path),
            },
        )
