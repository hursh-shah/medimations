from __future__ import annotations

import datetime as _dt
import random
from dataclasses import dataclass, replace
from pathlib import Path
from typing import List, Optional, Sequence

from .types import AgentResult, AgentRound, AnimationSpec, GenerationResult, ValidationReport, ValidationScore
from .validation.base import Validator


@dataclass(frozen=True)
class AgentConfig:
    # "max 2 reprompts" => 3 rounds total (initial + 2 retries).
    max_rounds: int = 3
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
        return replace(spec, prompt=new_prompt)


class GeminiPromptAdjuster(PromptAdjuster):
    """
    Uses Gemini to rewrite the Veo prompt based on validator feedback.

    Falls back to RuleBasedPromptAdjuster if Gemini is unavailable.
    """

    def __init__(self, *, model: str = "gemini-2.0-flash") -> None:
        self._model = model
        self._fallback = RuleBasedPromptAdjuster()

    def adjust(self, *, spec: AnimationSpec, report: ValidationReport, round_index: int) -> AnimationSpec:
        metadata = dict(spec.metadata or {})
        user_prompt = str(metadata.get("user_prompt") or "").strip() or spec.prompt

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

        system = """You are a prompt engineer for Veo 3.1 generating short biomedical animations.

You will be given:
- the original user request
- the previous Veo prompt
- validator scores + feedback
- suggested keywords to fix medical/physics issues

Output STRICT JSON (no markdown):
{
  "veo_prompt": "string",
  "negative_prompt": "string"
}
Keep the prompt a single concise paragraph. Include: subject, action, setting, camera, style, constraints (no on-screen text/watermark).
"""

        user = f"""original_user_request: {user_prompt}
previous_veo_prompt: {spec.prompt}
round_index: {round_index}
medical_score: {report.medical.score:.3f}
medical_feedback: {report.medical.feedback}
physics_score: {report.physics.score:.3f}
physics_feedback: {report.physics.feedback}
suggested_keywords: {", ".join(suggested)}
"""

        try:
            from .gemini import generate_json, get_optional_str, load_gemini_config

            cfg = load_gemini_config(model=self._model)
            data = generate_json(system=system, user=user, config=cfg)
            veo_prompt = get_optional_str(data, "veo_prompt")
            negative_prompt = get_optional_str(data, "negative_prompt")
            if not veo_prompt:
                raise RuntimeError("Gemini returned no veo_prompt")
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
