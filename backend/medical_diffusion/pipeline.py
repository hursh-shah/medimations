from __future__ import annotations

import datetime as _dt
import json
import math
import shutil
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .generation.veo_genai import VeoGenaiBackend
from .io.video import (
    VideoEncodeError,
    extract_frames_ffmpeg,
    get_video_duration_seconds,
)
from .prompt_rewriter import generate_extension_prompt
from .types import (
    AnimationSpec,
    GenerationResult,
    PipelineReport,
    SegmentResult,
    ValidationScore,
)
from .validation.base import Validator


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ExtensionPipelineConfig:
    target_duration_s: float = 60.0
    max_extensions: int = 20
    fps: int = 8
    gemini_model: str = "gemini-3-flash-preview"


# ---------------------------------------------------------------------------
# Score table formatting
# ---------------------------------------------------------------------------

def _fmt_score(vs: ValidationScore) -> str:
    if vs.skipped:
        return "skip"
    return f"{vs.score:.3f}"


def _print_score_table(scores: Dict[str, ValidationScore]) -> None:
    name_w = max(len(n) for n in scores) if scores else 9
    name_w = max(name_w, 9)
    score_w = 7
    fb_w = 50

    header = f"  {'Validator':<{name_w}} | {'Score':>{score_w}} | {'Feedback':<{fb_w}}"
    sep = f"  {'-' * name_w}-+-{'-' * score_w}-+-{'-' * fb_w}"
    print(sep)
    print(header)
    print(sep)
    for name, vs in scores.items():
        sc = _fmt_score(vs)
        fb = (vs.feedback or "")[:fb_w]
        print(f"  {name:<{name_w}} | {sc:>{score_w}} | {fb:<{fb_w}}")
    print(sep)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _validate_segment(
    *,
    generation: GenerationResult,
    validators: Sequence[Validator],
) -> Dict[str, ValidationScore]:
    scores: Dict[str, ValidationScore] = {}
    for v in validators:
        scores[v.name] = v.score(generation)
    return scores


def _build_generation_result_from_video(
    *,
    video_path: Path,
    frames_dir: Path,
    prompt: str,
    fps: int,
    backend_name: str = "veo",
) -> GenerationResult:
    """Extract frames and build a GenerationResult suitable for validators."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        extract_frames_ffmpeg(
            video_path=video_path,
            frames_dir=frames_dir,
            fps=fps,
        )
    except VideoEncodeError as e:
        raise RuntimeError(f"Failed to extract frames: {e}") from e

    frames = sorted(frames_dir.glob("frame_*.ppm"))
    if not frames:
        raise RuntimeError(f"No frames extracted from {video_path}")

    spec = AnimationSpec(prompt=prompt, fps=fps)
    return GenerationResult(
        spec=spec,
        frames=frames,
        frames_dir=frames_dir,
        backend=backend_name,
        metadata={"video_path": str(video_path)},
    )


def _estimate_segments(target_s: float, initial_s: float, ext_s: float) -> int:
    if target_s <= initial_s:
        return 1
    return 1 + math.ceil((target_s - initial_s) / ext_s)


def _save_report(report: PipelineReport, output_dir: Path) -> Path:
    """Persist the pipeline report as JSON."""
    from dataclasses import asdict

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "report.json"

    def _serialise(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        return str(obj)

    data = asdict(report)
    path.write_text(json.dumps(data, default=_serialise, indent=2))
    return path


# ---------------------------------------------------------------------------
# Interactive extension pipeline (human-in-the-loop)
# ---------------------------------------------------------------------------

class InteractiveExtensionPipeline:
    """
    Iteratively extend a Veo 3.1 video with human approval at each step.

    Veo extension returns the **full combined video** (original + extension)
    so there is no need for ffmpeg concatenation.  The API ``Video`` object
    from each generation is threaded through as ``veo_video_ref`` so that
    the next extension call can reference it server-side (raw bytes are not
    accepted by the extension endpoint).
    """

    def __init__(
        self,
        *,
        backend: VeoGenaiBackend,
        validators: Sequence[Validator],
        config: ExtensionPipelineConfig,
        initial_spec: AnimationSpec,
        run_root: Optional[Path] = None,
    ) -> None:
        self._backend = backend
        self._validators = list(validators)
        self._config = config
        self._spec = initial_spec
        self._run_root = run_root or Path("runs") / f"extend_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run(self) -> PipelineReport:
        self._run_root.mkdir(parents=True, exist_ok=True)
        est_initial_s = 8.0
        est_ext_s = 7.0
        est_total = _estimate_segments(self._config.target_duration_s, est_initial_s, est_ext_s)

        segments: List[SegmentResult] = []
        current_prompt = self._spec.prompt

        # ---- initial clip ----
        seg_idx = 0
        seg_dir = self._run_root / f"seg_{seg_idx:02d}"
        print(f"\n[Segment {seg_idx + 1}/~{est_total}] Generating initial clip...")
        generation = self._backend.generate(spec=self._spec, output_dir=seg_dir)

        video_path = Path(str(generation.metadata.get("video_path", seg_dir / "generated.mp4")))
        veo_ref = generation.metadata.get("veo_video_ref")
        dur = get_video_duration_seconds(video_path) or est_initial_s
        cumulative_s = dur
        print(f"  Generated: {video_path} ({dur:.1f}s)")

        scores = _validate_segment(generation=generation, validators=self._validators)
        _print_score_table(scores)

        accepted, new_hint = _ask_human(video_path)
        while not accepted:
            if new_hint is not None:
                current_prompt = new_hint
                self._spec = replace(self._spec, prompt=current_prompt)
            seg_dir = self._run_root / f"seg_{seg_idx:02d}_retry"
            print(f"  Regenerating initial clip...")
            generation = self._backend.generate(spec=self._spec, output_dir=seg_dir)
            video_path = Path(str(generation.metadata.get("video_path", seg_dir / "generated.mp4")))
            veo_ref = generation.metadata.get("veo_video_ref")
            dur = get_video_duration_seconds(video_path) or est_initial_s
            cumulative_s = dur
            print(f"  Generated: {video_path} ({dur:.1f}s)")
            scores = _validate_segment(generation=generation, validators=self._validators)
            _print_score_table(scores)
            accepted, new_hint = _ask_human(video_path)

        segments.append(SegmentResult(
            segment_index=seg_idx,
            video_path=video_path,
            duration_s=dur,
            prompt=current_prompt,
            validation_scores=scores,
            accepted=True,
        ))

        # ---- extension loop ----
        # Each extension returns the FULL combined video (original + extensions).
        # ``veo_ref`` is the API Video object needed for the next call.
        for ext_i in range(1, self._config.max_extensions + 1):
            if cumulative_s >= self._config.target_duration_s:
                print(f"\n  Target duration ({self._config.target_duration_s:.0f}s) reached at {cumulative_s:.1f}s.")
                break

            seg_idx = ext_i
            print(
                f"\n[Segment {seg_idx + 1}/~{est_total}] "
                f"Extending (+~{est_ext_s:.0f}s, cumulative ~{cumulative_s:.0f}s)..."
            )

            ext_prompt_result = generate_extension_prompt(
                original_prompt=current_prompt,
                model=self._config.gemini_model,
            )
            ext_prompt = ext_prompt_result.extension_prompt
            ext_neg = ext_prompt_result.negative_prompt
            print(f"  Extension prompt: {ext_prompt[:100]}{'...' if len(ext_prompt) > 100 else ''}")

            seg_dir = self._run_root / f"seg_{seg_idx:02d}"
            ext_result = self._backend.extend_video(
                source_video=veo_ref,
                prompt=ext_prompt,
                negative_prompt=ext_neg,
                output_dir=seg_dir,
                fps=self._config.fps,
            )
            ext_video = ext_result.video_path
            new_veo_ref = ext_result.metadata.get("veo_video_ref")
            ext_total_dur = get_video_duration_seconds(ext_video) or (cumulative_s + est_ext_s)
            print(f"  Generated: {ext_video} ({ext_total_dur:.1f}s total)")

            frames_dir = seg_dir / "val_frames"
            gen_result = _build_generation_result_from_video(
                video_path=ext_video,
                frames_dir=frames_dir,
                prompt=ext_prompt,
                fps=self._config.fps,
            )
            scores = _validate_segment(generation=gen_result, validators=self._validators)
            _print_score_table(scores)

            accepted, new_hint = _ask_human(ext_video)
            while not accepted:
                if new_hint is not None:
                    ext_prompt = new_hint
                retry_dir = self._run_root / f"seg_{seg_idx:02d}_retry"
                print(f"  Regenerating extension...")
                ext_result = self._backend.extend_video(
                    source_video=veo_ref,
                    prompt=ext_prompt,
                    negative_prompt=ext_neg,
                    output_dir=retry_dir,
                    fps=self._config.fps,
                )
                ext_video = ext_result.video_path
                new_veo_ref = ext_result.metadata.get("veo_video_ref")
                ext_total_dur = get_video_duration_seconds(ext_video) or (cumulative_s + est_ext_s)
                print(f"  Generated: {ext_video} ({ext_total_dur:.1f}s total)")
                frames_dir = retry_dir / "val_frames"
                gen_result = _build_generation_result_from_video(
                    video_path=ext_video,
                    frames_dir=frames_dir,
                    prompt=ext_prompt,
                    fps=self._config.fps,
                )
                scores = _validate_segment(generation=gen_result, validators=self._validators)
                _print_score_table(scores)
                accepted, new_hint = _ask_human(ext_video)

            veo_ref = new_veo_ref
            video_path = ext_video
            cumulative_s = ext_total_dur

            segments.append(SegmentResult(
                segment_index=seg_idx,
                video_path=ext_video,
                duration_s=ext_total_dur,
                prompt=ext_prompt,
                validation_scores=scores,
                accepted=True,
            ))

        # ---- final output ----
        # The last accepted video IS the full combined result; just copy it.
        final_path = self._run_root / "final.mp4"
        shutil.copyfile(video_path, final_path)
        final_dur = get_video_duration_seconds(final_path) or cumulative_s
        print(f"\n  Final video: {final_path} ({final_dur:.1f}s, {len(segments)} extensions)")

        agg = _aggregate_scores(segments)
        report = PipelineReport(
            segments=segments,
            mode="interactive",
            total_duration=final_dur,
            final_path=final_path,
            aggregate_scores=agg,
        )
        report_path = _save_report(report, self._run_root)
        print(f"  Report saved: {report_path}")
        return report


# ---------------------------------------------------------------------------
# One-shot pipeline (auto-extend, no human gates)
# ---------------------------------------------------------------------------

class OneshotPipeline:
    """Auto-extend to target duration without human gates (comparison baseline)."""

    def __init__(
        self,
        *,
        backend: VeoGenaiBackend,
        validators: Sequence[Validator],
        config: ExtensionPipelineConfig,
        initial_spec: AnimationSpec,
        run_root: Optional[Path] = None,
    ) -> None:
        self._backend = backend
        self._validators = list(validators)
        self._config = config
        self._spec = initial_spec
        self._run_root = run_root or Path("runs") / f"oneshot_{_dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def run(self) -> PipelineReport:
        self._run_root.mkdir(parents=True, exist_ok=True)
        est_initial_s = 8.0
        est_ext_s = 7.0
        est_total = _estimate_segments(self._config.target_duration_s, est_initial_s, est_ext_s)

        segments: List[SegmentResult] = []
        current_prompt = self._spec.prompt

        # ---- initial clip ----
        seg_dir = self._run_root / "seg_00"
        print(f"\n[Segment 1/~{est_total}] Generating initial clip... (auto-accept)")
        generation = self._backend.generate(spec=self._spec, output_dir=seg_dir)
        video_path = Path(str(generation.metadata.get("video_path", seg_dir / "generated.mp4")))
        veo_ref = generation.metadata.get("veo_video_ref")
        dur = get_video_duration_seconds(video_path) or est_initial_s
        cumulative_s = dur

        segments.append(SegmentResult(
            segment_index=0,
            video_path=video_path,
            duration_s=dur,
            prompt=current_prompt,
            accepted=True,
        ))
        print(f"  Generated: {video_path} ({dur:.1f}s)")

        # ---- extension loop ----
        for ext_i in range(1, self._config.max_extensions + 1):
            if cumulative_s >= self._config.target_duration_s:
                break

            print(
                f"[Segment {ext_i + 1}/~{est_total}] "
                f"Extending... (auto-accept, cumulative ~{cumulative_s:.0f}s)"
            )

            ext_prompt_result = generate_extension_prompt(
                original_prompt=current_prompt,
                model=self._config.gemini_model,
            )
            ext_prompt = ext_prompt_result.extension_prompt
            ext_neg = ext_prompt_result.negative_prompt

            seg_dir = self._run_root / f"seg_{ext_i:02d}"
            ext_result = self._backend.extend_video(
                source_video=veo_ref,
                prompt=ext_prompt,
                negative_prompt=ext_neg,
                output_dir=seg_dir,
                fps=self._config.fps,
            )
            ext_video = ext_result.video_path
            veo_ref = ext_result.metadata.get("veo_video_ref")
            ext_total_dur = get_video_duration_seconds(ext_video) or (cumulative_s + est_ext_s)
            video_path = ext_video
            cumulative_s = ext_total_dur
            print(f"  Generated: {ext_video} ({ext_total_dur:.1f}s total)")

            segments.append(SegmentResult(
                segment_index=ext_i,
                video_path=ext_video,
                duration_s=ext_total_dur,
                prompt=ext_prompt,
                accepted=True,
            ))

        # ---- final output ----
        final_path = self._run_root / "final.mp4"
        shutil.copyfile(video_path, final_path)
        final_dur = get_video_duration_seconds(final_path) or cumulative_s

        # ---- validate the full result ----
        print("\nFinal Validation (full video):")
        full_frames_dir = self._run_root / "final_frames"
        full_gen = _build_generation_result_from_video(
            video_path=final_path,
            frames_dir=full_frames_dir,
            prompt=current_prompt,
            fps=self._config.fps,
        )
        final_scores = _validate_segment(generation=full_gen, validators=self._validators)
        _print_score_table(final_scores)

        agg = _aggregate_scores(segments)
        print(f"\n  Final video: {final_path} ({final_dur:.1f}s, {len(segments)} extensions)")

        report = PipelineReport(
            segments=segments,
            mode="oneshot",
            total_duration=final_dur,
            final_path=final_path,
            aggregate_scores=agg,
        )
        report_path = _save_report(report, self._run_root)
        print(f"  Report saved: {report_path}")
        return report


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------

def _ask_human(video_path: Path) -> tuple[bool, Optional[str]]:
    """
    Prompt the human operator via CLI.

    Returns (accepted, optional_new_hint).
    If the user quits, raises SystemExit.
    """
    print(f"\n  >> Watch before deciding: {video_path}")
    while True:
        choice = input("  [a]ccept / [r]egenerate / [q]uit: ").strip().lower()
        if choice in ("a", "accept"):
            return True, None
        if choice in ("r", "regenerate"):
            hint = input("  Optional new instructions (Enter to keep same prompt): ").strip()
            return False, (hint if hint else None)
        if choice in ("q", "quit"):
            print("  Quitting pipeline.")
            raise SystemExit(0)
        print("  Invalid choice. Enter a, r, or q.")


def _aggregate_scores(segments: List[SegmentResult]) -> Dict[str, float]:
    """Average each validator score across accepted segments that have scores."""
    totals: Dict[str, List[float]] = {}
    for seg in segments:
        for name, vs in seg.validation_scores.items():
            if vs.skipped:
                continue
            totals.setdefault(name, []).append(vs.score)
    return {name: sum(vals) / len(vals) for name, vals in totals.items() if vals}
