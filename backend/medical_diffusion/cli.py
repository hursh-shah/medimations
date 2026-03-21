from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

from .agent import AgentConfig, ValidatorAgent
from .agent import GeminiPromptAdjuster
from .io.export import export_final_video
from .io.video import VideoEncodeError
from .prompt_processor import PromptProcessor
from .validation.biomedclip import BiomedCLIPMedicalValidator
from .validation.medical import FrameSanityCinematicValidator
from .validation.physics import PyBulletPhysicsValidator


# ---------------------------------------------------------------------------
# Shared argument helpers
# ---------------------------------------------------------------------------

def _add_veo_args(parser: argparse.ArgumentParser, *, extension: bool = False) -> None:
    parser.add_argument("--veo-model", default="veo-3.1-generate-preview", help="Veo model id")
    if extension:
        parser.add_argument("--veo-aspect-ratio", default="16:9", help="Veo aspect ratio (extension requires 16:9)")
        parser.add_argument("--veo-resolution", default="720p", help="Veo resolution (extension requires 720p)")
    else:
        parser.add_argument("--veo-aspect-ratio", default="9:16", help="Veo aspect ratio, e.g. 9:16")
        parser.add_argument("--veo-resolution", default="720p", help="Veo resolution, e.g. 720p")
    parser.add_argument("--veo-poll-seconds", type=int, default=20, help="Polling interval for Veo operations")


def _add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--prompt", required=True, help="User prompt for the animation")
    parser.add_argument("--negative-prompt", default=None, help="Optional negative prompt")
    parser.add_argument("--input-image", default=None, help="Optional reference image path")
    parser.add_argument("--reference-dir", default=None, help="Directory of reference images (up to 3 PNG/JPG/WebP) for subject-consistent generation")
    parser.add_argument("--prompt-rewrite", default="gemini", choices=["none", "rule", "gemini"], help="Prompt rewrite mode")
    parser.add_argument("--gemini-model", default="gemini-3-flash-preview", help="Gemini model for prompt rewriting")


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def _resolve_reference_images(raw_dir: Optional[str]) -> Optional[List[Path]]:
    """Load and validate reference images from a directory path.

    Returns None if raw_dir is None, or a list of 1-3 image Paths.
    Prints errors/warnings to stderr and raises SystemExit on fatal issues.
    """
    if raw_dir is None:
        return None
    ref_dir = Path(raw_dir)
    if not ref_dir.is_dir():
        print(f"Error: --reference-dir '{ref_dir}' is not a directory", file=sys.stderr)
        raise SystemExit(1)
    ref_images = sorted(
        p for p in ref_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not ref_images:
        print(f"Error: no images (PNG/JPG/WebP) found in --reference-dir '{ref_dir}'", file=sys.stderr)
        raise SystemExit(1)
    if len(ref_images) > 3:
        print(f"  Warning: Veo supports max 3 reference images, using first 3 of {len(ref_images)}", file=sys.stderr)
        ref_images = ref_images[:3]
    names = ", ".join(p.name for p in ref_images)
    print(f"  Reference images ({len(ref_images)}): {names}")
    return ref_images


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="medical_diffusion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # ---- run (existing) ----
    run_p = sub.add_parser("run", help="Generate + validate with agent loop")
    run_p.add_argument("--prompt", required=True, help="User prompt for the animation")
    run_p.add_argument("--out", default="runs/output.mp4", help="Output video path")
    run_p.add_argument("--no-video", action="store_true", help="Skip ffmpeg encode; leave frames on disk")
    run_p.add_argument("--negative-prompt", default=None, help="Optional negative prompt (backends that support it)")
    run_p.add_argument("--input-image", default=None, help="Optional reference image path for image+text → video")
    run_p.add_argument("--reference-dir", default=None, help="Directory of reference images (up to 3 PNG/JPG/WebP) for subject-consistent generation")
    run_p.add_argument("--veo-model", default="veo-3.1-generate-preview", help="Veo model id")
    run_p.add_argument("--veo-aspect-ratio", default="9:16", help="Veo aspect ratio, e.g. 9:16")
    run_p.add_argument("--veo-resolution", default="720p", help="Veo resolution, e.g. 720p")
    run_p.add_argument("--veo-poll-seconds", type=int, default=20, help="Polling interval for Veo operations")
    run_p.add_argument("--prompt-rewrite", default="gemini", choices=["none", "rule", "gemini"], help="Prompt rewrite mode")
    run_p.add_argument(
        "--gemini-model",
        default="gemini-3-flash-preview",
        help="Gemini model for prompt rewriting/reprompting",
    )
    run_p.add_argument("--biomedclip", action="store_true", help="(Deprecated) BiomedCLIP is no longer run on video frames by default")
    run_p.add_argument("--biomedclip-target", default=None, help="(Deprecated) Use image generation validation instead")
    run_p.add_argument("--biomedclip-labels", default=None, help="(Deprecated) Use image generation validation instead")
    run_p.add_argument("--biomedclip-frames", type=int, default=12, help="(Deprecated) Use image generation validation instead")
    run_p.add_argument("--max-rounds", type=int, default=2)
    run_p.add_argument("--candidates", type=int, default=1)
    run_p.add_argument("--medical-threshold", type=float, default=0.85)
    run_p.add_argument("--physics-threshold", type=float, default=0.85)
    run_p.add_argument("--fps", type=int, default=None)
    run_p.add_argument("--duration", type=float, default=None)
    run_p.add_argument("--width", type=int, default=None)
    run_p.add_argument("--height", type=int, default=None)

    # ---- extend-loop (new) ----
    ext_p = sub.add_parser(
        "extend-loop",
        help="Iteratively extend a video with human validation at each step",
    )
    _add_prompt_args(ext_p)
    _add_veo_args(ext_p, extension=True)
    ext_p.add_argument("--out", default="runs/extend_output.mp4", help="Output video path")
    ext_p.add_argument("--target-duration", type=float, default=60.0, help="Target total duration in seconds")
    ext_p.add_argument("--max-extensions", type=int, default=20, help="Max number of Veo extensions (each ~7s)")
    ext_p.add_argument("--fps", type=int, default=8, help="FPS for frame extraction / validation")
    ext_p.add_argument("--biomedclip-target", default=None, help="BiomedCLIP target label (inferred from prompt if omitted)")
    ext_p.add_argument("--biomedclip-labels", default=None, help="Comma-separated labels or path to label file")
    ext_p.add_argument("--biomedclip-frames", type=int, default=12, help="Frames to sample per segment for BiomedCLIP")

    # ---- oneshot (new) ----
    one_p = sub.add_parser(
        "oneshot",
        help="Auto-extend to target duration without human gates (comparison baseline)",
    )
    _add_prompt_args(one_p)
    _add_veo_args(one_p, extension=True)
    one_p.add_argument("--out", default="runs/oneshot_output.mp4", help="Output video path")
    one_p.add_argument("--target-duration", type=float, default=60.0, help="Target total duration in seconds")
    one_p.add_argument("--max-extensions", type=int, default=20, help="Max number of Veo extensions (each ~7s)")
    one_p.add_argument("--fps", type=int, default=8, help="FPS for frame extraction / validation")
    one_p.add_argument("--biomedclip-target", default=None, help="BiomedCLIP target label (inferred from prompt if omitted)")
    one_p.add_argument("--biomedclip-labels", default=None, help="Comma-separated labels or path to label file")
    one_p.add_argument("--biomedclip-frames", type=int, default=12, help="Frames to sample per segment for BiomedCLIP")

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _run(args)
    if args.cmd == "extend-loop":
        return _extend_loop(args)
    if args.cmd == "oneshot":
        return _oneshot(args)

    parser.error(f"Unknown command: {args.cmd}")
    return 2


def _run(args: argparse.Namespace) -> int:
    processor = PromptProcessor()
    spec = processor.parse(args.prompt)
    if args.fps is not None:
        spec = replace(spec, fps=args.fps)
    if args.duration is not None:
        spec = replace(spec, duration_s=args.duration)
    if (args.width is None) != (args.height is None):
        print("Error: --width and --height must be provided together")
        return 2
    if args.width is not None and args.height is not None:
        spec = replace(spec, width=args.width, height=args.height)
    if args.negative_prompt is not None:
        spec = replace(spec, negative_prompt=args.negative_prompt)
    if args.input_image:
        spec = replace(spec, input_image_path=Path(args.input_image))
    ref_images = _resolve_reference_images(args.reference_dir)
    if ref_images:
        spec = replace(spec, reference_image_paths=ref_images)

    metadata = dict(spec.metadata or {})
    if args.fps is not None:
        metadata["user_set_fps"] = True
    if args.duration is not None:
        metadata["user_set_duration"] = True
    if args.width is not None and args.height is not None:
        metadata["user_set_size"] = True
    if metadata:
        spec = replace(spec, metadata=metadata)

    prompt_adjuster = None
    original_prompt = spec.prompt
    if args.prompt_rewrite == "none":
        pass
    elif args.prompt_rewrite == "rule":
        spec = processor.rewrite_for_veo(spec)
    elif args.prompt_rewrite == "gemini":
        spec = processor.rewrite_for_veo_gemini(spec, model=args.gemini_model)
        prompt_adjuster = GeminiPromptAdjuster(model=args.gemini_model)
    else:
        raise ValueError(f"Unsupported --prompt-rewrite: {args.prompt_rewrite}")

    if spec.prompt != original_prompt:
        print(f"\n  Rewritten prompt ({args.prompt_rewrite}):\n  {spec.prompt}")
        if spec.negative_prompt:
            print(f"  Negative prompt: {spec.negative_prompt}")
        print()

    from .generation.veo_genai import VeoGenaiBackend

    backend = VeoGenaiBackend(
        model=args.veo_model,
        aspect_ratio=args.veo_aspect_ratio,
        resolution=args.veo_resolution,
        poll_seconds=args.veo_poll_seconds,
    )

    medical_validators = [FrameSanityCinematicValidator()]
    if args.biomedclip or args.biomedclip_target or args.biomedclip_labels:
        print("Warning: BiomedCLIP validation is no longer run on video frames (use /api/images/generate instead).")

    physics_validators = [PyBulletPhysicsValidator()]

    agent = ValidatorAgent(
        generator=backend,
        medical_validators=medical_validators,
        physics_validators=physics_validators,
        config=AgentConfig(
            max_rounds=min(2, int(args.max_rounds)),
            candidates_per_round=args.candidates,
            medical_threshold=args.medical_threshold,
            physics_threshold=args.physics_threshold,
        ),
        prompt_adjuster=prompt_adjuster,
    )

    result = agent.run(spec=spec)
    out_path = _normalize_out_path(args.out, no_video=bool(args.no_video))

    if args.no_video:
        print(f"Done. Final frames: {result.final.generation.frames_dir}")
        print(f"Accepted: {result.accepted} ({result.final.report.summary()})")
        return 0

    try:
        export_final_video(generation=result.final.generation, output_path=out_path)
    except VideoEncodeError as e:
        print(f"ffmpeg export failed: {e}")
        print(f"Frames are still available at: {result.final.generation.frames_dir}")
        return 2
    except Exception as e:
        print(f"Export failed: {e}")
        print(f"Frames are still available at: {result.final.generation.frames_dir}")
        return 2

    print(f"Wrote: {out_path}")
    print(f"Accepted: {result.accepted} ({result.final.report.summary()})")
    return 0


def _parse_label_list(raw: str) -> list[str]:
    path = Path(raw)
    if path.exists() and path.is_file():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [part.strip() for part in raw.split(",") if part.strip()]


def _normalize_out_path(raw: str, *, no_video: bool) -> Path:
    out_arg = (raw or "").strip()
    out_path = Path(out_arg)
    if no_video:
        return out_path

    # If the user passes a directory (common when line-breaking commands),
    # choose a default file name.
    if out_arg.endswith("/") or (out_path.exists() and out_path.is_dir()):
        return out_path / "output.mp4"

    if out_path.suffix == "":
        raise ValueError("`--out` must include a filename + extension (e.g. runs/demo.mp4) or end with `/` for a directory")

    return out_path


# ---------------------------------------------------------------------------
# Shared builder for extension pipeline validators
# ---------------------------------------------------------------------------

def _build_extension_validators(args: argparse.Namespace) -> list:
    """Build the full validator list used by extend-loop and oneshot."""
    validators: list = [FrameSanityCinematicValidator()]

    biomedclip_kwargs: dict = {}
    if args.biomedclip_target:
        biomedclip_kwargs["target_label"] = args.biomedclip_target
    if args.biomedclip_labels:
        biomedclip_kwargs["labels"] = _parse_label_list(args.biomedclip_labels)
    if args.biomedclip_frames:
        biomedclip_kwargs["n_frames"] = args.biomedclip_frames
    validators.append(BiomedCLIPMedicalValidator(**biomedclip_kwargs))

    validators.append(PyBulletPhysicsValidator())
    return validators


def _build_spec_for_extension(args: argparse.Namespace) -> "AnimationSpec":
    """Parse prompt and apply rewriting for extend-loop / oneshot."""
    from .types import AnimationSpec

    processor = PromptProcessor()
    spec = processor.parse(args.prompt)
    if args.fps is not None:
        spec = replace(spec, fps=args.fps)
    if args.negative_prompt is not None:
        spec = replace(spec, negative_prompt=args.negative_prompt)
    if args.input_image:
        spec = replace(spec, input_image_path=Path(args.input_image))
    ref_images = _resolve_reference_images(args.reference_dir)
    if ref_images:
        spec = replace(spec, reference_image_paths=ref_images)

    original_prompt = spec.prompt
    if args.prompt_rewrite == "rule":
        spec = processor.rewrite_for_veo(spec)
    elif args.prompt_rewrite == "gemini":
        spec = processor.rewrite_for_veo_gemini(spec, model=args.gemini_model)

    if spec.prompt != original_prompt:
        print(f"\n  Rewritten prompt ({args.prompt_rewrite}):\n  {spec.prompt}")
        if spec.negative_prompt:
            print(f"  Negative prompt: {spec.negative_prompt}")
        print()

    return spec


# ---------------------------------------------------------------------------
# extend-loop command
# ---------------------------------------------------------------------------

def _extend_loop(args: argparse.Namespace) -> int:
    from .generation.veo_genai import VeoGenaiBackend
    from .pipeline import ExtensionPipelineConfig, InteractiveExtensionPipeline

    spec = _build_spec_for_extension(args)

    backend = VeoGenaiBackend(
        model=args.veo_model,
        aspect_ratio=args.veo_aspect_ratio,
        resolution=args.veo_resolution,
        poll_seconds=args.veo_poll_seconds,
    )
    validators = _build_extension_validators(args)
    config = ExtensionPipelineConfig(
        target_duration_s=args.target_duration,
        max_extensions=args.max_extensions,
        fps=args.fps,
        gemini_model=args.gemini_model,
    )

    pipeline = InteractiveExtensionPipeline(
        backend=backend,
        validators=validators,
        config=config,
        initial_spec=spec,
    )

    try:
        report = pipeline.run()
    except SystemExit:
        return 0

    out_path = _normalize_out_path(args.out, no_video=False)
    if report.final_path and report.final_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report.final_path, out_path)
        print(f"\nWrote: {out_path}")
    return 0


# ---------------------------------------------------------------------------
# oneshot command
# ---------------------------------------------------------------------------

def _oneshot(args: argparse.Namespace) -> int:
    from .generation.veo_genai import VeoGenaiBackend
    from .pipeline import ExtensionPipelineConfig, OneshotPipeline

    spec = _build_spec_for_extension(args)

    backend = VeoGenaiBackend(
        model=args.veo_model,
        aspect_ratio=args.veo_aspect_ratio,
        resolution=args.veo_resolution,
        poll_seconds=args.veo_poll_seconds,
    )
    validators = _build_extension_validators(args)
    config = ExtensionPipelineConfig(
        target_duration_s=args.target_duration,
        max_extensions=args.max_extensions,
        fps=args.fps,
        gemini_model=args.gemini_model,
    )

    pipeline = OneshotPipeline(
        backend=backend,
        validators=validators,
        config=config,
        initial_spec=spec,
    )

    report = pipeline.run()

    out_path = _normalize_out_path(args.out, no_video=False)
    if report.final_path and report.final_path.exists():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(report.final_path, out_path)
        print(f"\nWrote: {out_path}")
    return 0
