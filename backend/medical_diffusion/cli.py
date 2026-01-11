from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Optional

from .agent import AgentConfig, ValidatorAgent
from .agent import GeminiPromptAdjuster
from .io.export import export_final_video
from .io.video import VideoEncodeError
from .prompt_processor import PromptProcessor
from .validation.biomedclip import BiomedCLIPMedicalValidator
from .validation.medical import FrameSanityMedicalValidator
from .validation.physics import PyBulletPhysicsValidator, RedDotGravityValidator


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="medical_diffusion")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_p = sub.add_parser("run", help="Generate + validate with agent loop")
    run_p.add_argument("--prompt", required=True, help="User prompt for the animation")
    run_p.add_argument("--out", default="runs/output.mp4", help="Output video path")
    run_p.add_argument("--no-video", action="store_true", help="Skip ffmpeg encode; leave frames on disk")
    run_p.add_argument("--negative-prompt", default=None, help="Optional negative prompt (backends that support it)")
    run_p.add_argument("--veo-model", default="veo-3.1-generate-preview", help="Veo model id")
    run_p.add_argument("--veo-aspect-ratio", default="9:16", help="Veo aspect ratio, e.g. 9:16")
    run_p.add_argument("--veo-resolution", default="720p", help="Veo resolution, e.g. 720p")
    run_p.add_argument("--veo-poll-seconds", type=int, default=20, help="Polling interval for Veo operations")
    run_p.add_argument("--prompt-rewrite", default="gemini", choices=["none", "rule", "gemini"], help="Prompt rewrite mode")
    run_p.add_argument("--gemini-model", default="gemini-2.0-flash", help="Gemini model for prompt rewriting/reprompting")
    run_p.add_argument("--biomedclip", action="store_true", help="Enable BiomedCLIP frame verifier (requires torch + open_clip_torch)")
    run_p.add_argument("--biomedclip-target", default=None, help="Expected target label (e.g. 'liver'); inferred from prompt if omitted")
    run_p.add_argument("--biomedclip-labels", default=None, help="Comma-separated labels (or path to a newline-delimited .txt)")
    run_p.add_argument("--biomedclip-frames", type=int, default=12, help="Frames sampled for BiomedCLIP scoring")
    run_p.add_argument("--max-rounds", type=int, default=2)
    run_p.add_argument("--candidates", type=int, default=1)
    run_p.add_argument("--medical-threshold", type=float, default=0.85)
    run_p.add_argument("--physics-threshold", type=float, default=0.85)
    run_p.add_argument("--fps", type=int, default=None)
    run_p.add_argument("--duration", type=float, default=None)
    run_p.add_argument("--width", type=int, default=None)
    run_p.add_argument("--height", type=int, default=None)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        return _run(args)

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
    if args.prompt_rewrite == "none":
        pass
    elif args.prompt_rewrite == "rule":
        spec = processor.rewrite_for_veo(spec)
    elif args.prompt_rewrite == "gemini":
        spec = processor.rewrite_for_veo_gemini(spec, model=args.gemini_model)
        prompt_adjuster = GeminiPromptAdjuster(model=args.gemini_model)
    else:
        raise ValueError(f"Unsupported --prompt-rewrite: {args.prompt_rewrite}")

    from .generation.veo_genai import VeoGenaiBackend

    backend = VeoGenaiBackend(
        model=args.veo_model,
        aspect_ratio=args.veo_aspect_ratio,
        resolution=args.veo_resolution,
        poll_seconds=args.veo_poll_seconds,
    )

    medical_validators = [FrameSanityMedicalValidator()]
    if args.biomedclip:
        medical_validators.append(
            BiomedCLIPMedicalValidator(
                target_label=args.biomedclip_target,
                labels=_parse_label_list(args.biomedclip_labels) if args.biomedclip_labels else None,
                n_frames=int(args.biomedclip_frames or 12),
            )
        )

    physics_validators = [RedDotGravityValidator(), PyBulletPhysicsValidator()]

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
