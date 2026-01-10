from __future__ import annotations

import shutil
from pathlib import Path

from ..types import GenerationResult
from .video import encode_video_ffmpeg


def export_final_video(*, generation: GenerationResult, output_path: Path) -> None:
    """
    Prefer copying a source .mp4 (e.g. Veo) when available; otherwise encode from frames.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    maybe = generation.metadata.get("video_path") if isinstance(generation.metadata, dict) else None
    if maybe:
        src = Path(str(maybe))
        if src.exists() and src.is_file():
            shutil.copyfile(src, output_path)
            return

    encode_video_ffmpeg(
        frames_dir=generation.frames_dir,
        output_path=output_path,
        fps=generation.spec.fps,
    )

