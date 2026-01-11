from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Optional


class VideoEncodeError(RuntimeError):
    pass


def encode_video_ffmpeg(
    *,
    frames_dir: Path,
    output_path: Path,
    fps: int,
    pattern: str = "frame_%04d.ppm",
    crf: int = 18,
    preset: str = "veryfast",
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise VideoEncodeError("ffmpeg not found on PATH (install ffmpeg or run with --no-video)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    input_pattern = str(frames_dir / pattern)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        input_pattern,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        str(crf),
        "-preset",
        preset,
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoEncodeError(proc.stderr.strip() or f"ffmpeg failed with exit code {proc.returncode}")


def mux_audio_ffmpeg(
    *,
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise VideoEncodeError("ffmpeg not found on PATH (required to mux audio)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-i",
        str(audio_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-c:v",
        "copy",
        "-c:a",
        str(audio_codec),
        "-b:a",
        str(audio_bitrate),
        "-shortest",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoEncodeError(proc.stderr.strip() or f"ffmpeg mux failed with exit code {proc.returncode}")


def extract_frames_ffmpeg(
    *,
    video_path: Path,
    frames_dir: Path,
    fps: int,
    pattern: str = "frame_%04d.ppm",
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise VideoEncodeError("ffmpeg not found on PATH (required to extract frames)")

    frames_dir.mkdir(parents=True, exist_ok=True)
    out_pattern = str(frames_dir / pattern)

    vf = f"fps={fps}"
    if width is not None and height is not None:
        vf += f",scale={width}:{height}"

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-vf",
        vf,
        "-start_number",
        "0",
        out_pattern,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise VideoEncodeError(proc.stderr.strip() or f"ffmpeg extract failed with exit code {proc.returncode}")
