from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..io.video import VideoEncodeError, extract_frames_ffmpeg
from ..types import AnimationSpec, GenerationResult
from .base import ensure_empty_dir


@dataclass
class VeoGenaiBackend:
    """
    Veo 3.1 video generation backend using the `google-genai` Python SDK.

    Install:
      pip install google-genai

    Auth:
      export GOOGLE_API_KEY=...
    """

    model: str = "veo-3.1-generate-preview"
    aspect_ratio: str = "9:16"
    resolution: str = "720p"
    poll_seconds: int = 20

    name: str = "veo"

    def generate(self, *, spec: AnimationSpec, output_dir: Path) -> GenerationResult:
        ensure_empty_dir(output_dir)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        video_path = output_dir / "generated.mp4"

        try:
            from google import genai
            from google.genai import types
        except Exception as e:
            raise RuntimeError(
                "google-genai is required for --backend veo. Install with: pip install google-genai"
            ) from e

        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (required for --backend veo)")
        client = genai.Client(api_key=api_key)

        image = None
        if spec.input_image_path is not None:
            image_path = Path(spec.input_image_path)
            if not image_path.exists():
                raise RuntimeError(f"Input image not found: {image_path}")
            if hasattr(types.Image, "from_file"):
                image = types.Image.from_file(location=str(image_path))
            else:
                # Fallback for older google-genai versions.
                mime_type = "image/png"
                ext = image_path.suffix.lower().lstrip(".")
                if ext in {"jpg", "jpeg"}:
                    mime_type = "image/jpeg"
                elif ext == "webp":
                    mime_type = "image/webp"
                image = types.Image(image_bytes=image_path.read_bytes(), mime_type=mime_type)

        operation = client.models.generate_videos(
            model=self.model,
            prompt=spec.prompt,
            image=image,
            config=types.GenerateVideosConfig(
                negative_prompt=spec.negative_prompt or None,
                aspect_ratio=self.aspect_ratio,
                resolution=self.resolution,
            ),
        )

        while not operation.done:
            time.sleep(max(1, int(self.poll_seconds)))
            operation = client.operations.get(operation)

        if not getattr(operation, "response", None) or not getattr(operation.response, "generated_videos", None):
            raise RuntimeError("Veo returned no videos")

        generated_video = operation.response.generated_videos[0]
        client.files.download(file=generated_video.video)
        generated_video.video.save(str(video_path))

        try:
            extract_frames_ffmpeg(
                video_path=video_path,
                frames_dir=frames_dir,
                fps=spec.fps,
                width=spec.width if spec.metadata.get("user_set_size") else None,
                height=spec.height if spec.metadata.get("user_set_size") else None,
            )
        except VideoEncodeError as e:
            raise RuntimeError(f"Failed to extract frames with ffmpeg: {e}") from e

        frames = sorted(frames_dir.glob("frame_*.ppm"))
        if not frames:
            raise RuntimeError("No frames extracted from generated video")

        return GenerationResult(
            spec=spec,
            frames=frames,
            frames_dir=frames_dir,
            backend=self.name,
            metadata={
                "video_path": str(video_path),
                "model": self.model,
                "input_image_path": str(spec.input_image_path) if spec.input_image_path is not None else None,
            },
        )
