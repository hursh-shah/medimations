from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from ..io.video import VideoEncodeError, extract_frames_ffmpeg
from ..types import AnimationSpec, GenerationResult
from .base import ensure_empty_dir


class VideoEditMode(str, Enum):
    """Mode for video editing operations."""
    INSERT = "insert"  # Add an object to the video
    REMOVE = "remove"  # Remove an object from the video


@dataclass
class ExtendVideoResult:
    """Result of a video extension operation."""
    video_path: Path
    frames: List[Path]
    frames_dir: Path
    source_video_path: Path
    prompt: str
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoEditResult:
    """Result of a video edit operation (insert/remove)."""
    video_path: Path
    frames: List[Path]
    frames_dir: Path
    source_video_path: Path
    mask_path: Path
    edit_mode: VideoEditMode
    prompt: Optional[str]  # For INSERT mode
    model: str
    metadata: Dict[str, Any] = field(default_factory=dict)


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

    def extend_video(
        self,
        *,
        source_video_path: Path,
        prompt: str,
        negative_prompt: Optional[str] = None,
        output_dir: Path,
        fps: int = 8,
    ) -> ExtendVideoResult:
        """
        Extend an existing video using Veo's video-to-video continuation.
        
        This uses the source video as a reference and generates a continuation
        based on the provided prompt.
        
        Args:
            source_video_path: Path to the source video file to extend
            prompt: The prompt describing what should happen in the continuation
            negative_prompt: Optional negative prompt for things to avoid
            output_dir: Directory to save the extended video and frames
            fps: Frames per second for frame extraction
            
        Returns:
            ExtendVideoResult with paths to the extended video and frames
        """
        ensure_empty_dir(output_dir)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        extended_video_path = output_dir / "extended.mp4"

        try:
            from google import genai
            from google.genai import types
        except Exception as e:
            raise RuntimeError(
                "google-genai is required for video extension. Install with: pip install google-genai"
            ) from e

        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (required for video extension)")
        
        client = genai.Client(api_key=api_key)

        # Validate source video exists
        source_video_path = Path(source_video_path)
        if not source_video_path.exists():
            raise RuntimeError(f"Source video not found: {source_video_path}")

        # Read video bytes and create Video object
        video_bytes = source_video_path.read_bytes()
        video = types.Video(
            video_bytes=video_bytes,
            mime_type="video/mp4",
        )

        # Generate extended video using source video as reference
        operation = client.models.generate_videos(
            model=self.model,
            prompt=prompt,
            video=video,
            config=types.GenerateVideosConfig(
                negative_prompt=negative_prompt or None,
                aspect_ratio=self.aspect_ratio,
                resolution=self.resolution,
            ),
        )

        # Poll until done
        while not operation.done:
            time.sleep(max(1, int(self.poll_seconds)))
            operation = client.operations.get(operation)

        if not getattr(operation, "response", None) or not getattr(operation.response, "generated_videos", None):
            raise RuntimeError("Veo returned no videos for extension")

        # Download and save the extended video
        generated_video = operation.response.generated_videos[0]
        client.files.download(file=generated_video.video)
        generated_video.video.save(str(extended_video_path))

        # Extract frames from the extended video
        try:
            extract_frames_ffmpeg(
                video_path=extended_video_path,
                frames_dir=frames_dir,
                fps=fps,
                width=None,
                height=None,
            )
        except VideoEncodeError as e:
            raise RuntimeError(f"Failed to extract frames from extended video: {e}") from e

        frames = sorted(frames_dir.glob("frame_*.ppm"))
        if not frames:
            raise RuntimeError("No frames extracted from extended video")

        return ExtendVideoResult(
            video_path=extended_video_path,
            frames=frames,
            frames_dir=frames_dir,
            source_video_path=source_video_path,
            prompt=prompt,
            model=self.model,
            metadata={
                "source_video_path": str(source_video_path),
                "extended_video_path": str(extended_video_path),
                "negative_prompt": negative_prompt,
            },
        )

    def edit_video(
        self,
        *,
        source_video_path: Path,
        mask_path: Path,
        mode: VideoEditMode,
        prompt: Optional[str] = None,
        output_dir: Path,
        fps: int = 8,
        edit_model: str = "veo-2.0-generate-preview",
    ) -> VideoEditResult:
        """
        Edit an existing video using Veo's mask-based editing (insert/remove).
        
        This uses a mask image to specify which region of the video to edit.
        - INSERT mode: Adds an object described by prompt to the masked region
        - REMOVE mode: Removes objects from the masked region (no prompt needed)
        
        Args:
            source_video_path: Path to the source video file to edit
            mask_path: Path to the mask image (PNG, white = edit region)
            mode: VideoEditMode.INSERT or VideoEditMode.REMOVE
            prompt: Description of object to insert (required for INSERT, ignored for REMOVE)
            output_dir: Directory to save the edited video and frames
            fps: Frames per second for frame extraction
            edit_model: Model to use for editing (veo-2.0 supports inpainting)
            
        Returns:
            VideoEditResult with paths to the edited video and frames
        """
        ensure_empty_dir(output_dir)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        edited_video_path = output_dir / "edited.mp4"

        try:
            from google import genai
            from google.genai import types
        except Exception as e:
            raise RuntimeError(
                "google-genai is required for video editing. Install with: pip install google-genai"
            ) from e

        api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is not set (required for video editing)")
        
        client = genai.Client(api_key=api_key)

        # Validate inputs
        source_video_path = Path(source_video_path)
        if not source_video_path.exists():
            raise RuntimeError(f"Source video not found: {source_video_path}")
        
        mask_path = Path(mask_path)
        if not mask_path.exists():
            raise RuntimeError(f"Mask image not found: {mask_path}")
        
        if mode == VideoEditMode.INSERT and not prompt:
            raise RuntimeError("prompt is required for INSERT mode")

        # Read video bytes
        video_bytes = source_video_path.read_bytes()
        video = types.Video(
            video_bytes=video_bytes,
            mime_type="video/mp4",
        )

        # Read mask image
        mask_bytes = mask_path.read_bytes()
        mask_mime = "image/png"
        if mask_path.suffix.lower() in {".jpg", ".jpeg"}:
            mask_mime = "image/jpeg"
        
        mask_image = types.Image(
            image_bytes=mask_bytes,
            mime_type=mask_mime,
        )

        # Determine mask mode
        if mode == VideoEditMode.INSERT:
            mask_mode = types.VideoGenerationMaskMode.INSERT
        else:
            mask_mode = types.VideoGenerationMaskMode.REMOVE

        # Build the source with video and optional prompt
        source = types.GenerateVideosSource(
            video=video,
            prompt=prompt if mode == VideoEditMode.INSERT else None,
        )

        # Generate edited video
        operation = client.models.generate_videos(
            model=edit_model,
            source=source,
            config=types.GenerateVideosConfig(
                mask=types.VideoGenerationMask(
                    image=mask_image,
                    mask_mode=mask_mode,
                ),
            ),
        )

        # Poll until done
        while not operation.done:
            time.sleep(max(1, int(self.poll_seconds)))
            operation = client.operations.get(operation)

        if not getattr(operation, "response", None) or not getattr(operation.response, "generated_videos", None):
            raise RuntimeError(f"Veo returned no videos for {mode.value} edit")

        # Download and save the edited video
        generated_video = operation.response.generated_videos[0]
        client.files.download(file=generated_video.video)
        generated_video.video.save(str(edited_video_path))

        # Extract frames from the edited video
        try:
            extract_frames_ffmpeg(
                video_path=edited_video_path,
                frames_dir=frames_dir,
                fps=fps,
                width=None,
                height=None,
            )
        except VideoEncodeError as e:
            raise RuntimeError(f"Failed to extract frames from edited video: {e}") from e

        frames = sorted(frames_dir.glob("frame_*.ppm"))
        if not frames:
            raise RuntimeError("No frames extracted from edited video")

        return VideoEditResult(
            video_path=edited_video_path,
            frames=frames,
            frames_dir=frames_dir,
            source_video_path=source_video_path,
            mask_path=mask_path,
            edit_mode=mode,
            prompt=prompt,
            model=edit_model,
            metadata={
                "source_video_path": str(source_video_path),
                "edited_video_path": str(edited_video_path),
                "mask_path": str(mask_path),
                "edit_mode": mode.value,
            },
        )

    def insert_object(
        self,
        *,
        source_video_path: Path,
        mask_path: Path,
        object_prompt: str,
        output_dir: Path,
        fps: int = 8,
    ) -> VideoEditResult:
        """
        Insert an object into a video at the masked region.
        
        Convenience method that wraps edit_video with INSERT mode.
        
        Args:
            source_video_path: Path to the source video
            mask_path: Path to mask image (white = where to insert)
            object_prompt: Description of the object to insert
            output_dir: Directory for output files
            fps: Frames per second for extraction
            
        Returns:
            VideoEditResult with the edited video
        """
        return self.edit_video(
            source_video_path=source_video_path,
            mask_path=mask_path,
            mode=VideoEditMode.INSERT,
            prompt=object_prompt,
            output_dir=output_dir,
            fps=fps,
        )

    def remove_object(
        self,
        *,
        source_video_path: Path,
        mask_path: Path,
        output_dir: Path,
        fps: int = 8,
    ) -> VideoEditResult:
        """
        Remove objects from a video at the masked region.
        
        Convenience method that wraps edit_video with REMOVE mode.
        
        Args:
            source_video_path: Path to the source video
            mask_path: Path to mask image (white = what to remove)
            output_dir: Directory for output files
            fps: Frames per second for extraction
            
        Returns:
            VideoEditResult with the edited video
        """
        return self.edit_video(
            source_video_path=source_video_path,
            mask_path=mask_path,
            mode=VideoEditMode.REMOVE,
            prompt=None,
            output_dir=output_dir,
            fps=fps,
        )
