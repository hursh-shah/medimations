from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw

from ..types import AnimationSpec, GenerationResult
from .base import ensure_empty_dir


@dataclass(frozen=True)
class DemoBackend:
    """
    Local, zero-dependency generator for testing the web/job flow without Veo.

    Generates a simple falling red dot (so the toy physics validator passes),
    and writes frames to `frame_%04d.ppm`.
    """

    name: str = "demo"

    def generate(self, *, spec: AnimationSpec, output_dir: Path) -> GenerationResult:
        ensure_empty_dir(output_dir)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        width = int(spec.width or 256)
        height = int(spec.height or 256)
        n_frames = max(1, int(spec.num_frames))

        margin = max(8, min(width, height) // 16)
        radius = max(4, min(width, height) // 20)

        x = width // 2
        y0 = margin + radius

        t_max = max(1, n_frames - 1)
        denom = t_max * (t_max + 1)
        max_y = max(y0 + 1, height - margin - radius)
        a = max(1, int(((max_y - y0) * 2) / max(1, denom)))

        tube_w = max(24, width // 3)
        left = x - tube_w // 2
        right = x + tube_w // 2

        for t in range(n_frames):
            img = Image.new("RGB", (width, height), (245, 245, 245))
            draw = ImageDraw.Draw(img)

            draw.rectangle(
                [left, margin, right, height - margin],
                outline=(180, 180, 180),
                width=max(2, width // 128),
            )

            y = y0 + (a * t * (t + 1)) // 2
            y = min(max_y, int(y))

            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                fill=(220, 0, 0),
                outline=(120, 0, 0),
            )

            img.save(frames_dir / f"frame_{t:04d}.ppm", format="PPM")

        frames = sorted(frames_dir.glob("frame_*.ppm"))
        if not frames:
            raise RuntimeError("DemoBackend produced no frames")

        return GenerationResult(
            spec=spec,
            frames=frames,
            frames_dir=frames_dir,
            backend=self.name,
            metadata={"demo": True},
        )

