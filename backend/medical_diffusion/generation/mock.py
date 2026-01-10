from __future__ import annotations

import math
import random
from pathlib import Path
from typing import List, Optional

from ..io.ppm import write_ppm
from ..types import AnimationSpec, GenerationResult
from .base import DiffusionBackend, ensure_empty_dir


class MockDiffusionBackend:
    """
    Generates a deterministic PPM frame sequence with a moving red dot.

    This keeps the rest of the system runnable without GPUs/APIs and gives the
    physics validator something trackable.
    """

    name = "mock"

    def generate(self, *, spec: AnimationSpec, output_dir: Path) -> GenerationResult:
        ensure_empty_dir(output_dir)
        frames_dir = output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        seed = spec.seed if spec.seed is not None else random.randint(0, 2**31 - 1)
        rng = random.Random(seed)

        num_frames = spec.num_frames
        frames: List[Path] = []
        for i in range(num_frames):
            t = 0.0 if num_frames <= 1 else i / (num_frames - 1)
            rgb = _render_frame(
                width=spec.width,
                height=spec.height,
                t=t,
                rng=rng,
            )
            frame_path = frames_dir / f"frame_{i:04d}.ppm"
            write_ppm(frame_path, spec.width, spec.height, rgb)
            frames.append(frame_path)

        return GenerationResult(
            spec=spec,
            frames=frames,
            frames_dir=frames_dir,
            backend=self.name,
            metadata={"seed": seed},
        )


def _render_frame(*, width: int, height: int, t: float, rng: random.Random) -> bytes:
    # Background: bluish gradient with mild noise.
    bg_r = 10
    bg_g = 20
    bg_b = 35

    # Red dot moves down with slight acceleration (gravity toy).
    cx = int(width * (0.2 + 0.6 * t))
    # y(t) = y0 + v0*t + 0.5*a*t^2
    y0 = height * 0.2
    v0 = height * 0.1
    a = height * 0.8
    cy = int(y0 + v0 * t + 0.5 * a * t * t)
    radius = max(4, int(min(width, height) * 0.06))

    out = bytearray(width * height * 3)
    for y in range(height):
        for x in range(width):
            i = (y * width + x) * 3
            # Subtle gradient.
            g = int(15 * (x / max(1, width - 1)) + 10 * (y / max(1, height - 1)))
            r = bg_r + g
            gg = bg_g + g
            b = bg_b + g

            # Mild noise to avoid totally flat frames.
            n = rng.randint(-2, 2)
            r = _clamp8(r + n)
            gg = _clamp8(gg + n)
            b = _clamp8(b + n)

            # Moving red dot.
            dx = x - cx
            dy = y - cy
            if dx * dx + dy * dy <= radius * radius:
                r = 220
                gg = 40
                b = 40

            out[i] = r
            out[i + 1] = gg
            out[i + 2] = b

    return bytes(out)


def _clamp8(v: int) -> int:
    if v < 0:
        return 0
    if v > 255:
        return 255
    return v
