from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..io.ppm import find_reddest_pixel, read_ppm, red_centroid
from ..types import GenerationResult, ValidationScore


class RedDotGravityValidator:
    """
    Toy physics validator (example only):
    - tracks the reddest pixel location across frames
    - checks that y(t) has roughly constant positive acceleration

    For real medical animations, replace this with:
    - pose/object tracking + kinematics constraints
    - a PyBullet validation pass for constrained motion
    """

    name = "physics_red_dot_gravity"

    def score(self, generation: GenerationResult) -> ValidationScore:
        if len(generation.frames) < 4:
            return ValidationScore(name=self.name, score=0.5, feedback="Too few frames to validate physics")

        ys: List[float] = []
        confidences: List[float] = []
        for frame_path in generation.frames[: min(24, len(generation.frames))]:
            ppm = read_ppm(frame_path)
            cx, cy, conf = red_centroid(ppm, stride=2, threshold=60.0)
            if conf == float("-inf"):
                _, y, conf = find_reddest_pixel(ppm, stride=2)
                ys.append(float(y))
                confidences.append(float(conf))
            else:
                ys.append(float(cy))
                confidences.append(float(conf))

        if max(confidences) < 20.0:
            return ValidationScore(
                name=self.name,
                score=0.0,
                skipped=True,
                details={"max_redness": max(confidences)},
                feedback="No trackable object found (plug in tracking for your domain)",
            )

        # Second difference approximates acceleration (up to scale).
        acc = []
        for i in range(2, len(ys)):
            acc.append(ys[i] - 2.0 * ys[i - 1] + ys[i - 2])

        if not acc:
            return ValidationScore(name=self.name, score=0.5, feedback="Not enough samples for acceleration")

        mean_acc = sum(acc) / len(acc)
        var = sum((a - mean_acc) ** 2 for a in acc) / max(1, len(acc) - 1)
        std = math.sqrt(var)

        # Prefer positive acceleration (downwards) and low variance.
        sign_ok = 1.0 if mean_acc > 0.0 else 0.2
        # Make this deliberately forgiving (hackathon): good tracking should score high.
        var_score = math.exp(-std / 5.0)
        score = float(max(0.0, min(1.0, sign_ok * var_score)))

        feedback = ""
        if score < 0.75:
            feedback = "Motion is not consistent with constant acceleration; reduce jitter / enforce dynamics"

        return ValidationScore(
            name=self.name,
            score=score,
            details={
                "mean_acc": mean_acc,
                "std_acc": std,
                "samples": len(ys),
                "max_redness": max(confidences),
            },
            feedback=feedback,
        )


@dataclass
class PyBulletPhysicsValidator:
    """
    Stub showing where Bullet fits.

    Expected flow (tomorrow):
    - Extract a coarse scene/state from frames (tracking / segmentation)
    - Build a Bullet scene with constraints
    - Simulate and compare trajectory/contacts against extracted state
    """

    name: str = "physics_pybullet"

    def score(self, generation: GenerationResult) -> ValidationScore:
        return ValidationScore(
            name=self.name,
            score=0.0,
            skipped=True,
            details={"validated": False},
            feedback="PyBulletPhysicsValidator is a stub (skipped)",
        )
