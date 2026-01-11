from __future__ import annotations

from typing import List

from ..io.ppm import mean_rgb, read_ppm
from ..types import GenerationResult, ValidationScore


class FrameSanityMedicalValidator:
    """
    Not "medical correctness" — just basic sanity checks to keep the loop real.

    Replace/extend with:
    - CLIP/OpenCLIP embeddings vs a labeled reference library
    - a VLM/LLM judge with structured rubric (anatomy, flow direction, etc.)
    """

    name = "medical_sanity"

    def score(self, generation: GenerationResult) -> ValidationScore:
        if not generation.frames:
            return ValidationScore(name=self.name, score=0.0, feedback="No frames produced")

        means = []
        for frame_path in generation.frames[: min(12, len(generation.frames))]:
            ppm = read_ppm(frame_path)
            r, g, b = mean_rgb(ppm, stride=32)
            means.append((r, g, b))

        # Penalize extreme flicker in mean brightness.
        diffs = []
        for (r0, g0, b0), (r1, g1, b1) in zip(means, means[1:]):
            diffs.append(abs((r1 + g1 + b1) - (r0 + g0 + b0)) / 3.0)

        flicker = (sum(diffs) / len(diffs)) if diffs else 0.0
        # Map flicker into score: 0 flicker => 1.0, 30+ => ~0.
        score = max(0.0, 1.0 - (flicker / 30.0))

        feedback = ""
        if score < 0.75:
            feedback = "Reduce frame-to-frame flicker; enforce temporal consistency"

        return ValidationScore(
            name=self.name,
            score=float(score),
            details={"mean_flicker": flicker, "sampled_frames": len(means)},
            feedback=feedback,
        )
