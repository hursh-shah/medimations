from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from ..io.ppm import read_ppm
from ..memory.organmnist3d import (
    OrganMNIST3DCacheConfig,
    embed_2d,
    infer_target_label,
    load_or_build_prototypes,
)
from ..types import GenerationResult, ValidationScore


@dataclass
class OrganMNIST3DVisualValidator:
    """
    Uses MedMNIST OrganMNIST3D as a lightweight visual reference “memory”.

    Flow:
    - Download/caches OrganMNIST3D into `root`
    - Builds per-organ prototypes (mean-projection -> small embedding vector)
    - Compares generated animation frames to prototypes via cosine similarity
    """

    root: Path = Path(".cache/medmnist")
    size: int = 28
    split: str = "train"
    download: bool = False
    embed_size: int = 32
    projection_axis: int = 0
    max_per_label: int = 64
    force_rebuild: bool = False
    export_gifs_dir: Optional[Path] = None
    gifs_per_label: int = 0

    name: str = "medmnist_organmnist3d_visual"

    _prototypes: Optional[object] = field(default=None, init=False, repr=False)
    _init_error: Optional[str] = field(default=None, init=False, repr=False)

    def score(self, generation: GenerationResult) -> ValidationScore:
        protos = self._ensure_prototypes()
        if protos is None:
            return ValidationScore(
                name=self.name,
                score=0.0,
                skipped=True,
                feedback=self._init_error or "OrganMNIST3D memory unavailable",
            )

        if not generation.frames:
            return ValidationScore(name=self.name, score=0.0, feedback="No frames produced")

        vec = _embed_generation_frames(generation, embed_size=self.embed_size)
        sims = protos.similarity(vec)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_label = protos.label_names[best_idx]

        target_idx = infer_target_label(generation.spec.prompt)
        if target_idx is not None:
            target_sim = float(sims[target_idx])
            target_label = protos.label_names[target_idx]
            score = _sim_to_score(target_sim)

            feedback = ""
            if best_idx != target_idx and (best_sim - target_sim) > 0.05:
                feedback = f"Looks closer to {best_label} than {target_label}; emphasize {target_label} anatomy"

            return ValidationScore(
                name=self.name,
                score=score,
                details={
                    "target_label": target_label,
                    "target_similarity": target_sim,
                    "best_label": best_label,
                    "best_similarity": best_sim,
                    "suggested_keywords": _suggest_keywords(target_label),
                },
                feedback=feedback,
            )

        # No explicit organ requested; score is best-match confidence.
        return ValidationScore(
            name=self.name,
            score=_sim_to_score(best_sim),
            details={
                "best_label": best_label,
                "best_similarity": best_sim,
                "suggested_keywords": _suggest_keywords(best_label),
            },
        )

    def _ensure_prototypes(self):
        if self._prototypes is not None:
            return self._prototypes
        if self._init_error is not None:
            return None
        try:
            cfg = OrganMNIST3DCacheConfig(
                root=self.root,
                size=self.size,
                split=self.split,
                embed_size=self.embed_size,
                projection_axis=self.projection_axis,
                max_per_label=self.max_per_label,
            )
            self._prototypes = load_or_build_prototypes(
                cfg=cfg,
                download=self.download,
                force_rebuild=self.force_rebuild,
                export_gifs_dir=self.export_gifs_dir,
                gifs_per_label=self.gifs_per_label,
            )
            return self._prototypes
        except Exception as e:
            self._init_error = str(e)
            return None


def _embed_generation_frames(generation: GenerationResult, *, embed_size: int) -> np.ndarray:
    # Sample up to 6 frames evenly.
    frames = generation.frames
    n = len(frames)
    k = min(6, n)
    idxs = np.linspace(0, n - 1, k).astype(np.int32).tolist()

    imgs = []
    for i in idxs:
        ppm = read_ppm(frames[i])
        rgb = np.frombuffer(ppm.rgb, dtype=np.uint8).reshape((ppm.height, ppm.width, 3))
        gray = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]).astype(np.float32)
        imgs.append(gray)

    mean_img = np.mean(np.stack(imgs, axis=0), axis=0)
    return embed_2d(mean_img, embed_size=embed_size)


def _sim_to_score(sim: float) -> float:
    # sim is cosine similarity in [-1, 1]; map to [0, 1].
    return float(max(0.0, min(1.0, (sim + 1.0) / 2.0)))


def _suggest_keywords(label_name: str) -> list[str]:
    # Keep these minimal; agent will append them to the prompt.
    base = label_name.replace("-", " ")
    return [base, "ct", "axial", "slice"]
