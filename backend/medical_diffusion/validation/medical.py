from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

from ..io.ppm import mean_rgb, read_ppm
from ..types import GenerationResult, ValidationScore


@dataclass(frozen=True)
class ReferenceEntry:
    id: str
    description: str
    keywords: Set[str]


@dataclass(frozen=True)
class ReferenceLibrary:
    entries: Sequence[ReferenceEntry]

    @staticmethod
    def from_dir(path: Path) -> "ReferenceLibrary":
        manifest_path = path / "manifest.json"
        raw = json.loads(manifest_path.read_text())
        entries: List[ReferenceEntry] = []
        for item in raw:
            entries.append(
                ReferenceEntry(
                    id=str(item["id"]),
                    description=str(item.get("description", "")),
                    keywords={str(k).lower() for k in item.get("keywords", [])},
                )
            )
        return ReferenceLibrary(entries=entries)


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


@dataclass
class KeywordLibraryMedicalValidator:
    """
    Hacky fallback: scores prompt against a keyword-only reference library.

    It does *not* inspect the pixels; it just gives the agent a direction for
    reprompting ("mention key anatomy terms", etc.).
    """

    reference: ReferenceLibrary
    name: str = "medical_library_keywords"

    def score(self, generation: GenerationResult) -> ValidationScore:
        prompt_tokens = _tokenize(generation.spec.prompt)
        if not prompt_tokens:
            return ValidationScore(name=self.name, score=0.0, feedback="Empty prompt")

        best = 0.0
        best_entry: Optional[ReferenceEntry] = None
        for entry in self.reference.entries:
            if not entry.keywords:
                continue
            j = _jaccard(prompt_tokens, entry.keywords)
            if j > best:
                best = j
                best_entry = entry

        score = float(min(1.0, best * 1.5))  # scale: j=0.66 -> 1.0
        feedback = ""
        details: Dict[str, Any] = {"best_jaccard": best}
        if best_entry:
            details["best_id"] = best_entry.id
            details["best_description"] = best_entry.description
            missing = sorted(list(best_entry.keywords - prompt_tokens))[:8]
            details["suggested_keywords"] = missing
            if score < 0.85 and missing:
                feedback = "Add key terms: " + ", ".join(missing)

        return ValidationScore(name=self.name, score=score, details=details, feedback=feedback)


def _tokenize(text: str) -> Set[str]:
    parts = []
    cur = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                parts.append("".join(cur))
                cur = []
    if cur:
        parts.append("".join(cur))
    stop = {"a", "an", "the", "and", "or", "to", "of", "in", "on", "with", "show", "showing", "animation"}
    return {p for p in parts if p and p not in stop and len(p) > 1}


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union
