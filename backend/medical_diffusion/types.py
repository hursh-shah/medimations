from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AnimationSpec:
    prompt: str
    duration_s: float = 2.0
    fps: int = 8
    width: int = 128
    height: int = 128
    input_image_path: Optional[Path] = None
    seed: Optional[int] = None
    negative_prompt: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        return max(1, int(round(self.duration_s * self.fps)))


@dataclass(frozen=True)
class GenerationResult:
    spec: AnimationSpec
    frames: List[Path]
    frames_dir: Path
    backend: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ValidationScore:
    name: str
    score: float
    details: Dict[str, Any] = field(default_factory=dict)
    feedback: str = ""
    skipped: bool = False


@dataclass(frozen=True)
class ValidationReport:
    medical: ValidationScore
    physics: ValidationScore

    def summary(self) -> str:
        parts = [
            f"medical={self.medical.score:.3f}",
            f"physics={self.physics.score:.3f}",
        ]
        return ", ".join(parts)


@dataclass(frozen=True)
class AgentRound:
    round_index: int
    prompt: str
    candidate_index: int
    generation: GenerationResult
    report: ValidationReport


@dataclass(frozen=True)
class AgentResult:
    accepted: bool
    final: AgentRound
    history: List[AgentRound]
