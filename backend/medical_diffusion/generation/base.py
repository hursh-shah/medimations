from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Protocol

from ..types import AnimationSpec, GenerationResult


class DiffusionBackend(Protocol):
    name: str

    def generate(self, *, spec: AnimationSpec, output_dir: Path) -> GenerationResult:
        ...


@dataclass(frozen=True)
class BackendUnavailable:
    name: str
    reason: str


def ensure_empty_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

