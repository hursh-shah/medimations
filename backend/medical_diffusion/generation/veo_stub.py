from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..types import AnimationSpec, GenerationResult
from .base import DiffusionBackend


@dataclass
class VeoBackend:
    """
    Stub for a Veo-style video diffusion API.

    Replace `generate()` with real API calls and frame downloads.
    """

    api_key: str
    name: str = "veo"

    def generate(self, *, spec: AnimationSpec, output_dir: Path) -> GenerationResult:
        raise NotImplementedError(
            "VeoBackend is a stub. Implement API call + frame download in "
            f"{__file__}."
        )

