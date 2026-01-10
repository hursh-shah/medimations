from __future__ import annotations

from typing import Protocol

from ..types import GenerationResult, ValidationScore


class Validator(Protocol):
    name: str

    def score(self, generation: GenerationResult) -> ValidationScore:
        ...

