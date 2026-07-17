"""Immutable innate priors: tendencies, never authored phrases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class PersonalityGenome:
    genome_id: str
    version: int
    sociability: float
    curiosity: float
    impulsivity: float
    chaos: float
    suspicion: float
    patience: float
    verbosity: float
    expressiveness: float
    memory_attachment: float
    novelty_seeking: float
    response_threshold: float
    initiative: float

    def __post_init__(self) -> None:
        if not self.genome_id or self.version <= 0:
            raise ValueError("genome identity and version must be valid")
        for name, value in self.traits().items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"genome trait {name} must be between 0 and 1")

    def traits(self) -> dict[str, float]:
        data = asdict(self)
        return {
            name: float(value)
            for name, value in data.items()
            if name not in {"genome_id", "version"}
        }

    @property
    def signature(self) -> str:
        encoded = json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()[:16]


DEBILGI_GENOME = PersonalityGenome(
    genome_id="debilgi-primary",
    version=1,
    sociability=0.64,
    curiosity=0.67,
    impulsivity=0.81,
    chaos=0.74,
    suspicion=0.58,
    patience=0.23,
    verbosity=0.39,
    expressiveness=0.79,
    memory_attachment=0.72,
    novelty_seeking=0.76,
    response_threshold=0.24,
    initiative=0.71,
)
