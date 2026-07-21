"""Top-level orchestration and fail-safe state handling (Stage 4)."""
"""Runtime coordination boundaries."""

from .sequential_part_a import (
    CoordinatorActionOutcome,
    PartAPhase,
    SequentialPartACoordinator,
)

__all__ = [
    "CoordinatorActionOutcome",
    "PartAPhase",
    "SequentialPartACoordinator",
]
