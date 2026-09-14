"""Core data types for salient event detection.

Everything here is **time-only**. A detected (or annotated) event is a half-open
time interval ``[onset_s, offset_s)`` carrying a salience ``score``. Methods that
internally produce 2-D time-frequency regions collapse each region to its time
span before constructing an :class:`Event`; frequency is never represented here and
never used downstream. Keeping a single, minimal event type is what lets every
detector be run and scored through one code path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Event:
    """A single detected or annotated salient event (time interval + score).

    Parameters
    ----------
    onset_s, offset_s
        Start and end of the event in seconds, with ``offset_s >= onset_s``.
    score
        Salience / confidence. Higher means more salient. The evaluator sweeps
        this to build precision-recall curves, so scores only need to be
        comparable *within* one detector's output, not across detectors. Ground
        truth events use a constant score (conventionally ``1.0``).
    """

    onset_s: float
    offset_s: float
    score: float = 1.0

    def __post_init__(self) -> None:
        if self.offset_s < self.onset_s:
            raise ValueError(
                f"offset_s ({self.offset_s}) must be >= onset_s ({self.onset_s})"
            )

    @property
    def duration_s(self) -> float:
        return self.offset_s - self.onset_s

    def overlap_s(self, other: "Event") -> float:
        """Length of the temporal intersection with ``other`` (0 if disjoint)."""
        lo = max(self.onset_s, other.onset_s)
        hi = min(self.offset_s, other.offset_s)
        return max(0.0, hi - lo)

    def iou(self, other: "Event") -> float:
        """Temporal intersection-over-union with ``other`` in ``[0, 1]``."""
        inter = self.overlap_s(other)
        if inter <= 0.0:
            return 0.0
        union = self.duration_s + other.duration_s - inter
        return inter / union if union > 0.0 else 0.0


@dataclass
class Salience:
    """An optional dense, frame-level salience function.

    Detectors whose internal detection function is a 1-D time series (energy,
    spectral flux, model logits, ...) expose it here so the evaluator can compute
    threshold-free frame-level metrics and so it can be plotted for diagnostics.
    Detectors without a natural dense function simply return ``None``.

    ``times_s`` and ``values`` are parallel 1-D arrays of equal length, where
    ``times_s[i]`` is the center time of the frame with salience ``values[i]``.
    """

    times_s: np.ndarray
    values: np.ndarray

    def __post_init__(self) -> None:
        self.times_s = np.asarray(self.times_s, dtype=np.float64)
        self.values = np.asarray(self.values, dtype=np.float64)
        if self.times_s.shape != self.values.shape:
            raise ValueError(
                f"times_s {self.times_s.shape} and values {self.values.shape} "
                "must have the same shape"
            )
        if self.times_s.ndim != 1:
            raise ValueError("Salience arrays must be 1-D")


@dataclass
class Detections:
    """All events a detector produced for one audio file.

    Parameters
    ----------
    file_id
        Identifier of the source recording (typically its basename), used to join
        predictions to ground truth during evaluation.
    events
        The detected events. Stored sorted by ``onset_s`` (the constructor sorts
        them) so downstream code can assume temporal order.
    detector
        The detector's registry name, recorded for provenance.
    params
        The hyperparameters the detector ran with, recorded for provenance.
    salience
        Optional dense salience function (see :class:`Salience`).
    """

    file_id: str
    events: list[Event]
    detector: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    salience: Salience | None = None

    def __post_init__(self) -> None:
        self.events = sorted(self.events, key=lambda e: (e.onset_s, e.offset_s))

    def __len__(self) -> int:
        return len(self.events)
