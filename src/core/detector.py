"""The :class:`Detector` interface and a name-based registry.

Every salient event detection method subclasses :class:`Detector` and implements
a single ``detect(audio, sr) -> Detections`` method that consumes a float32 mono
waveform and returns time-only :class:`~src.core.types.Event` objects with scores.
Audio loading, resampling, and long-file chunking are the *runner's* job, not the
detector's, so each method stays a small, testable unit.

Detectors self-register with :func:`register` so the benchmark CLI can select them
by name (``--detectors pcen_peak,maad_roi`` or ``all``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np

from src.core.types import Detections, Salience


class Detector(ABC):
    """Base class for all salient event detectors.

    Subclasses set a unique class attribute ``name`` and implement
    :meth:`detect`. Hyperparameters passed to ``__init__`` are stored on
    ``self.params`` and copied into every :class:`Detections` for provenance.
    """

    #: Unique registry key; set by subclasses (the :func:`register` decorator
    #: also assigns it to keep the two in sync).
    name: str = ""

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    @abstractmethod
    def detect(self, audio: np.ndarray, sr: int) -> Detections:
        """Detect salient events in ``audio`` (float32 mono at ``sr`` Hz).

        Returns a :class:`Detections`. Implementations should set ``detector`` to
        ``self.name`` and ``params`` to ``self.params``; the convenience helper
        :meth:`_pack` does this.
        """
        raise NotImplementedError

    def salience(self, audio: np.ndarray, sr: int) -> Salience | None:
        """Optional dense salience function; ``None`` unless a subclass overrides."""
        return None

    def _pack(
        self,
        file_id: str,
        events: list,
        salience: Salience | None = None,
    ) -> Detections:
        """Build a :class:`Detections` stamped with this detector's provenance."""
        return Detections(
            file_id=file_id,
            events=events,
            detector=self.name,
            params=dict(self.params),
            salience=salience,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        kv = ", ".join(f"{k}={v!r}" for k, v in self.params.items())
        return f"{type(self).__name__}({kv})"


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

REGISTRY: dict[str, type[Detector]] = {}


def register(name: str) -> Callable[[type[Detector]], type[Detector]]:
    """Class decorator registering a :class:`Detector` subclass under ``name``."""

    def deco(cls: type[Detector]) -> type[Detector]:
        if not issubclass(cls, Detector):
            raise TypeError(f"{cls.__name__} is not a Detector subclass")
        if name in REGISTRY and REGISTRY[name] is not cls:
            raise ValueError(f"detector name {name!r} is already registered")
        cls.name = name
        REGISTRY[name] = cls
        return cls

    return deco


def get_detector(name: str, **params: Any) -> Detector:
    """Instantiate a registered detector by name with the given hyperparameters."""
    try:
        cls = REGISTRY[name]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "(none registered)"
        raise KeyError(f"unknown detector {name!r}; registered: {known}") from None
    return cls(**params)
