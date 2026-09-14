"""Matchers deciding when a predicted event matches a ground-truth event.

All matching is **time-only**. A matcher is a callable that, given a predicted
event and a ground-truth event, returns either ``None`` (incompatible) or a
non-negative *quality* score (higher = better) used to break ties when a single
prediction could match several ground-truth events. Two matchers are provided:

* :class:`CollarMatcher` -- the **primary**, DCASE-style criterion: onset within a
  fixed collar and offset within a collar that grows with event duration.
* :class:`IoUMatcher` -- a secondary criterion: temporal IoU above a threshold.
"""

from __future__ import annotations

from typing import Protocol

from src.core.types import Event


class Matcher(Protocol):
    """Callable returning a match quality, or ``None`` if pred/gt are incompatible."""

    name: str

    def __call__(self, pred: Event, gt: Event) -> float | None: ...


class CollarMatcher:
    """DCASE-style onset/offset collar matching (the primary criterion).

    A prediction matches a ground-truth event when

    * ``|pred.onset - gt.onset| <= onset_collar`` and
    * ``|pred.offset - gt.offset| <= max(offset_collar, offset_fraction * gt.duration)``.

    The duration-scaled offset tolerance keeps long events from being penalized for
    small absolute offset errors while keeping short events strict.

    Quality is ``-|pred.onset - gt.onset|`` so that, among compatible events, the
    one with the closest onset is preferred.
    """

    def __init__(
        self,
        onset_collar: float = 0.2,
        offset_collar: float = 0.2,
        offset_fraction: float = 0.2,
    ) -> None:
        self.onset_collar = onset_collar
        self.offset_collar = offset_collar
        self.offset_fraction = offset_fraction
        self.name = f"collar(on={onset_collar},off={offset_collar},frac={offset_fraction})"

    def __call__(self, pred: Event, gt: Event) -> float | None:
        onset_err = abs(pred.onset_s - gt.onset_s)
        if onset_err > self.onset_collar:
            return None
        offset_tol = max(self.offset_collar, self.offset_fraction * gt.duration_s)
        if abs(pred.offset_s - gt.offset_s) > offset_tol:
            return None
        return -onset_err


class IoUMatcher:
    """Temporal intersection-over-union matching (the secondary criterion).

    A prediction matches a ground-truth event when their temporal IoU is at least
    ``threshold``. Quality is the IoU itself, so the tightest overlap is preferred.
    """

    def __init__(self, threshold: float = 0.5) -> None:
        self.threshold = threshold
        self.name = f"iou(>={threshold})"

    def __call__(self, pred: Event, gt: Event) -> float | None:
        iou = pred.iou(gt)
        return iou if iou >= self.threshold else None
