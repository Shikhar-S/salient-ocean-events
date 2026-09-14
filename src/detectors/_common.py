"""Shared converter: turn a detector's 1-D salience curve into scored events.

Used by ``pcen_peak``, ``spectral_entropy`` and ``energy_template`` (``maad_roi``
builds its spans from 2-D regions instead, so it does not go through here).

The threshold is deliberately low/generous so weak candidates survive the AP sweep
(high recall, many low-scored FPs). Events are scored by the **integrated salience**
over each span (area, not a single-sample peak) so a lone noise frame cannot earn a
high score -- this mirrors how the ROI detectors score by summed power and stops AP
collapsing to prevalence. Spans are real ``[onset, offset)`` intervals -- the offset
matters because :class:`~src.eval.matching.CollarMatcher` scores it too, so a bare
onset would fail.
"""

from __future__ import annotations

import numpy as np

from src.core.types import Event


def _threshold(values: np.ndarray, k: float) -> float:
    """Robust low threshold ``median + k * spread`` (MAD, std fallback)."""
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    spread = mad if mad > 0 else (float(values.std()) or 1e-9)
    return med + k * spread


def salience_to_events(
    times: np.ndarray,
    values: np.ndarray,
    *,
    k: float = 1.5,
    min_dur_s: float = 0.05,
    merge_gap_s: float = 0.05,
) -> list[Event]:
    """Turn a 1-D salience curve into scored :class:`Event` spans.

    Parameters
    ----------
    times, values
        Parallel 1-D arrays: frame-center times (s) and the salience at each.
    k
        Threshold sharpness. The cut is ``median + k * MAD`` of the curve (a
        robust, deliberately low floor so weak candidates survive the AP sweep).
    min_dur_s
        Drop merged spans shorter than this (blip rejection), applied last.
    merge_gap_s
        Bridge above-threshold runs separated by gaps no longer than this.

    Notes
    -----
    Each span is scored by the **integrated salience above the threshold** over
    its extent (area = ``sum(values - thr)+ * dt``), not the single-sample peak.
    A lone noise frame can momentarily match the peak, which made AP collapse to
    prevalence; the area rewards genuine sustained events (like the ROI detectors'
    summed-power score) and so separates real spans from noise blips.
    """
    times = np.asarray(times, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if times.size == 0:
        return []

    thr = _threshold(values, k)

    dt = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    half = dt / 2.0 if dt > 0 else 0.0
    area_dt = dt if dt > 0 else 1.0

    above = values >= thr
    spans: list[list[float]] = []  # [onset, offset, area]
    i, n = 0, len(above)
    while i < n:
        if not above[i]:
            i += 1
            continue
        j = i
        while j < n and above[j]:
            j += 1
        onset = max(0.0, times[i] - half)
        offset = times[j - 1] + half
        area = float((values[i:j] - thr).sum() * area_dt)
        spans.append([onset, offset, area])
        i = j

    merged: list[list[float]] = []
    for s in spans:
        if merged and s[0] - merged[-1][1] <= merge_gap_s:
            merged[-1][1] = s[1]
            merged[-1][2] += s[2]  # accumulate area across bridged runs
        else:
            merged.append(s)

    return [
        Event(onset_s=on, offset_s=off, score=max(area, 1e-12))
        for on, off, area in merged
        if off - on >= min_dur_s
    ]
