"""Audio loading and resampling.

Ported from ``cut_provenance_align.py`` so the whole project loads audio the same
way: float32, mono, with polyphase resampling. The runner uses these to hand each
detector a single waveform array; detectors never touch the filesystem.
"""

from __future__ import annotations

from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def load_mono(path: str) -> tuple[np.ndarray, int]:
    """Load a sound file as float32 mono. Returns ``(samples, samplerate)``."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), int(sr)


def resample_to(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase-resample ``x`` from ``sr_from`` to ``sr_to`` Hz (float32)."""
    if sr_from == sr_to:
        return np.ascontiguousarray(x, dtype=np.float32)
    g = gcd(int(sr_from), int(sr_to))
    up, down = int(sr_to) // g, int(sr_from) // g
    return resample_poly(x, up, down).astype(np.float32)


def load_mono_resampled(path: str, target_sr: int | None) -> tuple[np.ndarray, int]:
    """Load as mono float32 and optionally resample to ``target_sr``.

    If ``target_sr`` is ``None`` the native rate is kept. Returns
    ``(samples, sr)`` where ``sr`` is the rate of the returned samples.
    """
    x, sr = load_mono(path)
    if target_sr is not None and target_sr != sr:
        x = resample_to(x, sr, target_sr)
        sr = int(target_sr)
    return x, sr
