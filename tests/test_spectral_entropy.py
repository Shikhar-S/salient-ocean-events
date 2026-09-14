"""Unit tests for the tonality (spectral-entropy) detector.

The headline test is *discrimination*: a tonal burst must score above a
broadband-noise burst of similar energy (the property energy/flux methods lack).
"""

import numpy as np

import src.detectors.spectral_entropy  # noqa: F401  registers the detector
from src.core.detector import get_detector

SR = 48000


def _clip(dur=6.0, seed=0):
    """Silence with a TONAL burst at 1.0-1.4 s and a BROADBAND burst at 3.5-4.0 s."""
    n = int(dur * SR)
    x = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    # tonal: 3 kHz sine
    a, b = int(1.0 * SR), int(1.4 * SR)
    t = np.arange(b - a) / SR
    x[a:b] = 0.5 * np.sin(2 * np.pi * 3000 * t).astype(np.float32)
    # broadband: white noise, similar amplitude
    a2, b2 = int(3.5 * SR), int(4.0 * SR)
    x[a2:b2] = 0.5 * rng.standard_normal(b2 - a2).astype(np.float32)
    return x, (1.0, 1.4), (3.5, 4.0)


def _max_score(events, on, off):
    hits = [e.score for e in events if e.onset_s < off and e.offset_s > on]
    return max(hits) if hits else 0.0


def test_tonal_outscores_broadband():
    det = get_detector("spectral_entropy")
    x, tonal, broad = _clip()
    out = det.detect(x, SR)
    tonal_score = _max_score(out.events, *tonal)
    broad_score = _max_score(out.events, *broad)
    # The tonal burst is detected...
    assert tonal_score > 0
    # ...and scores strictly above the broadband burst (the discrimination property).
    assert tonal_score > broad_score


def test_salience_parallel_and_bounded():
    det = get_detector("spectral_entropy")
    x, _, _ = _clip()
    sal = det.salience(x, SR)
    assert sal.times_s.shape == sal.values.shape
    assert sal.values.min() >= 0.0


def test_empty_audio_yields_no_events():
    det = get_detector("spectral_entropy")
    out = det.detect(np.zeros(0, dtype=np.float32), SR)
    assert out.events == []
