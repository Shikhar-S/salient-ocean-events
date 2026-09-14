"""Unit tests for the energy-template self-similarity detector."""

import numpy as np

import src.detectors.energy_template  # noqa: F401  (registers the detector)
from src.core.detector import get_detector

SR = 48000


def _chirp(sr, dur_s, f0=4000.0, f1=9000.0, seed=0):
    """A short stereotyped tone-burst (linear sweep) windowed by a Hann taper."""
    n = int(dur_s * sr)
    t = np.arange(n) / sr
    f = f0 + (f1 - f0) * (t / dur_s)
    w = np.hanning(n)
    return (0.6 * w * np.sin(2 * np.pi * f * t)).astype(np.float32)


def _build_clip():
    """48 kHz clip: identical chirp at 1.0 s and 3.5 s, plus one odd-one-out burst.

    Returns ``(x, repeated_spans, odd_span)``.
    """
    dur = 6.0
    x = np.zeros(int(dur * SR), dtype=np.float32)

    chirp = _chirp(SR, 0.3)
    repeated = [(1.0, 1.3), (3.5, 3.8)]
    for on, _ in repeated:
        a = int(on * SR)
        x[a : a + chirp.size] += chirp

    # An odd-one-out: a broadband white-noise burst, different shape/spectrum.
    rng = np.random.default_rng(1)
    odd = (5.0, 5.3)
    a, b = int(odd[0] * SR), int(odd[1] * SR)
    x[a:b] += 0.5 * rng.standard_normal(b - a).astype(np.float32)

    return x, repeated, odd


def _overlaps(events, on, off):
    return any(e.onset_s < off and e.offset_s > on for e in events)


def _event_covering(events, on, off):
    matches = [e for e in events if e.onset_s < off and e.offset_s > on]
    return max(matches, key=lambda e: e.score) if matches else None


def test_recovers_implanted_events():
    det = get_detector("energy_template")
    x, repeated, odd = _build_clip()
    out = det.detect(x, SR)
    assert len(out.events) >= 2
    for on, off in repeated:
        assert _overlaps(out.events, on, off), (on, off)
    assert _overlaps(out.events, *odd)
    assert all(e.score > 0 for e in out.events)


def test_recurrence_is_rewarded():
    """The two identical chirps score above the odd-one-out white-noise burst."""
    det = get_detector("energy_template")
    x, repeated, odd = _build_clip()
    out = det.detect(x, SR)

    rep_events = [_event_covering(out.events, on, off) for on, off in repeated]
    assert all(e is not None for e in rep_events)
    odd_event = _event_covering(out.events, *odd)
    assert odd_event is not None

    rep_scores = [e.score for e in rep_events]
    # Recurrence rewarded: both repeats beat the one-off burst...
    assert min(rep_scores) > odd_event.score
    # ...and, robustly, both sit above the median event score.
    median = float(np.median([e.score for e in out.events]))
    assert all(s >= median for s in rep_scores)


def test_single_candidate_fallback():
    """A clip with one lone event still emits a positively-scored event."""
    det = get_detector("energy_template")
    x = np.zeros(int(4.0 * SR), dtype=np.float32)
    chirp = _chirp(SR, 0.3)
    a = int(1.5 * SR)
    x[a : a + chirp.size] += chirp
    out = det.detect(x, SR)
    assert len(out.events) >= 1
    assert all(e.score > 0 for e in out.events)


def test_salience_is_parallel():
    det = get_detector("energy_template")
    x, _, _ = _build_clip()
    sal = det.salience(x, SR)
    assert sal is not None
    assert sal.times_s.shape == sal.values.shape
    assert sal.times_s.ndim == 1


def test_empty_audio_yields_no_events():
    det = get_detector("energy_template")
    out = det.detect(np.zeros(0, dtype=np.float32), SR)
    assert out.events == []


def test_stage2_never_drops_candidates():
    """Recall == Stage-1 recall: every energy candidate survives to the output.

    The self-similarity stage only re-scores; it must not filter candidates.
    """
    from src.detectors._common import salience_to_events

    det = get_detector("energy_template")
    x, _, _ = _build_clip()
    times, energy_db, _ = det._mel(x, SR)
    cands = salience_to_events(
        times,
        energy_db,
        k=det.k,
        min_dur_s=det.min_dur_s,
        merge_gap_s=det.merge_gap_s,
    )
    out = det.detect(x, SR)
    # One emitted event per Stage-1 candidate, same spans, all finite-scored.
    assert len(out.events) == len(cands)
    cand_spans = sorted((round(c.onset_s, 6), round(c.offset_s, 6)) for c in cands)
    ev_spans = sorted((round(e.onset_s, 6), round(e.offset_s, 6)) for e in out.events)
    assert cand_spans == ev_spans
    assert all(np.isfinite(e.score) for e in out.events)


def test_one_frame_patch_feature_is_finite_unit_vector():
    """A degenerate 1-frame candidate yields a finite (unit or zero) feature."""
    det = get_detector("energy_template")
    rng = np.random.default_rng(0)
    vec = det._patch_feature(rng.standard_normal((det.n_mels, 1)))
    assert vec.shape == (det.n_mels * det.patch_w,)
    assert not np.isnan(vec).any()
    assert np.isclose(np.linalg.norm(vec), 1.0)


def test_flat_patch_feature_is_zero_no_nan():
    """A silent/flat patch (std == 0) must not produce NaNs (no div-by-zero)."""
    det = get_detector("energy_template")
    vec = det._patch_feature(np.full((det.n_mels, 5), -80.0))
    assert not np.isnan(vec).any()
    assert np.linalg.norm(vec) == 0.0
    # And such a candidate still scores finitely against a real peer.
    real = det._patch_feature(np.random.default_rng(0).standard_normal((det.n_mels, 8)))
    scores = det._recurrence_scores([vec, real])
    assert len(scores) == 2
    assert all(np.isfinite(s) for s in scores)


def test_two_candidate_topk_is_finite():
    """With exactly 2 candidates, top-k (k>peers) clamps without NaN/inf."""
    det = get_detector("energy_template")
    rng = np.random.default_rng(1)
    f1 = det._patch_feature(rng.standard_normal((det.n_mels, 10)))
    f2 = det._patch_feature(rng.standard_normal((det.n_mels, 10)))
    scores = det._recurrence_scores([f1, f2])
    assert len(scores) == 2
    assert all(np.isfinite(s) for s in scores)
    # Identical patches -> NCC ~ 1 for both.
    same = det._recurrence_scores([f1, f1])
    assert all(s > 0.99 for s in same)


def test_generous_threshold_default():
    """The default threshold k stays low/generous for recall."""
    det = get_detector("energy_template")
    assert det.k <= 0.5
