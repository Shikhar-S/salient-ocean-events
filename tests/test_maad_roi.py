"""Unit tests for the scikit-maad 2-D ROI detector."""

import numpy as np
from scipy.signal import butter, sosfilt

import src.detectors.maad_roi  # noqa: F401  -- registers maad_roi via import side-effect
from src.core.detector import get_detector


def _implant(sr=48000, dur=6.0, bursts=((1.0, 1.4, 3000, 5000), (3.5, 4.0, 7000, 9000)),
             seed=0):
    """Silent clip with band-limited noise bursts.

    Each burst is ``(onset_s, offset_s, lo_hz, hi_hz)``: white noise bandpassed to
    ``[lo, hi]`` makes a compact, high-contrast time-frequency blob -- an obvious
    ROI. Returns the waveform and the list of ``(onset_s, offset_s)`` ground truth.
    """
    x = np.zeros(int(dur * sr), dtype=np.float32)
    rng = np.random.default_rng(seed)
    gt = []
    for on, off, lo, hi in bursts:
        a, b = int(on * sr), int(off * sr)
        w = rng.standard_normal(b - a)
        sos = butter(4, [lo, hi], btype="band", fs=sr, output="sos")
        y = sosfilt(sos, w).astype(np.float32)
        y /= np.abs(y).max() + 1e-9
        x[a:b] = 0.8 * y
        gt.append((on, off))
    return x, gt


def test_recovers_implanted_bursts():
    det = get_detector("maad_roi")
    x, gt = _implant()
    out = det.detect(x, 48000)
    assert len(out.events) >= 1
    # Each implanted burst is overlapped by some detected event in time.
    for on, off in gt:
        assert any(e.onset_s < off and e.offset_s > on for e in out.events), (on, off)
    # Events are scored by ROI energy, strictly positive.
    assert all(e.score > 0 for e in out.events)
    # Time-only spans are well-formed.
    assert all(e.offset_s > e.onset_s for e in out.events)


def test_empty_audio_yields_no_events():
    det = get_detector("maad_roi")
    out = det.detect(np.zeros(0, dtype=np.float32), 48000)
    assert out.events == []


def test_too_short_audio_yields_no_events():
    det = get_detector("maad_roi")
    # Fewer samples than one STFT frame -> graceful no-op.
    out = det.detect(np.zeros(256, dtype=np.float32), 48000)
    assert out.events == []


def test_provenance_is_stamped():
    det = get_detector("maad_roi")
    x, _ = _implant()
    out = det.detect(x, 48000)
    assert out.detector == "maad_roi"
    assert out.params["bin_std"] == det.bin_std
    assert out.salience is None


def test_does_not_degenerate_to_whole_clip_roi():
    """A short event in a noisy clip must not collapse into one whole-file ROI.

    Real Watkins recordings carry a stationary broadband floor (rumble/hiss). With
    no per-frequency noise subtraction, that floor forms a single connected
    component spanning the entire clip, so every event becomes one whole-file span
    whose offset is nowhere near the truth -> recall collapses under the collar
    matcher. The noise-floor subtraction is what prevents this; guard it here.
    """
    sr, dur = 48000, 8.0
    rng = np.random.default_rng(3)
    # Stationary low-frequency rumble across the whole clip + faint broadband hiss.
    n = int(dur * sr)
    sos = butter(4, [40, 400], btype="band", fs=sr, output="sos")
    floor = sosfilt(sos, rng.standard_normal(n)).astype(np.float32)
    floor /= np.abs(floor).max() + 1e-9
    x = 0.2 * floor + 0.01 * rng.standard_normal(n).astype(np.float32)
    # One compact transient call at 5.0-5.4 s.
    a, b = int(5.0 * sr), int(5.4 * sr)
    sos2 = butter(4, [4000, 7000], btype="band", fs=sr, output="sos")
    call = sosfilt(sos2, rng.standard_normal(b - a)).astype(np.float32)
    call /= np.abs(call).max() + 1e-9
    x[a:b] += 0.8 * call

    out = get_detector("maad_roi").detect(x.astype(np.float32), sr)
    assert len(out.events) >= 1
    # No single event may span essentially the whole clip.
    assert all(e.duration_s < 0.9 * dur for e in out.events), [
        (round(e.onset_s, 2), round(e.offset_s, 2)) for e in out.events
    ]
    # The transient is recovered by some time-localized event.
    assert any(e.onset_s < 5.4 and e.offset_s > 5.0 for e in out.events)


def test_noise_floor_subtraction_breaks_up_background_band():
    """With subtraction off, the stationary band yields a near-whole-clip ROI;
    turning it on must localize the events instead."""
    sr, dur = 48000, 8.0
    rng = np.random.default_rng(5)
    n = int(dur * sr)
    sos = butter(4, [40, 400], btype="band", fs=sr, output="sos")
    floor = sosfilt(sos, rng.standard_normal(n)).astype(np.float32)
    floor /= np.abs(floor).max() + 1e-9
    x = (0.25 * floor).astype(np.float32)
    a, b = int(5.0 * sr), int(5.4 * sr)
    sos2 = butter(4, [4000, 7000], btype="band", fs=sr, output="sos")
    call = sosfilt(sos2, rng.standard_normal(b - a)).astype(np.float32)
    call /= np.abs(call).max() + 1e-9
    x[a:b] += 0.8 * call

    off = get_detector("maad_roi", subtract_noise_floor=False).detect(x, sr)
    on = get_detector("maad_roi", subtract_noise_floor=True).detect(x, sr)
    longest_off = max((e.duration_s for e in off.events), default=0.0)
    longest_on = max((e.duration_s for e in on.events), default=0.0)
    # Subtraction must not make the longest event longer (it should shorten it).
    assert longest_on <= longest_off + 1e-6


def test_silent_clip_is_robust_and_quiet():
    """A fully silent clip must neither crash nor invent events.

    (maad's median_equalizer raises on a silent spectrogram; the per-frequency
    median subtraction used here must degrade gracefully instead.)
    """
    det = get_detector("maad_roi")
    out = det.detect(np.zeros(int(4 * 48000), dtype=np.float32), 48000)
    assert out.events == []
