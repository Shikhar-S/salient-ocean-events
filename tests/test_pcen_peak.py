"""Unit tests for the PCEN peak-picking detector."""

import numpy as np

import src.detectors.pcen_peak  # noqa: F401  (registers the detector)
from src.core.detector import get_detector
from src.core.types import Event
from src.eval.matching import CollarMatcher


def _implant(sr=48000, dur=6.0, bursts=((1.0, 1.4), (3.5, 4.0)), seed=0):
    """Silent clip with white-noise bursts at the given (onset, offset) seconds."""
    x = np.zeros(int(dur * sr), dtype=np.float32)
    rng = np.random.default_rng(seed)
    for on, off in bursts:
        a, b = int(on * sr), int(off * sr)
        x[a:b] = 0.5 * rng.standard_normal(b - a).astype(np.float32)
    return x, bursts


def _tone(on, off, sr=48000, dur=8.0, freq=4000.0, amp=0.3):
    """Silent clip with one pure-tone (sustained tonal call) burst."""
    x = np.zeros(int(dur * sr), dtype=np.float32)
    a, b = int(on * sr), int(off * sr)
    t = np.arange(b - a) / sr
    x[a:b] = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return x


def _matches(events, on, off):
    """True if some detected event matches GT (on, off) under the primary collar."""
    mc = CollarMatcher()
    gt = Event(on, off, 1.0)
    return any(mc(e, gt) is not None for e in events)


def test_recovers_implanted_bursts():
    det = get_detector("pcen_peak")
    x, bursts = _implant()
    out = det.detect(x, 48000)
    assert len(out.events) >= 1
    # Each implanted burst is overlapped by some detected event.
    for on, off in bursts:
        assert any(e.onset_s < off and e.offset_s > on for e in out.events), (on, off)
    # Events are scored (peak salience), strictly positive.
    assert all(e.score > 0 for e in out.events)


def test_salience_is_parallel():
    det = get_detector("pcen_peak")
    x, _ = _implant()
    sal = det.salience(x, 48000)
    assert sal is not None
    assert sal.times_s.shape == sal.values.shape


def test_recovers_sustained_tonal_call_under_collar():
    """A multi-second tonal whistle must be recovered as one full-length span.

    This is the PCEN recall trap: an over-aggressive AGC (high ``gain`` / tiny
    ``time_constant``) adapts within the call and flattens its sustained body,
    so the detected event shrinks to a sliver near the onset and its onset AND
    offset both miss the matcher's collars. The default params must keep the
    whole 3-second call elevated. Scored against the *primary* CollarMatcher,
    which checks both onset and offset.
    """
    det = get_detector("pcen_peak")
    on, off = 2.0, 5.0
    x = _tone(on, off)
    out = det.detect(x, 48000)
    assert _matches(out.events, on, off), [
        (round(e.onset_s, 2), round(e.offset_s, 2)) for e in out.events
    ]


def test_recovers_short_tonal_call_under_collar():
    """A short (0.4 s) tonal call must also be recovered under the collar."""
    det = get_detector("pcen_peak")
    on, off = 3.0, 3.4
    x = _tone(on, off)
    out = det.detect(x, 48000)
    assert _matches(out.events, on, off), [
        (round(e.onset_s, 2), round(e.offset_s, 2)) for e in out.events
    ]


def test_noise_bursts_recovered_under_collar():
    """Both transient noise bursts match the primary collar (onset+offset)."""
    det = get_detector("pcen_peak")
    x, bursts = _implant()
    out = det.detect(x, 48000)
    for on, off in bursts:
        assert _matches(out.events, on, off), (on, off, len(out.events))


def test_silence_yields_no_events():
    """All-silence input must not hallucinate events (no NaN/Inf blow-ups)."""
    det = get_detector("pcen_peak")
    out = det.detect(np.zeros(int(3 * 48000), dtype=np.float32), 48000)
    assert out.events == []
    sal = det.salience(np.zeros(int(3 * 48000), dtype=np.float32), 48000)
    assert np.all(np.isfinite(sal.values))


def test_very_short_audio_does_not_crash():
    """A few samples (shorter than one FFT frame) must not raise."""
    det = get_detector("pcen_peak")
    out = det.detect(np.zeros(64, dtype=np.float32), 48000)
    assert isinstance(out.events, list)


def test_empty_audio_yields_no_events():
    det = get_detector("pcen_peak")
    out = det.detect(np.zeros(0, dtype=np.float32), 48000)
    assert out.events == []
