"""End-to-end smoke test: synthetic audio + GT table through the runner.

The runner is exercised with a trivial frame-RMS detector defined *here* rather
than a shipped one: the runner contract (chunking, seam merging, table writing) is
what is under test, and a deterministic local stub keeps that independent of any
real method's tuning. It is deliberately **not** ``@register``-ed, so the shipped
:data:`~src.core.detector.REGISTRY` stays exactly the four benchmarked methods.
"""

import os

import numpy as np
import soundfile as sf

from src.core.detector import Detector
from src.core.types import Detections, Event, Salience
from src.eval.matching import CollarMatcher
from src.eval.benchmark import run_benchmark, detect_file


class StubDetector(Detector):
    """Frame-wise RMS thresholding — a test fixture, not a benchmark method."""

    name = "stub"

    def __init__(self, win_s=0.05, hop_s=0.025, rel_threshold=0.5):
        super().__init__(win_s=win_s, hop_s=hop_s, rel_threshold=rel_threshold)
        self.win_s, self.hop_s, self.rel_threshold = win_s, hop_s, rel_threshold

    def _rms_frames(self, audio, sr):
        win = max(1, int(round(self.win_s * sr)))
        hop = max(1, int(round(self.hop_s * sr)))
        if len(audio) < win:
            audio = np.pad(audio, (0, win - len(audio)))
        n_frames = 1 + (len(audio) - win) // hop
        rms = np.empty(n_frames, dtype=np.float64)
        times = np.empty(n_frames, dtype=np.float64)
        for k in range(n_frames):
            start = k * hop
            frame = audio[start : start + win].astype(np.float64)
            rms[k] = np.sqrt(np.mean(frame * frame)) if frame.size else 0.0
            times[k] = (start + win / 2) / sr
        return times, rms

    def salience(self, audio, sr) -> Salience:
        times, rms = self._rms_frames(audio, sr)
        return Salience(times_s=times, values=rms)

    def detect(self, audio, sr) -> Detections:
        times, rms = self._rms_frames(audio, sr)
        file_id = ""  # set by the runner; left blank for direct calls
        if rms.size == 0 or rms.max() <= 0.0:
            return self._pack(file_id, [], Salience(times_s=times, values=rms))

        thr = self.rel_threshold * float(rms.max())
        above = rms >= thr
        hop = max(1, int(round(self.hop_s * sr)))
        win = max(1, int(round(self.win_s * sr)))

        events: list[Event] = []
        k, n = 0, len(above)
        while k < n:
            if not above[k]:
                k += 1
                continue
            j = k
            while j < n and above[j]:
                j += 1
            # Frames [k, j) form one event.
            events.append(
                Event(
                    onset_s=(k * hop) / sr,
                    offset_s=((j - 1) * hop + win) / sr,
                    score=float(rms[k:j].mean()),
                )
            )
            k = j

        return self._pack(file_id, events, Salience(times_s=times, values=rms))


def _make_wav(path, sr=16000, bursts=((1.0, 1.4), (3.0, 3.5))):
    """A silent clip with white-noise bursts at the given (onset, offset) seconds."""
    dur = 5.0
    x = np.zeros(int(dur * sr), dtype=np.float32)
    rng = np.random.default_rng(0)
    for on, off in bursts:
        a, b = int(on * sr), int(off * sr)
        x[a:b] = 0.5 * rng.standard_normal(b - a).astype(np.float32)
    sf.write(path, x, sr)
    return bursts


def test_stub_detects_bursts():
    detector = StubDetector()
    sr = 16000
    x = np.zeros(int(5 * sr), dtype=np.float32)
    rng = np.random.default_rng(1)
    x[int(1.0 * sr) : int(1.4 * sr)] = 0.5 * rng.standard_normal(int(0.4 * sr)).astype(np.float32)
    det = detector.detect(x, sr)
    assert len(det.events) >= 1
    # The detected event should overlap the [1.0, 1.4] burst.
    assert any(e.onset_s < 1.4 and e.offset_s > 1.0 for e in det.events)
    # Salience function is exposed and parallel.
    sal = detector.salience(x, sr)
    assert sal is not None and sal.times_s.shape == sal.values.shape


def test_run_benchmark_end_to_end(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()
    wav = audio_dir / "clip.wav"
    bursts = _make_wav(str(wav))

    files = {"clip": str(wav)}
    gt_by_file = {"clip": [Event(on, off) for on, off in bursts]}

    out_dir = tmp_path / "out"
    rows = run_benchmark(
        detectors=[StubDetector()],
        files=files,
        gt_by_file=gt_by_file,
        matchers=[CollarMatcher(onset_collar=0.3, offset_collar=0.3)],
        out_dir=str(out_dir),
    )

    assert len(rows) == 1
    r = rows[0].result
    assert r.n_gt == 2
    assert r.tp >= 1  # finds at least one burst
    # Artifacts written.
    assert os.path.exists(out_dir / "results.tsv")
    assert os.path.exists(out_dir / "preds_stub.tsv")


def test_chunked_matches_unchunked():
    # Chunking with seam-merge should reproduce roughly the same event count.
    sr = 16000
    detector = StubDetector()
    x = np.zeros(int(6 * sr), dtype=np.float32)
    rng = np.random.default_rng(2)
    for on in (0.5, 2.5, 4.5):
        x[int(on * sr) : int((on + 0.3) * sr)] = 0.5 * rng.standard_normal(int(0.3 * sr)).astype(np.float32)

    full = detect_file(detector, x, sr, "c")
    chunked = detect_file(detector, x, sr, "c", chunk_s=2.0, overlap_s=0.5)
    assert abs(len(full.events) - len(chunked.events)) <= 1
