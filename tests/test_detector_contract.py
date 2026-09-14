"""Contract tests that sweep *every* registered detector.

Per-detector test files assert each method's specific behaviour; this file
asserts the invariants the runner and evaluator rely on, uniformly across the
whole :data:`~src.core.detector.REGISTRY`, so a new detector that breaks the
contract fails here without needing a bespoke test. The contract (see
``src/core/detector.py``):

* ``detect(audio, sr)`` returns a :class:`Detections` stamped with the detector's
  own ``name``/``params`` (provenance) and with ``file_id == ""`` -- the *runner*
  fills the id, never the detector;
* events are **time-only**, well-ordered ``onset <= offset``, in-bounds, finite
  score, and sorted (the :class:`Detections` constructor sorts);
* empty audio is handled gracefully (no crash, no events) -- the runner can hand a
  detector a zero-length chunk;
* the optional dense ``salience`` is either ``None`` or parallel 1-D arrays.
"""

import importlib

import numpy as np
import pytest

import src.detectors  # noqa: F401  registers all bundled detectors
from src.core.detector import REGISTRY, get_detector
from src.core.types import Detections, Salience

SR = 48000
DUR_S = 6.0

# The four methods benchmarked in the paper (Table 2). Guards the import
# side-effect wiring in src/detectors/__init__.py against a forgotten line.
EXPECTED = {
    "maad_roi",
    "pcen_peak",
    "spectral_entropy",
    "energy_template",
}


def _clip(seed: int = 0) -> np.ndarray:
    """A 6 s clip: a 4 kHz tonal call (1.0-1.8 s) + a broadband burst (3.5-4.0 s).

    Carries both a tonal and a broadband event so structure- and energy-based
    detectors alike have something to fire on (the contract checks shape, not
    which events are found).
    """
    n = int(DUR_S * SR)
    x = np.zeros(n, dtype=np.float32)
    rng = np.random.default_rng(seed)
    a, b = int(1.0 * SR), int(1.8 * SR)
    t = np.arange(b - a) / SR
    x[a:b] = 0.5 * np.sin(2 * np.pi * 4000 * t).astype(np.float32)
    a2, b2 = int(3.5 * SR), int(4.0 * SR)
    x[a2:b2] = 0.5 * rng.standard_normal(b2 - a2).astype(np.float32)
    return x


def test_all_expected_detectors_registered():
    assert EXPECTED <= set(REGISTRY), f"missing: {EXPECTED - set(REGISTRY)}"


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_detect_returns_well_formed_detections(name):
    det = get_detector(name)
    out = det.detect(_clip(), SR)

    assert isinstance(out, Detections)
    # Provenance is stamped by the detector...
    assert out.detector == name
    assert out.params == det.params
    # ...but the file_id is the runner's job, never the detector's.
    assert out.file_id == ""

    onsets = [e.onset_s for e in out.events]
    assert onsets == sorted(onsets), "events must be sorted by onset"
    for e in out.events:
        assert e.onset_s <= e.offset_s
        assert e.onset_s >= -1e-6
        # In-bounds up to a frame's slack (converters pad by half a hop).
        assert e.offset_s <= DUR_S + 0.05
        assert np.isfinite(e.score)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_empty_audio_is_safe(name):
    det = get_detector(name)
    out = det.detect(np.zeros(0, dtype=np.float32), SR)
    assert isinstance(out, Detections)
    assert out.events == []


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_salience_is_none_or_parallel(name):
    det = get_detector(name)
    sal = det.salience(_clip(), SR)
    if sal is not None:
        assert isinstance(sal, Salience)
        assert sal.times_s.ndim == 1
        assert sal.times_s.shape == sal.values.shape


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_module_has_docstring_and_diagram(name):
    """Each detector module documents itself with a prose docstring + a diagram."""
    module = importlib.import_module(REGISTRY[name].__module__)
    doc = module.__doc__ or ""
    assert doc.strip(), f"{name}: module docstring is empty"
    assert "─►" in doc, f"{name}: module docstring is missing its pipeline diagram"
