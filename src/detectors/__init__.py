"""Detector implementations.

Importing this package registers every bundled detector (via ``@register``) so the
benchmark CLI can resolve them by name. Add a method as ``src/detectors/<name>.py``
with a ``@register(...)`` class and an import below.

The four methods here are the ones benchmarked in the paper (Table 2):
``maad_roi``, ``pcen_peak``, ``spectral_entropy``, ``energy_template``.
"""

from src.detectors import pcen_peak  # noqa: F401  (import for side-effect: registers)
from src.detectors import maad_roi  # noqa: F401  (import for side-effect: registers)
from src.detectors import energy_template  # noqa: F401  (import for side-effect: registers)
from src.detectors import spectral_entropy  # noqa: F401  (import for side-effect: registers)

__all__ = [
    "pcen_peak",
    "maad_roi",
    "energy_template",
    "spectral_entropy",
]
