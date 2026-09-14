"""Core data types and the detector interface shared by every method."""

from src.core.types import Event, Detections, Salience
from src.core.detector import Detector, register, REGISTRY, get_detector

__all__ = [
    "Event",
    "Detections",
    "Salience",
    "Detector",
    "register",
    "REGISTRY",
    "get_detector",
]
