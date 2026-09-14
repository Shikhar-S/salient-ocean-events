"""Tonality (spectral-flatness) detector (label-free).

Pipeline::

    audio ─► STFT magnitude ─► spectral_flatness ─► tonality = 1 − flatness
          ─► × (rms / max rms)  [optional energy gate] ─► salience_to_events ─► events

Scores per-frame tonality ``1 - spectral_flatness``: narrowband vocalizations
(whistles, moans, tonal calls) are peaky in frequency -> low flatness -> high
salience; broadband noise and silence are flat -> low salience. The optional energy
gate makes the response "tonal *and* present". By construction it under-responds to
broadband click events.
"""

from __future__ import annotations

import librosa
import numpy as np

from src.core.detector import Detector, register
from src.core.types import Detections, Salience
from src.detectors._common import salience_to_events


@register("spectral_entropy")
class SpectralEntropyDetector(Detector):
    """Per-frame tonality (1 - spectral flatness) -> scored event spans."""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        energy_weight: bool = True,
        k: float = 1.5,
        min_dur_s: float = 0.05,
        merge_gap_s: float = 0.1,
    ) -> None:
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            energy_weight=energy_weight,
            k=k,
            min_dur_s=min_dur_s,
            merge_gap_s=merge_gap_s,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.energy_weight = energy_weight
        self.k = k
        self.min_dur_s = min_dur_s
        self.merge_gap_s = merge_gap_s

    def _curve(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return np.zeros(0), np.zeros(0)
        # One STFT magnitude shared by the librosa flatness/rms primitives.
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))
        flatness = librosa.feature.spectral_flatness(S=S)[0]  # (t,), ~1 noise / ~0 tonal
        tonality = 1.0 - flatness
        if self.energy_weight:
            rms = librosa.feature.rms(S=S, frame_length=self.n_fft, hop_length=self.hop_length)[0]
            tonality = tonality * (rms / (rms.max() + 1e-12))
        times = librosa.frames_to_time(
            np.arange(tonality.shape[0]), sr=sr, hop_length=self.hop_length
        )
        return times, tonality

    def salience(self, audio: np.ndarray, sr: int) -> Salience:
        times, tonality = self._curve(audio, sr)
        return Salience(times_s=times, values=tonality)

    def detect(self, audio: np.ndarray, sr: int) -> Detections:
        times, tonality = self._curve(audio, sr)
        events = salience_to_events(
            times,
            tonality,
            k=self.k,
            min_dur_s=self.min_dur_s,
            merge_gap_s=self.merge_gap_s,
        )
        return self._pack("", events, Salience(times_s=times, values=tonality))
