"""PCEN-based salient event detector with adaptive peak-picking (label-free).

Pipeline::

    audio ─► mel (power) ─► PCEN (low gain, long τ → tracks recording-level floor)
          ─► Σ bands ─► activity curve ─► salience_to_events ─► events

Per-Channel Energy Normalization divides each mel band by a smoothed estimate of its
own recent energy, suppressing stationary background; summed across bands it gives a
1-D activity curve. The defaults are gentler than PCEN's usual transient-sharpening
(``gain=0.98, time_constant=0.06``): a low ``gain`` and long ``time_constant`` (>= any
event) make the AGC track the recording-level background rather than adapting *within*
a call, so a sustained tonal call stays elevated for its whole duration instead of
collapsing to a sliver near the onset. ``time_constant`` is frame-rate dependent.
"""

from __future__ import annotations

import librosa
import numpy as np

from src.core.detector import Detector, register
from src.core.types import Detections, Salience
from src.detectors._common import salience_to_events


@register("pcen_peak")
class PcenPeakDetector(Detector):
    """Mel-PCEN activity curve + adaptive span extraction."""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        gain: float = 0.3,
        bias: float = 2.0,
        power: float = 0.5,
        time_constant: float = 1.0,
        k: float = 1.0,
        min_dur_s: float = 0.05,
        merge_gap_s: float = 0.3,
    ) -> None:
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            gain=gain,
            bias=bias,
            power=power,
            time_constant=time_constant,
            k=k,
            min_dur_s=min_dur_s,
            merge_gap_s=merge_gap_s,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.gain = gain
        self.bias = bias
        self.power = power
        self.time_constant = time_constant
        self.k = k
        self.min_dur_s = min_dur_s
        self.merge_gap_s = merge_gap_s

    def _curve(self, audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        """Return parallel ``(times, pcen_activity)`` frame arrays."""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return np.zeros(0), np.zeros(0)
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        # librosa.pcen expects magnitudes on roughly the scale of a 16-bit STFT;
        # scaling the power spectrogram by 2**31 puts it in that regime.
        pcen = librosa.pcen(
            mel * (2**31),
            sr=sr,
            hop_length=self.hop_length,
            gain=self.gain,
            bias=self.bias,
            power=self.power,
            time_constant=self.time_constant,
        )
        activity = pcen.sum(axis=0)
        times = librosa.frames_to_time(
            np.arange(activity.shape[0]), sr=sr, hop_length=self.hop_length
        )
        return times, activity

    def salience(self, audio: np.ndarray, sr: int) -> Salience:
        times, activity = self._curve(audio, sr)
        return Salience(times_s=times, values=activity)

    def detect(self, audio: np.ndarray, sr: int) -> Detections:
        times, activity = self._curve(audio, sr)
        events = salience_to_events(
            times,
            activity,
            k=self.k,
            min_dur_s=self.min_dur_s,
            merge_gap_s=self.merge_gap_s,
        )
        return self._pack("", events, Salience(times_s=times, values=activity))
