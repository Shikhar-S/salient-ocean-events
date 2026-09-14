"""scikit-maad 2-D ROI detector (label-free).

Pipeline::

    audio ─► power spectrogram ─► dB + smooth ─► per-freq noise-floor subtract
          ─► relative double-threshold mask ─► label + area-select ROIs
          ─► bbox → [min_t, max_t)  (score = summed box power)
          ─► merge overlapping spans (max score) ─► events

Treats the spectrogram as an image and finds connected regions of interest in
time-frequency: power spectrogram -> dB + smooth -> per-frequency noise-floor
subtraction -> relative double-threshold mask (:func:`maad.rois.create_mask`) -> label
+ area-select (:func:`maad.rois.select_rois`) -> bbox in s/Hz. Each ROI collapses to its
span ``[min_t, max_t)``, scored by summed box power; overlapping spans merge.

The per-frequency noise-floor subtraction is the key step: without it a stationary
broadband band (rumble/hiss) is the strongest "feature" and forms one component spanning
the whole clip, collapsing every recording to a single whole-file ROI. ``dither_db`` only
matters for synthetic near-silent clips (it gives the relative threshold a defined
background); on real recordings it is ~a no-op.
"""

from __future__ import annotations

import numpy as np

from maad import rois, sound, util

from src.core.detector import Detector, register
from src.core.types import Detections, Event


@register("maad_roi")
class MaadRoiDetector(Detector):
    """scikit-maad relative-threshold ROI detection, collapsed to time spans."""

    def __init__(
        self,
        nperseg: int = 1024,
        noverlap: int = 512,
        smooth_std: float = 1.0,
        subtract_noise_floor: bool = True,
        dither_db: float = 1.0,
        bin_std: float = 4.0,
        bin_per: float = 0.5,
        min_roi: int = 25,
        max_roi: int | None = None,
        min_dur_s: float = 0.05,
        merge_gap_s: float = 0.05,
    ) -> None:
        super().__init__(
            nperseg=nperseg,
            noverlap=noverlap,
            smooth_std=smooth_std,
            subtract_noise_floor=subtract_noise_floor,
            dither_db=dither_db,
            bin_std=bin_std,
            bin_per=bin_per,
            min_roi=min_roi,
            max_roi=max_roi,
            min_dur_s=min_dur_s,
            merge_gap_s=merge_gap_s,
        )
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.smooth_std = smooth_std
        self.subtract_noise_floor = subtract_noise_floor
        self.dither_db = dither_db
        self.bin_std = bin_std
        self.bin_per = bin_per
        self.min_roi = min_roi
        self.max_roi = max_roi
        self.min_dur_s = min_dur_s
        self.merge_gap_s = merge_gap_s

    def detect(self, audio: np.ndarray, sr: int) -> Detections:
        audio = np.asarray(audio, dtype=np.float32)
        # Too short for a single STFT frame -> no events (maad would raise).
        if audio.size < self.nperseg:
            return self._pack("", [], salience=None)

        # 1. Power spectrogram: Sxx (freq x time), tn (s), fn (Hz).
        Sxx, tn, fn, _ext = sound.spectrogram(
            audio, sr, nperseg=self.nperseg, noverlap=self.noverlap
        )
        if Sxx.size == 0 or tn.size == 0:
            return self._pack("", [], salience=None)

        # 2. dB-scale and lightly smooth the spectrogram image.
        Sxx_db = util.power2dB(Sxx, db_range=96) + 96
        if self.smooth_std and self.smooth_std > 0:
            Sxx_db = sound.smooth(Sxx_db, std=self.smooth_std)

        # 2b. Subtract each frequency bin's temporal median to remove the
        # stationary broadband floor (see module docstring). Before the dither so
        # the floor is computed on real energy.
        if self.subtract_noise_floor and Sxx_db.shape[1] > 1:
            Sxx_db = Sxx_db - np.median(Sxx_db, axis=1, keepdims=True)
            Sxx_db = np.clip(Sxx_db, 0.0, None)

        # A small deterministic dither floor for near-silent (synthetic) clips.
        if self.dither_db and self.dither_db > 0:
            rng = np.random.default_rng(0)
            Sxx_db = Sxx_db + self.dither_db * np.abs(
                rng.standard_normal(Sxx_db.shape)
            )

        # 3. Relative double-threshold mask (seed + connected grow).
        im_mask = rois.create_mask(
            Sxx_db,
            mode_bin="relative",
            bin_std=self.bin_std,
            bin_per=self.bin_per,
        )

        # 4. Label + area-select ROIs.
        _im_rois, df_rois = rois.select_rois(
            im_mask, min_roi=self.min_roi, max_roi=self.max_roi
        )
        if df_rois is None or len(df_rois) == 0:
            return self._pack("", [], salience=None)

        # 5. Add time/frequency columns (min_t/max_t in s, min_f/max_f in Hz) while
        #    keeping the pixel bbox (min_y/min_x/max_y/max_x) for energy scoring.
        df_rois = util.format_features(df_rois, tn, fn)

        spans: list[list[float]] = []  # [onset_s, offset_s, score]
        for _, r in df_rois.iterrows():
            # Score = summed power inside the ROI's time-frequency bounding box.
            score = float(
                Sxx[int(r["min_y"]) : int(r["max_y"]) + 1,
                    int(r["min_x"]) : int(r["max_x"]) + 1].sum()
            )
            spans.append([float(r["min_t"]), float(r["max_t"]), score])

        # Merge overlapping / near-touching time spans, keeping the max score.
        spans.sort(key=lambda s: s[0])
        merged: list[list[float]] = []
        for s in spans:
            if merged and s[0] - merged[-1][1] <= self.merge_gap_s:
                merged[-1][1] = max(merged[-1][1], s[1])
                merged[-1][2] = max(merged[-1][2], s[2])
            else:
                merged.append(list(s))

        events = [
            Event(onset_s=on, offset_s=off, score=sc)
            for on, off, sc in merged
            if off - on >= self.min_dur_s
        ]
        return self._pack("", events, salience=None)
