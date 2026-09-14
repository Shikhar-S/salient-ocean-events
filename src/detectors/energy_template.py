"""Energy-candidate + spectrogram self-similarity detector (label-free).

Pipeline::

    Stage 1 (where):  audio ─► log-mel ─► broadband energy ─► salience_to_events ─► candidate spans
                                                                                          │
    Stage 2 (how      per candidate: mel patch ─► resample to fixed width ─► z + L2 norm  │
       confident):                                                                        ▼
                      NCC self-similarity vs other candidates ─► mean top-k ─► score ─► events

Two stages. Stage 1 finds *where*: peaks in a broadband log-mel energy envelope become
coarse candidate spans (a low threshold keeps recall high). Stage 2 scores *how
confident*: with no labelled template, confidence comes from recurrence -- a candidate
resembling other candidates in the same clip (a stereotyped, repeated call) scores higher
than a one-off. The recurrence score is a spectrogram self-similarity NCC: each
candidate's log-mel patch is time-resampled to a fixed width, z- and L2-normalized (so NCC
= dot product), and scored by the mean of its top-``topk`` NCCs to the others.
"""

from __future__ import annotations

import librosa
import numpy as np

from src.core.detector import Detector, register
from src.core.types import Detections, Event, Salience
from src.detectors._common import salience_to_events


@register("energy_template")
class EnergyTemplateDetector(Detector):
    """Energy candidates re-scored by spectrogram self-similarity (NCC)."""

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        k: float = 0.25,
        min_dur_s: float = 0.05,
        merge_gap_s: float = 0.05,
        patch_w: int = 32,
        topk: int = 3,
    ) -> None:
        super().__init__(
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            k=k,
            min_dur_s=min_dur_s,
            merge_gap_s=merge_gap_s,
            patch_w=patch_w,
            topk=topk,
        )
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.k = k
        self.min_dur_s = min_dur_s
        self.merge_gap_s = merge_gap_s
        self.patch_w = patch_w
        self.topk = topk

    def _mel(
        self, audio: np.ndarray, sr: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return ``(times, energy_db, log_mel)`` frame arrays.

        ``energy_db`` is the 1-D broadband envelope (the salience curve);
        ``log_mel`` is the 2-D ``ref=np.max`` dB spectrogram used for patches.
        """
        audio = np.asarray(audio, dtype=np.float32)
        if audio.size == 0:
            return np.zeros(0), np.zeros(0), np.zeros((self.n_mels, 0))
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            power=2.0,
        )
        energy_db = librosa.power_to_db(mel.sum(axis=0) + 1e-12)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        times = librosa.frames_to_time(
            np.arange(energy_db.shape[0]), sr=sr, hop_length=self.hop_length
        )
        return times, energy_db, log_mel

    def salience(self, audio: np.ndarray, sr: int) -> Salience:
        times, energy_db, _ = self._mel(audio, sr)
        return Salience(times_s=times, values=energy_db)

    def _patch_feature(self, patch: np.ndarray) -> np.ndarray:
        """Resample a ``(n_mels, t)`` patch to ``patch_w`` cols -> a unit vector.

        Time-resample by linear interpolation so patches of different durations
        are comparable, then z-normalize and L2-normalize so the NCC between two
        features is their dot product.
        """
        n_mels, t = patch.shape
        w = self.patch_w
        if t == 1:
            resampled = np.repeat(patch, w, axis=1)
        else:
            src = np.linspace(0.0, 1.0, t)
            dst = np.linspace(0.0, 1.0, w)
            resampled = np.empty((n_mels, w), dtype=np.float64)
            for r in range(n_mels):
                resampled[r] = np.interp(dst, src, patch[r])
        vec = resampled.reshape(-1).astype(np.float64)
        vec = vec - vec.mean()
        std = vec.std()
        if std > 1e-12:
            vec = vec / std
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        return vec

    def detect(self, audio: np.ndarray, sr: int) -> Detections:
        times, energy_db, log_mel = self._mel(audio, sr)
        sal = Salience(times_s=times, values=energy_db)
        if times.size == 0:
            return self._pack("", [], sal)

        # Stage 1: coarse energy candidates (generous, high recall).
        cands = salience_to_events(
            times,
            energy_db,
            k=self.k,
            min_dur_s=self.min_dur_s,
            merge_gap_s=self.merge_gap_s,
        )
        if not cands:
            return self._pack("", [], sal)

        # Stage 2: a self-similarity patch feature per candidate span.
        dt = float(np.median(np.diff(times))) if times.size > 1 else 0.0
        n_frames = log_mel.shape[1]
        feats: list[np.ndarray] = []
        for ev in cands:
            a = int(np.searchsorted(times, ev.onset_s, side="left")) if dt > 0 else 0
            b = int(np.searchsorted(times, ev.offset_s, side="right")) if dt > 0 else n_frames
            a = max(0, min(a, n_frames - 1))
            b = max(a + 1, min(b, n_frames))
            feats.append(self._patch_feature(log_mel[:, a:b]))

        scores = self._recurrence_scores(feats)
        events = [
            Event(onset_s=ev.onset_s, offset_s=ev.offset_s, score=float(s))
            for ev, s in zip(cands, scores)
        ]
        return self._pack("", events, sal)

    def _recurrence_scores(self, feats: list[np.ndarray]) -> list[float]:
        """Self-similarity NCC score per candidate.

        With >=2 candidates, score = mean of the top-``topk`` NCCs to the *other*
        candidates (NCC = dot of L2-normalized z-scored features). A lone
        candidate has no peer, so it scores ``1.0`` (single-event clips still
        emit a positive-scored event).
        """
        n = len(feats)
        if n == 1:
            return [1.0]
        sim = np.vstack(feats) @ np.vstack(feats).T  # pairwise NCC in [-1, 1]
        np.fill_diagonal(sim, -np.inf)  # exclude self
        kk = max(1, min(self.topk, n - 1))
        scores: list[float] = []
        for i in range(n):
            top = np.sort(sim[i])[::-1][:kk]
            top = top[np.isfinite(top)]
            scores.append(float(top.mean()) if top.size else 0.0)
        return scores
