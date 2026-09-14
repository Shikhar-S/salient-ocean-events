"""Orchestration: run detectors over a set of audio files and score them.

The runner owns everything around a detector: loading/resampling audio, optional
long-file chunking with overlap and seam de-duplication, stamping the ``file_id``,
writing predictions as selection tables, and evaluating against ground truth under
each matcher. Detectors themselves only ever see a single waveform array.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from src.core.detector import Detector
from src.core.types import Detections, Event
from src.eval.matching import Matcher
from src.eval.metrics import EvalResult, evaluate
from src.io.audio import load_mono_resampled
from src.io.selection_table import write_selection_table


@dataclass
class BenchmarkRow:
    """One ``(detector, matcher)`` row of the results table."""

    detector: str
    result: EvalResult


def _merge_events(events: list[Event], gap_s: float = 0.0) -> list[Event]:
    """Merge overlapping / near-touching events into one (keeps the max score).

    Used to stitch detections back together across chunk seams.
    """
    if not events:
        return []
    ordered = sorted(events, key=lambda e: e.onset_s)
    merged = [ordered[0]]
    for e in ordered[1:]:
        last = merged[-1]
        if e.onset_s <= last.offset_s + gap_s:
            merged[-1] = Event(
                onset_s=last.onset_s,
                offset_s=max(last.offset_s, e.offset_s),
                score=max(last.score, e.score),
            )
        else:
            merged.append(e)
    return merged


def detect_file(
    detector: Detector,
    audio,
    sr: int,
    file_id: str,
    chunk_s: float | None = None,
    overlap_s: float = 0.0,
) -> Detections:
    """Run ``detector`` on one already-loaded waveform, with optional chunking."""
    if chunk_s is None or len(audio) <= int(chunk_s * sr):
        det = detector.detect(audio, sr)
        det.file_id = file_id
        return det

    win = int(chunk_s * sr)
    hop = max(1, win - int(overlap_s * sr))
    all_events: list[Event] = []
    start = 0
    while start < len(audio):
        chunk = audio[start : start + win]
        sub = detector.detect(chunk, sr)
        offset = start / sr
        for e in sub.events:
            all_events.append(
                Event(e.onset_s + offset, e.offset_s + offset, e.score)
            )
        if start + win >= len(audio):
            break
        start += hop

    merged = _merge_events(all_events, gap_s=overlap_s)
    return Detections(
        file_id=file_id,
        events=merged,
        detector=detector.name,
        params=dict(detector.params),
    )


def run_detector(
    detector: Detector,
    files: dict[str, str],
    target_sr: int | None = None,
    chunk_s: float | None = None,
    overlap_s: float = 0.0,
) -> list[Detections]:
    """Run one detector over ``{file_id: path}``; returns one Detections per file."""
    out: list[Detections] = []
    for file_id, path in files.items():
        audio, sr = load_mono_resampled(path, target_sr)
        out.append(detect_file(detector, audio, sr, file_id, chunk_s, overlap_s))
    return out


def predictions_by_file(detections: list[Detections]) -> dict[str, list[Event]]:
    """Flatten a detector's per-file Detections into ``{file_id: [Event, ...]}``."""
    return {d.file_id: d.events for d in detections}


def _clip_durations(files: dict[str, str]) -> dict[str, float]:
    """Map ``file_id -> duration (s)`` from audio headers (no full decode)."""
    import soundfile as sf

    durations: dict[str, float] = {}
    for file_id, path in files.items():
        info = sf.info(path)
        durations[file_id] = info.frames / info.samplerate if info.samplerate else 0.0
    return durations


def run_benchmark(
    detectors: list[Detector],
    files: dict[str, str],
    gt_by_file: dict[str, list[Event]],
    matchers: list[Matcher],
    out_dir: str | None = None,
    target_sr: int | None = None,
    chunk_s: float | None = None,
    overlap_s: float = 0.0,
) -> list[BenchmarkRow]:
    """Run every detector over every file and score against ground truth.

    When ``out_dir`` is given, each detector's predictions are written to
    ``preds_<name>.tsv`` and the aggregated scores to ``results.tsv`` there.
    Returns the per-``(detector, matcher)`` rows.
    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    # Clip durations (seconds) for the label-free selection-rate / over-firing
    # guards; read from headers only, and invariant to any target_sr resampling.
    durations = _clip_durations(files)

    rows: list[BenchmarkRow] = []
    for detector in detectors:
        detections = run_detector(detector, files, target_sr, chunk_s, overlap_s)
        if out_dir:
            write_selection_table(
                os.path.join(out_dir, f"preds_{detector.name}.tsv"), detections
            )
        pred_by_file = predictions_by_file(detections)
        for matcher in matchers:
            result = evaluate(pred_by_file, gt_by_file, matcher, durations=durations)
            rows.append(BenchmarkRow(detector=detector.name, result=result))

    if out_dir:
        write_results_table(os.path.join(out_dir, "results.tsv"), rows)
    return rows


def write_results_table(path: str, rows: list[BenchmarkRow]) -> None:
    """Write the aggregated benchmark results as a TSV."""
    import csv

    fields = [
        "detector", "matcher", "n_pred", "n_gt", "ap", "max_f1",
        "precision_at_max_f1", "recall_at_max_f1", "threshold_at_max_f1",
        "precision", "recall", "f1", "macro_f1", "tp", "fp", "fn",
        "selection_rate", "recall_at_sel_budget", "auc_recall_sel",
        "op_threshold", "op_recall", "op_selection",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for row in rows:
            r = row.result
            w.writerow(
                {
                    "detector": row.detector,
                    "matcher": r.matcher,
                    "n_pred": r.n_pred,
                    "n_gt": r.n_gt,
                    "ap": f"{r.ap:.6f}",
                    "max_f1": f"{r.max_f1:.6f}",
                    "precision_at_max_f1": f"{r.precision_at_max_f1:.6f}",
                    "recall_at_max_f1": f"{r.recall_at_max_f1:.6f}",
                    "threshold_at_max_f1": f"{r.threshold_at_max_f1:.6g}",
                    "precision": f"{r.precision:.6f}",
                    "recall": f"{r.recall:.6f}",
                    "f1": f"{r.f1:.6f}",
                    "macro_f1": f"{r.macro_f1:.6f}",
                    "tp": r.tp,
                    "fp": r.fp,
                    "fn": r.fn,
                    "selection_rate": f"{r.selection_rate:.6f}",
                    "recall_at_sel_budget": f"{r.recall_at_sel_budget:.6f}",
                    "auc_recall_sel": f"{r.auc_recall_sel:.6f}",
                    "op_threshold": f"{r.op_threshold:.6g}",
                    "op_recall": f"{r.op_recall:.6f}",
                    "op_selection": f"{r.op_selection:.6f}",
                }
            )
