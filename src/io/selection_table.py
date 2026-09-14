"""Read/write events as tab-separated selection tables.

The on-disk format is a Raven-readable selection table, but **time-only**: one row
per event with columns

    file_id    Begin Time (s)    End Time (s)    score    detector

Frequency columns that Raven also understands are simply omitted. The reader is
tolerant of column-name aliases (``onset``/``start``, ``offset``/``end``,
``filename``) so the same function can ingest ground-truth tables that other tools
produced; see :func:`read_selection_table`.
"""

from __future__ import annotations

import csv
from collections import defaultdict

from src.core.types import Detections, Event

# Canonical output column names.
COL_FILE = "file_id"
COL_BEGIN = "Begin Time (s)"
COL_END = "End Time (s)"
COL_SCORE = "score"
COL_DETECTOR = "detector"
FIELDS = [COL_FILE, COL_BEGIN, COL_END, COL_SCORE, COL_DETECTOR]

# Accepted aliases (lower-cased) when *reading* foreign tables.
_BEGIN_ALIASES = {"begin time (s)", "onset", "onset_s", "start", "start_s", "begin"}
_END_ALIASES = {"end time (s)", "offset", "offset_s", "end", "end_s", "stop"}
_FILE_ALIASES = {"file_id", "filename", "file", "fname", "recording", "begin file"}
_SCORE_ALIASES = {"score", "confidence", "conf", "probability", "prob"}


def write_selection_table(path: str, detections: list[Detections]) -> None:
    """Write the events from many :class:`Detections` to one TSV at ``path``."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        for det in detections:
            for ev in det.events:
                w.writerow(
                    {
                        COL_FILE: det.file_id,
                        COL_BEGIN: f"{ev.onset_s:.6f}",
                        COL_END: f"{ev.offset_s:.6f}",
                        COL_SCORE: f"{ev.score:.6f}",
                        COL_DETECTOR: det.detector,
                    }
                )


def _resolve(header: list[str], aliases: set[str]) -> str | None:
    """Return the actual header name matching any alias (case-insensitive)."""
    for h in header:
        if h.strip().lower() in aliases:
            return h
    return None


def read_selection_table(
    path: str, default_file_id: str | None = None
) -> dict[str, list[Event]]:
    """Read a selection table into ``{file_id: [Event, ...]}``.

    Onset/offset columns are required (via their canonical names or aliases).
    ``score`` defaults to ``1.0`` when absent (e.g. ground-truth tables). When no
    file column is present, every event is filed under ``default_file_id`` (or the
    table's basename if that is ``None``).
    """
    import os

    with open(path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        header = reader.fieldnames or []
        c_begin = _resolve(header, _BEGIN_ALIASES)
        c_end = _resolve(header, _END_ALIASES)
        c_file = _resolve(header, _FILE_ALIASES)
        c_score = _resolve(header, _SCORE_ALIASES)
        if c_begin is None or c_end is None:
            raise ValueError(
                f"{path}: could not find onset/offset columns in header {header!r}"
            )

        fallback = default_file_id or os.path.basename(path)
        out: dict[str, list[Event]] = defaultdict(list)
        for row in reader:
            onset = float(row[c_begin])
            offset = float(row[c_end])
            score = float(row[c_score]) if c_score and row.get(c_score) else 1.0
            fid = (row.get(c_file) or "").strip() if c_file else ""
            fid = fid or fallback
            out[fid].append(Event(onset_s=onset, offset_s=offset, score=score))
    return dict(out)
