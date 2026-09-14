#!/usr/bin/env python3
"""Annotate long recordings with a detector to build an OpenBEATs pretraining corpus.

This is the export counterpart to :mod:`src.run_benchmark` — same detectors, but no
ground truth. Because ``maad_roi`` (the only method that generalized in the
benchmark) *saturates* on continuous master tapes — it marks ~all of the timeline
as salient — a raw event list is not selective. Instead we score the recording on a
fixed grid and keep only the most salient windows:

  ``score``   tile each recording into ``--window-s`` windows and run the detector
              **independently on each window** (exactly how it was validated on
              short clips); write a scored-grid TSV
              ``seg_id wav_id start end n_events max_score sum_score``.
  ``select``  read the (merged) scored grid and keep the top ``--keep-frac`` most
              salient windows, writing a Kaldi ``segments`` file
              (``<wav_id>-<idx> <wav_id> <start> <end>``) + a score sidecar.

The two phases split so ``score`` shards cleanly across an array job and ``select``
applies one global ranking. The ``segments`` output is consumed verbatim by
OpenBEATs' ``recipes/watkins/prepare_manifest.py`` into a JSONL manifest.

Example
-------
    uv run python -m src.annotate score  --wav-scp master_tapes/wav.scp --out grid.tsv
    uv run python -m src.annotate select --scored grid.tsv --out-segments segments \
        --keep-frac 0.5
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import src.detectors  # noqa: F401  registers all bundled detectors
from src.core.detector import Detector, get_detector
from src.io.audio import load_mono_resampled

SCORE_COLS = ("n_events", "max_score", "sum_score")


def read_wav_scp(path: str) -> list[tuple[str, str]]:
    """Parse Kaldi ``wav.scp`` (``<wav_id> <path>``) preserving order.

    Splits on the FIRST whitespace only, so paths may contain spaces
    (e.g. ``.../70017ch2 44kHz.wav``).
    """
    rows: list[tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            wid, p = line.split(None, 1)
            rows.append((wid, p))
    return rows


def grid_windows(duration_s: float, window_s: float) -> list[tuple[float, float]]:
    """Consecutive non-overlapping ``window_s`` windows over ``[0, duration_s)``.

    The final sub-window shorter than ``window_s`` is dropped (negligible over long
    recordings), so every span is exactly ``window_s`` — except a file shorter than
    one window, which yields a single whole-file span.
    """
    if duration_s < window_s:
        return [(0.0, duration_s)]
    n = int(duration_s // window_s)
    return [(k * window_s, (k + 1) * window_s) for k in range(n)]


def score_window(
    detector: Detector, audio, sr: int, start: float, end: float
) -> tuple[int, float, float]:
    """Run ``detector`` on the ``[start, end)`` slice; return ``(n_events, max, sum)``
    of the detected events' scores (a per-window salience summary)."""
    seg = audio[int(round(start * sr)) : int(round(end * sr))]
    events = detector.detect(seg, sr).events
    if not events:
        return (0, 0.0, 0.0)
    scores = [e.score for e in events]
    return (len(events), max(scores), float(sum(scores)))


# ----------------------------------------------------------------------- score
def score_grid(
    wav_scp: str,
    out_tsv: str,
    *,
    detector_name: str = "maad_roi",
    target_sr: int | None = 48000,
    window_s: float = 10.0,
    num_shards: int = 1,
    shard_id: int = 0,
) -> str:
    """Score every ``window_s`` grid window of each recording; write a scored-grid TSV.

    Recordings are sliced round-robin ``[shard_id::num_shards]`` so array tasks
    balance long/short files. Output columns:
    ``seg_id wav_id start end n_events max_score sum_score``.
    """
    rows = read_wav_scp(wav_scp)[shard_id::num_shards]
    detector = get_detector(detector_name)

    out_dir = os.path.dirname(os.path.abspath(out_tsv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    n_win = 0
    with open(out_tsv, "w") as f:
        f.write("seg_id\twav_id\tstart\tend\t" + "\t".join(SCORE_COLS) + "\n")
        for i, (wid, path) in enumerate(rows):
            try:
                audio, sr = load_mono_resampled(path, target_sr)
            except Exception as e:  # noqa: BLE001 - skip unreadable files, keep going
                print(f"skip {wid} ({path}): {e}", file=sys.stderr)
                continue
            dur = len(audio) / sr if sr else 0.0
            for k, (start, end) in enumerate(grid_windows(dur, window_s)):
                n_ev, mx, sm = score_window(detector, audio, sr, start, end)
                seg_id = f"{wid}-{k:09d}"
                f.write(f"{seg_id}\t{wid}\t{start:.2f}\t{end:.2f}\t"
                        f"{n_ev}\t{mx:.6g}\t{sm:.6g}\n")
                n_win += 1
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{len(rows)} recordings, {n_win} windows",
                      file=sys.stderr)

    print(f"wrote {out_tsv}: {n_win} windows from {len(rows)} recordings")
    return out_tsv


# ---------------------------------------------------------------------- select
def _read_scored(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        idx = {c: i for i, c in enumerate(header)}
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            p = line.split("\t")
            rows.append({
                "seg_id": p[idx["seg_id"]],
                "wav_id": p[idx["wav_id"]],
                "start": float(p[idx["start"]]),
                "end": float(p[idx["end"]]),
                "n_events": int(p[idx["n_events"]]),
                "max_score": float(p[idx["max_score"]]),
                "sum_score": float(p[idx["sum_score"]]),
            })
    return rows


def select_segments(
    scored,
    keep_frac: float,
    *,
    rank_by: str = "sum_score",
    scope: str = "per_recording",
) -> list[dict]:
    """Keep the top ``keep_frac`` most salient windows.

    Windows with no detected events are always dropped first. ``scope='per_recording'``
    keeps the top fraction *within each recording* (every recording stays represented,
    no cross-recording gain bias); ``scope='global'`` ranks all windows together.
    Returns the kept rows, sorted by ``(wav_id, start)``.
    """
    rows = _read_scored(scored) if isinstance(scored, str) else list(scored)
    rows = [r for r in rows if r["n_events"] > 0]
    if not rows:
        return []

    def top(group: list[dict]) -> list[dict]:
        k = max(1, math.ceil(keep_frac * len(group)))
        return sorted(group, key=lambda r: r[rank_by], reverse=True)[:k]

    if scope == "global":
        kept = top(rows)
    else:
        groups: dict[str, list[dict]] = {}
        for r in rows:
            groups.setdefault(r["wav_id"], []).append(r)
        kept = [r for g in groups.values() for r in top(g)]

    kept.sort(key=lambda r: (r["wav_id"], r["start"]))
    return kept


def write_segments(rows: list[dict], out_segments: str) -> tuple[str, str]:
    """Write kept windows as a Kaldi ``segments`` file + a ``.scores.tsv`` sidecar."""
    out_dir = os.path.dirname(os.path.abspath(out_segments))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    scores_path = out_segments + ".scores.tsv"
    with open(out_segments, "w") as seg_f, open(scores_path, "w") as sc_f:
        sc_f.write("seg_id\t" + "\t".join(SCORE_COLS) + "\n")
        for r in rows:
            seg_f.write(f"{r['seg_id']} {r['wav_id']} {r['start']:.2f} {r['end']:.2f}\n")
            sc_f.write(f"{r['seg_id']}\t{r['n_events']}\t"
                       f"{r['max_score']:.6g}\t{r['sum_score']:.6g}\n")
    print(f"wrote {out_segments}: {len(rows)} segments")
    return out_segments, scores_path


# -------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="score the grid windows of each recording")
    sc.add_argument("--wav-scp", required=True, help="Kaldi wav.scp (<wav_id> <path>)")
    sc.add_argument("--out", required=True, help="output scored-grid TSV")
    sc.add_argument("--detector", default="maad_roi", help="registry detector name")
    sc.add_argument("--target-sr", type=int, default=48000,
                    help="resample every recording to this rate (0/negative = native)")
    sc.add_argument("--window-s", type=float, default=10.0, help="grid window length (s)")
    sc.add_argument("--num-shards", type=int, default=1)
    sc.add_argument("--shard-id", type=int, default=0)

    se = sub.add_parser("select", help="keep the top-fraction salient windows")
    se.add_argument("--scored", required=True, help="scored-grid TSV (merged shards)")
    se.add_argument("--out-segments", required=True, help="output Kaldi segments file")
    se.add_argument("--keep-frac", type=float, default=0.5,
                    help="fraction of (non-empty) windows to keep")
    se.add_argument("--rank-by", default="sum_score", choices=SCORE_COLS)
    se.add_argument("--scope", default="per_recording",
                    choices=("per_recording", "global"))

    args = ap.parse_args(argv)
    if args.cmd == "score":
        target_sr = args.target_sr if args.target_sr and args.target_sr > 0 else None
        score_grid(args.wav_scp, args.out, detector_name=args.detector,
                   target_sr=target_sr, window_s=args.window_s,
                   num_shards=args.num_shards, shard_id=args.shard_id)
    else:
        kept = select_segments(args.scored, args.keep_frac,
                               rank_by=args.rank_by, scope=args.scope)
        write_segments(kept, args.out_segments)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
