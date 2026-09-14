#!/usr/bin/env python3
"""CLI: benchmark salient event detectors on a folder of audio against a GT table.

Example
-------
    uv run python -m src.run_benchmark \
        --audio-dir /path/to/wavs \
        --gt /path/to/ground_truth.tsv \
        --detectors all \
        --out-dir results/

Ground truth is a (Raven-style) selection table; see
``src.io.selection_table.read_selection_table`` for accepted columns. The
``file_id`` joining predictions to ground truth is each audio file's basename
without extension, so the GT table's file column should use the same.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys

import src.detectors  # noqa: F401  registers all bundled detectors
from src.core.detector import REGISTRY, get_detector
from src.eval.matching import CollarMatcher, IoUMatcher
from src.eval.benchmark import run_benchmark
from src.io.selection_table import read_selection_table

AUDIO_EXTS = (".wav", ".flac", ".ogg", ".aif", ".aiff")


def discover_audio(audio_dir: str) -> dict[str, str]:
    """Return ``{file_id: path}`` for audio under ``audio_dir`` (recursive)."""
    files: dict[str, str] = {}
    for ext in AUDIO_EXTS:
        for path in glob.glob(os.path.join(audio_dir, "**", f"*{ext}"), recursive=True):
            file_id = os.path.splitext(os.path.basename(path))[0]
            files[file_id] = path
    return files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--audio-dir", required=True, help="folder of audio files (recursive)")
    ap.add_argument("--gt", required=True, help="ground-truth selection table (TSV)")
    ap.add_argument(
        "--detectors", default="all",
        help="comma-separated detector names, or 'all' (default)",
    )
    ap.add_argument("--out-dir", default=None, help="write predictions + results here")
    ap.add_argument("--target-sr", type=int, default=None,
                    help="resample all audio to this rate before detection")
    ap.add_argument("--chunk-s", type=float, default=None,
                    help="process long files in chunks of this many seconds")
    ap.add_argument("--overlap-s", type=float, default=0.0,
                    help="overlap between chunks in seconds")
    # Matcher tolerances.
    ap.add_argument("--onset-collar", type=float, default=0.2)
    ap.add_argument("--offset-collar", type=float, default=0.2)
    ap.add_argument("--offset-fraction", type=float, default=0.2)
    ap.add_argument("--iou-threshold", type=float, default=0.5)
    args = ap.parse_args(argv)

    files = discover_audio(args.audio_dir)
    if not files:
        print(f"No audio found under {args.audio_dir!r}", file=sys.stderr)
        return 2
    gt_by_file = read_selection_table(args.gt)

    if args.detectors.strip().lower() == "all":
        names = sorted(REGISTRY)
    else:
        names = [n.strip() for n in args.detectors.split(",") if n.strip()]
    detectors = [get_detector(n) for n in names]

    matchers = [
        CollarMatcher(args.onset_collar, args.offset_collar, args.offset_fraction),
        IoUMatcher(args.iou_threshold),
    ]

    print(f"Detectors: {', '.join(names)}")
    print(f"Files: {len(files)}   GT events: {sum(len(v) for v in gt_by_file.values())}")
    rows = run_benchmark(
        detectors=detectors,
        files=files,
        gt_by_file=gt_by_file,
        matchers=matchers,
        out_dir=args.out_dir,
        target_sr=args.target_sr,
        chunk_s=args.chunk_s,
        overlap_s=args.overlap_s,
    )

    print("\nResults")
    print("-------")
    for row in rows:
        print(f"{row.detector:>16}  {row.result.summary()}")
    if args.out_dir:
        print(f"\nWrote predictions + results.tsv to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
