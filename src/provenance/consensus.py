#!/usr/bin/env python3
"""Derive cross-channel provenance consensus from a raw per-channel alignment table.

`cut_provenance_align.py` writes one row per (cut, channel), keeping every
channel's raw offset / ncc / peak_ratio so the confidence rule can be chosen
*after* the (expensive) correlation run. This script applies that rule:

A recording's channels are simultaneous, so a real match lands at the same time
on each. For one cut we keep the channels that matched well (`status==ok` and
`ncc_peak >= --min-ncc`), cluster their offsets within `--tol` seconds, and label
the cut:

  * ``ok``           - >=2 qualifying channels agree on the offset
  * ``ok_single``    - only one qualifying channel (agreement can't be tested)
  * ``no_consensus`` - qualifying channels disagree (offset almost certainly spurious)
  * ``no_match``     - no channel cleared the ncc floor
  * (else)           - a propagated non-ok status (no_master / master_shorter / ...)

The representative offset/score is the highest-`ncc_peak` channel in the agreeing
cluster. Re-run with different `--min-ncc` / `--tol` to retune -- no realignment.

Usage:
    python consensus.py raw.tsv --min-ncc 0.5 --tol 0.05 --out consensus.tsv
    python consensus.py raw.tsv --sweep            # just print locatable counts vs ncc floor
"""

import argparse
import csv
import sys
from collections import defaultdict

OUT_FIELDS = [
    "species", "cut_file", "tape_id", "cut_dur_s", "master_file",
    "offset_s", "end_s", "ncc_peak", "peak_ratio", "n_chan", "n_qual", "n_agree",
    "status",
]


def _f(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError):
        return 0.0


def consensus(chan_rows, min_ncc, tol_s):
    """Collapse one cut's per-channel rows into a single consensus dict."""
    n_chan = len(chan_rows)
    qual = [r for r in chan_rows if r["status"] == "ok" and _f(r, "ncc_peak") >= min_ncc]
    if not qual:
        # No channel cleared the floor: report the cut, but say why.
        base = chan_rows[0]
        any_ok = any(r["status"] == "ok" for r in chan_rows)
        status = "no_match" if any_ok else base["status"]
        rep = dict(base)
        if status == "no_match":   # blank the (rejected) offset to avoid implying a hit
            rep["offset_s"] = rep["end_s"] = rep["ncc_peak"] = rep["peak_ratio"] = ""
        rep.update(status=status, n_chan=n_chan, n_qual=0, n_agree=0)
        return rep

    best = []
    for r in qual:
        o = _f(r, "offset_s")
        cluster = [q for q in qual if abs(_f(q, "offset_s") - o) <= tol_s]
        if len(cluster) > len(best):
            best = cluster
    rep = dict(max(best, key=lambda r: _f(r, "ncc_peak")))
    if len(qual) == 1:
        status = "ok_single"
    elif len(best) >= 2:
        status = "ok"
    else:
        status = "no_consensus"
    rep.update(status=status, n_chan=n_chan, n_qual=len(qual), n_agree=len(best))
    return rep


def load_by_cut(path):
    by_cut = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            by_cut[(r["species"], r["cut_file"])].append(r)
    return by_cut


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", help="raw per-channel TSV from cut_provenance_align.py")
    ap.add_argument("--min-ncc", type=float, default=0.5,
                    help="a channel must reach this ncc_peak to count (default 0.5)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="offset agreement tolerance in seconds (default 0.05)")
    ap.add_argument("--out", default=None, help="write the per-cut consensus TSV here")
    ap.add_argument("--sweep", action="store_true",
                    help="print locatable-cut counts across a range of ncc floors")
    args = ap.parse_args(argv)

    by_cut = load_by_cut(args.raw)

    if args.sweep:
        print(f"{'min_ncc':>8} | {'ok':>6} | {'ok_single':>9} | {'tapes(ok)':>9}")
        for thr in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            n_ok = n_single = 0
            tapes = set()
            for (sp, _cf), rows in by_cut.items():
                c = consensus(rows, thr, args.tol)
                if c["status"] == "ok":
                    n_ok += 1
                    tapes.add((sp, c["tape_id"]))
                elif c["status"] == "ok_single":
                    n_single += 1
            print(f"{thr:>8.2f} | {n_ok:>6} | {n_single:>9} | {len(tapes):>9}")
        return 0

    results = [consensus(rows, args.min_ncc, args.tol) for rows in by_cut.values()]
    from collections import Counter
    counts = Counter(r["status"] for r in results)
    print(f"min_ncc={args.min_ncc}  tol={args.tol}s  -> {dict(counts)}", file=sys.stderr)

    # per-species locatable summary
    per_sp = defaultdict(lambda: [0, 0, set()])   # species -> [n_cuts, n_ok, ok_tapes]
    for r in results:
        s = per_sp[r["species"]]
        s[0] += 1
        if r["status"] == "ok":
            s[1] += 1
            s[2].add(r["tape_id"])
    print(f"{'species':<32} {'cuts':>6} {'ok':>5} {'ok_tapes':>8}", file=sys.stderr)
    for sp in sorted(per_sp, key=lambda k: -per_sp[k][1]):
        n, ok, tps = per_sp[sp]
        if ok:
            print(f"{sp:<32} {n:>6} {ok:>5} {len(tps):>8}", file=sys.stderr)

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=OUT_FIELDS, delimiter="\t", lineterminator="\n")
            w.writeheader()
            for r in results:
                w.writerow({k: r.get(k, "") for k in OUT_FIELDS})
        print(f"wrote {len(results)} cuts -> {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
