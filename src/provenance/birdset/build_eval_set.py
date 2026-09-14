#!/usr/bin/env python3
"""Build a stratified BirdSet salient-event clip benchmark from a download dump.

Consumes the dump that ``download.py`` writes
(``download/birdset/<config>/soundscapes/*`` + ``annotations.tsv``) and produces the
*same* eval-set layout the benchmark already scores -- a sibling of ``data/watkins/``:

    data/birdset/
      clips/<file_id>.wav   mono, --target-sr (32 kHz), --clip-s seconds
      gt.tsv                positives' event spans (clip-local): file_id  Begin Time (s)  End Time (s)
      metadata.csv          per-clip manifest (HuggingFace audiofolder)
      README.md             dataset card

Vocabulary maps onto Watkins: a BirdSet **soundscape** is a *master tape*; we cut our
own fixed-length *clips*; the annotation ``start``/``end`` are the *ground-truth time
references* (no NCC alignment needed -- BirdSet gives them). Unlike a Watkins cut
(one event per clip), a soundscape clip may contain several, possibly overlapping,
events; because evaluation is **time-only**, temporally overlapping events are merged
into one interval (co-vocalizing birds cannot be separated in time alone).

Construction (mirrors ``src/provenance/build_eval_set.py``)
-----------------------------------------------------------
1. Group annotations by master. Stratify **positives**: shuffle events (seeded), take
   up to ``--max-positives`` with at most ``--max-per-species`` per ``ebird_code`` so a
   common species can't dominate.
2. POSITIVE clip: a ``--clip-s`` window from the master with the anchor event placed at
   a random in-clip position (``>= --margin-s`` from each edge); GT = **all** events of
   that master overlapping the window, clipped to clip-local time and merged.
3. NEGATIVE clip: an event-free ``--clip-s`` window from the same masters
   (``>= --margin-s`` from every annotated event), one per positive -> recall from
   positives, precision/selection from negatives.

Standalone: numpy + soundfile + scipy only (no ``datasets``, no ``src/`` imports), so it
unit-tests against a tiny synthetic dump. Run via ``run_build.sbatch``.

Caveat: negatives are only *supposedly* event-free -- BirdSet strong labels may miss
faint background calls, so selection rate is a conservative upper bound (uniform across
methods). Recall stays the primary metric.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from collections import Counter, defaultdict
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


# --------------------------------------------------------------------------- #
# Audio I/O (mirrors the Watkins builder; soundfile reads .ogg and .wav)
# --------------------------------------------------------------------------- #
def resample_to(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x.astype(np.float32)
    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(x, int(sr_to) // g, int(sr_from) // g).astype(np.float32)


def read_region(path: str, t0: float, t1: float, target_sr: int, _cache={}) -> np.ndarray:
    """Read ``[t0, t1)`` seconds from ``path`` as mono float32 at ``target_sr`` (padded)."""
    if path not in _cache:
        info = sf.info(path)
        _cache[path] = (info.samplerate, info.frames)
    sr, n = _cache[path]
    a, b = max(0, int(round(t0 * sr))), min(n, int(round(t1 * sr)))
    x, _ = sf.read(path, start=a, stop=b, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = resample_to(x, sr, target_sr)
    want = int(round((t1 - t0) * target_sr))
    if len(x) < want:
        x = np.pad(x, (0, want - len(x)))
    return x[:want]


def master_len_s(path: str, _cache={}) -> float:
    if path not in _cache:
        info = sf.info(path)
        _cache[path] = info.frames / info.samplerate if info.samplerate else 0.0
    return _cache[path]


# --------------------------------------------------------------------------- #
# Pure helpers (unit-tested)
# --------------------------------------------------------------------------- #
def load_annotations(path: str) -> dict[str, list[tuple[float, float, str]]]:
    """Read ``annotations.tsv`` -> ``{master_id: [(start_s, end_s, ebird_code), ...]}``."""
    by_master: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            try:
                s, e = float(r["start_s"]), float(r["end_s"])
            except (TypeError, ValueError, KeyError):
                continue
            if e > s:
                by_master[r["master_id"]].append((s, e, r.get("ebird_code", "")))
    for mid in by_master:
        by_master[mid].sort()
    return dict(by_master)


def merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Union of (possibly overlapping) ``[start, end)`` intervals, sorted & merged."""
    out: list[list[float]] = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return [(s, e) for s, e in out]


def events_in_window(events, t0: float, t1: float) -> list[tuple[float, float]]:
    """Clip-local merged GT spans for the master ``events`` overlapping ``[t0, t1)``."""
    T = t1 - t0
    local = []
    for s, e, _code in events:
        if e > t0 and s < t1:
            ls, le = max(0.0, s - t0), min(T, e - t0)
            if le > ls:
                local.append((ls, le))
    return merge_intervals(local)


def _safe_id(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(s))


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dump", default="download/birdset", help="download dump root")
    ap.add_argument("--config", required=True, help="BirdSet region, e.g. HSN")
    ap.add_argument("--out", default="data/birdset")
    ap.add_argument("--clip-s", type=float, default=10.0)
    ap.add_argument("--margin-s", type=float, default=1.0)
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--target-sr", type=int, default=32000)
    ap.add_argument("--max-positives", type=int, default=400)
    ap.add_argument("--max-per-species", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    base = os.path.join(args.dump, args.config)
    ann_path = os.path.join(base, "annotations.tsv")
    if not os.path.exists(ann_path):
        raise SystemExit(f"no annotations at {ann_path}; run download.py first")
    by_master = load_annotations(ann_path)

    # master_id -> soundscape file (first match of any extension)
    master_path: dict[str, str] = {}
    for mid in by_master:
        hits = sorted(glob.glob(os.path.join(base, "soundscapes", f"{mid}.*")))
        if hits:
            master_path[mid] = hits[0]

    T, m, sr = args.clip_s, args.margin_s, args.target_sr

    # ---- stratified positive selection: cap per species, then overall ----
    flat = [(mid, s, e, code) for mid, evs in by_master.items()
            for (s, e, code) in evs if mid in master_path]
    rng.shuffle(flat)
    per_species: Counter = Counter()
    anchors = []
    for mid, s, e, code in flat:
        if len(anchors) >= args.max_positives:
            break
        if per_species[code] >= args.max_per_species:
            continue
        if master_len_s(master_path[mid]) <= T:   # too short to cut a clip
            continue
        anchors.append((mid, s, e, code))
        per_species[code] += 1

    os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)
    gt_rows, meta_rows = [], []
    pos_per_master: Counter = Counter()

    # ---- positive clips ----
    for i, (mid, a_s, a_e, code) in enumerate(anchors):
        path = master_path[mid]
        mlen = master_len_s(path)
        d = min(a_e - a_s, T - 2 * m)
        hi = max(m, T - d - m)
        rel = float(rng.uniform(m, hi)) if hi > m else m       # anchor onset within clip
        t0 = a_s - rel
        t0 = min(max(t0, 0.0), max(0.0, mlen - T))
        fid = f"{_safe_id(mid)}__p{i:04d}"
        clip = read_region(path, t0, t0 + T, sr)
        sf.write(os.path.join(args.out, "clips", f"{fid}.wav"), clip, sr)

        spans = events_in_window(by_master[mid], t0, t0 + T)   # clip-local merged GT
        for (gs, ge) in spans:
            gt_rows.append((fid, gs, ge))
        cover = sum(ge - gs for gs, ge in spans)
        a_on, a_off = max(0.0, a_s - t0), min(T, a_e - t0)
        meta_rows.append({
            "file_name": f"clips/{fid}.wav", "label": "event", "species": code,
            "split": args.config, "master_file": os.path.basename(path),
            "src_offset_s": f"{t0:.4f}", "event_onset_s": f"{a_on:.4f}",
            "event_offset_s": f"{a_off:.4f}", "sr": sr, "n_events": len(spans),
            "density": f"{cover / T:.4f}",
        })
        pos_per_master[mid] += 1

    # ---- negative clips: event-free windows, neg_per_pos per positive, per master ----
    neg_made = 0
    for mid, n_pos in pos_per_master.items():
        path = master_path[mid]
        mlen = master_len_s(path)
        if mlen <= T:
            continue
        forb = by_master[mid]
        n_need = n_pos * args.neg_per_pos
        accepted: list[float] = []
        attempts = 0
        while len(accepted) < n_need and attempts < 400 * max(1, n_need):
            attempts += 1
            t0 = float(rng.uniform(0, mlen - T))
            win = (t0 - m, t0 + T + m)
            if any(not (win[1] <= s or win[0] >= e) for s, e, _ in forb):
                continue
            if any(abs(t0 - a) < T for a in accepted):     # keep negatives disjoint
                continue
            accepted.append(t0)
        for k, t0 in enumerate(accepted):
            fid = f"{_safe_id(mid)}__n{k:03d}"
            clip = read_region(path, t0, t0 + T, sr)
            sf.write(os.path.join(args.out, "clips", f"{fid}.wav"), clip, sr)
            meta_rows.append({
                "file_name": f"clips/{fid}.wav", "label": "none", "species": "",
                "split": args.config, "master_file": os.path.basename(path),
                "src_offset_s": f"{t0:.4f}", "event_onset_s": "", "event_offset_s": "",
                "sr": sr, "n_events": 0, "density": "0.0000",
            })
            neg_made += 1

    # ---- write GT + metadata + card ----
    with open(os.path.join(args.out, "gt.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["file_id", "Begin Time (s)", "End Time (s)"])
        for fid, a, b in gt_rows:
            w.writerow([fid, f"{a:.4f}", f"{b:.4f}"])
    mfields = ["file_name", "label", "species", "split", "master_file", "src_offset_s",
               "event_onset_s", "event_offset_s", "sr", "n_events", "density"]
    with open(os.path.join(args.out, "metadata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mfields, lineterminator="\n")
        w.writeheader()
        w.writerows(meta_rows)

    n_pos = len(anchors)
    _write_card(args, n_pos, neg_made, len(pos_per_master), len(per_species))
    print(f"positives={n_pos}  negatives={neg_made}  masters={len(pos_per_master)}  "
          f"species={len(per_species)}  sr={sr}  -> {args.out}/")


def _write_card(args, n_pos, n_neg, n_masters, n_species):
    txt = f"""# BirdSet salient-event clip benchmark ({args.config})

A balanced clip dataset for **label-free salient bioacoustic event detection**, the bird
sibling of the Watkins set (`data/watkins/`). Built from the BirdSet `{args.config}`
soundscape `test` split by cutting fixed-length clips around annotated vocalizations
(masters + per-event `start`/`end`; no alignment needed -- BirdSet gives the times).

## Contents
- `clips/*.wav` -- mono, {args.target_sr} Hz, {args.clip_s:g}s each ({n_pos + n_neg} clips).
- `gt.tsv` -- positive clips' event spans (clip-local): `file_id  Begin Time (s)  End Time (s)`.
- `metadata.csv` -- per-clip manifest (HuggingFace audiofolder).

## Labels
- **{n_pos} positive** clips (`label=event`) over {n_masters} soundscapes, {n_species} species:
  contain >=1 annotated event; temporally overlapping events are merged (time-only).
- **{n_neg} negative** clips (`label=none`): event-free windows from the same soundscapes.

## Caveats
- Negatives are only *supposedly* event-free: BirdSet strong labels may miss faint background
  calls, so selection rate is a conservative upper bound (uniform across methods).
- Non-bird salient sound (insects, wind) can make a detector fire -- a false positive that
  isn't truly wrong. Recall is the primary metric.

## Run the benchmark
    uv run python -m src.run_benchmark --audio-dir clips --gt gt.tsv \\
        --detectors all --target-sr {args.target_sr}

## Attribution
Source: BirdSet (Rauch et al., arXiv 2403.10380; DBD-research-group/BirdSet on HuggingFace),
itself aggregating Xeno-Canto and soundscape annotations. Cite/credit BirdSet and the
underlying sources per their terms.
"""
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
