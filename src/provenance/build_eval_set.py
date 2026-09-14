#!/usr/bin/env python3
"""Build a balanced salient-event clip benchmark from Watkins provenance.

Each confidently-located cut becomes a POSITIVE clip (a short window of the master
recording containing one known salient event); an equal number of NEGATIVE clips
are sampled from the same recordings at event-free positions. The result is a
small, self-contained, HuggingFace-ready dataset under ``data/watkins/`` that the
existing benchmark scores directly (recall from positives, precision from negatives).

Inputs
------
* ``provenance_confident.tsv`` -- the cross-validated cuts (status==ok, the kept set).
* ``provenance_all.tsv``       -- raw per-channel results, for each cut's offset on
                                  the chosen canonical channel.

Construction
------------
1. Keep confident ``ok`` cuts; positives are those with event duration <= --max-event-s.
2. Per tape, pick ONE canonical channel (plurality of representative channels, ncc
   tie-break) and consolidate all the tape's events onto it.
3. POSITIVE clip: --clip-s seconds from the canonical channel with the event placed
   at a random position (>= --margin-s from each edge), resampled to --target-sr.
4. NEGATIVE clip: an event-free --clip-s window from the same canonical channel,
   >= --margin-s from every confident event; one per positive, per recording.

Outputs (under --out, default ``data/watkins/``)
------------------------------------------------
* ``clips/<file_id>.wav``  mono, --target-sr, uniform length.
* ``gt.tsv``               benchmark ground truth (positives only):
                           ``file_id  Begin Time (s)  End Time (s)``.
* ``metadata.csv``         HuggingFace audiofolder manifest (every clip).
* ``README.md``            dataset card (construction + caveats + attribution).

Caveat: negatives are only *supposedly* event-free -- a few may overlap unmatched
activity, since only the 445 confident events are avoided.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import Counter, defaultdict
from math import gcd

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

# Override with --root, or by exporting WATKINS_ROOT=/path/to/scraped_data.
DEFAULT_ROOT = os.environ.get("WATKINS_ROOT", "scraped_data")


def resample_to(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x.astype(np.float32)
    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(x, int(sr_to) // g, int(sr_from) // g).astype(np.float32)


def read_region(path, t0, t1, target_sr, _cache={}):
    """Read [t0, t1) seconds from `path` as mono float32 at `target_sr` (padded)."""
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


def master_len_s(path, _cache={}):
    if path not in _cache:
        info = sf.info(path)
        _cache[path] = info.frames / info.samplerate if info.samplerate else 0.0
    return _cache[path]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confident", default="results/provenance_confident.tsv")
    ap.add_argument("--raw", default="results/provenance_all.tsv")
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--out", default="data/watkins")
    ap.add_argument("--clip-s", type=float, default=10.0)
    ap.add_argument("--max-event-s", type=float, default=6.0)
    ap.add_argument("--margin-s", type=float, default=1.0)
    ap.add_argument("--neg-per-pos", type=int, default=1)
    ap.add_argument("--target-sr", type=int, default=48000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    rng = np.random.default_rng(args.seed)

    conf = [r for r in csv.DictReader(open(args.confident), delimiter="\t")
            if r["status"] == "ok"]

    # canonical channel per tape: plurality of representative channels, ncc tie-break
    by_tape = defaultdict(list)
    for r in conf:
        by_tape[(r["species"], r["tape_id"])].append(r)
    canon = {}
    for key, rs in by_tape.items():
        cnt = Counter(r["master_file"] for r in rs)
        ncc_sum = defaultdict(float)
        for r in rs:
            ncc_sum[r["master_file"]] += float(r["ncc_peak"] or 0)
        canon[key] = max(cnt, key=lambda mf: (cnt[mf], ncc_sum[mf]))

    # each cut's offset on the canonical channel (from the raw per-channel table)
    raw_off = {}
    for r in csv.DictReader(open(args.raw), delimiter="\t"):
        if r["status"] == "ok":
            raw_off[(r["species"], r["cut_file"], r["master_file"])] = \
                (float(r["offset_s"]), float(r["end_s"]))

    def chan_path(species, tape, mf):
        return os.path.join(args.root, "master_tapes", species, tape, mf)

    forbidden = defaultdict(list)   # (species,tape) -> [(on,off)] confident events on canon chan
    positives = []
    dropped_long = 0
    for r in conf:
        key = (r["species"], r["tape_id"]); cch = canon[key]
        dur = float(r["cut_dur_s"])
        on, off = raw_off.get((r["species"], r["cut_file"], cch),
                              (float(r["offset_s"]), float(r["end_s"])))
        forbidden[key].append((on, off))          # avoid all 445 when placing negatives
        if dur <= args.max_event_s:
            positives.append({"r": r, "key": key, "cch": cch, "on": on,
                              "dur": dur, "path": chan_path(r["species"], r["tape_id"], cch)})
        else:
            dropped_long += 1

    os.makedirs(os.path.join(args.out, "clips"), exist_ok=True)
    T, m, sr = args.clip_s, args.margin_s, args.target_sr
    meta_rows, gt_rows = [], []

    # ---- positive clips ----
    for p in positives:
        r = p["r"]; mlen = master_len_s(p["path"])
        d = min(p["dur"], T - 2 * m)
        hi = max(m, T - d - m)
        rel = float(rng.uniform(m, hi)) if hi > m else m          # event onset within clip
        t0 = p["on"] - rel
        t0 = min(max(t0, 0.0), max(0.0, mlen - T))
        rel = p["on"] - t0                                        # re-derive after clamping
        rel = min(max(rel, 0.0), T - d)
        fid = os.path.splitext(r["cut_file"])[0]
        clip = read_region(p["path"], t0, t0 + T, sr)
        sf.write(os.path.join(args.out, "clips", f"{fid}.wav"), clip, sr)
        try:
            csr = sf.info(os.path.join(args.root, "cut_tapes", r["species"],
                                       "19" + r["tape_id"][:2], r["cut_file"])).samplerate
        except Exception:
            csr = ""
        gt_rows.append((fid, rel, rel + d))
        meta_rows.append({"file_name": f"clips/{fid}.wav", "label": "event",
                          "species": r["species"], "tape": r["tape_id"],
                          "master_file": p["cch"], "src_offset_s": f"{p['on']:.4f}",
                          "event_onset_s": f"{rel:.4f}", "event_offset_s": f"{rel + d:.4f}",
                          "sr": sr, "cut_native_sr": csr,
                          "ncc": r["ncc_peak"], "n_agree": r["n_agree"]})

    # ---- negative clips: event-free windows, one per positive, per recording ----
    pos_per_tape = Counter(p["key"] for p in positives)
    neg_made = 0
    for key, n_need in pos_per_tape.items():
        n_need *= args.neg_per_pos
        species, tape = key; cch = canon[key]; path = chan_path(species, tape, cch)
        mlen = master_len_s(path)
        if mlen <= T:
            continue
        forb = forbidden[key]; accepted = []
        attempts = 0
        while len(accepted) < n_need and attempts < 400 * n_need:
            attempts += 1
            t0 = float(rng.uniform(0, mlen - T))
            win = (t0 - m, t0 + T + m)
            if any(not (win[1] <= on or win[0] >= off) for on, off in forb):
                continue
            if any(abs(t0 - a) < T for a in accepted):     # keep negatives disjoint
                continue
            accepted.append(t0)
        for k, t0 in enumerate(accepted):
            fid = f"neg_{tape}_{k:02d}"
            clip = read_region(path, t0, t0 + T, sr)
            sf.write(os.path.join(args.out, "clips", f"{fid}.wav"), clip, sr)
            meta_rows.append({"file_name": f"clips/{fid}.wav", "label": "none",
                              "species": species, "tape": tape, "master_file": cch,
                              "src_offset_s": f"{t0:.4f}", "event_onset_s": "",
                              "event_offset_s": "", "sr": sr, "cut_native_sr": "",
                              "ncc": "", "n_agree": ""})
            neg_made += 1

    # ---- write GT, metadata ----
    with open(os.path.join(args.out, "gt.tsv"), "w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["file_id", "Begin Time (s)", "End Time (s)"])
        for fid, a, b in gt_rows:
            w.writerow([fid, f"{a:.4f}", f"{b:.4f}"])
    mfields = ["file_name", "label", "species", "tape", "master_file", "src_offset_s",
               "event_onset_s", "event_offset_s", "sr", "cut_native_sr", "ncc", "n_agree"]
    with open(os.path.join(args.out, "metadata.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mfields, lineterminator="\n")
        w.writeheader()
        w.writerows(meta_rows)

    n_pos = len(positives)
    _write_card(args, n_pos, neg_made, dropped_long, len(by_tape))
    print(f"positives={n_pos}  negatives={neg_made}  dropped_long(>{args.max_event_s}s)={dropped_long}")
    print(f"clips={n_pos + neg_made}  tapes={len(by_tape)}  sr={sr}  -> {args.out}/")


def _write_card(args, n_pos, n_neg, dropped, n_tapes):
    txt = f"""# Watkins salient-event clip benchmark

A balanced clip dataset for **label-free salient bioacoustic event detection**,
derived from the Watkins Marine Mammal Sound Database by acoustically locating each
curated "cut" excerpt inside its source "master" recording (cross-correlation +
cross-channel agreement), then cutting fixed-length clips.

## Contents
- `clips/*.wav` -- mono, {args.target_sr} Hz, {args.clip_s:g}s each ({n_pos + n_neg} clips).
- `gt.tsv` -- ground truth for the positive clips: `file_id  Begin Time (s)  End Time (s)`.
- `metadata.csv` -- per-clip manifest (HuggingFace audiofolder).

## Labels
- **{n_pos} positive** clips (`label=event`): contain one located salient event;
  its in-clip span is in `gt.tsv` and `metadata.csv` (`event_onset_s`/`event_offset_s`).
- **{n_neg} negative** clips (`label=none`): event-free windows from the same
  recordings (no `gt.tsv` rows).

## Caveats
- Negatives are only *supposedly* event-free: only the confident events were avoided,
  so a few may overlap unmatched/unlabelled activity (uniform across methods).
- Clips are {args.target_sr} Hz (band <= {args.target_sr // 2} Hz); events whose energy was
  higher-frequency are attenuated -- see `cut_native_sr` to identify them.
- Event boundaries are curator-trimmed and carry ~50 ms localization uncertainty;
  score with onset/offset collars, not tight IoU.

## Run the benchmark
    uv run python -m src.run_benchmark --audio-dir clips --gt gt.tsv --detectors all

## Attribution
Source: Watkins Marine Mammal Sound Database (https://cis.whoi.edu/science/B/whalesounds/).
Please cite/credit Watkins MMSD per its terms when using this dataset.
"""
    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(txt)


if __name__ == "__main__":
    main()
