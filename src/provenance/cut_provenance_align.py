#!/usr/bin/env python3
r"""
cut_provenance_align.py

Locate the exact time offset of each Watkins "cut" clip within its source
"master" recording, using normalized cross-correlation (NCC).

Background
----------
The Watkins Marine Mammal Sound Database is laid out as two parallel trees:

    <data_root>/cut_tapes/<Species>/<Year>/<TAPEID><CUTIDX>.wav   (short clips)
    <data_root>/master_tapes/<Species>/<TAPEID>/<recording>.wav   (full tapes + .txt logs)

A cut filename is always 8 characters before ".wav":

    T T T T T  X X X
    \_______/  \____/
     tape id    cut index within that tape

  * TAPEID (first 5 chars) -> master_tapes/<Species>/<TAPEID>/   (the source tape)
  * The <Year> folder is "19" + first two digits of TAPEID.
  * A cut whose index begins with a letter (e.g. 84228A01) comes from the
    correspondingly-lettered sub-recording/reel (84228A...), if present.

This script confirms and refines that provenance acoustically: it finds *where*
in the master recording each cut occurs (start/end seconds) and how confident
that match is (NCC peak in [-1, 1]).

Provenance resolution per cut
-----------------------------
1. species + 5-char tape id  ->  master_tapes/<species>/<tapeid>/
2. Among the master recordings in that folder, pick the one whose "stem"
   (filename minus channel suffix) is the longest prefix of the cut basename.
   e.g. cut "51041C01" -> recording stem "51041C" (beats "51041").
3. Pick the best channel file for that recording:
     - cut_sr  > 48 kHz  -> prefer the high-rate channel  (hi1 / H1, 192 kHz)
     - cut_sr <= 48 kHz  -> prefer the audible channel    (ch1 44kHz)
   (falls back across ch/hi/H/bare as available).
4. Resample BOTH cut and master to a common correlation rate
   = min(--corr-rate, cut_sr, master_sr), then NCC the cut (template) against
   the whole master; the argmax is the start offset.

Reading the scores (validated on synthetic filtered+resampled+noisy excerpts):
  * ncc_peak is often LOW (~0.05-0.3) even for a correct match, because many
    cuts are band-pass-filtered copies of the master -- do NOT use its
    magnitude as confidence.
  * peak_ratio (peak vs. the best competing position) IS the confidence signal:
    in validation, true matches scored peak_ratio ~3-12 while the rate of false
    localization stayed near 1.0. Treat peak_ratio >= ~2.5 as confident,
    1.5-2.5 as plausible, < 1.5 as unreliable (likely a different version,
    heavily processed, or a repetitive click train with no unique alignment).

Orphaned cuts (whose master tape was never scraped) are reported with
status="no_master" and skipped, per request.

Usage
-----
    uv run python src/provenance/cut_provenance_align.py \
        --data-root /path/to/scraped_data \
        --out cut_provenance_alignment.tsv \
        --workers 8

Useful flags: --species BottlenoseDolphin   --limit 50   --corr-rate 16000
The run is resumable: re-running skips cuts already present in --out.

Reads the corpus read-only; writes only to --out (+ a .log next to it).
"""

import argparse
import csv
import os
import re
import sys
import glob
import time
import logging
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly, oaconvolve
from math import gcd
from multiprocessing import shared_memory

# Where the scraped Watkins corpus lives. Override with --data-root, or by exporting
# WATKINS_ROOT=/path/to/scraped_data.
DEFAULT_DATA_ROOT = os.environ.get("WATKINS_ROOT", "scraped_data")

# Output columns (tab-separated). One row PER (cut, channel): every channel's raw
# correlation result is kept so confidence thresholds (ncc floor, offset
# agreement) can be decided afterwards from the full record -- see consensus.py.
FIELDS = [
    "species", "cut_file", "tape_id", "cut_sr", "cut_dur_s",
    "master_file", "master_sr", "corr_sr",
    "offset_s", "end_s", "ncc_peak", "peak_ratio", "status",
]


# --------------------------------------------------------------------------- #
# Master-file parsing / channel selection
# --------------------------------------------------------------------------- #

# Trailing channel token: ch1 / ch1 44kHz / hi2 / H3 ...
_CHAN_RE = re.compile(r"(?i)\s*(?P<kind>ch|hi|h)\s*(?P<num>\d)\s*(?P<khz>44\s*k?hz)?\s*$")


def parse_master_name(filename):
    """Return (stem, chan_kind, prefers_high, rank) for a master wav filename.

    rank: lower is a more-preferred audible channel; high-rate handling is done
    by the caller based on the cut's own sample rate.
    """
    name = filename[:-4] if filename.lower().endswith(".wav") else filename
    m = _CHAN_RE.search(name)
    if m:
        stem = name[: m.start()].strip()
        kind = m.group("kind").lower()
        is_44k = bool(m.group("khz"))
        if kind == "ch":
            rank = 0 if is_44k else 1          # audible band, preferred at <=48k
            prefers_high = False
        elif kind == "hi":
            rank = 2
            prefers_high = True                # 192 kHz high-rate channel
        else:  # 'h' / 'H' series (older naming, also high-rate)
            rank = 3
            prefers_high = True
        return stem, kind, prefers_high, rank
    # No recognizable channel token (bare "<stem>.wav" or compilation like 12/34)
    return name.strip(), "", False, 4


def index_master_recordings(data_root):
    """species -> tapeid -> list of dicts {path, stem, prefers_high, rank}."""
    idx = defaultdict(lambda: defaultdict(list))
    pattern = os.path.join(data_root, "master_tapes", "*", "*", "*.wav")
    for path in glob.glob(pattern):
        parts = path.split(os.sep)
        species, tapeid, fn = parts[-3], parts[-2], parts[-1]
        stem, kind, prefers_high, rank = parse_master_name(fn)
        idx[species][tapeid].append(
            {"path": path, "stem": stem, "prefers_high": prefers_high, "rank": rank}
        )
    return idx


def choose_channels(cut_base, cut_sr, recordings):
    """Return (channel_paths, stem) for the recording a cut belongs to, or (None, None).

    The recording is the longest stem that is a prefix of the cut basename; its
    channel files (hi1..hi4 / ch1..ch4 / bare) are all returned, ordered by how
    well each suits the cut's band. They are *simultaneous* recordings, so a true
    match must land at the same time on each -- ``consensus.py`` uses that
    agreement as the real confidence signal instead of a single channel's
    `peak_ratio`. The best-suited channel is first (used when only one exists).
    """
    cands = [r for r in recordings if cut_base.startswith(r["stem"]) and r["stem"]]
    if not cands:
        return None, None
    longest = max(len(r["stem"]) for r in cands)
    stem = next(r["stem"] for r in cands if len(r["stem"]) == longest)
    cands = [r for r in cands if len(r["stem"]) == longest]

    want_high = cut_sr > 48000
    def key(r):
        # Prefer matching the cut's band first, then the audible-rank ordering.
        band_mismatch = 0 if (r["prefers_high"] == want_high) else 1
        return (band_mismatch, r["rank"])
    cands.sort(key=key)
    return [r["path"] for r in cands], stem


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #

def load_mono(path):
    """Load a wav as float32 mono. Returns (samples, samplerate)."""
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return np.ascontiguousarray(data, dtype=np.float32), sr


def resample_to(x, sr_from, sr_to):
    if sr_from == sr_to:
        return x
    g = gcd(int(sr_from), int(sr_to))
    up, down = int(sr_to) // g, int(sr_from) // g
    return resample_poly(x, up, down).astype(np.float32)


def envelope(x, sr, env_sr):
    """Amplitude (RMS) envelope downsampled to ~env_sr Hz.

    Robust to band-pass filtering and spectral differences between a cut and
    its master: it matches the temporal pattern of clicks/whistles/song, not
    the raw waveform. Returns (env, actual_env_sr).
    """
    win = max(1, int(round(sr / env_sr)))
    e = x.astype(np.float64)
    e *= e
    csum = np.cumsum(np.insert(e, 0, 0.0))
    sm = (csum[win:] - csum[:-win]) / win          # moving-average power
    env = np.sqrt(sm[::win]).astype(np.float32)    # decimate, take RMS
    return env, sr / win


def ncc_offset(master, cut, block=4_000_000):
    """Normalized cross-correlation of `cut` (template) sliding over `master`.

    Returns (best_index, peak_value, peak_ratio); both inputs at one sample rate.
    The offset axis is swept in chunks of `block`, with the numerator (overlap-add
    convolution) and the windowed-energy denominator (a block-local prefix sum)
    built one chunk at a time -- so peak memory is ~`block` samples regardless of
    how long the master is (corpus masters reach ~140 min). peak_ratio = peak vs.
    the best competing position outside a +/- n/2 guard.
    """
    n = len(cut)
    L = len(master) - n + 1
    if L <= 0:
        return None
    c = cut - cut.mean()
    c_norm = float(np.sqrt(np.dot(c, c)))
    if c_norm < 1e-9:
        return None  # silent template
    rev = np.ascontiguousarray(c[::-1])
    guard = max(1, n // 2)

    cands = []   # (value, position) local maxima -- best-in-block plus its runner-up
    s = 0
    while s < L:
        e = min(s + block, L)
        m = e - s
        seg = master[s : e + n - 1]                     # covers offsets s .. e-1
        # numerator: sum_i master[k+i] * c[i]  (c is zero-mean, window mean drops out)
        num = oaconvolve(seg, rev, mode="valid")[:m]
        seg64 = seg.astype(np.float64)
        ps = np.empty(seg64.size + 1); ps[0] = 0.0; np.cumsum(seg64, out=ps[1:])
        ps2 = np.empty(seg64.size + 1); ps2[0] = 0.0; np.cumsum(seg64 * seg64, out=ps2[1:])
        del seg64
        s1 = ps[n : n + m] - ps[:m]
        we = ps2[n : n + m] - ps2[:m]
        we -= (s1 * s1) / n
        del s1, ps, ps2
        np.clip(we, 1e-9, None, out=we)
        np.sqrt(we, out=we)
        we *= c_norm
        ncc = num / we
        del num, we

        i1 = int(np.argmax(ncc))
        cands.append((float(ncc[i1]), s + i1))
        lo, hi = max(0, i1 - guard), min(m, i1 + guard + 1)
        ncc[lo:hi] = -np.inf                            # mask, find an in-block runner-up
        if m > hi - lo:
            i2 = int(np.argmax(ncc))
            if np.isfinite(ncc[i2]):
                cands.append((float(ncc[i2]), s + i2))
        del ncc
        s = e

    best_val, best_pos = max(cands)
    second = max((v for v, p in cands if abs(p - best_pos) > guard), default=0.0)
    ratio = best_val / second if second > 1e-6 else float("inf")
    return best_pos, best_val, ratio


# --------------------------------------------------------------------------- #
# Worker: process all cuts that share one master file
# --------------------------------------------------------------------------- #

def _to_shm(arr):
    """Copy `arr` into a new shared-memory block; return (handle, meta).

    Allocates at least one byte so empty arrays (e.g. a 0-frame master) can be
    shared without `SharedMemory` rejecting a zero size; the true shape rides in
    the meta so the view is reconstructed empty.
    """
    arr = np.ascontiguousarray(arr)
    shm = shared_memory.SharedMemory(create=True, size=max(arr.nbytes, 1))
    if arr.nbytes:
        np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
    return shm, (shm.name, arr.shape, arr.dtype.str)


def _from_shm(meta):
    """Attach a shared-memory block read-only; return (handle, array view)."""
    name, shape, dtype = meta
    shm = shared_memory.SharedMemory(name=name)
    return shm, np.ndarray(shape, dtype=np.dtype(dtype), buffer=shm.buf)


def _cut_worker(task):
    """Correlate one cut against a master that already lives in shared memory."""
    (species, cut_path, tape, master_path, mr_meta,
     series_sr, mode, env_rate, m_sr) = task
    shms = []
    try:
        shm_mr, mr = _from_shm(mr_meta); shms.append(shm_mr)

        cut, c_sr = load_mono(cut_path)
        cut_dur = len(cut) / c_sr
        if mode == "envelope":
            ce, ce_sr = envelope(cut, c_sr, env_rate)
            c = resample_to(ce, ce_sr, env_rate)
        else:
            c = resample_to(cut, c_sr, series_sr)

        if len(c) < 8:
            return _row(species, cut_path, tape, master_path, cut_sr=c_sr,
                        cut_dur=cut_dur, m_sr=m_sr, corr_sr=series_sr,
                        status="cut_too_short")
        if len(mr) < len(c):
            # master recording is shorter than the cut -- usually an empty or
            # truncated master wav in the scrape (e.g. a 0-second file).
            return _row(species, cut_path, tape, master_path, cut_sr=c_sr,
                        cut_dur=cut_dur, m_sr=m_sr, corr_sr=series_sr,
                        status="master_shorter_than_cut")

        res = ncc_offset(mr, c)
        if res is None:
            return _row(species, cut_path, tape, master_path, cut_sr=c_sr,
                        cut_dur=cut_dur, m_sr=m_sr, corr_sr=series_sr,
                        status="no_corr")
        best, peak, ratio = res
        offset = best / series_sr
        return _row(species, cut_path, tape, master_path, cut_sr=c_sr,
                    cut_dur=cut_dur, m_sr=m_sr, corr_sr=series_sr, offset=offset,
                    end=offset + cut_dur, peak=peak, ratio=ratio, status="ok")
    except Exception as e:
        return _row(species, cut_path, tape, master_path,
                    status=f"error:{type(e).__name__}:{e}"[:120])
    finally:
        for s in shms:
            s.close()


def _process_master(master_path, cut_jobs, corr_rate, mode, env_rate, mapper):
    """cut_jobs: list of (species, cut_path, tape_id, cut_sr). Returns row dicts.

    Loads/resamples the master once per correlation rate, stages it in shared
    memory, and fans the master's cuts out via `mapper` so all workers share that
    one copy instead of reloading it per cut/group. ncc_offset streams the offset
    axis, so per-worker memory stays small even for ~140-min masters.

    mode == "envelope": correlate RMS envelopes at `env_rate` (filtering-robust).
    mode == "wave":     correlate raw waveforms at min(corr_rate, cut_sr, m_sr).
    """
    rows = []
    try:
        master_raw, m_sr = load_mono(master_path)
    except Exception as e:  # unreadable master
        for species, cut_path, tape, _c_sr in cut_jobs:
            rows.append(_row(species, cut_path, tape, master_path,
                             status=f"master_read_error:{type(e).__name__}"))
        return rows

    cuts_by_sr = defaultdict(list)
    if mode == "envelope":
        cuts_by_sr[env_rate] = list(cut_jobs)
    else:
        for species, cut_path, tape, c_sr in cut_jobs:
            series_sr = int(min(corr_rate, c_sr, m_sr)) if c_sr else int(min(corr_rate, m_sr))
            if series_sr < 100:
                rows.append(_row(species, cut_path, tape, master_path,
                                 cut_sr=c_sr, m_sr=m_sr, status="rate_too_low"))
                continue
            cuts_by_sr[series_sr].append((species, cut_path, tape, c_sr))

    if mode == "envelope":
        e, e_sr = envelope(master_raw, m_sr, env_rate)
        env_series = resample_to(e, e_sr, env_rate)

    for series_sr, cuts in cuts_by_sr.items():
        if not cuts:
            continue
        # Build (and stage) the master at this rate only when needed, one rate at
        # a time, so several resampled copies never coexist in memory.
        if mode == "envelope":
            mr = np.ascontiguousarray(env_series, dtype=np.float32)
        else:
            mr = np.ascontiguousarray(resample_to(master_raw, m_sr, series_sr),
                                      dtype=np.float32)
        shm_mr, mr_meta = _to_shm(mr)
        del mr
        try:
            tasks = [
                (species, cut_path, tape, master_path, mr_meta,
                 series_sr, mode, env_rate, m_sr)
                for species, cut_path, tape, c_sr in cuts
            ]
            rows.extend(mapper(_cut_worker, tasks))
        finally:
            shm_mr.close()
            shm_mr.unlink()
    return rows


def _row(species, cut_path, tape, master_path, cut_sr="", cut_dur="",
         m_sr="", corr_sr="", offset="", end="", peak="", ratio="", status=""):
    return {
        "species": species,
        "cut_file": os.path.basename(cut_path),
        "tape_id": tape,
        "cut_sr": cut_sr,
        "cut_dur_s": f"{cut_dur:.4f}" if cut_dur != "" else "",
        "master_file": os.path.basename(master_path) if master_path else "",
        "master_sr": m_sr,
        "corr_sr": corr_sr,
        "offset_s": f"{offset:.4f}" if offset != "" else "",
        "end_s": f"{end:.4f}" if end != "" else "",
        "ncc_peak": f"{peak:.4f}" if peak != "" else "",
        "peak_ratio": (f"{ratio:.3f}" if ratio != "" and ratio != float("inf")
                       else ("inf" if ratio == float("inf") else "")),
        "status": status,
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT)
    ap.add_argument("--out", default="cut_provenance_alignment.tsv")
    ap.add_argument("--species", default=None, help="restrict to one species folder")
    ap.add_argument("--tape", default=None, help="restrict to one 5-char tape id (testing)")
    ap.add_argument("--limit", type=int, default=0, help="process at most N cuts (testing)")
    ap.add_argument("--mode", choices=["wave", "envelope"], default="wave",
                    help="wave (default, recommended): correlate raw waveforms; "
                         "gives a sharp, reliable peak even for band-pass-filtered "
                         "copies -- judge confidence by peak_ratio, not ncc_peak. "
                         "envelope: correlate RMS envelopes; a fallback, but the "
                         "correlation surface is flat (unreliable argmax) for "
                         "repetitive click trains -- use only if wave fails.")
    ap.add_argument("--env-rate", type=int, default=1000,
                    help="envelope sample rate in Hz (envelope mode)")
    ap.add_argument("--corr-rate", type=int, default=48000,
                    help="max common rate for waveform correlation (Hz); "
                         "effective rate = min(this, cut_sr, master_sr)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore any existing --out and start fresh")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stderr),
                  logging.FileHandler(args.out + ".log")],
    )
    log = logging.getLogger("align")

    log.info("Indexing master recordings under %s ...", args.data_root)
    master_idx = index_master_recordings(args.data_root)
    log.info("Indexed %d species of master tapes.", len(master_idx))

    # Resume: collect already-done cut filenames.
    done = set()
    if os.path.exists(args.out) and not args.no_resume:
        with open(args.out, newline="") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                done.add((r["species"], r["cut_file"]))
        log.info("Resuming: %d cuts already in %s", len(done), args.out)

    # Resolve every cut -> its recording (a stem + all of its channel files).
    cut_glob = os.path.join(args.data_root, "cut_tapes",
                            args.species or "*", "*", "*.wav")
    recordings = {}    # (species, tape, stem) -> {"channels": [...], "cuts": [...]}
    orphans = []       # cuts with no master folder/recording
    n_cuts = 0
    for cut_path in glob.glob(cut_glob):
        parts = cut_path.split(os.sep)
        species, cut_fn = parts[-3], parts[-1]
        if (species, cut_fn) in done:
            continue
        base = cut_fn[:-4]
        tape = base[:5]
        if args.tape and tape != args.tape:
            continue
        recs = master_idx.get(species, {}).get(tape)
        if not recs:
            orphans.append((species, cut_path, tape))
            continue
        try:
            c_sr = sf.info(cut_path).samplerate
        except Exception:
            c_sr = 0
        channels, stem = choose_channels(base, c_sr, recs)
        if not channels:
            orphans.append((species, cut_path, tape))
            continue
        rec = recordings.setdefault((species, tape, stem),
                                    {"channels": channels, "cuts": []})
        rec["cuts"].append((species, cut_path, tape, c_sr))
        n_cuts += 1
        if args.limit and n_cuts >= args.limit:
            break

    log.info("To process: %d cuts in %d recordings; %d orphans (ignored).",
             n_cuts, len(recordings), len(orphans))

    # Open output (append if resuming).
    new_file = args.no_resume or not os.path.exists(args.out)
    out_f = open(args.out, "w" if new_file else "a", newline="")
    writer = csv.DictWriter(out_f, fieldnames=FIELDS, delimiter="\t",
                            lineterminator="\n")
    if new_file:
        writer.writeheader()
    # Record orphans once (only when starting fresh, to avoid dupes on resume).
    if new_file:
        for species, cut_path, tape in orphans:
            writer.writerow(_row(species, cut_path, tape, "", status="no_master"))
        out_f.flush()

    def run_channel(ch_path, cuts, mapper):
        try:
            return _process_master(ch_path, cuts, args.corr_rate, args.mode,
                                   args.env_rate, mapper)
        except Exception as e:  # never silently drop a channel's cuts
            log.error("channel failed %s: %s", ch_path, e)
            return [_row(s, p, t, ch_path, status=f"error:{type(e).__name__}")
                    for s, p, t, _c_sr in cuts]

    def process_recording(rec, mapper):
        # Emit every channel's raw result (one row per cut per channel); the
        # cross-channel consensus / thresholds are derived later by consensus.py.
        channels, cuts = rec["channels"], rec["cuts"]
        rows = []
        for ch_path in channels:
            rows.extend(run_channel(ch_path, cuts, mapper))
        return rows

    t0 = time.time()
    n_done = 0
    # Heaviest recordings first for a shorter makespan.
    ordered = sorted(recordings.items(), key=lambda kv: len(kv[1]["cuts"]),
                     reverse=True)

    def drive(mapper):
        nonlocal n_done
        for (species, tape, stem), rec in ordered:
            for row in process_recording(rec, mapper):
                writer.writerow(row)
            n_done += len(rec["cuts"])
            out_f.flush()
            log.info("[%d/%d] %s/%s (%d ch)", n_done, n_cuts, tape, stem,
                     len(rec["channels"]))

    if args.workers <= 1:
        drive(lambda fn, it: list(map(fn, it)))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            drive(lambda fn, it: list(ex.map(fn, it, chunksize=1)))

    out_f.close()
    log.info("Done. %d cuts in %.1f min -> %s",
             n_done, (time.time() - t0) / 60, args.out)


if __name__ == "__main__":
    main()
