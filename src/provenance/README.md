# Watkins evaluation set

Recovers the timestamp of each short **cut** clip inside its full **master** recording in
the Watkins Marine Mammal Sound Database, then cuts the balanced clip set the benchmark
scores. Standalone: these scripts do not import from the benchmark package (`src/`).

## Corpus

Obtained with <https://github.com/Shikhar-S/watkins-marine-sound-scraper>. `--data-root`
must point at a tree of this shape; it is only ever read. Set it per run, or export
`WATKINS_ROOT=/path/to/scraped_data` once:

```
scraped_data/
├── cut_tapes/<Species>/<Year>/<TAPEID><CUTIDX>.wav    # short curated clips
└── master_tapes/<Species>/<TAPEID>/<recording>.wav    # full tapes, several channels each
```

A cut filename is always 8 characters before `.wav`: `TAPEID` (first 5) resolves the source
tape, `CUTIDX` (last 3) indexes the clip within it. `<Year>` is `"19" + TAPEID[:2]`. A
`CUTIDX` beginning with a letter names a sub-reel, so `51041C01` comes from recording
`51041C`.

## 1. Align

Resamples cut and master to a common rate and locates the cut by normalized
cross-correlation. Writes **one row per (cut, channel)** so the confidence threshold can be
chosen afterwards without re-correlating. Resumable and incremental, so it is safe to
interrupt.

```bash
# whole corpus, one SLURM task per species
mkdir -p results/by_species
ls <data-root>/cut_tapes > results/by_species/species.txt
N=$(wc -l < results/by_species/species.txt)
sbatch --array=0-$((N-1))%12 src/provenance/run_provenance_array.sbatch

# or directly, one species at a time
uv run python src/provenance/cut_provenance_align.py \
    --species BottlenoseDolphin --out results/by_species/BottlenoseDolphin.tsv --workers 16
```

| Flag | Default | Notes |
|------|---------|-------|
| `--data-root` | `$WATKINS_ROOT` or `scraped_data` | read only, never written to |
| `--out` | `cut_provenance_alignment.tsv` | results TSV (+ `<out>.log`) |
| `--species` / `--tape` / `--limit` | all | restrict the run |
| `--workers` | 4 | master groups processed in parallel |
| `--corr-rate` | 48000 | max common correlation rate (Hz); lower is much faster |
| `--mode` | `wave` | keep `wave`; `envelope` has a flat correlation surface on click trains |
| `--no-resume` | off | otherwise a re-run skips cuts already in `--out` |

Columns:

```
species  cut_file  tape_id  cut_sr  cut_dur_s  master_file  master_sr  corr_sr
         offset_s  end_s  ncc_peak  peak_ratio  status
```

`master_file` is the channel this row is for, `offset_s`/`end_s` the match inside that
channel, `ncc_peak` the correlation there. `status` is `ok` or a reason
(`master_shorter_than_cut`, `cut_too_short`, `rate_too_low`, `no_corr`, `no_master`).

## 2. Consensus

A recording's channels are simultaneous, so a true match lands at the same offset on every
one. Cross-channel agreement above an `ncc` floor is the confidence signal; `peak_ratio` is
a within-channel sharpness cue that is high for spurious matches too, so never use it alone.

```bash
# merge the per-species tables, keeping a single header row
head -1 "$(ls results/by_species/*.tsv | head -1)" > results/provenance_all.tsv
tail -q -n +2 results/by_species/*.tsv                       >> results/provenance_all.tsv

uv run python src/provenance/consensus.py results/provenance_all.tsv \
    --min-ncc 0.5 --tol 0.05 --out results/provenance_confident.tsv
uv run python src/provenance/consensus.py results/provenance_all.tsv --sweep
```

Channels with `status==ok` and `ncc_peak >= --min-ncc` are clustered within `--tol` seconds,
labelling each cut `ok` (≥2 agree), `ok_single` (one qualifying channel, cannot be
cross-validated), `no_consensus` (they disagree, the offset is almost certainly spurious),
`no_match`, or a propagated non-ok status. Two channels can agree on the same *noise*, so
the floor is required: use `--sweep` to see locatable counts against it and pick the point
where matches check out by ear (0.3–0.5 here).

## 3. Cut the clip set

Keeps `ok` cuts, picks one canonical channel per tape, and writes one positive clip per
event with the event at a random position, plus an equal number of event-free negatives
from the same recordings.

```bash
uv run python src/provenance/build_eval_set.py \
    --confident results/provenance_confident.tsv --raw results/provenance_all.tsv \
    --out data/watkins --target-sr 48000
```

Tunable: `--clip-s`, `--max-event-s`, `--margin-s`, `--neg-per-pos`, `--seed`. Output is
`clips/<file_id>.wav`, `gt.tsv`, `metadata.csv`.
