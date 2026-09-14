# Drowning in Noise: On Signal Saliency in Self-Supervised Learning for Ocean Acoustics

The saliency half of the paper: the label-free salient event detection benchmark (RQ2) and
the detector-based filtering that builds a pre-training corpus (RQ3). Encoder pre-training
and probing live in [OpenBEATs](https://github.com/Shikhar-S/OpenBEATs).

| Path | What it does |
|------|--------------|
| `src/provenance/` | builds the evaluation sets (Watkins, BirdSet) |
| `src/detectors/`, `src/eval/` | the four detectors and the runner that scores them |
| `src/annotate.py` | turns a detector into OpenBEATs pre-training annotations |

Detector registry names are `maad_roi`, `pcen_peak`, `spectral_entropy` and
`energy_template`.

## Install

`uv`, Python ≥ 3.12. Run everything from the repo root.

```bash
uv sync
uv run python -m pytest
```

The SLURM scripts resolve the repo from the directory you submit from, so run `sbatch` at
the repo root, or export `REPO=/path/to/repo`. Their `--account` and `--partition` lines are
site-specific and will need editing. If `$HOME` has a small quota, send the uv wheel cache
to scratch with `export UV_CACHE_DIR=/path/to/scratch/.uv-cache`.

## Build the evaluation sets

A Watkins cut carries no timestamp in its master tape, so it is located by
cross-correlation and kept only where at least two channels agree on the offset:

```bash
# 1. locate (per-species job array, resumable)
sbatch --array=0-$((N-1))%12 src/provenance/run_provenance_array.sbatch

# 2. keep only cross-channel-agreeing localizations
uv run python src/provenance/consensus.py results/provenance_all.tsv \
    --out results/provenance_confident.tsv

# 3. cut positives + event-free negatives
uv run python src/provenance/build_eval_set.py \
    --confident results/provenance_confident.tsv --raw results/provenance_all.tsv \
    --out data/watkins --target-sr 48000
```

BirdSet ships event times, so one script downloads, cuts and scores all seven habitats:

```bash
sbatch src/provenance/birdset/run_sweep.sbatch
```

Details in [`src/provenance/README.md`](src/provenance/README.md) and
[`src/provenance/birdset/README.md`](src/provenance/birdset/README.md).

## Run the benchmark

```bash
uv run python -m src.run_benchmark \
    --audio-dir data/watkins/clips --gt data/watkins/gt.tsv \
    --detectors all --out-dir eval/results_watkins
```

Add `--target-sr 32000` for BirdSet; `--help` lists the rest. `sbatch run_benchmark.sbatch`
runs the four detectors in parallel and merges their output. Table 2 reads `recall`,
`selection_rate` and `auc_recall_sel` from the `iou(>=0.5)` rows of `results.tsv`.

Ground truth is a TSV of `file_id`, `Begin Time (s)`, `End Time (s)`, where `file_id` is the
audio basename without extension. Aliases such as `filename`, `onset` and `offset` are
also accepted.

## Annotation bridge

```bash
uv run python -m src.annotate score --wav-scp <wav.scp> --out grid.tsv --target-sr 48000
uv run python -m src.annotate select --scored grid.tsv --out-segments segments --keep-frac 0.5
```

On SLURM, `sbatch run_annotate.sbatch` shards the scoring; rerun it with `select` once the
array finishes. The `segments` file is consumed verbatim by OpenBEATs'
`recipes/watkins/prepare_manifest.py`.