# BirdSet evaluation sets

Turns the **BirdSet** soundscape benchmark (`DBD-research-group/BirdSet` on HuggingFace)
into the same `clips/ + gt.tsv + metadata.csv` layout the benchmark scores. BirdSet's
`test` split gives full soundscape recordings with per-event `start_time`/`end_time`, so
unlike Watkins no alignment is needed — the scripts only parse and cut. Standalone: no
import from the benchmark package (`src/`).

## Run

```bash
# all seven habitats: download -> build -> benchmark
sbatch src/provenance/birdset/run_sweep.sbatch

# or one config at a time
CONFIG=HSN sbatch src/provenance/birdset/run_download.sbatch   # -> download/birdset/HSN/
CONFIG=HSN sbatch src/provenance/birdset/run_build.sbatch      # -> data/birdset/HSN/
```

Sanity-check a config first by adding `LIMIT=50` to the download job. NBP is excluded: its
recordings are shorter than the 10 s clip length.

**Requirements.** Only `download.py` needs `datasets`. BirdSet is a script-based dataset, so
it requires `datasets<=3.6.0` (pinned in `pyproject.toml`) and `trust_remote_code=True`;
newer versions refuse it with "Dataset scripts are no longer supported".

**Disk.** `load_dataset` downloads and extracts *all* splits, roughly 14 GB for HSN and more
for larger configs. The cache lands in `download/birdset/.hf_cache` and is disposable once
the dump exists, since `soundscapes/` holds independent copies. `run_sweep.sbatch` reclaims
it automatically.

## Dump format

`download.py` fetches only the requested `--split` (default `test`) and copies each
soundscape once, undecoded:

```
download/birdset/<config>/
├── soundscapes/<master_id>.<ext>   unique full soundscapes
├── annotations.tsv                 master_id  start_s  end_s  ebird_code   (one row/event)
└── meta.tsv                        config  split  n_masters  n_events
```

`ebird_code` is a `ClassLabel`, so the integer index is resolved back to the species name
before writing.

## Eval set

`build_eval_set.py` cuts 10 s mono clips at 32 kHz into `data/birdset/<CONFIG>/`:
`clips/<file_id>.wav`, `gt.tsv` (clip-local event spans), `metadata.csv`, and a dataset
card. Positives are capped per species (`--max-per-species`) and overall
(`--max-positives`); negatives are event-free windows from the same soundscapes, 1:1 with
positives. Because evaluation is time-only, temporally overlapping events from
co-vocalizing birds are merged into one ground-truth interval.
