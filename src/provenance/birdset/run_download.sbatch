#!/usr/bin/env bash
# Download one BirdSet soundscape split into the dump (SLURM; BirdSet is large).
#
#   One config (default HSN):  CONFIG=HSN sbatch src/provenance/birdset/run_download.sbatch
#   Another region:            CONFIG=POW sbatch src/provenance/birdset/run_download.sbatch
#   Sanity (few examples):     CONFIG=HSN LIMIT=50 sbatch src/provenance/birdset/run_download.sbatch
#
# Needs the `datasets` dep (uv add datasets, synced on the login node first). The
# HuggingFace cache is pinned under download/birdset/.hf_cache -- NEVER a quota-limited $HOME.
#
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --job-name=bs-download
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
mkdir -p logs

CONFIG=${CONFIG:-HSN}
SPLIT=${SPLIT:-test}
OUT=${OUT:-download/birdset}
LIMIT=${LIMIT:-}

# Pin every HuggingFace cache into the repo (a small $HOME quota would overflow).
# NB: load_dataset extracts ALL splits (train+test+test_5s), ~14 GB for HSN — this
# cache is disposable once the dump is built (soundscapes/ holds independent copies).
export HF_HOME="$REPO/download/birdset/.hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
mkdir -p "$HF_DATASETS_CACHE"

echo "Downloading BirdSet config=$CONFIG split=$SPLIT -> $OUT/$CONFIG"
"$UV" run --no-sync python src/provenance/birdset/download.py \
    --config "$CONFIG" --split "$SPLIT" --out "$OUT" \
    --cache-dir "$HF_DATASETS_CACHE" ${LIMIT:+--limit "$LIMIT"}
echo "done"
