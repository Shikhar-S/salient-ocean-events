#!/usr/bin/env bash
# Build the stratified BirdSet eval set from a download dump (cpu node).
#
#   Default (HSN):   CONFIG=HSN sbatch src/provenance/birdset/run_build.sbatch
#   Tune size:       CONFIG=HSN MAX_POS=400 MAX_PER_SP=50 sbatch src/provenance/birdset/run_build.sbatch
#
# Reads download/birdset/<CONFIG>/ (run run_download.sbatch first); writes data/birdset/.
#
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --job-name=bs-build
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
mkdir -p logs

CONFIG=${CONFIG:-HSN}
OUT=${OUT:-data/birdset}
TARGET_SR=${TARGET_SR:-32000}
MAX_POS=${MAX_POS:-400}
MAX_PER_SP=${MAX_PER_SP:-50}
SEED=${SEED:-0}

echo "Building BirdSet eval set config=$CONFIG -> $OUT"
"$UV" run --no-sync python src/provenance/birdset/build_eval_set.py \
    --dump download/birdset --config "$CONFIG" --out "$OUT" \
    --target-sr "$TARGET_SR" --max-positives "$MAX_POS" \
    --max-per-species "$MAX_PER_SP" --seed "$SEED"
echo "done"
