#!/usr/bin/env bash
# Annotate the Watkins master tapes with maad_roi -> a Kaldi `segments` file for
# OpenBEATs pretraining. Two phases:
#
#   1. score the 10 s grid (array-sharded):
#        sbatch run_annotate.sbatch
#   2. merge shards + select the top-K% most salient windows:
#        sbatch --dependency=afterok:<arrayjobid> run_annotate.sbatch select
#
# Override WAV_SCP / OUT / TARGET_SR / WINDOW_S / KEEP_FRAC / RANK_BY / SCOPE via
# --export=ALL,VAR=...  `uv run --no-sync` so the parallel array tasks never race
# on the lockfile/cache (env assumed already synced on the login node).
#
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --array=0-31
#SBATCH --job-name=mt-annotate
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail
cd "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

WAV_SCP=${WAV_SCP:?set WAV_SCP=/path/to/wav.scp (Kaldi wav.scp of the recordings to annotate)}
OUT=${OUT:-data/master_tapes_maad}
TARGET_SR=${TARGET_SR:-48000}
WINDOW_S=${WINDOW_S:-10.0}
KEEP_FRAC=${KEEP_FRAC:-0.5}
RANK_BY=${RANK_BY:-sum_score}
SCOPE=${SCOPE:-per_recording}
mkdir -p "$OUT/shards"

export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONWARNINGS=ignore

# --- select mode: merge scored shards, keep the top-K% salient windows ----------
if [ "${1:-}" = "select" ]; then
    { head -n1 "$(ls "$OUT"/shards/grid.shard*.tsv | head -n1)";
      tail -q -n +2 "$OUT"/shards/grid.shard*.tsv; } > "$OUT/grid.tsv"
    $UV run --no-sync python -m src.annotate select \
        --scored "$OUT/grid.tsv" --out-segments "$OUT/segments" \
        --keep-frac "$KEEP_FRAC" --rank-by "$RANK_BY" --scope "$SCOPE"
    exit 0
fi

# --- score mode (array task): score the grid windows of this shard --------------
N=${SLURM_ARRAY_TASK_COUNT:-1}
I=${SLURM_ARRAY_TASK_ID:-0}
$UV run --no-sync python -m src.annotate score \
    --wav-scp "$WAV_SCP" --out "$OUT/shards/grid.shard${I}.tsv" \
    --target-sr "$TARGET_SR" --window-s "$WINDOW_S" \
    --num-shards "$N" --shard-id "$I"
echo "shard $I/$N done"
