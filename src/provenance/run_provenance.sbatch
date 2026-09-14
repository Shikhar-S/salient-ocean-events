#!/usr/bin/env bash
# Generate Watkins cut -> master provenance alignment as a SLURM batch job.
#
# Small-set-first: this defaults to a single species (ClymeneDolphin, ~328 cuts)
# so the output can be eyeballed before scaling to the whole corpus. The run is
# resumable -- re-submitting with the same --out continues where it left off.
#
#   Submit small set:  sbatch src/provenance/run_provenance.sbatch
#   Another species:   SPECIES=Walrus sbatch src/provenance/run_provenance.sbatch
#   Whole corpus:      SPECIES="" sbatch src/provenance/run_provenance.sbatch
#                      (bump --time/--mem below first; the full run is much longer)
#   Fresh output dir:  OUTDIR=results/bench sbatch src/provenance/run_provenance.sbatch
#
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96g
#SBATCH --job-name=provenance
#SBATCH --output=logs/%x-%j.out
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

SPECIES="${SPECIES-ClymeneDolphin}"          # set to "" for the whole corpus
OUTDIR="${OUTDIR:-results/provenance}"       # override to write into a fresh dir
OUT="$OUTDIR/provenance_${SPECIES:-all}.tsv"
WORKERS="${SLURM_CPUS_PER_TASK:-16}"
mkdir -p logs "$OUTDIR"

species_arg=()
[[ -n "$SPECIES" ]] && species_arg=(--species "$SPECIES")

echo "Species: ${SPECIES:-<all>}   Workers: $WORKERS   Out: $OUT"
"$UV" run python src/provenance/cut_provenance_align.py \
    "${species_arg[@]}" \
    --out "$OUT" \
    --workers "$WORKERS"
