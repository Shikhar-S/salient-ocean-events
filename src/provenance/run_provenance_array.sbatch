#!/usr/bin/env bash
# Full-corpus Watkins provenance as a SLURM job array: one task per species, run
# concurrently. Each task writes its own raw per-channel table (resumable);
# afterwards, combine + derive consensus:
#
#   # one line per species:
#   ls .../scraped_data/cut_tapes > results/by_species/species.txt
#   N=$(wc -l < results/by_species/species.txt)
#   sbatch --array=0-$((N-1))%12 src/provenance/run_provenance_array.sbatch
#
#   # when the array is done:
#   head -1 results/by_species/AtlanticSpottedDolphin.tsv > results/provenance_all.tsv
#   tail -q -n +2 results/by_species/*.tsv            >> results/provenance_all.tsv
#   python src/provenance/consensus.py results/provenance_all.tsv --sweep
#
# Per-worker memory is ~0.2 GB (ncc_offset streams the offset axis), so 2 GB/CPU
# is ample at any core count even for the ~140-min masters.
#
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --mem-per-cpu=2g
#SBATCH --job-name=prov_sp
#SBATCH --output=logs/%x-%A_%a.out
set -euo pipefail

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"
mkdir -p logs results/by_species

LIST="${SPECIES_LIST:-results/by_species/species.txt}"
SP=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$LIST")
[[ -z "$SP" ]] && { echo "no species at array index $SLURM_ARRAY_TASK_ID"; exit 0; }

echo "species='$SP'  cpus=$SLURM_CPUS_PER_TASK"
"$UV" run python src/provenance/cut_provenance_align.py \
    --species "$SP" \
    --out "results/by_species/${SP}.tsv" \
    --workers "$SLURM_CPUS_PER_TASK" \
    --corr-rate 48000
