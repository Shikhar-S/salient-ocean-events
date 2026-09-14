#!/usr/bin/env bash
# Run the salient-event benchmark on a compute node, all detectors in PARALLEL.
#
# Each detector is an independent process writing to its own sub-dir; the runner
# is single-detector per process, so the wall-clock is ~the slowest detector
# (maad_roi) instead of the sum. Results are merged into one results.tsv at the
# end. `uv run --no-sync` is used so the concurrent processes never race on the
# lockfile/cache (the env is assumed already synced on the login node).
#
#   Watkins (default):  sbatch run_benchmark.sbatch
#   BirdSet (32 kHz):   AUDIO=data/birdset/clips GT=data/birdset/gt.tsv OUT=eval/results_birdset \
#                           TARGET_SR=32000 sbatch run_benchmark.sbatch
#   Subset:             DETECTORS="maad_roi pcen_peak" sbatch run_benchmark.sbatch
#
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=2g
#SBATCH --job-name=edbench
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p logs
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

AUDIO=${AUDIO:-data/watkins/clips}
GT=${GT:-data/watkins/gt.tsv}
OUT=${OUT:-eval/results_watkins}
TARGET_SR=${TARGET_SR:-}   # optional: resample every clip to this rate (e.g. 32000 for BirdSet)
DETECTORS=${DETECTORS:-"maad_roi pcen_peak spectral_entropy energy_template"}

rm -rf "$OUT"; mkdir -p "$OUT"

# Keep per-process BLAS/OMP threading modest so the parallel processes share
# the cores without oversubscribing.
export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONWARNINGS=ignore

echo "Launching detectors in parallel: $DETECTORS"
pids=()
for d in $DETECTORS; do
    $UV run --no-sync python -m src.run_benchmark \
        --audio-dir "$AUDIO" --gt "$GT" --detectors "$d" \
        --out-dir "$OUT/$d" ${TARGET_SR:+--target-sr "$TARGET_SR"} &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 0 ] || { echo "a detector process failed"; exit 1; }

# Merge per-detector results.tsv into one, and lift preds_*.tsv up to $OUT.
$UV run --no-sync python - "$OUT" <<'PY'
import csv, glob, os, shutil, sys
out = sys.argv[1]
header, rows = None, []
for f in sorted(glob.glob(os.path.join(out, "*", "results.tsv"))):
    with open(f) as fh:
        r = list(csv.reader(fh, delimiter="\t"))
    header = r[0]
    rows += r[1:]
    for p in glob.glob(os.path.join(os.path.dirname(f), "preds_*.tsv")):
        shutil.copy(p, os.path.join(out, os.path.basename(p)))
with open(os.path.join(out, "results.tsv"), "w", newline="") as fh:
    w = csv.writer(fh, delimiter="\t", lineterminator="\n")
    w.writerow(header)
    w.writerows(rows)
print(f"merged {len(rows)} rows -> {out}/results.tsv")
PY
echo "done"
