#!/usr/bin/env bash
# Full BirdSet sweep: for every region, download (soundscapes-only `_scape`) -> build a
# stratified clip set -> benchmark all detectors -> reclaim the HF cache. Resumable
# (skips a stage whose output already exists) and fault-tolerant (a failed region does
# not abort the rest). Aggregates IoU auc_recall_sel across regions at the end.
#
#   Whole sweep:     sbatch src/provenance/birdset/run_sweep.sbatch
#   Subset:          REGIONS="HSN POW PER" sbatch src/provenance/birdset/run_sweep.sbatch
#   Full configs:    SCAPE=0 sbatch ...        (download focal train too — much larger)
#
#SBATCH --account=bbjs-delta-cpu
#SBATCH --partition=cpu
#SBATCH --time=18:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48g
#SBATCH --job-name=bs-sweep
#SBATCH --output=logs/%x-%j.out
set -uo pipefail   # NOT -e: continue past a region's failure

REPO="${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$REPO"; mkdir -p logs
UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

REGIONS=${REGIONS:-"HSN UHH POW SNE SSW NES PER"}   # NBP excluded: its recordings are shorter than the 10 s clip length
DETECTORS=${DETECTORS:-"maad_roi pcen_peak spectral_entropy energy_template"}
TARGET_SR=32000
SCAPE=${SCAPE:-1}
scape_flag=--scape; [ "$SCAPE" = "0" ] && scape_flag=--no-scape

export OMP_NUM_THREADS=3 OPENBLAS_NUM_THREADS=3 MKL_NUM_THREADS=3 PYTHONWARNINGS=ignore
export HF_HOME="$REPO/download/birdset/.hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

for R in $REGIONS; do
    echo "==================== REGION $R ===================="
    # 1. DOWNLOAD (soundscapes only) — skip if dump already present.
    if [ ! -f "download/birdset/$R/annotations.tsv" ]; then
        mkdir -p "$HF_DATASETS_CACHE"
        $UV run --no-sync python src/provenance/birdset/download.py \
            --config "$R" --split test --out download/birdset \
            --cache-dir "$HF_DATASETS_CACHE" $scape_flag \
            || { echo "!! download failed for $R, skipping"; rm -rf "$HF_HOME"; continue; }
    else
        echo "  download: dump exists, skipping"
    fi
    rm -rf "$HF_HOME"   # reclaim cache now; soundscapes/ holds independent copies

    # 2. BUILD — skip if the clip set already exists.
    if [ ! -f "data/birdset/$R/gt.tsv" ]; then
        $UV run --no-sync python src/provenance/birdset/build_eval_set.py \
            --dump download/birdset --config "$R" --out "data/birdset/$R" \
            --target-sr "$TARGET_SR" --seed 0 \
            || { echo "!! build failed for $R, skipping"; continue; }
    else
        echo "  build: clip set exists, skipping"
    fi

    # 3. BENCHMARK (detectors in parallel) — skip if results already present.
    OUT="eval/results_birdset/$R"
    if [ -f "$OUT/results.tsv" ]; then
        echo "  benchmark: results exist, skipping"
        continue
    fi
    rm -rf "$OUT"; mkdir -p "$OUT"
    pids=()
    for d in $DETECTORS; do
        $UV run --no-sync python -m src.run_benchmark \
            --audio-dir "data/birdset/$R/clips" --gt "data/birdset/$R/gt.tsv" \
            --detectors "$d" --out-dir "$OUT/$d" --target-sr "$TARGET_SR" &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p" || true; done
    $UV run --no-sync python - "$OUT" <<'PY'
import csv, glob, os, sys
out = sys.argv[1]; header=None; rows=[]
for f in sorted(glob.glob(os.path.join(out, "*", "results.tsv"))):
    r = list(csv.reader(open(f), delimiter="\t")); header = r[0]; rows += r[1:]
if header:
    w = csv.writer(open(os.path.join(out, "results.tsv"), "w", newline=""),
                   delimiter="\t", lineterminator="\n")
    w.writerow(header); w.writerows(rows)
print(f"  merged {len(rows)} rows -> {out}/results.tsv")
PY
    echo "  region $R done"
done

# ---- aggregate IoU auc_recall_sel across regions ----
echo "==================== SWEEP SUMMARY (IoU auc_recall_sel) ===================="
$UV run --no-sync python - <<'PY'
import csv, glob
data, regions = {}, []
for f in sorted(glob.glob("eval/results_birdset/*/results.tsv")):
    R = f.split("/")[-2]
    if R not in regions: regions.append(R)
    for r in csv.DictReader(open(f), delimiter="\t"):
        if "iou" in r["matcher"]:
            data.setdefault(r["detector"], {})[R] = float(r["auc_recall_sel"])
print("detector".ljust(17) + "".join(R.rjust(8) for R in regions) + "    mean")
def mean(d):
    vs=[d.get(R) for R in regions if d.get(R) is not None]; return sum(vs)/len(vs) if vs else 0
for det in sorted(data, key=lambda d: -mean(data[d])):
    row = "".join(f"{data[det].get(R, float('nan')):8.3f}" for R in regions)
    print(det.ljust(17) + row + f"  {mean(data[det]):6.3f}")
PY
echo "SWEEP COMPLETE"
