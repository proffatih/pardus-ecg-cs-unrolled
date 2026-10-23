#!/usr/bin/env bash
# run_phase4.sh — final evaluation pass.
#
# BSBL now runs batched on the GPU (18x faster, numerically identical to 1e-9),
# so the evaluation set is raised from 250 to 1000 windows per recording and
# every fold is re-evaluated with the SAME setting, instead of mixing sizes.
# No network is retrained: all models are on disk and are loaded.
#
# Outputs go to $RUN_OUT (outside the git tree).
set -u
cd "$(dirname "$0")"
export RUN_OUT=/home/pardus/papers_runs/0027
export BSBL_GPU=1
LOG=$RUN_OUT/run_phase4.log
mkdir -p "$RUN_OUT"
echo "=== PHASE4 START $(date -Is) ===" | tee -a "$LOG"

echo ">>> [1/3] all folds, all CR, 1000 windows/record + QRS endpoint" | tee -a "$LOG"
python3 run_cv3.py --eval_cap 1000 --full_cap 2000 --qrs_seconds 120 \
        --out cv4_metrics.csv >> "$LOG" 2>&1
echo "    rc=$? $(date -Is)" | tee -a "$LOG"

echo ">>> [2/3] seed replicates on fold 0 (seeds 1,2)" | tee -a "$LOG"
for s in 1 2; do
  python3 run_cv3.py --folds 0 --seed "$s" --eval_cap 1000 --full_cap 2000 \
     --no_qrs --out cv4_metrics.csv >> "$LOG" 2>&1
done
echo "    rc=$? $(date -Is)" | tee -a "$LOG"

echo ">>> [3/3] sensing-matrix families on fold 0" | tee -a "$LOG"
for p in bernoulli sparse_binary; do
  python3 run_cv3.py --folds 0 --phi "$p" --eval_cap 1000 --full_cap 2000 \
     --no_qrs --out cv4_metrics.csv >> "$LOG" 2>&1
done
echo "    rc=$? $(date -Is)" | tee -a "$LOG"
echo "=== PHASE4 DONE $(date -Is) ===" | tee -a "$LOG"
