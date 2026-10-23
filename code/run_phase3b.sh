#!/usr/bin/env bash
# run_phase3b.sh — recover the evaluation blocks lost when a `git pull` merge
# from another machine replaced the output file while the job held it open
# (the job kept writing to the detached inode; folds 1-4 vanished).
#
# Training is NOT repeated: every model from the first run is on disk and is
# loaded.  Outputs go to $RUN_OUT, which is outside the git working tree.
set -u
cd "$(dirname "$0")"
export RUN_OUT=/home/pardus/papers_runs/0027
LOG=$RUN_OUT/run_phase3b.log
mkdir -p "$RUN_OUT"
echo "=== PHASE3B START $(date -Is) ===" | tee -a "$LOG"

echo ">>> [1/4] missing: fold 1, CR 0.75" | tee -a "$LOG"
python3 run_cv3.py --folds 1 --crs 0.75 --eval_cap 250 --full_cap 2000 \
        --qrs_seconds 120 >> "$LOG" 2>&1
echo "    rc=$? $(date -Is)" | tee -a "$LOG"

echo ">>> [2/4] missing: folds 2,3,4 (all CR)" | tee -a "$LOG"
python3 run_cv3.py --folds 2,3,4 --eval_cap 250 --full_cap 2000 \
        --qrs_seconds 120 >> "$LOG" 2>&1
echo "    rc=$? $(date -Is)" | tee -a "$LOG"

echo ">>> [3/4] seed replicates on fold 0 (seeds 1,2)" | tee -a "$LOG"
for s in 1 2; do
  python3 run_cv3.py --folds 0 --seed "$s" --eval_cap 250 --full_cap 2000 \
     --no_qrs >> "$LOG" 2>&1
done
echo "    rc=$? $(date -Is)" | tee -a "$LOG"

echo ">>> [4/4] sensing-matrix families on fold 0" | tee -a "$LOG"
for p in bernoulli sparse_binary; do
  python3 run_cv3.py --folds 0 --phi "$p" --eval_cap 250 --full_cap 2000 \
     --no_qrs >> "$LOG" 2>&1
done
echo "    rc=$? $(date -Is)" | tee -a "$LOG"
echo "=== PHASE3B DONE $(date -Is) ===" | tee -a "$LOG"
