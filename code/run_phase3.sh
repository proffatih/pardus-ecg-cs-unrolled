#!/usr/bin/env bash
# run_phase3.sh — definitive benchmark: common evaluation windows for every
# decoder, saved models, and a QRS-detection endpoint on contiguous signal.
set -u
cd "$(dirname "$0")"
LOG=../results_v2/run_phase3.log
echo "=== PHASE3 START $(date -Is) ===" | tee -a $LOG

echo ">>> [1/3] main run: 5 folds x 3 CR x 8 decoders, common windows + QRS" | tee -a $LOG
python3 run_cv3.py --eval_cap 250 --full_cap 2000 --qrs_seconds 120 >> $LOG 2>&1
echo "    rc=$? $(date -Is)" | tee -a $LOG

echo ">>> [2/3] seed replicates on fold 0 (seeds 1,2)" | tee -a $LOG
for s in 1 2; do
  python3 run_cv3.py --folds 0 --seed $s --eval_cap 250 --full_cap 2000 \
     --no_qrs >> $LOG 2>&1
done
echo "    rc=$? $(date -Is)" | tee -a $LOG

echo ">>> [3/3] sensing-matrix families on fold 0, common windows" | tee -a $LOG
for p in bernoulli sparse_binary; do
  python3 run_cv3.py --folds 0 --phi $p --eval_cap 250 --full_cap 2000 \
     --no_qrs >> $LOG 2>&1
done
echo "    rc=$? $(date -Is)" | tee -a $LOG
echo "=== PHASE3 DONE $(date -Is) ===" | tee -a $LOG
