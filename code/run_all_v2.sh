#!/usr/bin/env bash
# run_all_v2.sh — full v2 benchmark pipeline for the SPL resubmission.
set -u
cd "$(dirname "$0")"
LOG=../results_v2/run_all.log
mkdir -p ../results_v2
echo "=== START $(date -Is) ===" | tee -a $LOG

echo ">>> [1/5] main record-wise CV (44 records, 5 folds, 9 methods)" | tee -a $LOG
python3 run_cv.py --epochs 25 >> $LOG 2>&1
echo "    rc=$? $(date -Is)" | tee -a $LOG

echo ">>> [2/5] seed + sensing-matrix variability (fold 0, 3 extra seeds)" | tee -a $LOG
for s in 1 2 3; do
  python3 run_cv.py --folds 0 --seed $s --epochs 25 \
    --methods ista,fista,omp,ksvd_omp,bsbl,istanet,hua,dilated >> $LOG 2>&1
done
for p in bernoulli sparse_binary; do
  python3 run_cv.py --folds 0 --phi $p --epochs 25 \
    --methods fista,omp,bsbl,istanet >> $LOG 2>&1
done
echo "    rc=$? $(date -Is)" | tee -a $LOG

echo ">>> [3/5] quantisation / energy / complexity + ablation" | tee -a $LOG
python3 run_extras.py --what all --epochs 20 >> $LOG 2>&1
echo "    rc=$? $(date -Is)" | tee -a $LOG

echo ">>> [4/5] cross-database (train MIT-BIH -> test PTB)" | tee -a $LOG
python3 run_crossdb.py --epochs 25 >> $LOG 2>&1
echo "    rc=$? $(date -Is)" | tee -a $LOG

echo ">>> [5/7] training-budget control (fairness of re-implemented baselines)" | tee -a $LOG
python3 run_budget.py --epochs 25,50,100 >> $LOG 2>&1
echo "    rc=$? $(date -Is)" | tee -a $LOG

echo ">>> [6/7] figures" | tee -a $LOG
python3 make_figures2.py >> $LOG 2>&1
echo "    rc=$? $(date -Is)" | tee -a $LOG

echo ">>> [7/7] statistics + tables" | tee -a $LOG
python3 stats_v2.py >> $LOG 2>&1
python3 make_supplement.py >> $LOG 2>&1
python3 make_numbers.py >> $LOG 2>&1
echo "    rc=$? $(date -Is)" | tee -a $LOG
echo "=== DONE $(date -Is) ===" | tee -a $LOG
