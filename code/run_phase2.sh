#!/usr/bin/env bash
# run_phase2.sh — fairness re-runs after two methodological fixes:
#   (1) learned decoders trained to convergence (early stopping) instead of a
#       fixed 25-epoch budget, so a published decoder is not judged on fewer
#       epochs than its authors used;
#   (2) unrolled step size initialised at 1/||Phi||^2, which the sparse binary
#       sensing matrix needs (fixed 0.1 diverges there).
set -u
cd "$(dirname "$0")"
LOG=../results_v2/run_phase2.log
echo "=== PHASE2 START $(date -Is) ===" | tee -a $LOG

echo ">>> [1/3] learned decoders, early stopping, 5 folds" | tee -a $LOG
python3 run_cv.py --methods istanet,hua,dilated --patience 12 --max_epochs 120 \
        --out cv_metrics_es.csv >> $LOG 2>&1
echo "    rc=$? $(date -Is)" | tee -a $LOG

echo ">>> [2/3] sensing-matrix variability with the corrected step size" | tee -a $LOG
for p in bernoulli sparse_binary; do
  python3 run_cv.py --folds 0 --phi $p --patience 12 --max_epochs 120 \
    --methods fista,omp,bsbl,istanet --out cv_metrics_es.csv >> $LOG 2>&1
done
echo "    rc=$? $(date -Is)" | tee -a $LOG

echo ">>> [3/3] quantisation / energy re-run (sparse binary was diverging)" | tee -a $LOG
mv -f ../results_v2/quant_energy.csv ../results_v2/quant_energy_BROKEN.csv 2>/dev/null
mv -f ../results_v2/complexity.csv ../results_v2/complexity_old.csv 2>/dev/null
python3 run_extras.py --what quant --epochs 40 >> $LOG 2>&1
echo "    rc=$? $(date -Is)" | tee -a $LOG
echo "=== PHASE2 DONE $(date -Is) ===" | tee -a $LOG
