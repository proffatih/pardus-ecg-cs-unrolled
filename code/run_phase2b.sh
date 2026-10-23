#!/usr/bin/env bash
# run_phase2b.sh — re-run of the step that died on a CUDA OOM in the
# early-stopping validation pass (now batched).
set -u
cd "$(dirname "$0")"
LOG=../results_v2/run_phase2b.log
echo "=== PHASE2B START $(date -Is) ===" | tee -a $LOG
python3 run_cv.py --methods istanet,hua,dilated --patience 12 --max_epochs 120 \
        --out cv_metrics_es.csv >> $LOG 2>&1
echo "    rc=$? $(date -Is)" | tee -a $LOG
echo "=== PHASE2B DONE $(date -Is) ===" | tee -a $LOG
