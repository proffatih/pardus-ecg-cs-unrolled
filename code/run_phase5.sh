#!/usr/bin/env bash
# run_phase5.sh — seed replicates across ALL folds for the learned decoders.
#
# Why: the fold-0 seed study showed that ISTA-Net+ is highly sensitive to the
# training seed (PRD 9.79 / 12.99 / 16.18 at CR = 0.75, s.d. 3.19), while the
# two published decoders are stable (s.d. 0.26 and 0.80).  The margin by which
# ISTA-Net+ beats BSBL at CR = 0.75 is 1.60 PRD, i.e. *smaller than its own
# seed spread*, so a single-seed table cannot support that comparison.  Every
# learned decoder is therefore retrained with three seeds on every fold, and
# the main table is built from seed-averaged per-record values.
#
# Classical decoders are deterministic given the sensing matrix and are not
# repeated (their only randomness, Phi, is covered by the matrix-family study).
set -u
cd "$(dirname "$0")"
export RUN_OUT=/home/pardus/papers_runs/0027
export BSBL_GPU=1
LOG=$RUN_OUT/run_phase5.log
mkdir -p "$RUN_OUT"
echo "=== PHASE5 START $(date -Is) ===" | tee -a "$LOG"

for s in 1 2; do
  echo ">>> seed $s, all folds, learned decoders" | tee -a "$LOG"
  python3 run_cv3.py --seed "$s" --methods istanet,hua,dilated \
     --eval_cap 1000 --full_cap 2000 --no_qrs \
     --out cv5_seeds.csv >> "$LOG" 2>&1
  echo "    rc=$? $(date -Is)" | tee -a "$LOG"
done
echo "=== PHASE5 DONE $(date -Is) ===" | tee -a "$LOG"
