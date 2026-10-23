"""merge_cv.py — build cv_metrics_final.csv.

Classical decoders are unaffected by the two fixes (their rows come from the
first run); learned decoders are taken from the early-stopped re-run.  Rows are
selected per method so that every number in the paper comes from the protocol
described in the paper.
"""
import os
import pandas as pd

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results_v2")
CLASSICAL = {"ista", "fista", "omp", "ksvd_omp", "bsbl"}
LEARNED = {"istanet", "hua", "dilated"}

a = pd.read_csv(os.path.join(RES, "cv_metrics.csv"))
b = pd.read_csv(os.path.join(RES, "cv_metrics_es.csv"))
out = pd.concat([a[a.method.isin(CLASSICAL)], b[b.method.isin(LEARNED)]],
                ignore_index=True)
out = out.drop_duplicates(subset=["fold", "cr", "phi", "bits", "seed",
                                  "method", "record"], keep="last")
out.to_csv(os.path.join(RES, "cv_metrics_final.csv"), index=False)
print(f"{len(out)} rows -> cv_metrics_final.csv")
print(out.groupby('method').record.count().to_string())
