"""
prepare_data2.py — build per-record window arrays for ALL non-paced MIT-BIH
records (44 of 48; 102/104/107/217 are paced and excluded).

v1 used 14 records with one fixed 9/5 split, which the reviewers correctly
flagged as too small a statistical base.  Here every record is windowed once
and cached as its own .npy, so any record-wise fold can be composed without
re-reading the signals.

Window: N = 256 samples (~0.71 s at 360 Hz), non-overlapping, channel 0 (MLII
for most records), per-record z-normalisation.

Run:  python3 prepare_data2.py
"""
import os
import numpy as np
import wfdb

from cs_common2 import N, RECORDS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "mitdb")
OUT = os.path.join(ROOT, "data", "windows")


def window_record(rec, raw=RAW, n=N):
    sig, fields = wfdb.rdsamp(os.path.join(raw, rec))
    x = sig[:, 0].astype(np.float64)
    x = (x - x.mean()) / (x.std() + 1e-8)
    nw = len(x) // n
    return x[: nw * n].reshape(nw, n).astype(np.float32), fields


def main():
    os.makedirs(OUT, exist_ok=True)
    tot = 0
    missing = []
    for rec in RECORDS:
        if not os.path.exists(os.path.join(RAW, rec + ".dat")):
            missing.append(rec)
            continue
        X, f = window_record(rec)
        np.save(os.path.join(OUT, f"{rec}.npy"), X)
        tot += X.shape[0]
        print(f"  {rec}: {X.shape[0]:5d} windows  (fs={f['fs']}, lead={f['sig_name'][0]})")
    print(f"\n{len(RECORDS) - len(missing)}/{len(RECORDS)} records, {tot} windows total")
    if missing:
        print(f"MISSING (not downloaded): {missing}")


if __name__ == "__main__":
    main()
