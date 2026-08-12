"""
prepare_data.py
Load MIT-BIH Arrhythmia Database records (downloaded via wfdb), take channel 0
(MLII), z-normalise per record, and segment into non-overlapping fixed-length
windows of N=256 samples (~0.71 s at 360 Hz). Records are split into a train set
and a held-out test set (record-wise, no leakage). Outputs .npy arrays.

Run:  python prepare_data.py
"""
import os
import numpy as np
import wfdb

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data", "mitdb")
OUT = os.path.join(ROOT, "data")

N = 256  # window length (signal dimension)

# Record-wise split (no patient/record overlap between train and test).
TRAIN_RECS = ["100", "101", "103", "105", "106", "107", "108", "109", "111"]
TEST_RECS = ["112", "115", "117", "118", "119"]


def load_windows(records):
    segs = []
    for rec in records:
        path = os.path.join(DATA, rec)
        sig, fields = wfdb.rdsamp(path)
        x = sig[:, 0].astype(np.float64)  # channel 0 (MLII)
        # per-record z-normalisation (robust to baseline/gain differences)
        x = (x - x.mean()) / (x.std() + 1e-8)
        n_win = len(x) // N
        x = x[: n_win * N].reshape(n_win, N)
        segs.append(x)
    return np.concatenate(segs, axis=0)


def main():
    rng = np.random.default_rng(0)
    Xtr = load_windows(TRAIN_RECS)
    Xte = load_windows(TEST_RECS)
    # shuffle training windows
    rng.shuffle(Xtr)
    np.save(os.path.join(OUT, "X_train.npy"), Xtr.astype(np.float32))
    np.save(os.path.join(OUT, "X_test.npy"), Xte.astype(np.float32))
    print(f"Sampling frequency assumed 360 Hz (MIT-BIH).")
    print(f"Window length N = {N}")
    print(f"Train windows: {Xtr.shape}  (records {TRAIN_RECS})")
    print(f"Test  windows: {Xte.shape}  (records {TEST_RECS})")


if __name__ == "__main__":
    main()
