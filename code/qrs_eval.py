"""
qrs_eval.py — diagnostic endpoint: does the reconstruction still support beat
detection?

Reviewer 2 objected, correctly, that a PRD figure does not license the phrase
"diagnostically usable".  PRD is a waveform-fidelity proxy; what a telemetry
link has to preserve is the clinical readout.  This module measures that
directly: a standard QRS detector is run on the reconstructed signal and its
detections are scored against the database's reference beat annotations, with
the same detector on the original signal as the ceiling.

Protocol
  * a contiguous stretch of each test record (default 120 s) is windowed
    exactly as in training, reconstructed window by window, and concatenated;
  * `wfdb.processing.xqrs_detect` is run on the original and on the
    reconstruction;
  * detections are matched to the reference annotations with a +-150 ms
    tolerance (the standard EC57 matching window), giving sensitivity, positive
    predictive value and F1;
  * the quantity of interest is the *drop* in F1 relative to the detector run
    on the original signal, which separates reconstruction loss from detector
    error.
"""
import os
import numpy as np
import wfdb
from wfdb import processing

from cs_common2 import N, FS

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "mitdb")

# AAMI beat annotation symbols (non-beat annotations are ignored)
BEAT_SYMS = set("NLRBAaJSVrFejnEf/Q?")
TOL_S = 0.15


def load_segment(rec, seconds=120.0, start_s=0.0, raw=RAW):
    """Return (windows, ref_sample_idx, offset) for a contiguous stretch."""
    sig, fields = wfdb.rdsamp(os.path.join(raw, rec))
    x = sig[:, 0].astype(np.float64)
    mu, sd = x.mean(), x.std() + 1e-8
    x = (x - mu) / sd                       # same per-record normalisation
    fs = fields["fs"]
    a0 = int(start_s * fs)
    n_win = int(seconds * fs) // N
    seg = x[a0: a0 + n_win * N]
    if len(seg) < n_win * N:
        return None, None, None
    ann = wfdb.rdann(os.path.join(raw, rec), "atr")
    keep = [(s, y) for s, y in zip(ann.sample, ann.symbol)
            if y in BEAT_SYMS and a0 <= s < a0 + n_win * N]
    ref = np.array([s - a0 for s, _ in keep], dtype=int)
    return seg.reshape(n_win, N).astype(np.float32), ref, fs


def match(ref, det, fs, tol_s=TOL_S):
    """Greedy one-to-one matching within +-tol; returns (TP, FP, FN)."""
    if len(ref) == 0:
        return 0, len(det), 0
    if len(det) == 0:
        return 0, 0, len(ref)
    tol = tol_s * fs
    used = np.zeros(len(det), bool)
    tp = 0
    for r in ref:
        d = np.abs(det.astype(float) - float(r))
        d[used] = np.inf
        j = int(np.argmin(d))
        if d[j] <= tol:
            used[j] = True
            tp += 1
    return tp, int((~used).sum()), int(len(ref) - tp)


def prf(tp, fp, fn):
    se = tp / (tp + fn) if tp + fn else 0.0
    pp = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * se * pp / (se + pp) if se + pp else 0.0
    return se, pp, f1


def detect(x, fs):
    try:
        return processing.xqrs_detect(sig=np.asarray(x, dtype=np.float64),
                                      fs=fs, verbose=False)
    except Exception:
        return np.array([], dtype=int)


def score(seg_windows, recon_windows, ref, fs):
    """F1 of the detector on original and on reconstruction."""
    x0 = np.concatenate(seg_windows)
    xr = np.concatenate(recon_windows)
    d0, dr = detect(x0, fs), detect(xr, fs)
    o = prf(*match(ref, d0, fs))
    r = prf(*match(ref, dr, fs))
    return {"n_beats": int(len(ref)),
            "se_orig": o[0], "ppv_orig": o[1], "f1_orig": o[2],
            "se_rec": r[0], "ppv_rec": r[1], "f1_rec": r[2],
            "f1_drop": o[2] - r[2]}


if __name__ == "__main__":
    # smoke test: detector on the untouched signal must be near-perfect
    for rec in ("100", "106", "208"):
        w, ref, fs = load_segment(rec, seconds=120.0)
        if w is None:
            print(f"{rec}: segment too short")
            continue
        s = score(w, w, ref, fs)
        print(f"{rec}: {s['n_beats']:4d} beats  F1(original)={s['f1_orig']:.4f} "
              f"Se={s['se_orig']:.4f} PPV={s['ppv_orig']:.4f}")
