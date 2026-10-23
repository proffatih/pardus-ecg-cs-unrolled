"""
morph_eval.py — morphology endpoint, complementing the beat-detection endpoint.

Beat detection turns out to be almost insensitive to reconstruction error (see
qrs_eval.py): even a PRD of 44 % barely moves the detector's F1.  That result is
real but it is also the *easiest* clinical readout, and a letter that stopped
there would be choosing a favourable endpoint.  This module measures what beat
detection does not: the fidelity of the beat waveform itself, through three
quantities a monitor actually reads off the trace.

For every reference beat (annotation-aligned, so detector errors do not enter):

  ST level      amplitude at J+60 ms relative to the PR-segment baseline
                (the quantity used for ischaemia monitoring);
  R amplitude   peak amplitude of the QRS complex;
  QRS width     duration above 50 % of the R peak.

Each is computed on the original and on the reconstruction, and the error is
reported in the signal's own normalised units (the recordings are per-record
z-normalised), together with the fraction of beats whose ST level moves by more
than 0.1 (a tenth of a standard deviation of the record).

Output: results_v2/morph.csv
"""
import os
import csv
import argparse
import numpy as np
import torch

from cs_common2 import N, FS, CRS, folds, m_for_cr, make_phi, quantize
import classical2 as C
import learned2 as L
import qrs_eval as Q
from run_cv3 import reconstruct_all, CLASSICAL, LEARNED

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIN = os.path.join(ROOT, "data", "windows")
RES = os.environ.get("RUN_OUT", "/home/pardus/papers_runs/0027")
MOD = os.environ.get("RUN_MODELS", os.path.join(ROOT, "models_v2"))
DEV = "cuda" if torch.cuda.is_available() else "cpu"

PR_OFF = int(0.08 * FS)      # 80 ms before the fiducial point -> PR baseline
J_OFF = int(0.06 * FS)       # J+60 ms, measured from the fiducial point
WIN_L, WIN_R = int(0.10 * FS), int(0.12 * FS)


def beat_features(x, peaks):
    """ST level, R amplitude and QRS width for each annotated beat."""
    st, ramp, width = [], [], []
    for p in peaks:
        if p - PR_OFF - 5 < 0 or p + J_OFF + 5 >= len(x):
            continue
        base = float(np.mean(x[p - PR_OFF - 5: p - PR_OFF + 5]))
        st.append(float(x[p + J_OFF]) - base)
        a, b = max(0, p - WIN_L), min(len(x), p + WIN_R)
        seg = x[a:b] - base
        r = float(np.max(np.abs(seg)))
        ramp.append(r)
        above = np.abs(seg) > 0.5 * r
        width.append(float(np.sum(above)) / FS * 1000.0)       # ms
    return np.array(st), np.array(ramp), np.array(width)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=120.0)
    ap.add_argument("--eval_cap", type=int, default=1000)
    args = ap.parse_args()

    methods = CLASSICAL + LEARNED
    W = C.wavelet_matrix()
    f = open(os.path.join(RES, "morph.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["fold", "cr", "method", "record", "n_beats",
                "st_mae", "st_frac_gt_0p1", "ramp_mape", "qrs_width_mae_ms"])

    for fi, (tr_recs, te_recs) in enumerate(folds()):
        Xtr = np.concatenate([np.load(os.path.join(WIN, f"{r}.npy")) for r in tr_recs])
        rng = np.random.default_rng(0)
        if len(Xtr) > 30000:
            Xtr = Xtr[rng.choice(len(Xtr), 30000, replace=False)]
        Xtr = Xtr.astype(np.float32)
        n_val = min(2000, len(Xtr) // 10)
        Xval, Xfit = Xtr[:n_val], Xtr[n_val:]
        D_ksvd = C.ksvd_fit(Xfit, n_atoms=256, sparsity=16, seed=0)

        for cr in CRS:
            m = m_for_cr(cr)
            Phi = make_phi(m, seed=0, kind="gaussian")
            Yval = quantize(Xval @ Phi.T, None)
            lam = {"ista": C.tune_lambda(Yval, Xval, Phi, W, C.ista_l1)[0],
                   "fista": C.tune_lambda(Yval, Xval, Phi, W, C.fista_l1)[0]}
            nets = {}
            for name in LEARNED:
                tag = f"{name}_f{fi}_cr{int(cr*100)}_s0_gaussian"
                path = os.path.join(MOD, tag + ".pt")
                if not os.path.exists(path):
                    continue
                if name == "istanet":
                    net = L.ISTANetPlus(Phi, n_stages=8, n_feat=32)
                elif name == "dilated":
                    net = L.DilatedUnrolled(Phi, n_stages=6, n_feat=32)
                else:
                    net = L.HuaNet(Phi=Phi, n=N, m=m, sensing="fixed")
                net.load_state_dict(torch.load(path, map_location=DEV))
                nets[name] = net.to(DEV).eval()

            for rec in te_recs:
                seg, ref, fs = Q.load_segment(rec, seconds=args.seconds)
                if seg is None or len(ref) < 5:
                    continue
                x0 = np.concatenate(seg)
                st0, ra0, wd0 = beat_features(x0, ref)
                if len(st0) < 3:
                    continue
                res = reconstruct_all(seg, Phi, W, D_ksvd, nets, lam, m, cr,
                                      None, methods)
                for meth, (Xh, _) in res.items():
                    xr = np.concatenate(Xh)
                    st1, ra1, wd1 = beat_features(xr, ref)
                    k = min(len(st0), len(st1))
                    st_mae = float(np.mean(np.abs(st0[:k] - st1[:k])))
                    st_frac = float(np.mean(np.abs(st0[:k] - st1[:k]) > 0.1))
                    ramp = float(np.mean(np.abs(ra0[:k] - ra1[:k]) /
                                         (np.abs(ra0[:k]) + 1e-9)) * 100)
                    wmae = float(np.mean(np.abs(wd0[:k] - wd1[:k])))
                    w.writerow([fi, cr, meth, rec, k, f"{st_mae:.5f}",
                                f"{st_frac:.4f}", f"{ramp:.3f}", f"{wmae:.3f}"])
                f.flush()
                print(f"  fold{fi} CR={cr} rec {rec} ok", flush=True)
            del nets
            torch.cuda.empty_cache()
    f.close()
    print("done ->", os.path.join(RES, "morph.csv"))


if __name__ == "__main__":
    main()
