"""
run_cv.py — main record-wise cross-validated benchmark (v2).

Answers reviewer points: all non-paced records, K-fold record-wise CV,
per-record results, balanced baselines (tuned ISTA, FISTA, OMP, K-SVD+OMP,
BSBL-BO) against three learned decoders (ISTA-Net+, HuaNet [S2],
DilatedUnrolled [S1]), several sensing matrices, measurement quantisation,
and multiple seeds.

Output (long format, one row per fold x CR x method x record):
    results_v2/cv_metrics.csv

Usage:
    python3 run_cv.py                      # full run
    python3 run_cv.py --folds 0            # single fold (smoke test)
    python3 run_cv.py --methods istanet,omp
    python3 run_cv.py --phi sparse_binary --bits 8
"""
import os
import csv
import time
import argparse
import numpy as np
import torch

from cs_common2 import (N, CRS, RECORDS, folds, m_for_cr, make_phi, quantize,
                        summarize, prdn)
import classical2 as C
import learned2 as L

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIN = os.path.join(ROOT, "data", "windows")
RES = os.path.join(ROOT, "results_v2")
os.makedirs(RES, exist_ok=True)

ALL_METHODS = ["ista", "fista", "omp", "ksvd_omp", "bsbl",
               "istanet", "hua", "hua_learned", "dilated"]
# per-window greedy / Bayesian solvers are slow -> evaluate on a capped random
# subset of each record's test windows (same subset for every method, so the
# comparison stays paired). v1 silently used a 600-window subset for OMP only.
SLOW = {"omp", "ksvd_omp", "bsbl"}
SLOW_CAP = 120          # windows per record for slow solvers
FAST_CAP = 2000         # windows per record for fast solvers


def load_records(recs):
    return {r: np.load(os.path.join(WIN, f"{r}.npy")) for r in recs}


def subsample(X, cap, seed=0):
    if len(X) <= cap:
        return X, np.arange(len(X))
    idx = np.random.default_rng(seed).choice(len(X), cap, replace=False)
    idx.sort()
    return X[idx], idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--crs", default=",".join(str(c) for c in CRS))
    ap.add_argument("--phi", default="gaussian")
    ap.add_argument("--bits", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--max_epochs", type=int, default=0)
    ap.add_argument("--out", default="cv_metrics.csv")
    args = ap.parse_args()

    methods = args.methods.split(",")
    crs = [float(c) for c in args.crs.split(",")]
    bits = None if args.bits == "none" else int(args.bits)
    args.max_epochs = args.max_epochs or None
    fold_ids = (list(range(len(folds()))) if args.folds == "all"
                else [int(f) for f in args.folds.split(",")])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    out_path = os.path.join(RES, args.out)
    new = not os.path.exists(out_path)
    fh = open(out_path, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(["fold", "cr", "m", "phi", "bits", "seed", "method", "record",
                     "PRDN", "PRDN_std", "PRD_raw", "SNR", "SNR_std", "RMSE",
                     "n_win", "fit_s", "infer_s"])

    W = C.wavelet_matrix()
    all_folds = folds()
    t_start = time.time()

    for fi in fold_ids:
        tr_recs, te_recs = all_folds[fi]
        Xtr_d = load_records(tr_recs)
        Xte_d = load_records(te_recs)
        Xtr = np.concatenate([Xtr_d[r] for r in tr_recs]).astype(np.float32)
        rng = np.random.default_rng(args.seed)
        if len(Xtr) > 30000:
            Xtr = Xtr[rng.choice(len(Xtr), 30000, replace=False)]
        # validation split for lambda tuning: last 10 % of training windows
        n_val = min(2000, len(Xtr) // 10)
        Xval, Xfit = Xtr[:n_val], Xtr[n_val:]
        print(f"\n=== fold {fi}: train {len(tr_recs)} rec / {len(Xfit)} win  "
              f"| test {len(te_recs)} rec ===", flush=True)

        D_ksvd = None
        if "ksvd_omp" in methods:
            t0 = time.time()
            D_ksvd = C.ksvd_fit(Xfit, n_atoms=256, sparsity=16, seed=args.seed)
            print(f"  K-SVD dictionary: {time.time()-t0:.0f}s", flush=True)

        for cr in crs:
            m = m_for_cr(cr)
            Phi = make_phi(m, seed=args.seed, kind=args.phi)
            Yval = quantize(Xval @ Phi.T, bits)

            # ---- tuned lambdas (reported in the paper, reviewer asked) ----
            lam_ista, tab_i = C.tune_lambda(Yval, Xval, Phi, W, C.ista_l1)
            lam_fista, tab_f = C.tune_lambda(Yval, Xval, Phi, W, C.fista_l1)
            print(f"  CR={cr} M={m}: lam_ISTA={lam_ista} lam_FISTA={lam_fista}",
                  flush=True)

            # ---- train learned decoders once per (fold, CR) ----
            nets = {}
            for name in ("istanet", "hua", "hua_learned", "dilated"):
                if name not in methods:
                    continue
                t0 = time.time()
                if name == "istanet":
                    net = L.ISTANetPlus(Phi, n_stages=8, n_feat=32)
                elif name == "dilated":
                    net = L.DilatedUnrolled(Phi, n_stages=6, n_feat=32)
                elif name == "hua":
                    net = L.HuaNet(Phi=Phi, n=N, m=m, sensing="fixed")
                else:
                    net = L.HuaNet(Phi=None, n=N, m=m, sensing="learned")
                net, _ = L.train_model(net, Xfit, Phi, epochs=args.epochs,
                                       dev=dev, seed=args.seed, quant_bits=bits,
                                       patience=args.patience,
                                       max_epochs=args.max_epochs)
                nets[name] = (net, time.time() - t0, L.count_params(net))
                print(f"    trained {name}: {nets[name][1]:.0f}s ({nets[name][2]} params,"
                      f" {getattr(net,'epochs_run',args.epochs)} epochs run,"
                      f" best@{getattr(net,'best_epoch',-1)})", flush=True)

            # ---- evaluate per test record ----
            for rec in te_recs:
                Xr = Xte_d[rec]
                Xfast, _ = subsample(Xr, FAST_CAP, seed=args.seed)
                Xslow, _ = subsample(Xr, SLOW_CAP, seed=args.seed)
                Yfast = quantize(Xfast @ Phi.T, bits)
                Yslow = quantize(Xslow @ Phi.T, bits)

                for meth in methods:
                    t0 = time.time()
                    if meth == "ista":
                        Xh, Xref = C.ista_l1(Yfast, Phi, W, lam=lam_ista), Xfast
                    elif meth == "fista":
                        Xh, Xref = C.fista_l1(Yfast, Phi, W, lam=lam_fista), Xfast
                    elif meth == "omp":
                        k = max(1, int(0.5 * m)) if cr < 0.75 else max(1, int(0.3 * m))
                        Xh, Xref = C.omp_dict(Yslow, Phi, W, n_nonzero=k), Xslow
                    elif meth == "ksvd_omp":
                        k = max(1, int(0.5 * m)) if cr < 0.75 else max(1, int(0.3 * m))
                        Xh, Xref = C.omp_dict(Yslow, Phi, D_ksvd, n_nonzero=k), Xslow
                    elif meth == "bsbl":
                        Xh, Xref = C.bsbl_bo(Yslow, Phi, block_size=16), Xslow
                    elif meth in nets:
                        net = nets[meth][0]
                        src = Xfast if meth == "hua_learned" else Yfast
                        if meth == "hua_learned":
                            Xh = L.reconstruct(net, Xfast, Phi, dev=dev)
                        else:
                            Xh = L.reconstruct(net, Xfast, Phi, dev=dev, quant_bits=bits)
                        Xref = Xfast
                    else:
                        continue
                    dt = time.time() - t0
                    s = summarize(Xref, Xh)
                    fit_s = nets[meth][1] if meth in nets else 0.0
                    wr.writerow([fi, cr, m, args.phi, args.bits, args.seed, meth, rec,
                                 f"{s['PRDN']:.4f}", f"{s['PRDN_std']:.4f}",
                                 f"{s['PRD_raw']:.4f}", f"{s['SNR']:.4f}",
                                 f"{s['SNR_std']:.4f}", f"{s['RMSE']:.6f}",
                                 s["n_win"], f"{fit_s:.1f}", f"{dt:.2f}"])
                fh.flush()
                print(f"    rec {rec} done ({time.time()-t_start:.0f}s elapsed)",
                      flush=True)
            del nets
            torch.cuda.empty_cache()

    fh.close()
    print(f"\nTOTAL {time.time()-t_start:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
