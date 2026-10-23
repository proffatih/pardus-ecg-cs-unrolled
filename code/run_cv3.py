"""
run_cv3.py — definitive benchmark run.

Three changes over run_cv.py, all of them things a careful reviewer would
otherwise raise:

 1. **Common evaluation windows.**  run_cv.py scored the per-window solvers
    (OMP, K-SVD+OMP, BSBL) on a 120-window subset and the fast solvers on 2000
    windows, then compared the per-record means as if they were paired.  Here
    *every* decoder is scored on exactly the same `--eval_cap` windows of each
    record; the fast decoders are additionally scored on the full record and
    reported separately (scope = "full") for reference.

 2. **Diagnostic endpoint.**  A QRS detector is run on a contiguous
    reconstructed stretch of each test record and scored against the reference
    beat annotations, so "diagnostically usable" is measured rather than
    inferred from PRD (see qrs_eval.py).

 3. **Trained models are saved**, so the reconstructions used for the figures
    and for the diagnostic endpoint come from exactly the evaluated models.

Output: results_v2/cv3_metrics.csv, results_v2/cv3_qrs.csv, models_v2/*.pt
"""
import os
import csv
import time
import argparse
import numpy as np
import torch

from cs_common2 import (N, CRS, folds, m_for_cr, make_phi, quantize, summarize)
import classical2 as C
import learned2 as L
import qrs_eval as Q

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIN = os.path.join(ROOT, "data", "windows")
# Run outputs go OUTSIDE the git working tree by default: a `git pull` merge
# triggered from another machine replaces tracked files, which detaches the
# inode a long-running job is writing to and silently discards everything it
# writes afterwards (this cost folds 1-4 of the first phase-3 run).
RES = os.environ.get("RUN_OUT", "/home/pardus/papers_runs/0027")
MOD = os.environ.get("RUN_MODELS", os.path.join(ROOT, "models_v2"))
os.makedirs(RES, exist_ok=True)
os.makedirs(MOD, exist_ok=True)

CLASSICAL = ["ista", "fista", "omp", "ksvd_omp", "bsbl"]
LEARNED = ["istanet", "hua", "dilated"]
FAST = {"ista", "fista", "istanet", "hua", "dilated"}
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def subsample(X, cap, seed=0):
    if len(X) <= cap:
        return X
    idx = np.random.default_rng(seed).choice(len(X), cap, replace=False)
    idx.sort()
    return X[idx]


def reconstruct_all(Xw, Phi, W, D_ksvd, nets, lam, m, cr, bits, methods):
    """Reconstruct the same window block with every requested decoder."""
    Y = quantize(Xw @ Phi.T, bits)
    k = max(1, int(0.5 * m)) if cr < 0.75 else max(1, int(0.3 * m))
    out = {}
    for meth in methods:
        t0 = time.time()
        if meth == "ista":
            Xh = C.ista_l1(Y, Phi, W, lam=lam["ista"])
        elif meth == "fista":
            Xh = C.fista_l1(Y, Phi, W, lam=lam["fista"])
        elif meth == "omp":
            Xh = C.omp_dict(Y, Phi, W, n_nonzero=k)
        elif meth == "ksvd_omp":
            Xh = C.omp_dict(Y, Phi, D_ksvd, n_nonzero=k)
        elif meth == "bsbl":
            Xh = C.bsbl_bo(Y, Phi, block_size=16, max_iter=80)
        elif meth in nets:
            Xh = L.reconstruct(nets[meth], Xw, Phi, dev=DEV, quant_bits=bits)
        else:
            continue
        out[meth] = (Xh, time.time() - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", default="all")
    ap.add_argument("--methods", default=",".join(CLASSICAL + LEARNED))
    ap.add_argument("--crs", default=",".join(str(c) for c in CRS))
    ap.add_argument("--phi", default="gaussian")
    ap.add_argument("--bits", default="none")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--max_epochs", type=int, default=120)
    ap.add_argument("--eval_cap", type=int, default=250)
    ap.add_argument("--full_cap", type=int, default=2000)
    ap.add_argument("--qrs_seconds", type=float, default=120.0)
    ap.add_argument("--no_qrs", action="store_true")
    ap.add_argument("--out", default="cv3_metrics.csv")
    args = ap.parse_args()

    methods = args.methods.split(",")
    crs = [float(c) for c in args.crs.split(",")]
    bits = None if args.bits == "none" else int(args.bits)
    all_folds = folds()
    fold_ids = (list(range(len(all_folds))) if args.folds == "all"
                else [int(f) for f in args.folds.split(",")])

    fm = open(os.path.join(RES, args.out), "a", newline="")
    wm = csv.writer(fm)
    if fm.tell() == 0:
        wm.writerow(["fold", "cr", "m", "phi", "bits", "seed", "method", "record",
                     "scope", "PRDN", "PRDN_std", "PRD_raw", "SNR", "RMSE",
                     "n_win", "sec"])
    fq = open(os.path.join(RES, "cv3_qrs.csv"), "a", newline="")
    wq = csv.writer(fq)
    if fq.tell() == 0:
        wq.writerow(["fold", "cr", "seed", "method", "record", "n_beats",
                     "se_orig", "ppv_orig", "f1_orig",
                     "se_rec", "ppv_rec", "f1_rec", "f1_drop"])

    W = C.wavelet_matrix()
    t_start = time.time()

    for fi in fold_ids:
        tr_recs, te_recs = all_folds[fi]
        Xtr = np.concatenate([np.load(os.path.join(WIN, f"{r}.npy"))
                              for r in tr_recs])
        rng = np.random.default_rng(args.seed)
        if len(Xtr) > 30000:
            Xtr = Xtr[rng.choice(len(Xtr), 30000, replace=False)]
        Xtr = Xtr.astype(np.float32)
        n_val = min(2000, len(Xtr) // 10)
        Xval, Xfit = Xtr[:n_val], Xtr[n_val:]
        print(f"\n=== fold {fi}: {len(tr_recs)} train rec / {len(Xfit)} win "
              f"| {len(te_recs)} test rec ===", flush=True)

        D_ksvd = None
        if "ksvd_omp" in methods:
            t0 = time.time()
            D_ksvd = C.ksvd_fit(Xfit, n_atoms=256, sparsity=16, seed=args.seed)
            print(f"  K-SVD: {time.time()-t0:.0f}s", flush=True)

        for cr in crs:
            m = m_for_cr(cr)
            Phi = make_phi(m, seed=args.seed, kind=args.phi)
            Yval = quantize(Xval @ Phi.T, bits)
            lam = {}
            lam["ista"], _ = C.tune_lambda(Yval, Xval, Phi, W, C.ista_l1)
            lam["fista"], _ = C.tune_lambda(Yval, Xval, Phi, W, C.fista_l1)
            print(f"  CR={cr} M={m}: lam_ISTA={lam['ista']} "
                  f"lam_FISTA={lam['fista']}", flush=True)

            nets = {}
            for name in LEARNED:
                if name not in methods:
                    continue
                tag = f"{name}_f{fi}_cr{int(cr*100)}_s{args.seed}_{args.phi}"
                path = os.path.join(MOD, tag + ".pt")
                if name == "istanet":
                    net = L.ISTANetPlus(Phi, n_stages=8, n_feat=32)
                elif name == "dilated":
                    net = L.DilatedUnrolled(Phi, n_stages=6, n_feat=32)
                else:
                    net = L.HuaNet(Phi=Phi, n=N, m=m, sensing="fixed")
                if os.path.exists(path):
                    net.load_state_dict(torch.load(path, map_location=DEV))
                    net = net.to(DEV).eval()
                    print(f"    loaded {name}", flush=True)
                else:
                    t0 = time.time()
                    net, _ = L.train_model(net, Xfit, Phi, dev=DEV,
                                           seed=args.seed, quant_bits=bits,
                                           patience=args.patience,
                                           max_epochs=args.max_epochs)
                    torch.save(net.state_dict(), path)
                    print(f"    trained {name}: {time.time()-t0:.0f}s "
                          f"({getattr(net,'epochs_run','?')} ep, "
                          f"best@{getattr(net,'best_epoch','?')})", flush=True)
                nets[name] = net

            for rec in te_recs:
                Xr = np.load(os.path.join(WIN, f"{rec}.npy"))
                # --- common window set: every decoder sees exactly these ---
                Xc = subsample(Xr, args.eval_cap, seed=args.seed)
                res = reconstruct_all(Xc, Phi, W, D_ksvd, nets, lam, m, cr,
                                      bits, methods)
                for meth, (Xh, dt) in res.items():
                    s = summarize(Xc, Xh)
                    wm.writerow([fi, cr, m, args.phi, args.bits, args.seed, meth,
                                 rec, "common", f"{s['PRDN']:.4f}",
                                 f"{s['PRDN_std']:.4f}", f"{s['PRD_raw']:.4f}",
                                 f"{s['SNR']:.4f}", f"{s['RMSE']:.6f}",
                                 s["n_win"], f"{dt:.2f}"])
                # --- fast decoders additionally on the full record ---
                Xf = subsample(Xr, args.full_cap, seed=args.seed)
                fast = [mth for mth in methods if mth in FAST]
                resf = reconstruct_all(Xf, Phi, W, D_ksvd, nets, lam, m, cr,
                                       bits, fast)
                for meth, (Xh, dt) in resf.items():
                    s = summarize(Xf, Xh)
                    wm.writerow([fi, cr, m, args.phi, args.bits, args.seed, meth,
                                 rec, "full", f"{s['PRDN']:.4f}",
                                 f"{s['PRDN_std']:.4f}", f"{s['PRD_raw']:.4f}",
                                 f"{s['SNR']:.4f}", f"{s['RMSE']:.6f}",
                                 s["n_win"], f"{dt:.2f}"])
                fm.flush()

                # --- diagnostic endpoint on a contiguous stretch ---
                if not args.no_qrs:
                    seg, ref, fs = Q.load_segment(rec, seconds=args.qrs_seconds)
                    if seg is not None and len(ref) > 5:
                        resq = reconstruct_all(seg, Phi, W, D_ksvd, nets, lam,
                                               m, cr, bits, methods)
                        for meth, (Xh, _) in resq.items():
                            sc = Q.score(seg, Xh, ref, fs)
                            wq.writerow([fi, cr, args.seed, meth, rec,
                                         sc["n_beats"],
                                         f"{sc['se_orig']:.4f}",
                                         f"{sc['ppv_orig']:.4f}",
                                         f"{sc['f1_orig']:.4f}",
                                         f"{sc['se_rec']:.4f}",
                                         f"{sc['ppv_rec']:.4f}",
                                         f"{sc['f1_rec']:.4f}",
                                         f"{sc['f1_drop']:.4f}"])
                        fq.flush()
                print(f"    rec {rec} done ({time.time()-t_start:.0f}s)",
                      flush=True)
            del nets
            torch.cuda.empty_cache()

    fm.close()
    fq.close()
    print(f"\nTOTAL {time.time()-t_start:.0f}s")


if __name__ == "__main__":
    main()
