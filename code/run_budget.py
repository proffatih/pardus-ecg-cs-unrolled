"""
run_budget.py — training-budget control (fairness of the learned baselines).

Reviewer 2's central complaint about the previous version was that the baselines
were handicapped relative to the learned decoder.  The rebuilt benchmark trains
every network under one identical protocol (25 epochs), but the published
decoders we re-implement were trained far longer in their original papers
([S2] reports 200 epochs).  Training a re-implemented competitor for fewer
epochs than its authors did would repeat the same error in the opposite
direction, so this control sweeps the training budget for all three networks and
reports whether the ranking depends on it.

Output: results_v2/budget.csv
"""
import os
import csv
import time
import argparse
import numpy as np
import torch

from cs_common2 import N, CRS, folds, m_for_cr, make_phi, summarize
import learned2 as L

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIN = os.path.join(ROOT, "data", "windows")
RES = os.path.join(ROOT, "results_v2")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", default="25,50,100")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--train_cap", type=int, default=28000)
    ap.add_argument("--test_cap", type=int, default=3000)
    args = ap.parse_args()
    budgets = [int(e) for e in args.epochs.split(",")]

    tr, te = folds()[args.fold]
    rng = np.random.default_rng(0)
    Xtr = np.concatenate([np.load(os.path.join(WIN, f"{r}.npy")) for r in tr])
    if len(Xtr) > args.train_cap:
        Xtr = Xtr[rng.choice(len(Xtr), args.train_cap, replace=False)]
    Xtr = Xtr.astype(np.float32)
    Xte = np.concatenate([np.load(os.path.join(WIN, f"{r}.npy"))
                          for r in te])[:args.test_cap].astype(np.float32)

    f = open(os.path.join(RES, "budget.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["cr", "m", "method", "epochs", "PRDN", "SNR", "train_s"])
    for cr in CRS:
        m = m_for_cr(cr)
        Phi = make_phi(m, kind="gaussian")
        for name in ("istanet", "hua", "hua_learned", "dilated"):
            for ep in budgets:
                if name == "istanet":
                    net = L.ISTANetPlus(Phi, 8, 32)
                elif name == "dilated":
                    net = L.DilatedUnrolled(Phi, 6, 32)
                elif name == "hua":
                    net = L.HuaNet(Phi=Phi, n=N, m=m, sensing="fixed")
                else:
                    net = L.HuaNet(Phi=None, n=N, m=m, sensing="learned")
                t0 = time.time()
                net, _ = L.train_model(net, Xtr, Phi, epochs=ep, dev=DEV, seed=0)
                dt = time.time() - t0
                s = summarize(Xte, L.reconstruct(net, Xte, Phi, dev=DEV))
                w.writerow([cr, m, name, ep, f"{s['PRDN']:.4f}",
                            f"{s['SNR']:.4f}", f"{dt:.0f}"])
                f.flush()
                print(f"  CR={cr} {name:12s} {ep:3d}ep  PRD={s['PRDN']:6.2f}  "
                      f"({dt:.0f}s)", flush=True)
                del net
                torch.cuda.empty_cache()
    f.close()


if __name__ == "__main__":
    main()
