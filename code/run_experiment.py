"""
run_experiment.py
Main compressed-sensing ECG experiment.

For each compression ratio CR in {0.25, 0.50, 0.75} (fraction of measurements
discarded), compare three reconstruction methods on the held-out MIT-BIH test
records:
    1. ISTA-L1   : classical wavelet-domain L1 (iterative soft-thresholding)
    2. OMP       : classical greedy Orthogonal Matching Pursuit
    3. ISTA-Net  : learned deep-unrolled network (trained per CR)

Outputs:
    results/metrics_by_cr.csv     : mean +/- std PRD, SNR, RMSE per method/CR
    results/examples_cr050.npz    : example windows (orig + 3 reconstructions)
    results/training_curves.csv   : learned-net training loss per CR

Reproducible: fixed seeds, record-wise train/test split (see prepare_data.py).
"""
import os
import csv
import time
import numpy as np
import torch

from cs_common import N, CRS, make_phi, m_for_cr, summarize
from classical import wavelet_matrix, ista_l1, omp
from learned import ISTANet, train_model, reconstruct

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
os.makedirs(RES, exist_ok=True)

# Per-CR classical hyperparameters (tuned on a small held-out subset, see report)
ISTA_LAM = {0.25: 0.02, 0.50: 0.06, 0.75: 0.08}
OMP_FRAC = {0.25: 0.5, 0.50: 0.5, 0.75: 0.3}  # n_nonzero = frac * M

# Limit OMP test count (per-window greedy solve is slow); evaluate on a fixed
# random subset of the test set for OMP, full test set for the others.
OMP_EVAL = 600
SEED = 0


def main():
    t0 = time.time()
    Xtr = np.load(os.path.join(ROOT, "data", "X_train.npy"))
    Xte = np.load(os.path.join(ROOT, "data", "X_test.npy"))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={dev}  Xtr={Xtr.shape}  Xte={Xte.shape}")

    W = wavelet_matrix()
    rng = np.random.default_rng(SEED)
    omp_idx = rng.choice(Xte.shape[0], size=min(OMP_EVAL, Xte.shape[0]), replace=False)
    Xte_omp = Xte[omp_idx]

    rows = []
    curves = []
    examples = {}

    for cr in CRS:
        M = m_for_cr(cr)
        print(f"\n=== CR={cr:.2f}  (keep M={M}/{N} measurements) ===")
        Phi = make_phi(M, seed=SEED)
        Yte = Xte @ Phi.T
        Yte_omp = Xte_omp @ Phi.T

        # --- classical ISTA-L1 (full test set) ---
        ti = time.time()
        x_ista = ista_l1(Yte, Phi, W, lam=ISTA_LAM[cr], n_iter=400)
        s_ista = summarize(Xte, x_ista)
        print(f"  ISTA-L1  PRD={s_ista['PRD']:.2f}  SNR={s_ista['SNR']:.2f}  "
              f"({time.time()-ti:.1f}s)")

        # --- classical OMP (subset) ---
        ti = time.time()
        k = max(1, int(OMP_FRAC[cr] * M))
        x_omp = omp(Yte_omp, Phi, W, n_nonzero=k)
        s_omp = summarize(Xte_omp, x_omp)
        print(f"  OMP      PRD={s_omp['PRD']:.2f}  SNR={s_omp['SNR']:.2f}  "
              f"k={k}  ({time.time()-ti:.1f}s)")

        # --- learned ISTA-Net (full data) ---
        ti = time.time()
        torch.manual_seed(SEED)
        model = ISTANet(Phi, n_stages=9, n_feat=32)
        model = train_model(model, Xtr, dev, epochs=60, batch=256, lr=1e-3,
                            gamma=0.01, seed=SEED, verbose=False)
        x_net = reconstruct(model, Xte, dev)
        s_net = summarize(Xte, x_net)
        print(f"  ISTA-Net PRD={s_net['PRD']:.2f}  SNR={s_net['SNR']:.2f}  "
              f"({time.time()-ti:.1f}s)")

        for name, s in [("ISTA-L1", s_ista), ("OMP", s_omp), ("ISTA-Net", s_net)]:
            rows.append({
                "CR": cr, "keep_M": M, "method": name,
                "PRD": round(s["PRD"], 3), "PRD_std": round(s["PRD_std"], 3),
                "SNR": round(s["SNR"], 3), "SNR_std": round(s["SNR_std"], 3),
                "RMSE": round(s["RMSE"], 5),
            })

        if abs(cr - 0.50) < 1e-9:
            # save a few example windows for the qualitative figure
            sel = [3, 17, 42]
            examples = {
                "orig": Xte[sel],
                "ista": x_ista[sel],
                "net": x_net[sel],
                "Phi_M": M,
            }
            # OMP example: recompute on those windows
            x_omp_ex = omp(Xte[sel] @ Phi.T, Phi, W, n_nonzero=k)
            examples["omp"] = x_omp_ex

    # write CSVs
    with open(os.path.join(RES, "metrics_by_cr.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    np.savez(os.path.join(RES, "examples_cr050.npz"), **examples)
    print(f"\nWrote results/metrics_by_cr.csv and examples_cr050.npz")
    print(f"Total time {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
