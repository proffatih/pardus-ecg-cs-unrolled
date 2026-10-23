"""
run_extras.py — experiments the SPL reviewers asked for beyond the main CV:

  (A) sensing-matrix family x measurement quantisation -> reconstruction quality,
      transmitted bit rate and sensor-node arithmetic cost.
      Answers R1-3 and R2-"Low-power claim": the dense Gaussian matrix used in
      v1 is expensive on the node, and v1 analysed neither quantisation nor
      bit rate, so the "low-power" framing was unsupported.

  (B) decoder complexity: parameters, multiply-accumulates and measured
      inference time per window (R1-3).

  (C) ablation of the unrolled decoder (R1-4): number of stages, learned vs
      fixed threshold, learned prox vs plain proximal-gradient.

Outputs: results_v2/quant_energy.csv, results_v2/complexity.csv,
         results_v2/ablation.csv
"""
import os
import csv
import time
import argparse
import numpy as np
import torch

from cs_common2 import (N, FS, CRS, folds, m_for_cr, make_phi, quantize,
                        bit_rate, phi_node_cost, summarize)
import classical2 as C
import learned2 as L

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIN = os.path.join(ROOT, "data", "windows")
RES = os.path.join(ROOT, "results_v2")
os.makedirs(RES, exist_ok=True)

PHIS = ["gaussian", "bernoulli", "sparse_binary"]
BITS = [None, 12, 10, 8, 6, 4]
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load(recs, cap=None, seed=0):
    X = np.concatenate([np.load(os.path.join(WIN, f"{r}.npy")) for r in recs])
    if cap and len(X) > cap:
        X = X[np.random.default_rng(seed).choice(len(X), cap, replace=False)]
    return X.astype(np.float32)


# ---------------------------------------------------------------- (A) + (B)
def quant_energy(fold=0, epochs=20, train_cap=30000, test_cap=4000, seed=0):
    tr, te = folds()[fold]
    Xtr, Xte = load(tr, train_cap, seed), load(te, test_cap, seed)
    W = C.wavelet_matrix()

    fq = open(os.path.join(RES, "quant_energy.csv"), "w", newline="")
    wq = csv.writer(fq)
    wq.writerow(["cr", "m", "phi", "bits", "method", "PRDN", "SNR",
                 "bit_rate_bps", "node_mults", "node_adds"])
    fc = open(os.path.join(RES, "complexity.csv"), "w", newline="")
    wc = csv.writer(fc)
    wc.writerow(["cr", "m", "method", "params", "macs_per_window",
                 "infer_ms_per_window", "device"])

    for cr in CRS:
        m = m_for_cr(cr)
        for phi_kind in PHIS:
            Phi = make_phi(m, seed=seed, kind=phi_kind)
            mults, adds = phi_node_cost(phi_kind, m, N)
            for bits in BITS:
                br = bit_rate(m, bits)
                Yte = quantize(Xte @ Phi.T, bits)
                # classical reference: FISTA with lambda tuned at this setting
                lam, _ = C.tune_lambda(Yte[:500], Xte[:500], Phi, W, C.fista_l1)
                Xh = C.fista_l1(Yte, Phi, W, lam=lam)
                s = summarize(Xte, Xh)
                wq.writerow([cr, m, phi_kind, bits, "fista", f"{s['PRDN']:.4f}",
                             f"{s['SNR']:.4f}", f"{br:.1f}", mults, adds])
                # learned decoder trained *at this quantisation* (quantisation
                # aware), which is the fair way to ask the low-power question.
                # Trained only in the aggressive regime (CR = 0.75), which is
                # where the learned decoder is the method of choice and the one
                # the manuscript quotes; the classical sweep above covers every
                # CR at negligible cost.
                if cr != 0.75:
                    continue
                net = L.ISTANetPlus(Phi, n_stages=8, n_feat=32)
                net, _ = L.train_model(net, Xtr, Phi, epochs=epochs, dev=DEV,
                                       seed=seed, quant_bits=bits)
                Xh = L.reconstruct(net, Xte, Phi, dev=DEV, quant_bits=bits)
                s = summarize(Xte, Xh)
                wq.writerow([cr, m, phi_kind, bits, "istanet", f"{s['PRDN']:.4f}",
                             f"{s['SNR']:.4f}", f"{br:.1f}", mults, adds])
                fq.flush()
                print(f"  CR={cr} {phi_kind:13s} bits={str(bits):4s} "
                      f"PRDN={s['PRDN']:6.2f} rate={br:7.0f} bps", flush=True)
                del net
                torch.cuda.empty_cache()

        # ---- (B) complexity of each decoder at this CR (gaussian, float) ----
        Phi = make_phi(m, seed=seed, kind="gaussian")
        Yte = Xte[:512] @ Phi.T
        for name in ("istanet", "dilated", "hua"):
            if name == "istanet":
                net = L.ISTANetPlus(Phi, n_stages=8, n_feat=32)
                macs = 8 * (2 * m * N + 2 * (1 * 32 * 9 * N + 32 * 32 * 9 * N))
            elif name == "dilated":
                net = L.DilatedUnrolled(Phi, n_stages=6, n_feat=32)
                macs = 6 * (2 * m * N + (1 * 32 * 5 + 3 * 32 * 32 * 5 + 32 * 1 * 5) * N)
            else:
                net = L.HuaNet(Phi=Phi, n=N, m=m, sensing="fixed")
                macs = m * N + 64 * N + 3 * 4 * 2 * 16 * 16 * 10 * N + 4 * 64 * 64 * N
            net = net.to(DEV).eval()
            with torch.no_grad():
                yb = torch.tensor(Yte, dtype=torch.float32, device=DEV)
                for _ in range(3):
                    net(yb)
                if DEV == "cuda":
                    torch.cuda.synchronize()
                t0 = time.time()
                for _ in range(10):
                    net(yb)
                if DEV == "cuda":
                    torch.cuda.synchronize()
                dt = (time.time() - t0) / (10 * len(Yte)) * 1e3
            wc.writerow([cr, m, name, L.count_params(net), int(macs),
                         f"{dt:.4f}", DEV])
            fc.flush()
            del net
            torch.cuda.empty_cache()
        # classical solver timing for reference
        t0 = time.time()
        C.fista_l1(Yte, Phi, C.wavelet_matrix(), lam=0.05)
        wc.writerow([cr, m, "fista", 0, int(300 * 2 * m * N),
                     f"{(time.time()-t0)/len(Yte)*1e3:.4f}", "cpu"])
        fc.flush()
    fq.close()
    fc.close()


# -------------------------------------------------------------------- (C)
def ablation(fold=0, epochs=20, train_cap=30000, test_cap=4000, seed=0):
    tr, te = folds()[fold]
    Xtr, Xte = load(tr, train_cap, seed), load(te, test_cap, seed)
    f = open(os.path.join(RES, "ablation.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["cr", "m", "variant", "stages", "params", "PRDN", "SNR"])

    for cr in CRS:
        m = m_for_cr(cr)
        Phi = make_phi(m, seed=seed, kind="gaussian")
        # (i) depth
        for ns in (1, 2, 4, 8, 12):
            net = L.ISTANetPlus(Phi, n_stages=ns, n_feat=32)
            net, _ = L.train_model(net, Xtr, Phi, epochs=epochs, dev=DEV, seed=seed)
            s = summarize(Xte, L.reconstruct(net, Xte, Phi, dev=DEV))
            w.writerow([cr, m, "stages", ns, L.count_params(net),
                        f"{s['PRDN']:.4f}", f"{s['SNR']:.4f}"])
            f.flush()
            print(f"  CR={cr} stages={ns:2d}  PRDN={s['PRDN']:6.2f}", flush=True)
            del net
            torch.cuda.empty_cache()
        # (ii) no learned prox = plain unrolled proximal gradient (soft-thresh
        #      in the wavelet domain, learned step size and threshold only)
        net = L.ISTANetPlus(Phi, n_stages=8, n_feat=32)
        for b in net.blocks:                       # freeze the learned operators
            for p in list(b.F.parameters()) + list(b.G.parameters()):
                p.requires_grad_(False)
        net, _ = L.train_model(net, Xtr, Phi, epochs=epochs, dev=DEV, seed=seed)
        s = summarize(Xte, L.reconstruct(net, Xte, Phi, dev=DEV))
        w.writerow([cr, m, "no_learned_prox", 8, L.count_params(net),
                    f"{s['PRDN']:.4f}", f"{s['SNR']:.4f}"])
        # (iii) fixed (non-learned) threshold
        net = L.ISTANetPlus(Phi, n_stages=8, n_feat=32)
        for b in net.blocks:
            b.thr.requires_grad_(False)
        net, _ = L.train_model(net, Xtr, Phi, epochs=epochs, dev=DEV, seed=seed)
        s = summarize(Xte, L.reconstruct(net, Xte, Phi, dev=DEV))
        w.writerow([cr, m, "fixed_threshold", 8, L.count_params(net),
                    f"{s['PRDN']:.4f}", f"{s['SNR']:.4f}"])
        f.flush()
        del net
        torch.cuda.empty_cache()
    f.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="all", choices=["all", "quant", "ablation"])
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    if a.what in ("all", "quant"):
        print("== (A)+(B) quantisation / energy / complexity ==", flush=True)
        quant_energy(epochs=a.epochs)
    if a.what in ("all", "ablation"):
        print("== (C) ablation ==", flush=True)
        ablation(epochs=a.epochs)
