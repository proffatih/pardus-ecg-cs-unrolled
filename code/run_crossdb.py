"""
run_crossdb.py — cross-database generalisation (reviewer 2, "Statistics").

Train every decoder on MIT-BIH (360 Hz, lead MLII) and test, without any
retraining or recalibration, on a *different* database recorded with different
hardware, patients and sampling rate.  Test set: the PTB Diagnostic ECG
Database (PhysioNet, `ptbdb`, 1000 Hz, lead ii), resampled to 360 Hz and
windowed identically.  PTB is used rather than PTB-XL because the full PTB-XL
archive is ~2 GB per sampling rate; PTB gives an independent patient population
at a fraction of the download, and the point of the experiment is the domain
shift, not the corpus size.

Output: results_v2/crossdb.csv
"""
import os
import csv
import argparse
import numpy as np
import torch
import wfdb
from scipy.signal import resample_poly

from cs_common2 import N, CRS, FS, folds, m_for_cr, make_phi, summarize
import classical2 as C
import learned2 as L

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
WIN = os.path.join(ROOT, "data", "windows")
PTB = os.path.join(ROOT, "data", "ptbdb")
RES = os.path.join(ROOT, "results_v2")
os.makedirs(RES, exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def fetch_ptb(n_records=40):
    """Download a subset of PTB records (patient-disjoint from MIT-BIH)."""
    os.makedirs(PTB, exist_ok=True)
    recs = wfdb.get_record_list("ptbdb")
    picked, got = [], 0
    for r in recs:
        if got >= n_records:
            break
        try:
            wfdb.dl_database("ptbdb", PTB, records=[r], annotators=None)
            picked.append(r)
            got += 1
        except Exception as e:
            print(f"    skip {r}: {e}")
    return picked


def ptb_windows(max_rec=40, lead_name="ii"):
    """Load PTB, take lead ii, resample 1000 -> 360 Hz, z-norm, window."""
    out = {}
    for root, _, files in os.walk(PTB):
        for fn in sorted(files):
            if not fn.endswith(".hea"):
                continue
            base = os.path.join(root, fn[:-4])
            try:
                sig, fields = wfdb.rdsamp(base)
            except Exception:
                continue
            names = [s.lower() for s in fields["sig_name"]]
            if lead_name not in names:
                continue
            x = sig[:, names.index(lead_name)].astype(np.float64)
            x = x[np.isfinite(x)]
            if len(x) < fields["fs"] * 10:
                continue
            x = resample_poly(x, int(FS), int(fields["fs"]))      # 1000 -> 360
            x = (x - x.mean()) / (x.std() + 1e-8)
            nw = len(x) // N
            if nw < 5:
                continue
            key = os.path.relpath(base, PTB)
            out[key] = x[: nw * N].reshape(nw, N).astype(np.float32)
            if len(out) >= max_rec:
                return out
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--train_cap", type=int, default=30000)
    ap.add_argument("--fetch", type=int, default=40)
    args = ap.parse_args()

    if not os.path.isdir(PTB) or not any(f.endswith(".hea")
                                         for _, _, fs in os.walk(PTB) for f in fs):
        print("downloading PTB subset ...", flush=True)
        fetch_ptb(args.fetch)

    ptb = ptb_windows(args.fetch)
    print(f"PTB test records: {len(ptb)}  windows: {sum(len(v) for v in ptb.values())}",
          flush=True)
    if not ptb:
        print("NO PTB DATA — skipping cross-database experiment")
        return

    tr, te_mit = folds()[0]
    Xtr = np.concatenate([np.load(os.path.join(WIN, f"{r}.npy")) for r in tr])
    rng = np.random.default_rng(0)
    if len(Xtr) > args.train_cap:
        Xtr = Xtr[rng.choice(len(Xtr), args.train_cap, replace=False)]
    Xtr = Xtr.astype(np.float32)
    Xmit = np.concatenate([np.load(os.path.join(WIN, f"{r}.npy")) for r in te_mit])[:4000]
    W = C.wavelet_matrix()

    f = open(os.path.join(RES, "crossdb.csv"), "w", newline="")
    w = csv.writer(f)
    w.writerow(["cr", "m", "method", "test_db", "record", "PRDN", "SNR", "n_win"])

    for cr in CRS:
        m = m_for_cr(cr)
        Phi = make_phi(m, kind="gaussian")
        lam, _ = C.tune_lambda(Xmit[:500] @ Phi.T, Xmit[:500], Phi, W, C.fista_l1)
        nets = {}
        for name, ctor in (("istanet", lambda: L.ISTANetPlus(Phi, 8, 32)),
                           ("dilated", lambda: L.DilatedUnrolled(Phi, 6, 32)),
                           ("hua", lambda: L.HuaNet(Phi=Phi, n=N, m=m, sensing="fixed"))):
            net, _ = L.train_model(ctor(), Xtr, Phi, epochs=args.epochs, dev=DEV)
            nets[name] = net
        print(f"  CR={cr}: trained {list(nets)}", flush=True)

        for db, data in (("mitbih", {"mit_pool": Xmit}), ("ptb", ptb)):
            for rec, Xr in data.items():
                Xr = Xr[:2000]
                Y = Xr @ Phi.T
                res = {"fista": C.fista_l1(Y, Phi, W, lam=lam),
                       "bsbl": C.bsbl_bo(Y[:120], Phi, block_size=16, max_iter=80)}
                for nm, net in nets.items():
                    res[nm] = L.reconstruct(net, Xr, Phi, dev=DEV)
                for nm, Xh in res.items():
                    ref = Xr[:len(Xh)]
                    s = summarize(ref, Xh)
                    w.writerow([cr, m, nm, db, rec, f"{s['PRDN']:.4f}",
                                f"{s['SNR']:.4f}", s["n_win"]])
                f.flush()
        del nets
        torch.cuda.empty_cache()
    f.close()
    print("done ->", os.path.join(RES, "crossdb.csv"))


if __name__ == "__main__":
    main()
