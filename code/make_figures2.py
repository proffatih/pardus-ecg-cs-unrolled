"""
make_figures2.py — figures for the resubmission.

Fig. 1  PRD vs compression ratio for every decoder, with per-record spread
        (replaces the v1 figure that showed only three methods).
Fig. 2  PRD vs transmitted bit rate for the three sensing-matrix families and
        the quantiser word lengths (the "low-power" claim, now measured).
Fig. 3  Reconstructions at CR = 0.75 with a zoomed inset on the QRS complex and
        a per-sample error trace below each panel (reviewer 2, minor point 3).
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys as _sys, os as _os
for _c in (_os.environ.get("PRO_STYLE_DIR"),
           _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "..", "..", "_templates")):
    if _c and _os.path.isdir(_c):
        _sys.path.insert(0, _c)
try:
    from pro_style import apply as _pa
    _pa()
except Exception:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results_v2")
FIG = os.path.join(ROOT, "resubmission", "manuscript")
os.makedirs(FIG, exist_ok=True)

LABEL = {"ista": "ISTA", "fista": "FISTA", "omp": "OMP",
         "ksvd_omp": "K-SVD+OMP", "bsbl": "BSBL",
         "istanet": "ISTA-Net$^+$ (unrolled)", "hua": "Hua'22 (fixed $\\Phi$)",
         "hua_learned": "Hua'22 (learned $\\Phi$)", "dilated": "Erkoç'26 (dilated)"}
ORDER = ["ista", "fista", "omp", "ksvd_omp", "bsbl", "hua", "dilated", "istanet"]


def fig_main():
    cv = pd.read_csv(os.path.join(RES, "cv_metrics.csv"))
    cv = cv[(cv.phi == "gaussian") & (cv.bits == "none")]
    per = (cv.groupby(["cr", "method", "record"]).PRDN.mean().reset_index())
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for meth in ORDER:
        g = per[per.method == meth]
        if g.empty:
            continue
        crs = sorted(g.cr.unique())
        mu = [g[g.cr == c].PRDN.mean() for c in crs]
        sd = [g[g.cr == c].PRDN.std() for c in crs]
        ls = "-" if meth == "istanet" else "--"
        lw = 2.2 if meth == "istanet" else 1.2
        ax[0].errorbar(crs, mu, yerr=sd, marker="o", ms=3, ls=ls, lw=lw,
                       capsize=2, label=LABEL.get(meth, meth))
    ax[0].axhline(9.0, color="0.4", lw=0.9, ls=":")
    ax[0].text(0.255, 9.6, "PRD = 9 % band [17]", fontsize=6, color="0.35")
    ax[0].set_xlabel("compression ratio (fraction discarded)")
    ax[0].set_ylabel("PRD (%)")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize=5.4, ncol=2, frameon=False)

    # paired per-record difference, learned minus best classical
    cls = ["ista", "fista", "omp", "ksvd_omp", "bsbl"]
    for c in sorted(per.cr.unique()):
        w = per[per.cr == c].pivot(index="record", columns="method", values="PRDN")
        if "istanet" not in w:
            continue
        have = [m for m in cls if m in w.columns]
        d = (w["istanet"] - w[have].min(axis=1)).dropna()
        ax[1].scatter(np.full(len(d), c) + np.random.uniform(-0.012, 0.012, len(d)),
                      d, s=7, alpha=0.65)
        ax[1].plot([c - 0.03, c + 0.03], [d.median()] * 2, color="k", lw=1.6)
    ax[1].axhline(0, color="0.3", lw=0.9)
    ax[1].set_xlabel("compression ratio")
    ax[1].set_ylabel("PRD: unrolled $-$ best classical")
    ax[1].text(0.02, 0.96, "learned better below 0", transform=ax[1].transAxes,
               fontsize=6, va="top", color="0.3")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_main.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  fig_main")


def fig_rate():
    p = os.path.join(RES, "quant_energy.csv")
    if not os.path.exists(p):
        return
    qe = pd.read_csv(p)
    qe = qe[qe.cr == 0.75]
    fig, ax = plt.subplots(figsize=(3.5, 2.7))
    marks = {"gaussian": "o", "bernoulli": "s", "sparse_binary": "^"}
    for (phi, meth), g in qe.groupby(["phi", "method"]):
        g = g.sort_values("bit_rate_bps")
        ax.plot(g.bit_rate_bps, g.PRDN, marker=marks.get(phi, "o"), ms=3.5,
                lw=1.3, ls="-" if meth == "istanet" else "--",
                label=f"{phi.replace('_',' ')}, {'unrolled' if meth=='istanet' else 'FISTA'}")
    ax.axhline(9.0, color="0.4", lw=0.9, ls=":")
    ax.set_xlabel("transmitted bit rate (bit/s)")
    ax.set_ylabel("PRD (%)")
    ax.set_yscale("log")
    ax.legend(fontsize=5.4, frameon=False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_rate.{ext}"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  fig_rate")


def fig_examples():
    """Reconstructions at CR=0.75 with zoom inset and error trace."""
    import torch
    sys_path = os.path.dirname(os.path.abspath(__file__))
    _sys.path.insert(0, sys_path)
    from cs_common2 import m_for_cr, make_phi, folds, N
    import classical2 as C
    import learned2 as L

    tr, te = folds()[0]
    Xtr = np.concatenate([np.load(os.path.join(ROOT, "data", "windows", f"{r}.npy"))
                          for r in tr])[:30000].astype(np.float32)
    Xte = np.load(os.path.join(ROOT, "data", "windows", f"{te[0]}.npy"))[:512].astype(np.float32)
    cr = 0.75
    m = m_for_cr(cr)
    Phi = make_phi(m)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    W = C.wavelet_matrix()
    Y = Xte @ Phi.T
    lam, _ = C.tune_lambda(Y[:200], Xte[:200], Phi, W, C.fista_l1)
    # single-column figure: the contrast that carries the message is
    # fixed-basis recovery against the best learned decoder
    recs = {"FISTA (wavelet)": C.fista_l1(Y, Phi, W, lam=lam)}
    net, _ = L.train_model(L.DilatedUnrolled(Phi, 6, 32), Xtr, Phi,
                           patience=12, max_epochs=120, dev=dev)
    recs["Dilated unrolled"] = L.reconstruct(net, Xte, Phi, dev=dev)

    k = int(np.argmax(np.ptp(Xte[:64], axis=1)))      # a window with a clear QRS
    t = np.arange(N) / 360.0
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 2.9), sharex="col",
                             gridspec_kw={"height_ratios": [2.4, 1]})
    for j, (name, Xh) in enumerate(recs.items()):
        a, b = axes[0, j], axes[1, j]
        a.plot(t, Xte[k], color="0.25", lw=1.0, label="original")
        a.plot(t, Xh[k], lw=1.0, label=name)
        a.set_title(name, fontsize=6)
        if j == 0:
            a.set_ylabel("amplitude (a.u.)")
            a.legend(fontsize=4.8, frameon=False, loc="upper right")
        q = int(np.argmax(np.abs(Xte[k])))
        lo, hi = max(0, q - 24), min(N, q + 24)
        ins = a.inset_axes([0.58, 0.08, 0.4, 0.42])
        ins.plot(t[lo:hi], Xte[k][lo:hi], color="0.25", lw=0.9)
        ins.plot(t[lo:hi], Xh[k][lo:hi], lw=0.9)
        ins.set_xticks([]); ins.set_yticks([])
        ins.set_title("QRS", fontsize=5, pad=1)
        b.plot(t, Xh[k] - Xte[k], lw=0.8, color="firebrick")
        b.axhline(0, color="0.5", lw=0.6)
        b.set_xlabel("time (s)")
        if j == 0:
            b.set_ylabel("error")
    ymax = max(abs(ax.get_ylim()[0]) for ax in axes[1])
    for ax in axes[1]:
        ax.set_ylim(-ymax, ymax)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig_examples_cr075.{ext}"), dpi=300,
                    bbox_inches="tight")
    plt.close(fig)
    print("  fig_examples_cr075")


if __name__ == "__main__":
    fig_main()
    fig_rate()
    fig_examples()
    print("figures ->", FIG)
