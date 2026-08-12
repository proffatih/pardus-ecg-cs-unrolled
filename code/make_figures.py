"""
make_figures.py
Generate publication-quality figures (PDF + PNG, dpi>=200) from results/.

Figures:
  fig_examples_cr050  : original vs ISTA-L1 vs OMP vs ISTA-Net reconstructions
                        at CR=0.50, for a few held-out ECG windows.
  fig_prd_snr_vs_cr   : PRD and SNR versus compression ratio for the three
                        methods (with std error bars).
"""
import os
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys, os as _os
for _c in (_os.environ.get('PRO_STYLE_DIR'), _os.path.dirname(_os.path.abspath(__file__)),
           _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..', '..', '_templates')):
    if _c and _os.path.isdir(_c): _sys.path.insert(0, _c)
try:
    from pro_style import apply as _pa; _pa()
except Exception as _e:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "font.family": "serif",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.6,
})

FS = 360.0  # Hz


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("wrote", name + ".pdf/.png")


def load_metrics():
    rows = []
    with open(os.path.join(RES, "metrics_by_cr.csv")) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def fig_examples():
    d = np.load(os.path.join(RES, "examples_cr050.npz"))
    orig, ista, omp_, net = d["orig"], d["ista"], d["omp"], d["net"]
    n = orig.shape[0]
    t = np.arange(orig.shape[1]) / FS
    fig, axes = plt.subplots(n, 1, figsize=(7.0, 2.1 * n), sharex=True)
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.plot(t, orig[i], color="black", lw=1.8, label="Original", zorder=3)
        ax.plot(t, ista[i], color="#1f77b4", lw=1.2, ls="--", label="ISTA-L1")
        ax.plot(t, omp_[i], color="#2ca02c", lw=1.2, ls=":", label="OMP")
        ax.plot(t, net[i], color="#d62728", lw=1.4, label="ISTA-Net (learned)")
        ax.set_ylabel("Amplitude (norm.)")
        ax.margins(x=0)
    axes[0].legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.32),
                   frameon=False)
    axes[-1].set_xlabel("Time (s)")
    save(fig, "fig_examples_cr050")


def fig_curves():
    rows = load_metrics()
    methods = ["ISTA-L1", "OMP", "ISTA-Net"]
    styles = {"ISTA-L1": ("#1f77b4", "o", "--"),
              "OMP": ("#2ca02c", "s", ":"),
              "ISTA-Net": ("#d62728", "^", "-")}
    labels = {"ISTA-L1": "ISTA-L1 (classical)",
              "OMP": "OMP (classical)",
              "ISTA-Net": "ISTA-Net (learned)"}
    crs = sorted({float(r["CR"]) for r in rows})

    def series(method, key):
        out = []
        for cr in crs:
            for r in rows:
                if r["method"] == method and abs(float(r["CR"]) - cr) < 1e-9:
                    out.append(float(r[key]))
        return np.array(out)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.4))
    xcr = [int(c * 100) for c in crs]
    for m in methods:
        c, mk, ls = styles[m]
        a1.errorbar(xcr, series(m, "PRD"), yerr=series(m, "PRD_std"),
                    color=c, marker=mk, ls=ls, capsize=3, label=labels[m])
        a2.errorbar(xcr, series(m, "SNR"), yerr=series(m, "SNR_std"),
                    color=c, marker=mk, ls=ls, capsize=3, label=labels[m])
    a1.set_xlabel("Compression ratio (% measurements discarded)")
    a1.set_ylabel("PRD (%)")
    a1.set_title("(a) Reconstruction PRD")
    a2.set_xlabel("Compression ratio (% measurements discarded)")
    a2.set_ylabel("SNR (dB)")
    a2.set_title("(b) Reconstruction SNR")
    a1.set_xticks(xcr)
    a2.set_xticks(xcr)
    a1.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig_prd_snr_vs_cr")


if __name__ == "__main__":
    fig_examples()
    fig_curves()
