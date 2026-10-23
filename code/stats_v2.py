"""
stats_v2.py — aggregation and significance testing for the v2 benchmark.

Reviewer 2 asked for per-record results, variability across seeds and sensing
matrices, and significance testing.  All comparisons are *paired at the record
level* (the natural experimental unit: windows within a record are not
independent), using the Wilcoxon signed-rank test over per-record PRDN, with
Holm correction across the methods compared at each compression ratio, plus
bootstrap confidence intervals on the median paired difference.

Outputs: results_v2/summary_by_method.csv, results_v2/significance.csv,
         results_v2/table_main.tex
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
CV_FILE = os.environ.get("CV_FILE", "cv_metrics.csv")
RES = os.path.join(os.path.dirname(HERE), "results_v2")

NICE = {"ista": "ISTA ($\\ell_1$, wavelet)", "fista": "FISTA ($\\ell_1$, wavelet)",
        "omp": "OMP (wavelet)", "ksvd_omp": "K-SVD + OMP~\\cite{aharon2006ksvd}",
        "bsbl": "BSBL-EM~\\cite{zhang2013fetal}",
        "istanet": "ISTA-Net$^+$ (unrolled)~\\cite{zhang2018}",
        "hua": "Hua \\emph{et al.}~\\cite{hua2022} (fixed $\\Phi$)",
        "hua_learned": "Hua \\emph{et al.}~\\cite{hua2022} (learned $\\Phi$)",
        "dilated": "Dilated unrolled~\\cite{erkoc2026}"}
ORDER = ["ista", "fista", "omp", "ksvd_omp", "bsbl", "hua", "dilated",
         "istanet", "hua_learned"]


def bootstrap_ci(d, n=10000, seed=0, q=(2.5, 97.5)):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    meds = np.median(np.asarray(d)[idx], axis=1)
    return float(np.percentile(meds, q[0])), float(np.percentile(meds, q[1]))


def main():
    cv = pd.read_csv(os.path.join(RES, CV_FILE))
    cv = cv[cv.phi == "gaussian"]

    # ---- per-record means over folds/seeds, then summary per method ----
    per_rec = (cv.groupby(["cr", "method", "record"])
                 .agg(PRDN=("PRDN", "mean"), SNR=("SNR", "mean"))
                 .reset_index())
    summ = (per_rec.groupby(["cr", "method"])
                   .agg(PRDN_mean=("PRDN", "mean"), PRDN_sd=("PRDN", "std"),
                        PRDN_med=("PRDN", "median"),
                        SNR_mean=("SNR", "mean"), SNR_sd=("SNR", "std"),
                        n_rec=("record", "nunique"))
                   .reset_index())
    summ.to_csv(os.path.join(RES, "summary_by_method.csv"), index=False)

    # ---- paired significance tests against the learned decoder ----
    rows = []
    for cr, g in per_rec.groupby("cr"):
        wide = g.pivot(index="record", columns="method", values="PRDN").dropna()
        if "istanet" not in wide.columns:
            continue
        others = [c for c in wide.columns if c != "istanet"]
        pvals = {}
        for o in others:
            d = wide["istanet"].values - wide[o].values
            try:
                stat, p = wilcoxon(d)
            except ValueError:
                stat, p = np.nan, 1.0
            lo, hi = bootstrap_ci(d)
            pvals[o] = (p, float(np.median(d)), lo, hi, len(d))
        # Holm correction within this CR
        order = sorted(pvals, key=lambda k: pvals[k][0])
        mt = len(order)
        prev = 0.0
        for i, o in enumerate(order):
            p, med, lo, hi, nrec = pvals[o]
            p_adj = max(prev, min(1.0, (mt - i) * p))
            prev = p_adj
            rows.append({"cr": cr, "reference": "istanet", "method": o,
                         "median_diff_PRDN": med, "ci_lo": lo, "ci_hi": hi,
                         "p_raw": p, "p_holm": p_adj, "n_records": nrec,
                         "verdict": ("learned better" if med < 0 and p_adj < 0.05
                                     else "other better" if med > 0 and p_adj < 0.05
                                     else "no significant difference")})
    sig = pd.DataFrame(rows)
    sig.to_csv(os.path.join(RES, "significance.csv"), index=False)

    # ---- seed / sensing-matrix variability, if present ----
    var_lines = []
    full = pd.read_csv(os.path.join(RES, CV_FILE))
    for key in ("seed", "phi"):
        if full[key].nunique() > 1:
            v = (full.groupby(["cr", "method", key]).PRDN.mean()
                     .groupby(level=[0, 1]).std().reset_index()
                     .rename(columns={"PRDN": f"sd_across_{key}"}))
            v.to_csv(os.path.join(RES, f"variability_{key}.csv"), index=False)
            var_lines.append(f"  variability across {key} -> variability_{key}.csv")

    # ---- LaTeX main table: methods as rows, compression ratios as columns ----
    # (27 rows does not fit a Letter; this layout is 9 rows.)
    crs = sorted(summ.cr.unique())
    head = " & ".join([r"\multicolumn{2}{c}{CR $=" + f"{c:g}" + "$}" for c in crs])
    lines = [r"\begin{tabular}{l" + "cc" * len(crs) + "}", r"\toprule",
             r"Decoder & " + head + r" \\"]
    cmid = " ".join([f"\\cmidrule(lr){{{2+2*i}-{3+2*i}}}" for i in range(len(crs))])
    lines.append(cmid)
    lines.append(" & " + " & ".join([r"PRD (\%) & SNR (dB)"] * len(crs)) + r" \\")
    lines.append(r"\midrule")
    best = {c: summ[summ.cr == c].set_index("method").PRDN_mean.idxmin() for c in crs}
    sigmap = {}
    if len(sig):
        for _, r in sig.iterrows():
            sigmap[(r.cr, r.method)] = r.p_holm
    group_done = False
    for meth in ORDER:
        cells = []
        present = False
        for c in crs:
            sub = summ[(summ.cr == c) & (summ.method == meth)]
            if sub.empty:
                cells += ["---", "---"]
                continue
            present = True
            r = sub.iloc[0]
            prd = f"{r.PRDN_mean:.2f}"
            snr = f"{r.SNR_mean:.2f}"
            p = sigmap.get((c, meth))
            if p is not None and p < 0.05:
                prd += r"$^{\ast}$"
            if best[c] == meth:
                prd, snr = r"\textbf{" + prd + "}", r"\textbf{" + snr + "}"
            cells += [prd, snr]
        if not present:
            continue
        if meth in ("hua", "dilated", "istanet") and not group_done:
            lines.append(r"\midrule")
            group_done = True
        lines.append(f"{NICE.get(meth, meth)} & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(RES, "table_main.tex"), "w").write("\n".join(lines))

    print(summ.to_string(index=False))
    print()
    if len(sig):
        print(sig.to_string(index=False))
    print("\nwrote summary_by_method.csv, significance.csv, table_main.tex")
    for l in var_lines:
        print(l)


if __name__ == "__main__":
    main()
