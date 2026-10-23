"""
make_supplement2.py — SPL-compliant supplement: exactly two labelled pages.

The journal allows at most one page of "Supplementary Material" and one further
page of "Information for Reproducibility", both labelled separately, neither
essential to the paper (editorial office, 22 Oct 2026).  The earlier five-page
supplement violated that limit, so the content is triaged:

  page 1  Supplementary Material      -> the pairwise tests and the
                                         cross-database result behind two
                                         statements in the letter
  page 2  Information for Reproducibility -> tuned parameters, seed and
                                         sensing-matrix spread, ablation,
                                         clinical endpoints, and the exact
                                         commands that regenerate everything

Per-recording tables (44 x 3 x 8 values) cannot fit two pages and are not
needed to follow the paper; they ship in the results archive as CSV.
"""
import os
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results_v2")
OUT = os.path.join(ROOT, "resubmission", "manuscript", "supplement_body.tex")

NICE = {"ista": "ISTA", "fista": "FISTA", "omp": "OMP", "ksvd_omp": "K-SVD+OMP",
        "bsbl": "BSBL", "istanet": "ISTA-Net$^+$", "hua": "Hua'22",
        "dilated": "Erko\\c{c}'26"}
ORDER = ["ista", "fista", "omp", "ksvd_omp", "bsbl", "hua", "dilated", "istanet"]


def tab(rows, header, align, caption, label, note=""):
    L = [r"\begin{table}[h]", r"\centering", r"\scriptsize",
         r"\setlength{\tabcolsep}{3pt}",
         f"\\caption{{{caption}}}", f"\\label{{{label}}}",
         r"\begin{tabular}{" + align + "}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]
    L += [" & ".join(r) + r" \\" for r in rows]
    L += [r"\bottomrule", r"\end{tabular}"]
    if note:
        L.append(r"\\[2pt]\begin{minipage}{\linewidth}\scriptsize " + note + r"\end{minipage}")
    L.append(r"\end{table}")
    return "\n".join(L)


def main():
    out = []

    # ================= PAGE 1: Supplementary Material =====================
    out.append(r"\section*{Supplementary Material}")
    out.append(r"\setcounter{table}{0}\renewcommand{\thetable}{S\arabic{table}}")

    sig = pd.read_csv(os.path.join(RES, "significance.csv"))
    rows = []
    for cr in sorted(sig.cr.unique()):
        g = sig[sig.cr == cr].sort_values("median_diff_PRDN")
        for _, r in g.iterrows():
            p = "$<$0.001" if r.p_holm < 1e-3 else f"{r.p_holm:.3f}"
            v = ("unrolled" if r.median_diff_PRDN < 0 and r.p_holm < 0.05
                 else "other" if r.p_holm < 0.05 else "n.s.")
            rows.append([f"{cr:g}", NICE.get(r.method, r.method),
                         f"{r.median_diff_PRDN:+.2f}",
                         f"[{r.ci_lo:+.2f}, {r.ci_hi:+.2f}]", p, v])
    out.append(tab(rows, ["CR", "Compared with", r"$\Delta$PRD", r"95\,\% CI",
                          r"$p$ (Holm)", "Better"],
                   "llcccc",
                   "Pairwise comparison of the unrolled decoder against every "
                   "other decoder. $\\Delta$PRD is the median per-recording "
                   "difference (unrolled minus the named method); negative "
                   "favours the unrolled decoder. Wilcoxon signed-rank across "
                   "the 44 recordings, Holm corrected within each ratio. "
                   "\"n.s.\" = not significant at the 5\\,\\% level.",
                   "tab:S1"))

    cd = pd.read_csv(os.path.join(RES, "crossdb.csv"))
    g = (cd.groupby(["cr", "method", "test_db"]).PRDN.mean().unstack().reset_index())
    rows = []
    for cr in sorted(g.cr.unique()):
        sub = g[g.cr == cr]
        for meth in ORDER:
            r = sub[sub.method == meth]
            if len(r):
                rows.append([f"{cr:g}", NICE.get(meth, meth),
                             f"{r.mitbih.iloc[0]:.2f}", f"{r.ptb.iloc[0]:.2f}"])
    out.append(tab(rows, ["CR", "Decoder", "MIT-BIH", "PTB"], "llcc",
                   "Cross-database generalisation. Every decoder is trained on "
                   "MIT-BIH training recordings and applied, without "
                   "retraining, to 40 recordings of the PTB Diagnostic ECG "
                   "Database resampled to 360\\,Hz. Values are mean PRD "
                   "(\\%).", "tab:S2"))

    # ============ PAGE 2: Information for Reproducibility =================
    out.append(r"\clearpage")
    out.append(r"\section*{Information for Reproducibility}")

    ab = pd.read_csv(os.path.join(RES, "ablation.csv"))
    a = ab[ab.cr == 0.75]
    rows = []
    for _, r in a.iterrows():
        v = {"stages": f"stages = {int(r.stages)}",
             "no_learned_prox": "frozen prox operator",
             "fixed_threshold": "frozen threshold"}.get(r.variant, r.variant)
        rows.append([v, f"{int(r.params):,}".replace(",", "\\,"),
                     f"{r.PRDN:.2f}", f"{r.SNR:.2f}"])
    out.append(tab(rows, ["Variant", "Params", "PRD (\\%)", "SNR (dB)"], "lccc",
                   "Ablation of the unrolled decoder at $\\mathrm{CR}=0.75$.",
                   "tab:S3"))

    cv = pd.read_csv(os.path.join(RES, "cv_final.csv"))
    cv = cv[(cv.scope == "common") & (cv.phi == "gaussian")]
    sd = (cv[cv.method.isin(["istanet", "hua", "dilated"])]
          .groupby(["cr", "method", "seed"]).PRDN.mean()
          .groupby(level=[0, 1]).agg(["mean", "std"]).reset_index())
    rows = [[f"{r.cr:g}", NICE.get(r.method, r.method), f"{r['mean']:.2f}",
             f"{r['std']:.2f}"] for _, r in sd.iterrows()]
    out.append(tab(rows, ["CR", "Decoder", "mean PRD", "s.d. (seeds)"], "llcc",
                   "Spread of the learned decoders across three training "
                   "seeds. The unrolled decoder is the least stable, which is "
                   "why the main table reports seed-averaged per-recording "
                   "values.", "tab:S4"))

    q_p, m_p = os.path.join(RES, "cv3_qrs.csv"), os.path.join(RES, "morph.csv")
    if os.path.exists(q_p) and os.path.exists(m_p):
        q = (pd.read_csv(q_p)
             .drop_duplicates(subset=["fold", "cr", "seed", "method", "record"],
                              keep="last")
             .groupby(["cr", "method"]).f1_drop.mean())
        mo = (pd.read_csv(m_p).groupby(["cr", "method"])
              .agg(st=("st_mae", "mean"), fr=("st_frac_gt_0p1", "mean")))
        rows = []
        for cr in (0.25, 0.5, 0.75):
            for meth in ORDER:
                if (cr, meth) in mo.index:
                    rows.append([f"{cr:g}", NICE.get(meth, meth),
                                 f"{q.get((cr, meth), float('nan')):+.4f}",
                                 f"{mo.loc[(cr, meth), 'st']:.3f}",
                                 f"{mo.loc[(cr, meth), 'fr']*100:.0f}"])
        out.append(tab(rows, ["CR", "Decoder", r"$\Delta F_1$", "ST MAE",
                              r"ST $>$0.1 (\%)"], "llccc",
                       "Clinical endpoints on 6\\,672 annotated beats of "
                       "contiguous reconstructed signal. $\\Delta F_1$ is the "
                       "beat-detection loss relative to the same detector on "
                       "the original recording (negative = reconstruction "
                       "scores higher). ST level is measured at J+60\\,ms "
                       "against the PR baseline, in units of the per-recording "
                       "standard deviation.", "tab:S5"))

    out.append(r"""
\subsection*{Regenerating the results}
All data are public. MIT-BIH and the PTB Diagnostic ECG Database are downloaded
by the pipeline from PhysioNet; 44 non-paced recordings are used (102, 104, 107
and 217 excluded), windowed into non-overlapping segments of $N=256$ samples
with per-recording $z$-normalisation. Folds partition recordings, never windows.
Regularisation parameters are selected per fold and ratio on a validation split
of the training recordings over the grid
$\lambda\in\{0.005,0.01,0.02,0.04,0.06,0.08,0.12,0.20\}$; networks are trained
with Adam, early stopping on a held-out split (patience 12, cap 120 epochs), and
the step size of each unrolled stage is initialised at $1/\|\Phi\|_2^2$. The
accompanying archive regenerates every number and figure:

\begin{verbatim}
python3 prepare_data2.py   # PhysioNet -> windows
bash    run_phase4.sh      # cross-validation + endpoints
bash    run_phase5.sh      # seed replicates
python3 stats_v2.py        # paired tests, tables
python3 make_figures2.py   # figures
\end{verbatim}

Per-recording values for all 44 recordings, every fold, ratio, seed and decoder
are provided as CSV in the archive (\texttt{results/cv\_final.csv},
\texttt{cv3\_qrs.csv}, \texttt{morph.csv}); they are not reproduced here because
they are not needed to follow the paper.
""")

    open(OUT, "w").write("\n\n".join(out) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
