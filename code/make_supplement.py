"""
make_supplement.py — build the supplemental material (Tables S1-S5) and the
main-table wrapper, all from results_v2/*.csv.

S1  tuned regularisation parameters per fold and CR          (reviewer 2)
S2  per-recording PRD for every method                       (reviewer 2)
S3  variability across seeds and sensing-matrix families     (reviewer 2)
S4  ablation of the unrolled decoder                         (reviewer 1)
S5  cross-database results (MIT-BIH -> PTB)                  (reviewer 2)
"""
import os
import re
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CV_FILE = os.environ.get("CV_FILE", "cv_metrics.csv")
RES = os.path.join(ROOT, "results_v2")
SUB = os.path.join(ROOT, "resubmission", "manuscript")
os.makedirs(SUB, exist_ok=True)

NICE = {"ista": "ISTA", "fista": "FISTA", "omp": "OMP", "ksvd_omp": "K-SVD+OMP",
        "bsbl": "BSBL", "istanet": "ISTA-Net$^+$", "hua": "Hua'22 (fixed $\\Phi$)",
        "hua_learned": "Hua'22 (learned $\\Phi$)", "dilated": "Erko\\c{c}'26"}
ORDER = ["ista", "fista", "omp", "ksvd_omp", "bsbl", "hua", "dilated",
         "istanet", "hua_learned"]


def wrapper():
    """Wrap the generated tabular in a table float for the main text."""
    body = open(os.path.join(RES, "table_main.tex")).read()
    tex = (r"\begin{table}[!t]" "\n" r"\centering" "\n"
           r"\caption{Reconstruction quality over all \NRECORDS{} non-paced "
           r"MIT-BIH recordings under record-wise \NFOLDS-fold cross-validation. "
           r"PRD and SNR are means over recordings. $^{\ast}$ marks a difference "
           r"from the unrolled decoder (ISTA-Net$^+$) that is significant at the "
           r"5\,\% level by a Wilcoxon signed-rank test across recordings with "
           r"Holm correction; per-recording values, exact $p$ and bootstrap "
           r"intervals are in Tables~S2 and S6. Learned decoders are listed "
           r"below the rule. Best per CR in \textbf{bold}.}" "\n"
           r"\label{tab:main}" "\n" r"\footnotesize" "\n"
           r"\setlength{\tabcolsep}{3pt}" "\n"
           r"\resizebox{\linewidth}{!}{%" "\n" + body + "}\n"
           r"\end{table}" "\n")
    open(os.path.join(SUB, "table_main_wrapper.tex"), "w").write(tex)


def _tab(df, cols, header, caption, label, fmt=None, small=True):
    lines = [r"\begin{table}[!t]", r"\centering", f"\\caption{{{caption}}}",
             f"\\label{{{label}}}"]
    if small:
        lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{" + "l" * 1 + "c" * (len(cols) - 1) + "}")
    lines.append(r"\toprule")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:.2f}" if fmt is None else fmt(c, v))
            else:
                cells.append(str(v))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def main():
    out = [r"\section*{Supplemental Material}",
           r"\setcounter{table}{0}",
           r"\renewcommand{\thetable}{S\arabic{table}}", ""]

    log = os.path.join(RES, "run_all.log")
    # ---- S1 tuned lambdas (parsed from the run log) ----
    if os.path.exists(log):
        rows = []
        fold = None
        for line in open(log, errors="ignore"):
            if line.startswith("=== fold"):
                fold = int(re.search(r"fold (\d+)", line).group(1))
            m = re.search(r"CR=([\d.]+) M=(\d+): lam_ISTA=([\d.]+) lam_FISTA=([\d.]+)", line)
            if m and fold is not None:
                rows.append({"fold": fold, "CR": float(m.group(1)),
                             "M": int(m.group(2)), "lam_ISTA": float(m.group(3)),
                             "lam_FISTA": float(m.group(4))})
        if rows:
            d = pd.DataFrame(rows).drop_duplicates()
            out.append(_tab(d, ["fold", "CR", "M", "lam_ISTA", "lam_FISTA"],
                            ["Fold", "CR", "$M$", r"$\lambda_{\mathrm{ISTA}}$",
                             r"$\lambda_{\mathrm{FISTA}}$"],
                            "Regularisation parameters selected per fold and "
                            "compression ratio on a validation split of the "
                            "training recordings (grid "
                            "$\\{0.005,\\dots,0.20\\}$).", "tab:S1"))
            out.append("")

    cv_p = os.path.join(RES, CV_FILE)
    if os.path.exists(cv_p):
        cv = pd.read_csv(cv_p)
        base = cv[(cv.phi == "gaussian") & (cv.bits == "none") & (cv.seed == 0)]
        # ---- S2 per-record PRD ----
        per = (base.groupby(["cr", "record", "method"]).PRDN.mean()
                   .reset_index())
        for cr in sorted(per.cr.unique()):
            w = per[per.cr == cr].pivot(index="record", columns="method",
                                        values="PRDN").reset_index()
            cols = ["record"] + [m for m in ORDER if m in w.columns]
            head = ["Rec."] + [NICE.get(m, m) for m in cols[1:]]
            out.append(_tab(w[cols], cols, head,
                            f"Per-recording PRD (\\%) at CR = {cr}.",
                            f"tab:S2{int(cr*100)}"))
            out.append("")
        # ---- S3 variability ----
        var_rows = []
        for key, nice in (("seed", "training seed"), ("phi", "sensing matrix")):
            if cv[key].nunique() > 1:
                g = (cv.groupby(["cr", "method", key]).PRDN.mean()
                       .groupby(level=[0, 1]).agg(["mean", "std"]).reset_index())
                g["source"] = nice
                var_rows.append(g)
        if var_rows:
            v = pd.concat(var_rows)
            v["method"] = v.method.map(lambda m: NICE.get(m, m))
            out.append(_tab(v, ["source", "cr", "method", "mean", "std"],
                            ["Varied", "CR", "Method", "mean PRD", "s.d."],
                            "Variability of mean PRD across four training seeds "
                            "and across the three sensing-matrix families.",
                            "tab:S3"))
            out.append("")

    # ---- S4 ablation ----
    ab_p = os.path.join(RES, "ablation.csv")
    if os.path.exists(ab_p):
        ab = pd.read_csv(ab_p)
        ab["variant"] = ab.variant.replace(
            {"stages": "stages", "no_learned_prox": "frozen prox",
             "fixed_threshold": "frozen threshold"})
        out.append(_tab(ab, ["cr", "variant", "stages", "params", "PRDN", "SNR"],
                        ["CR", "Variant", "Stages", "Params", "PRD (\\%)",
                         "SNR (dB)"],
                        "Ablation of the unrolled decoder.", "tab:S4"))
        out.append("")

    # ---- S5 cross-database ----
    cd_p = os.path.join(RES, "crossdb.csv")
    if os.path.exists(cd_p):
        cd = pd.read_csv(cd_p)
        g = (cd.groupby(["cr", "test_db", "method"])
               .agg(PRD=("PRDN", "mean"), sd=("PRDN", "std"),
                    n=("record", "nunique")).reset_index())
        g["method"] = g.method.map(lambda m: NICE.get(m, m))
        out.append(_tab(g, ["cr", "test_db", "method", "PRD", "sd", "n"],
                        ["CR", "Test DB", "Method", "PRD (\\%)", "s.d.",
                         "$n$ rec."],
                        "Cross-database generalisation: every decoder is "
                        "trained on MIT-BIH training recordings and applied "
                        "without retraining to the PTB Diagnostic ECG Database "
                        "resampled to 360\\,Hz.", "tab:S5"))
        out.append("")


    # ---- S7 clinical endpoints ----
    q_p = os.path.join(RES, "cv3_qrs.csv")
    m_p = os.path.join(RES, "morph.csv")
    if os.path.exists(q_p) and os.path.exists(m_p):
        q = pd.read_csv(q_p).drop_duplicates(
            subset=["fold", "cr", "seed", "method", "record"], keep="last")
        mo = pd.read_csv(m_p)
        a = q.groupby(["cr", "method"]).agg(f1_rec=("f1_rec", "mean"),
                                            drop=("f1_drop", "mean")).reset_index()
        b = mo.groupby(["cr", "method"]).agg(st=("st_mae", "mean"),
                                             frac=("st_frac_gt_0p1", "mean"),
                                             ramp=("ramp_mape", "mean"),
                                             qw=("qrs_width_mae_ms", "mean")).reset_index()
        t = a.merge(b, on=["cr", "method"])
        t["frac"] = t.frac * 100
        t["method"] = t.method.map(lambda m: NICE.get(m, m))
        out.append(_tab(t, ["cr", "method", "f1_rec", "drop", "st", "frac",
                            "ramp", "qw"],
                        ["CR", "Decoder", "$F_1$ (rec.)", "$\\Delta F_1$",
                         "ST MAE", "ST $>$0.1 (\\%)", "$R$ err. (\\%)",
                         "QRS width (ms)"],
                        "Clinical endpoints on contiguous reconstructed signal. "
                        "$\\Delta F_1$ is the loss relative to the same detector "
                        "run on the original recording (negative = reconstruction "
                        "scores higher, a mild denoising effect). ST level is "
                        "measured at J+60\\,ms against the PR baseline, in units "
                        "of the per-recording standard deviation.", "tab:S7"))
        out.append("")

    # ---- S6 significance detail ----
    sg_p = os.path.join(RES, "significance.csv")
    if os.path.exists(sg_p):
        sg = pd.read_csv(sg_p)
        sg["method"] = sg.method.map(lambda m: NICE.get(m, m))
        sg["CI"] = sg.apply(lambda r: f"[{r.ci_lo:.2f}, {r.ci_hi:.2f}]", axis=1)
        sg["p"] = sg.p_holm.map(lambda p: "$<$0.001" if p < 1e-3 else f"{p:.3f}")
        out.append(_tab(sg, ["cr", "method", "median_diff_PRDN", "CI", "p",
                             "n_records", "verdict"],
                        ["CR", "Compared with", "median $\\Delta$PRD", "95\\,\\% CI",
                         "$p$ (Holm)", "$n$", "Verdict"],
                        "Pairwise comparison of the unrolled decoder against "
                        "every other decoder. $\\Delta$PRD is the median "
                        "per-recording difference (unrolled minus the named "
                        "method); negative means the unrolled decoder is "
                        "better. Wilcoxon signed-rank across recordings, Holm "
                        "corrected within each compression ratio.", "tab:S6"))
        out.append("")

    open(os.path.join(SUB, "supplement_body.tex"), "w").write("\n".join(out))
    wrapper()
    print("wrote supplement_body.tex and table_main_wrapper.tex")


if __name__ == "__main__":
    main()
