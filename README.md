# A Record-Wise Benchmark of Classical, Bayesian and Learned Decoders for Compressed-Sensing ECG

Reproducibility archive for the manuscript

> **A Record-Wise Benchmark of Classical, Bayesian and Learned Decoders for Compressed-Sensing ECG**
> Fatih Gül, Department of Electrical and Electronics Engineering, Department of Biomedical
> Engineering, and the Artificial Intelligence and Internet of Things Research Laboratory,
> Recep Tayyip Erdoğan University, Rize, Türkiye
> (ORCID [0000-0001-5072-2122](https://orcid.org/0000-0001-5072-2122)) — *under review, 2026*

All signals are **real public recordings**: the MIT-BIH Arrhythmia Database (44 non-paced records;
102, 104, 107 and 217 excluded) and the PTB Diagnostic ECG Database, both downloaded from
PhysioNet by the scripts. Splits are **record-wise (subject-disjoint)**.

## What this reproduces

Nine decoders receive the identical measurement operator, compression ratios and evaluation
windows: wavelet-domain ISTA and FISTA, OMP, OMP over a K-SVD dictionary learned from the
training records, BSBL (EM, DCT domain), ISTA-Net+, and re-implementations of two published deep
CS-ECG decoders. Five-fold record-wise cross-validation, three training seeds per learned
decoder, paired Wilcoxon tests across records, quantised and multiplier-free acquisition, a
cross-database test on PTB, and two clinical endpoints (QRS detection and ST-segment morphology).

| Path | Description |
|---|---|
| `code/prepare_data2.py` | PhysioNet to per-record windows (N = 256, per-record z-norm) |
| `code/classical2.py` | ISTA, FISTA, OMP, K-SVD dictionary, BSBL-EM (DCT domain) |
| `code/classical_gpu.py` | batched GPU BSBL (18x faster, numerically identical to 1e-9; `selftest()`) |
| `code/learned2.py` | ISTA-Net+, Hua'22 re-implementation, dilated unrolled decoder |
| `code/run_cv3.py` | main benchmark: common evaluation windows, saved models, QRS endpoint |
| `code/qrs_eval.py`, `code/morph_eval.py` | clinical endpoints (beat detection, ST level) |
| `code/stats_v2.py` | paired signed-rank tests, Holm correction, bootstrap intervals |
| `results/` | every result file the manuscript and its tables are built from |

## Reproducing

```bash
python3 -m pip install numpy scipy scikit-learn matplotlib torch pywavelets wfdb pandas
cd code
python3 prepare_data2.py     # PhysioNet -> windows
bash    run_phase4.sh        # cross-validation + clinical endpoints
bash    run_phase5.sh        # seed replicates
python3 stats_v2.py          # paired tests and tables
python3 make_figures2.py     # figures
```

Seeds are fixed; `results/cv_final.csv`, `cv3_qrs.csv` and `morph.csv` are reproduced from
scratch by the pipeline.

## Headline result

Learning is not uniformly better. A correctly tuned block-sparse Bayesian decoder is the best
method at light and moderate compression, and at CR = 0.75 it is statistically indistinguishable
from the canonical unrolled network once training variability is accounted for. The two published
architectures beat that network at every ratio and are far more stable across seeds.

## License

Code under the MIT License; results and figures under CC BY 4.0.
