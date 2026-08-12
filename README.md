# Deep-Unrolled Compressed Sensing for Low-Power ECG Reconstruction in Wearable Telemetry

Reproducibility archive for the manuscript

> **Deep-Unrolled Compressed Sensing for Low-Power ECG Reconstruction in Wearable Telemetry**
> Fatih Gül, Department of Electrical and Electronics Engineering, Department of Biomedical
> Engineering, and the Artificial Intelligence and Internet of Things Research Laboratory,
> Recep Tayyip Erdoğan University, Rize, Türkiye
> (ORCID [0000-0001-5072-2122](https://orcid.org/0000-0001-5072-2122)) — *under peer review, 2026*

All signals are **real recordings** from the public MIT-BIH Arrhythmia Database
(PhysioNet, channel 0 / MLII). No synthetic or simulated ECG is used. The train/test
split is **record-wise (subject-disjoint)**, so no window from a test record is ever seen
during training.

## Contents

| Path | Description |
|---|---|
| `code/prepare_data.py` | Downloads/loads MIT-BIH records via `wfdb`, z-normalises per record, segments into non-overlapping windows of N = 256 samples (~0.71 s at 360 Hz), record-wise train/test split |
| `code/cs_common.py` | Shared settings: window length, compression ratios, random Gaussian sensing matrix, PRD/SNR metrics |
| `code/classical.py` | Wavelet-domain ℓ1 recovery (ISTA) and Orthogonal Matching Pursuit baselines |
| `code/learned.py` | ISTA-Net-style unrolled proximal-gradient network (PyTorch) and its training loop |
| `code/run_experiment.py` | Main sweep over compression ratios 25 / 50 / 75 % → `results/metrics_by_cr.csv`, `results/examples_cr050.npz` |
| `code/make_figures.py` | Regenerates both paper figures from the result files |
| `results/metrics_by_cr.csv` | Every number reported in Table I |
| `results/run_log.txt` | Full console log of the reported run (22 851 train / 12 695 test windows) |
| `figures/*.{pdf,png}` | Figures 2 and 3 of the paper |

## Reproducing every number

```bash
python3 -m pip install numpy scipy scikit-learn matplotlib torch pywavelets wfdb
cd code
python3 prepare_data.py      # MIT-BIH -> ../data/*.npy   (record-wise split)
python3 run_experiment.py    # ISTA / OMP / ISTA-Net at CR = 0.25, 0.50, 0.75
python3 make_figures.py      # ../figures/*.pdf and *.png
```

Seeds are fixed, so `results/metrics_by_cr.csv` and both figures are reproduced bit-for-bit.
The reported run used one GPU (`device=cuda`); CPU-only execution gives the same numbers,
only slower.

## Headline result

| Compression | ISTA-ℓ1 PRD | OMP PRD | **ISTA-Net PRD** | ISTA-Net SNR |
|---|---|---|---|---|
| 25 % | 6.52 | 5.04 | **5.53** | 25.93 dB |
| 50 % | 12.84 | 9.73 | **7.27** | 23.51 dB |
| 75 % | 54.35 | 35.93 | **11.43** | 19.50 dB |

## License

Code released under the MIT License; results and figures under CC BY 4.0.

## Citation

If you use this archive, please cite the manuscript above (citation updated on acceptance).
