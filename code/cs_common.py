"""
cs_common.py
Shared compressed-sensing utilities for ECG reconstruction.

Convention
----------
Signal x in R^N (N=256). Measurement matrix Phi in R^{M x N} with M < N.
Measurements y = Phi @ x.  Compression ratio (CR) is the fraction of samples
DISCARDED, i.e. CR = 1 - M/N.  So CR=0.50 keeps M = N/2 measurements.

Metrics (standard in CS-ECG literature)
---------------------------------------
PRD  (percentage RMS difference) = 100 * ||x - xhat|| / ||x||
SNR  (dB)                        = 20 * log10(||x|| / ||x - xhat||)
RMSE                             = sqrt(mean((x - xhat)^2))
PRD and SNR are computed on the zero-mean signal, which is the common CS-ECG
definition (PRDN). Values are averaged over the test windows.
"""
import numpy as np

N = 256
CRS = [0.25, 0.50, 0.75]  # fraction of measurements discarded
SEED = 0


def m_for_cr(cr, n=N):
    return int(round((1.0 - cr) * n))


def make_phi(m, n=N, seed=SEED, kind="gaussian"):
    """Random measurement matrix with unit-norm-ish columns (1/sqrt(m) scaling)."""
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        phi = rng.standard_normal((m, n)) / np.sqrt(m)
    elif kind == "bernoulli":
        phi = rng.integers(0, 2, size=(m, n)).astype(np.float64) * 2 - 1
        phi /= np.sqrt(m)
    else:
        raise ValueError(kind)
    return phi


# ----------------------------- metrics ------------------------------------
def _prep(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=-1, keepdims=True)


def prd(x, xhat):
    x0 = _prep(x)
    e = x0 - _prep(xhat)
    num = np.sqrt((e ** 2).sum(axis=-1))
    den = np.sqrt((x0 ** 2).sum(axis=-1)) + 1e-12
    return 100.0 * num / den


def snr_db(x, xhat):
    x0 = _prep(x)
    e = x0 - _prep(xhat)
    num = (x0 ** 2).sum(axis=-1)
    den = (e ** 2).sum(axis=-1) + 1e-12
    return 10.0 * np.log10(num / den + 1e-12)


def rmse(x, xhat):
    e = np.asarray(x, np.float64) - np.asarray(xhat, np.float64)
    return np.sqrt((e ** 2).mean(axis=-1))


def summarize(x, xhat):
    return {
        "PRD": float(np.mean(prd(x, xhat))),
        "PRD_std": float(np.std(prd(x, xhat))),
        "SNR": float(np.mean(snr_db(x, xhat))),
        "SNR_std": float(np.std(snr_db(x, xhat))),
        "RMSE": float(np.mean(rmse(x, xhat))),
    }
