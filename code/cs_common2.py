"""
cs_common2.py — shared utilities for the expanded CS-ECG benchmark (v2).

Changes w.r.t. v1 (driven by the SPL review):
  * all 44 non-paced MIT-BIH records, record-wise 5-fold cross-validation
    (v1 used a single 9/5-record split = 14 of 48 records);
  * three sensing-matrix families: dense Gaussian, Bernoulli +-1 and
    *sparse binary* (d ones per column) which is the hardware-cheap option on
    the sensor node;
  * measurement quantisation to b bits + bit-rate / node-energy accounting;
  * both PRD conventions are reported: PRDN (mean removed, used in v1 and by
    most CS-ECG papers) and raw PRD (no mean removal), because the literature
    mixes them and the numbers differ by a large factor.

Convention: x in R^N, Phi in R^{MxN}, y = Phi x.
  sensing rate  SR = M / N          (used by [S2] Hua 2022 and most DL papers)
  compression   CR = 1 - M/N        (fraction of measurements discarded, v1)
"""
import numpy as np

N = 256
FS = 360.0                      # MIT-BIH sampling rate [Hz]
CRS = [0.25, 0.50, 0.75]        # fraction of measurements DISCARDED
SEED = 0

# ------------------------------- records ----------------------------------
ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
# paced beats -> excluded, standard practice in the CS-ECG / AAMI literature
PACED = {"102", "104", "107", "217"}
RECORDS = [r for r in ALL_RECORDS if r not in PACED]      # 44 records

N_FOLDS = 5


def folds(records=None, n_folds=N_FOLDS, seed=SEED):
    """Record-wise (subject-disjoint) K-fold split: list of (train, test)."""
    recs = list(RECORDS if records is None else records)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(recs))
    chunks = np.array_split(idx, n_folds)
    out = []
    for k in range(n_folds):
        te = [recs[i] for i in chunks[k]]
        tr = [recs[i] for i in idx if recs[i] not in te]
        out.append((tr, te))
    return out


def m_for_cr(cr, n=N):
    return int(round((1.0 - cr) * n))


def sr_for_cr(cr):
    return 1.0 - cr


# --------------------------- sensing matrices ------------------------------
def make_phi(m, n=N, seed=SEED, kind="gaussian", d=8):
    """
    kind = "gaussian"      : dense i.i.d. N(0,1/m)   (v1 default; costly on node)
           "bernoulli"     : dense +-1/sqrt(m)
           "sparse_binary" : exactly d ones per column, column-normalised.
                             Node cost is d additions per column, no multiplies.
    """
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        phi = rng.standard_normal((m, n)) / np.sqrt(m)
    elif kind == "bernoulli":
        phi = (rng.integers(0, 2, size=(m, n)).astype(np.float64) * 2 - 1) / np.sqrt(m)
    elif kind == "sparse_binary":
        phi = np.zeros((m, n))
        dd = min(d, m)
        for j in range(n):
            rows = rng.choice(m, size=dd, replace=False)
            phi[rows, j] = 1.0
        phi /= np.sqrt(dd)
    else:
        raise ValueError(kind)
    return phi


def phi_node_cost(kind, m, n=N, d=8):
    """Per-window sensor-node cost of computing y = Phi x.
    Returns (multiplies, additions). Sparse binary needs no multiplies."""
    if kind == "sparse_binary":
        return 0, d * n
    if kind == "bernoulli":
        return 0, m * n            # +-1 -> sign flips + adds
    return m * n, m * n            # dense Gaussian


# ---------------------------- quantisation ---------------------------------
def quantize(Y, bits, per_window=True):
    """Uniform mid-rise quantisation of the measurements to `bits` bits.
    bits=None -> float (no quantisation). Scale is sent as side information
    (its cost is accounted for separately in bit_rate())."""
    if bits is None:
        return np.asarray(Y, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    ax = -1 if per_window else None
    a = np.max(np.abs(Y), axis=ax, keepdims=True) + 1e-12
    levels = 2 ** (bits - 1) - 1
    Q = np.round(Y / a * levels) / levels * a
    return Q


def bit_rate(m, bits, n=N, fs=FS, scale_bits=16):
    """Transmitted bits per second for one window of n samples at fs Hz."""
    if bits is None:
        bits = 32
    per_window = m * bits + scale_bits
    return per_window * fs / n


# ------------------------------- metrics -----------------------------------
def _zm(x):
    x = np.asarray(x, dtype=np.float64)
    return x - x.mean(axis=-1, keepdims=True)


def prdn(x, xhat):
    """PRD on mean-removed signals (PRDN) — the v1 definition."""
    x0, e = _zm(x), _zm(x) - _zm(xhat)
    return 100.0 * np.sqrt((e ** 2).sum(-1)) / (np.sqrt((x0 ** 2).sum(-1)) + 1e-12)


def prd_raw(x, xhat):
    """PRD without mean removal — the other convention found in the literature."""
    x = np.asarray(x, np.float64)
    e = x - np.asarray(xhat, np.float64)
    return 100.0 * np.sqrt((e ** 2).sum(-1)) / (np.sqrt((x ** 2).sum(-1)) + 1e-12)


def snr_db(x, xhat):
    x0, e = _zm(x), _zm(x) - _zm(xhat)
    return 10.0 * np.log10((x0 ** 2).sum(-1) / ((e ** 2).sum(-1) + 1e-12) + 1e-12)


def rmse(x, xhat):
    e = np.asarray(x, np.float64) - np.asarray(xhat, np.float64)
    return np.sqrt((e ** 2).mean(-1))


def summarize(x, xhat):
    p, pr, s, r = prdn(x, xhat), prd_raw(x, xhat), snr_db(x, xhat), rmse(x, xhat)
    return {
        "PRDN": float(np.mean(p)), "PRDN_std": float(np.std(p)),
        "PRD_raw": float(np.mean(pr)),
        "SNR": float(np.mean(s)), "SNR_std": float(np.std(s)),
        "RMSE": float(np.mean(r)), "n_win": int(np.asarray(p).size),
    }


# Quality grades used in the ECG-compression literature.
# Zigel, Cohen & Katz, IEEE TBME 47(11):1422-1430, 2000 (doi:10.1109/10.880093)
# relate PRD bands to diagnostic distortion; the widely quoted operating point
# is PRD < 9 % ("very good"/"good" reconstruction quality).
PRD_GOOD = 9.0
