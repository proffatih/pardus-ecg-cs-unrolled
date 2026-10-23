"""
classical2.py — classical sparse-recovery baselines for CS-ECG (v2).

v1 had only wavelet ISTA + OMP.  The SPL review asked for (i) a properly tuned
l1 solver and FISTA, and (ii) a *data-adapted* classical baseline, because
comparing a trained network against a fixed wavelet basis is not a fair fight.
This module therefore provides:

  ista_l1   wavelet-domain l1 via ISTA            (lambda tuned on a val split)
  fista_l1  wavelet-domain l1 via FISTA           (accelerated, same objective)
  omp       orthogonal matching pursuit, wavelet dictionary
  ksvd_fit / omp_dict   K-SVD learned dictionary + OMP  [S4] Aharon et al. 2006
  bsbl_bo   block sparse Bayesian learning (BSBL-BO), Zhang & Rao
            — the method used for CS-ECG telemonitoring in [6] zhang2013fetal

All solvers take measurements Y (B,M), Phi (M,N) and return Xhat (B,N).
"""
import numpy as np
import pywt
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.decomposition import MiniBatchDictionaryLearning

from cs_common2 import N


# ----------------------------- dictionaries --------------------------------
_W_CACHE = {}


def wavelet_matrix(n=N, wavelet="db4"):
    """Orthonormal DWT synthesis matrix W (n x n): x = W @ s."""
    key = (n, wavelet)
    if key in _W_CACHE:
        return _W_CACHE[key]
    w = pywt.Wavelet(wavelet)
    level = pywt.dwt_max_level(n, w.dec_len)
    cols = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        coeffs = pywt.wavedec(e, w, level=level, mode="periodization")
        cols.append(np.concatenate([c.ravel() for c in coeffs]))
    W = np.array(cols).T.T          # synthesis matrix
    _W_CACHE[key] = W
    return W


def ksvd_fit(Xtr, n_atoms=256, sparsity=16, n_iter=20, seed=0, max_train=20000):
    """
    Learn an overcomplete dictionary from *training* ECG windows (K-SVD family).
    Implemented with scikit-learn's MiniBatchDictionaryLearning (online K-SVD
    variant, OMP sparse coder), which scales to the ~10^5 windows we have.
    Returns D (N x n_atoms) with unit-norm columns so that x ~ D @ s.
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(Xtr, dtype=np.float64)
    if X.shape[0] > max_train:
        X = X[rng.choice(X.shape[0], max_train, replace=False)]
    dl = MiniBatchDictionaryLearning(
        n_components=n_atoms, transform_n_nonzero_coefs=sparsity,
        transform_algorithm="omp", max_iter=n_iter, batch_size=256,
        fit_algorithm="lars", random_state=seed,
    )
    dl.fit(X)
    D = dl.components_.T                       # (N, n_atoms)
    D /= (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    return D


# ------------------------------- solvers -----------------------------------
def _soft(z, thr):
    return np.sign(z) * np.maximum(np.abs(z) - thr, 0.0)


def ista_l1(Y, Phi, D, lam=0.05, n_iter=300):
    """min_s 0.5||Phi D s - y||^2 + lam||s||_1  (ISTA), x = D s."""
    A = Phi @ D
    L = np.linalg.norm(A, 2) ** 2
    step = 1.0 / L
    S = np.zeros((Y.shape[0], A.shape[1]))
    for _ in range(n_iter):
        S = _soft(S - step * ((S @ A.T - Y) @ A), lam * step)
    return S @ D.T


def fista_l1(Y, Phi, D, lam=0.05, n_iter=300):
    """Same objective, Nesterov-accelerated (FISTA, Beck & Teboulle 2009)."""
    A = Phi @ D
    L = np.linalg.norm(A, 2) ** 2
    step = 1.0 / L
    S = np.zeros((Y.shape[0], A.shape[1]))
    Z = S.copy()
    t = 1.0
    for _ in range(n_iter):
        S_new = _soft(Z - step * ((Z @ A.T - Y) @ A), lam * step)
        t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
        Z = S_new + ((t - 1.0) / t_new) * (S_new - S)
        S, t = S_new, t_new
    return S @ D.T


def omp_dict(Y, Phi, D, n_nonzero=None):
    """Orthogonal matching pursuit over dictionary D."""
    A = Phi @ D
    M = A.shape[0]
    if n_nonzero is None:
        n_nonzero = max(1, int(0.5 * M))
    n_nonzero = int(min(n_nonzero, M - 1 if M > 1 else 1))
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero, fit_intercept=False)
    Xhat = np.empty((Y.shape[0], D.shape[0]))
    for i in range(Y.shape[0]):
        model.fit(A, Y[i])
        Xhat[i] = D @ model.coef_
    return Xhat


def tune_lambda(Yval, Xval, Phi, D, solver=ista_l1, grid=None, n_iter=200):
    """Grid-search lambda on a validation split; returns (best_lam, table)."""
    from cs_common2 import prdn
    if grid is None:
        grid = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.20]
    table = []
    best = (None, np.inf)
    for lam in grid:
        Xh = solver(Yval, Phi, D, lam=lam, n_iter=n_iter)
        v = float(np.mean(prdn(Xval, Xh)))
        table.append((lam, v))
        if v < best[1]:
            best = (lam, v)
    return best[0], table


# ------------------------------- BSBL-BO -----------------------------------
def _ar1_toeplitz(R, block_size, floor=0.0):
    """Zhang & Rao's robust AR(1) parameterisation of the intra-block
    covariance: r = m1/m0 with m0 the mean of the main diagonal and m1 the mean
    of the first sub-diagonal (means, not traces — the two diagonals have
    different lengths and using raw traces biases r towards 0)."""
    m0 = np.mean(np.diag(R))
    m1 = np.mean(np.diag(R, 1)) if block_size > 1 else 0.0
    r = 0.0 if m0 <= 1e-12 else float(np.clip(m1 / m0, -0.95, 0.95))
    idx = np.arange(block_size)
    B = r ** np.abs(idx[:, None] - idx[None, :])
    return B + floor * np.eye(block_size)


def dct_matrix(n=N):
    """Orthonormal DCT-II synthesis matrix D (n x n): x = D @ theta."""
    k = np.arange(n)
    D = np.cos(np.pi * (2 * k[:, None] + 1) * k[None, :] / (2 * n))
    D[:, 0] *= np.sqrt(1.0 / n)
    D[:, 1:] *= np.sqrt(2.0 / n)
    return D


def bsbl_bo(Y, Phi, D=None, block_size=16, max_iter=60, lam_noise=1e-8,
            prune=1e-4, learn_corr=True, learn_noise=False, tol=1e-4):
    """BSBL-EM.  Dispatches to the batched GPU implementation when one is
    available (14x faster, numerically identical to 1e-9); set BSBL_GPU=0 to
    force the reference NumPy path."""
    import os as _os
    if _os.environ.get("BSBL_GPU", "1") == "1" and not learn_noise:
        try:
            import torch as _torch
            if _torch.cuda.is_available():
                from classical_gpu import bsbl_bo_gpu
                return bsbl_bo_gpu(Y, Phi, D=D, block_size=block_size,
                                   max_iter=max_iter, lam_noise=lam_noise,
                                   prune=prune, learn_corr=learn_corr)
        except Exception as _e:
            print(f"  [bsbl] GPU path unavailable ({_e}); using NumPy", flush=True)
    return _bsbl_bo_cpu(Y, Phi, D, block_size, max_iter, lam_noise, prune,
                        learn_corr, learn_noise, tol)


def _bsbl_bo_cpu(Y, Phi, D=None, block_size=16, max_iter=60, lam_noise=1e-8,
                 prune=1e-4, learn_corr=True, learn_noise=False, tol=1e-4):
    """
    Block Sparse Bayesian Learning, bound-optimisation variant (BSBL-BO),
    Zhang & Rao, IEEE JSTSP 2011 / TBME 2013.  Operates directly in the time
    domain, which is why it works on ECG without a sparsifying transform: the
    signal is block-sparse in blocks of `block_size` consecutive samples with
    intra-block correlation.

    Y: (B, M) measurements, Phi: (M, N).  Returns Xhat (B, N).
    """
    # Zhang & Rao apply BSBL to ECG through a representation in which the
    # signal is block sparse (the DCT domain); raw time-domain ECG is not
    # block sparse, so recovering x directly cripples the baseline.
    Phi = np.asarray(Phi, dtype=np.float64)
    if D is None:
        D = dct_matrix(Phi.shape[1])
    D = np.asarray(D, dtype=np.float64)
    Phi = Phi @ D                                   # recover theta, x = D theta
    M, n = Phi.shape
    nb = n // block_size
    assert nb * block_size == n, "N must be divisible by block_size"
    blocks = [slice(i * block_size, (i + 1) * block_size) for i in range(nb)]

    Xhat = np.zeros((Y.shape[0], n))
    for bi in range(Y.shape[0]):
        y = Y[bi]
        gammas = np.ones(nb)
        Bs = [np.eye(block_size) for _ in range(nb)]
        active = list(range(nb))
        mu = np.zeros(n)
        lam = float(lam_noise)
        prev = None
        for _ in range(max_iter):
            # Sigma_0 = blkdiag(gamma_i B_i) restricted to active blocks
            Sig0 = np.zeros((n, n))
            for i in active:
                Sig0[blocks[i], blocks[i]] = gammas[i] * Bs[i]
            PS = Phi @ Sig0
            Sigma_y = PS @ Phi.T + lam * np.eye(M)
            try:
                Sy_inv = np.linalg.inv(Sigma_y)
            except np.linalg.LinAlgError:
                Sy_inv = np.linalg.pinv(Sigma_y)
            mu = PS.T @ (Sy_inv @ y)
            # posterior block covariances and gamma update (bound optimisation)
            new_active, sig_trace = [], 0.0
            Rsum = np.zeros((block_size, block_size))
            for i in active:
                sl = blocks[i]
                Phi_i = Phi[:, sl]
                Sig_i = gammas[i] * Bs[i]
                Sigma_i = Sig_i - Sig_i @ Phi_i.T @ Sy_inv @ Phi_i @ Sig_i
                mu_i = mu[sl]
                Bi_inv = np.linalg.pinv(Bs[i] + 1e-6 * np.eye(block_size))
                # EM update (BSBL-EM, Zhang & Rao): gamma_i <- Tr[B_i^{-1}
                # (Sigma_i + mu_i mu_i^T)] / d.  The bound-optimisation form
                # sqrt(num/den) has a non-zero fixed point for inactive blocks,
                # so it never prunes; the EM form shrinks them to zero.
                gammas[i] = float(max(
                    np.trace(Bi_inv @ (Sigma_i + np.outer(mu_i, mu_i))), 0.0
                ) / block_size)
                if gammas[i] > prune:
                    new_active.append(i)
                    sig_trace += np.trace(Phi_i @ Sigma_i @ Phi_i.T)
                    # pooled intra-block correlation: one B shared by all blocks
                    # (Zhang & Rao recommend pooling; per-block B overfits with
                    #  a single window and makes the bound diverge)
                    Rsum += (Sigma_i + np.outer(mu_i, mu_i)) / (gammas[i] + 1e-12)
            active = new_active
            if not active:
                break
            if learn_corr:
                B = _ar1_toeplitz(Rsum / len(active), block_size, floor=1e-3)
                for i in active:
                    Bs[i] = B
            if learn_noise:
                resid = y - Phi @ mu
                lam = float(max((resid @ resid + sig_trace) / M, 1e-10))
            if prev is not None and np.linalg.norm(mu - prev) <= tol * (
                    np.linalg.norm(prev) + 1e-12):
                break
            prev = mu.copy()
        Xhat[bi] = D @ mu
    return Xhat
