"""
classical.py
Classical sparse-recovery baselines for compressed-sensing ECG.

Sparsifying basis: orthonormal Daubechies-4 (db4) discrete wavelet transform,
implemented as an explicit N x N matrix W so that x = W @ s with s the wavelet
coefficients (W orthonormal => W.T = W^{-1}).  Reconstruction operates in the
wavelet domain: A = Phi @ W maps coefficients to measurements.

Two solvers:
  * ista_l1  : wavelet-domain L1 minimisation via the Iterative Shrinkage-
               Thresholding Algorithm (ISTA / soft-thresholding).
  * omp      : Orthogonal Matching Pursuit on A (greedy support selection).
"""
import numpy as np
import pywt
from sklearn.linear_model import OrthogonalMatchingPursuit

from cs_common import N


def wavelet_matrix(n=N, wavelet="db4"):
    """Build orthonormal DWT synthesis matrix W (n x n): x = W @ s."""
    w = pywt.Wavelet(wavelet)
    level = pywt.dwt_max_level(n, w.dec_len)
    # analysis matrix: rows are basis vectors -> A_dwt @ x = coeffs
    cols = []
    for i in range(n):
        e = np.zeros(n)
        e[i] = 1.0
        coeffs = pywt.wavedec(e, w, level=level, mode="periodization")
        cols.append(np.concatenate([c.ravel() for c in coeffs]))
    A_dwt = np.array(cols).T  # A_dwt @ x = coeffs ; orthonormal
    W = A_dwt.T  # synthesis: x = W @ s
    return W


def _soft(z, thr):
    return np.sign(z) * np.maximum(np.abs(z) - thr, 0.0)


def ista_l1(Y, Phi, W, lam=0.05, n_iter=300):
    """
    Wavelet-domain L1 recovery via ISTA.
    Y: (B, M) measurements; returns xhat: (B, N).
    Solve  min_s 0.5||Phi W s - y||^2 + lam||s||_1, then x = W s.
    """
    A = Phi @ W  # (M, N)
    # step size = 1 / Lipschitz constant
    L = np.linalg.norm(A, 2) ** 2
    step = 1.0 / L
    AT = A.T
    B = Y.shape[0]
    S = np.zeros((B, A.shape[1]))
    for _ in range(n_iter):
        grad = (S @ AT - Y) @ A  # (B,N)
        S = _soft(S - step * grad, lam * step)
    Xhat = S @ W.T
    return Xhat


def omp(Y, Phi, W, n_nonzero=None):
    """Orthogonal Matching Pursuit recovery, per-window."""
    A = Phi @ W  # (M, N)
    M = A.shape[0]
    if n_nonzero is None:
        n_nonzero = max(1, int(0.5 * M))
    model = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero, fit_intercept=False)
    Xhat = np.empty((Y.shape[0], N))
    for i in range(Y.shape[0]):
        model.fit(A, Y[i])
        Xhat[i] = W @ model.coef_
    return Xhat
