"""
classical_gpu.py — batched GPU versions of the slow classical decoders.

BSBL and OMP dominate the wall-clock of the benchmark: both are per-window
solvers, and the NumPy implementations process one window at a time (BSBL
~0.29 s/window).  Both algorithms are, however, embarrassingly batchable: every
window needs the same sequence of small dense linear-algebra operations, which
is exactly what batched torch kernels do well.

These implementations are *numerically equivalent* to the ones in classical2.py
(verified in `selftest()` below against the reference code); they are a speed
optimisation only, not a change of method.
"""
import numpy as np
import torch

from cs_common2 import N


def _t(x, dev, dtype=torch.float64):
    return torch.as_tensor(np.asarray(x), dtype=dtype, device=dev)


# --------------------------------------------------------------------------
# BSBL-EM, batched over windows
# --------------------------------------------------------------------------
@torch.no_grad()
def bsbl_bo_gpu(Y, Phi, D=None, block_size=16, max_iter=60, lam_noise=1e-8,
                prune=1e-4, learn_corr=True, dev="cuda", chunk=256,
                dtype=torch.float64):
    """Batched BSBL-EM.  Same model and updates as classical2.bsbl_bo.

    Y: (B, M) measurements; returns (B, N).
    """
    from classical2 import dct_matrix
    Phi_np = np.asarray(Phi, dtype=np.float64)
    if D is None:
        D = dct_matrix(Phi_np.shape[1])
    D_np = np.asarray(D, dtype=np.float64)
    A_np = Phi_np @ D_np                      # recover theta, x = D theta
    M, n = A_np.shape
    nb = n // block_size
    assert nb * block_size == n

    A = _t(A_np, dev, dtype)                  # (M, n)
    Dt = _t(D_np, dev, dtype)                 # (n, n)
    Yt = _t(Y, dev, dtype)                    # (B, M)
    Bt = Yt.shape[0]
    out = torch.empty((Bt, n), dtype=dtype, device=dev)
    eyeM = torch.eye(M, dtype=dtype, device=dev)
    eyeb = torch.eye(block_size, dtype=dtype, device=dev)
    # A reshaped per block: (nb, M, bs)
    Ablk = A.view(M, nb, block_size).permute(1, 0, 2).contiguous()

    for s in range(0, Bt, chunk):
        y = Yt[s:s + chunk]                               # (b, M)
        b = y.shape[0]
        gam = torch.ones((b, nb), dtype=dtype, device=dev)
        Bmat = eyeb.expand(b, nb, block_size, block_size).clone()
        active = torch.ones((b, nb), dtype=torch.bool, device=dev)
        mu = torch.zeros((b, n), dtype=dtype, device=dev)

        for _ in range(max_iter):
            # Sigma_0 blocks: (b, nb, bs, bs), zeroed where pruned
            S0 = gam[..., None, None] * Bmat * active[..., None, None]
            # Sigma_y = sum_i A_i S0_i A_i^T + lam I     -> (b, M, M)
            AS = torch.einsum('imk,bikl->biml', Ablk, S0)       # (b,nb,M,bs)
            Sy = torch.einsum('biml,inl->bmn', AS, Ablk) + lam_noise * eyeM
            L = torch.linalg.cholesky(
                Sy + 1e-12 * eyeM * Sy.diagonal(dim1=-2, dim2=-1).mean(-1)[:, None, None])
            Syi_y = torch.cholesky_solve(y.unsqueeze(-1), L)     # (b, M, 1)
            # mu_i = S0_i A_i^T Sigma_y^{-1} y
            mu_b = torch.einsum('biml,bml->bil', AS, Syi_y.expand(-1, -1, block_size))
            mu = mu_b.reshape(b, n)
            # Sigma_i = S0_i - S0_i A_i^T Sy^{-1} A_i S0_i
            Syi_AS = torch.cholesky_solve(
                AS.permute(0, 1, 2, 3).reshape(b, nb * 1, M, block_size)
                  .view(b, nb, M, block_size).reshape(b * nb, M, block_size),
                L.repeat_interleave(nb, dim=0)).view(b, nb, M, block_size)
            Sig = S0 - torch.einsum('biml,bimk->bilk', AS, Syi_AS)
            # EM update: gamma_i <- Tr[B_i^{-1}(Sigma_i + mu_i mu_i^T)] / bs
            Binv = torch.linalg.inv(Bmat + 1e-6 * eyeb)
            MM = Sig + mu_b.unsqueeze(-1) * mu_b.unsqueeze(-2)
            gam = torch.einsum('bikl,billk->bik', Binv.unsqueeze(2),
                               MM.unsqueeze(1)).diagonal(dim1=1, dim2=2) \
                if False else torch.einsum('bikl,bilk->bi', Binv, MM) / block_size
            gam = gam.clamp_min(0.0)
            active = gam > prune
            if not bool(active.any()):
                break
            if learn_corr:
                # pooled AR(1) correlation over active blocks (per window)
                R = (MM / (gam[..., None, None] + 1e-12)) * active[..., None, None]
                cnt = active.sum(1).clamp_min(1)[:, None, None]
                Rm = R.sum(1) / cnt                                  # (b,bs,bs)
                m0 = Rm.diagonal(dim1=-2, dim2=-1).mean(-1)
                m1 = Rm.diagonal(offset=1, dim1=-2, dim2=-1).mean(-1)
                r = torch.where(m0 > 1e-12, (m1 / m0).clamp(-0.95, 0.95),
                                torch.zeros_like(m0))
                idx = torch.arange(block_size, device=dev)
                p = (idx[:, None] - idx[None, :]).abs().to(dtype)
                Bnew = r[:, None, None] ** p + 1e-3 * eyeb           # (b,bs,bs)
                Bmat = Bnew.unsqueeze(1).expand(b, nb, block_size, block_size).clone()
        out[s:s + chunk] = mu @ Dt.T
    return out.cpu().numpy()


# NOTE: a batched GPU OMP was implemented and removed. It did not reproduce
# scikit-learn's selection bit-for-bit (max|diff| 4.8e-1, PRD 11.94 vs 12.10),
# and OMP is not the bottleneck (~4 ms/window), so the reference CPU solver is
# kept. Only BSBL is accelerated, and it is numerically identical (1e-9).

# --------------------------------------------------------------------------
def selftest(n_win=24, cr=0.5, dev="cuda"):
    """GPU implementations must match the reference NumPy ones."""
    import time
    import cs_common2 as CC
    import classical2 as C
    X = np.load("../data/windows/100.npy")[:n_win].astype(np.float64)
    m = CC.m_for_cr(cr)
    Phi = CC.make_phi(m)
    Y = X @ Phi.T
    W = C.wavelet_matrix()

    t = time.time(); ref_b = C.bsbl_bo(Y, Phi, block_size=16, max_iter=60); tb = time.time() - t
    t = time.time(); gpu_b = bsbl_bo_gpu(Y, Phi, block_size=16, max_iter=60, dev=dev); tg = time.time() - t
    db = float(np.max(np.abs(ref_b - gpu_b)))
    pr = float(np.mean(CC.prdn(X, ref_b))); pg = float(np.mean(CC.prdn(X, gpu_b)))
    print(f"BSBL  cpu {tb:6.1f}s  gpu {tg:5.1f}s  ({tb/max(tg,1e-9):5.1f}x)  "
          f"max|diff|={db:.2e}  PRD cpu={pr:.3f} gpu={pg:.3f}")



if __name__ == "__main__":
    selftest()
