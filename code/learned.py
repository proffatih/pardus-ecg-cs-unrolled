"""
learned.py
Deep-unrolled compressed-sensing reconstruction network for ECG (ISTA-Net-style).

Each of K stages unrolls one proximal-gradient (ISTA) step:

    r = x_{k-1} - rho_k * Phi^T (Phi x_{k-1} - y)      (gradient/data step)
    x_k = r + G_k( soft( F_k(r), theta_k ) )           (learned proximal step)

where F_k and G_k are small 1-D convolutional operators and soft() is
element-wise soft-thresholding with a learned threshold (the ISTA-Net design of
Zhang & Ghanem, CVPR 2018).  A symmetry/constraint loss encourages G_k(F_k(.))
to behave like an (approximate) inverse pair, as in the original ISTA-Net.

The same architecture is trained separately for each compression ratio.
"""
import numpy as np
import torch
import torch.nn as nn


class ProxBlock(nn.Module):
    """One ISTA-Net stage operating on 1-D signals of length N."""

    def __init__(self, n_feat=32, ksize=9):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(0.1))
        self.thr = nn.Parameter(torch.tensor(0.01))
        pad = ksize // 2
        # F: lift to feature space (2 convs); G: project back (2 convs)
        self.F = nn.Sequential(
            nn.Conv1d(1, n_feat, ksize, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feat, n_feat, ksize, padding=pad),
        )
        self.G = nn.Sequential(
            nn.Conv1d(n_feat, n_feat, ksize, padding=pad),
            nn.ReLU(inplace=True),
            nn.Conv1d(n_feat, 1, ksize, padding=pad),
        )
        # near-identity init: last conv of G starts at zero so each stage begins
        # as an (approximate) plain proximal-gradient step -> stable training.
        nn.init.zeros_(self.G[-1].weight)
        nn.init.zeros_(self.G[-1].bias)

    def soft(self, z):
        return torch.sign(z) * torch.relu(torch.abs(z) - torch.abs(self.thr))

    def forward(self, x, y, Phi):
        # x: (B,N)  y: (B,M)  Phi: (M,N)
        resid = x @ Phi.t() - y          # (B,M)
        grad = resid @ Phi               # (B,N)
        r = x - self.rho * grad
        ru = r.unsqueeze(1)              # (B,1,N)
        f = self.F(ru)
        st = self.soft(f)
        g = self.G(st)
        x_new = (r + g.squeeze(1))
        # symmetry constraint term: G(F(r)) ~ r  (returned for loss)
        sym = self.G(f).squeeze(1) - r
        return x_new, sym


class ISTANet(nn.Module):
    def __init__(self, Phi, n_stages=8, n_feat=32):
        super().__init__()
        self.register_buffer("Phi", torch.tensor(np.asarray(Phi), dtype=torch.float32))
        self.blocks = nn.ModuleList([ProxBlock(n_feat) for _ in range(n_stages)])

    def forward(self, y):
        # initial estimate x0 = Phi^T y
        x = y @ self.Phi                  # (B,N)
        syms = []
        for blk in self.blocks:
            x, sym = blk(x, y, self.Phi)
            syms.append(sym)
        return x, syms


def train_model(model, Xtr, device, epochs=40, batch=256, lr=1e-3,
                gamma=0.01, seed=0, verbose=True):
    torch.manual_seed(seed)
    model.to(device)
    Phi = model.Phi
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    n = Xtr_t.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n, generator=g)
        tot = 0.0
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            x = Xtr_t[idx]
            y = x @ Phi.t()
            xhat, syms = model(y)
            mse = ((xhat - x) ** 2).mean()
            sym = sum((s ** 2).mean() for s in syms) / len(syms)
            loss = mse + gamma * sym
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * x.shape[0]
        sched.step()
        if verbose and (ep % 5 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  loss {tot / n:.5f}")
    return model


@torch.no_grad()
def reconstruct(model, Xte, device, batch=512):
    model.eval()
    Phi = model.Phi
    out = np.empty_like(Xte)
    Xte_t = torch.tensor(Xte, dtype=torch.float32, device=device)
    for i in range(0, Xte_t.shape[0], batch):
        x = Xte_t[i:i + batch]
        y = x @ Phi.t()
        xhat, _ = model(y)
        out[i:i + batch] = xhat.cpu().numpy()
    return out
