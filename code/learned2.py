"""
learned2.py — learned CS-ECG decoders (v2).

The SPL review required comparison against *published* deep CS-ECG decoders,
not only against classical recovery.  Three networks are implemented here and
trained under one identical protocol:

  ISTANetPlus   deep-unrolled proximal gradient, ISTA-Net+ formulation
                (Zhang & Ghanem, CVPR 2018) — the decoder studied in v1.
  HuaNet        re-implementation of the modified-Inception + LSTM decoder of
                Hua, Rao, Peng, Liu & Tang, Entropy 24(8):1024, 2022
                (doi:10.3390/e24081024).  Published description: adaptive
                compression by three Conv1d(k=4, s=4) layers with
                C = 64 * SR filters; initial reconstruction Conv1d(k=1) with 64
                filters + LeakyReLU and reshape to N; a four-branch Inception
                block with kernels 7/9/11/13, each branch two convolutions with
                an additive skip; LSTM for temporal dependencies; MSE loss,
                Adam, lr 5e-4, batch 32, N = 256.
  DilatedUnrolled  re-implementation following the description of Erkoc,
                Circuits Syst. Signal Process. 45(9):7210-7233, 2026
                (doi:10.1007/s00034-026-03506-5): compressed sensing with an
                unrolled architecture whose proximal operator is built from
                *dilated* CNN blocks.  The paper is not open access; only the
                architectural family (unrolled + dilated CNN prox) is specified
                in the public record, so this is a faithful-family
                re-implementation and is labelled as such everywhere.

Two sensing protocols, both reported:
  "fixed"   : y = Phi x with the *same* fixed Phi for every method
              (decoder-only comparison — the fair one for our question);
  "learned" : the network learns its own measurement operator, as published
              for HuaNet (an upper bound that no fixed-matrix decoder can match).
"""
import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------
# sensing front-ends
# --------------------------------------------------------------------------
class FixedSensing(nn.Module):
    """y = Phi x, Phi fixed (not trained)."""

    def __init__(self, Phi):
        super().__init__()
        self.register_buffer("Phi", torch.tensor(np.asarray(Phi), dtype=torch.float32))

    def forward(self, x):
        return x @ self.Phi.t()


class LearnedSensing(nn.Module):
    """Hua et al.: three Conv1d(kernel 4, stride 4) layers, C = 64 * SR."""

    def __init__(self, n=256, m=128):
        super().__init__()
        sr = m / n
        c = max(1, int(round(64 * sr)))
        self.c = c
        self.net = nn.Sequential(
            nn.Conv1d(1, c, 4, stride=4),
            nn.Conv1d(c, c, 4, stride=4),
            nn.Conv1d(c, c, 4, stride=4),
        )
        self.out_dim = c * (n // 64)

    def forward(self, x):
        return self.net(x.unsqueeze(1)).flatten(1)


# --------------------------------------------------------------------------
# 1) ISTA-Net+  (ours / v1)
# --------------------------------------------------------------------------

def _step0(Phi):
    """Initial gradient step size 1/||Phi||_2^2 (the ISTA/FISTA step).

    A fixed initial step is only safe when the operator is well conditioned.
    A sparse binary Phi has a much larger Lipschitz constant than a Gaussian one
    (33 vs 8 here), and a fixed 0.1 makes the unrolled gradient step diverge.
    Scaling the initial step by the Lipschitz constant is what the classical
    solvers do, and it is applied identically to every network and every
    sensing matrix.
    """
    L2 = float(np.linalg.norm(np.asarray(Phi, dtype=np.float64), 2) ** 2)
    return 1.0 / max(L2, 1e-8)


class ProxBlock(nn.Module):
    def __init__(self, n_feat=32, ksize=9, rho0=0.1):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(float(rho0)))
        self.thr = nn.Parameter(torch.tensor(0.01))
        pad = ksize // 2
        self.F = nn.Sequential(
            nn.Conv1d(1, n_feat, ksize, padding=pad), nn.ReLU(inplace=True),
            nn.Conv1d(n_feat, n_feat, ksize, padding=pad))
        self.G = nn.Sequential(
            nn.Conv1d(n_feat, n_feat, ksize, padding=pad), nn.ReLU(inplace=True),
            nn.Conv1d(n_feat, 1, ksize, padding=pad))
        nn.init.zeros_(self.G[-1].weight)
        nn.init.zeros_(self.G[-1].bias)

    def soft(self, z):
        return torch.sign(z) * torch.relu(torch.abs(z) - torch.abs(self.thr))

    def forward(self, x, y, Phi):
        r = x - self.rho * ((x @ Phi.t() - y) @ Phi)
        f = self.F(r.unsqueeze(1))
        g = self.G(self.soft(f)).squeeze(1)
        sym = self.G(f).squeeze(1) - r
        return r + g, sym


class ISTANetPlus(nn.Module):
    def __init__(self, Phi, n_stages=8, n_feat=32):
        super().__init__()
        self.register_buffer("Phi", torch.tensor(np.asarray(Phi), dtype=torch.float32))
        r0 = _step0(Phi)
        self.blocks = nn.ModuleList([ProxBlock(n_feat, rho0=r0)
                                     for _ in range(n_stages)])

    def forward(self, y):
        x = y @ self.Phi                      # Phi^T y  (initialisation)
        syms = []
        for b in self.blocks:
            x, s = b(x, y, self.Phi)
            syms.append(s)
        return x, syms


# --------------------------------------------------------------------------
# 2) HuaNet  [S2] Entropy 2022
# --------------------------------------------------------------------------
class InceptionBranch(nn.Module):
    def __init__(self, ch, k):
        super().__init__()
        pad = k // 2
        self.f1 = nn.Conv1d(ch, ch, k, padding=pad)
        self.f2 = nn.Conv1d(ch, ch, k, padding=pad)
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        return self.act(self.f2(self.act(self.f1(x)))) + x


class ModifiedInception(nn.Module):
    """Four parallel branches, kernels 7/9/11/13, concatenated to 64 channels."""

    def __init__(self, ch_total=64):
        super().__init__()
        cb = ch_total // 4
        self.split = cb
        self.branches = nn.ModuleList([InceptionBranch(cb, k) for k in (7, 9, 11, 13)])

    def forward(self, x):
        parts = torch.split(x, self.split, dim=1)
        return torch.cat([b(p) for b, p in zip(self.branches, parts)], dim=1)


class HuaNet(nn.Module):
    def __init__(self, Phi=None, n=256, m=128, sensing="fixed",
                 n_incep=3, lstm_hidden=64, lstm_layers=1):
        super().__init__()
        self.n = n
        self.sensing_mode = sensing
        if sensing == "learned":
            self.sense = LearnedSensing(n, m)
            in_dim = self.sense.out_dim
        else:
            self.sense = FixedSensing(Phi)
            in_dim = Phi.shape[0]
        # initial reconstruction: 1x1 conv with 64 filters + LeakyReLU, reshape
        self.init_fc = nn.Linear(in_dim, n)
        self.init_conv = nn.Sequential(nn.Conv1d(1, 64, 1), nn.LeakyReLU(0.2, inplace=True))
        self.incep = nn.Sequential(*[ModifiedInception(64) for _ in range(n_incep)])
        self.lstm = nn.LSTM(64, lstm_hidden, num_layers=lstm_layers, batch_first=True)
        self.out = nn.Conv1d(lstm_hidden, 1, 1)

    def forward(self, x_or_y, y=None):
        if self.sensing_mode == "learned":
            meas = self.sense(x_or_y)
        else:
            meas = x_or_y if y is None else y
        h = self.init_conv(self.init_fc(meas).unsqueeze(1))     # (B,64,N)
        h = self.incep(h)
        h, _ = self.lstm(h.transpose(1, 2))                      # (B,N,H)
        return self.out(h.transpose(1, 2)).squeeze(1), []


# --------------------------------------------------------------------------
# 3) DilatedUnrolled  [S1] CSSP 2026 (family re-implementation)
# --------------------------------------------------------------------------
class DilatedProx(nn.Module):
    def __init__(self, n_feat=32, ksize=5, dilations=(1, 2, 4, 8), rho0=0.1):
        super().__init__()
        self.rho = nn.Parameter(torch.tensor(float(rho0)))
        layers, ch_in = [], 1
        for d in dilations:
            layers += [nn.Conv1d(ch_in, n_feat, ksize, padding=d * (ksize // 2), dilation=d),
                       nn.LeakyReLU(0.2, inplace=True)]
            ch_in = n_feat
        layers += [nn.Conv1d(n_feat, 1, ksize, padding=ksize // 2)]
        self.body = nn.Sequential(*layers)
        nn.init.zeros_(self.body[-1].weight)
        nn.init.zeros_(self.body[-1].bias)

    def forward(self, x, y, Phi):
        r = x - self.rho * ((x @ Phi.t() - y) @ Phi)
        return r + self.body(r.unsqueeze(1)).squeeze(1)


class DilatedUnrolled(nn.Module):
    def __init__(self, Phi, n_stages=6, n_feat=32):
        super().__init__()
        self.register_buffer("Phi", torch.tensor(np.asarray(Phi), dtype=torch.float32))
        r0 = _step0(Phi)
        self.blocks = nn.ModuleList([DilatedProx(n_feat, rho0=r0)
                                     for _ in range(n_stages)])

    def forward(self, y):
        x = y @ self.Phi
        for b in self.blocks:
            x = b(x, y, self.Phi)
        return x, []


# --------------------------------------------------------------------------
# shared training / inference
# --------------------------------------------------------------------------
MODELS = {"istanet": ISTANetPlus, "hua": HuaNet, "dilated": DilatedUnrolled}


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(model, Xtr, Phi, epochs=30, bs=64, lr=1e-3, dev="cuda",
                sym_w=0.01, seed=0, quant_bits=None, verbose=False,
                val_frac=0.1, patience=0, max_epochs=None):
    """One identical training protocol for every learned decoder.

    With `patience > 0` the budget is decided by the model itself: training runs
    until the validation loss has not improved for `patience` epochs (or until
    `max_epochs`), and the best-validation weights are restored.  This removes
    the fixed-epoch budget as a confound — a published decoder trained for 200
    epochs by its authors must not be judged on 25 — and makes the comparison
    budget-independent rather than budget-matched.
    """
    from cs_common2 import quantize
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = model.to(dev).train()
    total_epochs = int(max_epochs or epochs)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs)
    Xall = torch.tensor(Xtr, dtype=torch.float32)
    if patience > 0 and val_frac > 0:
        nv = max(256, int(len(Xall) * val_frac))
        Xva, Xt = Xall[:nv], Xall[nv:]
    else:
        Xva, Xt = None, Xall
    Phit = torch.tensor(np.asarray(Phi), dtype=torch.float32, device=dev)
    nb = int(np.ceil(len(Xt) / bs))
    learned_sensing = getattr(model, "sensing_mode", "fixed") == "learned"
    hist = []
    best = (np.inf, 0, None)                       # (val loss, epoch, state)
    for ep in range(total_epochs):
        perm = torch.randperm(len(Xt))
        tot = 0.0
        for i in range(nb):
            xb = Xt[perm[i * bs:(i + 1) * bs]].to(dev)
            if learned_sensing:
                out = model(xb)
            else:
                yb = xb @ Phit.t()
                if quant_bits is not None:
                    yb = torch.tensor(quantize(yb.cpu().numpy(), quant_bits),
                                      dtype=torch.float32, device=dev)
                out = model(yb)
            xh, syms = out if isinstance(out, tuple) else (out, [])
            loss = torch.mean((xh - xb) ** 2)
            if syms:
                loss = loss + sym_w * sum(torch.mean(s ** 2) for s in syms)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(xb)
        sched.step()
        hist.append(tot / len(Xt))
        if Xva is not None:
            model.eval()
            # batched: the recurrent decoder needs several GiB if the whole
            # validation split is pushed through in one call
            tot_v, n_v = 0.0, 0
            with torch.no_grad():
                for j in range(0, len(Xva), 256):
                    vb = Xva[j:j + 256].to(dev)
                    if learned_sensing:
                        out = model(vb)
                    else:
                        yv = vb @ Phit.t()
                        if quant_bits is not None:
                            yv = torch.tensor(quantize(yv.cpu().numpy(), quant_bits),
                                              dtype=torch.float32, device=dev)
                        out = model(yv)
                    xh = out[0] if isinstance(out, tuple) else out
                    tot_v += float(torch.sum((xh - vb) ** 2))
                    n_v += vb.numel()
            vloss = tot_v / max(n_v, 1)
            model.train()
            if vloss < best[0] - 1e-6:
                best = (vloss, ep,
                        {k: v.detach().clone() for k, v in model.state_dict().items()})
            elif ep - best[1] >= patience:
                break
        if verbose and (ep % 5 == 0 or ep == total_epochs - 1):
            print(f"    ep{ep:3d} loss={hist[-1]:.5f}", flush=True)
    if best[2] is not None:
        model.load_state_dict(best[2])
    model.epochs_run = len(hist)
    model.best_epoch = best[1]
    return model.eval(), hist


@torch.no_grad()
def reconstruct(model, X, Phi, dev="cuda", bs=512, quant_bits=None):
    from cs_common2 import quantize
    model = model.to(dev).eval()
    Phit = torch.tensor(np.asarray(Phi), dtype=torch.float32, device=dev)
    learned_sensing = getattr(model, "sensing_mode", "fixed") == "learned"
    out = np.empty_like(np.asarray(X, dtype=np.float32))
    for i in range(0, len(X), bs):
        xb = torch.tensor(X[i:i + bs], dtype=torch.float32, device=dev)
        if learned_sensing:
            res = model(xb)
        else:
            yb = xb @ Phit.t()
            if quant_bits is not None:
                yb = torch.tensor(quantize(yb.cpu().numpy(), quant_bits),
                                  dtype=torch.float32, device=dev)
            res = model(yb)
        xh = res[0] if isinstance(res, tuple) else res
        out[i:i + bs] = xh.cpu().numpy()
    return out
