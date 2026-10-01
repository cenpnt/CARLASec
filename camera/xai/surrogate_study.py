"""Differentiable surrogate for the gradient-boosted tree detectors, and the
study that selected it.

Candidates are scored on three fidelity measures against the tree:
  AUC        discriminative power
  spearman   rank agreement with the tree's score
  agree@5%   same flag decision as the tree at a 5% clean FPR

Selected (2026-09-19): label targets, rank transform, 256 hidden, depth 4,
4000 epochs. Distillation on the tree's probabilities did not help, because
the trees separate the training set perfectly and their probabilities
saturate. The rank transform gives the MLP a tree's invariance to monotone
feature transforms.

Run the study (features are cached after the first run):
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u surrogate_study.py
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from model import load_trained, IMG_SIZE
from xai_detect import CKPT, DEVICE
from aim1_attack import attack as plain_attack, predict, load_data
from aim1_adaptive import xai_feats, pixel_feats

CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "_surrogate_cache.npz")
BEST = dict(transform="rank", hidden=256, depth=4, epochs=4000, lr=2e-3)


class MLP(nn.Module):
    """Surrogate with a selectable input transform:
    "std" mean/std, "robust" median/IQR, "rank" differentiable quantile map."""

    def __init__(self, n_feat, hidden=128, depth=3, transform="std"):
        super().__init__()
        self.transform = transform
        self.register_buffer("a", torch.zeros(n_feat))
        self.register_buffer("b", torch.ones(n_feat))
        self.register_buffer("q", torch.zeros(65, n_feat))
        layers, d = [], n_feat
        for _ in range(depth - 1):
            layers += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def fit_transform(self, F):
        t = torch.as_tensor(F, dtype=torch.float32)
        if self.transform == "robust":
            self.a.copy_(t.median(0).values)
            self.b.copy_(t.quantile(0.75, dim=0) - t.quantile(0.25, dim=0) + 1e-6)
        else:
            self.a.copy_(t.mean(0))
            self.b.copy_(t.std(0) + 1e-8)
        levels = torch.linspace(0, 1, 65)
        self.q.copy_(torch.stack([t[:, j].quantile(levels)
                                  for j in range(t.shape[1])], dim=1))

    def _rank(self, F):
        """Piecewise-linear lookup into the training quantiles, to [-1, 1]."""
        lo = self.q[:-1].unsqueeze(0)
        hi = self.q[1:].unsqueeze(0)
        frac = ((F.unsqueeze(1) - lo) / (hi - lo + 1e-9)).clamp(0, 1)
        return (frac.sum(1) / 64.0) * 2.0 - 1.0

    def forward(self, F):
        z = self._rank(F) if self.transform == "rank" else (F - self.a) / self.b
        return self.net(z).squeeze(1)


def train(F, target, transform, hidden, depth, epochs, lr, seed=0):
    torch.manual_seed(seed)
    m = MLP(F.shape[1], hidden, depth, transform)
    m.fit_transform(F)
    m.to(DEVICE)
    Ft = torch.as_tensor(F, dtype=torch.float32, device=DEVICE)
    yt = torch.as_tensor(target, dtype=torch.float32, device=DEVICE)
    opt = torch.optim.Adam(m.parameters(), lr=lr, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    m.train()
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        Fn.binary_cross_entropy_with_logits(m(Ft), yt).backward()
        opt.step()
        sch.step()
    return m.eval()


def fit_surrogate(F, lab, seed=0):
    """Differentiable stand-in for a gradient-boosted tree detector."""
    return train(F, lab, BEST["transform"], BEST["hidden"], BEST["depth"],
                 BEST["epochs"], BEST["lr"], seed=seed)


def score(m, F):
    with torch.no_grad():
        return m(torch.as_tensor(F, dtype=torch.float32,
                                 device=DEVICE)).cpu().numpy()


def fidelity(s_mlp, s_tree, lab, n_clean):
    auc = roc_auc_score(lab, s_mlp)
    rho = spearmanr(s_mlp, s_tree).correlation
    tm = np.quantile(s_mlp[:n_clean], 0.95)
    tt = np.quantile(s_tree[:n_clean], 0.95)
    agree = float(((s_mlp > tm) == (s_tree > tt)).mean())
    return auc, rho, agree


def build_cache(n, eps, steps):
    model, ckpt = load_trained(CKPT, DEVICE)
    X, Y = load_data(model, ckpt.get("img_size", IMG_SIZE), n)
    advs = [plain_attack(model, X, Y, eps=eps, k=96, lam=0.0, steps=steps).float(),
            plain_attack(model, X, Y, eps=eps, k=6, lam=0.0, steps=steps).float()]
    Fx = np.concatenate([xai_feats(model, X, predict(model, X))]
                        + [xai_feats(model, A, predict(model, A)) for A in advs])
    Fp = np.concatenate([pixel_feats(X)] + [pixel_feats(A) for A in advs])
    lab = np.concatenate([np.zeros(len(X))] + [np.ones(len(A)) for A in advs])
    np.savez_compressed(CACHE, Fx=Fx, Fp=Fp, lab=lab, n_clean=len(X))
    return Fx, Fp, lab, len(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if os.path.exists(CACHE) and not args.rebuild:
        d = np.load(CACHE)
        Fx, Fp, lab, n_clean = d["Fx"], d["Fp"], d["lab"], int(d["n_clean"])
    else:
        Fx, Fp, lab, n_clean = build_cache(args.n, args.eps, args.steps)
    print(f"features: xai {Fx.shape}, pixel {Fp.shape}")

    candidates = [
        # name,                     target,   transform, hidden, depth, ep,   lr
        ("labels  std  300",        "label",  "std",      96,    3,     300,  1e-3),
        ("labels  std  2000",       "label",  "std",     128,    3,    2000,  2e-3),
        ("distil  std  2000",       "soft",   "std",     128,    3,    2000,  2e-3),
        ("distil  robust 2000",     "soft",   "robust",  128,    3,    2000,  2e-3),
        ("distil  rank 2000",       "soft",   "rank",    128,    3,    2000,  2e-3),
        ("distil  rank 4000 deep",  "soft",   "rank",    256,    4,    4000,  2e-3),
        ("labels  rank 2000",       "label",  "rank",    128,    3,    2000,  2e-3),
        ("labels  rank 4000 deep",  "label",  "rank",    256,    4,    4000,  2e-3),
        ("labels  std  4000 deep",  "label",  "std",     256,    4,    4000,  2e-3),
        ("labels  rank 8000 deep",  "label",  "rank",    256,    4,    8000,  2e-3),
    ]

    for nm, F in [("PIXEL", Fp), ("XAI", Fx)]:
        tree = HistGradientBoostingClassifier(max_iter=400,
                                              random_state=0).fit(F, lab)
        p = np.clip(tree.predict_proba(F)[:, 1], 1e-6, 1 - 1e-6)
        s_tree = np.log(p / (1 - p))
        soft = np.clip(p, 1e-4, 1 - 1e-4)
        print(f"\n{nm} detector   tree AUC {roc_auc_score(lab, p):.4f}   "
              f"({F.shape[1]} features)")
        print(f"{'surrogate':>24} {'AUC':>8} {'spearman':>10} {'agree@5%':>10}")
        for cname, tgt, tr, h, dp, ep, lr in candidates:
            y = lab if tgt == "label" else soft
            m = train(F, y, tr, h, dp, ep, lr)
            a, r, g = fidelity(score(m, F), s_tree, lab, n_clean)
            print(f"{cname:>24} {a:>8.4f} {r:>10.4f} {g:>10.4f}", flush=True)


if __name__ == "__main__":
    main()
