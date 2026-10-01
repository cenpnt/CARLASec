"""White-box attack on the OR-composition of the pixel and XAI detectors.

Each detector score is put in threshold-relative units, u <= 0 meaning "not
flagged", and the attack minimises a soft maximum of the two, since the OR
fires if either does:

    objective = L_CW + lambda * T * logsumexp([u_pixel, u_xai] / T)

The detectors are the differentiable surrogates from `surrogate_study`.
"""
import numpy as np
import torch

from xai_detect import DEVICE
from feat_diff import features_t
from pixel_feat_diff import pixel_features_t
from aim1_attack import cw_margin, upsample
from aim1_adaptive import all_maps, thr_at_fpr

SOFT_T = 0.25          # temperature of the differentiable maximum


def soft_max2(a, b, T=SOFT_T):
    """Differentiable max of two per-sample score vectors."""
    return T * torch.logsumexp(torch.stack([a, b], 0) / T, dim=0)


class Norm:
    """Turns a raw detector score into threshold-relative units."""

    def __init__(self, clean_scores, fpr):
        self.thr = thr_at_fpr(clean_scores, fpr)
        self.sd = float(np.std(clean_scores)) + 1e-8

    def t(self, s):
        return (s - self.thr) / self.sd


def joint_attack(model, det_x, det_p, nx, np_, x, y, eps, k, lam, steps=100,
                 lr=0.08, kappa=0.0, bs=64, warm=0.4, restarts=2, seed=0):
    """Returns adversarial images on CPU. Per sample, keeps the misclassified
    iterate with the lowest soft-max score across all restarts."""
    warm_steps = int(warm * steps)
    out = []

    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEVICE)
        yb = y[i:i + bs].to(DEVICE)
        H, n = xb.shape[-1], len(xb)
        best_z = torch.zeros(n, 3, k, k, device=DEVICE)
        best = torch.full((n,), float("inf"), device=DEVICE)
        seen = torch.zeros(n, dtype=torch.bool, device=DEVICE)

        for r in range(restarts):
            if r == 0:
                z = torch.zeros(n, 3, k, k, device=DEVICE)
            else:
                g = torch.Generator(device="cpu").manual_seed(seed + 977 * r)
                z = (0.5 * torch.randn(n, 3, k, k, generator=g)).to(DEVICE)
            z = z.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)

            for step in range(steps):
                delta = eps * torch.tanh(upsample(z, H))
                x_adv = (xb + delta).clamp(0, 1)
                logits = model(x_adv)
                loss = cw_margin(logits, yb, kappa).sum()
                joint = None

                if step >= warm_steps and lam > 0:
                    tgt = logits.argmax(1).detach()
                    ux = nx.t(det_x(features_t(
                        all_maps(model, x_adv, tgt, create_graph=True))))
                    up = np_.t(det_p(pixel_features_t(x_adv)))
                    joint = soft_max2(up, ux)
                    loss = loss + lam * joint.sum()

                with torch.no_grad():
                    mis = logits.argmax(1) != yb
                    s = joint.detach() if joint is not None \
                        else torch.full((n,), 1e3, device=DEVICE)
                    cand = torch.where(mis, s, torch.full_like(s, float("inf")))
                    better = cand < best
                    best = torch.where(better, cand, best)
                    best_z[better] = z.detach()[better]
                    seen |= mis

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                if (~seen).any():
                    best_z[~seen] = z.detach()[~seen]

        with torch.no_grad():
            delta = eps * torch.tanh(upsample(best_z, H))
            out.append((xb + delta).clamp(0, 1).cpu())

    return torch.cat(out)
