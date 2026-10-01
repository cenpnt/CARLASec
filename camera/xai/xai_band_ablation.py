"""Is the XAI detector reading high-frequency aliasing in the attribution maps?

Low-pass filters the maps at decreasing cutoffs (after FORGrad, Muzellec et al.
ICML 2024) and reports detector AUC with all 54 features, without the total
variation features, with them alone, and with the nine agreement features alone.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u xai_band_ablation.py
"""
import argparse

import numpy as np
import torch
from captum.attr import IntegratedGradients, InputXGradient, Saliency
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from model import load_trained, IMG_SIZE
from xai_detect import map_features, disagreement_features, CKPT, DEVICE
from aim1_attack import attack as smooth_attack, predict, load_data

# Per-method feature layout in xai_detect.map_features (15 per method):
#   0 log total, 1 mean, 2 std, 3 max, 4 entropy, 5 top1%, 6 top5%, 7 top20%,
#   8 tv, 9 tv_h, 10 tv_w, 11 centre, 12 pos_frac, 13 signed mean, 14 signed sd
TV_IDX_PER_METHOD = [8, 9, 10]
N_PER_METHOD = 15
N_METHODS = 3
TV_IDX = [m * N_PER_METHOD + i
          for m in range(N_METHODS) for i in TV_IDX_PER_METHOD]
DISAGREE_IDX = list(range(N_METHODS * N_PER_METHOD,
                          N_METHODS * N_PER_METHOD + 9))
CUTOFFS = [1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.08]


def lowpass(attr, frac):
    """Radial FFT low-pass on a signed attribution tensor (B,C,H,W), keeping
    `frac` of the Nyquist radius. 1.0 is a no-op."""
    if frac >= 1.0:
        return attr
    H, W = attr.shape[-2:]
    F = torch.fft.fftshift(torch.fft.fft2(attr.float()), dim=(-2, -1))
    fy = torch.fft.fftshift(torch.fft.fftfreq(H, device=attr.device))
    fx = torch.fft.fftshift(torch.fft.fftfreq(W, device=attr.device))
    R = (fy[:, None] ** 2 + fx[None, :] ** 2).sqrt()
    mask = (R <= 0.5 * frac).to(F.dtype)
    return torch.fft.ifft2(torch.fft.ifftshift(F * mask, dim=(-2, -1))).real


def features_at_cutoff(model, x, target, frac, bs=64):
    """The detector's 54 features, computed on band-limited attribution maps."""
    ig, ixg, sal = (IntegratedGradients(model), InputXGradient(model),
                    Saliency(model))
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEVICE).requires_grad_(True)
        tb = target[i:i + bs].to(DEVICE)
        m = {
            "ig": ig.attribute(xb, target=tb, n_steps=32,
                               internal_batch_size=bs),
            "ixg": ixg.attribute(xb, target=tb),
            "sal": sal.attribute(xb, target=tb, abs=False),
        }
        m = {k: lowpass(v.detach(), frac) for k, v in m.items()}
        per = [map_features(m[n]) for n in sorted(m)]
        out.append(np.concatenate(per + [disagreement_features(m)], axis=1))
    return np.concatenate(out, 0)


def auc(F, lab, grp, n_splits=5):
    a = []
    for s in range(n_splits):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3,
                                        random_state=s).split(F, lab,
                                                              groups=grp))
        d = HistGradientBoostingClassifier(max_iter=400, random_state=s)
        d.fit(F[tr], lab[tr])
        a.append(roc_auc_score(lab[te], d.predict_proba(F[te])[:, 1]))
    return float(np.mean(a)), float(np.std(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=150)
    args = ap.parse_args()

    model, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    X, Y = load_data(model, img_size, args.n)
    print(f"classifier: {ckpt.get('backbone')} @ {img_size}px, "
          f"clean acc {ckpt['test_acc'] * 100:.2f}%")
    print(f"{len(X)} images, eps={args.eps}\n")

    attacks = {
        "unconstrained k=96": smooth_attack(model, X, Y, eps=args.eps, k=96,
                                            lam=0.0, steps=args.steps).float(),
        "smooth k=6": smooth_attack(model, X, Y, eps=args.eps, k=6,
                                    lam=0.0, steps=args.steps).float(),
    }

    for name, Xadv in attacks.items():
        pred = predict(model, Xadv)
        idx = (pred != Y).nonzero().squeeze(1)
        print("=" * 70)
        print(f"{name}   ({len(idx)} successful of {len(X)})")
        print("=" * 70)
        ix = idx.numpy()
        lab = np.concatenate([np.zeros(len(ix)), np.ones(len(ix))])
        grp = np.concatenate([ix, ix])

        print(f"{'cutoff':>8} {'kept band':>11} {'AUC all 54':>18} "
              f"{'no TV (45)':>18} {'TV only (9)':>18}")
        print("-" * 70)
        for frac in CUTOFFS:
            Fc = features_at_cutoff(model, X[idx], Y[idx], frac)
            Fa = features_at_cutoff(model, Xadv[idx], pred[idx], frac)
            F = np.concatenate([Fc, Fa])
            keep_no_tv = [i for i in range(F.shape[1]) if i not in TV_IDX]
            a_all = auc(F, lab, grp)
            a_notv = auc(F[:, keep_no_tv], lab, grp)
            a_tv = auc(F[:, TV_IDX], lab, grp)
            band = "all" if frac >= 1.0 else f"{frac:.2f} x Nyq"
            print(f"{frac:>8.2f} {band:>11} "
                  f"{a_all[0]:>10.4f}+-{a_all[1]:<6.4f} "
                  f"{a_notv[0]:>10.4f}+-{a_notv[1]:<6.4f} "
                  f"{a_tv[0]:>10.4f}+-{a_tv[1]:<6.4f}", flush=True)

        # agreement features alone, unfiltered
        Fc = features_at_cutoff(model, X[idx], Y[idx], 1.0)
        Fa = features_at_cutoff(model, Xadv[idx], pred[idx], 1.0)
        F = np.concatenate([Fc, Fa])
        a_dis = auc(F[:, DISAGREE_IDX], lab, grp)
        print(f"{'':>8} {'disagree(9)':>11} "
              f"{a_dis[0]:>10.4f}+-{a_dis[1]:<6.4f}")
        print()



if __name__ == "__main__":
    main()
