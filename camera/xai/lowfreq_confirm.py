"""Confirmation run for the low-frequency result, with a STRONGER attack.

The sweep found one cell (k=3, eps=0.03) where XAI beat the pixel baseline
0.9514 vs 0.8374, with a clean monotone mechanism behind it: as the attack gets
smoother, the pixel baseline degrades and XAI does not. Two problems with that
evidence:

  1. only ~180 successful attacks at that cell, so the AUC interval is wide
  2. 9% attack success is too weak to be a real threat

This script fixes both:

  STRONGER ATTACK. Cross-entropy saturates once a sample is misclassified and
  gives poor gradients near the boundary, so we switch to a CW-style margin
  loss (push the true logit below the best other logit), add random restarts,
  more steps, and a cosine LR schedule.

  ERROR BARS. Each configuration is scored over several train/test splits and
  reported as mean +/- std, so a lucky split cannot masquerade as a result.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe lowfreq_confirm.py
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import attribute_all, DATA, CKPT, DEVICE
from control_baseline import image_features

N_SEEDS = 5


def lowfreq_strong(model, x, y, eps, k, steps=300, restarts=2, lr=0.15, bs=256):
    """Smooth untargeted attack using a CW-style margin loss.

    The perturbation is a k x k tensor upsampled to image size, so it is
    low-frequency by construction and carries almost no Laplacian signature.
    """
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEVICE)
        yb = y[i:i + bs].to(DEVICE)
        H = xb.shape[-1]
        best = xb.clone()
        done = torch.zeros(len(xb), dtype=torch.bool, device=DEVICE)

        for r in range(restarts):
            z = (torch.zeros(len(xb), 3, k, k, device=DEVICE) if r == 0
                 else 0.5 * torch.randn(len(xb), 3, k, k, device=DEVICE))
            z.requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
            for _ in range(steps):
                delta = eps * torch.tanh(
                    Fn.interpolate(z, size=(H, H), mode="bilinear", align_corners=False))
                logits = model((xb + delta).clamp(0, 1))
                true_logit = logits.gather(1, yb[:, None]).squeeze(1)
                other = logits.scatter(1, yb[:, None], -1e9).max(1).values
                # minimise (true - other): drives the true class below the best rival
                loss = (true_logit - other).clamp(min=-0.10).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()

            with torch.no_grad():
                delta = eps * torch.tanh(
                    Fn.interpolate(z, size=(H, H), mode="bilinear", align_corners=False))
                xa = (xb + delta).clamp(0, 1)
                ok = model(xa).argmax(1) != yb
                fresh = ok & ~done            # keep the first restart that worked
                best[fresh] = xa[fresh]
                done |= ok
        out.append(best.cpu())
    return torch.cat(out)


def predict(model, x, bs=512):
    with torch.no_grad():
        return torch.cat([model(x[i:i + bs].to(DEVICE)).argmax(1).cpu()
                          for i in range(0, len(x), bs)])


def auc_spread(F, lab, grp):
    """AUC over several splits -> (mean, std)."""
    vals = []
    for s in range(N_SEEDS):
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s)
                      .split(F, lab, groups=grp))
        det = HistGradientBoostingClassifier(max_iter=400, random_state=s)
        det.fit(F[tr], lab[tr])
        vals.append(roc_auc_score(lab[te], det.predict_proba(F[te])[:, 1]))
    return float(np.mean(vals)), float(np.std(vals))


def roughness(xa, xc):
    d = (xa - xc).mean(1)
    lap = (d[:, 1:-1, 2:] + d[:, 1:-1, :-2] + d[:, 2:, 1:-1]
           + d[:, :-2, 1:-1] - 4 * d[:, 1:-1, 1:-1])
    return lap.abs().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    args = ap.parse_args()

    model, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    print(f"classifier: {ckpt.get('backbone')} @ {img_size}px, "
          f"clean acc {ckpt['test_acc']*100:.2f}%   ({N_SEEDS} splits per cell)\n")

    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)
    X, Y = [], []
    for x, y in DataLoader(ds, batch_size=512, num_workers=8):
        X.append(x); Y.append(y)
    X, Y = torch.cat(X), torch.cat(Y)
    keep = (predict(model, X) == Y).nonzero().squeeze(1)
    gen = torch.Generator().manual_seed(0)
    keep = keep[torch.randperm(len(keep), generator=gen)[:args.n]]
    X, Y = X[keep], Y[keep]
    print(f"{len(X)} correctly-classified clean images")

    clf = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(),
                            input_shape=(3, img_size, img_size), nb_classes=NUM_CLASSES,
                            clip_values=(0.0, 1.0),
                            device_type="gpu" if DEVICE == "cuda" else "cpu")

    print("clean features ...")
    Fx_clean, _ = attribute_all(model, X, predict(model, X))
    Fi_clean = np.concatenate([image_features(X[i:i+512]) for i in range(0, len(X), 512)])

    configs = [
        ("FGSM       eps=0.03", lambda: torch.from_numpy(
            FastGradientMethod(estimator=clf, eps=0.03, batch_size=256).generate(x=X.numpy()))),
        ("PGD full   eps=0.03", lambda: torch.from_numpy(
            ProjectedGradientDescent(estimator=clf, eps=0.03, eps_step=0.005, max_iter=40,
                                     batch_size=256, verbose=False).generate(x=X.numpy()))),
        ("LowFreq k=6  e=0.03", lambda: lowfreq_strong(model, X, Y, 0.03, 6)),
        ("LowFreq k=4  e=0.03", lambda: lowfreq_strong(model, X, Y, 0.03, 4)),
        ("LowFreq k=3  e=0.03", lambda: lowfreq_strong(model, X, Y, 0.03, 3)),
        ("LowFreq k=2  e=0.03", lambda: lowfreq_strong(model, X, Y, 0.03, 2)),
        ("LowFreq k=3  e=0.06", lambda: lowfreq_strong(model, X, Y, 0.06, 3)),
        ("LowFreq k=2  e=0.06", lambda: lowfreq_strong(model, X, Y, 0.06, 2)),
        ("LowFreq k=3  e=0.10", lambda: lowfreq_strong(model, X, Y, 0.10, 3)),
    ]

    print(f"\n{'attack':>20} {'succ':>7} {'n':>6} {'rough':>9} | "
          f"{'IMAGE':>15} {'XAI':>15}   gap")
    print("-" * 92)
    for name, fn in configs:
        Xadv = fn().float()
        adv_pred = predict(model, Xadv)
        idx = (adv_pred != Y).nonzero().squeeze(1)
        succ, rgh = len(idx) / len(X), roughness(Xadv, X)
        if len(idx) < 150:
            print(f"{name:>20} {succ*100:>6.1f}% {len(idx):>6} {rgh:>9.5f} |  too few")
            continue
        ix = idx.numpy()
        Fx_adv, _ = attribute_all(model, Xadv[idx], adv_pred[idx])
        Fi_adv = np.concatenate([image_features(Xadv[idx][i:i+512])
                                 for i in range(0, len(idx), 512)])
        lab = np.concatenate([np.zeros(len(ix)), np.ones(len(ix))])
        grp = np.concatenate([ix, ix])
        mi, si = auc_spread(np.concatenate([Fi_clean[ix], Fi_adv]), lab, grp)
        mx, sx = auc_spread(np.concatenate([Fx_clean[ix], Fx_adv]), lab, grp)
        gap = mx - mi
        tag = "XAI WINS" if gap > 3 * max(si, sx, 1e-6) else ("XAI ahead" if gap > 0 else "")
        print(f"{name:>20} {succ*100:>6.1f}% {len(ix):>6} {rgh:>9.5f} | "
              f"{mi:>7.4f}+/-{si:<6.4f} {mx:>7.4f}+/-{sx:<6.4f} {gap:+.4f} {tag}")

    print("\nGap counts only if it exceeds ~3x the split-to-split noise.")


if __name__ == "__main__":
    main()
