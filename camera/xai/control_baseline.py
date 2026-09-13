"""Control experiment: is the XAI actually doing the work?

FGSM perturbs every pixel, so an attacked image is measurably noisier than a
clean one. An attribution map computed on a noisy image inherits that noise.
The XAI detector may therefore be an expensive noise meter rather than a probe
of the model's reasoning.

This script trains the SAME detector on three feature sets and compares them:

  IMAGE : plain noise statistics of the raw image (no model, no gradients)
  XAI   : attribution-map features only (as in xai_detect.py)
  BOTH  : concatenation, to see whether XAI adds anything on top of IMAGE

If IMAGE alone matches XAI, the explainability framing is not supported.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe control_baseline.py
"""
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import attribute_all, WORK, DATA, CKPT, DEVICE


def image_features(x):
    """Noise statistics of the raw image. Deliberately a STRONG baseline:
    these are the standard ways to measure pixel-level noise in an image."""
    x = x.to(DEVICE)
    g = x.mean(1)                                   # grayscale (B,H,W)
    H, W = g.shape[-2:]
    f = []

    f.append(x.flatten(2).std(2))                   # per-channel std   (3)
    f.append(x.flatten(2).mean(2))                  # per-channel mean  (3)

    # total variation, whole image and per channel
    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    f.append(tv_h.mean((2, 3)))                     # (3)
    f.append(tv_w.mean((2, 3)))                     # (3)
    f.append(torch.stack([tv_h.mean((1, 2, 3)), tv_w.mean((1, 2, 3)),
                          tv_h.std((1, 2, 3)), tv_w.std((1, 2, 3))], 1))

    # Laplacian energy (classic sharpness / noise measure)
    lap = (g[:, 1:-1, 2:] + g[:, 1:-1, :-2] + g[:, 2:, 1:-1]
           + g[:, :-2, 1:-1] - 4 * g[:, 1:-1, 1:-1])
    f.append(torch.stack([lap.abs().mean((1, 2)), lap.pow(2).mean((1, 2)),
                          lap.abs().max(1).values.max(1).values], 1))

    # high-frequency energy ratio from the 2D FFT
    mag = torch.fft.fftshift(torch.fft.fft2(g).abs(), dim=(-2, -1))
    total = mag.sum((1, 2)) + 1e-9
    ch, cw = H // 4, W // 4
    centre = mag[:, H // 2 - ch:H // 2 + ch, W // 2 - cw:W // 2 + cw].sum((1, 2))
    f.append(((total - centre) / total).unsqueeze(1))

    # robust noise estimate: MAD of the high-pass residual
    hp = (g[:, 1:, :] - g[:, :-1, :]).flatten(1)
    med = hp.median(1).values.unsqueeze(1)
    f.append((hp - med).abs().median(1).values.unsqueeze(1))

    # saturation / clipping, since FGSM pushes pixels against [0,1]
    f.append(torch.stack([(x <= 1e-6).float().mean((1, 2, 3)),
                          (x >= 1 - 1e-6).float().mean((1, 2, 3))], 1))

    return torch.cat(f, 1).detach().float().cpu().numpy()


def fit_eval(F, lab, grp):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                  .split(F, lab, groups=grp))
    det = HistGradientBoostingClassifier(max_iter=400, random_state=0)
    det.fit(F[tr], lab[tr])
    return roc_auc_score(lab[te], det.predict_proba(F[te])[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.10])
    args = ap.parse_args()

    model, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    print(f"classifier: {ckpt.get('backbone')} @ {img_size}px, "
          f"clean acc {ckpt['test_acc']*100:.2f}%\n")

    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)
    X, Y = [], []
    for x, y in DataLoader(ds, batch_size=512, num_workers=8):
        X.append(x); Y.append(y)
    X, Y = torch.cat(X), torch.cat(Y)

    with torch.no_grad():
        pred = torch.cat([model(X[i:i+512].to(DEVICE)).argmax(1).cpu()
                          for i in range(0, len(X), 512)])
    keep = (pred == Y).nonzero().squeeze(1)
    g = torch.Generator().manual_seed(0)
    keep = keep[torch.randperm(len(keep), generator=g)[:args.n]]
    X, Y = X[keep], Y[keep]
    print(f"using {len(X)} correctly-classified clean images")

    clf = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(),
                            input_shape=(3, img_size, img_size), nb_classes=NUM_CLASSES,
                            clip_values=(0.0, 1.0),
                            device_type="gpu" if DEVICE == "cuda" else "cpu")

    with torch.no_grad():
        clean_pred = torch.cat([model(X[i:i+512].to(DEVICE)).argmax(1).cpu()
                                for i in range(0, len(X), 512)])
    print("computing clean features ...")
    Fx_clean, _ = attribute_all(model, X, clean_pred)
    Fi_clean = np.concatenate([image_features(X[i:i+512])
                               for i in range(0, len(X), 512)])
    print(f"  XAI {Fx_clean.shape}   IMAGE {Fi_clean.shape}\n")

    print(f"{'eps':>6} {'n_adv':>7} {'IMAGE':>8} {'XAI':>8} {'BOTH':>8}   verdict")
    print("-" * 62)
    for eps in args.eps:
        Xadv = torch.from_numpy(
            FastGradientMethod(estimator=clf, eps=eps, batch_size=256).generate(x=X.numpy()))
        with torch.no_grad():
            adv_pred = torch.cat([model(Xadv[i:i+512].to(DEVICE)).argmax(1).cpu()
                                  for i in range(0, len(Xadv), 512)])
        idx = (adv_pred != Y).nonzero().squeeze(1)
        if len(idx) < 100:
            continue
        ix = idx.numpy()

        Fx_adv, _ = attribute_all(model, Xadv[idx], adv_pred[idx])
        Fi_adv = np.concatenate([image_features(Xadv[idx][i:i+512])
                                 for i in range(0, len(idx), 512)])

        lab = np.concatenate([np.zeros(len(ix)), np.ones(len(ix))])
        grp = np.concatenate([ix, ix])
        a_img = fit_eval(np.concatenate([Fi_clean[ix], Fi_adv]), lab, grp)
        a_xai = fit_eval(np.concatenate([Fx_clean[ix], Fx_adv]), lab, grp)
        a_both = fit_eval(np.concatenate([
            np.concatenate([Fi_clean[ix], Fx_clean[ix]], 1),
            np.concatenate([Fi_adv, Fx_adv], 1)]), lab, grp)

        gap = a_xai - a_img
        verdict = ("XAI adds nothing" if gap < 0.005 else
                   "XAI helps a little" if gap < 0.03 else "XAI clearly helps")
        print(f"{eps:>6.3f} {len(idx):>7} {a_img:>8.4f} {a_xai:>8.4f} {a_both:>8.4f}   {verdict}")

    print("\nIMAGE = raw-image noise statistics only (no model, no explanations)")
    print("XAI   = attribution-map features only")
    print("BOTH  = concatenated")


if __name__ == "__main__":
    main()
