"""Why does the trivial image baseline win? Feature-group ablation.

control_baseline.py showed raw-image statistics reaching AUC 1.0000 on FGSM,
beating the XAI detector. Before concluding anything, we need to know WHICH
image statistic is doing it, because they mean very different things:

  clip : fraction of pixels pinned to exactly 0.0 or 1.0. This is an ARTEFACT of
         generating attacks digitally and clipping to [0,1]. A physical attack,
         or any image that has been through a camera/JPEG pipeline, would not
         carry it. If detection rests on this, the whole benchmark is measuring
         a setup artefact rather than adversarial perturbation.
  tv/lap/fft/mad : genuine high-frequency noise measures. If these carry it,
         the baseline is really detecting the perturbation, and the finding
         against XAI stands on firmer ground.
  chan : per-channel mean/std. Should be near chance; sanity check.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe diagnose_baseline.py
"""
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
from xai_detect import DATA, CKPT, DEVICE


def image_features_grouped(x):
    """Same statistics as control_baseline.image_features, but tagged by group."""
    x = x.to(DEVICE)
    g = x.mean(1)
    H, W = g.shape[-2:]
    parts = []

    parts.append(("chan", x.flatten(2).std(2)))
    parts.append(("chan", x.flatten(2).mean(2)))

    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    parts.append(("tv", tv_h.mean((2, 3))))
    parts.append(("tv", tv_w.mean((2, 3))))
    parts.append(("tv", torch.stack([tv_h.mean((1, 2, 3)), tv_w.mean((1, 2, 3)),
                                     tv_h.std((1, 2, 3)), tv_w.std((1, 2, 3))], 1)))

    lap = (g[:, 1:-1, 2:] + g[:, 1:-1, :-2] + g[:, 2:, 1:-1]
           + g[:, :-2, 1:-1] - 4 * g[:, 1:-1, 1:-1])
    parts.append(("lap", torch.stack([lap.abs().mean((1, 2)), lap.pow(2).mean((1, 2)),
                                      lap.abs().max(1).values.max(1).values], 1)))

    mag = torch.fft.fftshift(torch.fft.fft2(g).abs(), dim=(-2, -1))
    total = mag.sum((1, 2)) + 1e-9
    ch, cw = H // 4, W // 4
    centre = mag[:, H // 2 - ch:H // 2 + ch, W // 2 - cw:W // 2 + cw].sum((1, 2))
    parts.append(("fft", ((total - centre) / total).unsqueeze(1)))

    hp = (g[:, 1:, :] - g[:, :-1, :]).flatten(1)
    med = hp.median(1).values.unsqueeze(1)
    parts.append(("mad", (hp - med).abs().median(1).values.unsqueeze(1)))

    parts.append(("clip", torch.stack([(x <= 1e-6).float().mean((1, 2, 3)),
                                       (x >= 1 - 1e-6).float().mean((1, 2, 3))], 1)))

    groups, cols = [], []
    for name, t in parts:
        t = t if t.dim() == 2 else t.unsqueeze(1)
        groups += [name] * t.shape[1]
        cols.append(t)
    return torch.cat(cols, 1).detach().float().cpu().numpy(), np.array(groups)


def fit_eval(F, lab, grp):
    if F.shape[1] == 0:
        return float("nan")
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                  .split(F, lab, groups=grp))
    det = HistGradientBoostingClassifier(max_iter=400, random_state=0)
    det.fit(F[tr], lab[tr])
    return roc_auc_score(lab[te], det.predict_proba(F[te])[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000)
    ap.add_argument("--eps", type=float, nargs="+", default=[0.005, 0.01, 0.03, 0.10])
    args = ap.parse_args()

    model, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    print(f"classifier: {ckpt.get('backbone')} @ {img_size}px, clean acc {ckpt['test_acc']*100:.2f}%\n")

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
    gen = torch.Generator().manual_seed(0)
    sel = keep[torch.randperm(len(keep), generator=gen)[:args.n]]
    X, Y = X[sel], Y[sel]          # one index set, so X and Y stay aligned
    print(f"using {len(X)} correctly-classified clean images")

    clf = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(),
                            input_shape=(3, img_size, img_size), nb_classes=NUM_CLASSES,
                            clip_values=(0.0, 1.0),
                            device_type="gpu" if DEVICE == "cuda" else "cpu")

    Fi_clean, groups = image_features_grouped(X)
    names = ["ALL", "no_clip", "clip", "tv", "lap", "fft", "mad", "chan"]
    print(f"feature groups: {dict(zip(*np.unique(groups, return_counts=True)))}\n")
    print("     eps   n_adv " + "".join(f"{n:>9}" for n in names))
    print("-" * (16 + 9 * len(names)))

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
        Fi_adv, _ = image_features_grouped(Xadv[idx])

        lab = np.concatenate([np.zeros(len(ix)), np.ones(len(ix))])
        grp = np.concatenate([ix, ix])
        F = np.concatenate([Fi_clean[ix], Fi_adv])

        row = []
        for n in names:
            if n == "ALL":
                sel = np.ones(len(groups), bool)
            elif n == "no_clip":
                sel = groups != "clip"
            else:
                sel = groups == n
            row.append(fit_eval(F[:, sel], lab, grp))
        print(f"{eps:>8.3f} {len(idx):>7} " + "".join(f"{v:>9.4f}" for v in row))

    print("\nIf 'clip' alone is near 1.0 and 'no_clip' drops sharply, the baseline is")
    print("exploiting a digital-attack artefact rather than detecting perturbation.")


if __name__ == "__main__":
    main()
