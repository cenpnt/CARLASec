"""Low-frequency (smooth) attacks: the setting where pixel statistics should go blind.

WHY THIS ATTACK. Our control experiments showed a Laplacian/total-variation
detector reaching AUC 1.0000 on FGSM and on patches, beating XAI every time.
That detector works by measuring HIGH-FREQUENCY ROUGHNESS. Adversarial
perturbations of naturally-trained models concentrate in high frequency
(Yin et al. 2019), which is exactly what it looks for.

So we constrain the attack to be SMOOTH. The perturbation is parameterised at
low resolution (k x k) and bilinearly upsampled to full size, so it has almost
no high-frequency content by construction. This mirrors the low-frequency and
perceptually-constrained attacks built in the literature specifically to evade
detection-based defences.

Prediction: as k decreases the perturbation gets smoother, the IMAGE baseline
should collapse toward chance, and XAI may survive because the model's decision
is still being hijacked regardless of how smooth the cause is.

HONESTY NOTE: this is a sweep. Every cell is reported, not just favourable ones.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe lowfreq_attack.py
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


def lowfreq_pgd(model, x, y, eps, k, steps=150, lr=0.08, bs=256):
    """Untargeted attack whose perturbation is smooth by construction.

    The perturbation is a k x k tensor upsampled to the image size, so its
    spectrum is concentrated at low frequencies. tanh keeps it inside the
    L_inf ball of radius eps.
    """
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEVICE)
        yb = y[i:i + bs].to(DEVICE)
        H = xb.shape[-1]
        z = torch.zeros(len(xb), 3, k, k, device=DEVICE, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        for _ in range(steps):
            delta = eps * torch.tanh(
                Fn.interpolate(z, size=(H, H), mode="bilinear", align_corners=False))
            loss = -nn.functional.cross_entropy(model((xb + delta).clamp(0, 1)), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            delta = eps * torch.tanh(
                Fn.interpolate(z, size=(H, H), mode="bilinear", align_corners=False))
            out.append((xb + delta).clamp(0, 1).cpu())
    return torch.cat(out)


def predict(model, x, bs=512):
    with torch.no_grad():
        return torch.cat([model(x[i:i + bs].to(DEVICE)).argmax(1).cpu()
                          for i in range(0, len(x), bs)])


def fit_eval(F, lab, grp):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                  .split(F, lab, groups=grp))
    det = HistGradientBoostingClassifier(max_iter=400, random_state=0)
    det.fit(F[tr], lab[tr])
    return roc_auc_score(lab[te], det.predict_proba(F[te])[:, 1])


def roughness(xa, xc):
    """Mean |Laplacian| of the perturbation, i.e. how rough the attack is."""
    d = (xa - xc).mean(1)
    lap = (d[:, 1:-1, 2:] + d[:, 1:-1, :-2] + d[:, 2:, 1:-1]
           + d[:, :-2, 1:-1] - 4 * d[:, 1:-1, 1:-1])
    return lap.abs().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
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
        ("FGSM        eps=0.03", lambda: torch.from_numpy(
            FastGradientMethod(estimator=clf, eps=0.03, batch_size=256).generate(x=X.numpy()))),
        ("PGD  full   eps=0.03", lambda: torch.from_numpy(
            ProjectedGradientDescent(estimator=clf, eps=0.03, eps_step=0.005,
                                     max_iter=40, batch_size=256, verbose=False
                                     ).generate(x=X.numpy()))),
        ("LowFreq k=24 eps=0.03", lambda: lowfreq_pgd(model, X, Y, 0.03, 24)),
        ("LowFreq k=12 eps=0.03", lambda: lowfreq_pgd(model, X, Y, 0.03, 12)),
        ("LowFreq k=6  eps=0.03", lambda: lowfreq_pgd(model, X, Y, 0.03, 6)),
        ("LowFreq k=3  eps=0.03", lambda: lowfreq_pgd(model, X, Y, 0.03, 3)),
        ("LowFreq k=6  eps=0.06", lambda: lowfreq_pgd(model, X, Y, 0.06, 6)),
        ("LowFreq k=3  eps=0.06", lambda: lowfreq_pgd(model, X, Y, 0.06, 3)),
        ("LowFreq k=2  eps=0.08", lambda: lowfreq_pgd(model, X, Y, 0.08, 2)),
    ]

    print(f"\n{'attack':>22} {'succ':>7} {'rough':>9} | {'IMAGE':>8} {'XAI':>8} {'BOTH':>8}  verdict")
    print("-" * 84)
    for name, fn in configs:
        Xadv = fn().float()
        adv_pred = predict(model, Xadv)
        idx = (adv_pred != Y).nonzero().squeeze(1)
        succ = len(idx) / len(X)
        rgh = roughness(Xadv, X)
        if len(idx) < 100:
            print(f"{name:>22} {succ*100:>6.1f}% {rgh:>9.5f} |  too few successful attacks")
            continue
        ix = idx.numpy()
        Fx_adv, _ = attribute_all(model, Xadv[idx], adv_pred[idx])
        Fi_adv = np.concatenate([image_features(Xadv[idx][i:i+512])
                                 for i in range(0, len(idx), 512)])
        lab = np.concatenate([np.zeros(len(ix)), np.ones(len(ix))])
        grp = np.concatenate([ix, ix])
        a_i = fit_eval(np.concatenate([Fi_clean[ix], Fi_adv]), lab, grp)
        a_x = fit_eval(np.concatenate([Fx_clean[ix], Fx_adv]), lab, grp)
        a_b = fit_eval(np.concatenate([
            np.concatenate([Fi_clean[ix], Fx_clean[ix]], 1),
            np.concatenate([Fi_adv, Fx_adv], 1)]), lab, grp)
        gap = a_x - a_i
        v = ("XAI WINS" if gap > 0.03 else
             "XAI ahead" if gap > 0.005 else
             "tie" if gap > -0.005 else "IMAGE wins")
        print(f"{name:>22} {succ*100:>6.1f}% {rgh:>9.5f} | "
              f"{a_i:>8.4f} {a_x:>8.4f} {a_b:>8.4f}  {v}")

    print("\n'rough' = mean |Laplacian| of the perturbation. Lower means smoother,")
    print("which is what the IMAGE baseline relies on seeing.")


if __name__ == "__main__":
    main()
