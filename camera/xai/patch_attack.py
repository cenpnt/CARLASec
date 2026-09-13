"""Adversarial patch attack: the case where XAI should beat image statistics.

Motivation from the FGSM control experiment and the literature:

  FGSM spreads a small perturbation over EVERY pixel. Adversarial perturbations
  of naturally-trained models concentrate in high frequency (Yin et al. 2019),
  so global noise statistics detect them trivially, and treating the problem as
  steganalysis works well (Liu et al. 2018). Our control confirmed this: a
  Laplacian energy measure reached AUC 1.0000 and beat the XAI detector.

  A PATCH is different. It is localised and structurally natural, so global
  average statistics are diluted by the unperturbed remainder of the image.
  Saliency-based defences (SentiNet, Chou et al.) exploit exactly this: the
  patch dominates the model's attention, which is an explanation-space signal.

Hypothesis: on patch attacks the IMAGE baseline should degrade while the XAI
features should hold up or improve. That is the regime where explainability
earns its place.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe patch_attack.py
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import AdversarialPatchPyTorch
from art.utils import to_categorical

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import attribute_all, DATA, CKPT, DEVICE
from control_baseline import image_features


def fit_eval(F, lab, grp):
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                  .split(F, lab, groups=grp))
    det = HistGradientBoostingClassifier(max_iter=400, random_state=0)
    det.fit(F[tr], lab[tr])
    return roc_auc_score(lab[te], det.predict_proba(F[te])[:, 1])


def batched_image_features(x, bs=512):
    return np.concatenate([image_features(x[i:i + bs]) for i in range(0, len(x), bs)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=1500, help="images used to learn the patch")
    ap.add_argument("--n-eval", type=int, default=3000, help="held-out images to attack")
    ap.add_argument("--target", type=int, default=1, help="target class for the patch")
    ap.add_argument("--max-iter", type=int, default=250)
    ap.add_argument("--scale", type=float, nargs="+", default=[0.25, 0.35, 0.45])
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
    gen = torch.Generator().manual_seed(0)
    keep = keep[torch.randperm(len(keep), generator=gen)]

    # Disjoint sets: the patch is LEARNED on one set and EVALUATED on another,
    # so the detector is never scored on images the patch was optimised for.
    tr_idx = keep[:args.n_train]
    ev_idx = keep[args.n_train:args.n_train + args.n_eval]
    Xtr = X[tr_idx]
    Xev, Yev = X[ev_idx], Y[ev_idx]
    print(f"patch training images: {len(Xtr)}   held-out eval images: {len(Xev)}")

    clf = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(),
                            input_shape=(3, img_size, img_size), nb_classes=NUM_CLASSES,
                            clip_values=(0.0, 1.0),
                            device_type="gpu" if DEVICE == "cuda" else "cpu")

    print(f"learning a universal patch targeting class {args.target} ...")
    attack = AdversarialPatchPyTorch(
        estimator=clf,
        patch_shape=(3, img_size, img_size),
        patch_type="square",
        rotation_max=10.0,
        scale_min=0.20, scale_max=0.45,
        learning_rate=0.5,
        max_iter=args.max_iter,
        batch_size=64,
        targeted=True,
        verbose=False,
    )
    y_target = to_categorical(np.full(len(Xtr), args.target), nb_classes=NUM_CLASSES)
    patch, mask = attack.generate(x=Xtr.numpy(), y=y_target)
    np.save(r"C:\Users\s4990998\xai_work\patch.npy", patch)
    print("patch learned.\n")

    print("computing clean features on held-out set ...")
    with torch.no_grad():
        clean_pred = torch.cat([model(Xev[i:i+512].to(DEVICE)).argmax(1).cpu()
                                for i in range(0, len(Xev), 512)])
    Fx_clean, _ = attribute_all(model, Xev, clean_pred)
    Fi_clean = batched_image_features(Xev)
    print(f"  XAI {Fx_clean.shape}   IMAGE {Fi_clean.shape}\n")

    print(f"{'scale':>6} {'succ':>7} {'n_adv':>7} {'IMAGE':>8} {'XAI':>8} {'BOTH':>8}   verdict")
    print("-" * 66)
    for scale in args.scale:
        Xadv = torch.from_numpy(
            attack.apply_patch(Xev.numpy(), scale=scale)).float().clamp(0, 1)
        with torch.no_grad():
            adv_pred = torch.cat([model(Xadv[i:i+512].to(DEVICE)).argmax(1).cpu()
                                  for i in range(0, len(Xadv), 512)])
        idx = (adv_pred != Yev).nonzero().squeeze(1)
        succ = len(idx) / len(Xev)
        if len(idx) < 100:
            print(f"{scale:>6.2f} {succ*100:>6.1f}% {len(idx):>7}   too few successful attacks")
            continue
        ix = idx.numpy()

        Fx_adv, _ = attribute_all(model, Xadv[idx], adv_pred[idx])
        Fi_adv = batched_image_features(Xadv[idx])

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
        print(f"{scale:>6.2f} {succ*100:>6.1f}% {len(idx):>7} {a_img:>8.4f} "
              f"{a_xai:>8.4f} {a_both:>8.4f}   {verdict}")

    print("\nCompare against FGSM, where IMAGE reached 1.0000 and beat XAI everywhere.")


if __name__ == "__main__":
    main()
