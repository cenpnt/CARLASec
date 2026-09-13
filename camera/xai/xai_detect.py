"""XAI-only detection of FGSM adversarial examples on a SOTA GTSRB classifier.

Attacks come from ART (reference implementations). Every detector feature is
derived from attribution maps produced by input-level explanation methods:
Integrated Gradients, Input x Gradient and Saliency. No softmax confidence, no
prediction-stability / feature-squeezing cue, and no Grad-CAM. If the detector
works, the signal is attributable to explainability alone.

Two feature families, matching the proposal:
  1. per-method attribution statistics (mass, dispersion, entropy, total
     variation, spatial concentration)
  2. cross-method disagreement (correlation, cosine, top-k overlap between the
     attributions that different methods give for the SAME decision)

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe xai_detect.py
"""
import os
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from captum.attr import IntegratedGradients, InputXGradient, Saliency
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit

from model import load_trained, IMG_SIZE, NUM_CLASSES

WORK = r"C:\Users\s4990998\xai_work"
DATA = os.path.join(WORK, "data")
CKPT = os.path.join(WORK, "gtsrb_sota.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# attribution -> feature vector
# --------------------------------------------------------------------------
def map_features(attr):
    """Summary statistics of one batch of attribution maps.

    attr: (B,3,H,W) tensor. Returns (B,F) numpy array.
    All features are shape/scale descriptors of the attribution map itself.
    """
    a = attr.abs().sum(1)                      # (B,H,W) magnitude per pixel
    B, H, W = a.shape
    flat = a.reshape(B, -1)
    total = flat.sum(1) + 1e-12

    p = flat / total.unsqueeze(1)              # normalised to a distribution
    entropy = -(p * (p + 1e-12).log()).sum(1)

    srt = p.sort(dim=1, descending=True).values
    top1 = srt[:, : max(1, int(0.01 * H * W))].sum(1)   # mass in top 1% pixels
    top5 = srt[:, : max(1, int(0.05 * H * W))].sum(1)
    top20 = srt[:, : max(1, int(0.20 * H * W))].sum(1)

    # total variation: how high-frequency the attribution map is. FGSM adds a
    # pixel-level sign pattern, which should roughen the attribution.
    tv_h = (a[:, 1:, :] - a[:, :-1, :]).abs().mean((1, 2))
    tv_w = (a[:, :, 1:] - a[:, :, :-1]).abs().mean((1, 2))
    mean_a = flat.mean(1) + 1e-12
    tv = (tv_h + tv_w) / mean_a                # scale-invariant roughness

    # spatial concentration: signs are centred, so attribution drifting to the
    # border is informative.
    c0, c1 = H // 4, 3 * H // 4
    centre = a[:, c0:c1, c0:c1].sum((1, 2)) / total

    # signed statistics (attribution polarity balance)
    s = attr.sum(1).reshape(B, -1)
    pos_frac = (s > 0).float().mean(1)

    feats = torch.stack([
        total.log(), flat.mean(1), flat.std(1), flat.max(1).values,
        entropy, top1, top5, top20, tv, tv_h / mean_a, tv_w / mean_a,
        centre, pos_frac, s.mean(1), s.std(1),
    ], dim=1)
    return feats.detach().float().cpu().numpy()


def disagreement_features(maps):
    """Cross-method disagreement between attribution maps of the same decision.

    maps: dict name -> (B,3,H,W). Returns (B,F) numpy array.
    """
    names = sorted(maps)
    mags = {n: maps[n].abs().sum(1).flatten(1) for n in names}   # (B,HW)
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            u, v = mags[names[i]], mags[names[j]]
            un = (u - u.mean(1, keepdim=True)) / (u.std(1, keepdim=True) + 1e-12)
            vn = (v - v.mean(1, keepdim=True)) / (v.std(1, keepdim=True) + 1e-12)
            pearson = (un * vn).mean(1)
            cos = torch.nn.functional.cosine_similarity(u, v, dim=1)

            # overlap of the most-important 10% of pixels
            k = max(1, int(0.10 * u.shape[1]))
            tu = u.topk(k, dim=1).indices
            tv_ = v.topk(k, dim=1).indices
            mu = torch.zeros_like(u, dtype=torch.bool).scatter_(1, tu, True)
            mv = torch.zeros_like(v, dtype=torch.bool).scatter_(1, tv_, True)
            iou = (mu & mv).sum(1).float() / (mu | mv).sum(1).float().clamp(min=1)

            out += [pearson, cos, iou]
    return torch.stack(out, dim=1).detach().float().cpu().numpy()


# --------------------------------------------------------------------------
def attribute_all(model, x, target, batch=64):
    """Compute all attribution methods for x w.r.t. the PREDICTED class.

    The true label is unavailable at runtime, so attributions must be taken
    against the model's own prediction.
    """
    ig = IntegratedGradients(model)
    ixg = InputXGradient(model)
    sal = Saliency(model)

    feats, maps_acc = [], {"ig": [], "ixg": [], "sal": []}
    for i in range(0, len(x), batch):
        xb = x[i:i + batch].to(DEVICE).requires_grad_(True)
        tb = target[i:i + batch].to(DEVICE)
        m = {
            "ig": ig.attribute(xb, target=tb, n_steps=32, internal_batch_size=batch),
            "ixg": ixg.attribute(xb, target=tb),
            "sal": sal.attribute(xb, target=tb, abs=False),
        }
        per_method = [map_features(m[n]) for n in sorted(m)]
        feats.append(np.concatenate(per_method + [disagreement_features(m)], axis=1))
        for n in m:
            maps_acc[n].append(m[n].detach().cpu())
    return np.concatenate(feats, 0), {n: torch.cat(v) for n, v in maps_acc.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000, help="number of source images")
    ap.add_argument("--eps", type=float, nargs="+",
                    default=[0.005, 0.01, 0.02, 0.03, 0.05, 0.10])
    args = ap.parse_args()

    model, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    print(f"loaded classifier: {ckpt.get('backbone')} @ {img_size}px, "
          f"clean test acc {ckpt['test_acc']*100:.2f}%  device {DEVICE}")

    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)
    ld = DataLoader(ds, batch_size=512, shuffle=False, num_workers=8)

    xs, ys = [], []
    for x, y in ld:
        xs.append(x); ys.append(y)
    X = torch.cat(xs); Y = torch.cat(ys)

    # Keep only images the classifier gets right when clean. Detecting an attack
    # on an image the model already misreads is not a meaningful test.
    with torch.no_grad():
        preds = []
        for i in range(0, len(X), 512):
            preds.append(model(X[i:i + 512].to(DEVICE)).argmax(1).cpu())
        preds = torch.cat(preds)
    keep = (preds == Y).nonzero().squeeze(1)
    g = torch.Generator().manual_seed(0)
    keep = keep[torch.randperm(len(keep), generator=g)[:args.n]]
    X, Y = X[keep], Y[keep]
    print(f"using {len(X)} correctly-classified clean images\n")

    classifier = PyTorchClassifier(
        model=model, loss=nn.CrossEntropyLoss(),
        input_shape=(3, img_size, img_size), nb_classes=NUM_CLASSES,
        clip_values=(0.0, 1.0), device_type="gpu" if DEVICE == "cuda" else "cpu",
    )

    print("computing attributions for clean images ...")
    t0 = time.time()
    with torch.no_grad():
        clean_pred = torch.cat([model(X[i:i+512].to(DEVICE)).argmax(1).cpu()
                                for i in range(0, len(X), 512)])
    F_clean, _ = attribute_all(model, X, clean_pred)
    print(f"  clean features {F_clean.shape}  ({time.time()-t0:.1f}s)\n")

    print(f"{'eps':>6} {'atk succ':>9} {'n_adv':>7} {'AUC':>8} {'bal acc':>8}")
    print("-" * 44)
    results = []
    pool_F, pool_lab, pool_grp = [], [], []
    for eps in args.eps:
        atk = FastGradientMethod(estimator=classifier, eps=eps, batch_size=256)
        Xadv = torch.from_numpy(atk.generate(x=X.numpy()))

        with torch.no_grad():
            adv_pred = torch.cat([model(Xadv[i:i+512].to(DEVICE)).argmax(1).cpu()
                                  for i in range(0, len(Xadv), 512)])
        flipped = (adv_pred != Y)
        succ = flipped.float().mean().item()
        idx = flipped.nonzero().squeeze(1)
        if len(idx) < 100:
            print(f"{eps:>6.3f} {succ*100:>8.1f}% {len(idx):>7}   too few successful attacks")
            continue

        F_adv, _ = attribute_all(model, Xadv[idx], adv_pred[idx])

        # MATCHED PAIRS. The clean side is restricted to exactly the same source
        # images that the attack succeeded on. Comparing all 3000 clean images
        # against only the successfully-attacked subset would let the detector
        # learn "this image is intrinsically hard to classify" instead of "this
        # image has been attacked", and would also leave the classes imbalanced
        # enough to make plain accuracy meaningless.
        ix = idx.numpy()
        Fx = np.concatenate([F_clean[ix], F_adv])
        lab = np.concatenate([np.zeros(len(ix)), np.ones(len(F_adv))])
        grp = np.concatenate([ix, ix])          # a pair never straddles the split

        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                      .split(Fx, lab, groups=grp))
        det = HistGradientBoostingClassifier(max_iter=400, random_state=0)
        det.fit(Fx[tr], lab[tr])
        score = det.predict_proba(Fx[te])[:, 1]
        auc = roc_auc_score(lab[te], score)
        acc = balanced_accuracy_score(lab[te], score > 0.5)
        results.append((eps, succ, len(idx), auc, acc))
        print(f"{eps:>6.3f} {succ*100:>8.1f}% {len(idx):>7} {auc:>8.4f} {acc:>8.4f}")

        pool_F.append(Fx); pool_lab.append(lab); pool_grp.append(grp)

    # Deployment-realistic test: one detector, epsilon unknown at inference.
    Fx = np.concatenate(pool_F); lab = np.concatenate(pool_lab)
    grp = np.concatenate(pool_grp)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                  .split(Fx, lab, groups=grp))
    det = HistGradientBoostingClassifier(max_iter=400, random_state=0)
    det.fit(Fx[tr], lab[tr])
    score = det.predict_proba(Fx[te])[:, 1]
    print("-" * 44)
    print(f"{'pooled':>6} {'(eps unknown)':>19} {roc_auc_score(lab[te], score):>8.4f} "
          f"{balanced_accuracy_score(lab[te], score > 0.5):>8.4f}")

    print("\nDetector: XAI attributions only (IG + InputXGradient + Saliency),")
    print("per-method map statistics plus cross-method disagreement.")
    print("Matched clean/adversarial pairs; balanced accuracy; grouped split.")
    np.save(os.path.join(WORK, "xai_detect_results.npy"), np.array(results))


if __name__ == "__main__":
    main()
