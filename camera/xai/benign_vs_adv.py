"""THE GATE EXPERIMENT: can we tell an adversarial patch from innocent graffiti?

Everything so far asked "is this image perturbed?", which is trivial. FGSM was
solved by a Laplacian filter (AUC 1.0000) and patches were solved by everything
(AUC 1.0000 for all methods). Neither is a research problem.

A detector that fires on ANY mark on a sign is useless in a vehicle, because
real signs carry graffiti, stickers, rust and dirt. The only useful question is:

    is this sticker an ATTACK, or just vandalism?

This is the first task where the answer is not encoded in the pixels. A benign
sticker and an adversarial sticker are both coloured blobs; the difference is
that only one HIJACKS THE MODEL'S DECISION, which is visible in explanation
space and nowhere else.

Prediction: IMAGE statistics should fail (near chance), XAI should succeed.
If XAI also fails, explanation-based detection has no niche here.

FAIRNESS: benign and adversarial patches are pasted at the SAME size and the
SAME location (same RNG), so the detector cannot cheat on geometry.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe benign_vs_adv.py
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit

from model import load_trained, IMG_SIZE
from xai_detect import attribute_all, DATA, CKPT, DEVICE
from control_baseline import image_features

PATCH_PATH = r"C:\Users\s4990998\xai_work\patch.npy"


# ---------------------------------------------------------------- patches
def make_benign(kind, n, size, pool, gen):
    """Benign occlusions: the kind of thing that lands on a real sign."""
    if kind == "solid":                      # a plain sticker
        c = torch.rand(n, 3, 1, 1, generator=gen)
        return c.expand(n, 3, size, size).clone()
    if kind == "noise":                      # spray-paint speckle
        return torch.rand(n, 3, size, size, generator=gen)
    if kind == "texture":                    # a torn poster / another sign
        idx = torch.randint(len(pool), (n,), generator=gen)
        return Fn.interpolate(pool[idx], size=(size, size), mode="bilinear",
                              align_corners=False)
    if kind == "scribble":                   # graffiti strokes
        t = torch.rand(n, 3, 1, 1, generator=gen).expand(n, 3, size, size).clone()
        for i in range(n):
            for _ in range(int(torch.randint(2, 5, (1,), generator=gen))):
                col = torch.rand(3, generator=gen)
                y = int(torch.randint(0, size, (1,), generator=gen))
                w = int(torch.randint(1, max(2, size // 6), (1,), generator=gen))
                if int(torch.randint(0, 2, (1,), generator=gen)):
                    t[i, :, y:y + w, :] = col.view(3, 1, 1)
                else:
                    t[i, :, :, y:y + w] = col.view(3, 1, 1)
        return t
    raise ValueError(kind)


def paste(imgs, tiles, locs):
    """Paste tiles into imgs at locs. tiles (B,3,s,s), locs (B,2)."""
    out = imgs.clone()
    s = tiles.shape[-1]
    for i, (y, x) in enumerate(locs):
        out[i, :, y:y + s, x:x + s] = tiles[i]
    return out.clamp(0, 1)


def predict(model, x, bs=512):
    with torch.no_grad():
        return torch.cat([model(x[i:i + bs].to(DEVICE)).argmax(1).cpu()
                          for i in range(0, len(x), bs)])


def feats(model, x, pred, bs=512):
    fx, _ = attribute_all(model, x, pred)
    fi = np.concatenate([image_features(x[i:i + bs]) for i in range(0, len(x), bs)])
    return fi, fx


def fit_eval(F, lab, grp):
    if len(np.unique(lab)) < 2 or len(lab) < 60:
        return float("nan")
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
                  .split(F, lab, groups=grp))
    if len(np.unique(lab[te])) < 2:
        return float("nan")
    det = HistGradientBoostingClassifier(max_iter=400, random_state=0)
    det.fit(F[tr], lab[tr])
    return roc_auc_score(lab[te], det.predict_proba(F[te])[:, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--scale", type=float, default=0.35)
    ap.add_argument("--kinds", nargs="+",
                    default=["solid", "noise", "texture", "scribble"])
    args = ap.parse_args()

    model, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    size = int(args.scale * img_size)
    print(f"classifier: {ckpt.get('backbone')} @ {img_size}px, "
          f"clean acc {ckpt['test_acc']*100:.2f}%")
    print(f"patch size: {size}x{size}px ({args.scale:.0%} of image side)\n")

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

    # identical geometry for every condition
    locs = torch.stack([
        torch.randint(0, img_size - size, (len(X),), generator=gen),
        torch.randint(0, img_size - size, (len(X),), generator=gen)], 1)

    adv_tile = Fn.interpolate(torch.from_numpy(np.load(PATCH_PATH)).float().unsqueeze(0),
                              size=(size, size), mode="bilinear", align_corners=False)
    Xadv = paste(X, adv_tile.expand(len(X), -1, -1, -1), locs)
    adv_pred = predict(model, Xadv)
    adv_ok = (adv_pred != Y)
    print(f"adversarial patch success: {adv_ok.float().mean()*100:.1f}% "
          f"({adv_ok.sum()} images)\n")
    if adv_ok.sum() < 150:
        print("too few successful attacks; raise --scale"); return

    ai = adv_ok.nonzero().squeeze(1)
    Fi_adv, Fx_adv = feats(model, Xadv[ai], adv_pred[ai])
    Fi_cln, Fx_cln = feats(model, X[ai], predict(model, X[ai]))

    print(f"{'benign kind':>12} {'flip%':>7} {'n':>6} | {'IMAGE':>8} {'XAI':>8} {'BOTH':>8}  verdict")
    print("-" * 74)
    for kind in args.kinds:
        tiles = make_benign(kind, len(X), size, X, torch.Generator().manual_seed(1))
        Xben = paste(X, tiles, locs)
        ben_pred = predict(model, Xben)
        flip = (ben_pred != Y).float().mean().item()

        # benign side uses the SAME source images as the adversarial side
        Fi_ben, Fx_ben = feats(model, Xben[ai], ben_pred[ai])

        lab = np.concatenate([np.zeros(len(ai)), np.ones(len(ai))])
        grp = np.concatenate([ai.numpy(), ai.numpy()])
        a_i = fit_eval(np.concatenate([Fi_ben, Fi_adv]), lab, grp)
        a_x = fit_eval(np.concatenate([Fx_ben, Fx_adv]), lab, grp)
        a_b = fit_eval(np.concatenate([np.concatenate([Fi_ben, Fx_ben], 1),
                                       np.concatenate([Fi_adv, Fx_adv], 1)]), lab, grp)
        gap = a_x - a_i
        v = ("XAI adds nothing" if gap < 0.005 else
             "XAI helps a little" if gap < 0.03 else "XAI CLEARLY HELPS")
        print(f"{kind:>12} {flip*100:>6.1f}% {len(ai):>6} | "
              f"{a_i:>8.4f} {a_x:>8.4f} {a_b:>8.4f}  {v}")

    # reference point: the easy task we already knew was trivial
    lab = np.concatenate([np.zeros(len(ai)), np.ones(len(ai))])
    grp = np.concatenate([ai.numpy(), ai.numpy()])
    print("-" * 74)
    print(f"{'(clean ref)':>12} {'-':>7} {len(ai):>6} | "
          f"{fit_eval(np.concatenate([Fi_cln, Fi_adv]), lab, grp):>8.4f} "
          f"{fit_eval(np.concatenate([Fx_cln, Fx_adv]), lab, grp):>8.4f} "
          f"{'':>8}  clean vs adversarial")

    print("\nThe benign rows are the real test: same patch size, same location,")
    print("only the CONTENT differs. IMAGE cannot see intent; XAI might.")


if __name__ == "__main__":
    main()
