"""Why did the adaptive attack fail to evade? Locate the break in the chain.

The adaptive attack cost the attacker 20 points of success and moved the XAI
detector's AUC by nothing (0.9961 -> 0.9964). Tramer's rule of thumb says a
failed adaptive attack usually means a weak attack, not a strong defence. There
are three places this can break, and they need different fixes:

  A. the evasion loss does not even reduce what it targets
     -> the attack is broken (bug, or lambda/steps too small)
  B. it reduces the SURROGATE features (23-dim, Saliency + InputxGradient) but
     not the DETECTOR features (54-dim, adds IG and 3-way disagreement)
     -> surrogate gap; fix by putting IG in the evasion loss
  C. it reduces both, yet the detector still separates them
     -> the detector is genuinely using something else; the defence is real

This measures all three. Distances are normalised per feature by the clean
standard deviation, so they are comparable across the two feature sets.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe diag_adaptive.py
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from sklearn.ensemble import HistGradientBoostingClassifier

from model import load_trained, IMG_SIZE
from xai_detect import attribute_all, DATA, CKPT, DEVICE
from adaptive_attack import adaptive_lowfreq, diff_attr_features, predict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--lams", type=float, nargs="+", default=[0.0, 10.0, 100.0])
    args = ap.parse_args()

    model, ckpt = load_trained(CKPT, DEVICE)
    s = ckpt.get("img_size", IMG_SIZE)
    tf = transforms.Compose([transforms.Resize((s, s)), transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)
    X, Y = [], []
    for a, b in DataLoader(ds, batch_size=512, num_workers=8):
        X.append(a); Y.append(b)
    X, Y = torch.cat(X), torch.cat(Y)
    keep = (predict(model, X) == Y).nonzero().squeeze(1)
    g = torch.Generator().manual_seed(0)
    sel = keep[torch.randperm(len(keep), generator=g)[:args.n]]
    X, Y = X[sel], Y[sel]          # one index set, so X and Y stay aligned
    print(f"{len(X)} images, k={args.k}, eps={args.eps}, {args.steps} steps\n", flush=True)

    # clean references
    xc = X[:256].to(DEVICE).requires_grad_(True)
    f0, _ = diff_attr_features(model, xc, create_graph=False)
    scale_sur = f0.detach().std(0).clamp(min=1e-6)

    sur_clean = []
    for i in range(0, len(X), 128):
        xb = X[i:i + 128].to(DEVICE).requires_grad_(True)
        f, _ = diff_attr_features(model, xb, create_graph=False)
        sur_clean.append(f.detach().cpu())
    sur_clean = torch.cat(sur_clean)

    det_clean, _ = attribute_all(model, X, predict(model, X))
    scale_det = det_clean.std(0)
    scale_det[scale_det < 1e-6] = 1e-6

    print(f"{'lambda':>7} {'succ':>7} | {'surrogate dist':>15} {'detector dist':>14} "
          f"{'det score':>10}")
    print("-" * 60)

    base = None
    for lam in args.lams:
        Xadv = adaptive_lowfreq(model, X, Y, args.eps, args.k, lam,
                                scale_sur, steps=args.steps)
        pa = predict(model, Xadv)
        idx = (pa != Y).nonzero().squeeze(1)
        if len(idx) < 30:
            print(f"{lam:>7.1f} {len(idx)/len(X)*100:>6.1f}% |  too few successes")
            continue

        sur_adv = []
        for i in range(0, len(idx), 128):
            xb = Xadv[idx][i:i + 128].to(DEVICE).requires_grad_(True)
            f, _ = diff_attr_features(model, xb, create_graph=False)
            sur_adv.append(f.detach().cpu())
        sur_adv = torch.cat(sur_adv)
        det_adv, _ = attribute_all(model, Xadv[idx], pa[idx])

        d_sur = (((sur_adv - sur_clean[idx]) / scale_sur.cpu()) ** 2).mean().sqrt().item()
        d_det = float(np.sqrt((((det_adv - det_clean[idx.numpy()]) / scale_det) ** 2).mean()))

        if base is None:
            det = HistGradientBoostingClassifier(max_iter=400, random_state=0)
            det.fit(np.concatenate([det_clean, det_adv]),
                    np.concatenate([np.zeros(len(det_clean)), np.ones(len(det_adv))]))
            base = det
        score = base.predict_proba(det_adv)[:, 1].mean()

        print(f"{lam:>7.1f} {len(idx)/len(X)*100:>6.1f}% | {d_sur:>15.4f} {d_det:>14.4f} "
              f"{score:>10.4f}", flush=True)

    print("\nsurrogate dist = what the attack optimises (23 feats, Saliency + IxG)")
    print("detector dist  = what the detector reads (54 feats, adds IG + 3-way)")
    print("det score      = mean P(adversarial) from a detector fit on lambda=0")
    print("\nIf surrogate dist does not fall  -> the attack is broken (case A)")
    print("If it falls but detector dist does not -> surrogate gap (case B)")
    print("If both fall but score stays high -> the defence is real (case C)")


if __name__ == "__main__":
    main()
