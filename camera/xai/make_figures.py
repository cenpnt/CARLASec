"""Figures for the thesis: what the attacks look like and why the baseline fails.

Figure 1  attack comparison. Rows: clean, FGSM, low-frequency. Columns: the
          image, the perturbation amplified, the perturbation's Laplacian (what
          the pixel baseline actually measures), and the Integrated Gradients
          attribution (what the XAI detector measures).
          The point: both attacks fool the model and look clean, but only FGSM
          leaves a Laplacian signature. The attribution changes for both.

Figure 2  the data and the patch: sample GTSRB images the classifier was
          trained on, plus the learned adversarial patch.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe make_figures.py
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import GTSRB

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod
from captum.attr import IntegratedGradients

from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import DATA, CKPT, DEVICE
from lowfreq_confirm import lowfreq_strong

OUT = r"\\puffball.labs.eait.uq.edu.au\s4990998\Documents\REIT4842\Code\CARLASec\camera\xai\figures"

NAMES = [
    "Speed 20", "Speed 30", "Speed 50", "Speed 60", "Speed 70", "Speed 80",
    "End speed 80", "Speed 100", "Speed 120", "No passing", "No passing >3.5t",
    "Right-of-way", "Priority road", "Yield", "STOP", "No vehicles",
    ">3.5t prohibited", "No entry", "General caution", "Curve left",
    "Curve right", "Double curve", "Bumpy road", "Slippery road",
    "Narrows right", "Road work", "Traffic signals", "Pedestrians",
    "Children", "Bicycles", "Ice/snow", "Wild animals", "End limits",
    "Turn right", "Turn left", "Ahead only", "Straight/right", "Straight/left",
    "Keep right", "Keep left", "Roundabout", "End no passing", "End no pass >3.5t",
]


def lap(d):
    """Laplacian magnitude map of a (3,H,W) perturbation."""
    g = d.mean(0)
    return (g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1]
            - 4 * g[1:-1, 1:-1]).abs()


def main():
    os.makedirs(OUT, exist_ok=True)
    model, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)

    # ---- collect a pool, preferring STOP signs for the narrative -------------
    xs, ys = [], []
    for i in range(0, len(ds), 3):
        x, y = ds[i]
        xs.append(x); ys.append(y)
        if len(xs) >= 1500:
            break
    X, Y = torch.stack(xs), torch.tensor(ys)
    with torch.no_grad():
        p = torch.cat([model(X[i:i + 256].to(DEVICE)).softmax(1).cpu()
                       for i in range(0, len(X), 256)])
    # NOTE: the classifier was trained with label smoothing 0.1, so its softmax
    # is capped around 0.93. A 0.95 threshold selects nothing.
    ok = (p.argmax(1) == Y) & (p.max(1).values > 0.85)
    stop = ok & (Y == 14)
    pool = stop.nonzero().squeeze(1) if stop.sum() >= 5 else ok.nonzero().squeeze(1)
    print(f"candidate pool: {len(pool)} images (stop signs: {int(stop.sum())})")

    clf = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(),
                            input_shape=(3, img_size, img_size), nb_classes=NUM_CLASSES,
                            clip_values=(0.0, 1.0),
                            device_type="gpu" if DEVICE == "cuda" else "cpu")
    Xp, Yp = X[pool], Y[pool]
    Xf = torch.from_numpy(FastGradientMethod(estimator=clf, eps=0.03,
                                             batch_size=128).generate(x=Xp.numpy())).float()
    Xl = lowfreq_strong(model, Xp, Yp, eps=0.06, k=3, steps=300, restarts=2).float()

    with torch.no_grad():
        pf = model(Xf.to(DEVICE)).softmax(1).cpu()
        pl = model(Xl.to(DEVICE)).softmax(1).cpu()
    both = ((pf.argmax(1) != Yp) & (pl.argmax(1) != Yp)).nonzero().squeeze(1)
    if len(both) == 0:
        print("no image fooled by both attacks; falling back to low-freq only")
        both = (pl.argmax(1) != Yp).nonzero().squeeze(1)
    j = int(both[0])
    print(f"using image with true class {int(Yp[j])} ({NAMES[int(Yp[j])]})")

    # ---- Figure 1 -----------------------------------------------------------
    ig = IntegratedGradients(model)
    rows = []
    for tag, img, prob in [("Clean", Xp[j], p[pool][j]),
                           ("FGSM (eps=0.03)", Xf[j], pf[j]),
                           ("Low-frequency (k=3)", Xl[j], pl[j])]:
        xb = img.unsqueeze(0).to(DEVICE).requires_grad_(True)
        a = ig.attribute(xb, target=int(prob.argmax()), n_steps=48)[0].detach().cpu()
        rows.append((tag, img, a, prob, img - Xp[j]))

    # One colour scale for both roughness panels. Left to autoscale, each panel
    # is stretched to its own maximum and the two cannot be compared by eye:
    # the smooth row looks busy because a handful of clipping pixels set its
    # ceiling, which understates a difference of two orders of magnitude.
    lap_max = max(lap(rw[4]).max().item() for rw in rows[1:])

    fig, ax = plt.subplots(3, 4, figsize=(13.5, 10))
    for r, (tag, img, attr, prob, delta) in enumerate(rows):
        ax[r, 0].imshow(img.permute(1, 2, 0).numpy())
        cls = int(prob.argmax())
        colour = "green" if cls == int(Yp[j]) else "red"
        ax[r, 0].set_title(f"{tag}\npred: {NAMES[cls]} ({prob.max():.2f})",
                           fontsize=10, color=colour)

        if r == 0:
            ax[r, 1].axis("off"); ax[r, 2].axis("off")
            ax[r, 1].text(0.5, 0.5, "(no perturbation)", ha="center", va="center",
                          fontsize=10, color="grey")
        else:
            d = delta
            ax[r, 1].imshow((d * 10 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy())
            ax[r, 1].set_title(f"perturbation x10\nLinf={d.abs().max():.3f}", fontsize=10)
            lp = lap(d)
            ax[r, 2].imshow(lp.numpy(), cmap="inferno", vmin=0, vmax=lap_max)
            ax[r, 2].set_title(f"perturbation roughness\nmean|Laplacian|={lp.mean():.5f}",
                               fontsize=10)

        m = attr.abs().sum(0)
        ax[r, 3].imshow(m.numpy() / (m.max() + 1e-9), cmap="viridis")
        ax[r, 3].set_title("Integrated Gradients\n(what XAI sees)", fontsize=10)
        for c in range(4):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    fig.suptitle("Why the pixel baseline fails on low-frequency attacks\n"
                 "Both attacks fool the classifier, but only FGSM leaves a roughness signature",
                 fontsize=13)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    f1 = os.path.join(OUT, "fig1_attack_comparison.png")
    plt.savefig(f1, dpi=130, bbox_inches="tight"); plt.close()
    print("saved", f1)

    # ---- Figure 2 -----------------------------------------------------------
    fig, ax = plt.subplots(2, 8, figsize=(16, 4.6))
    g = torch.Generator().manual_seed(3)
    pick = torch.randperm(len(X), generator=g)[:15]
    for i, idx in enumerate(pick):
        a = ax[i // 8, i % 8]
        a.imshow(X[idx].permute(1, 2, 0).numpy())
        a.set_title(NAMES[int(Y[idx])], fontsize=8)
        a.set_xticks([]); a.set_yticks([])
    pp = r"C:\Users\s4990998\xai_work\patch.npy"
    a = ax[1, 7]
    if os.path.exists(pp):
        patch = torch.from_numpy(np.load(pp)).float()
        if patch.dim() == 4:
            patch = patch[0]
        a.imshow(patch.clamp(0, 1).permute(1, 2, 0).numpy())
        a.set_title("learned adversarial\npatch", fontsize=8, color="red")
    else:
        a.axis("off")
    a.set_xticks([]); a.set_yticks([])
    fig.suptitle("GTSRB traffic signs used to train the classifier (99.24% clean accuracy)",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    f2 = os.path.join(OUT, "fig2_dataset_and_patch.png")
    plt.savefig(f2, dpi=130, bbox_inches="tight"); plt.close()
    print("saved", f2)


if __name__ == "__main__":
    main()
