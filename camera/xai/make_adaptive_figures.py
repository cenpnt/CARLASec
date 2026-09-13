"""Figures for the adaptive-attack experiment.

Figure 3  the attacker cost curve. As lambda rises the attacker tries harder to
          keep its attributions looking clean. Detection AUC should fall and
          attack success should fall with it. The shape of that trade-off is the
          result: a defence is good if evasion is expensive.

Figure 4  what evasion looks like. Same stop sign under the non-adaptive attack
          and the adaptive one, with Integrated Gradients underneath. If the
          adaptive row's attribution looks like the clean row's, the attacker
          has hidden from the explanation.

Run AFTER adaptive_attack.py:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe make_adaptive_figures.py
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import GTSRB
from captum.attr import IntegratedGradients

from model import load_trained, IMG_SIZE
from xai_detect import DATA, CKPT, DEVICE
from adaptive_attack import adaptive_lowfreq, diff_attr_features
from make_figures import NAMES, lap

OUT = r"\\puffball.labs.eait.uq.edu.au\s4990998\Documents\REIT4842\Code\CARLASec\camera\xai\figures"
NPZ = r"C:\Users\s4990998\xai_work\adaptive_results.npz"


def fig_curve():
    d = np.load(NPZ, allow_pickle=True)
    r = d["rows"]                      # lam, succ, n, img_auc, xai_auc
    lam, succ, img, xai = r[:, 0], r[:, 1] * 100, r[:, 3], r[:, 4]
    xpos = np.arange(len(lam))

    fig, ax1 = plt.subplots(figsize=(9, 5.4))
    ax1.plot(xpos, xai, "o-", color="#1f77b4", lw=2.4, ms=8, label="XAI detector AUC")
    ax1.plot(xpos, img, "s--", color="#7a7a7a", lw=1.8, ms=7, label="pixel-statistics AUC")
    ax1.axhline(0.5, color="red", ls=":", lw=1.4)
    ax1.text(xpos[-1], 0.512, "chance", color="red", fontsize=9, ha="right")
    ax1.set_ylabel("detection AUC", fontsize=11)
    ax1.set_xlabel("lambda  (how hard the attacker tries to hide from the explanation)",
                   fontsize=11)
    ax1.set_ylim(0.45, 1.02)
    ax1.set_xticks(xpos)
    ax1.set_xticklabels([f"{v:g}" for v in lam])
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.plot(xpos, succ, "^-", color="#d62728", lw=2, ms=8, label="attack success rate")
    ax2.set_ylabel("attack success rate (%)", color="#d62728", fontsize=11)
    ax2.tick_params(axis="y", colors="#d62728")
    ax2.set_ylim(0, max(succ.max() * 1.25, 5))

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=9)
    plt.title("Adaptive attack: what evading the XAI detector costs the attacker\n"
              "(detector trained on non-adaptive attacks, tested on adaptive ones)",
              fontsize=12)
    plt.tight_layout()
    p = os.path.join(OUT, "fig3_adaptive_tradeoff.png")
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print("saved", p)


def fig_visual(lam_hi=10.0, eps=0.06, k=3, steps=200):
    model, ckpt = load_trained(CKPT, DEVICE)
    s = ckpt.get("img_size", IMG_SIZE)
    tf = transforms.Compose([transforms.Resize((s, s)), transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)

    xs, ys = [], []
    for i in range(0, len(ds), 3):
        x, y = ds[i]
        if int(y) == 14:                       # stop signs
            xs.append(x); ys.append(y)
        if len(xs) >= 64:
            break
    X, Y = torch.stack(xs), torch.tensor(ys)
    with torch.no_grad():
        p = model(X.to(DEVICE)).softmax(1).cpu()
    keep = (p.argmax(1) == Y).nonzero().squeeze(1)
    X, Y = X[keep], Y[keep]

    xc = X[:32].to(DEVICE).requires_grad_(True)
    f0, _ = diff_attr_features(model, xc, create_graph=False)
    scale = f0.detach().std(0).clamp(min=1e-6)

    A0 = adaptive_lowfreq(model, X, Y, eps, k, 0.0, scale, steps=steps)
    A1 = adaptive_lowfreq(model, X, Y, eps, k, lam_hi, scale, steps=steps)
    with torch.no_grad():
        p0 = model(A0.to(DEVICE)).softmax(1).cpu()
        p1 = model(A1.to(DEVICE)).softmax(1).cpu()
    both = ((p0.argmax(1) != Y) & (p1.argmax(1) != Y)).nonzero().squeeze(1)
    if len(both) == 0:
        print("no stop sign fooled by both settings; using the non-adaptive one only")
        both = (p0.argmax(1) != Y).nonzero().squeeze(1)
    j = int(both[0])

    ig = IntegratedGradients(model)
    rows = [("Clean", X[j], p[keep][j], None),
            (f"Non-adaptive (lambda=0)", A0[j], p0[j], A0[j] - X[j]),
            (f"Adaptive (lambda={lam_hi:g})", A1[j], p1[j], A1[j] - X[j])]

    fig, ax = plt.subplots(3, 3, figsize=(10.5, 10))
    for r, (tag, img, prob, delta) in enumerate(rows):
        xb = img.unsqueeze(0).to(DEVICE).requires_grad_(True)
        a = ig.attribute(xb, target=int(prob.argmax()), n_steps=48)[0].detach().cpu()
        cls = int(prob.argmax())
        col = "green" if cls == int(Y[j]) else "red"
        ax[r, 0].imshow(img.permute(1, 2, 0).numpy())
        ax[r, 0].set_title(f"{tag}\npred: {NAMES[cls]} ({prob.max():.2f})",
                           fontsize=10, color=col)
        if delta is None:
            ax[r, 1].axis("off")
            ax[r, 1].text(0.5, 0.5, "(no perturbation)", ha="center", va="center",
                          color="grey", fontsize=10)
        else:
            ax[r, 1].imshow((delta * 10 + 0.5).clamp(0, 1).permute(1, 2, 0).numpy())
            ax[r, 1].set_title(f"perturbation x10\nroughness={lap(delta).mean():.5f}",
                               fontsize=10)
        m = a.abs().sum(0)
        ax[r, 2].imshow(m.numpy() / (m.max() + 1e-9), cmap="viridis")
        ax[r, 2].set_title("Integrated Gradients", fontsize=10)
        for c in range(3):
            ax[r, c].set_xticks([]); ax[r, c].set_yticks([])

    fig.suptitle("Does the adaptive attacker hide from the explanation?\n"
                 "If row 3's attribution resembles row 1's, the XAI detector is evaded",
                 fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    p_ = os.path.join(OUT, "fig4_adaptive_visual.png")
    plt.savefig(p_, dpi=130, bbox_inches="tight"); plt.close()
    print("saved", p_)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    if os.path.exists(NPZ):
        fig_curve()
    else:
        print("no npz yet; skipping the trade-off curve")
    fig_visual()
