"""Figure 6 (slides), real images: one sign clean, under published ADV2
(sign-PGD) and under our reformulation (Adam on a tanh perturbation), with the
three maps and the XAI detector's verdict; plus the map loss per step of both
optimisers. Same detector recipe and split as aim1_improve.py (seed 0).

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u make_adv2_example_figure.py
"""
import os

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import CKPT, DEVICE, attribute_all
from relu_surrogate import swap_relu
from aim1_attack import attack as plain_attack, predict, load_data
from aim1_adaptive import thr_at_fpr, xai_feats, pixel_feats
from aim1_improve import pgd_attack, blind_attack, PIX_LIVE
from make_figures import NAMES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CACHE = os.path.join(r"C:\Users\s4990998\xai_work", "fig6_cache.pt")
EPS, FPR, W, STOP = 0.03, 0.05, {"adv2_all": 10.0}, 14
PGD_C, OURS_C = "#8c8c8c", "#d95f02"


def mag(a):
    return a.abs().sum(0).numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    plain, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    surro, _ = load_trained(CKPT, DEVICE)
    swap_relu(surro, kind="adv2")

    X, Y = load_data(plain, img_size, 1000, seed=0)
    Xtr, Ytr, Xte, Yte = X[:500], Y[:500], X[500:], Y[500:]

    print("fitting the XAI detector ...", flush=True)
    clf = PyTorchClassifier(model=plain, loss=nn.CrossEntropyLoss(),
                            input_shape=(3, img_size, img_size),
                            nb_classes=NUM_CLASSES, clip_values=(0.0, 1.0),
                            device_type="gpu" if DEVICE == "cuda" else "cpu")
    advs = [plain_attack(plain, Xtr, Ytr, eps=EPS, k=k, lam=0.0,
                         steps=100).float() for k in (96, 6, 3)]
    advs.append(torch.from_numpy(FastGradientMethod(
        estimator=clf, eps=EPS, batch_size=256).generate(
            x=Xtr.numpy())).float())
    Fx_c = xai_feats(plain, Xtr, predict(plain, Xtr))
    Fx = np.concatenate([Fx_c] + [xai_feats(plain, A, predict(plain, A))
                                  for A in advs])
    lab = np.concatenate([np.zeros(len(Xtr))] + [np.ones(len(A)) for A in advs])
    tree = HistGradientBoostingClassifier(max_iter=400,
                                          random_state=0).fit(Fx, lab)
    tx = thr_at_fpr(tree.predict_proba(
        xai_feats(plain, Xte, predict(plain, Xte)))[:, 1], FPR)
    sd_x = torch.as_tensor(Fx_c.std(0) + 1e-6, dtype=torch.float32,
                           device=DEVICE)
    sd_p = torch.as_tensor(pixel_feats(Xtr)[:, PIX_LIVE].std(0) + 1e-6,
                           dtype=torch.float32, device=DEVICE)

    if os.path.exists(CACHE):
        A_pgd, A_ours, tr_pgd, tr_ours = torch.load(CACHE)
        print("loaded cached attacks", CACHE)
    else:
        print("running published ADV2 and ours ...", flush=True)
        tr_pgd, tr_ours = [], []
        A_pgd = pgd_attack(surro, Xte, Yte, EPS, W, sd_x, sd_p,
                           trace=tr_pgd).float()
        A_ours = blind_attack(surro, Xte, Yte, EPS, W, sd_x, sd_p,
                              trace=tr_ours).float()
        torch.save((A_pgd, A_ours, tr_pgd, tr_ours), CACHE)

    def score(A):
        p = predict(plain, A)
        return p, tree.predict_proba(xai_feats(plain, A, p))[:, 1]

    pp, sp = score(A_pgd)
    po, so = score(A_ours)
    mis_p, mis_o = (pp != Yte).numpy(), (po != Yte).numpy()
    for n, m, s in [("published", mis_p, sp), ("ours", mis_o, so)]:
        print(f"{n:>9}: mis {m.mean()*100:.1f}%  XAI evasion "
              f"{(m & (s < tx)).mean()*100:.1f}%")

    # convergence of the map loss (mean over the first batch of 64 signs)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(np.arange(1, len(tr_pgd) + 1), tr_pgd, color=PGD_C, lw=2,
            label="published ADV$^2$ (sign steps)")
    ax.plot(np.arange(1, len(tr_ours) + 1), tr_ours, color=OURS_C, lw=2.5,
            label="ours (Adam + tanh)")
    ax.set_xlabel("steps spent on the map loss")
    ax.set_ylabel("map distance from clean\n(lower = harder to detect)")
    ax.set_title("Real attack, 64 signs: sign steps flatten out, Adam keeps "
                 "lowering\nthe map distance, in 60 steps against 200",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    path = os.path.join(OUT, "fig6_map_loss_curve.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)

    # example: published caught, ours evades; prefer a STOP sign
    cand = np.nonzero(mis_p & mis_o & (sp >= tx) & (so < tx))[0]
    print(f"{len(cand)} signs where published is caught and ours evades")
    n_cand = len(cand)
    stops = [i for i in cand if int(Yte[i]) == STOP]
    i = max(stops or list(cand), key=lambda j: sp[j] - so[j])
    print(f"using index {i} ({NAMES[int(Yte[i])]})")

    xs = torch.stack([Xte[i], A_pgd[i], A_ours[i]])
    ts = torch.stack([Yte[i], pp[i], po[i]])
    _, maps = attribute_all(plain, xs, ts)
    with torch.no_grad():
        conf = torch.softmax(plain(xs.to(DEVICE)), 1).max(1).values.cpu()

    rows = [("Clean", None), ("Published ADV$^2$", sp[i]),
            ("ADV$^2$ reformulated (ours)", so[i])]
    fig, ax = plt.subplots(3, 5, figsize=(15, 9.6))
    for r, (title, s) in enumerate(rows):
        c = ax[r, 0]
        c.imshow(xs[r].permute(1, 2, 0).numpy())
        head = f"{title}\nmodel says: {NAMES[int(ts[r])]} ({conf[r]:.2f})"
        colour = "black"
        if s is not None:
            caught = s >= tx
            head += "\nXAI detector: " + ("CAUGHT" if caught else "missed")
            colour = "#c51b7d" if caught else "#1b9e77"
        c.set_title(head, color=colour, fontsize=10)

        c = ax[r, 1]
        if r == 0:
            c.text(0.5, 0.5, "(no change)", ha="center", va="center",
                   color="grey", transform=c.transAxes)
        else:
            d = xs[r] - xs[0]
            c.imshow((0.5 + 10 * d).clamp(0, 1).permute(1, 2, 0).numpy())
            c.set_title(f"change x10\nmax {d.abs().max()*255:.1f} / 255",
                        fontsize=10)
        for col, (k, lab) in enumerate(
                [("ig", "Integrated Gradients"), ("ixg", "Input x Gradient"),
                 ("sal", "Saliency")], start=2):
            m = mag(maps[k][r])
            ax[r, col].imshow(m, cmap="viridis", vmax=np.quantile(m, 0.995))
            ax[r, col].set_title(lab, fontsize=10)
    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("Same goal (fool the model, keep all three maps clean), "
                 f"same budget (eps = {EPS}), no detector access. "
                 f"Published caught, ours missed on {n_cand} of {len(Yte)} "
                 "test signs; this is one of them.",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=2.5)
    path = os.path.join(OUT, "fig6_example.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)

    for name, x in zip(["clean", "published", "ours"], xs):
        im = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round()
                             .astype(np.uint8))
        p = os.path.join(OUT, f"fig6_img_{name}.png")
        im.resize((384, 384), Image.NEAREST).save(p)
        print("saved", p)


if __name__ == "__main__":
    main()
