"""Figure 5: one sign clean, under a plain CW attack, and under the
detector-blind agreement attack, with all three attribution maps and the
detectors' verdicts. Both attacks use `disagree_attack` with the same budget.
The example is selected (plain caught, agreement missed), not representative.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u make_agreement_figure.py
"""
import os

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.ensemble import HistGradientBoostingClassifier

from model import load_trained, IMG_SIZE
from xai_detect import CKPT, DEVICE, attribute_all
from relu_surrogate import swap_relu
from aim1_attack import attack as plain_attack, predict, load_data
from aim1_adaptive import thr_at_fpr, xai_feats, pixel_feats
from aim1_disagree import disagree_attack
from make_figures import NAMES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
EPS, STEPS, RESTARTS, FPR = 0.03, 100, 2, 0.05
STOP = 14

# disagreement block starts at feature 45; pairs in sorted-name order
PAIRS = [("IG", "IxG"), ("IG", "Sal"), ("IxG", "Sal")]


def mag(a):
    """Per-pixel attribution magnitude, as the detector's features use it."""
    return a.abs().sum(0).numpy()


def main():
    os.makedirs(OUT, exist_ok=True)
    plain, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    surro, _ = load_trained(CKPT, DEVICE)
    swap_relu(surro, kind="adv2")

    X, Y = load_data(plain, img_size, 1000)
    Xtr, Ytr, Xte, Yte = X[:500], Y[:500], X[500:], Y[500:]

    # detectors: same recipe as aim1_disagree
    print("fitting the XAI and pixel trees ...", flush=True)
    advs = [plain_attack(plain, Xtr, Ytr, eps=EPS, k=96, lam=0.0,
                         steps=STEPS).float(),
            plain_attack(plain, Xtr, Ytr, eps=EPS, k=6, lam=0.0,
                         steps=STEPS).float()]
    Fx = np.concatenate([xai_feats(plain, Xtr, predict(plain, Xtr))]
                        + [xai_feats(plain, A, predict(plain, A)) for A in advs])
    Fp = np.concatenate([pixel_feats(Xtr)] + [pixel_feats(A) for A in advs])
    lab = np.concatenate([np.zeros(len(Xtr))] + [np.ones(len(A)) for A in advs])
    tree_x = HistGradientBoostingClassifier(max_iter=400,
                                            random_state=0).fit(Fx, lab)
    tree_p = HistGradientBoostingClassifier(max_iter=400,
                                            random_state=0).fit(Fp, lab)
    tx = thr_at_fpr(tree_x.predict_proba(
        xai_feats(plain, Xte, predict(plain, Xte)))[:, 1], FPR)
    tp = thr_at_fpr(tree_p.predict_proba(pixel_feats(Xte))[:, 1], FPR)

    # both attacks through the same function
    print("running the plain and agreement attacks ...", flush=True)
    kw = dict(eps=EPS, k=96, steps=STEPS, restarts=RESTARTS)
    cache = os.path.join(r"C:\Users\s4990998\xai_work", "fig5_cache.pt")
    if os.path.exists(cache):
        A_plain, A_agree = torch.load(cache)
        print("loaded cached attacks", cache)
    else:
        A_plain = disagree_attack(surro, None, None, Xte, Yte, lam_det=0.0,
                                  lam_dis=0.0, **kw).float()
        A_agree = disagree_attack(surro, None, None, Xte, Yte, lam_det=0.0,
                                  lam_dis=1.0, **kw).float()
        torch.save((A_plain, A_agree), cache)

    def score(A):
        p = predict(plain, A)
        sx = tree_x.predict_proba(xai_feats(plain, A, p))[:, 1]
        sp = tree_p.predict_proba(pixel_feats(A))[:, 1]
        return p, sx, sp

    pp, sxp, spp = score(A_plain)
    pa, sxa, spa = score(A_agree)
    mis_p, mis_a = (pp != Yte).numpy(), (pa != Yte).numpy()
    print(f"plain     mis {mis_p.mean()*100:.1f}%  XAI evasion "
          f"{(mis_p & (sxp < tx)).mean()*100:.1f}%")
    print(f"agreement mis {mis_a.mean()*100:.1f}%  XAI evasion "
          f"{(mis_a & (sxa < tx)).mean()*100:.1f}%")

    # Select on the XAI detector only; the pixel verdict is still shown.
    good = mis_p & mis_a & (sxp >= tx) & (sxa < tx)
    cand = np.nonzero(good)[0]
    if len(cand) == 0:
        raise SystemExit("no image where plain is caught and agreement evades")
    stops = [i for i in cand if int(Yte[i]) == STOP]
    pool = stops if stops else list(cand)
    i = max(pool, key=lambda j: sxp[j] - sxa[j])    # clearest contrast
    print(f"{len(cand)} candidate images, {len(stops)} of them STOP signs; "
          f"using index {i} ({NAMES[int(Yte[i])]})")

    # attributions at evaluation quality (Captum, 32-step IG)
    xs = torch.stack([Xte[i], A_plain[i], A_agree[i]])
    ts = torch.stack([Yte[i], pp[i], pa[i]])
    F, maps = attribute_all(plain, xs, ts)
    dis = F[:, 45:54].reshape(3, 3, 3)             # row, pair, [pearson,cos,iou]
    with torch.no_grad():
        conf = torch.softmax(plain(xs.to(DEVICE)), 1).max(1).values.cpu()

    rows = [
        ("Clean", Yte[i], None, None, None),
        ("Plain attack (CW only)", pp[i], sxp[i], spp[i], A_plain[i] - Xte[i]),
        ("Agreement attack (NEW, blind)", pa[i], sxa[i], spa[i],
         A_agree[i] - Xte[i]),
    ]

    fig, ax = plt.subplots(3, 6, figsize=(19, 12),
                           gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1, 1.15]))
    for r, (title, pred, sx, sp, delta) in enumerate(rows):
        c = ax[r, 0]
        c.imshow(xs[r].permute(1, 2, 0).numpy())
        colour = "green" if r == 0 else "red"
        head = f"{title}\npred: {NAMES[int(pred)]} ({conf[r]:.2f})"
        if sx is not None:
            vx = "CAUGHT" if sx >= tx else "missed"
            vp = "CAUGHT" if sp >= tp else "missed"
            head += f"\nXAI: {vx}   pixel: {vp}"
        c.set_title(head, color=colour, fontsize=10)

        c = ax[r, 1]
        if delta is None:
            c.text(0.5, 0.5, "(no change)", ha="center", va="center",
                   color="grey", transform=c.transAxes)
        else:
            c.imshow((0.5 + 10 * delta).clamp(0, 1).permute(1, 2, 0).numpy())
            c.set_title(f"change x10\nmax {delta.abs().max()*255:.1f} / 255",
                        fontsize=10)

        for col, (key, label) in enumerate(
                [("ig", "Integrated Gradients"), ("ixg", "Input x Gradient"),
                 ("sal", "Saliency")], start=2):
            m = mag(maps[key][r])
            ax[r, col].imshow(m, cmap="viridis",
                              vmax=np.quantile(m, 0.995))
            ax[r, col].set_title(label, fontsize=10)

        c = ax[r, 5]
        c.axis("off")
        lines = ["How much the three maps agree", "(Pearson   top-10% overlap)", ""]
        for p, (a, b) in enumerate(PAIRS):
            lines.append(f"{a:>3} vs {b:<3}   {dis[r, p, 0]:+.2f}      "
                         f"{dis[r, p, 2]:.2f}")
        if r > 0:
            err = ((dis[r, :, :2] - dis[0, :, :2]) ** 2).sum()
            lines += ["", f"distance from clean: {err:.4f}"]
        c.text(0.02, 0.5, "\n".join(lines), family="monospace", fontsize=10,
               va="center", transform=c.transAxes)

    for a in ax.ravel():
        a.set_xticks([])
        a.set_yticks([])
    fig.suptitle("The agreement attack: fool the model, but keep the three "
                 "explanations agreeing the way they did on the clean image\n"
                 f"eps = {EPS} (at most {EPS*255:.0f}/255 per pixel), "
                 f"detector threshold at {int(FPR*100)}% false alarms. "
                 "Selected example; the overall rate is in the results table.",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=3.0)
    path = os.path.join(OUT, "fig5_agreement_attack.png")
    fig.savefig(path, dpi=150)
    print("saved", path)

    # the three images on their own, enlarged, for slides
    for name, x in zip(["img_clean", "img_plain", "img_agreement"], xs):
        im = Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round()
                             .astype(np.uint8))
        p = os.path.join(OUT, f"fig5_{name}.png")
        im.resize((384, 384), Image.NEAREST).save(p)
        print("saved", p)


if __name__ == "__main__":
    main()
