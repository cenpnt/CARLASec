"""Figure 6 (slides): published ADV2 (sign-PGD) vs our reformulation (Adam on
a tanh perturbation, CW margin). Concept plots plus result charts; the numbers
are the Run 16-17 results in RESULTS_aim1_log.md. No model needed.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe make_adv2_figures.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
PGD_C, OURS_C, GREY = "#8c8c8c", "#d95f02", "#bdbdbd"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


def main_result():
    pgd, ours = [40.8, 48.4, 50.0], [76.4, 75.2, 76.6]
    labels = ["AdvEdge+\n(published)", "AdvEdge\n(published)",
              "ADV$^2$\n(published)", "ADV$^2$\nreformulated\n(ours)"]
    means = [23.8, 34.6, np.mean(pgd), np.mean(ours)]
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    bars = ax.bar(labels, means, color=[GREY, GREY, PGD_C, OURS_C], width=0.6)
    for x, pts in [(2, pgd), (3, ours)]:
        ax.scatter([x] * 3, pts, color="black", s=18, zorder=3,
                   label="each data split" if x == 2 else None)
    for b, m in zip(bars, means):
        top = max([m] + (pgd if b is bars[2] else ours if b is bars[3] else []))
        ax.text(b.get_x() + b.get_width() / 2, top + 2, f"{m:.1f}%",
                ha="center", fontweight="bold")
    ax.set_ylabel("Gets past the XAI detector (%)")
    ax.set_ylim(0, 90)
    ax.set_title("Same goal, same images, same detector, no detector access",
                 fontsize=12)
    ax.legend(frameon=False, loc="upper left")
    save(fig, "fig6_main_result.png")


def tradeoff():
    # mean misclassification and evasion over the three splits (Run 17)
    rows = [("ADV$^2$ (published)", 96.9, 46.4),
            ("ADV$^2$ reformulated (ours)", 86.2, 76.1)]
    fig, ax = plt.subplots(figsize=(8, 2.8))
    for i, (name, mis, ev) in enumerate(rows):
        parts = [(ev, "#1b9e77", "fooled and NOT caught"),
                 (mis - ev, "#e7298a", "fooled but caught"),
                 (100 - mis, GREY, "not fooled")]
        left = 0
        for w, c, lab in parts:
            ax.barh(i, w, left=left, color=c, label=lab if i == 0 else None)
            if w > 6:
                ax.text(left + w / 2, i, f"{w:.0f}%", ha="center",
                        va="center", color="white", fontweight="bold")
            left += w
    ax.set_yticks([0, 1], [r[0] for r in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% of 500 test signs")
    ax.legend(frameon=False, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.35))
    save(fig, "fig6_tradeoff.png")


def ablation_and_sweep():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ["ours:\n2 tries,\nkeep best", "ours:\n1 try,\nkeep best",
             "ours:\n1 try,\nkeep last", "published:\n200 steps",
             "published:\n800 steps"]
    vals = [76.4, 76.4, 75.0, 40.8, 44.0]
    bars = a.bar(names, vals, color=[OURS_C] * 3 + [PGD_C] * 2, width=0.6)
    for r, v in zip(bars, vals):
        a.text(r.get_x() + r.get_width() / 2, v + 1.5, f"{v:.1f}%",
               ha="center")
    a.set_ylim(0, 90)
    a.set_ylabel("Gets past the XAI detector (%)")
    a.set_title("Removing our extra tricks costs ~1 point;\n"
                "giving published 4x more steps gains ~3", fontsize=11)
    a.tick_params(axis="x", labelsize=9)

    w, v = [10, 100, 1000], [40.8, 30.0, 24.4]
    b.plot(w, v, "o-", color=PGD_C, lw=2, label="ADV$^2$ (published)")
    b.axhline(76.4, color=OURS_C, ls="--", lw=2, label="ours (weight 10)")
    for x, y in zip(w, v):
        b.text(x, y + 2.5, f"{y:.1f}%", ha="center")
    b.set_xscale("log")
    b.set_xticks(w, ["10", "100", "1000"])
    b.set_ylim(0, 90)
    b.set_xlabel("weight on 'keep the maps clean'")
    b.set_title("Published ADV$^2$ gets worse when told\nto care more "
                "about the maps", fontsize=11)
    b.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig6_ablation.png")


def toy_paths():
    """Sign steps vs Adam on a narrow tilted valley, a 2-pixel stand-in."""
    c, s = np.cos(0.5), np.sin(0.5)

    def grad(p):
        u, v = c * p[0] + s * p[1], -s * p[0] + c * p[1]
        gu, gv = u, 30 * v
        return np.array([c * gu - s * gv, s * gu + c * gv])

    def loss(x, y):
        u, v = c * x + s * y, -s * x + c * y
        return 0.5 * u ** 2 + 15 * v ** 2

    start, n, lr = np.array([-2.2, 1.4]), 150, 0.08
    p, sign_path = start.copy(), [start.copy()]
    for _ in range(n):
        p = p - lr * np.sign(grad(p))
        sign_path.append(p.copy())
    p, m, v, adam_path = start.copy(), 0, 0, [start.copy()]
    for t in range(1, n + 1):
        g = grad(p)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * g ** 2
        p = p - lr * (m / (1 - 0.9 ** t)) / (np.sqrt(v / (1 - 0.999 ** t))
                                               + 1e-8)
        adam_path.append(p.copy())
    sign_path, adam_path = np.array(sign_path), np.array(adam_path)

    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.6),
                               gridspec_kw=dict(width_ratios=[1.3, 1]))
    xx, yy = np.meshgrid(np.linspace(-2.6, 1.2, 300), np.linspace(-1.2, 1.8, 300))
    a.contour(xx, yy, loss(xx, yy), levels=np.geomspace(0.02, 40, 14),
              colors="#cccccc", linewidths=0.8)
    a.plot(*sign_path.T, "o-", color=PGD_C, ms=3, lw=1.2,
           label="sign steps (published ADV$^2$)")
    a.plot(*adam_path.T, "o-", color=OURS_C, ms=3, lw=1.6, label="Adam (ours)")
    a.plot(0, 0, "*", color="black", ms=14, label="best point")
    a.plot(*start, "s", color="black", ms=7)
    a.set_xlabel("pixel 1")
    a.set_ylabel("pixel 2")
    a.set_title("Toy example: reaching the bottom of a narrow valley",
                fontsize=11)
    a.legend(frameon=False, fontsize=9, loc="upper right")
    a.set_aspect("equal")

    b.semilogy([loss(*q) for q in sign_path], color=PGD_C, lw=2,
               label="sign steps")
    b.semilogy([loss(*q) for q in adam_path], color=OURS_C, lw=2, label="Adam")
    b.set_xlabel("step")
    b.set_ylabel("loss (log scale)")
    b.set_title("Same step size: sign steps stall, Adam keeps improving",
                fontsize=11)
    b.legend(frameon=False)
    fig.tight_layout()
    save(fig, "fig6_toy_adam_vs_sign.png")


def cw_and_tanh():
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    gap = np.linspace(-6, 8, 400)          # correct logit minus best wrong logit
    a.plot(gap, np.log1p(np.exp(gap)), color=PGD_C, lw=2,
           label="cross-entropy style: never reaches 0")
    a.plot(gap, np.clip(gap + 2, 0, None), color=OURS_C, lw=2.5,
           label="CW margin ($\\kappa$ = 2)")
    a.axvspan(-6, -2, color="#1b9e77", alpha=0.12)
    a.text(-4, 3.0, "wrong by 2 or more:\nloss = 0, effort goes\nto hiding",
           ha="center", fontsize=9, color="#1b9e77")
    a.axvline(0, color="black", lw=0.6, ls=":")
    a.text(0.2, 5.0, "decision\nboundary", fontsize=8)
    a.set_xlabel("correct class score minus best wrong class score")
    a.set_ylabel("classifier loss")
    a.set_ylim(0, 10)
    a.set_title("CW margin: stop pushing once it is wrong enough",
                fontsize=11)
    a.legend(frameon=False, fontsize=9, loc="upper left")

    eps = 0.03
    z = np.linspace(-3, 3, 400)
    b.plot(z, np.clip(z * eps / 1.5, -eps, eps) * 255, color=PGD_C, lw=2,
           label="clip (published): hard wall")
    b.plot(z, eps * np.tanh(z) * 255, color=OURS_C, lw=2.5,
           label="$\\varepsilon$ tanh(z) (CW, ours): soft cushion")
    for y in (eps * 255, -eps * 255):
        b.axhline(y, color="black", lw=0.8, ls="--")
    b.text(-3, eps * 255 + 0.4, "limit $\\varepsilon$ = 0.03 (about 8/255)", fontsize=9)
    b.set_xlabel("what the optimiser moves")
    b.set_ylabel("change to the pixel (/255)")
    b.set_ylim(-11, 11)
    b.set_title("tanh: the limit can never be broken,\n"
                "and the optimiser never gets stuck on it", fontsize=11)
    b.legend(frameon=False, fontsize=9, loc="lower right")
    fig.tight_layout()
    save(fig, "fig6_cw_and_tanh.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    main_result()
    tradeoff()
    ablation_and_sweep()
    toy_paths()
    cw_and_tanh()
