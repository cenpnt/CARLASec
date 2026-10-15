"""Seminar background figures: why ADV2 needs a smooth ReLU, and a timeline of
the related work. No model needed.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe make_related_figures.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
GREY, ORANGE = "#8c8c8c", "#d95f02"
TAU = 1e-4                     # relu_surrogate.py, kind "adv2"
plt.rcParams.update({"font.size": 12, "axes.spines.top": False,
                     "axes.spines.right": False})


def relu_figure():
    z = np.linspace(-0.06, 0.06, 1201)
    r = np.sqrt(z ** 2 + TAU)
    cols = [
        ("the layer", "output", np.maximum(z, 0), (z + r) / 2),
        ("its slope\n(what explanation maps are made of)", "slope",
         (z > 0).astype(float), (1 + z / r) / 2),
        ("slope of the slope\n(what ADV$^2$ needs to attack a map)",
         "change in slope", np.zeros_like(z), TAU / (2 * r ** 3)),
    ]
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, (title, ylab, real, smooth) in zip(ax, cols):
        a.plot(z, real, color=GREY, lw=3, label="ReLU (real network)")
        a.plot(z, smooth, color=ORANGE, lw=2, ls="--",
               label="smooth stand-in (ADV$^2$)")
        a.set_title(title, fontsize=11)
        a.set_xlabel("input to the neuron")
        a.set_ylabel(ylab)
        a.axvline(0, color="black", lw=0.5, ls=":")
    ax[2].text(0.042, 25, "ReLU: zero everywhere\nso the attack gets\nno signal",
               color=GREY, fontsize=10, ha="center")
    ax[2].text(-0.035, 35, "stand-in: non-zero\nso the attack can\nsteer the map",
               color=ORANGE, fontsize=10, ha="center")
    ax[0].legend(frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    save(fig, "fig7_relu_surrogate.png")


# (year, label, lane, label dx in years, label dy, ha, va); lanes top to bottom
PAPERS = [
    (2014.0, "Saliency@Simonyan", 0, 0, .13, "center", "bottom"),
    (2016.6, "Input x Gradient@Shrikumar", 0, 0, .13, "center", "bottom"),
    (2017.4, "Integrated Gradients@Sundararajan", 0, 0, -.13, "center", "top"),
    (2017.4, "CW attack@Carlini & Wagner", 1, 0, .13, "center", "bottom"),
    (2018.3, "PGD@Madry", 1, 0, -.13, "center", "top"),
    (2018.6, "BPDA@Athalye", 1, 0.2, .13, "center", "bottom"),
    (2019.1, "Fragile explanations@Ghorbani", 2, -0.3, .13, "center", "bottom"),
    (2019.9, "Manipulated explanations@Dombrowski", 2, 0, -.13, "center",
     "top"),
    (2020.6, "ADV$^2$@Zhang (USENIX Sec)", 2, 0, .13, "center", "bottom"),
    (2021.9, "AdvEdge@Abdukhamidov", 2, 0.5, .13, "center", "bottom"),
    (2025.2, "AdViT@Abdukhamidov", 2, 0, .13, "center", "bottom"),
    (2020.1, "ML-LOO@Yang (AAAI)", 3, -1.0, 0, "right", "center"),
    (2020.55, "SHAP signatures@Fidel", 3, 0, -.13, "center", "top"),
    (2020.65, "X-Ensemble@Wang (KDD)", 3, 0.5, .3, "left", "bottom"),
]
LANES = ["Explanation@methods", "Attack@tools", "Attacks on@explanations",
         "Detectors built on@explanations"]


def timeline():
    fig, ax = plt.subplots(figsize=(14, 6.2))
    ys = [3, 2, 1, 0]
    for y, name in zip(ys, LANES):
        ax.axhline(y, color="#e0e0e0", lw=1, zorder=0)
        ax.text(2012.4, y, name.replace("@", "\n"), ha="right", va="center",
                fontsize=11, fontweight="bold")
    for yr, lab, lane, dx, dy, ha, va in PAPERS:
        y = ys[lane]
        key = "Zhang" in lab
        ax.plot(yr, y, "o", ms=10 if key else 7,
                color=ORANGE if key else "#4f5d75", zorder=3)
        if dx and abs(dx) > 0.4:
            ax.plot([yr, yr + dx], [y, y + dy], color="#bbbbbb", lw=0.8,
                    zorder=1)
        ax.text(yr + dx, y + dy, lab.replace("@", "\n"), ha=ha, va=va,
                fontsize=8.5, fontweight="bold" if key else None)
    star = (2026.6, 0.5)
    ax.plot(*star, "*", ms=24, color=ORANGE, zorder=3)
    ax.text(star[0], star[1] + 0.17, "This work\n(2026)", ha="center",
            va="bottom", fontsize=10, fontweight="bold", color=ORANGE)
    ax.annotate("", xy=(star[0] - 0.25, star[1] + 0.06), xytext=(2020.75, 0.93),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5))
    ax.annotate("", xy=(star[0] - 0.25, star[1] - 0.06), xytext=(2020.8, 0.05),
                arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.5,
                                ls="--"))
    ax.text(2023.6, 0.66, "improve the attack", color=ORANGE, fontsize=9.5,
            ha="center", va="top")
    ax.text(2023.6, 0.06, "test it against this kind of detector",
            color=ORANGE, fontsize=9.5, ha="center", va="top")
    ax.set_xlim(2012.5, 2027.4)
    ax.set_ylim(-0.4, 3.75)
    ax.set_yticks([])
    ax.spines[["left"]].set_visible(False)
    ax.set_xticks(range(2014, 2027, 2))
    save(fig, "fig7_related_timeline.png")


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    relu_figure()
    timeline()
