"""Seminar figure: the XAI detector pipeline, from sign image to alarm. The
sign and the three maps are cut from fig_attribution_methods.png (same image,
same model), so no model is needed.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe make_detector_diagram.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
SRC = os.path.join(HERE, "..", "..", "..", "..", "fig_attribution_methods.png")
GREY, ORANGE, INK = "#8c8c8c", "#d95f02", "#222222"
ORANGE_TINT = "#fbe3d1"


def panels(path):
    """The four image panels of the source figure, found from their black frames."""
    im = plt.imread(path)[..., :3]
    dark = im.mean(2) < 0.25
    cols = np.where(dark.mean(0) > 0.4)[0]           # frame verticals
    edges = np.split(cols, np.where(np.diff(cols) > 5)[0] + 1)
    # frames are thin lines; wide dark runs are the sign itself
    xs = [(e[0], e[-1]) for e in edges if e[-1] - e[0] <= 3]
    # horizontal frame lines: rows that are dark across the whole first panel
    rows = np.where(dark[:, xs[0][1]:xs[1][0]].mean(1) > 0.9)[0]
    top, bottom = rows.min(), rows.max()
    out = []
    for (l0, l1), (r0, r1) in zip(xs[0::2], xs[1::2]):
        out.append(im[top + 4:bottom - 3, l1 + 3:r0 - 2])
    return out


def box(ax, x, y, w, h, title, sub=None, focal=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=ORANGE_TINT if focal else "white",
                                ec=ORANGE if focal else INK, lw=2))
    ax.text(x + w / 2, y + h / 2 + (0.22 if sub else 0), title, ha="center",
            va="center", fontsize=17, fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.32, sub, ha="center", va="center",
                fontsize=13, color="#444444", linespacing=1.3)


def arrow(ax, x0, x1, y):
    ax.annotate("", xy=(x1, y), xytext=(x0, y),
                arrowprops=dict(arrowstyle="-|>", color=GREY, lw=2.2,
                                mutation_scale=20, shrinkA=0, shrinkB=0))


def main():
    sign, ig, ixg, sal = panels(SRC)
    fig, ax = plt.subplots(figsize=(16, 3.4))
    ax.set_xlim(0, 16)
    ax.set_ylim(0.6, 4.0)
    ax.axis("off")
    y = 2.1                                           # centre line

    # 1. sign image
    ax.imshow(sign, extent=(0.2, 2.0, y - 0.9, y + 0.9))
    ax.text(1.1, y - 1.25, "sign image", ha="center", fontsize=14, color=INK)

    # 2. classifier
    arrow(ax, 2.15, 2.65, y)
    box(ax, 2.7, y - 0.75, 2.1, 1.5, "Classifier", "ResNet-34\nreads: STOP")

    # 3. three maps
    arrow(ax, 4.95, 5.45, y)
    for i, (m, name) in enumerate([(ig, "IG"), (ixg, "IxG"), (sal, "Saliency")]):
        x0 = 5.5 + i * 1.12
        ax.imshow(m, extent=(x0, x0 + 1.0, y - 0.5, y + 0.5))
        ax.add_patch(plt.Rectangle((x0, y - 0.5), 1.0, 1.0, fill=False,
                                   ec=GREY, lw=1))
        ax.text(x0 + 0.5, y - 0.8, name, ha="center", fontsize=13, color=INK)
    ax.text(7.18, y + 0.75, "3 explanation maps", ha="center", fontsize=14,
            color=INK)

    # 4. 54 numbers
    arrow(ax, 8.95, 9.35, y)
    box(ax, 9.4, y - 0.75, 2.6, 1.5, "54 numbers",
        "shape of each map\n+ agreement")

    # 5. trees (the detector, focal)
    arrow(ax, 12.0, 12.4, y)
    box(ax, 12.45, y - 0.75, 1.7, 1.5, "Detector", "decision\ntrees", focal=True)

    # 6. verdict
    arrow(ax, 14.2, 14.6, y)
    ax.text(14.7, y + 0.25, "normal", ha="left", va="center", fontsize=16,
            color=INK)
    ax.text(14.7, y - 0.25, "or ATTACKED", ha="left", va="center",
            fontsize=16, fontweight="bold", color=ORANGE)

    # bracket over what the detector reads
    ax.plot([5.45, 5.45, 14.15, 14.15], [3.25, 3.45, 3.45, 3.25], color=ORANGE,
            lw=1.5)
    ax.text(9.8, 3.65, "the XAI detector reads the explanations, not the pixels",
            ha="center", fontsize=14, color=ORANGE)

    fig.savefig(os.path.join(OUT, "fig8_detector_pipeline.png"), dpi=200,
                bbox_inches="tight", facecolor="white")
    print("wrote fig8_detector_pipeline.png")


if __name__ == "__main__":
    main()
