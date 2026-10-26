"""Meeting figures asked for by Dr Kim (week 13):
  fig10_progress_pipeline.png   thesis pipeline with a tick on every finished step
  fig11_experiment_io.png       one experiment as input -> our program -> output
  fig12_carlasec_framework.png  the CARLASec framework with plug-in choices

Only fig11 needs the model (two thumbnails and the dataset counts, cached in
WORK/fig11_thumbs_adv2.pt after the first run). Edit STEPS to move a tick.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe make_week13_figures.py
"""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "figures")
WORK = r"C:\Users\s4990998\xai_work"
GREY, ORANGE, INK, SOFT = "#8c8c8c", "#d95f02", "#222222", "#555555"
ORANGE_TINT, GREEN, PANEL = "#fbe3d1", "#1b9e77", "#f3f3f3"
DONE, NOW, TODO = "done", "now", "todo"

# (lane, [(title, detail, status)])
STEPS = [
    ("Aim 1\nAttack", [
        ("Sign model", "ResNet-34, 99.2%", DONE),
        ("XAI detector", "3 maps, 54 features", DONE),
        ("ADV$^2$ reproduced", "46% evasion", DONE),
        ("Better optimiser", "76% evasion", DONE),
        ("Ablations", "Aim 1 closed", DONE),
    ]),
    ("Aim 2\nDefence", [
        ("Prior-art check", "45 papers verified", DONE),
        ("Reproduce AdvEdge", "published detector", NOW),
        ("Baseline number", "vs ADV$^2$, ours", TODO),
        ("Improve it", "train on our ADV$^2$", TODO),
        ("Adaptive attack", "on the improved one", TODO),
    ]),
    ("Aim 3\nCARLASec", [
        ("Signs in CARLA", "crop from CARLA boxes", TODO),
        ("Plug-in framework", "pick or add your own", TODO),
        ("Closed-loop runs", "reuse CARLASec+ CAN", TODO),
    ]),
    ("Paper\n(attack)", [
        ("Adam vs tanh", "ablation", TODO),
        ("Model zoo on HPC", "8 backbones", TODO),
        ("Other datasets", "LISA, Mapillary", TODO),
    ]),
]


def box(ax, x, y, w, h, title, sub=None, style="plain", fs=13, sfs=10.5):
    fc, ec, ls = {"plain": ("white", INK, "-"), "focal": (ORANGE_TINT, ORANGE, "-"),
                  "todo": ("white", GREY, (0, (4, 3))),
                  "panel": (PANEL, "#cfcfcf", "-")}[style]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=fc, ec=ec, lw=1.8, ls=ls))
    if title:
        ax.text(x + w / 2, y + h / 2 + (0.17 if sub else 0), title, ha="center",
                va="center", fontsize=fs, fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.22, sub, ha="center", va="center",
                fontsize=sfs, color=SOFT)


def arrow(ax, x0, y0, x1, y1, lw=1.8, ms=16, color=GREY):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                                mutation_scale=ms, shrinkA=0, shrinkB=0))


def status(ax, x, y, s, r=0.17):
    if s == DONE:
        ax.add_patch(Circle((x, y), r, fc=GREEN, ec=GREEN, zorder=5))
        ax.text(x, y - 0.01, "\u2713", ha="center", va="center", fontsize=12,
                color="white", fontweight="bold", zorder=6)
    elif s == NOW:
        ax.add_patch(Circle((x, y), r, fc="white", ec=ORANGE, lw=2, zorder=5))
        ax.add_patch(Circle((x, y), r * 0.45, fc=ORANGE, ec=ORANGE, zorder=6))
    else:
        ax.add_patch(Circle((x, y), r, fc="white", ec=GREY, lw=1.6, zorder=5))


def progress():
    w, gap, h, lane_h, x0 = 2.9, 0.4, 1.0, 1.6, 2.3
    n_lanes = len(STEPS)
    n_cols = max(len(st) for _, st in STEPS)
    fig, ax = plt.subplots(figsize=(3.3 * n_cols + 2.7, 1.6 * n_lanes + 1.6))
    ax.set_xlim(0, x0 + n_cols * (w + gap) + 0.1)
    ax.set_ylim(-1.0, n_lanes * lane_h + 0.3)
    ax.axis("off")
    for li, (lane, steps) in enumerate(STEPS):
        y = (n_lanes - 1 - li) * lane_h + 0.2
        ax.text(0.1, y + h / 2, lane, ha="left", va="center", fontsize=14,
                fontweight="bold", color=INK, linespacing=1.25)
        for si, (title, sub, s) in enumerate(steps):
            x = x0 + si * (w + gap)
            box(ax, x, y, w, h, title, sub,
                {DONE: "plain", NOW: "focal", TODO: "todo"}[s])
            status(ax, x + 0.02, y + h + 0.02, s)
            if si:
                arrow(ax, x - gap + 0.04, y + h / 2, x - 0.04, y + h / 2)
        if li < n_lanes - 1:
            ax.plot([0.1, ax.get_xlim()[1] - 0.1], [y - 0.28] * 2, color="#e2e2e2", lw=1)
    done = sum(s == DONE for _, st in STEPS for *_, s in st)
    total = sum(len(st) for _, st in STEPS)
    lx = x0
    for s, label in [(DONE, "done"), (NOW, "next up"), (TODO, "to do")]:
        status(ax, lx, -0.6, s)
        ax.text(lx + 0.3, -0.6, label, va="center", fontsize=12, color=INK)
        lx += 3.3
    ax.text(ax.get_xlim()[1] - 0.1, -0.6, f"{done} of {total} steps done",
            ha="right", va="center", fontsize=13, fontweight="bold", color=INK)
    save(fig, "fig10_progress_pipeline.png")


def thumbs():
    """Clean sign, the same sign after our ADV2 (digital, as attacked), and the
    dataset counts. Same sign as fig9 (seed 0 split, index 8)."""
    import torch
    cache = os.path.join(WORK, "fig11_thumbs_adv2.pt")
    if os.path.exists(cache):
        return torch.load(cache, weights_only=False)     # our own file, numpy arrays
    from model import load_trained, IMG_SIZE
    from xai_detect import CKPT, DEVICE
    from aim1_attack import load_data, predict
    from make_figures import NAMES
    plain, ckpt = load_trained(CKPT, DEVICE)
    X, Y = load_data(plain, ckpt.get("img_size", IMG_SIZE), 1000, seed=0)
    Xa, Ya = X[500:628], Y[500:628]
    A_ours, _ = torch.load(os.path.join(WORK, "fig9_cache.pt"))
    i = 8
    read = NAMES[int(predict(plain, A_ours[i:i + 1]))]
    from torchvision.datasets import GTSRB
    n_test = len(GTSRB(os.path.join(WORK, "data"), split="test", download=False))
    n_train = len(GTSRB(os.path.join(WORK, "data"), split="train", download=False))
    out = {"clean": Xa[i].permute(1, 2, 0).numpy(), "adv": A_ours[i].permute(1, 2, 0).numpy(),
           "true": NAMES[int(Ya[i])], "read": read, "n_test": n_test,
           "n_train": n_train, "acc": float(ckpt["test_acc"])}
    torch.save(out, cache)
    return out


def experiment_io(cost=None):
    t = thumbs()
    n_ok = round(t["acc"] * t["n_test"])
    fig, ax = plt.subplots(figsize=(20, 8.4))
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 8.4)
    ax.axis("off")
    cols = [(0.1, 5.4, "INPUT"), (6.3, 7.4, "OUR PROGRAM"), (14.6, 5.3, "OUTPUT")]
    for x, w, head in cols:
        box(ax, x, 0.1, w, 7.6, None, style="panel")
        ax.text(x + w / 2, 7.95, head, ha="center", va="center", fontsize=16,
                fontweight="bold", color=INK)
    arrow(ax, 5.6, 4.0, 6.2, 4.0, lw=3, ms=26, color=INK)
    arrow(ax, 13.8, 4.0, 14.5, 4.0, lw=3, ms=26, color=INK)

    def lines(x, y, rows, fs=12.5, dy=0.42):
        for r in rows:
            bold = r.startswith("*")
            ax.text(x, y, r.lstrip("*"), fontsize=fs, va="top", color=INK,
                    fontweight="bold" if bold else "normal")
            y -= dy
        return y

    # input column
    ax.imshow(t["clean"], extent=(1.85, 3.65, 5.55, 7.35), zorder=3)
    ax.text(2.75, 5.35, f"clean sign: {t['true']}", ha="center", fontsize=11, color=SOFT)
    lines(0.35, 4.85, [
        "*Data: GTSRB (43 sign classes)",
        f"test set {t['n_test']:,} signs",
        f"model reads {n_ok:,} correctly",
        f"random 1,000 of them ({1000 / n_ok:.0%}) per split",
        "  500 train the detector",
        "  500 are attacked",
        "3 splits (seeds 0, 1, 2)",
        "*Model: ResNet-34, 96 px",
        f"trained on {t['n_train']:,} training signs",
        f"clean accuracy {t['acc']:.2%}",
    ], dy=0.41)

    # program column: three numbered steps
    steps = [
        ("1  Train the XAI detector",
         ["500 clean + 2,000 attacked signs (CW x3, FGSM)",
          "IG, IxG, Saliency maps -> 54 features",
          "boosted trees, alarm threshold at 5% false alarms"]),
        ("2  Attack the other 500 signs (eps = 0.03)",
         ["goal: misread, keep all 3 maps like the clean ones",
          "published ADV$^2$: sign-PGD, 200 + 200 steps",
          "ours: Adam on tanh, 100 steps x 2 restarts"]),
        ("3  Score each attacked sign",
         ["the detector never sees the attack (blind)",
          "success = misread AND no alarm",
          "same 500 signs for every attack"]),
    ]
    y = 7.45
    for si, (title, rows) in enumerate(steps):
        box(ax, 6.55, y - 2.1, 6.9, 2.0, None, style="focal" if si == 1 else "plain")
        lines(6.75, y - 0.2, ["*" + title] + rows, fs=12.5, dy=0.44)
        if si < 2:
            arrow(ax, 10.0, y - 2.12, 10.0, y - 2.38)
        y -= 2.45

    # output column
    ax.imshow(t["adv"], extent=(16.35, 18.15, 5.55, 7.35), zorder=3)
    ax.text(17.25, 5.35, f"our ADV$^2$, read as: {t['read']}", ha="center",
            fontsize=11, color=SOFT)
    y = lines(14.85, 4.85, ["*Evade the detector, mean of 3 splits:"], dy=0.41)
    for name, val, bold in [("plain CW (no map term)", "13%", False),
                            ("published ADV$^2$", "46%", False),
                            ("our ADV$^2$", "76%", True)]:
        for x, s, ha in [(14.85, name, "left"), (19.0, val, "right")]:
            ax.text(x, y, s, fontsize=12.5, va="top", ha=ha, color=ORANGE if bold else INK,
                    fontweight="bold" if bold else "normal")
        y -= 0.41
    rows = ["*Misread at all:", "published 97%, ours 86%",
            "same objective and detector;", "only the optimiser differs"]
    if cost:
        rows += ["*Cost per sign (RTX 4080 SUPER):",
                 f"ours {cost[0]:.1f} s, published {cost[1]:.1f} s"]
    lines(14.85, y - 0.12, rows, dy=0.41)
    save(fig, "fig11_experiment_io.png")


# (stage, subtitle, options): "*" exists today, "+" a file the user adds,
# "~" continues the line above, no mark = to build
FRAMEWORK = [
    ("Scenario", "CARLA world",
     ["*CARLA + CARLASec+", "scenario config file", "~town, weather, sign spots"]),
    ("Attack", "plug-in slot",
     ["*none (clean run)", "*FGSM, PGD, CW", "*ADV$^2$ (published)", "*ADV$^2$ (ours)",
      "+attacks/your_attack.py"]),
    ("Sign crop", "camera -> 96 px",
     ["*RGB camera, 20 FPS", "crop at CARLA's own", "~sign boxes (no detector)"]),
    ("Classifier", "plug-in slot",
     ["*ResNet-34 (GTSRB)", "+models/your_model.py"]),
    ("Defence", "plug-in slot",
     ["*none", "*XAI detector (ours)", "AdvEdge detector (Aim 2)",
      "+defences/your_defence.py"]),
    ("Response", "CARLASec+ CAN",
     ["*IDS alarm on CAN bus", "*RSS braking, safe state"]),
]
METRICS = ("Logged every run: misread rate, attack success, detection rate, "
           "false alarms, latency per frame (ms), sign violations")


def framework():
    n, w, gap = len(FRAMEWORK), 3.25, 0.38
    fig, ax = plt.subplots(figsize=(22, 6.6))
    ax.set_xlim(0, n * w + (n - 1) * gap + 0.2)
    ax.set_ylim(-1.3, 5.9)
    ax.axis("off")
    for i, (stage, sub, opts) in enumerate(FRAMEWORK):
        x = 0.1 + i * (w + gap)
        slot = sub == "plug-in slot"
        box(ax, x, 4.4, w, 1.2, stage, sub, "focal" if slot else "plain", fs=15, sfs=12)
        if i:
            arrow(ax, x - gap + 0.04, 5.0, x - 0.04, 5.0, lw=2.2, ms=20, color=INK)
        box(ax, x, 0.35, w, 3.8, None, style="panel")
        y = 3.75
        for o in opts:
            kind = o[0] if o[0] in "*+~" else ""
            if kind == "~":
                y += 0.28           # continuation: tuck under the line above
            mark, col, tcol, font = {"*": ("\u25CF", GREEN, INK, None),
                                     "+": ("+", ORANGE, INK, "monospace"),
                                     "~": ("", GREY, SOFT, None),
                                     "": ("\u25CB", GREY, SOFT, None)}[kind]
            ax.text(x + 0.22, y, mark, fontsize=12 if kind == "+" else 11,
                    fontweight="bold" if kind == "+" else "normal",
                    color=col, va="center")
            ax.text(x + 0.5, y, o.lstrip("*+~"), fontsize=10.5 if font else 11.5,
                    va="center", color=tcol, family=font or "sans-serif")
            y -= 0.66
    xr = ax.get_xlim()[1]
    box(ax, 0.1, -0.75, xr - 0.2, 0.62, None, style="plain")
    ax.text(xr / 2, -0.44, METRICS, ha="center", va="center", fontsize=13, color=INK)
    lx = 0.1
    for mark, col, label in [("\u25CF", GREEN, "exists in our code or CARLASec+"),
                             ("\u25CB", GREY, "to build"),
                             ("+", ORANGE, "users add their own: one Python file in that folder")]:
        ax.text(lx, -1.1, mark, fontsize=12, color=col, va="center",
                fontweight="bold" if mark == "+" else "normal")
        ax.text(lx + 0.3, -1.1, label, fontsize=12, color=SOFT, va="center")
        lx += 0.6 + 0.135 * len(label)
    ax.text(xr - 0.1, -1.1, "orange box = plug-in slot, picked in the config file",
            ha="right", fontsize=12, color=SOFT, va="center")
    save(fig, "fig12_carlasec_framework.png")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    import sys
    progress()
    framework()
    if "--no-io" not in sys.argv:
        # attack time per sign, measured by time_eot.py (rows with no views)
        tp = os.path.join(WORK, "time_eot.npz")
        cost = None
        if os.path.exists(tp):
            sec = {r[0]: r[2] for r in np.load(tp, allow_pickle=True)["rows"] if r[1] == 0}
            cost = (sec["ADV2 ours"], sec["ADV2 published"])
        experiment_io(cost)
