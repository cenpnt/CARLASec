"""Architecture diagram and Gantt chart for the project proposal.

Dr Kim's proposal guidance asks for the proposed approach to be explained with a
figure showing the proposed architecture, and for the plan to be shown as a
Gantt-style chart. Both are generated here as PNGs for inclusion in the LaTeX
proposal.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe make_proposal_figures.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.dates as mdates
import datetime as dt

OUT = r"\\puffball.labs.eait.uq.edu.au\s4990998\Documents\REIT4842"

BUILT = "#2b6cb0"     # implemented and measured
PLAN = "#b7791f"      # planned work
REUSE = "#718096"     # reused from CARLASec
ATTACK = "#c53030"    # the threat


def box(ax, x, y, w, h, text, colour, style="solid", fs=8.5, fc=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                linewidth=1.6, edgecolor=colour,
                                facecolor=fc if fc else "white",
                                linestyle=style, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=4, color="#1a202c", linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, colour="#4a5568", style="-", lw=1.5):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=13, linewidth=lw,
                                 color=colour, linestyle=style, zorder=2,
                                 shrinkA=0, shrinkB=0))


def architecture():
    fig, ax = plt.subplots(figsize=(13.5, 6.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")

    # --- attack ---------------------------------------------------------
    box(ax, 1, 33, 17, 10,
        "Adversarial attack\n(low-frequency, smooth)\nAim 1",
        ATTACK, fc="#fff5f5", fs=9)
    arrow(ax, 9.5, 33, 9.5, 26.5, ATTACK)

    # --- perception -----------------------------------------------------
    box(ax, 1, 16, 17, 10, "Camera\n(CARLA / GTSRB)", REUSE, fs=9)
    arrow(ax, 18, 21, 23, 21)
    box(ax, 23, 16, 18, 10,
        "Sign classifier\nResNet-34, 96 px\n99.24% clean", BUILT, fs=9)

    # --- explanation ----------------------------------------------------
    arrow(ax, 41, 21, 46, 21)
    box(ax, 46, 14.5, 19, 13,
        "Explanation methods\nIntegrated Gradients\nInput x Gradient, Saliency",
        BUILT, fs=8.5)
    arrow(ax, 65, 21, 70, 21)
    box(ax, 70, 14.5, 18, 13,
        "Detector\nmap statistics +\ncross-method\ndisagreement", BUILT, fs=8.5)

    # control baseline
    box(ax, 46, 1.5, 19, 9,
        "Control detector\nimage statistics only\n(no model access)", REUSE,
        style=(0, (5, 3)), fs=8.5)
    arrow(ax, 32, 16, 32, 6, "#4a5568", style=(0, (4, 3)))
    arrow(ax, 32, 6, 46, 6, "#4a5568", style=(0, (4, 3)))
    arrow(ax, 65, 6, 79, 6, "#4a5568", style=(0, (4, 3)))
    arrow(ax, 79, 6, 79, 14.5, "#4a5568", style=(0, (4, 3)))
    ax.text(55.5, 11.6, "comparison baseline", fontsize=7.5,
            style="italic", color="#4a5568", ha="center")

    # --- fusion and response -------------------------------------------
    box(ax, 66, 33, 26, 10,
        "Noisy-OR risk score\nIRS = 1 - prod(1 - w r)\nAim 2", PLAN,
        style=(0, (5, 3)), fs=9)
    arrow(ax, 79, 27.5, 79, 33, PLAN, style=(0, (4, 3)))
    arrow(ax, 92, 38, 93.4, 38, PLAN, style=(0, (4, 3)))
    box(ax, 93.5, 30.5, 5.6, 15,
        "Safety\nstate\nmachine\n\nNormal\nObserve\nSafety\nMRC", REUSE, fs=7.5)

    ax.text(50, 46.4, "Proposed architecture", fontsize=13,
            ha="center", weight="bold")

    handles = [
        plt.Line2D([], [], color=BUILT, lw=2.4, label="Implemented and measured"),
        plt.Line2D([], [], color=PLAN, lw=2.4, ls="--", label="Planned (Aims 2 and 3)"),
        plt.Line2D([], [], color=REUSE, lw=2.4, label="Reused (CARLASec / control)"),
        plt.Line2D([], [], color=ATTACK, lw=2.4, label="Adversarial input (Aim 1)"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5,
              frameon=False, ncol=2, bbox_to_anchor=(0.20, 0.115))

    plt.tight_layout()
    p = os.path.join(OUT, "fig_architecture.png")
    plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
    print("saved", p)


def gantt():
    tasks = [
        ("Project proposal",                       "2026-09-01", "2026-09-17", PLAN),
        ("Complete adaptive evaluation",           "2026-09-10", "2026-10-10", BUILT),
        ("Progress seminar",                       "2026-10-01", "2026-10-16", PLAN),
        ("Randomised multi-method detector (Aim 2)", "2026-10-10", "2026-11-25", BUILT),
        ("Re-run adaptive attacker on new defence", "2026-11-05", "2026-12-05", BUILT),
        ("CARLASec integration: sign in control loop", "2026-11-01", "2026-12-15", ATTACK),
        ("Summer break (buffer)",                  "2026-12-15", "2027-02-01", REUSE),
        ("Thesis plan submission",                 "2027-03-01", "2027-03-19", PLAN),
        ("World-space patch + consequence metrics", "2027-02-01", "2027-04-05", ATTACK),
        ("Live attribution visualisation",         "2027-03-01", "2027-04-20", BUILT),
        ("Final evaluation",                       "2027-04-01", "2027-05-05", BUILT),
        ("Thesis writing (due 24 May)",            "2027-04-05", "2027-05-24", PLAN),
        ("Poster, demo and 3MP",                   "2027-05-24", "2027-06-11", PLAN),
    ]
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ypos, ylab = [], []
    for i, (name, s, e, c) in enumerate(tasks):
        s = dt.datetime.strptime(s, "%Y-%m-%d")
        e = dt.datetime.strptime(e, "%Y-%m-%d")
        y = len(tasks) - i
        ax.barh(y, (e - s).days, left=s, height=0.6, color=c, alpha=0.9,
                edgecolor="white")
        ypos.append(y); ylab.append(name)

    for d, lab in [("2026-09-17", "Proposal"), ("2027-05-24", "Thesis"),
                   ("2027-06-11", "Demo")]:
        x = dt.datetime.strptime(d, "%Y-%m-%d")
        ax.axvline(x, color="#c53030", ls=":", lw=1.4, zorder=0)
        ax.text(x, len(tasks) + 0.75, lab, rotation=0, fontsize=8,
                color="#c53030", ha="center")

    # task names on the axis, so long labels cannot overflow short bars
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylab, fontsize=8.5)
    ax.set_ylim(0.2, len(tasks) + 1.4)
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b\n%Y"))
    ax.grid(axis="x", alpha=0.25)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    plt.title("Project schedule against assessment deadlines", fontsize=12)
    plt.tight_layout()
    p = os.path.join(OUT, "fig_gantt.png")
    plt.savefig(p, dpi=200, bbox_inches="tight"); plt.close()
    print("saved", p)


if __name__ == "__main__":
    architecture()
    gantt()
