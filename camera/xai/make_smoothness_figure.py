"""Show what 'smooth' actually looks like as k is swept.

One sign, attacked five ways at the same budget: full-resolution PGD, then the
low-frequency attack at k = 6, 4, 3, 2. Three rows:

  1. the adversarial image      (all five are misread by the classifier)
  2. the perturbation           (shared amplitude scale, amplified about grey)
  3. the perturbation Laplacian (what the pixel-statistics control measures)

The third row is the argument: it goes from dense static to black while the
attack keeps working.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u make_smoothness_figure.py
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import ProjectedGradientDescent

from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import WORK, DATA, CKPT, DEVICE
from lowfreq_confirm import lowfreq_strong
from make_background_figures import CLASS_NAMES, laplacian_energy

OUT = r"\\puffball.labs.eait.uq.edu.au\s4990998\Documents\REIT4842"

EPS = 0.03
KS = [6, 4, 3, 2]
POOL = 512          # candidate images to attack before choosing one


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    model, _ = load_trained(CKPT, DEVICE)
    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)
    X, Y = next(iter(DataLoader(ds, batch_size=POOL, shuffle=True)))

    # keep only what the model already gets right
    with torch.no_grad():
        pred = model(X.to(DEVICE)).argmax(1).cpu()
    keep = pred == Y

    # Drop near-saturated images. Where the clean image sits against 0 or 1 the
    # perturbation is clipped, and the clip edge is a hard discontinuity that
    # dominates the Laplacian no matter how smooth the attack is. On such an
    # image roughness measures the clipping, not the attack.
    sat = ((X > 0.97) | (X < 0.03)).float().mean((1, 2, 3))
    keep &= sat < 0.02
    X, Y = X[keep], Y[keep]
    print(f"{len(X)} correctly classified, unsaturated candidates")

    clf = PyTorchClassifier(
        model=model, loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, IMG_SIZE, IMG_SIZE), nb_classes=NUM_CLASSES,
        clip_values=(0.0, 1.0), device_type="gpu" if DEVICE == "cuda" else "cpu",
    )

    advs, labels = {}, []
    pgd = ProjectedGradientDescent(estimator=clf, eps=EPS, eps_step=EPS / 8,
                                   max_iter=40, num_random_init=1, verbose=False)
    advs["PGD"] = torch.from_numpy(pgd.generate(x=X.numpy(), y=Y.numpy()))
    labels.append(("PGD", "PGD, full resolution"))
    for k in KS:
        print(f"low-frequency k={k} ...")
        advs[k] = lowfreq_strong(model, X, Y, eps=EPS, k=k)
        labels.append((k, f"low-frequency, $k={k}$"))

    # a sample every attack succeeds on, and that is legible
    ok = torch.ones(len(X), dtype=torch.bool)
    for key in advs:
        with torch.no_grad():
            p = model(advs[key].to(DEVICE)).argmax(1).cpu()
        ok &= p != Y
    idx = torch.nonzero(ok).squeeze(1)
    if len(idx) == 0:
        raise RuntimeError("no image was flipped by every attack; enlarge POOL")
    i = idx[X[idx].std((1, 2, 3)).argmax()].item()   # highest contrast
    print(f"using index {i}: {CLASS_NAMES[Y[i].item()]}, "
          f"{len(idx)} images flipped by all five attacks")

    xc = X[i]
    cols = [(advs[key][i], txt) for key, txt in labels]
    gmax = max(np.abs((xa - xc).numpy()).max() for xa, _ in cols)

    fig, ax = plt.subplots(3, len(cols), figsize=(3.0 * len(cols), 9.3))
    for j, (xa, txt) in enumerate(cols):
        with torch.no_grad():
            p = model(xa[None].to(DEVICE)).softmax(1)[0]

        ax[0, j].imshow(xa.numpy().transpose(1, 2, 0))
        ax[0, j].set_title(txt, fontsize=12, pad=8)
        ax[0, j].set_xlabel(f"read as {CLASS_NAMES[p.argmax().item()]}",
                            fontsize=10.5, color="#c0392b")

        d = (xa - xc).numpy()
        ax[1, j].imshow(np.clip(d / gmax * 0.5 + 0.5, 0, 1).transpose(1, 2, 0))
        ax[1, j].set_xlabel(fr"$\|\delta\|_\infty = {np.abs(d).max():.3f}$",
                            fontsize=10.5, color="#555555")

        g = d.mean(0)
        lap = np.abs(g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1]
                     - 4 * g[1:-1, 1:-1])
        ax[2, j].imshow(lap, cmap="inferno", vmin=0, vmax=0.06)
        ax[2, j].set_xlabel(f"mean |Laplacian| = {lap.mean():.5f}",
                            fontsize=10.5, color="#555555")

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("adversarial image", fontsize=12)
    ax[1, 0].set_ylabel("perturbation\n(amplified)", fontsize=12)
    ax[2, 0].set_ylabel("roughness\n(Laplacian)", fontsize=12)

    fig.text(0.5, 0.012,
             "All five attacks use the same budget and all five fool the classifier. "
             "As $k$ falls the perturbation becomes a smooth gradient and its "
             "Laplacian goes dark, which is the signal the image-statistics control "
             "detector depends on.",
             ha="center", fontsize=11, color="#333333", style="italic")

    plt.tight_layout(rect=[0, 0.04, 1, 1], h_pad=3.0)
    f = os.path.join(OUT, "fig_smoothness_sweep.png")
    plt.savefig(f, dpi=130, bbox_inches="tight"); plt.close()
    print("wrote", f)


if __name__ == "__main__":
    main()
