"""Figures for the Background chapter of the proposal.

Two figures, both generated from this project's own model and data rather than
reproduced from a paper:

  fig_adversarial_examples.png
      A single GTSRB sign under FGSM at three budgets, with the model's own
      prediction and confidence, and the corresponding perturbations below.
      Illustrates Equation (2.1) and the epsilon tension: the perturbation
      that is easy to see is also easy to detect, and the one that is stealthy
      still flips a confident prediction.

  fig_attribution_methods.png
      Integrated Gradients, Input x Gradient and Saliency on the same clean
      decision. The three disagree, which is what the disagreement feature
      family in the detector is built on, and is also visible evidence for the
      Adebayo et al. caution about edge-detector-like behaviour.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u make_background_figures.py
"""
import os

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.datasets import GTSRB

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import FastGradientMethod

from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import attribute_all, WORK, DATA, CKPT, DEVICE

OUT = r"\\puffball.labs.eait.uq.edu.au\s4990998\Documents\REIT4842"

# Short display names; the full GTSRB names are too long for a panel title.
CLASS_NAMES = [
    "Speed limit 20", "Speed limit 30", "Speed limit 50", "Speed limit 60",
    "Speed limit 70", "Speed limit 80", "End speed limit 80", "Speed limit 100",
    "Speed limit 120", "No passing", "No passing >3.5t",
    "Right of way", "Priority road", "Yield", "Stop", "No vehicles",
    "No vehicles >3.5t", "No entry", "General caution", "Curve left",
    "Curve right", "Double curve", "Bumpy road", "Slippery road",
    "Road narrows", "Road work", "Traffic signals", "Pedestrians",
    "Children crossing", "Bicycles crossing", "Ice or snow",
    "Wild animals", "End all limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Straight or right", "Straight or left", "Keep right",
    "Keep left", "Roundabout", "End no passing", "End no passing >3.5t",
]

EPS = [0.005, 0.01, 0.03]
PREFERRED = [14, 17, 13]        # Stop, No entry, Yield: the safety-critical ones


def laplacian_energy(img):
    """Mean absolute Laplacian of the greyscale image: a roughness measure."""
    g = img.mean(0)
    lap = (g[1:-1, 2:] + g[1:-1, :-2] + g[2:, 1:-1] + g[:-2, 1:-1]
           - 4 * g[1:-1, 1:-1])
    return lap.abs().mean().item()


def pick_sample(model, clf, ds):
    """A confidently correct sample of a preferred class that FGSM actually flips.

    A figure illustrating adversarial examples is worthless if the prediction
    never changes, so success at the largest budget is a selection criterion.
    """
    atk = FastGradientMethod(estimator=clf, eps=EPS[-1], norm=np.inf)
    for want in PREFERRED:
        idx = [i for i in range(len(ds)) if ds._samples[i][1] == want]
        cand = []
        for i in idx:
            x, y = ds[i]
            # a dark or washed-out crop makes an unreadable figure
            lum = x.mean().item()
            if not 0.30 < lum < 0.70:
                continue
            with torch.no_grad():
                p = model(x[None].to(DEVICE)).softmax(1)[0]
            if p.argmax().item() != y or p.max().item() <= 0.85:
                continue
            xa = atk.generate(x=x.numpy()[None], y=np.array([y]))
            with torch.no_grad():
                q = model(torch.from_numpy(xa).to(DEVICE)).softmax(1)[0]
            if q.argmax().item() == y:
                continue
            cand.append((x.std().item(), i, x, y, q.argmax().item()))
        if cand:
            # highest contrast among the survivors: the most legible sign
            _, i, x, y, flip = max(cand, key=lambda c: c[0])
            print(f"using sample {i} of {len(cand)} candidates: "
                  f"{CLASS_NAMES[y]} flips to {CLASS_NAMES[flip]}")
            return x, y
    raise RuntimeError("no sample found that FGSM flips at the largest budget")


def fig_adversarial(model, clf, x, y):
    """Clean and FGSM rows: prediction above, perturbation below."""
    xs = [x.numpy()[None]]
    for e in EPS:
        atk = FastGradientMethod(estimator=clf, eps=e, norm=np.inf)
        xs.append(atk.generate(x=x.numpy()[None], y=np.array([y])))

    gmax = max(np.abs(xa[0] - x.numpy()).max() for xa in xs[1:])
    fig, ax = plt.subplots(2, 4, figsize=(11.5, 6.6))
    for j, xa in enumerate(xs):
        t = torch.from_numpy(xa)
        with torch.no_grad():
            p = model(t.to(DEVICE)).softmax(1)[0]
        pred, conf = p.argmax().item(), p.max().item()
        ok = pred == y

        ax[0, j].imshow(np.clip(xa[0].transpose(1, 2, 0), 0, 1))
        head = "Clean" if j == 0 else fr"FGSM $\varepsilon = {EPS[j-1]}$"
        ax[0, j].set_title(head, fontsize=12, pad=8)
        ax[0, j].set_xlabel(f"{CLASS_NAMES[pred]}\n{conf:.1%} confidence",
                            fontsize=10.5,
                            color="#1a7f37" if ok else "#c0392b")

        # perturbation, amplified about mid grey so sign is visible
        d = xa[0] - x.numpy()
        amp = np.clip(d / gmax * 0.5 + 0.5, 0, 1)
        ax[1, j].imshow(amp.transpose(1, 2, 0))
        if j == 0:
            ax[1, j].set_xlabel(
                "no perturbation\n"
                fr"roughness {laplacian_energy(x):.4f}",
                fontsize=10.5, color="#555555")
        else:
            rough = laplacian_energy(torch.from_numpy(xa[0]))
            ax[1, j].set_xlabel(
                fr"$\|\delta\|_\infty = {np.abs(d).max():.3f}$"
                "\n" fr"roughness {rough:.4f}",
                fontsize=10.5, color="#555555")

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([])
    ax[0, 0].set_ylabel("input", fontsize=12)
    ax[1, 0].set_ylabel("perturbation\n(amplified)", fontsize=12)

    fig.text(0.5, 0.015,
             "Roughness is the mean absolute Laplacian. Every FGSM perturbation "
             "raises it above the clean image, which is why a detector that uses "
             "no model at all can separate these images.",
             ha="center", fontsize=10, color="#333333", style="italic")

    plt.tight_layout(rect=[0, 0.05, 1, 1], h_pad=3.2)
    f = os.path.join(OUT, "fig_adversarial_examples.png")
    plt.savefig(f, dpi=130, bbox_inches="tight"); plt.close()
    print("wrote", f)


def fig_attributions(model, x, y):
    """IG, IxG and Saliency on the same clean decision."""
    with torch.no_grad():
        pred = model(x[None].to(DEVICE)).argmax(1)
    _, maps = attribute_all(model, x[None], pred)

    names = [("ig", "Integrated Gradients"),
             ("ixg", r"Input $\times$ Gradient"),
             ("sal", "Saliency")]

    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.9))
    ax[0].imshow(x.numpy().transpose(1, 2, 0))
    ax[0].set_title(f"Input: {CLASS_NAMES[y]}", fontsize=12, pad=8)

    flat = {}
    for k, _ in names:
        m = maps[k][0].sum(0).numpy()
        flat[k] = m / (np.abs(m).max() + 1e-12)

    for j, (k, label) in enumerate(names, start=1):
        ax[j].imshow(flat[k], cmap="seismic", vmin=-1, vmax=1)
        ax[j].set_title(label, fontsize=12, pad=8)

    # pairwise correlation, the quantity the disagreement features capture
    def corr(a, b):
        a, b = a.ravel(), b.ravel()
        return float(np.corrcoef(a, b)[0, 1])

    ax[1].set_xlabel(f"corr with IxG {corr(flat['ig'], flat['ixg']):.2f}",
                     fontsize=10.5, color="#555555")
    ax[2].set_xlabel(f"corr with Saliency {corr(flat['ixg'], flat['sal']):.2f}",
                     fontsize=10.5, color="#555555")
    ax[3].set_xlabel(f"corr with IG {corr(flat['sal'], flat['ig']):.2f}",
                     fontsize=10.5, color="#555555")

    for a in ax:
        a.set_xticks([]); a.set_yticks([])

    fig.text(0.5, 0.005,
             "Red is evidence for the predicted class, blue against it. All three "
             "explain the same decision on the same image, yet agree only loosely: "
             "Integrated Gradients concentrates on the sign face while the other two "
             "spread across the background. That disagreement is what the detector's "
             "second feature family measures.",
             ha="center", fontsize=10, color="#333333", style="italic")

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    f = os.path.join(OUT, "fig_attribution_methods.png")
    plt.savefig(f, dpi=130, bbox_inches="tight"); plt.close()
    print("wrote", f)


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    model, _ = load_trained(CKPT, DEVICE)
    tf = transforms.Compose([transforms.Resize((IMG_SIZE, IMG_SIZE)),
                             transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)

    clf = PyTorchClassifier(
        model=model, loss=torch.nn.CrossEntropyLoss(),
        input_shape=(3, IMG_SIZE, IMG_SIZE), nb_classes=NUM_CLASSES,
        clip_values=(0.0, 1.0), device_type="gpu" if DEVICE == "cuda" else "cpu",
    )

    x, y = pick_sample(model, clf, ds)
    fig_adversarial(model, clf, x, y)
    fig_attributions(model, x, y)


if __name__ == "__main__":
    main()
