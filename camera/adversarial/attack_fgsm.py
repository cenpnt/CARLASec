"""
Step 2 of the adversarial thread: FGSM attack on the GTSRB traffic-sign classifier.
Matches the baseline paper's setup (GTSRB, 32x32 CNN, FGSM, eps 0.01-0.10).

FGSM is done in [0,1] pixel space (normalization folded into the forward pass), so
the perturbation budget epsilon is directly comparable to the paper. Reports clean
vs adversarial accuracy and attack success rate per epsilon, and saves a montage
(clean | adversarial | amplified perturbation) to show near-invisibility.

Run:  python attack_fgsm.py   (venv active; gtsrb_cnn.pt must exist)
"""
import os
import numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB
from train_gtsrb import TrafficSignNet   # safe import (defines only; no main run)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(HERE, "gtsrb_cnn.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = torch.tensor([0.34, 0.31, 0.32]).view(1, 3, 1, 1)
STD  = torch.tensor([0.27, 0.26, 0.27]).view(1, 3, 1, 1)
EPSILONS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10]

# dataset in [0,1] pixel space (no normalization here; done in forward)
tfm = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

class Normalized(nn.Module):
    """Wrap the classifier so it takes [0,1] images and normalizes internally,
    which lets FGSM perturb real pixels within a [0,1] budget."""
    def __init__(self, model):
        super().__init__(); self.model = model
        self.register_buffer("mean", MEAN); self.register_buffer("std", STD)
    def forward(self, x):
        return self.model((x - self.mean) / self.std)

def fgsm(model, x, y, eps):
    if eps == 0:
        return x
    x = x.clone().detach().requires_grad_(True)
    loss = nn.functional.cross_entropy(model(x), y)
    loss.backward()
    x_adv = x + eps * x.grad.sign()
    return x_adv.clamp(0, 1).detach()

def main():
    ckpt = torch.load(MODEL_PATH, map_location=device)
    base = TrafficSignNet(ckpt.get("n_classes", 43)); base.load_state_dict(ckpt["state_dict"])
    model = Normalized(base).to(device).eval()

    test_ds = GTSRB(DATA, split="test", download=False, transform=tfm)
    # use a fixed subset for speed/repeatability
    idx = list(range(0, len(test_ds), max(1, len(test_ds)//2000)))[:2000]
    dl = DataLoader(Subset(test_ds, idx), batch_size=128, shuffle=False, num_workers=0)

    print(f"FGSM on {len(idx)} GTSRB test images (device={device})\n", flush=True)
    print(f"{'eps':>6} {'accuracy':>10} {'attack_success':>16}")
    saved_example = False
    for eps in EPSILONS:
        correct = total = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            xadv = fgsm(model, xb, yb, eps)
            with torch.no_grad():
                pred = model(xadv).argmax(1)
            correct += (pred == yb).sum().item(); total += yb.size(0)
            # save one visual example at a mid epsilon
            if eps == 0.05 and not saved_example:
                clean = (xb[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)[..., ::-1]
                adv   = (xadv[0].cpu().numpy().transpose(1,2,0)*255).astype(np.uint8)[..., ::-1]
                pert  = np.clip((xadv[0]-xb[0]).cpu().numpy().transpose(1,2,0)*10*255+127,0,255).astype(np.uint8)[..., ::-1]
                big = lambda im: cv2.resize(im,(160,160),interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(HERE,"fgsm_example.png"), np.hstack([big(clean),big(adv),big(pert)]))
                saved_example = True
        acc = correct/total
        print(f"{eps:>6.2f} {acc*100:>9.2f}% {(1-acc)*100:>15.2f}%", flush=True)
    print("\nSaved fgsm_example.png  [clean | adversarial | perturbation x10]", flush=True)

if __name__ == "__main__":
    main()
