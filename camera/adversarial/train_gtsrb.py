"""
Step 1 of the adversarial thread: train a small CNN traffic-sign classifier on
GTSRB (German Traffic Sign Recognition Benchmark). This is the neural network we
will later attack with adversarial perturbations and defend with XAI.

Downloads GTSRB via torchvision, trains a small CNN, reports test accuracy, and
saves the model to gtsrb_cnn.pt.

Run:  python train_gtsrb.py   (venv active). CPU is fine, just slower.
"""
import os, time
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
MODEL_PATH = os.path.join(HERE, "gtsrb_cnn.pt")
EPOCHS = 8
BATCH = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tfm = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.34, 0.31, 0.32), (0.27, 0.26, 0.27)),  # GTSRB-ish stats
])

class TrafficSignNet(nn.Module):
    def __init__(self, n_classes=43):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1);  self.c2 = nn.Conv2d(32, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1); self.c4 = nn.Conv2d(64, 64, 3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 256);     self.fc2 = nn.Linear(256, n_classes)
        self.drop = nn.Dropout(0.3)
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c2(F.relu(self.c1(x)))), 2)   # 32->16
        x = F.max_pool2d(F.relu(self.c4(F.relu(self.c3(x)))), 2)   # 16->8
        x = x.flatten(1)
        x = self.drop(F.relu(self.fc1(x)))
        return self.fc2(x)

def evaluate(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            correct += (model(xb).argmax(1) == yb).sum().item(); total += yb.size(0)
    return correct / total

def main():
    os.makedirs(DATA, exist_ok=True)
    print("Loading GTSRB (downloads on first run)...", flush=True)
    train_ds = GTSRB(DATA, split="train", download=True, transform=tfm)
    test_ds  = GTSRB(DATA, split="test",  download=True, transform=tfm)
    print(f"train={len(train_ds)}  test={len(test_ds)}  device={device}", flush=True)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    test_dl  = DataLoader(test_ds,  batch_size=BATCH, shuffle=False, num_workers=0)

    model = TrafficSignNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = nn.CrossEntropyLoss()

    for ep in range(1, EPOCHS + 1):
        model.train(); t0 = time.time(); running = 0.0
        for i, (xb, yb) in enumerate(train_dl):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = lossf(model(xb), yb); loss.backward(); opt.step()
            running += loss.item()
            if (i + 1) % 100 == 0:
                print(f"  ep{ep} batch {i+1}/{len(train_dl)} loss {running/(i+1):.3f}", flush=True)
        acc = evaluate(model, test_dl)
        print(f"Epoch {ep}/{EPOCHS}  train_loss {running/len(train_dl):.3f}  test_acc {acc:.4f}  ({time.time()-t0:.0f}s)", flush=True)

    torch.save({"state_dict": model.state_dict(), "n_classes": 43}, MODEL_PATH)
    print(f"Saved model to {MODEL_PATH}", flush=True)

if __name__ == "__main__":
    main()
