"""Fine-tune a pretrained backbone on GTSRB to state-of-the-art accuracy.

Replaces the earlier 32x32 prototype CNN (94.2%). Target is >99% clean test
accuracy, which is the level a deployed sign reader would be expected to reach
and the level at which attack/defence results become representative.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe train_sota.py
"""
import os
import time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from model import SignNet, IMG_SIZE, BACKBONE, NUM_CLASSES

WORK = r"C:\Users\s4990998\xai_work"
DATA = os.path.join(WORK, "data")


def build_loaders(batch_size, workers, img_size=IMG_SIZE):
    # Colour is semantically meaningful on traffic signs (red = prohibition,
    # blue = mandatory), so saturation/hue jitter is kept mild.
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=10, translate=(0.10, 0.10), scale=(0.9, 1.1),
                                shear=5),
        transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15),
        transforms.ToTensor(),
    ])
    # No augmentation at test time, and no normalisation: the model does that.
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    train_ds = GTSRB(DATA, split="train", download=False, transform=train_tf)
    test_ds = GTSRB(DATA, split="test", download=False, transform=test_tf)

    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=workers, pin_memory=True, drop_last=True,
                          persistent_workers=workers > 0)
    test_ld = DataLoader(test_ds, batch_size=512, shuffle=False,
                         num_workers=workers, pin_memory=True,
                         persistent_workers=workers > 0)
    return train_ld, test_ld, len(train_ds), len(test_ds)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--img-size", type=int, default=IMG_SIZE)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(WORK, "gtsrb_sota.pt"))
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}  ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")

    train_ld, test_ld, n_tr, n_te = build_loaders(args.batch_size, args.workers, args.img_size)
    print(f"train images: {n_tr}   test images: {n_te}   backbone: {BACKBONE}  input: {args.img_size}px")

    model = SignNet(pretrained=True).to(device)
    model = model.to(memory_format=torch.channels_last)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr * 10, epochs=args.epochs, steps_per_epoch=len(train_ld))
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    best = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, running = time.time(), 0.0
        for x, y in train_ld:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                loss = crit(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            running += loss.item()

        acc = evaluate(model, test_ld, device)
        print(f"epoch {epoch:2d}/{args.epochs}  loss {running/len(train_ld):.4f}  "
              f"test acc {acc*100:.2f}%  ({time.time()-t0:.1f}s)")

        if acc > best:
            best = acc
            torch.save({
                "state_dict": model.state_dict(),
                "backbone": BACKBONE,
                "num_classes": NUM_CLASSES,
                "img_size": args.img_size,
                "test_acc": acc,
            }, args.out)

    print(f"\nbest clean test accuracy: {best*100:.2f}%")
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
