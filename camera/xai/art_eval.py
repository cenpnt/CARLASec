"""Rigorous evaluation of the XAI detector, using ART attacks only.

Two problems with the previous run, both standard traps:

 1. GRADIENT MASKING. The adaptive (gradient-based) attack did WORSE than the
    non-adaptive one, which is impossible for an attacker with strictly more
    information. Athalye et al. ("Obfuscated Gradients Give a False Sense of
    Security", ICML 2018) list exactly this as a symptom of masked gradients,
    not robustness. The standard test is to also run a GRADIENT-FREE attack:
    if a black-box attack beats the white-box one, gradients were the problem.
    ART ships SquareAttack (score-based, gradient-free) for this.

 2. DISTRIBUTION MISMATCH. The detector was trained only on low-frequency
    attacks and then tested against PGD, so it was out of distribution. Here it
    is trained on a MIXTURE of attack families, which is also what a deployed
    detector would face.

Methodology follows RobustBench/Tramer: run several attacks and score each
sample by the BEST attack against it, so a masked gradient cannot flatter the
defence.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe art_eval.py
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import (FastGradientMethod, ProjectedGradientDescent,
                                 AutoProjectedGradientDescent, SquareAttack)
from sklearn.metrics import roc_auc_score

from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import DATA, CKPT, DEVICE
from adaptive_attack import adaptive_lowfreq, predict
from art_adaptive import DiffXAIDetector, DefendedModel, batched_features

DETECT_CLASS = NUM_CLASSES          # index of the extra "adversarial" class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=800, help="images to train the detector")
    ap.add_argument("--n-eval", type=int, default=250, help="held-out images to attack")
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--pgd-iter", type=int, default=40)
    ap.add_argument("--square-iter", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()

    # Seed everything. Without this the detector head trains differently on every
    # run and the evaluation numbers move for no reason, which made two otherwise
    # identical runs disagree (PGD evasion 10% vs 3%).
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)

    model, ckpt = load_trained(CKPT, DEVICE)
    s = ckpt.get("img_size", IMG_SIZE)
    print(f"classifier: {ckpt.get('backbone')} @ {s}px, clean acc {ckpt['test_acc']*100:.2f}%\n",
          flush=True)

    tf = transforms.Compose([transforms.Resize((s, s)), transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)
    X, Y = [], []
    for a, b in DataLoader(ds, batch_size=512, num_workers=8):
        X.append(a); Y.append(b)
    X, Y = torch.cat(X), torch.cat(Y)
    keep = (predict(model, X) == Y).nonzero().squeeze(1)
    g = torch.Generator().manual_seed(0)
    keep = keep[torch.randperm(len(keep), generator=g)]
    tr, ev = keep[:args.n_train], keep[args.n_train:args.n_train + args.n_eval]
    Xtr, Ytr = X[tr], Y[tr]
    Xev, Yev = X[ev], Y[ev]
    print(f"detector training images: {len(Xtr)}   held-out eval images: {len(Xev)}", flush=True)

    plain = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(),
                              input_shape=(3, s, s), nb_classes=NUM_CLASSES,
                              clip_values=(0.0, 1.0),
                              device_type="gpu" if DEVICE == "cuda" else "cpu")

    # ---- detector training set: a MIXTURE of attack families ----------------
    print("building mixed-attack training set for the detector ...", flush=True)
    fam = {
        "FGSM": torch.from_numpy(FastGradientMethod(estimator=plain, eps=0.03,
                                                    batch_size=128).generate(x=Xtr.numpy())).float(),
        "PGD": torch.from_numpy(ProjectedGradientDescent(estimator=plain, eps=0.03,
                                                         eps_step=0.005, max_iter=20,
                                                         batch_size=128, verbose=False
                                                         ).generate(x=Xtr.numpy())).float(),
        "LowFreq": adaptive_lowfreq(model, Xtr, Ytr, args.eps, 3, 0.0,
                                    torch.ones(23, device=DEVICE), steps=150),
    }
    advs = []
    for name, Xa in fam.items():
        ok = (predict(model, Xa) != Ytr)
        print(f"  {name:<8} success {ok.float().mean()*100:5.1f}%", flush=True)
        advs.append(Xa[ok])
    Xadv_tr = torch.cat(advs)

    stub = DiffXAIDetector(model, torch.zeros(23), torch.ones(23)).to(DEVICE).eval()
    F_cln = batched_features(stub, Xtr)
    F_adv = batched_features(stub, Xadv_tr)
    fmean, fstd = F_cln.mean(0), F_cln.std(0).clamp(min=1e-6)

    det = DiffXAIDetector(model, fmean, fstd).to(DEVICE).eval()
    for p in det.base.parameters():
        p.requires_grad_(False)
    Ftr = ((torch.cat([F_cln, F_adv]).to(DEVICE) - fmean.to(DEVICE)) / fstd.to(DEVICE))
    ytr = torch.cat([torch.zeros(len(F_cln)), torch.ones(len(F_adv))]).to(DEVICE)[:, None]
    opt = torch.optim.Adam(det.head.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([len(F_cln) / max(1, len(F_adv))],
                                                         device=DEVICE))
    for _ in range(args.epochs):
        perm = torch.randperm(len(Ftr), device=DEVICE)
        for i in range(0, len(Ftr), 256):
            b = perm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            lossf(det.head(Ftr[b]), ytr[b]).backward()
            opt.step()
    print(f"  detector trained on {len(F_cln)} clean + {len(F_adv)} adversarial\n", flush=True)

    defended_model = DefendedModel(model, det).to(DEVICE).eval()
    defended = PyTorchClassifier(model=defended_model, loss=nn.CrossEntropyLoss(),
                                 input_shape=(3, s, s), nb_classes=NUM_CLASSES + 1,
                                 clip_values=(0.0, 1.0),
                                 device_type="gpu" if DEVICE == "cuda" else "cpu")

    # ---- evaluation ---------------------------------------------------------
    with torch.no_grad():
        sc_clean = torch.cat([det(Xev[i:i + 128].to(DEVICE)).squeeze(1).cpu()
                              for i in range(0, len(Xev), 128)])
    fpr = (sc_clean > 0).float().mean().item()
    print(f"false-positive rate on clean held-out images: {fpr*100:.1f}%\n", flush=True)

    # ALL attacks are TARGETED at a random wrong SIGN class. Untargeted attacks
    # on the defended model have a degenerate solution: they can raise the loss
    # by triggering the detector class, which the optimiser scores as success
    # but is a detection, not an evasion. Targeting a real sign class forces the
    # attacker to beat the classifier and the detector simultaneously.
    rng = np.random.default_rng(0)
    tgt = rng.integers(0, NUM_CLASSES, size=len(Xev))
    clash = tgt == Yev.numpy()
    tgt[clash] = (tgt[clash] + 1) % NUM_CLASSES
    y_plain = np.eye(NUM_CLASSES)[tgt]
    y_def = np.eye(NUM_CLASSES + 1)[tgt]

    print(f"{'attack':>34} | {'wrong sign':>11} {'flagged':>8} {'EVADED':>8}")
    print("-" * 68)
    evaded_any = torch.zeros(len(Xev), dtype=torch.bool)

    def report(name, Xa, track=True):
        nonlocal evaded_any
        pred = predict(model, Xa)
        with torch.no_grad():
            sc = torch.cat([det(Xa[i:i + 128].to(DEVICE)).squeeze(1).cpu()
                            for i in range(0, len(Xa), 128)])
        wrong = (pred != Yev)
        flagged = sc > 0
        ev_ = wrong & ~flagged
        if track:
            evaded_any |= ev_
        print(f"{name:>34} | {wrong.float().mean()*100:10.1f}% "
              f"{flagged.float().mean()*100:7.1f}% {ev_.float().mean()*100:7.1f}%", flush=True)
        return ev_

    xe = Xev.numpy()
    report("FGSM (non-adaptive)", torch.from_numpy(
        FastGradientMethod(estimator=plain, eps=args.eps, batch_size=64,
                           targeted=True).generate(x=xe, y=y_plain)).float())
    report("PGD (non-adaptive)", torch.from_numpy(
        ProjectedGradientDescent(estimator=plain, eps=args.eps, eps_step=args.eps / 8,
                                 max_iter=args.pgd_iter, batch_size=64, targeted=True,
                                 verbose=False).generate(x=xe, y=y_plain)).float())
    report("LowFreq (non-adaptive)",
           adaptive_lowfreq(model, Xev, Yev, args.eps, 3, 0.0,
                            torch.ones(23, device=DEVICE), steps=200))

    print("-" * 68)
    report("PGD vs defended (adaptive)", torch.from_numpy(
        ProjectedGradientDescent(estimator=defended, eps=args.eps, eps_step=args.eps / 8,
                                 max_iter=args.pgd_iter, batch_size=32, targeted=True,
                                 verbose=False).generate(x=xe, y=y_def)).float())
    report("APGD vs defended (adaptive)", torch.from_numpy(
        AutoProjectedGradientDescent(estimator=defended, eps=args.eps,
                                     eps_step=args.eps / 4, max_iter=args.pgd_iter,
                                     nb_random_init=1, batch_size=32, targeted=True,
                                     verbose=False).generate(x=xe, y=y_def)).float())
    print("   (gradient-free: immune to gradient masking)", flush=True)

    # SquareAttack takes an adv_criterion callable rather than a `targeted` flag,
    # which lets us encode EVASION exactly: misread as a wrong SIGN class and not
    # pushed into the detector class. That removes the degenerate solution
    # without needing a fixed target.
    def evasion_criterion(y_pred, y):
        p = np.argmax(y_pred, axis=1)
        t = np.argmax(y, axis=1)
        return (p != t) & (p != DETECT_CLASS)

    y_true_def = np.eye(NUM_CLASSES + 1)[Yev.numpy()]
    report("SquareAttack vs defended", torch.from_numpy(
        SquareAttack(estimator=defended, eps=args.eps, max_iter=args.square_iter,
                     nb_restarts=1, batch_size=32, adv_criterion=evasion_criterion,
                     verbose=False).generate(x=xe, y=y_true_def)).float())

    print("-" * 68)
    print(f"{'BEST-OF-ALL (per sample)':>34} | {'':>11} {'':>8} "
          f"{evaded_any.float().mean()*100:7.1f}%", flush=True)
    print("\nIf SquareAttack (gradient-free) evades more than PGD/APGD, the white-box")
    print("numbers were depressed by gradient masking and must not be believed.")


if __name__ == "__main__":
    main()
