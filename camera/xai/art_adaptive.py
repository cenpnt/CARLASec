"""Adaptive attack the RIGHT way: ART's DetectorClassifier + ART's PGD.

My hand-written adaptive attack was broken (diag_adaptive.py showed the evasion
term INCREASED the attribution distance it was supposed to minimise). Rather
than debug a bespoke optimiser, use the library.

ART 1.20 ships `DetectorClassifier`, which is literally the construction from
Carlini and Wagner, "Adversarial Examples Are Not Easily Detected"
(arXiv:1705.07263), the paper that broke ten detectors. It wraps a classifier
and a detector into ONE model with nb_classes+1 outputs, where the extra class
means "adversarial":

    logit_{n+1} = (D(x) + 1) * max_i Z_i(x)

Any standard ART attack against that combined model must therefore misclassify
the sign AND keep the detector quiet, which is exactly an adaptive attack, with
no custom loss or optimiser of ours involved.

To plug in, our detector has to be a differentiable network with ONE output
(positive = adversarial). So the attribution features are computed inside an
nn.Module (Saliency and Input x Gradient via create_graph), followed by a small
MLP head trained on clean vs non-adaptive adversarial examples.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe art_adaptive.py
"""
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from art.estimators.classification import PyTorchClassifier, DetectorClassifier
from art.attacks.evasion import ProjectedGradientDescent
from sklearn.metrics import roc_auc_score

from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import DATA, CKPT, DEVICE
from adaptive_attack import diff_map_feats, adaptive_lowfreq, predict


class DiffXAIDetector(nn.Module):
    """Image -> attribution features -> MLP -> one logit (positive = adversarial).

    Differentiable end to end, so ART can attack through the explanations.
    """

    def __init__(self, base, fmean, fstd, hidden=64):
        super().__init__()
        self.base = base
        self.register_buffer("fmean", fmean)
        self.register_buffer("fstd", fstd)
        self.head = nn.Sequential(
            nn.Linear(fmean.numel(), hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def attr_features(self, x):
        with torch.enable_grad():
            xg = x if x.requires_grad else x.clone().requires_grad_(True)
            logits = self.base(xg)
            tgt = logits.argmax(1).detach()
            sel = logits.gather(1, tgt[:, None]).sum()
            g = torch.autograd.grad(sel, xg, create_graph=True)[0]
        ixg = g * xg
        u = g.abs().sum(1).flatten(1)
        v = ixg.abs().sum(1).flatten(1)
        un = (u - u.mean(1, keepdim=True)) / (u.std(1, keepdim=True) + 1e-12)
        vn = (v - v.mean(1, keepdim=True)) / (v.std(1, keepdim=True) + 1e-12)
        corr = (un * vn).mean(1, keepdim=True)
        return torch.cat([diff_map_feats(g), diff_map_feats(ixg), corr], 1)

    def forward(self, x):
        f = (self.attr_features(x) - self.fmean) / self.fstd
        return self.head(f)


class DefendedModel(nn.Module):
    """Classifier and detector fused into one (n+1)-class model.

    Implements Carlini and Wagner's combination rule (arXiv:1705.07263), which is
    what ART's DetectorClassifier also implements:
        logit_{n+1} = (D(x) + 1) * max_i Z_i(x)
    so the extra class wins whenever the detector score D(x) is positive.
    """

    def __init__(self, base, detector):
        super().__init__()
        self.base = base
        self.detector = detector

    def forward(self, x):
        z = self.base(x)
        d = self.detector(x).squeeze(1)
        extra = (d + 1.0) * z.max(1).values
        return torch.cat([z, extra[:, None]], dim=1)


def batched_features(det, x, bs=128):
    out = []
    for i in range(0, len(x), bs):
        f = det.attr_features(x[i:i + bs].to(DEVICE))
        out.append(f.detach().cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--pgd-iter", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()

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
    sel = keep[torch.randperm(len(keep), generator=g)[:args.n]]
    X, Y = X[sel], Y[sel]
    print(f"{len(X)} correctly-classified clean images", flush=True)

    # ---- training data for the detector: clean vs NON-ADAPTIVE low-frequency ----
    print("generating non-adaptive attacks to train the detector ...", flush=True)
    stub = DiffXAIDetector(model, torch.zeros(23), torch.ones(23)).to(DEVICE).eval()
    Xadv = adaptive_lowfreq(model, X, Y, args.eps, args.k, 0.0,
                            torch.ones(23, device=DEVICE), steps=200)
    ok = (predict(model, Xadv) != Y)
    print(f"  non-adaptive success {ok.float().mean()*100:.1f}%", flush=True)

    F_cln = batched_features(stub, X)
    F_adv = batched_features(stub, Xadv[ok])
    fmean, fstd = F_cln.mean(0), F_cln.std(0).clamp(min=1e-6)

    det = DiffXAIDetector(model, fmean, fstd).to(DEVICE).eval()
    for p in det.base.parameters():
        p.requires_grad_(False)

    # train only the MLP head, on pre-computed (detached) features
    Ftr = torch.cat([F_cln, F_adv]).to(DEVICE)
    ytr = torch.cat([torch.zeros(len(F_cln)), torch.ones(len(F_adv))]).to(DEVICE)[:, None]
    Ftr = (Ftr - fmean.to(DEVICE)) / fstd.to(DEVICE)
    opt = torch.optim.Adam(det.head.parameters(), lr=2e-3)
    lossf = nn.BCEWithLogitsLoss()
    for ep in range(args.epochs):
        perm = torch.randperm(len(Ftr), device=DEVICE)
        for i in range(0, len(Ftr), 256):
            b = perm[i:i + 256]
            opt.zero_grad(set_to_none=True)
            lossf(det.head(Ftr[b]), ytr[b]).backward()
            opt.step()
    with torch.no_grad():
        sc = det.head(Ftr).squeeze(1).cpu().numpy()
    auc0 = roc_auc_score(ytr.squeeze(1).cpu().numpy(), sc)
    print(f"  detector head trained: AUC on its own training data {auc0:.4f}", flush=True)

    # ---- wire into ART -------------------------------------------------------
    # ART's DetectorClassifier needs a 1-output detector, but PyTorchClassifier
    # refuses nb_classes < 2, so that path is unusable as shipped. We instead
    # fuse both into ONE module using Carlini's own combination rule
    # (arXiv:1705.07263, the same formula DetectorClassifier implements):
    #     logit_{n+1} = (D(x) + 1) * max_i Z_i(x)
    # This is strictly better here: ART differentiates the fused graph natively
    # instead of stitching two estimators' gradients together in numpy.
    defended_model = DefendedModel(model, det).to(DEVICE).eval()

    sign_clf = PyTorchClassifier(model=model, loss=nn.CrossEntropyLoss(),
                                 input_shape=(3, s, s), nb_classes=NUM_CLASSES,
                                 clip_values=(0.0, 1.0),
                                 device_type="gpu" if DEVICE == "cuda" else "cpu")
    defended = PyTorchClassifier(model=defended_model, loss=nn.CrossEntropyLoss(),
                                 input_shape=(3, s, s), nb_classes=NUM_CLASSES + 1,
                                 clip_values=(0.0, 1.0),
                                 device_type="gpu" if DEVICE == "cuda" else "cpu")
    print(f"defended model built: {NUM_CLASSES + 1} classes "
          f"({NUM_CLASSES} signs + 1 'adversarial')\n", flush=True)

    # Targeted attacks, so success is unambiguous. An UNtargeted attack could
    # "succeed" by pushing the input into the detector class, which is a
    # detection, not an evasion. Targeting a specific wrong SIGN class forces
    # the attacker to beat the classifier and the detector at once.
    rng = np.random.default_rng(0)
    tgt = rng.integers(0, NUM_CLASSES, size=len(X))
    same = tgt == Y.numpy()
    tgt[same] = (tgt[same] + 1) % NUM_CLASSES
    y_t = np.eye(NUM_CLASSES)[tgt]
    y_t_def = np.eye(NUM_CLASSES + 1)[tgt]

    # ---- attacks -------------------------------------------------------------
    with torch.no_grad():
        sc_clean = torch.cat([det(X[i:i + 128].to(DEVICE)).squeeze(1).cpu()
                              for i in range(0, len(X), 128)])

    def report(name, Xa):
        pred = predict(model, Xa)
        hit = torch.from_numpy(pred.numpy() == tgt)      # reached the target class
        miscls = (pred != Y)
        with torch.no_grad():
            sc = torch.cat([det(Xa[i:i + 128].to(DEVICE)).squeeze(1).cpu()
                            for i in range(0, len(Xa), 128)])
        flagged = sc > 0
        evaded = miscls & ~flagged
        lab = np.concatenate([np.zeros(len(X)), np.ones(int(miscls.sum()))])
        auc = (roc_auc_score(lab, np.concatenate([sc_clean.numpy(), sc[miscls].numpy()]))
               if miscls.sum() > 5 else float("nan"))
        print(f"{name:>28} | target hit {hit.float().mean()*100:5.1f}%  "
              f"misclassified {miscls.float().mean()*100:5.1f}%  "
              f"flagged {flagged.float().mean()*100:5.1f}%  "
              f"EVADED {evaded.float().mean()*100:5.1f}%  AUC {auc:.4f}", flush=True)

    print("attacking the UNDEFENDED classifier (baseline) ...", flush=True)
    pgd_plain = ProjectedGradientDescent(estimator=sign_clf, eps=args.eps,
                                         eps_step=args.eps / 8, max_iter=args.pgd_iter,
                                         batch_size=64, targeted=True, verbose=False)
    report("PGD vs classifier only", torch.from_numpy(pgd_plain.generate(x=X.numpy(), y=y_t)).float())

    print("attacking the DEFENDED model (adaptive, via ART) ...", flush=True)
    pgd_adapt = ProjectedGradientDescent(estimator=defended, eps=args.eps,
                                         eps_step=args.eps / 8, max_iter=args.pgd_iter,
                                         batch_size=32, targeted=True, verbose=False)
    report("PGD vs classifier+detector",
           torch.from_numpy(pgd_adapt.generate(x=X.numpy(), y=y_t_def)).float())

    print("\n'EVADED' = misclassified AND not flagged. That is the number that matters.")
    print("If the adaptive row evades far more often, the detector is broken.")


if __name__ == "__main__":
    main()
