"""Headline table: every attack against the same detectors, images and protocol.

Evasion = misclassified AND unflagged, measured on the gradient-boosted trees
at 5% and 8.4% clean FPR, for the XAI detector, the pixel detector, and their
OR-composition. The last row scores each image by the strongest attack.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u aim1_headline.py
"""
import argparse
import os

os.environ.setdefault('TQDM_DISABLE', '1')   # ART/AutoAttack progress bars

import numpy as np
import torch

if not hasattr(np, 'mat'):          # NumPy 2.0 removed it; GeoDA uses it
    np.mat = np.asmatrix
import torch.nn as nn
from art.attacks.evasion import (FastGradientMethod, ProjectedGradientDescent,
                                 AutoProjectedGradientDescent, SquareAttack,
                                 AutoAttack, CarliniL2Method, DeepFool,
                                 ElasticNet, ShadowAttack, HopSkipJump,
                                 SimBA, BoundaryAttack, GeoDA,
                                 SpatialTransformation, SignOPTAttack)
from art.estimators.classification import PyTorchClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import CKPT, DEVICE
from relu_surrogate import swap_relu
from surrogate_study import fit_surrogate
from aim1_attack import attack as plain_attack, predict, load_data
from aim1_adaptive import thr_at_fpr, mlp_scores, xai_feats, pixel_feats
from aim1_minimax import Norm, joint_attack
from aim1_disagree import disagree_attack

FPRS = [0.05, 0.084]


class _Wrap(nn.Module):
    """Cast inputs to float32 (ART's decision-based attacks send float64) and
    optionally return probabilities (SimBA needs them)."""

    def __init__(self, m, softmax=False):
        super().__init__()
        self.m = m
        self.softmax = softmax

    def forward(self, x):
        out = self.m(x.float())
        return torch.softmax(out, dim=1) if self.softmax else out


def _shadow(clf, Xn, args, n_sub=60):
    """ShadowAttack takes one sample per call, so only a subset is attacked.
    The rest are returned unchanged, so this row is a lower bound."""
    a = ShadowAttack(estimator=clf, batch_size=1, nb_steps=100, verbose=False)
    out = Xn.copy()
    for i in range(min(n_sub, len(Xn))):
        out[i] = a.generate(x=Xn[i:i + 1])[0]
    return torch.from_numpy(out).float()


def or_thresholds(sp, sx, fpr, n=400):
    """Two thresholds whose OR flags `fpr` of clean inputs, so the composition
    gets no larger false-positive budget than either detector alone."""
    qs = np.linspace(1 - fpr, 1.0, n)
    best = min(qs, key=lambda q: abs(
        ((sp > np.quantile(sp, q)) | (sx > np.quantile(sx, q))).mean() - fpr))
    return float(np.quantile(sp, best)), float(np.quantile(sx, best))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1600)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=2)
    args = ap.parse_args()

    plain, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    surro, _ = load_trained(CKPT, DEVICE)
    swap_relu(surro, kind="adv2")

    X, Y = load_data(plain, img_size, args.n)
    ntr = len(X) // 2
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    print(f"classifier: {ckpt.get('backbone')} @ {img_size}px, "
          f"clean acc {ckpt['test_acc'] * 100:.2f}%")
    print(f"{len(Xtr)} detector-train / {len(Xte)} attack-eval, "
          f"eps={args.eps}, steps={args.steps}\n")

    # ---- detectors -------------------------------------------------------
    print("fitting detectors on a mixture of non-adaptive attacks ...",
          flush=True)
    clf_tr = PyTorchClassifier(model=plain, loss=nn.CrossEntropyLoss(),
                               input_shape=(3, img_size, img_size),
                               nb_classes=NUM_CLASSES, clip_values=(0.0, 1.0),
                               device_type="gpu" if DEVICE == "cuda" else "cpu")
    advs = [plain_attack(plain, Xtr, Ytr, eps=args.eps, k=96, lam=0.0,
                         steps=args.steps).float(),
            plain_attack(plain, Xtr, Ytr, eps=args.eps, k=6, lam=0.0,
                         steps=args.steps).float(),
            plain_attack(plain, Xtr, Ytr, eps=args.eps, k=3, lam=0.0,
                         steps=args.steps).float(),
            torch.from_numpy(FastGradientMethod(
                estimator=clf_tr, eps=args.eps,
                batch_size=256).generate(x=Xtr.numpy())).float()]
    Fx = np.concatenate([xai_feats(plain, Xtr, predict(plain, Xtr))]
                        + [xai_feats(plain, A, predict(plain, A)) for A in advs])
    Fp = np.concatenate([pixel_feats(Xtr)] + [pixel_feats(A) for A in advs])
    lab = np.concatenate([np.zeros(len(Xtr))] + [np.ones(len(A)) for A in advs])

    tree_x = HistGradientBoostingClassifier(max_iter=400,
                                            random_state=0).fit(Fx, lab)
    tree_p = HistGradientBoostingClassifier(max_iter=400,
                                            random_state=0).fit(Fp, lab)
    det_x, det_p = fit_surrogate(Fx, lab), fit_surrogate(Fp, lab)

    Fx_c = xai_feats(plain, Xte, predict(plain, Xte))
    Fp_c = pixel_feats(Xte)
    cx = tree_x.predict_proba(Fx_c)[:, 1]
    cp = tree_p.predict_proba(Fp_c)[:, 1]
    nx = Norm(mlp_scores(det_x, Fx_c), 0.05)     # for the attack's objective
    np_ = Norm(mlp_scores(det_p, Fp_c), 0.05)

    thr = {}
    for f in FPRS:
        thr[f] = dict(x=thr_at_fpr(cx, f), p=thr_at_fpr(cp, f),
                      o=or_thresholds(cp, cx, f))
    print("detectors fitted and thresholds calibrated on clean held-out data\n")

    # ---- attack pool -----------------------------------------------------
    clf = PyTorchClassifier(model=plain, loss=nn.CrossEntropyLoss(),
                            input_shape=(3, img_size, img_size),
                            nb_classes=NUM_CLASSES, clip_values=(0.0, 1.0),
                            device_type="gpu" if DEVICE == "cuda" else "cpu")
    def _clf(softmax=False):
        return PyTorchClassifier(
            model=_Wrap(plain, softmax).to(DEVICE), loss=nn.CrossEntropyLoss(),
            input_shape=(3, img_size, img_size), nb_classes=NUM_CLASSES,
            clip_values=(0.0, 1.0),
            device_type="gpu" if DEVICE == "cuda" else "cpu")

    clf_cast = _clf(False)       # float64-tolerant, for decision-based attacks
    clf_prob = _clf(True)        # probability outputs, for SimBA
    Xn = Xte.numpy()

    def art(a):
        return torch.from_numpy(a.generate(x=Xn)).float()

    # "seen" marks attacks the detectors were trained on.
    pool = [
        # --- cheap white-box, plus the standard parameter-free benchmark ---
        ("FGSM         ART seen", lambda: art(
            FastGradientMethod(estimator=clf, eps=args.eps, batch_size=256))),
        ("PGD          ART seen", lambda: art(
            ProjectedGradientDescent(estimator=clf, eps=args.eps,
                                     eps_step=args.eps / 6, max_iter=40,
                                     batch_size=256, verbose=False))),
        ("APGD         ART     ", lambda: art(
            AutoProjectedGradientDescent(estimator=clf, eps=args.eps,
                                         max_iter=40, nb_random_init=1,
                                         batch_size=256, verbose=False))),
        ("AutoAttack   ART     ", lambda: art(
            AutoAttack(estimator=clf, eps=args.eps, batch_size=128,
                       targeted=False))),
        ("DeepFool     ART     ", lambda: art(
            DeepFool(classifier=clf, max_iter=50, batch_size=128,
                     verbose=False))),
        # --- black-box, score-based ---
        ("Square       ART     ", lambda: art(
            SquareAttack(estimator=clf, eps=args.eps, max_iter=1000,
                         batch_size=256, verbose=False))),
        ("SimBA        ART     ", lambda: art(
            SimBA(classifier=clf_prob, max_iter=800, epsilon=args.eps))),
        # --- black-box, decision-based: the obfuscated-gradients test ---
        ("HopSkipJump  ART     ", lambda: art(
            HopSkipJump(classifier=clf_cast, targeted=False, max_iter=10,
                        max_eval=500, init_eval=10, verbose=False))),
        ("GeoDA        ART     ", lambda: art(
            GeoDA(estimator=clf_cast, max_iter=1000, batch_size=64,
                  verbose=False))),
        ("SignOPT      ART     ", lambda: art(
            SignOPTAttack(estimator=clf_cast, targeted=False, max_iter=100,
                          query_limit=2000, batch_size=64, verbose=False))),
        # --- geometric: no Lp perturbation at all ---
        ("SpatialTrans ART     ", lambda: art(
            SpatialTransformation(classifier=clf, max_translation=8,
                                  max_rotation=15, num_translations=5,
                                  num_rotations=5, verbose=False))),
        # --- slow stragglers last, so a timeout costs the least ---
        ("CW-L2        ART     ", lambda: art(
            CarliniL2Method(classifier=clf, max_iter=50,
                            binary_search_steps=7, batch_size=128,
                            verbose=False))),
        ("ElasticNet   ART     ", lambda: art(
            ElasticNet(classifier=clf, max_iter=20, binary_search_steps=5,
                       batch_size=128, verbose=False))),
        ("ShadowAttack ART     ", lambda: _shadow(clf, Xn, args)),
        ("Boundary     ART     ", lambda: art(
            BoundaryAttack(estimator=clf_cast, targeted=False, max_iter=100,
                           num_trial=3, sample_size=5, verbose=False))),
    ]
    THREAT = {
        "FGSM": "WB", "PGD": "WB", "APGD": "WB", "AutoAttack": "WB",
        "CW-L2": "WB", "DeepFool": "WB", "ElasticNet": "WB",
        "ShadowAttack": "WB", "Square": "BB-S", "SimBA": "BB-S",
        "HopSkipJump": "BB-D", "Boundary": "BB-D", "GeoDA": "BB-D",
        "SignOPT": "BB-D", "SpatialTrans": "GEO",
        "oursDIS": "WB-nd",      # white-box classifier, NO detector access
        "oursBOTH": "WB*",
    }
    ours = []
    for k, lam in [(96, 1.0), (12, 1.0), (6, 1.0), (6, 3.0), (4, 1.0)]:
        ours.append((f"ours  k={k:<3d} lam={lam:.0f}  WB*",
                     (lambda k=k, lam=lam: joint_attack(
                         surro, det_x, det_p, nx, np_, Xte, Yte, eps=args.eps,
                         k=k, lam=lam, steps=args.steps,
                         restarts=args.restarts).float())))
    # Agreement attack: "blind" never queries the detector.
    ours.append(("oursDIS blind       ", lambda: disagree_attack(
        surro, det_x, nx, Xte, Yte, eps=args.eps, k=96, lam_det=0.0,
        lam_dis=1.0, steps=args.steps, restarts=args.restarts).float()))
    ours.append(("oursBOTH det+dis    ", lambda: disagree_attack(
        surro, det_x, nx, Xte, Yte, eps=args.eps, k=96, lam_det=1.0,
        lam_dis=1.0, steps=args.steps, restarts=args.restarts).float()))

    # Insert ours ahead of the slow stragglers (the last five ART rows).
    pool = pool[:-5] + ours + pool[-5:]

    # ---- evaluate --------------------------------------------------------
    hdr = (f"{'attack':>24} {'threat':>7} {'mis':>6} | "
           + " | ".join(f"{'XAI':>6} {'PIXEL':>6} {'OR':>6}  @{f:.0%}"
                        for f in FPRS))
    print(hdr)
    print("-" * len(hdr))

    any_ev = {f: {k: np.zeros(len(Xte), bool) for k in "xpo"} for f in FPRS}
    for name, fn in pool:
        try:
            A = fn()
        except Exception as e:
            print(f"{name:>24}   SKIPPED: {type(e).__name__}: "
                  f"{str(e)[:50]}", flush=True)
            continue
        pa = predict(plain, A)
        mis = (pa != Yte).numpy()
        sx = tree_x.predict_proba(xai_feats(plain, A, pa))[:, 1]
        sp = tree_p.predict_proba(pixel_feats(A))[:, 1]
        cells = []
        for f in FPRS:
            t = thr[f]
            ev = dict(x=mis & (sx < t["x"]),
                      p=mis & (sp < t["p"]),
                      o=mis & ~((sp > t["o"][0]) | (sx > t["o"][1])))
            for k in "xpo":
                any_ev[f][k] |= ev[k]
            cells.append(f"{ev['x'].mean() * 100:>5.1f}% "
                         f"{ev['p'].mean() * 100:>5.1f}% "
                         f"{ev['o'].mean() * 100:>5.1f}%      ")
        tm = THREAT.get(name.split()[0], "WB*")
        print(f"{name:>24} {tm:>7} {mis.mean() * 100:>5.1f}% | "
              + " | ".join(cells), flush=True)

    print("-" * len(hdr))
    cells = [f"{any_ev[f]['x'].mean() * 100:>5.1f}% "
             f"{any_ev[f]['p'].mean() * 100:>5.1f}% "
             f"{any_ev[f]['o'].mean() * 100:>5.1f}%      " for f in FPRS]
    print(f"{'STRONGEST PER SAMPLE':>24} {'all':>7} {'---':>6} | "
          + " | ".join(cells))
    print()
    print("Evasion = misclassified AND unflagged, over all attacked inputs.")
    print("Threat models: WB white-box; BB-S black-box score-based; BB-D")
    print("black-box decision-based; GEO geometric; WB* white-box on classifier")
    print("and detector; WB-nd white-box on classifier, no detector access.")


if __name__ == "__main__":
    main()
