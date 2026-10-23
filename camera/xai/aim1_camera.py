"""Aim 1 under camera viewing. Do the map-alignment attacks still fool the
classifier and pass the XAI detector once a car camera sees the sign from a
random angle, distance and light (camera_view.py)? And does optimising over
those views (Expectation over Transformation) keep them working?

Each attacked sign is scored on the same `--views` fixed camera views as its
clean version. Three detector settings:
  lab      the aim1_improve detector and threshold, untouched
  recal    the same detector, threshold reset to 5% FPR on clean camera views
  camera   a detector trained on camera views, 5% FPR on clean camera views

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u aim1_camera.py
"""
import argparse
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import camera_view
from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import CKPT, DEVICE, WORK
from relu_surrogate import swap_relu
from aim1_attack import attack as plain_attack, predict, load_data
from aim1_adaptive import thr_at_fpr, xai_feats
from aim1_improve import blind_attack, pgd_attack

ADV2 = {"adv2_all": 10.0}
EOT = {"view": True}
# name -> (procedure, stealth weights, extra attack options). Without a map
# term keep-best keeps the FIRST misclassified step, so the CW + EOT control
# also runs on the final step.
ATTACKS = {
    "CW only":              ("blind", {}, {}),
    "CW + EOT":             ("blind", {}, EOT),
    "CW + EOT, last step":  ("blind", {}, {**EOT, "keep_best": False}),
    "ADV2 published":       ("pgd", ADV2, {}),
    "ADV2 ours":            ("blind", ADV2, {}),
    "ADV2 published + EOT": ("pgd", ADV2, EOT),
    "ADV2 ours + EOT":      ("blind", ADV2, EOT),
    "ADV2 ours + EOT, fixed ref": ("blind", ADV2, {**EOT, "view_ref": False}),
}


def feats_scores(model, tree, x):
    pred = predict(model, x)
    return pred, tree.predict_proba(xai_feats(model, x, pred))[:, 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--pgd-steps", type=int, default=200)
    ap.add_argument("--views", type=int, default=10, help="camera views per test sign")
    ap.add_argument("--eot", type=int, default=2, help="camera views per attack step")
    ap.add_argument("--seed", type=int, default=0, help="image sample and split")
    ap.add_argument("--only", nargs="*", default=None,
                    help="attack names to run (exact), else substrings of them")
    ap.add_argument("--lam", type=float, nargs="+", default=[1.0],
                    help="multiplier applied to every stealth weight")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()

    plain, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    surro, _ = load_trained(CKPT, DEVICE)
    swap_relu(surro, kind="adv2")

    X, Y = load_data(plain, img_size, args.n, seed=args.seed)
    ntr = len(X) // 2
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    K = args.views
    print(f"{len(Xtr)} detector-train / {len(Xte)} attack-eval, eps={args.eps}, "
          f"steps={args.steps}, restarts={args.restarts}, views={K}, "
          f"eot={args.eot}, seed={args.seed}\n", flush=True)

    # lab detector: same recipe as aim1_improve
    clf = PyTorchClassifier(model=plain, loss=nn.CrossEntropyLoss(),
                            input_shape=(3, img_size, img_size),
                            nb_classes=NUM_CLASSES, clip_values=(0.0, 1.0),
                            device_type="gpu" if DEVICE == "cuda" else "cpu")
    advs = [plain_attack(plain, Xtr, Ytr, eps=args.eps, k=k, lam=0.0,
                         steps=args.steps).float() for k in (96, 6, 3)]
    advs.append(torch.from_numpy(FastGradientMethod(
        estimator=clf, eps=args.eps, batch_size=256).generate(
            x=Xtr.numpy())).float())
    lab = np.concatenate([np.zeros(len(Xtr))] + [np.ones(len(A)) for A in advs])
    Fx = np.concatenate([xai_feats(plain, Xtr, predict(plain, Xtr))]
                        + [xai_feats(plain, A, predict(plain, A)) for A in advs])
    tree_lab = HistGradientBoostingClassifier(max_iter=400,
                                              random_state=0).fit(Fx, lab)

    # camera detector: one camera view of every training image
    vseed = 1000 + args.seed
    Fc = np.concatenate([xai_feats(plain, V, predict(plain, V)) for V in
                         [camera_view.views(A, 1, seed=vseed) for A in [Xtr] + advs]])
    tree_cam = HistGradientBoostingClassifier(max_iter=400,
                                              random_state=0).fit(Fc, lab)
    print(f"detectors trained ({time.time() - t0:.0f}s)", flush=True)

    # thresholds at 5% FPR
    _, s_lab = feats_scores(plain, tree_lab, Xte)
    t_lab = thr_at_fpr(s_lab, 0.05)
    eseed = 2000 + args.seed
    Vte = camera_view.views(Xte, K, seed=eseed)
    Yv = Yte.repeat(K)
    pv, sv_lab = feats_scores(plain, tree_lab, Vte)
    _, sv_cam = feats_scores(plain, tree_cam, Vte)
    t_rec, t_cam = thr_at_fpr(sv_lab, 0.05), thr_at_fpr(sv_cam, 0.05)
    print(f"clean camera views: accuracy {(pv == Yv).float().mean() * 100:.1f}%, "
          f"lab threshold flags {(sv_lab > t_lab).mean() * 100:.1f}% "
          f"(5.0% without views)\n", flush=True)

    hdr = (f"{'attack':>26} | {'no view':^15} | {'camera views':^34}\n"
           f"{'':>26} | {'mis':>6} {'lab':>7} | {'mis':>6} {'lab':>7} "
           f"{'recal':>7} {'camera':>7}   (evasion @5% FPR)")
    print(hdr)
    print("-" * 74, flush=True)
    rows = {}
    exact = args.only and any(s in ATTACKS for s in args.only)
    for name, (proc, w, kw) in ATTACKS.items():
        if args.only and not (name in args.only if exact
                              else any(s in name for s in args.only)):
            continue
        kw = {**kw, "eot": args.eot if kw.get("view") else 1}
        for lam in (args.lam if w else [1.0]):
            t1 = time.time()
            ws = {k: v * lam for k, v in w.items()}
            label = name if lam == 1.0 else f"{name} x{lam:g}"
            if proc == "pgd":
                A = pgd_attack(surro, Xte, Yte, args.eps, ws, None, None,
                               steps=args.pgd_steps, **kw).float()
            else:
                A = blind_attack(surro, Xte, Yte, args.eps, ws, None, None,
                                 steps=args.steps, restarts=args.restarts,
                                 **kw).float()
            pa, sa = feats_scores(plain, tree_lab, A)
            mis0 = (pa != Yte).numpy()
            VA = camera_view.views(A, K, seed=eseed)
            pva, sva_lab = feats_scores(plain, tree_lab, VA)
            _, sva_cam = feats_scores(plain, tree_cam, VA)
            mis = (pva != Yv).numpy()
            r = {"mis0": mis0.mean(), "ev0": (mis0 & (sa < t_lab)).mean(),
                 "mis": mis.mean(), "lab": (mis & (sva_lab < t_lab)).mean(),
                 "recal": (mis & (sva_lab < t_rec)).mean(),
                 "camera": (mis & (sva_cam < t_cam)).mean()}
            rows[label] = r
            print(f"{label:>26} | {r['mis0'] * 100:>5.1f}% "
                  f"{r['ev0'] * 100:>6.1f}% | "
                  f"{r['mis'] * 100:>5.1f}% {r['lab'] * 100:>6.1f}% "
                  f"{r['recal'] * 100:>6.1f}% {r['camera'] * 100:>6.1f}%"
                  f"   ({time.time() - t1:.0f}s)", flush=True)

    tag = "_" + "_".join(args.only).replace(" ", "") if args.only else ""
    if args.lam != [1.0]:
        tag += "_lam" + "-".join(f"{v:g}" for v in args.lam)
    out = os.path.join(WORK, f"aim1_camera_seed{args.seed}{tag}.npz")
    np.savez(out, **{f"{n}|{k}": v for n, r in rows.items() for k, v in r.items()})
    print(f"\nsaved {out} ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
