"""Aim 2 feasibility: does the way explanation features VARY across the frames
of one sign separate camera-robust attacks from clean signs?

Each test sign is seen through K camera views (stand-in for the frames of an
approaching car). Per sign we get a (K, 54) block of XAI features. Detectors,
all scored per sign:
  output     fraction of views that disagree with the majority reading
             (AdvIT-style output consistency, no training)
  1 frame    trees on the features of one view (the retrain-on-EOT baseline)
  vote       the same per-frame trees, score averaged over the K views
  mean       trees on the K-view mean of each feature (more evidence)
  spread     trees on the K-view standard deviation of each feature
             (explanation consistency, the Aim 2 idea)
  mean+spread
The trained detectors use 5 folds grouped by sign, so a sign's clean and
attacked versions are never split between train and test. "unseen" trains on
CW + EOT and tests on ADV2 + EOT, which the detector never saw. Positives are
attacked signs misread in at least half their views.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u aim2_consistency.py
"""
import argparse
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

import camera_view
from model import load_trained, IMG_SIZE
from xai_detect import CKPT, DEVICE, WORK
from relu_surrogate import swap_relu
from aim1_attack import predict, load_data
from aim1_adaptive import thr_at_fpr, xai_feats
from aim1_improve import blind_attack

ADV2 = {"adv2_all": 10.0}


def per_view(model, x, k, seed):
    """Readings (N, K) and XAI features (N, K, 54) of k fixed views of x."""
    V = camera_view.views(x, k, seed=seed)
    p = predict(model, V)
    F = xai_feats(model, V, p)
    n = len(x)
    return p.view(k, n).T.numpy(), F.reshape(k, n, -1).transpose(1, 0, 2)


def tree():
    return HistGradientBoostingClassifier(max_iter=400, random_state=0)


def summarise(F, kind):
    if kind == "1 frame":
        return F[:, 0]
    if kind == "mean":
        return F.mean(1)
    if kind == "spread":
        return F.std(1)
    return np.concatenate([F.mean(1), F.std(1)], 1)


def report(name, s_clean, s_att):
    y = np.r_[np.zeros(len(s_clean)), np.ones(len(s_att))]
    auc = roc_auc_score(y, np.r_[s_clean, s_att])
    tpr = (s_att > thr_at_fpr(s_clean, 0.05)).mean()
    print(f"  {name:>12}  AUC {auc:.3f}   caught @5% FPR {tpr * 100:5.1f}%", flush=True)
    return auc, tpr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--views", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    t0 = time.time()

    plain, ckpt = load_trained(CKPT, DEVICE)
    surro, _ = load_trained(CKPT, DEVICE)
    swap_relu(surro, kind="adv2")
    X, Y = load_data(plain, ckpt.get("img_size", IMG_SIZE), args.n, seed=args.seed)
    X, Y = X[len(X) // 2:], Y[len(Y) // 2:]          # the aim1_camera attack half
    K, vseed = args.views, 2000 + args.seed
    print(f"{len(X)} signs, {K} views each, eps={args.eps}, seed={args.seed}", flush=True)

    cache = os.path.join(WORK, f"aim2_consistency_seed{args.seed}.npz")
    if os.path.exists(cache):
        d = np.load(cache)
        P = {k: d["P_" + k] for k in ("clean", "cw", "adv2")}
        F = {k: d["F_" + k] for k in ("clean", "cw", "adv2")}
        print("loaded", cache)
    else:
        A = {"clean": X,
             "cw": blind_attack(surro, X, Y, args.eps, {}, None, None,
                                keep_best=False, view=True, eot=2).float(),
             "adv2": blind_attack(surro, X, Y, args.eps, ADV2, None, None,
                                  view=True, eot=2).float()}
        print(f"attacks done ({time.time() - t0:.0f}s)", flush=True)
        P, F = {}, {}
        for k, x in A.items():
            P[k], F[k] = per_view(plain, x, K, vseed)
        np.savez(cache, **{f"P_{k}": v for k, v in P.items()},
                 **{f"F_{k}": v for k, v in F.items()})
        print(f"features done ({time.time() - t0:.0f}s), saved {cache}", flush=True)

    y = Y.numpy()
    ok = {k: (P[k] != y[:, None]).mean(1) >= 0.5 for k in ("cw", "adv2")}
    print(f"clean views read correctly {(P['clean'] == y[:, None]).mean() * 100:.1f}%; "
          f"signs misread in >= half their views: CW + EOT {ok['cw'].mean() * 100:.1f}%, "
          f"ADV2 + EOT {ok['adv2'].mean() * 100:.1f}%\n")

    def disagree(p):
        maj = np.array([np.bincount(r).argmax() for r in p])
        return (p != maj[:, None]).mean(1)

    folds = list(KFold(5, shuffle=True, random_state=0).split(X))
    kinds = ["1 frame", "vote", "mean", "spread", "mean+spread"]
    out = {}
    for train_on, test_on in [("cw", "cw"), ("adv2", "adv2"), ("cw", "adv2")]:
        tag = test_on if train_on == test_on else "unseen"
        print(f"train on {train_on.upper()} + EOT, test on {test_on.upper()} + EOT"
              f"{' (unseen attack)' if tag == 'unseen' else ''}:")
        out[f"{tag}|output"] = report("output", disagree(P["clean"]),
                                      disagree(P[test_on])[ok[test_on]])
        for kind in kinds:
            sc, sa = np.zeros(len(X)), np.zeros(len(X))
            for tr, te in folds:
                pos = tr[ok[train_on][tr]]
                if kind == "vote":       # per-frame trees, scores averaged over views
                    Xf = np.concatenate([F["clean"][tr].reshape(-1, 54),
                                         F[train_on][pos].reshape(-1, 54)])
                    yf = np.r_[np.zeros(len(tr) * K), np.ones(len(pos) * K)]
                    m = tree().fit(Xf, yf)
                    sc[te] = m.predict_proba(F["clean"][te].reshape(-1, 54))[:, 1].reshape(-1, K).mean(1)
                    sa[te] = m.predict_proba(F[test_on][te].reshape(-1, 54))[:, 1].reshape(-1, K).mean(1)
                else:
                    Xf = np.concatenate([summarise(F["clean"][tr], kind),
                                         summarise(F[train_on][pos], kind)])
                    yf = np.r_[np.zeros(len(tr)), np.ones(len(pos))]
                    m = tree().fit(Xf, yf)
                    sc[te] = m.predict_proba(summarise(F["clean"][te], kind))[:, 1]
                    sa[te] = m.predict_proba(summarise(F[test_on][te], kind))[:, 1]
            out[f"{tag}|{kind}"] = report(kind, sc, sa[ok[test_on]])
        print()

    np.savez(os.path.join(WORK, f"aim2_consistency_result_seed{args.seed}.npz"),
             **{k: np.array(v) for k, v in out.items()})
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
