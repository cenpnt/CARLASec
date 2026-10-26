"""Cost of EOT. How much longer does an attack take as it averages over more
camera views per step, what does that buy (misread rate on camera views), and
what does the defender pay per frame at run time?

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u time_eot.py
"""
import argparse
import os
import time

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

import camera_view
from model import load_trained, IMG_SIZE
from xai_detect import CKPT, DEVICE, WORK
from relu_surrogate import swap_relu
from aim1_attack import predict, load_data
from aim1_adaptive import xai_feats
from aim1_improve import blind_attack, pgd_attack

ADV2 = {"adv2_all": 10.0}


def timed(fn):
    torch.cuda.synchronize()
    t = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--views", type=int, default=10)
    ap.add_argument("--eot", type=int, nargs="+", default=[1, 2, 4, 8])
    args = ap.parse_args()
    torch.manual_seed(0)

    plain, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    surro, _ = load_trained(CKPT, DEVICE)
    swap_relu(surro, kind="adv2")
    X, Y = load_data(plain, img_size, args.n, seed=0)
    Yv = Y.repeat(args.views)
    print(f"{len(X)} images, eps={args.eps}, {torch.cuda.get_device_name(0)}\n")

    def misread(A):
        mis0 = (predict(plain, A) != Y).float().mean().item()
        V = camera_view.views(A, args.views, seed=2000)
        return mis0, (predict(plain, V) != Yv).float().mean().item()

    # attacker cost: (name, attack) pairs, the view-free baseline first
    runs = [("CW", 0, lambda: blind_attack(surro, X, Y, args.eps, {}, None, None,
                                           keep_best=False))]
    runs += [(f"CW + EOT", k, lambda k=k: blind_attack(
        surro, X, Y, args.eps, {}, None, None, keep_best=False, view=True, eot=k))
        for k in args.eot]
    runs += [("ADV2 ours", 0, lambda: blind_attack(surro, X, Y, args.eps, ADV2,
                                                   None, None))]
    runs += [("ADV2 ours + EOT", k, lambda k=k: blind_attack(
        surro, X, Y, args.eps, ADV2, None, None, view=True, eot=k))
        for k in args.eot if k <= 4]
    runs += [("ADV2 published", 0, lambda: pgd_attack(surro, X, Y, args.eps, ADV2,
                                                      None, None, steps=200))]
    runs += [("ADV2 published + EOT", 2, lambda: pgd_attack(
        surro, X, Y, args.eps, ADV2, None, None, steps=200, view=True, eot=2))]

    print(f"{'attack':>22} {'views/step':>10} {'s/image':>8} {'x base':>7} "
          f"{'mis exact':>9} {'mis views':>9}")
    rows, base = [], {}
    for name, k, fn in runs:
        A, dt = timed(fn)
        m0, mv = misread(A.float())
        fam = name.split(" +")[0]
        base.setdefault(fam, dt)
        rows.append((name, k, dt / len(X), dt / base[fam], m0, mv))
        print(f"{name:>22} {k if k else '-':>10} {dt / len(X):>8.3f} "
              f"{dt / base[fam]:>6.1f}x {m0 * 100:>8.1f}% {mv * 100:>8.1f}%",
              flush=True)

    # defender cost per frame: classifier alone, then the full XAI detector
    tree = HistGradientBoostingClassifier(max_iter=400, random_state=0).fit(
        np.random.rand(200, 54), np.r_[np.zeros(100), np.ones(100)])
    x1 = X[:1].to(DEVICE)
    for _ in range(3):
        predict(plain, x1); xai_feats(plain, x1, predict(plain, x1))
    reps = 50
    _, t_cls = timed(lambda: [predict(plain, x1) for _ in range(reps)])
    _, t_xai = timed(lambda: [tree.predict_proba(xai_feats(plain, x1, predict(plain, x1)))
                              for _ in range(reps)])
    _, t_bat = timed(lambda: xai_feats(plain, X, predict(plain, X)))
    print(f"\ndefender, one frame: classifier {t_cls / reps * 1000:.1f} ms, "
          f"classifier + XAI detector {t_xai / reps * 1000:.1f} ms; "
          f"batched detector {t_bat / len(X) * 1000:.1f} ms/frame")

    np.savez(os.path.join(WORK, "time_eot.npz"),
             rows=np.array(rows, dtype=object),
             defender_ms=np.array([t_cls / reps, t_xai / reps, t_bat / len(X)]) * 1000)


if __name__ == "__main__":
    main()
