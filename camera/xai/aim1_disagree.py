"""Agreement attack: fool the classifier while keeping the pairwise agreement
between IG, IxG and Saliency at its clean value. With lam_det = 0 it never
queries the detector.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u aim1_disagree.py
"""
import argparse

import numpy as np
import torch
from sklearn.ensemble import HistGradientBoostingClassifier

from model import load_trained, IMG_SIZE
from xai_detect import CKPT, DEVICE
from relu_surrogate import swap_relu
from feat_diff import features_t, disagreement_features_t
from surrogate_study import fit_surrogate
from aim1_attack import attack as plain_attack, cw_margin, upsample, predict, load_data
from aim1_adaptive import all_maps, thr_at_fpr, mlp_scores, xai_feats, pixel_feats
from aim1_minimax import Norm

# [pearson, cos, iou] per method pair; the IoU terms carry no gradient
DIFF_IDX = [0, 1, 3, 4, 6, 7]


def disagree_vec(maps):
    """The six differentiable pairwise agreement measures."""
    return disagreement_features_t(maps)[:, DIFF_IDX]


def disagree_attack(model, det_x, nx, x, y, eps, k, lam_det, lam_dis,
                    steps=100, lr=0.08, kappa=2.0, bs=64, warm=0.4,
                    restarts=2, seed=0):
    """Returns adversarial images on CPU. lam_det=0 gives the detector-blind
    agreement attack; lam_dis=0 the detector-score attack."""
    warm_steps = int(warm * steps)
    out = []

    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEVICE)
        yb = y[i:i + bs].to(DEVICE)
        H, n = xb.shape[-1], len(xb)

        # clean agreement, against the true class
        d_clean = disagree_vec(
            all_maps(model, xb.clone(), yb, create_graph=False)).detach()

        best_z = torch.zeros(n, 3, k, k, device=DEVICE)
        best = torch.full((n,), float("inf"), device=DEVICE)
        seen = torch.zeros(n, dtype=torch.bool, device=DEVICE)

        for r in range(restarts):
            if r == 0:
                z = torch.zeros(n, 3, k, k, device=DEVICE)
            else:
                g = torch.Generator(device="cpu").manual_seed(seed + 613 * r)
                z = (0.5 * torch.randn(n, 3, k, k, generator=g)).to(DEVICE)
            z = z.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)

            for step in range(steps):
                delta = eps * torch.tanh(upsample(z, H))
                x_adv = (xb + delta).clamp(0, 1)
                logits = model(x_adv)
                loss = cw_margin(logits, yb, kappa).sum()
                rank = None

                if step >= warm_steps and (lam_det > 0 or lam_dis > 0):
                    tgt = logits.argmax(1).detach()
                    maps = all_maps(model, x_adv, tgt, create_graph=True)
                    term = torch.zeros(n, device=DEVICE)
                    if lam_det > 0:
                        term = term + lam_det * nx.t(det_x(features_t(maps)))
                    if lam_dis > 0:
                        term = term + lam_dis * (
                            (disagree_vec(maps) - d_clean) ** 2).sum(1)
                    loss = loss + term.sum()
                    # rank by the attack's own objective, never the detector's
                    rank = term.detach()

                with torch.no_grad():
                    mis = logits.argmax(1) != yb
                    s = rank if rank is not None \
                        else torch.full((n,), 1e3, device=DEVICE)
                    cand = torch.where(mis, s, torch.full_like(s, float("inf")))
                    better = cand < best
                    best = torch.where(better, cand, best)
                    best_z[better] = z.detach()[better]
                    seen |= mis

                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

            with torch.no_grad():
                if (~seen).any():
                    best_z[~seen] = z.detach()[~seen]

        with torch.no_grad():
            delta = eps * torch.tanh(upsample(best_z, H))
            out.append((xb + delta).clamp(0, 1).cpu())

    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--fpr", type=float, default=0.05)
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
          f"eps={args.eps}, k={args.k}, steps={args.steps}\n")

    print("fitting detectors ...", flush=True)
    advs = [plain_attack(plain, Xtr, Ytr, eps=args.eps, k=96, lam=0.0,
                         steps=args.steps).float(),
            plain_attack(plain, Xtr, Ytr, eps=args.eps, k=6, lam=0.0,
                         steps=args.steps).float()]
    Fx = np.concatenate([xai_feats(plain, Xtr, predict(plain, Xtr))]
                        + [xai_feats(plain, A, predict(plain, A)) for A in advs])
    Fp = np.concatenate([pixel_feats(Xtr)] + [pixel_feats(A) for A in advs])
    lab = np.concatenate([np.zeros(len(Xtr))] + [np.ones(len(A)) for A in advs])

    tree_x = HistGradientBoostingClassifier(max_iter=400,
                                            random_state=0).fit(Fx, lab)
    tree_p = HistGradientBoostingClassifier(max_iter=400,
                                            random_state=0).fit(Fp, lab)
    det_x = fit_surrogate(Fx, lab)

    Fx_c = xai_feats(plain, Xte, predict(plain, Xte))
    Fp_c = pixel_feats(Xte)
    nx = Norm(mlp_scores(det_x, Fx_c), args.fpr)
    tx = thr_at_fpr(tree_x.predict_proba(Fx_c)[:, 1], args.fpr)
    tp = thr_at_fpr(tree_p.predict_proba(Fp_c)[:, 1], args.fpr)
    print("detectors fitted\n")

    configs = [
        ("CW only        (no stealth)", 0.0, 0.0),
        ("detector score (prev best) ", 1.0, 0.0),
        ("disagreement   (NEW, blind)", 0.0, 1.0),
        ("disagreement   (NEW, x10)  ", 0.0, 10.0),
        ("both                       ", 1.0, 1.0),
        ("both  det-heavy            ", 1.0, 0.1),
    ]

    hdr = (f"{'objective':>28} {'succ':>7} | {'dis err':>8} | "
           f"{'XAI ev':>7} {'PIX ev':>7}")
    print(hdr)
    print("-" * len(hdr))

    for name, ld, ls in configs:
        A = disagree_attack(surro, det_x, nx, Xte, Yte, eps=args.eps,
                            k=args.k, lam_det=ld, lam_dis=ls,
                            steps=args.steps, restarts=args.restarts).float()
        pa = predict(plain, A)
        mis = (pa != Yte).numpy()
        sx = tree_x.predict_proba(xai_feats(plain, A, pa))[:, 1]
        sp = tree_p.predict_proba(pixel_feats(A))[:, 1]

        # agreement error on successful attacks (maps need grad enabled)
        idx = torch.from_numpy(mis).nonzero().squeeze(1)[:256]
        if len(idx):
            with torch.enable_grad():
                dc = disagree_vec(all_maps(plain, Xte[idx].to(DEVICE),
                                           Yte[idx].to(DEVICE), False)).detach()
                da = disagree_vec(all_maps(plain, A[idx].to(DEVICE),
                                           pa[idx].to(DEVICE), False)).detach()
            derr = ((da - dc) ** 2).sum(1).mean().item()
        else:
            derr = float("nan")

        print(f"{name:>28} {mis.mean() * 100:>6.1f}% | {derr:>8.4f} | "
              f"{(mis & (sx < tx)).mean() * 100:>6.1f}% "
              f"{(mis & (sp < tp)).mean() * 100:>6.1f}%", flush=True)

    print("\n'dis err': squared distance between adversarial and clean")
    print("agreement vectors, on successful attacks.")


if __name__ == "__main__":
    main()
