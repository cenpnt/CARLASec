"""Detector-blind attack variants against the XAI, pixel and OR detectors:
ADV^2-style map alignment, the agreement attack, and full-fingerprint matching.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u aim1_improve.py
"""
import argparse
import os

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import CKPT, DEVICE
from relu_surrogate import swap_relu
from attrib_diff import magnitude
from feat_diff import features_t
from pixel_feat_diff import pixel_features_t
from aim1_attack import attack as plain_attack, cw_margin, upsample, predict, load_data
from aim1_adaptive import all_maps, thr_at_fpr, xai_feats, pixel_feats
from aim1_headline import or_thresholds

SOFT_TOPK_T = 0.1          # temperature of the soft top-k mask, in units of map std
POS_FRAC = [12, 27, 42]    # pos_frac of each method
IOU = [47, 50, 53]         # top-10% IoU of each method pair
AGREE6 = [45, 46, 48, 49, 51, 52]
AGREE9 = list(range(45, 54))
PIX_LIVE = list(range(21))  # the two saturation fractions carry no gradient


def soft_topk_mask(u, frac=0.10):
    """Sigmoid relaxation of the top-`frac` mask of each row of u."""
    k = max(1, int(frac * u.shape[1]))
    thr = u.topk(k, dim=1).values[:, -1:].detach()
    scale = u.std(1, keepdim=True).detach() * SOFT_TOPK_T + 1e-12
    return torch.sigmoid((u - thr) / scale)


def xai_fingerprint(maps):
    """The 54 detector features with the four step-function features replaced
    by smooth stand-ins, so every entry carries gradient."""
    F = features_t(maps)
    names = sorted(maps)
    soft = {}
    for m, n in enumerate(names):
        s = maps[n].sum(1).flatten(1)
        scale = s.std(1, keepdim=True).detach() * SOFT_TOPK_T + 1e-12
        soft[POS_FRAC[m]] = torch.sigmoid(s / scale).mean(1)
    mags = {n: maps[n].abs().sum(1).flatten(1) for n in names}
    p = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            mu = soft_topk_mask(mags[names[i]])
            mv = soft_topk_mask(mags[names[j]])
            soft[IOU[p]] = (torch.minimum(mu, mv).sum(1)
                            / torch.maximum(mu, mv).sum(1).clamp(min=1e-6))
            p += 1
    cols = [soft.get(c, F[:, c]) for c in range(F.shape[1])]
    return torch.stack(cols, dim=1)


def stealth_terms(model, x_adv, tgt, ref, w, sd_x, sd_p):
    """Per-sample stealth penalty. `w` weights the terms; `ref` holds the
    clean-image references."""
    term = torch.zeros(len(x_adv), device=DEVICE)
    need_maps = any(w.get(k, 0) > 0 for k in
                    ("adv2_sal", "adv2_all", "agree6raw", "agree6", "agree9",
                     "xai"))
    if need_maps:
        maps = all_maps(model, x_adv, tgt, create_graph=True)
    if w.get("adv2_sal", 0) > 0:
        term = term + w["adv2_sal"] * (magnitude(maps["sal"])
                                       - ref["mag"]["sal"]).abs().sum(1)
    if w.get("adv2_all", 0) > 0:
        term = term + w["adv2_all"] * sum(
            (magnitude(maps[n]) - ref["mag"][n]).abs().sum(1) for n in maps) / 3
    if any(w.get(k, 0) > 0 for k in ("agree6raw", "agree6", "agree9", "xai")):
        raw = xai_fingerprint(maps) - ref["fx"]
        if w.get("agree6raw", 0) > 0:      # the original aim1_disagree loss
            term = term + w["agree6raw"] * (raw[:, AGREE6] ** 2).sum(1)
        fx = raw / sd_x
        if w.get("agree6", 0) > 0:
            term = term + w["agree6"] * (fx[:, AGREE6] ** 2).mean(1)
        if w.get("agree9", 0) > 0:
            term = term + w["agree9"] * (fx[:, AGREE9] ** 2).mean(1)
        if w.get("xai", 0) > 0:
            term = term + w["xai"] * (fx ** 2).mean(1)
    if w.get("pix", 0) > 0:
        fp = (pixel_features_t(x_adv)[:, PIX_LIVE] - ref["fp"]) / sd_p
        term = term + w["pix"] * (fp ** 2).mean(1)
    return term


def edge_weights(x, mode):
    """AdvEdge step weights, as in the authors' notebook: skimage.filters.sobel
    on the grayscale image ("soft"), or 1 where it exceeds 0.1 ("bin", AdvEdge+)."""
    g = torch.nn.functional.pad(x.mean(1, keepdim=True), (1, 1, 1, 1),
                                mode="reflect")
    kh = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]],
                      device=x.device).view(1, 1, 3, 3) / 4
    gh = torch.nn.functional.conv2d(g, kh)
    gv = torch.nn.functional.conv2d(g, kh.transpose(2, 3))
    e = ((gh ** 2 + gv ** 2) / 2).sqrt()
    return (e > 0.1).float() if mode == "bin" else e


def clean_refs(model, xb, yb):
    """Clean-image references for the stealth terms."""
    with torch.enable_grad():
        m0 = all_maps(model, xb.clone(), yb, create_graph=False)
        return {"mag": {k: magnitude(v).detach() for k, v in m0.items()},
                "fx": xai_fingerprint(m0).detach(),
                "fp": pixel_features_t(xb)[:, PIX_LIVE].detach()}


def pgd_attack(model, x, y, eps, w, sd_x, sd_p, edge=None, pre_steps=200,
               steps=200, alpha=1 / 255, kappa=2.0, bs=64):
    """ADV2 / AdvEdge with the authors' PGD procedure: `pre_steps` sign steps on
    the classifier loss, weighted by the edge map if `edge` is set, then `steps`
    unweighted sign steps on classifier loss plus stealth terms. Keeps the
    misclassified iterate with the lowest stealth term."""
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEVICE)
        yb = y[i:i + bs].to(DEVICE)
        n = len(xb)
        E = edge_weights(xb, edge) if edge else 1.0
        ref = clean_refs(model, xb, yb)
        xa = xb.clone()

        def project(v):
            return (xb + (v - xb).clamp(-eps, eps)).clamp(0, 1)

        for _ in range(pre_steps):
            xa.requires_grad_(True)
            loss = cw_margin(model(xa), yb, kappa).sum()
            g, = torch.autograd.grad(loss, xa)
            xa = project(xa.detach() - E * alpha * g.sign())

        best_x = xa.detach().clone()
        best = torch.full((n,), float("inf"), device=DEVICE)
        for _ in range(steps):
            xa.requires_grad_(True)
            logits = model(xa)
            term = stealth_terms(model, xa, logits.argmax(1).detach(), ref, w,
                                 sd_x, sd_p)
            loss = cw_margin(logits, yb, kappa).sum() + term.sum()
            g, = torch.autograd.grad(loss, xa)
            with torch.no_grad():
                mis = logits.argmax(1) != yb
                cand = torch.where(mis, term.detach(),
                                   torch.full_like(best, float("inf")))
                better = cand < best
                best = torch.where(better, cand, best)
                best_x[better] = xa.detach()[better]
            xa = project(xa.detach() - alpha * g.sign())
        with torch.no_grad():
            never = torch.isinf(best)
            best_x[never] = xa.detach()[never]
        out.append(best_x.cpu())
    return torch.cat(out)


def blind_attack(model, x, y, eps, w, sd_x, sd_p, steps=100, lr=0.08,
                 kappa=2.0, bs=64, warm=0.4, restarts=2, seed=0,
                 keep_best=True):
    """CW margin plus the weighted stealth terms. Never queries a detector:
    candidates are ranked by the attack's own objective. keep_best=False
    returns the final iterate of the last restart instead (ablation)."""
    warm_steps = int(warm * steps)
    active = any(v > 0 for v in w.values())
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEVICE)
        yb = y[i:i + bs].to(DEVICE)
        H, n = xb.shape[-1], len(xb)
        ref = clean_refs(model, xb, yb)

        best_z = torch.zeros(n, 3, H, H, device=DEVICE)
        best = torch.full((n,), float("inf"), device=DEVICE)
        seen = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        for r in range(restarts):
            if r == 0:
                z = torch.zeros(n, 3, H, H, device=DEVICE)
            else:
                g = torch.Generator(device="cpu").manual_seed(seed + 613 * r)
                z = (0.5 * torch.randn(n, 3, H, H, generator=g)).to(DEVICE)
            z = z.clone().requires_grad_(True)
            opt = torch.optim.Adam([z], lr=lr)

            for step in range(steps):
                x_adv = (xb + eps * torch.tanh(z)).clamp(0, 1)
                logits = model(x_adv)
                loss = cw_margin(logits, yb, kappa).sum()
                rank = None
                if active and step >= warm_steps:
                    term = stealth_terms(model, x_adv, logits.argmax(1).detach(),
                                         ref, w, sd_x, sd_p)
                    loss = loss + term.sum()
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

        if not keep_best:
            best_z = z.detach()
        with torch.no_grad():
            out.append((xb + eps * torch.tanh(best_z)).clamp(0, 1).cpu())
    return torch.cat(out)


CONFIGS = {
    "CW only":               {},
    "ADV2-single (Sal map)": {"adv2_sal": 10.0},
    "ADV2-all (3 maps)":     {"adv2_all": 10.0},
    "agree6 (current)":      {"agree6raw": 1.0},
    "agree6 (scaled)":       {"agree6": 1.0},
    "agree9 (+soft IoU)":    {"agree9": 1.0},
    "fingerprint XAI":       {"xai": 1.0},
    "fingerprint XAI+pix":   {"xai": 1.0, "pix": 1.0},
    "ADV2-all + agree6":     {"adv2_all": 10.0, "agree6raw": 1.0},
    "ADV2-all + pix":        {"adv2_all": 10.0, "pix": 1.0},
    "ADV2-all + XAI + pix":  {"adv2_all": 10.0, "xai": 1.0, "pix": 1.0},
    # the authors' PGD procedure (pgd_attack); "pgd" names the phase-1 edge mode
    "PGD ADV2 (3 maps)":     {"adv2_all": 10.0, "pgd": None},
    "PGD AdvEdge (3 maps)":  {"adv2_all": 10.0, "pgd": "soft"},
    "PGD AdvEdge+ (3 maps)": {"adv2_all": 10.0, "pgd": "bin"},
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--lam", type=float, nargs="+", default=[1.0],
                    help="multiplier applied to every stealth weight")
    ap.add_argument("--only", nargs="*", default=None,
                    help="substrings of config names to run")
    ap.add_argument("--seed", type=int, default=0, help="image sample and split")
    ap.add_argument("--surrogate", default="adv2", choices=["adv2", "adv2-exact"])
    ap.add_argument("--no-best", action="store_true",
                    help="return the final iterate, not the best one")
    ap.add_argument("--pgd-steps", type=int, default=200,
                    help="phase-2 steps of the PGD procedure")
    args = ap.parse_args()

    plain, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    surro, _ = load_trained(CKPT, DEVICE)
    swap_relu(surro, kind=args.surrogate)

    X, Y = load_data(plain, img_size, args.n, seed=args.seed)
    ntr = len(X) // 2
    Xtr, Ytr, Xte, Yte = X[:ntr], Y[:ntr], X[ntr:], Y[ntr:]
    print(f"{len(Xtr)} detector-train / {len(Xte)} attack-eval, eps={args.eps}, "
          f"steps={args.steps}, restarts={args.restarts}, "
          f"surrogate={args.surrogate}\n", flush=True)

    # detectors: same recipe as aim1_headline
    clf = PyTorchClassifier(model=plain, loss=nn.CrossEntropyLoss(),
                            input_shape=(3, img_size, img_size),
                            nb_classes=NUM_CLASSES, clip_values=(0.0, 1.0),
                            device_type="gpu" if DEVICE == "cuda" else "cpu")
    advs = [plain_attack(plain, Xtr, Ytr, eps=args.eps, k=k, lam=0.0,
                         steps=args.steps).float() for k in (96, 6, 3)]
    advs.append(torch.from_numpy(FastGradientMethod(
        estimator=clf, eps=args.eps, batch_size=256).generate(
            x=Xtr.numpy())).float())
    Fx_tr_c = xai_feats(plain, Xtr, predict(plain, Xtr))
    Fp_tr_c = pixel_feats(Xtr)
    Fx = np.concatenate([Fx_tr_c] + [xai_feats(plain, A, predict(plain, A))
                                     for A in advs])
    Fp = np.concatenate([Fp_tr_c] + [pixel_feats(A) for A in advs])
    lab = np.concatenate([np.zeros(len(Xtr))] + [np.ones(len(A)) for A in advs])
    tree_x = HistGradientBoostingClassifier(max_iter=400,
                                            random_state=0).fit(Fx, lab)
    tree_p = HistGradientBoostingClassifier(max_iter=400,
                                            random_state=0).fit(Fp, lab)
    cx = tree_x.predict_proba(xai_feats(plain, Xte, predict(plain, Xte)))[:, 1]
    cp = tree_p.predict_proba(pixel_feats(Xte))[:, 1]
    tx, tp = thr_at_fpr(cx, 0.05), thr_at_fpr(cp, 0.05)
    to = or_thresholds(cp, cx, 0.05)

    # feature scales from CLEAN training images only: no detector information
    sd_x = torch.as_tensor(Fx_tr_c.std(0) + 1e-6, dtype=torch.float32,
                           device=DEVICE)
    sd_p = torch.as_tensor(Fp_tr_c[:, PIX_LIVE].std(0) + 1e-6,
                           dtype=torch.float32, device=DEVICE)

    hdr = (f"{'attack':>24} {'lam':>5} {'mis':>6} | {'XAI':>6} {'PIXEL':>6} "
           f"{'OR':>6}   (evasion @5% FPR)")
    print(hdr)
    print("-" * len(hdr), flush=True)
    for name, w in CONFIGS.items():
        if args.only and not any(s in name for s in args.only):
            continue
        w = dict(w)
        use_pgd = "pgd" in w
        edge = w.pop("pgd", None)
        for lam in (args.lam if w else [0.0]):
            ws = {k: v * lam for k, v in w.items()}
            if use_pgd:
                A = pgd_attack(surro, Xte, Yte, args.eps, ws, sd_x, sd_p,
                               edge=edge, steps=args.pgd_steps).float()
            else:
                A = blind_attack(surro, Xte, Yte, args.eps, ws, sd_x, sd_p,
                                 steps=args.steps,
                                 restarts=args.restarts,
                                 keep_best=not args.no_best).float()
            pa = predict(plain, A)
            mis = (pa != Yte).numpy()
            sx = tree_x.predict_proba(xai_feats(plain, A, pa))[:, 1]
            sp = tree_p.predict_proba(pixel_feats(A))[:, 1]
            ev_x = (mis & (sx < tx)).mean() * 100
            ev_p = (mis & (sp < tp)).mean() * 100
            ev_o = (mis & (sp <= to[0]) & (sx <= to[1])).mean() * 100
            print(f"{name:>24} {lam:>5.1f} {mis.mean() * 100:>5.1f}% | "
                  f"{ev_x:>5.1f}% {ev_p:>5.1f}% {ev_o:>5.1f}%", flush=True)


if __name__ == "__main__":
    main()
