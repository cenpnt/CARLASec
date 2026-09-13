"""ADAPTIVE ATTACK: the attacker knows the XAI detector exists and evades it.

Every result so far assumed an attacker who ignores the defence. That is exactly
the setting where detection defences look excellent and then die: Carlini and
Wagner broke ten detectors this way, Tramer et al. broke thirteen more, several
of which had already run their own adaptive evaluation.

Our low-frequency attacker already evades the PIXEL detector, by being smooth.
Here it additionally evades the XAI detector, by keeping its attribution maps
statistically close to those of the clean image:

    L = margin loss (misclassify) + lambda * || f_attr(x_adv) - f_attr(x_clean) ||^2

f_attr is differentiable, so we backpropagate THROUGH THE EXPLANATION into the
perturbation (second-order gradients). To keep that tractable the evasion term
uses Saliency and Input x Gradient (one backward pass each); the DETECTOR being
evaluated still uses the full IG + IxG + Saliency feature set, so this is an
honest surrogate-transfer attack rather than attacking the exact scorer.

PROTOCOL. The detector is trained on clean vs NON-ADAPTIVE attacks (what a
deployer would have), then tested on clean vs ADAPTIVE attacks (what they would
actually face). Train and test are split by source image so none straddles.

Sweeping lambda traces the attacker cost curve: evasion should buy a lower
detection rate but cost attack success. That trade-off is the real deliverable.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe adaptive_attack.py
"""
import argparse

import numpy as np
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from model import load_trained, IMG_SIZE
from xai_detect import attribute_all, DATA, CKPT, DEVICE
from control_baseline import image_features


# ------------------------------------------------------------------ features
def diff_map_feats(a):
    """Differentiable version of the detector's attribution-map statistics."""
    m = a.abs().sum(1)
    B, H, W = m.shape
    flat = m.reshape(B, -1)
    total = flat.sum(1) + 1e-12
    p = flat / total.unsqueeze(1)
    ent = -(p * (p + 1e-12).log()).sum(1)
    srt = p.sort(dim=1, descending=True).values
    t1 = srt[:, :max(1, int(0.01 * H * W))].sum(1)
    t5 = srt[:, :max(1, int(0.05 * H * W))].sum(1)
    t20 = srt[:, :max(1, int(0.20 * H * W))].sum(1)
    tvh = (m[:, 1:, :] - m[:, :-1, :]).abs().mean((1, 2))
    tvw = (m[:, :, 1:] - m[:, :, :-1]).abs().mean((1, 2))
    mean_m = flat.mean(1) + 1e-12
    c0, c1 = H // 4, 3 * H // 4
    centre = m[:, c0:c1, c0:c1].sum((1, 2)) / total
    return torch.stack([total.log(), flat.mean(1), flat.std(1), flat.max(1).values,
                        ent, t1, t5, t20, tvh / mean_m, tvw / mean_m, centre], 1)


def diff_attr_features(model, x, create_graph):
    """Saliency + Input x Gradient features, differentiable with respect to x."""
    logits = model(x)
    tgt = logits.argmax(1).detach()
    sel = logits.gather(1, tgt[:, None]).sum()
    g = torch.autograd.grad(sel, x, create_graph=create_graph)[0]
    ixg = g * x
    u = g.abs().sum(1).flatten(1)
    v = ixg.abs().sum(1).flatten(1)
    un = (u - u.mean(1, keepdim=True)) / (u.std(1, keepdim=True) + 1e-12)
    vn = (v - v.mean(1, keepdim=True)) / (v.std(1, keepdim=True) + 1e-12)
    corr = (un * vn).mean(1, keepdim=True)
    return torch.cat([diff_map_feats(g), diff_map_feats(ixg), corr], 1), logits


# ------------------------------------------------------------------ attack
def adaptive_lowfreq(model, x, y, eps, k, lam, scale, steps=200, lr=0.15, bs=128):
    """Smooth attack that also matches the clean attribution statistics."""
    out = []
    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEVICE)
        yb = y[i:i + bs].to(DEVICE)
        H = xb.shape[-1]

        xc = xb.clone().requires_grad_(True)
        f_clean, _ = diff_attr_features(model, xc, create_graph=False)
        f_clean = f_clean.detach()

        z = torch.zeros(len(xb), 3, k, k, device=DEVICE, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
        need = lam > 0
        for _ in range(steps):
            delta = eps * torch.tanh(
                Fn.interpolate(z, size=(H, H), mode="bilinear", align_corners=False))
            xa = (xb + delta).clamp(0, 1)
            # At lambda=0 there is no evasion term, so skip attributions entirely.
            # (Calling them with create_graph=False frees the graph that
            # loss.backward() still needs, and is wasted work besides.)
            if need:
                f_adv, logits = diff_attr_features(model, xa, create_graph=True)
            else:
                logits = model(xa)

            true_logit = logits.gather(1, yb[:, None]).squeeze(1)
            other = logits.scatter(1, yb[:, None], -1e9).max(1).values
            loss = (true_logit - other).clamp(min=-0.10).mean()
            if need:
                loss = loss + lam * (((f_adv - f_clean) / scale) ** 2).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()

        with torch.no_grad():
            delta = eps * torch.tanh(
                Fn.interpolate(z, size=(H, H), mode="bilinear", align_corners=False))
            out.append((xb + delta).clamp(0, 1).cpu())
    return torch.cat(out)


def predict(model, x, bs=512):
    with torch.no_grad():
        return torch.cat([model(x[i:i + bs].to(DEVICE)).argmax(1).cpu()
                          for i in range(0, len(x), bs)])


def feats(model, x, pred, bs=512):
    fx, _ = attribute_all(model, x, pred)
    fi = np.concatenate([image_features(x[i:i + bs]) for i in range(0, len(x), bs)])
    return fi, fx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--eps", type=float, default=0.06)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--lams", type=float, nargs="+",
                    default=[0.0, 0.3, 1.0, 3.0, 10.0, 30.0])
    args = ap.parse_args()

    model, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    print(f"classifier: {ckpt.get('backbone')} @ {img_size}px, "
          f"clean acc {ckpt['test_acc']*100:.2f}%")
    print(f"attack: low-frequency k={args.k}, eps={args.eps}, {args.steps} steps\n",
          flush=True)

    tf = transforms.Compose([transforms.Resize((img_size, img_size)), transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)
    X, Y = [], []
    for a, b in DataLoader(ds, batch_size=512, num_workers=8):
        X.append(a); Y.append(b)
    X, Y = torch.cat(X), torch.cat(Y)
    keep = (predict(model, X) == Y).nonzero().squeeze(1)
    gen = torch.Generator().manual_seed(0)
    keep = keep[torch.randperm(len(keep), generator=gen)[:args.n]]
    X, Y = X[keep], Y[keep]
    n = len(X)
    print(f"{n} correctly-classified clean images", flush=True)

    perm = torch.randperm(n, generator=torch.Generator().manual_seed(7))
    tr_i, te_i = perm[:int(0.7 * n)], perm[int(0.7 * n):]
    set_tr = set(tr_i.tolist())

    xc = X[:256].to(DEVICE).requires_grad_(True)
    f0, _ = diff_attr_features(model, xc, create_graph=False)
    scale = f0.detach().std(0).clamp(min=1e-6)

    print("clean features ...", flush=True)
    Fi_cln, Fx_cln = feats(model, X, predict(model, X))

    results = {}
    samples = {}
    for lam in args.lams:
        Xadv = adaptive_lowfreq(model, X, Y, args.eps, args.k, lam, scale, steps=args.steps)
        pa = predict(model, Xadv)
        ok = (pa != Y)
        succ = ok.float().mean().item()
        idx = ok.nonzero().squeeze(1)
        if len(idx) < 100:
            print(f"lambda={lam:<5} success {succ*100:.1f}%  too few", flush=True)
            continue
        Fi_adv, Fx_adv = feats(model, Xadv[idx], pa[idx])
        results[lam] = (succ, idx, Fi_adv, Fx_adv)
        # keep a few successful examples per lambda for the figures
        samples[lam] = (Xadv[idx[:6]].clone(), idx[:6].clone(), pa[idx[:6]].clone())
        print(f"lambda={lam:<5} attack success {succ*100:5.1f}%  ({len(idx)} images)",
              flush=True)

    if 0.0 not in results:
        print("baseline (lambda=0) failed; cannot train the detector")
        return

    b_succ, b_idx, b_Fi, b_Fx = results[0.0]
    bm_tr = np.array([int(i) in set_tr for i in b_idx.tolist()])

    rows = []
    print(f"\n{'lambda':>7} {'succ':>7} {'n':>6} | {'IMAGE AUC':>10} {'XAI AUC':>9}   note")
    print("-" * 68)
    for lam in sorted(results):
        succ, idx, Fi_adv, Fx_adv = results[lam]
        m_te = ~np.array([int(i) in set_tr for i in idx.tolist()])
        row = []
        for which in ("img", "xai"):
            Fc = Fx_cln if which == "xai" else Fi_cln
            Fa = Fx_adv if which == "xai" else Fi_adv
            Fb = b_Fx if which == "xai" else b_Fi
            Xtr = np.concatenate([Fc[tr_i.numpy()], Fb[bm_tr]])
            ytr = np.concatenate([np.zeros(len(tr_i)), np.ones(int(bm_tr.sum()))])
            det = HistGradientBoostingClassifier(max_iter=400, random_state=0)
            det.fit(Xtr, ytr)
            Xte = np.concatenate([Fc[te_i.numpy()], Fa[m_te]])
            yte = np.concatenate([np.zeros(len(te_i)), np.ones(int(m_te.sum()))])
            row.append(roc_auc_score(yte, det.predict_proba(Xte)[:, 1]))
        note = ("non-adaptive baseline" if lam == 0
                else f"success cost {(succ - b_succ) * 100:+.1f}pp")
        print(f"{lam:>7.1f} {succ*100:>6.1f}% {len(idx):>6} | "
              f"{row[0]:>10.4f} {row[1]:>9.4f}   {note}", flush=True)
        rows.append([lam, succ, len(idx), row[0], row[1]])

    save = {"rows": np.array(rows), "clean": X[:6].numpy(),
            "clean_idx": np.arange(6)}
    for lam, (xs, ids, pr) in samples.items():
        save[f"adv_{lam}"] = xs.numpy()
        save[f"idx_{lam}"] = ids.numpy()
        save[f"pred_{lam}"] = pr.numpy()
        save[f"src_{lam}"] = X[ids].numpy()
        save[f"true_{lam}"] = Y[ids].numpy()
    np.savez(r"C:\Users\s4990998\xai_work\adaptive_results.npz", **save)
    print("\nsaved C:\\Users\\s4990998\\xai_work\\adaptive_results.npz")
    print("Detector trained on clean vs NON-ADAPTIVE attacks, tested on clean vs")
    print("the adaptive attack at each lambda. Falling XAI AUC means evasion works.")


if __name__ == "__main__":
    main()
