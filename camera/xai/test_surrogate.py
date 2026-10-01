"""Checks that attribution maps have a usable gradient under each ReLU surrogate:
  1. the "adv2" surrogate leaves the forward pass unchanged
  2. the hand-rolled maps in attrib_diff match Captum
  3. ||dD/dz|| per attribution method and surrogate, against plain ReLU

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe test_surrogate.py
"""
import argparse

import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from captum.attr import IntegratedGradients, InputXGradient, Saliency

from model import load_trained, IMG_SIZE
from xai_detect import DATA, CKPT, DEVICE
from relu_surrogate import swap_relu, count_relu
from attrib_diff import ATTRIB, map_distance

KINDS = ["relu", "adv2", "adv2-literal", "softplus"]
IG_STEPS = 8


def load_batch(model, img_size, n):
    """n correctly-classified test images, deterministically chosen."""
    tf = transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)
    X, Y = [], []
    for x, y in DataLoader(ds, batch_size=512, num_workers=4):
        X.append(x)
        Y.append(y)
        if sum(len(t) for t in X) >= 4096:
            break
    X, Y = torch.cat(X), torch.cat(Y)
    with torch.no_grad():
        pred = model(X[:4096].to(DEVICE)).argmax(1).cpu()
    keep = (pred == Y[:4096]).nonzero().squeeze(1)[:n]
    return X[keep].to(DEVICE), Y[keep].to(DEVICE)


def attrib_loss_grad(model, x, y, kind, k, eps, ig_steps=IG_STEPS, seed=0):
    """||dD/dz|| for attribution distance D and k x k code z. z starts away
    from zero, because D = 0 is a minimum there and the gradient would vanish."""
    fn = ATTRIB[kind]
    extra = {"steps": ig_steps} if kind == "ig" else {}

    with torch.enable_grad():
        a_clean = fn(model, x.clone(), y, create_graph=False, **extra).detach()

        g = torch.Generator(device="cpu").manual_seed(seed)
        z = (0.5 * torch.randn(len(x), 3, k, k, generator=g)).to(DEVICE)
        z.requires_grad_(True)

        delta = eps * torch.tanh(
            Fn.interpolate(z, size=x.shape[-2:], mode="bilinear",
                           align_corners=False))
        x_adv = (x + delta).clamp(0, 1)

        a_adv = fn(model, x_adv, y, create_graph=True, **extra)
        loss = map_distance(a_adv, a_clean).sum()
        (gz,) = torch.autograd.grad(loss, z)

    return gz.norm().item(), loss.item() / len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--eps", type=float, default=0.03)
    ap.add_argument("--beta", type=float, default=50.0)
    args = ap.parse_args()

    base, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    print(f"classifier: {ckpt.get('backbone')} @ {img_size}px, "
          f"clean acc {ckpt['test_acc'] * 100:.2f}%")
    print(f"batch n={args.n}, k={args.k}, eps={args.eps}, IG steps={IG_STEPS}\n")

    X, Y = load_batch(base, img_size, args.n)
    with torch.no_grad():
        logits_relu = base(X).clone()

    # ---- 1. forward identity and accuracy under each surrogate -------------
    print("=" * 72)
    print("1. FORWARD IDENTITY  (adv2 must be exact: same forward, same accuracy)")
    print("=" * 72)
    print(f"{'surrogate':>14} {'swapped':>8} {'max|dlogit|':>13} {'batch acc':>10}")
    print("-" * 72)
    for kind in KINDS:
        m, _ = load_trained(CKPT, DEVICE)
        n_sw = swap_relu(m, kind=kind, beta=args.beta)
        relu_left, surr = count_relu(m)
        with torch.no_grad():
            lg = m(X)
        drift = (lg - logits_relu).abs().max().item()
        acc = (lg.argmax(1) == Y).float().mean().item()
        print(f"{kind:>14} {n_sw:>8} {drift:>13.3e} {acc * 100:>9.1f}%"
              + ("   <- leftover nn.ReLU!" if kind != "relu" and relu_left else ""))
    print()

    # ---- 2. hand-rolled maps vs Captum on the unmodified model -------------
    print("=" * 72)
    print("2. CAPTUM AGREEMENT  (attrib_diff vs captum, unmodified model)")
    print("=" * 72)
    m, _ = load_trained(CKPT, DEVICE)
    xg = X.clone().requires_grad_(True)
    ref = {
        "sal": Saliency(m).attribute(xg, target=Y, abs=False),
        "ixg": InputXGradient(m).attribute(xg, target=Y),
        "ig": IntegratedGradients(m).attribute(
            xg, target=Y, n_steps=IG_STEPS, method="riemann_middle",
            internal_batch_size=len(X)),
    }
    print(f"{'method':>8} {'max abs diff':>14} {'rel diff':>12}")
    print("-" * 72)
    for name in ["sal", "ixg", "ig"]:
        extra = {"steps": IG_STEPS} if name == "ig" else {}
        mine = ATTRIB[name](m, X.clone(), Y, create_graph=False, **extra)
        d = (mine - ref[name]).abs()
        rel = (d.max() / ref[name].abs().max().clamp(min=1e-12)).item()
        print(f"{name:>8} {d.max().item():>14.3e} {rel:>12.3e}"
              + ("   OK" if rel < 1e-3 else "   <- CHECK"))
    print()

    # ---- 3. the gate -------------------------------------------------------
    print("=" * 72)
    print("3. THE GATE  ||dD/dz||, D = L1 distance between attribution maps")
    print("=" * 72)
    print(f"{'method':>8} {'surrogate':>14} {'||dD/dz||':>13} {'vs relu':>10} {'D':>9}")
    print("-" * 72)
    verdict = {}
    for name in ["sal", "ixg", "ig"]:
        base_norm = None
        for kind in KINDS:
            m, _ = load_trained(CKPT, DEVICE)
            swap_relu(m, kind=kind, beta=args.beta)
            try:
                gn, dval = attrib_loss_grad(m, X, Y, name, args.k, args.eps)
            except RuntimeError as e:
                print(f"{name:>8} {kind:>14}   RuntimeError: {str(e)[:40]}")
                continue
            if kind == "relu":
                base_norm = gn
                ratio = "-"
            else:
                ratio = ("inf" if base_norm == 0 else f"{gn / base_norm:.1f}x")
                verdict[(name, kind)] = (gn, base_norm)
            print(f"{name:>8} {kind:>14} {gn:>13.3e} {ratio:>10} {dval:>9.4f}")
        print("-" * 72)

    # ---- 4. read the result -----------------------------------------------
    print()
    print("=" * 72)
    print("VERDICT")
    print("=" * 72)
    # Usable: clear of numerical noise and at least 2x plain ReLU. IxG and IG
    # keep a first-order path through their input factor, so their ratios are
    # modest; Saliency's is infinite because plain ReLU gives exactly zero.
    usable = {}
    for name in ["sal", "ixg", "ig"]:
        g_adv2, g_relu = verdict.get((name, "adv2"), (0.0, 0.0))
        u = g_adv2 > 1e-3 and (g_relu == 0.0 or g_adv2 > 2.0 * g_relu)
        usable[name] = u
        ratio = "inf" if g_relu == 0 else f"{g_adv2 / g_relu:.1f}x"
        print(f"  {name:>4}: relu {g_relu:.3e} -> adv2 {g_adv2:.3e}  "
              f"({ratio:>6})  " + ("usable" if u else "NOT usable"))
    print()
    if usable.get("sal") and usable.get("ixg"):
        print("  PASS: the attribution term is optimisable.")
    else:
        print("  FAIL: no usable gradient through the attribution maps.")


if __name__ == "__main__":
    main()
