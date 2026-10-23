"""Random camera views of a sign crop: viewing angle, rotation, distance
(scale and resolution loss), framing offset, focus blur, brightness and
contrast. Every image in a batch gets its own parameters and every step is
differentiable, so the same views serve evaluation and Expectation over
Transformation (Athalye et al. 2018) inside an attack.

The ranges sit just beyond the classifier's training augmentation
(train_sota.py: 10 deg, 10% shift, 0.9-1.1 scale, 0.25 brightness/contrast),
adding perspective, blur and resolution loss that training never saw.
"""
import math

import torch
import torch.nn.functional as Fn

from xai_detect import DEVICE

RANGES = {
    "persp": 0.25,          # max inward corner shift, fraction of half-width
    "rot": 10.0,            # degrees
    "scale": (0.85, 1.15),
    "shift": 0.08,          # fraction of half-width
    "res": (0.5, 1.0),      # downsampled side / full side
    "blur": (0.0, 1.0),     # Gaussian sigma in pixels
    "bright": (0.7, 1.3),
    "contrast": (0.7, 1.3),
}
RES_BINS = 6


def _u(n, lo, hi, g):
    return lo + (hi - lo) * torch.rand(n, generator=g)


def sample(n, seed=None, ranges=RANGES):
    """Parameters of n random views, on CPU. A fixed seed gives the same views
    to every attack."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None
    r = ranges
    return {
        "persp": r["persp"] * torch.rand(n, 4, 2, generator=g),
        "rot": _u(n, -r["rot"], r["rot"], g) * math.pi / 180,
        "scale": _u(n, *r["scale"], g),
        "shift": _u(n * 2, -r["shift"], r["shift"], g).view(n, 2),
        "res": _u(n, *r["res"], g),
        "blur": _u(n, *r["blur"], g),
        "bright": _u(n, *r["bright"], g),
        "contrast": _u(n, *r["contrast"], g),
    }


def subset(p, idx):
    return {k: v[idx] for k, v in p.items()}


def _homography(src, dst):
    """3x3 maps taking each dst point to its src point, solved per image."""
    n = src.shape[0]
    x, y = dst[..., 0], dst[..., 1]
    u, v = src[..., 0], src[..., 1]
    z, o = torch.zeros_like(x), torch.ones_like(x)
    A = torch.cat([torch.stack([x, y, o, z, z, z, -u * x, -u * y], -1),
                   torch.stack([z, z, z, x, y, o, -v * x, -v * y], -1)], 1)
    b = torch.cat([u, v], 1)
    h = torch.linalg.solve(A, b)
    return torch.cat([h, torch.ones(n, 1)], 1).view(n, 3, 3)


def _geometry(x, p):
    """Where the sign's corners land in the camera crop, then a bilinear warp
    with edge pixels continuing the background."""
    n, _, H, W = x.shape
    src = torch.tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]]).expand(n, 4, 2)
    c, s = p["rot"].cos(), p["rot"].sin()
    R = torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], 1)
    dst = (src @ R.transpose(1, 2)) * p["scale"].view(n, 1, 1) \
        + p["shift"].view(n, 1, 2) - src.sign() * p["persp"]
    Hm = _homography(src, dst).to(x.device)
    ys, xs = torch.meshgrid(torch.linspace(-1 + 1 / H, 1 - 1 / H, H, device=x.device),
                            torch.linspace(-1 + 1 / W, 1 - 1 / W, W, device=x.device),
                            indexing="ij")
    pts = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(1, -1, 3)
    q = pts @ Hm.transpose(1, 2)
    grid = (q[..., :2] / q[..., 2:]).view(n, H, W, 2)
    return Fn.grid_sample(x, grid, mode="bilinear", padding_mode="border",
                          align_corners=False)


def _resolution(x, p):
    """Down- and up-sample to mimic a distant sign; sizes binned so each bin
    is one batched call."""
    H = x.shape[-1]
    lo, hi = RANGES["res"]
    sizes = torch.linspace(lo * H, hi * H, RES_BINS).round().long()
    pick = (p["res"] - lo) / (hi - lo + 1e-9) * (RES_BINS - 1)
    pick = pick.round().long().clamp(0, RES_BINS - 1)
    out = x.clone()
    for b in pick.unique().tolist():
        idx = (pick == b).nonzero().squeeze(1).to(x.device)
        s = int(sizes[b])
        if s >= H:
            continue
        small = Fn.interpolate(x[idx], size=(s, s), mode="bilinear",
                               antialias=True, align_corners=False)
        out[idx] = Fn.interpolate(small, size=(H, H), mode="bilinear",
                                  align_corners=False)
    return out


def _blur(x, p, k=5):
    n, C, H, W = x.shape
    sig = p["blur"].clamp(min=1e-3).to(x.device).view(n, 1)
    t = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-t.view(1, -1) ** 2 / (2 * sig ** 2))
    g = g / g.sum(1, keepdim=True)
    ker = (g.view(n, k, 1) * g.view(n, 1, k)).repeat_interleave(C, 0)
    xp = Fn.pad(x.reshape(1, n * C, H, W), (k // 2,) * 4, mode="reflect")
    return Fn.conv2d(xp, ker.unsqueeze(1), groups=n * C).view(n, C, H, W)


def _photometric(x, p):
    n = x.shape[0]
    b = p["bright"].to(x.device).view(n, 1, 1, 1)
    c = p["contrast"].to(x.device).view(n, 1, 1, 1)
    gray = (0.299 * x[:, 0] + 0.587 * x[:, 1] + 0.114 * x[:, 2])
    m = gray.mean((1, 2)).view(n, 1, 1, 1)
    return (((x * b) - m * b) * c + m * b).clamp(0, 1)


def apply(x, p):
    """Camera view of x (n,3,H,W in [0,1]) under parameters p."""
    x = _geometry(x, p)
    x = _resolution(x, p)
    x = _blur(x, p)
    return _photometric(x, p)


def views(x, k, seed=0, bs=256):
    """k fixed views of every image in x (CPU in, CPU out), view-major: rows
    j*len(x) to (j+1)*len(x) hold view j."""
    out = []
    for j in range(k):
        p = sample(len(x), seed=seed * 1000 + j)
        for i in range(0, len(x), bs):
            with torch.no_grad():
                out.append(apply(x[i:i + bs].to(DEVICE), subset(p, slice(i, i + bs))).cpu())
    return torch.cat(out)
