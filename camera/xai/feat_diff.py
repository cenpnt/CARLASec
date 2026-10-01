"""Differentiable versions of the XAI detector's 54 features. Run it to check
them, and the pixel mirror, against the numpy versions.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe feat_diff.py
"""
import numpy as np
import torch

from xai_detect import map_features, disagreement_features

__all__ = ["map_features_t", "disagreement_features_t", "features_t",
           "test_feat_diff"]


def map_features_t(attr):
    """Torch mirror of xai_detect.map_features. Returns (B,15) with graph."""
    a = attr.abs().sum(1)
    B, H, W = a.shape
    flat = a.reshape(B, -1)
    total = flat.sum(1) + 1e-12

    p = flat / total.unsqueeze(1)
    entropy = -(p * (p + 1e-12).log()).sum(1)

    srt = p.sort(dim=1, descending=True).values
    top1 = srt[:, : max(1, int(0.01 * H * W))].sum(1)
    top5 = srt[:, : max(1, int(0.05 * H * W))].sum(1)
    top20 = srt[:, : max(1, int(0.20 * H * W))].sum(1)

    tv_h = (a[:, 1:, :] - a[:, :-1, :]).abs().mean((1, 2))
    tv_w = (a[:, :, 1:] - a[:, :, :-1]).abs().mean((1, 2))
    mean_a = flat.mean(1) + 1e-12
    tv = (tv_h + tv_w) / mean_a

    c0, c1 = H // 4, 3 * H // 4
    centre = a[:, c0:c1, c0:c1].sum((1, 2)) / total

    s = attr.sum(1).reshape(B, -1)
    pos_frac = (s > 0).float().mean(1)        # no gradient

    return torch.stack([
        total.log(), flat.mean(1), flat.std(1), flat.max(1).values,
        entropy, top1, top5, top20, tv, tv_h / mean_a, tv_w / mean_a,
        centre, pos_frac, s.mean(1), s.std(1),
    ], dim=1)


def disagreement_features_t(maps):
    """Torch mirror of xai_detect.disagreement_features. Returns (B,9):
    [pearson, cosine, top-10% IoU] for each pair of methods."""
    names = sorted(maps)
    mags = {n: maps[n].abs().sum(1).flatten(1) for n in names}
    out = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            u, v = mags[names[i]], mags[names[j]]
            un = (u - u.mean(1, keepdim=True)) / (u.std(1, keepdim=True) + 1e-12)
            vn = (v - v.mean(1, keepdim=True)) / (v.std(1, keepdim=True) + 1e-12)
            pearson = (un * vn).mean(1)
            cos = torch.nn.functional.cosine_similarity(u, v, dim=1)

            k = max(1, int(0.10 * u.shape[1]))
            tu = u.topk(k, dim=1).indices
            tv_ = v.topk(k, dim=1).indices
            mu = torch.zeros_like(u, dtype=torch.bool).scatter_(1, tu, True)
            mv = torch.zeros_like(v, dtype=torch.bool).scatter_(1, tv_, True)
            iou = (mu & mv).sum(1).float() / (mu | mv).sum(1).float().clamp(min=1)

            out += [pearson, cos, iou]      # iou carries no gradient
    return torch.stack(out, dim=1)


def features_t(maps):
    """Full 54-feature vector, in `xai_detect.attribute_all` order."""
    per_method = [map_features_t(maps[n]) for n in sorted(maps)]
    return torch.cat(per_method + [disagreement_features_t(maps)], dim=1)


def test_feat_diff(maps, atol=1e-4):
    """Compare against the numpy features the detector is fitted on.
    Returns (n_features, n_mismatches, max_abs_diff)."""
    mine = features_t(maps).detach().float().cpu().numpy()
    ref = np.concatenate(
        [map_features(maps[n]) for n in sorted(maps)]
        + [disagreement_features(maps)], axis=1)
    assert mine.shape == ref.shape, (mine.shape, ref.shape)
    bad = np.abs(mine - ref) > atol * (1.0 + np.abs(ref))
    return mine.shape[1], int(bad.sum()), float(np.abs(mine - ref).max())


if __name__ == "__main__":
    from model import load_trained, IMG_SIZE
    from xai_detect import attribute_all, CKPT, DEVICE
    from aim1_attack import predict, load_data
    from pixel_feat_diff import test_pixel_feat_diff

    model, ckpt = load_trained(CKPT, DEVICE)
    X, _ = load_data(model, ckpt.get("img_size", IMG_SIZE), 32)
    _, maps = attribute_all(model, X, predict(model, X))
    for name, res in [
            ("XAI", test_feat_diff({n: v.to(DEVICE) for n, v in maps.items()})),
            ("pixel", test_pixel_feat_diff(X.to(DEVICE)))]:
        nf, nbad, mx = res
        print(f"{name:>5}: {nf} features, {nbad} mismatches, "
              f"max abs diff {mx:.3e}")
