"""Differentiable mirror of `control_baseline.image_features` (23 features).

The two saturation fractions are threshold counts and carry no gradient.
"""
import numpy as np
import torch

from control_baseline import image_features

__all__ = ["pixel_features_t", "test_pixel_feat_diff"]


def pixel_features_t(x):
    """Torch mirror of image_features. Returns (B,23) with the graph intact."""
    g = x.mean(1)
    H, W = g.shape[-2:]
    f = []

    f.append(x.flatten(2).std(2))
    f.append(x.flatten(2).mean(2))

    tv_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs()
    tv_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs()
    f.append(tv_h.mean((2, 3)))
    f.append(tv_w.mean((2, 3)))
    f.append(torch.stack([tv_h.mean((1, 2, 3)), tv_w.mean((1, 2, 3)),
                          tv_h.std((1, 2, 3)), tv_w.std((1, 2, 3))], 1))

    lap = (g[:, 1:-1, 2:] + g[:, 1:-1, :-2] + g[:, 2:, 1:-1]
           + g[:, :-2, 1:-1] - 4 * g[:, 1:-1, 1:-1])
    f.append(torch.stack([lap.abs().mean((1, 2)), lap.pow(2).mean((1, 2)),
                          lap.abs().max(1).values.max(1).values], 1))

    mag = torch.fft.fftshift(torch.fft.fft2(g).abs(), dim=(-2, -1))
    total = mag.sum((1, 2)) + 1e-9
    ch, cw = H // 4, W // 4
    centre = mag[:, H // 2 - ch:H // 2 + ch, W // 2 - cw:W // 2 + cw].sum((1, 2))
    f.append(((total - centre) / total).unsqueeze(1))

    hp = (g[:, 1:, :] - g[:, :-1, :]).flatten(1)
    med = hp.median(1).values.unsqueeze(1)
    f.append((hp - med).abs().median(1).values.unsqueeze(1))

    # no gradient
    f.append(torch.stack([(x <= 1e-6).float().mean((1, 2, 3)),
                          (x >= 1 - 1e-6).float().mean((1, 2, 3))], 1))

    return torch.cat(f, 1)


def test_pixel_feat_diff(x, atol=1e-4):
    mine = pixel_features_t(x).detach().float().cpu().numpy()
    ref = image_features(x)
    assert mine.shape == ref.shape, (mine.shape, ref.shape)
    bad = np.abs(mine - ref) > atol * (1.0 + np.abs(ref))
    return mine.shape[1], int(bad.sum()), float(np.abs(mine - ref).max())
