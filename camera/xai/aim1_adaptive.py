"""Shared helpers for the Aim 1 attacks: attribution maps with graph,
detector features, and evasion thresholds."""
import numpy as np
import torch

from xai_detect import attribute_all, DEVICE
from control_baseline import image_features
from attrib_diff import saliency, integrated_grad

IG_LOOP_STEPS = 4      # inside the attack loop, where the graph is retained


def all_maps(model, x, target, create_graph, ig_steps=IG_LOOP_STEPS):
    """The three attribution maps the detector consumes, optionally with graph."""
    sal = saliency(model, x, target, create_graph=create_graph)
    return {
        "sal": sal,
        "ixg": x * sal,
        "ig": integrated_grad(model, x, target, steps=ig_steps,
                              create_graph=create_graph),
    }


def thr_at_fpr(clean_scores, fpr):
    """Threshold flagging exactly `fpr` of clean inputs (higher = adversarial)."""
    return float(np.quantile(clean_scores, 1.0 - fpr))


def mlp_scores(det, F):
    with torch.no_grad():
        return det(torch.as_tensor(F, dtype=torch.float32,
                                   device=DEVICE)).cpu().numpy()


def xai_feats(model, x, target, bs=64):
    F, _ = attribute_all(model, x, target, batch=bs)
    return F


def pixel_feats(x):
    return np.concatenate([image_features(x[i:i + 512])
                           for i in range(0, len(x), 512)])
