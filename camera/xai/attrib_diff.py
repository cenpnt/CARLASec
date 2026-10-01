"""Differentiable attribution maps for use inside an attack loop.

Captum (in `xai_detect`) stays the reference for evaluation. These versions
exist because an attack must back-propagate through the map, which needs every
gradient call made with create_graph=True. `test_surrogate.py` checks them
against Captum.

`target` is always explicit: at runtime the detector attributes against the
model's own prediction, so for an adversarial input that is the wrong class.
"""
import torch

__all__ = ["saliency", "input_x_grad", "integrated_grad", "ATTRIB",
           "magnitude", "map_distance"]


def _selected_logit(model, x, target):
    """Sum over the batch of the logit for each sample's target class."""
    logits = model(x)
    return logits.gather(1, target.view(-1, 1)).sum()


def saliency(model, x, target, create_graph=False):
    """dF_target/dx, signed. Captum's Saliency(abs=False)."""
    if not x.requires_grad:
        x = x.requires_grad_(True)
    (g,) = torch.autograd.grad(_selected_logit(model, x, target), x,
                               create_graph=create_graph)
    return g


def input_x_grad(model, x, target, create_graph=False):
    """x * dF_target/dx. Captum's InputXGradient."""
    return x * saliency(model, x, target, create_graph=create_graph)


def integrated_grad(model, x, target, steps=16, baseline=None,
                    create_graph=False):
    """Integrated Gradients by midpoint Riemann sum from a black baseline."""
    if baseline is None:
        baseline = torch.zeros_like(x)
    diff = x - baseline
    total = None
    for i in range(steps):
        a = (i + 0.5) / steps
        xi = baseline + a * diff
        if not xi.requires_grad:
            xi = xi.requires_grad_(True)
        (g,) = torch.autograd.grad(_selected_logit(model, xi, target), xi,
                                   create_graph=create_graph)
        total = g if total is None else total + g
    return diff * total / steps


ATTRIB = {
    "sal": saliency,
    "ixg": input_x_grad,
    "ig": integrated_grad,
}


def magnitude(attr, eps=1e-12):
    """Per-pixel attribution magnitude, summed over channels, summing to one."""
    flat = attr.abs().sum(1).flatten(1)
    return flat / (flat.sum(1, keepdim=True) + eps)


def map_distance(attr_adv, attr_clean):
    """L1 distance between normalised magnitude maps, per sample."""
    return (magnitude(attr_adv) - magnitude(attr_clean)).abs().sum(1)
