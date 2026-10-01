"""ReLU surrogates from ADV^2 (Zhang et al. 2020): exact ReLU forward, smoothed
derivative backward, so gradients can flow through attribution maps."""
import torch
import torch.nn as nn

TAU = 1e-4


def h_adv2(z, tau=TAU):
    """Smoothed step: the ADV^2 surrogate for relu'(z)."""
    return 0.5 * (1.0 + z / torch.sqrt(z * z + tau))


def h_adv2_literal(z, tau=TAU):
    """The paper's two-branch expression read literally. Discontinuous at 0."""
    r = torch.sqrt(z * z + tau)
    return torch.where(z < 0, 1.0 + z / r, z / r)


class _SurrogateReLUFn(torch.autograd.Function):
    """relu forward, smoothed-step backward. The backward uses differentiable
    ops so double-backward works."""

    @staticmethod
    def forward(ctx, z, tau, literal):
        ctx.save_for_backward(z)
        ctx.tau = tau
        ctx.literal = literal
        return z.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_out):
        (z,) = ctx.saved_tensors
        h = h_adv2_literal if ctx.literal else h_adv2
        return grad_out * h(z, ctx.tau), None, None


class SurrogateReLU(nn.Module):
    """Drop-in nn.ReLU replacement with a twice-differentiable backward."""

    def __init__(self, tau=TAU, literal=False):
        super().__init__()
        self.tau = tau
        self.literal = literal

    def forward(self, z):
        return _SurrogateReLUFn.apply(z, self.tau, self.literal)

    def extra_repr(self):
        return f"tau={self.tau}, literal={self.literal}"


class SoftplusReLU(nn.Module):
    """softplus_beta in both directions. Changes the forward pass."""

    def __init__(self, beta=50.0):
        super().__init__()
        self.beta = beta

    def forward(self, z):
        return nn.functional.softplus(z, beta=self.beta)

    def extra_repr(self):
        return f"beta={self.beta}"


def _make(kind, tau=TAU, beta=50.0):
    if kind == "adv2":
        return SurrogateReLU(tau=tau, literal=False)
    if kind == "adv2-literal":
        return SurrogateReLU(tau=tau, literal=True)
    if kind == "softplus":
        return SoftplusReLU(beta=beta)
    raise ValueError(f"unknown surrogate kind: {kind!r}")


def swap_relu(model, kind="adv2", tau=TAU, beta=50.0):
    """Recursively replace every nn.ReLU in `model`, in place. Returns the
    number replaced. kind="relu" is a no-op."""
    if kind == "relu":
        return 0
    n = 0
    for name, child in model.named_children():
        if isinstance(child, nn.ReLU):
            setattr(model, name, _make(kind, tau=tau, beta=beta))
            n += 1
        else:
            n += swap_relu(child, kind=kind, tau=tau, beta=beta)
    return n


def count_relu(model):
    """(plain ReLU count, surrogate count), to verify a swap."""
    relu = surrogate = 0
    for m in model.modules():
        if isinstance(m, nn.ReLU):
            relu += 1
        elif isinstance(m, (SurrogateReLU, SoftplusReLU)):
            surrogate += 1
    return relu, surrogate
