"""ADV^2-style attack (CW loss plus Saliency map distance) and shared attack
utilities. With lam = 0 it is a plain CW attack."""
import torch
import torch.nn.functional as Fn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import GTSRB

from xai_detect import DATA, DEVICE
from attrib_diff import saliency, map_distance


def cw_margin(logits, y, kappa=0.0):
    """Untargeted Carlini-Wagner margin, hinged at zero once misclassified
    by at least kappa."""
    onehot = Fn.one_hot(y, logits.shape[1]).bool()
    true = logits.gather(1, y.view(-1, 1)).squeeze(1)
    other = logits.masked_fill(onehot, -1e4).max(1).values
    return (true - other + kappa).clamp(min=0)


def upsample(z, size):
    """k x k code to full resolution. Identity when already full size."""
    if z.shape[-1] == size:
        return z
    return Fn.interpolate(z, size=(size, size), mode="bilinear",
                          align_corners=False)


def attack(model, x, y, eps, k, lam, steps=150, lr=0.08, kappa=0.0, bs=128,
           warm=0.4):
    """Returns adversarial images on CPU. Needs the ReLU surrogate when lam > 0.
    The map term switches on after the first `warm` fraction of steps."""
    warm_steps = int(warm * steps)
    out = []

    for i in range(0, len(x), bs):
        xb = x[i:i + bs].to(DEVICE)
        yb = y[i:i + bs].to(DEVICE)
        H = xb.shape[-1]
        if lam > 0:
            a_clean = saliency(model, xb.clone(), yb).detach()

        z = torch.zeros(len(xb), 3, k, k, device=DEVICE, requires_grad=True)
        opt = torch.optim.Adam([z], lr=lr)

        for step in range(steps):
            delta = eps * torch.tanh(upsample(z, H))
            x_adv = (xb + delta).clamp(0, 1)
            logits = model(x_adv)
            loss = cw_margin(logits, yb, kappa).sum()

            if lam > 0 and step >= warm_steps:
                # Attribute against the current prediction, as the detector does.
                tgt = logits.argmax(1).detach()
                a_adv = saliency(model, x_adv, tgt, create_graph=True)
                loss = loss + lam * map_distance(a_adv, a_clean).sum()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        with torch.no_grad():
            delta = eps * torch.tanh(upsample(z, H))
            out.append((xb + delta).clamp(0, 1).cpu())

    return torch.cat(out)


def predict(model, x, bs=512):
    with torch.no_grad():
        return torch.cat([model(x[i:i + bs].to(DEVICE)).argmax(1).cpu()
                          for i in range(0, len(x), bs)])


def load_data(model, img_size, n, seed=0):
    """n random GTSRB test images that the model classifies correctly."""
    tf = transforms.Compose([transforms.Resize((img_size, img_size)),
                             transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tf)
    X, Y = [], []
    for x, y in DataLoader(ds, batch_size=512, num_workers=4):
        X.append(x)
        Y.append(y)
    X, Y = torch.cat(X), torch.cat(Y)
    keep = (predict(model, X) == Y).nonzero().squeeze(1)
    g = torch.Generator().manual_seed(seed)
    keep = keep[torch.randperm(len(keep), generator=g)[:n]]
    return X[keep], Y[keep]
