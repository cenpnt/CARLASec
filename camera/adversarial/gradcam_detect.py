"""
Step: first XAI-based adversarial detector using Grad-CAM.

For each image we compute the Grad-CAM heatmap of the classifier's predicted
class (clean vs FGSM-adversarial), extract simple heatmap statistics (how
focused vs scattered / unstable the explanation is), and train a detector to
tell clean from adversarial. Also saves a visual comparison.

This is the first, single-method XAI detector. Later we make it hybrid
(multiple XAI methods + uncertainty) per Dr. Kim's feedback.

Run:  python gradcam_detect.py   (venv active; gtsrb_cnn.pt must exist)
"""
import os, numpy as np, cv2
import torch, torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from train_gtsrb import TrafficSignNet

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.34,0.31,0.32]).view(1,3,1,1); STD = torch.tensor([0.27,0.26,0.27]).view(1,3,1,1)

class Normalized(nn.Module):
    def __init__(s, m): super().__init__(); s.model=m; s.register_buffer("mean",MEAN); s.register_buffer("std",STD)
    def forward(s, x): return s.model((x - s.mean) / s.std)

def fgsm(model, x, y, eps):
    x = x.clone().detach().requires_grad_(True)
    nn.functional.cross_entropy(model(x), y).backward()
    return (x + eps*x.grad.sign()).clamp(0,1).detach()

class GradCAM:
    """Grad-CAM on the last conv layer (c4) of TrafficSignNet."""
    def __init__(self, model, target_layer):
        self.model = model; self.a = None; self.g = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)
    def _fwd(self, m, i, o): self.a = o.detach()
    def _bwd(self, m, gi, go): self.g = go[0].detach()
    def __call__(self, x, cls=None):
        self.model.zero_grad()
        logits = self.model(x)
        if cls is None: cls = logits.argmax(1)
        score = logits.gather(1, cls.view(-1,1)).sum()
        score.backward()
        alpha = self.g.mean(dim=(2,3), keepdim=True)          # [B,C,1,1]
        cam = torch.relu((alpha * self.a).sum(1))             # [B,h,w]
        cam = cam - cam.amin(dim=(1,2), keepdim=True)
        cam = cam / (cam.amax(dim=(1,2), keepdim=True) + 1e-9)
        return cam.cpu().numpy()                              # [B,h,w] in [0,1]

def cam_features(cam):   # cam: 2D [h,w] in [0,1]
    s = cam.sum() + 1e-9; H, W = cam.shape
    ys, xs = np.mgrid[0:H, 0:W]
    cy = (cam*ys).sum()/s; cx = (cam*xs).sum()/s
    spread = np.sqrt((cam*((ys-cy)**2 + (xs-cx)**2)).sum()/s)   # how spatially spread out
    p = cam.flatten()/s
    entropy = -np.sum(p*np.log(p+1e-12))                        # how diffuse
    return [float(cam.mean()), float(cam.std()), float(cam.max()),
            float(entropy), float(spread), float(np.mean(cam>0.5))]

def overlay(img01, cam):  # img01: [3,32,32] tensor in [0,1]; cam: [h,w]
    bgr = (img01.numpy().transpose(1,2,0)*255).astype(np.uint8)[..., ::-1]
    bgr = cv2.resize(bgr, (96,96), interpolation=cv2.INTER_NEAREST)
    hm = cv2.applyColorMap(cv2.resize((cam*255).astype(np.uint8),(96,96)), cv2.COLORMAP_JET)
    return cv2.addWeighted(bgr, 0.5, hm, 0.5, 0)

def main():
    ckpt = torch.load(os.path.join(HERE,"gtsrb_cnn.pt"), map_location=device)
    base = TrafficSignNet(ckpt.get("n_classes",43)); base.load_state_dict(ckpt["state_dict"])
    model = Normalized(base).to(device).eval()
    cam = GradCAM(model, base.c4)

    tfm = transforms.Compose([transforms.Resize((32,32)), transforms.ToTensor()])
    ds = GTSRB(DATA, split="test", download=False, transform=tfm)
    idx = list(range(0, len(ds), max(1, len(ds)//500)))[:500]
    dl = DataLoader(Subset(ds, idx), batch_size=1, shuffle=False)

    for eps in [0.01, 0.03, 0.05]:
        Xc, Xa = [], []; saved = 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            cc = cam(xb)[0]; Xc.append(cam_features(cc))
            xadv = fgsm(model, xb, yb, eps)
            ca = cam(xadv)[0]; Xa.append(cam_features(ca))
            if eps == 0.03 and saved < 3:
                row = np.hstack([overlay(xb[0].cpu(), cc), overlay(xadv[0].cpu(), ca)])
                cv2.imwrite(os.path.join(HERE, f"gradcam_ex{saved}.png"), row); saved += 1
        X = np.vstack([Xc, Xa]); y = np.r_[np.zeros(len(Xc)), np.ones(len(Xa))]
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
        clf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
        pred = clf.predict(Xte); prob = clf.predict_proba(Xte)[:,1]
        print(f"eps={eps:.2f}  Grad-CAM detector accuracy={accuracy_score(yte,pred)*100:5.1f}%  AUC={roc_auc_score(yte,prob):.3f}", flush=True)
    print("\nSaved gradcam_ex0..2.png  [clean img+CAM | adversarial img+CAM]", flush=True)

if __name__ == "__main__":
    main()
