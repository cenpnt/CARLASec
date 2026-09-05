"""
Hybrid XAI adversarial detector (the contribution taking shape).

Single-method Grad-CAM was ~chance. Here we combine three complementary signal
groups and run an ablation (each alone vs all combined), at the stealthy low-eps
regime where simple detection fails:

  1. XAI (Grad-CAM heatmap statistics)   - how focused / scattered the explanation is
  2. Uncertainty (softmax)               - confidence, entropy, top-2 margin
  3. Stability (feature squeezing)       - how much the prediction shifts under
                                           bit-depth reduction / blur / small noise
                                           (adversarial inputs are fragile, clean stable)

Run:  python hybrid_detect.py   (venv active; gtsrb_cnn.pt must exist)
"""
import os, numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import GTSRB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
from train_gtsrb import TrafficSignNet

HERE = os.path.dirname(os.path.abspath(__file__)); DATA = os.path.join(HERE, "data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = torch.tensor([0.34,0.31,0.32]).view(1,3,1,1); STD = torch.tensor([0.27,0.26,0.27]).view(1,3,1,1)

class Normalized(nn.Module):
    def __init__(s,m): super().__init__(); s.model=m; s.register_buffer("mean",MEAN); s.register_buffer("std",STD)
    def forward(s,x): return s.model((x-s.mean)/s.std)

def fgsm(model,x,y,eps):
    x=x.clone().detach().requires_grad_(True)
    F.cross_entropy(model(x),y).backward()
    return (x+eps*x.grad.sign()).clamp(0,1).detach()

class GradCAM:
    def __init__(s,model,layer):
        s.model=model; s.a=None; s.g=None
        layer.register_forward_hook(lambda m,i,o: setattr(s,'a',o.detach()))
        layer.register_full_backward_hook(lambda m,gi,go: setattr(s,'g',go[0].detach()))
    def __call__(s,x):
        s.model.zero_grad(); logits=s.model(x); cls=logits.argmax(1)
        logits.gather(1,cls.view(-1,1)).sum().backward()
        cam=F.relu((s.g.mean((2,3),keepdim=True)*s.a).sum(1))[0]
        cam=cam-cam.min(); cam=cam/(cam.max()+1e-9)
        return cam.cpu().numpy()

def cam_feats(cam):
    s=cam.sum()+1e-9; H,W=cam.shape; ys,xs=np.mgrid[0:H,0:W]
    cy=(cam*ys).sum()/s; cx=(cam*xs).sum()/s
    spread=np.sqrt((cam*((ys-cy)**2+(xs-cx)**2)).sum()/s)
    p=cam.flatten()/s; ent=-np.sum(p*np.log(p+1e-12))
    return [cam.mean(),cam.std(),cam.max(),ent,spread,np.mean(cam>0.5)]

def unc_feats(p):
    ps=np.sort(p)[::-1]
    return [float(ps[0]), float(-np.sum(p*np.log(p+1e-12))), float(ps[0]-ps[1])]

def squeeze(x, kind):
    if kind=="bit":  return (x*15).round()/15.0
    if kind=="blur": return gaussian_blur(x,kernel_size=3,sigma=1.0)
    if kind=="noise":return (x+0.04*torch.randn_like(x)).clamp(0,1)

def stab_feats(model,x,p0):
    ds=[]
    for k in ("bit","blur","noise"):
        with torch.no_grad(): pk=F.softmax(model(squeeze(x,k)),1)[0].cpu().numpy()
        ds.append(float(np.abs(p0-pk).sum()))
    return ds+[max(ds)]

def collect(model,cam,dl,eps):
    G,U,S,lab=[],[],[],[]
    for xb,yb in dl:
        xb,yb=xb.to(device),yb.to(device)
        for x in (xb, fgsm(model,xb,yb,eps)):
            with torch.no_grad(): p=F.softmax(model(x),1)[0].cpu().numpy()
            G.append(cam_feats(cam(x))); U.append(unc_feats(p)); S.append(stab_feats(model,x,p))
        lab += [0,1]   # clean, adversarial
    return np.array(G),np.array(U),np.array(S),np.array(lab)

def auc_of(X,y):
    Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    clf=RandomForestClassifier(n_estimators=300,random_state=0).fit(Xtr,ytr)
    prob=clf.predict_proba(Xte)[:,1]
    return roc_auc_score(yte,prob), accuracy_score(yte,clf.predict(Xte))

def main():
    ckpt=torch.load(os.path.join(HERE,"gtsrb_cnn.pt"),map_location=device)
    base=TrafficSignNet(ckpt.get("n_classes",43)); base.load_state_dict(ckpt["state_dict"])
    model=Normalized(base).to(device).eval(); cam=GradCAM(model,base.c4)
    tfm=transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor()])
    ds=GTSRB(DATA,split="test",download=False,transform=tfm)
    idx=list(range(0,len(ds),max(1,len(ds)//400)))[:400]
    dl=DataLoader(Subset(ds,idx),batch_size=1,shuffle=False)

    for eps in (0.01,0.03):
        G,U,S,y=collect(model,cam,dl,eps)
        groups={"Grad-CAM (XAI)":G,"Uncertainty":U,"Stability":S,
                "HYBRID (all)":np.hstack([G,U,S])}
        print(f"\n===== eps={eps:.2f}  (detection: clean vs adversarial) =====")
        print(f"  {'signal':22s} {'AUC':>7} {'acc':>7}")
        for name,X in groups.items():
            a,ac=auc_of(X,y)
            print(f"  {name:22s} {a:>7.3f} {ac*100:>6.1f}%", flush=True)

if __name__=="__main__":
    main()
