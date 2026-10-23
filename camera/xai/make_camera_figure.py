"""Figure 9 (slides): one sign clean, under our ADV2 and under our ADV2 with
EOT, each seen as attacked and through four camera views, with the model's
reading and the XAI detector's verdict on every image. Detector and split as
aim1_camera.py (seed 0); threshold at 5% FPR on clean camera views.

Run:
    C:\\Users\\s4990998\\xai-venv\\Scripts\\python.exe -u make_camera_figure.py
"""
import os

os.environ.setdefault("TQDM_DISABLE", "1")

import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import PyTorchClassifier
from sklearn.ensemble import HistGradientBoostingClassifier

import camera_view
from model import load_trained, IMG_SIZE, NUM_CLASSES
from xai_detect import CKPT, DEVICE, WORK
from relu_surrogate import swap_relu
from aim1_attack import attack as plain_attack, predict, load_data
from aim1_adaptive import thr_at_fpr, xai_feats
from aim1_improve import blind_attack
from make_figures import NAMES

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
CACHE = os.path.join(WORK, "fig9_cache.pt")
EPS, W, STOP, N_ATT, K = 0.03, {"adv2_all": 10.0}, 14, 128, 4
VIEW_SEED = 2000            # same evaluation views as aim1_camera.py, seed 0
CAUGHT_C, MISSED_C = "#c51b7d", "#1b9e77"


def main():
    os.makedirs(OUT, exist_ok=True)
    plain, ckpt = load_trained(CKPT, DEVICE)
    img_size = ckpt.get("img_size", IMG_SIZE)
    surro, _ = load_trained(CKPT, DEVICE)
    swap_relu(surro, kind="adv2")
    torch.manual_seed(0)

    X, Y = load_data(plain, img_size, 1000, seed=0)
    Xtr, Ytr, Xte, Yte = X[:500], Y[:500], X[500:], Y[500:]

    print("fitting the XAI detector ...", flush=True)
    clf = PyTorchClassifier(model=plain, loss=nn.CrossEntropyLoss(),
                            input_shape=(3, img_size, img_size),
                            nb_classes=NUM_CLASSES, clip_values=(0.0, 1.0),
                            device_type="gpu" if DEVICE == "cuda" else "cpu")
    advs = [plain_attack(plain, Xtr, Ytr, eps=EPS, k=k, lam=0.0,
                         steps=100).float() for k in (96, 6, 3)]
    advs.append(torch.from_numpy(FastGradientMethod(
        estimator=clf, eps=EPS, batch_size=256).generate(
            x=Xtr.numpy())).float())
    lab = np.concatenate([np.zeros(len(Xtr))] + [np.ones(len(A)) for A in advs])
    Fx = np.concatenate([xai_feats(plain, Xtr, predict(plain, Xtr))]
                        + [xai_feats(plain, A, predict(plain, A)) for A in advs])
    tree = HistGradientBoostingClassifier(max_iter=400,
                                          random_state=0).fit(Fx, lab)

    def score(x):
        p = predict(plain, x)
        return p, tree.predict_proba(xai_feats(plain, x, p))[:, 1]

    _, s_clean = score(camera_view.views(Xte, 10, seed=VIEW_SEED))
    thr = thr_at_fpr(s_clean, 0.05)

    Xa, Ya = Xte[:N_ATT], Yte[:N_ATT]
    if os.path.exists(CACHE):
        A_ours, A_eot = torch.load(CACHE)
        print("loaded cached attacks", CACHE)
    else:
        print(f"attacking {N_ATT} signs ...", flush=True)
        A_ours = blind_attack(surro, Xa, Ya, EPS, W, None, None).float()
        A_eot = blind_attack(surro, Xa, Ya, EPS, W, None, None,
                             view=True, eot=2).float()
        torch.save((A_ours, A_eot), CACHE)

    def per_view(x):
        """x and its K camera views (N, 1+K, ...), with the reading and the
        detector score of each. Same views for every set of N signs."""
        V = torch.cat([x, camera_view.views(x, K, seed=VIEW_SEED)])
        p, s = score(V)
        V = V.view(1 + K, len(x), *x.shape[1:]).transpose(0, 1)
        return V, p.view(1 + K, -1).T, torch.as_tensor(s).view(1 + K, -1).T

    V_c, p_c, _ = per_view(Xa)
    V_o, p_o, s_o = per_view(A_ours)
    V_e, p_e, s_e = per_view(A_eot)
    ok_o = (p_o != Ya[:, None]) & (s_o < thr)        # misread and missed
    ok_e = (p_e != Ya[:, None]) & (s_e < thr)
    # ours passes on the fixed image but fails on views; EOT passes on views
    gain = ok_e[:, 1:].sum(1) - ok_o[:, 1:].sum(1)
    good = ok_o[:, 0] & ok_e[:, 1:].all(1)
    pool = good.nonzero().squeeze(1).tolist() or list(range(N_ATT))
    stops = [i for i in pool if int(Ya[i]) == STOP]
    i = max(stops or pool, key=lambda j: int(gain[j]))
    print(f"using index {i} ({NAMES[int(Ya[i])]}); "
          f"{int(good.sum())} of {N_ATT} signs fit the pattern")

    rows = [("Clean sign", V_c[i], p_c[i], None),
            ("Our ADV$^2$", V_o[i], p_o[i], s_o[i]),
            ("Our ADV$^2$ + EOT\n(trained over camera views)",
             V_e[i], p_e[i], s_e[i])]
    fig, ax = plt.subplots(3, 1 + K, figsize=(3.0 * (1 + K), 11.6))
    for r, (title, V, p, s) in enumerate(rows):
        for c in range(1 + K):
            a = ax[r, c]
            a.imshow(V[c].permute(1, 2, 0).numpy())
            a.set_xticks([])
            a.set_yticks([])
            head = "as attacked" if c == 0 else f"camera view {c}"
            read = NAMES[int(p[c])]
            if r == 0:
                a.set_title(f"{head}\nreads: {read}", fontsize=10)
                continue
            fooled = int(p[c]) != int(Ya[i])
            caught = s[c] >= thr
            if not fooled:
                verdict, col = "attack failed", "grey"
            elif caught:
                verdict, col = "detector: CAUGHT", CAUGHT_C
            else:
                verdict, col = "detector: missed", MISSED_C
            a.set_title(f"{head}\nreads: {read}\n{verdict}", fontsize=10,
                        color=col)
        ax[r, 0].set_ylabel(title, fontsize=11)
    fig.suptitle("Without EOT the attack only works on the exact image; "
                 "trained over camera views it keeps fooling the model "
                 "and the detector", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96], h_pad=3.0)
    path = os.path.join(OUT, "fig9_camera_views.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print("saved", path)


if __name__ == "__main__":
    main()
