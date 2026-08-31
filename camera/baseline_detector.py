"""
Baseline camera-attack detector: simple image-statistics features + random forest,
clean vs attack. Establishes a first accuracy number to compare the later
XAI-based detector against.

- Features (cheap, interpretable): edge density, mean brightness, contrast,
  Laplacian variance (sharpness/noise), saturated-pixel fraction.
- Split: grouped by SCENE (all 6 versions of a frame stay on one side) -> no leakage.
- Reports overall accuracy, confusion matrix, feature importances, and per-attack
  detection rate.

Run:  python baseline_detector.py     (with the venv active; dataset/ must exist)
"""
import os, csv, numpy as np, cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = os.path.join(HERE, "dataset")
MANIFEST = os.path.join(DATASET, "manifest.csv")

FEATURES = ["edge_density", "mean_brightness", "contrast", "laplacian_var", "saturated_frac"]

def extract_features(path):
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    edge_density = float(np.mean(edges > 0))
    mean_brightness = float(gray.mean())
    contrast = float(gray.std())
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    saturated_frac = float(np.mean((gray < 8) | (gray > 247)))
    return [edge_density, mean_brightness, contrast, laplacian_var, saturated_frac]

def scene_id(path):
    # .../blur/blur_0007.jpg -> "0007"
    return os.path.splitext(os.path.basename(path))[0].split("_")[-1]

def main():
    rows = []
    with open(MANIFEST) as f:
        for r in csv.DictReader(f):
            rows.append(r)
    print(f"Extracting features from {len(rows)} images...", flush=True)

    X, y, groups, atk = [], [], [], []
    for i, r in enumerate(rows):
        p = r["path"]
        if not os.path.isabs(p):
            p = os.path.join(HERE, p) if not p.startswith("dataset") else os.path.join(HERE, p)
        p = p.replace("/", os.sep)
        X.append(extract_features(p))
        y.append(0 if r["label"] == "clean" else 1)
        groups.append(scene_id(r["path"]))
        atk.append(r["attack_type"])
        if (i + 1) % 300 == 0:
            print(f"  {i+1}/{len(rows)}", flush=True)
    X = np.array(X); y = np.array(y); groups = np.array(groups); atk = np.array(atk)

    # scene-grouped 70/30 split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    tr, te = next(gss.split(X, y, groups))
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])

    print("\n==================== BASELINE RESULTS ====================")
    print(f"Train: {len(tr)} imgs ({len(set(groups[tr]))} scenes) | Test: {len(te)} imgs ({len(set(groups[te]))} scenes)")
    print(f"Overall accuracy (clean vs attack): {accuracy_score(y[te], pred):.4f}")
    print("\nConfusion matrix [rows=true, cols=pred] (0=clean,1=attack):")
    print(confusion_matrix(y[te], pred))
    print("\n" + classification_report(y[te], pred, target_names=["clean", "attack"]))

    print("Feature importances:")
    for name, imp in sorted(zip(FEATURES, clf.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:16s} {imp:.3f}")

    print("\nPer-attack detection rate (recall on test attacked images):")
    for t in ["blur", "darken", "gaussian_noise", "salt_pepper", "brightness_flicker"]:
        mask = (atk[te] == t)
        if mask.sum():
            rate = np.mean(pred[mask] == 1)
            print(f"  {t:20s} {rate*100:5.1f}%  ({mask.sum()} imgs)")
    cm = confusion_matrix(y[te], pred)
    clean_mask = (atk[te] == "none")
    if clean_mask.sum():
        fpr = np.mean(pred[clean_mask] == 1)
        print(f"\nClean false-alarm rate: {fpr*100:.1f}%  ({clean_mask.sum()} clean imgs)")

if __name__ == "__main__":
    main()
