"""E7: do the 500x images predict baseline air resistance on unseen sensors? (development only)

Target: log10 r_air_median (data/derived/r_air.csv, from robustness.py E3).
Same pipeline as the response model (dev_eval 'img': standardise -> PCA -> ridge, inner CV),
20 repeats of sensor-held-out 5-fold CV, 1,000-permutation null (one repeat per permutation).
  A  frozen ViT-B (dinov2_vitb14_n448_pct)       -> log R_air, all development sensors
  B  49 texture statistics (texture_tiles.npz)   -> log R_air, comparison
  C  ViT-B -> log R_air outside the high-deposition group (PC1 >= -30): beyond overgrowth?
  D  Spearman of the out-of-fold image prediction of R_air (from A) with response,
     for reference, not a test
The locked 15 are never touched. Writes data/derived/r_air_predict.json.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from dev_eval import cv_score, locked, read_labels, sensor_features

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features500x"
NPERM = 1000


def test(X, y, label):
    rho, r2, _, pred = cv_score(X, y, "img", 20)
    rng = np.random.default_rng(0)
    null = np.array([cv_score(X, rng.permutation(y), "img", 1, seed=k)[0][0] for k in range(NPERM)])
    p = (1 + (null >= rho.mean()).sum()) / (NPERM + 1)
    res = dict(n=len(y), rho=round(float(rho.mean()), 3), rho_sd=round(float(rho.std()), 3),
               r2=round(float(r2.mean()), 3), null_mean=round(float(null.mean()), 3),
               null_sd=round(float(null.std()), 3), perm_p=round(float(p), 4))
    print(f"{label:34s} {res}", flush=True)
    return res, pred


def main():
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test_set = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in lab if s not in test_set)
    ra = {r["sensor"]: np.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    dev = [s for s in dev if s in ra]
    y = np.array([ra[s] for s in dev])
    resp = np.array([lab[s]["response"] for s in dev])
    X = sensor_features(FEAT / "dinov2_vitb14_n448_pct.npz", FEAT / "images.csv", dev, "mean")
    T = np.load(ROOT / "data/derived/texture_tiles.npz")
    Xt = np.stack([T["feats"][T["sensor"] == s].mean(0) for s in dev])
    out = {}
    out["A_vitb_r_air"], pred = test(X, y, "A ViT-B -> log R_air")
    out["B_texture_r_air"], _ = test(Xt, y, "B texture -> log R_air")
    pc1 = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(X))[:, 0]
    keep = pc1 >= -30
    out["C_vitb_r_air_outside_high_dep"], _ = test(X[keep], y[keep], "C ViT-B -> log R_air (PC1>=-30)")
    r = spearmanr(pred, resp)
    out["D_pred_r_air_vs_response"] = dict(rho=round(float(r.statistic), 3), n=len(y))
    print("D image-predicted R_air vs response", out["D_pred_r_air_vs_response"])
    json.dump(out, open(ROOT / "data/derived/r_air_predict.json", "w"), indent=1)


if __name__ == "__main__":
    main()
