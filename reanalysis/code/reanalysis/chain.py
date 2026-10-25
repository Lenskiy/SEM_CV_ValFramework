"""E9: recipe -> microstructure (500x images) -> baseline resistance -> response (development only).

Recipe block: log C, log I, log t and log h (regime.py metallic-Zn equivalent thickness, A 0.048 cm2).
Each model: 20 repeats of sensor-held-out 5-fold CV (dev_eval), out-of-fold predictions averaged
over repeats.
  recipe -> log R_air;  images -> log R_air;  images + recipe -> log R_air
  recipe -> log response;  images -> log response (frozen model, as in the paper)
Increments, with predictions held fixed (10,000 permutations of the residual):
  a  image prediction of R_air vs the residual of R_air after the recipe prediction
     (do the images see device-level variation the recipe does not set?)
  b  log R_air vs the residual of response after the recipe prediction
  c  image prediction of response vs the residual of response after log R_air (linear fit)
     (does the image signal reach response through resistance?)
Writes data/derived/chain.json and data/derived/chain_oof.csv (out-of-fold predictions).
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from dev_eval import cv_score, locked, read_labels, sensor_features

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features500x"
rng = np.random.default_rng(0)


def resid(y, x):
    X = np.column_stack([np.ones(len(x)), x])
    return y - X @ np.linalg.lstsq(X, y, rcond=None)[0]


def inc(x, r, n=10000):
    rho = spearmanr(x, r).statistic
    null = np.array([spearmanr(x, rng.permutation(r)).statistic for _ in range(n)])
    return dict(rho=round(float(rho), 3), perm_p=round(float((1 + (null >= rho).sum()) / (n + 1)), 4))


def fit(X, y, kind, label, out):
    rho, r2, _, pred = cv_score(X, y, kind, 20)
    out[label] = dict(rho=round(float(rho.mean()), 3), rho_sd=round(float(rho.std()), 3), r2=round(float(r2.mean()), 3))
    print(f"{label:24s} {out[label]}", flush=True)
    return pred


def main():
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    reg = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/derived/regime_sensors.csv"))}
    dev = [s for s in sorted(lab) if s not in test and s in reg]
    R = np.log10([[float(reg[s][k]) for k in ("C", "I", "t", "h_0.048")] for s in dev])
    lr = np.log10([float(reg[s]["r_air"]) for s in dev])
    y = np.array([lab[s]["response"] for s in dev])
    X = sensor_features(FEAT / "dinov2_vitb14_n448_pct.npz", FEAT / "images.csv", dev, "mean")
    out = dict(n=len(dev))
    p_rec_r = fit(R, lr, "params", "recipe->R_air", out)
    p_img_r = fit(X, lr, "img", "images->R_air", out)
    fit(np.hstack([X, R]), lr, "img|params", "images+recipe->R_air", out)
    p_rec_y = fit(R, y, "params", "recipe->response", out)
    p_img_y = fit(X, y, "img", "images->response", out)
    out["a_images_vs_R_air_beyond_recipe"] = inc(p_img_r, lr - p_rec_r)
    out["b_R_air_vs_response_beyond_recipe"] = inc(lr, y - p_rec_y)
    out["c_images_vs_response_beyond_R_air"] = inc(p_img_y, resid(y, lr))
    out["R_air_vs_response_rho"] = round(float(spearmanr(lr, y).statistic), 3)
    for k in ("a_images_vs_R_air_beyond_recipe", "b_R_air_vs_response_beyond_recipe", "c_images_vs_response_beyond_R_air"):
        print(k, out[k])
    json.dump(out, open(ROOT / "data/derived/chain.json", "w"), indent=1)
    with open(ROOT / "data/derived/chain_oof.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "log_r_air", "log_response", "img_pred_r_air", "recipe_pred_r_air", "img_pred_response"])
        for k, s in enumerate(dev):
            w.writerow([s] + [round(float(v[k]), 5) for v in (lr, y, p_img_r, p_rec_r, p_img_y)])


if __name__ == "__main__":
    main()
