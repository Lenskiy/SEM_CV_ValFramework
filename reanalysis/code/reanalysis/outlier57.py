"""E19: sensitivity to the excluded sensor (0.1M-250uA-60s), development set of 57 instead of 56.

The sensor was excluded before any image model, as the outlier of the earlier signal-to-noise label;
on the relative-response label it is unremarkable (8.7%). Here it is put back and the main
development results are recomputed with the paper's procedures (20 x 5-fold sensor-held-out CV):
resistance-response correlation; resistance screen (AUC, leave-one-out threshold, 10 kOhm);
frozen ViT-B (pre-registered configuration) -> log response and -> log R_air; linear recipe ->
log R_air; images beyond the linear recipe for R_air. The locked set is not touched.
Writes data/derived/outlier57.json.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import common as C
from dev_eval import TARGETS, cv_score, locked, sensor_features
from qc_metrics import auc
from screen import best_threshold, metrics

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features500x"


def main():
    lab = {r["sensor"]: np.log10(float(r["rel_resp_median"])) for r in csv.DictReader(open(ROOT / "data/derived/labels_v2.csv"))}
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    out = {}
    for name, dev in (("56", sorted(s for s in lab if s not in test and s != "0.1M-250uA-60s")),
                      ("57", sorted(s for s in lab if s not in test))):
        ra = {r["sensor"]: math.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
        lr = np.array([ra[s] for s in dev])
        y = np.array([lab[s] for s in dev])
        dead = 10 ** y < 0.01
        flag = np.zeros(len(dev), bool)
        for k in range(len(dev)):
            m = np.arange(len(dev)) != k
            flag[k] = lr[k] <= best_threshold(lr[m], dead[m])
        X = sensor_features(FEAT / "dinov2_vitb14_n448_pct.npz", FEAT / "images.csv", dev, "mean")
        R = np.stack([C.fab_features(s) for s in dev])
        ri, _, _, pi = cv_score(X, lr, "img", 20)
        rr, _, _, pr = cv_score(R, lr, "params", 20)
        ry, _, _, _ = cv_score(X, y, "img", 20)
        b = np.polyfit(pr, lr, 1)
        out[name] = dict(n=len(dev), rho_r_air_response=round(float(spearmanr(lr, y).statistic), 3),
                         screen_auc=round(float(auc(lr, dead)), 3), screen_loo=metrics(flag, dead),
                         screen_10k=metrics(lr < 4, dead),
                         vitb_r_air=round(float(ri.mean()), 3), vitb_response=round(float(ry.mean()), 3),
                         recipe_r_air=round(float(rr.mean()), 3),
                         images_beyond_recipe_r_air=round(float(spearmanr(pi, lr - np.polyval(b, pr)).statistic), 3))
        print(name, out[name], flush=True)
    json.dump(out, open(ROOT / "data/derived/outlier57.json", "w"), indent=1)


if __name__ == "__main__":
    main()
