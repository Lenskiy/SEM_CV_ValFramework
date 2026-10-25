"""Small CNNs trained from scratch (train_small.py) against the frozen ViT-B on the same split.

Split: fold k of repeat 0 of RepeatedKFold(5, random_state=0) over the 56 development sensors
(the split used for the fine-tuning comparison). Per model and target: Spearman rho of the pooled
held-out sensor predictions for each seed (mean +/- s.d. over 3 seeds) and of the seed-averaged
predictions; frozen ViT-B (dinov2_vitb14_n448_pct, ridge pipeline) on the same split.
Writes data/derived/compare_small.csv.
"""
import csv
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from dev_eval import cv_score, locked, read_labels, sensor_features

ROOT = Path(__file__).resolve().parents[2]


def main():
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in lab if s not in test)
    ra = {r["sensor"]: math.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    Y = {"response": np.array([lab[s]["response"] for s in dev]), "r_air": np.array([ra[s] for s in dev])}
    X = sensor_features(ROOT / "data/features500x/dinov2_vitb14_n448_pct.npz", ROOT / "data/features500x/images.csv", dev, "mean")
    rows = []
    for target, y in Y.items():
        rows.append(dict(model="frozen ViT-B + ridge", target=target, rho_seeds="", rho_mean=round(float(cv_score(X, y, "img", 1)[0][0]), 3)))
        for model in ("small_resnet", "simple_cnn"):
            P = []
            for seed in range(3):
                p = {}
                for f in range(5):
                    for r in csv.DictReader(open(ROOT / f"data/small_runs/{model}_{target}_s{seed}/fold{f}/preds.csv")):
                        if r["heldout"] == "True":
                            p[r["sensor"]] = float(r["pred"])
                P.append(np.array([p[s] for s in dev]))
            per = [spearmanr(q, y).statistic for q in P]
            rows.append(dict(model=model, target=target, rho_seeds=" ".join(f"{v:+.3f}" for v in per),
                             rho_mean=round(float(np.mean(per)), 3), rho_sd=round(float(np.std(per)), 3),
                             rho_seed_avg=round(float(spearmanr(np.mean(P, 0), y).statistic), 3)))
    with open(ROOT / "data/derived/compare_small.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "target", "rho_mean", "rho_sd", "rho_seed_avg", "rho_seeds"])
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)


if __name__ == "__main__":
    main()
