"""Two-head (CE + lambda x MSE) small CNNs against their regression-only versions and frozen ViT-B.

Same split as compare_small.py (repeat 0 of RepeatedKFold(5, random_state=0), 56 development
sensors), 3 seeds. Per model, target and lambda: pooled held-out Spearman rho of the regression
output and of the expected class (mean over seeds, s.d.), and balanced accuracy of the predicted
class against the sensor's class (quartiles of each fold's training labels; chance 0.25).
Writes data/derived/compare_dual.csv.
"""
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from dev_eval import locked, read_labels

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "data/small_runs"


def pooled(run):
    rows = {}
    for f in range(5):
        for r in csv.DictReader(open(RUNS / run / f"fold{f}/preds.csv")):
            if r["heldout"] == "True":
                rows[r["sensor"]] = r
    return rows


def balanced_acc(true, pred):
    return np.mean([np.mean(pred[true == c] == c) for c in np.unique(true)])


def main():
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in lab if s not in test)
    out = [dict(r) for r in csv.DictReader(open(ROOT / "data/derived/compare_small.csv"))]
    for r in out:
        r["loss"] = "regression only" if r["model"] != "frozen ViT-B + ridge" else "ridge"
    for model in ("small_resnet", "simple_cnn"):
        for target in ("response", "r_air"):
            for lam in ("0.001", "0.01", "0.1", "1"):
                reg, cls, acc = [], [], []
                for seed in range(3):
                    p = pooled(f"{model}_{target}_dual{lam}_s{seed}")
                    y = np.array([float(p[s]["y"]) for s in dev])
                    reg.append(spearmanr([float(p[s]["pred"]) for s in dev], y).statistic)
                    cls.append(spearmanr([float(p[s]["exp_class"]) for s in dev], y).statistic)
                    acc.append(balanced_acc(np.array([int(p[s]["true_class"]) for s in dev]),
                                            np.array([int(p[s]["pred_class"]) for s in dev])))
                out.append(dict(model=model, target=target, loss=f"CE + {lam} MSE",
                                rho_mean=round(float(np.mean(reg)), 3), rho_sd=round(float(np.std(reg)), 3),
                                rho_expected_class=round(float(np.mean(cls)), 3),
                                rho_expected_class_sd=round(float(np.std(cls)), 3),
                                balanced_accuracy=round(float(np.mean(acc)), 3)))
    fields = ["model", "target", "loss", "rho_mean", "rho_sd", "rho_expected_class", "rho_expected_class_sd",
              "balanced_accuracy"]
    with open(ROOT / "data/derived/compare_dual.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(out)
    for r in sorted(out, key=lambda r: (r["target"], r["model"], r["loss"])):
        print(f"{r['target']:8s} {r['model']:22s} {r['loss']:16s} reg {float(r['rho_mean']):+.3f} "
              f"exp-class {r.get('rho_expected_class', '')!s:>7} bal-acc {r.get('balanced_accuracy', '')!s}")


if __name__ == "__main__":
    main()
