"""Frozen ViT-B on the 4-class task of the two-head CNNs (same split, same classes).

Split: repeat 0 of RepeatedKFold(5, random_state=0) over the 56 development sensors; classes are
quartiles of each fold's training labels. Two readouts: (a) the ridge regression prediction binned
with the training quartile edges; (b) a multinomial logistic classifier (standardise -> PCA 16 ->
LogisticRegression, C by inner 5-fold CV). Balanced accuracy (chance 0.25) with a permutation null
over the held-out predictions (2,000 shuffles). Writes data/derived/vitb_classes.json.
"""
import json
import math
import csv
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dev_eval import locked, make_model, read_labels, sensor_features

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features500x"


def balanced_acc(t, p):
    return float(np.mean([np.mean(p[t == c] == c) for c in np.unique(t)]))


def main():
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in lab if s not in test)
    X = sensor_features(FEAT / "dinov2_vitb14_n448_pct.npz", FEAT / "images.csv", dev, "mean")
    ra = {r["sensor"]: math.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    rng = np.random.default_rng(0)
    out = {}
    for target, y in (("response", np.array([lab[s]["response"] for s in dev])), ("r_air", np.array([ra[s] for s in dev]))):
        tc, pb, pl = np.zeros(len(y), int), np.zeros(len(y), int), np.zeros(len(y), int)
        for tr, te in RepeatedKFold(n_splits=5, n_repeats=1, random_state=0).split(X):
            edges = np.quantile(y[tr], [0.25, 0.5, 0.75])
            tc[te] = np.digitize(y[te], edges)
            pb[te] = np.digitize(make_model("img", len(tr)).fit(X[tr], y[tr]).predict(X[te]), edges)
            clf = make_pipeline(StandardScaler(), PCA(16), LogisticRegressionCV(Cs=10, cv=5, max_iter=5000))
            pl[te] = clf.fit(X[tr], np.digitize(y[tr], edges)).predict(X[te])
        res = {}
        for name, p in (("ridge_binned", pb), ("logistic", pl)):
            ba = balanced_acc(tc, p)
            null = np.array([balanced_acc(tc, rng.permutation(p)) for _ in range(2000)])
            res[name] = dict(balanced_accuracy=round(ba, 3), perm_p=round(float((null >= ba).mean()), 4),
                             confusion=np.histogram2d(tc, p, bins=[np.arange(5) - .5] * 2)[0].astype(int).tolist())
        out[target] = res
        print(target, json.dumps(res), flush=True)
    json.dump(out, open(ROOT / "data/derived/vitb_classes.json", "w"), indent=1)


if __name__ == "__main__":
    main()
