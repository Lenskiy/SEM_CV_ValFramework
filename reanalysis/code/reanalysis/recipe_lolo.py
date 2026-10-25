"""E14c: nearest-neighbour recipe model under leave-one-level-out CV (development sensors only).

Complements robustness_chain.py R3 (ViT-B and the linear recipe): the nearest-neighbour recipe
model of recipe_nonlinear.py (log C, log I, log t, log h; k and weights by inner CV) with each level
of ZnCl2 concentration (4), current (6) or time (3) held out in turn; pooled held-out Spearman rho
for log10 baseline resistance and log10 response. Writes data/derived/recipe_lolo.json.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import common as C
import recipe_nonlinear as RN
from dev_eval import locked, read_labels

ROOT = Path(__file__).resolve().parents[2]


def main():
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in lab if s not in test)
    ra = {r["sensor"]: math.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    hh = {r["sensor"]: float(r["h_0.048"]) for r in csv.DictReader(open(ROOT / "data/derived/regime_sensors.csv"))}
    keys = np.array([C.parse_sensor_key(s) for s in dev])
    X = np.column_stack([np.log10(keys), np.log10([hh[s] for s in dev])])
    out = {}
    for tname, y in (("r_air", np.array([ra[s] for s in dev])), ("response", np.array([lab[s]["response"] for s in dev]))):
        for j, lv in enumerate(("concentration", "current", "time")):
            p = np.zeros(len(y))
            for v in np.unique(keys[:, j]):
                te = keys[:, j] == v
                p[te] = RN.model("knn").fit(X[~te], y[~te]).predict(X[te])
            out[f"{tname}|knn_recipe|{lv}"] = round(float(spearmanr(p, y).statistic), 3)
    json.dump(out, open(ROOT / "data/derived/recipe_lolo.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
