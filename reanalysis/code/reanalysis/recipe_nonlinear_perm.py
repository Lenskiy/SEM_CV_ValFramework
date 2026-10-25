"""E14b: permutation null for the nonlinear recipe baselines (development sensors only).

For each target (log10 response, log10 baseline resistance), the six recipe models of
recipe_nonlinear.py (knn / rf / gp, inputs C,I,t or C,I,t,h) are scored on 200 label permutations,
one repeat of 5-fold CV per permutation (as perm_full.py: permutation k uses seed k for the split;
permutations from default_rng(0) over the sorted development sensors, the same ones perm_full.py
used). p: per model, and corrected by the max over the six models (the best model was chosen
among them). Writes data/derived/recipe_nonlinear_perm.json.
Chunked for a cluster: --start/--stop writes <root>/data/derived/recipe_perm_parts/part_<start>.npy; --merge combines.
Paths: --root (holds data/derived) and --images (images.csv listing the imaged sensors).
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
import csv
import json
import math
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold

import common as C
import recipe_nonlinear as RN
from dev_eval import locked, read_labels

ROOT = Path(__file__).resolve().parents[2]
JOBS = []
NAMES = []


def score(kind, X, y, seed):
    RN.NF = X.shape[1]
    p = np.zeros(len(y))
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=1, random_state=seed).split(X):
        p[te] = RN.model(kind).fit(X[tr], y[tr]).predict(X[te])
    return spearmanr(p, y).statistic


def one(args):
    k, perm = args
    return [score(kind, X, y[perm], k) for kind, X, y in JOBS]


def setup(images):
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    have = {r["sensor"] for r in csv.DictReader(open(images))}
    dev = sorted(s for s in lab if s not in test and s in have)
    ra = {r["sensor"]: math.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    hh = {r["sensor"]: float(r["h_0.048"]) for r in csv.DictReader(open(ROOT / "data/derived/regime_sensors.csv"))}
    rec = np.array([[math.log10(c), math.log10(i), math.log10(t)] for c, i, t in map(C.parse_sensor_key, dev)])
    rech = np.column_stack([rec, np.log10([hh[s] for s in dev])])
    names = []
    for tname, y in (("response", np.array([lab[s]["response"] for s in dev])), ("r_air", np.array([ra[s] for s in dev]))):
        for kind in ("knn", "rf", "gp"):
            for xn, X in (("C,I,t", rec), ("C,I,t,h", rech)):
                JOBS.append((kind, X, y))
                names.append(f"{kind}|{xn}|{tname}")
    NAMES.extend(names)
    return dev


def main():
    global ROOT
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--images", default=None)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    ROOT = Path(a.root)
    part_dir = ROOT / "data/derived/recipe_perm_parts"
    part_dir.mkdir(exist_ok=True)
    dev = setup(a.images or ROOT / "data/features500x/images.csv")
    if not a.merge:
        rng = np.random.default_rng(0)
        perms = [(k, rng.permutation(len(dev))) for k in range(200)][a.start:a.stop]
        with get_context("fork").Pool(a.workers) as pool:
            null = np.array(pool.map(one, perms))
        np.save(part_dir / f"part_{a.start:04d}.npy", null)
        return
    null = np.concatenate([np.load(f) for f in sorted(part_dir.glob("part_*.npy"))])
    assert len(null) == 200, len(null)
    names = NAMES
    obs = json.load(open(ROOT / "data/derived/recipe_nonlinear.json"))["models"]
    out = {}
    for t in ("response", "r_air"):
        idx = [j for j, n in enumerate(names) if n.endswith(t)]
        mx = null[:, idx].max(1)
        for j in idx:
            o = obs[names[j]]["rho_mean"]
            out[names[j]] = dict(rho=o, null_mean=round(float(null[:, j].mean()), 3), null_sd=round(float(null[:, j].std()), 3),
                                 p=round(float((1 + (null[:, j] >= o).sum()) / 201), 4),
                                 p_max_over_6=round(float((1 + (mx >= o).sum()) / 201), 4))
            print(names[j], out[names[j]], flush=True)
    json.dump(out, open(ROOT / "data/derived/recipe_nonlinear_perm.json", "w"), indent=1)
    np.save(ROOT / "data/derived/recipe_nonlinear_perm.npy", null)


if __name__ == "__main__":
    main()
