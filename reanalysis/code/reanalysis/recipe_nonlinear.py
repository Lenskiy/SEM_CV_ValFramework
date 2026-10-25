"""E14: nonlinear recipe baselines (review check; development sensors only, locked sensors never read).

The recipe models in the paper are linear (ridge). Here the recipe (log10 ZnCl2 concentration,
log10 current, log10 time; optionally the zinc-thickness estimate h) is given to nonlinear models
with their hyperparameters chosen by inner 5-fold CV inside each outer training fold:
  knn  k-nearest neighbours on standardised inputs, k in 1..10, uniform or distance weights
  rf   random forest, 500 trees, min_samples_leaf in {1, 2, 4, 8}
  gp   Gaussian process, anisotropic RBF + white noise (hyperparameters by marginal likelihood)
Outer CV: the paper's 20 repeats of 5-fold sensor-held-out CV (RepeatedKFold random_state 0).
Targets: log10 response and log10 baseline resistance. Reported: mean +- s.d. of Spearman rho
over repeats, and the "beyond recipe" checks recomputed with the best nonlinear recipe model:
  images vs resistance left by the recipe; resistance-left-by-recipe vs response-left-by-recipe.
Image out-of-fold predictions are the 20-repeat means in data/derived/chain_oof.csv.
Writes data/derived/recipe_nonlinear.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import json
import math
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C
from dev_eval import locked, read_labels

ROOT = Path(__file__).resolve().parents[2]
INNER = KFold(5, shuffle=True, random_state=0)


def model(kind):
    if kind == "knn":
        return GridSearchCV(make_pipeline(StandardScaler(), KNeighborsRegressor()),
                            {"kneighborsregressor__n_neighbors": list(range(1, 11)),
                             "kneighborsregressor__weights": ["uniform", "distance"]},
                            cv=INNER, scoring="neg_mean_squared_error")
    if kind == "rf":
        return GridSearchCV(RandomForestRegressor(500, random_state=0), {"min_samples_leaf": [1, 2, 4, 8]},
                            cv=INNER, scoring="neg_mean_squared_error")
    k = ConstantKernel() * RBF(length_scale=np.ones(NF)) + WhiteKernel()
    return make_pipeline(StandardScaler(), GaussianProcessRegressor(k, normalize_y=True, n_restarts_optimizer=2,
                                                                     random_state=0))


def job(args):
    kind, X, y = args
    global NF
    NF = X.shape[1]
    preds = np.zeros((20, len(y)))
    for i, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=20, random_state=0).split(X)):
        preds[i // 5, te] = model(kind).fit(X[tr], y[tr]).predict(X[te])
    return np.array([spearmanr(p, y).statistic for p in preds]), preds.mean(0)


def resid(y, p):
    b = np.polyfit(p, y, 1)
    return y - np.polyval(b, p)


def perm_p(a, b, n=10000, seed=0):
    r = spearmanr(a, b).statistic
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(a, rng.permutation(b)).statistic for _ in range(n)])
    return float(r), float((1 + np.sum(np.abs(null) >= abs(r))) / (n + 1))


def main():
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in lab if s not in test)
    ra = {r["sensor"]: math.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    hh = {r["sensor"]: float(r["h_0.048"]) for r in csv.DictReader(open(ROOT / "data/derived/regime_sensors.csv"))}
    oof = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/derived/chain_oof.csv"))}
    lr = np.array([ra[s] for s in dev])
    resp = np.array([lab[s]["response"] for s in dev])
    rec = np.array([[math.log10(c), math.log10(i), math.log10(t)] for c, i, t in map(C.parse_sensor_key, dev)])
    rech = np.column_stack([rec, np.log10([hh[s] for s in dev])])
    jobs = [(k, X, y) for k in ("knn", "rf", "gp") for X in (rec, rech) for y in (resp, lr)]
    names = [f"{k}|{'C,I,t' if X.shape[1] == 3 else 'C,I,t,h'}|{'response' if y is resp else 'r_air'}"
             for k, X, y in jobs]
    with get_context("fork").Pool(12) as pool:
        res = pool.map(job, jobs)
    out = {"models": {}}
    for n, (rho, p) in zip(names, res):
        out["models"][n] = dict(rho_mean=round(float(rho.mean()), 3), rho_sd=round(float(rho.std()), 3))
        print(n, out["models"][n], flush=True)
    # beyond-recipe checks with the best nonlinear recipe model for resistance
    best = max((n for n in names if n.endswith("r_air")), key=lambda n: out["models"][n]["rho_mean"])
    pr = dict(zip(names, [p for _, p in res]))
    img_r = np.array([float(oof[s]["img_pred_r_air"]) for s in dev])
    r1 = perm_p(img_r, resid(lr, pr[best]))
    bestr = max((n for n in names if n.endswith("response")), key=lambda n: out["models"][n]["rho_mean"])
    r2 = perm_p(resid(lr, pr[best]), resid(resp, pr[bestr]))
    out["best_recipe_r_air"], out["best_recipe_response"] = best, bestr
    out["images_vs_r_air_beyond_nonlinear_recipe"] = dict(rho=round(r1[0], 3), p_two_sided=round(r1[1], 4))
    out["r_air_vs_response_both_beyond_nonlinear_recipe"] = dict(rho=round(r2[0], 3), p_two_sided=round(r2[1], 4))
    print(json.dumps({k: v for k, v in out.items() if k != "models"}, indent=1))
    json.dump(out, open(ROOT / "data/derived/recipe_nonlinear.json", "w"), indent=1)


if __name__ == "__main__":
    main()
