"""E18: imaging session as a confound for the image -> resistance result (development sensors only).

The 500x images were taken on 13 days (data/features500x/images.csv, 'time'). Each sensor is assigned
its primary session: the day with most of its images, preferring a later day over the August 2024
screening days. Checks for log10 baseline resistance:
  1. Kruskal-Wallis of resistance across primary sessions.
  2. Session-only baseline: one-hot session -> ridge, 20 x 5-fold sensor-held-out CV (as the paper).
  3. Image model with whole sessions held out (leave-one-session-out CV): pooled Spearman rho, and
     the same for the linear recipe model, both against a 1,000-permutation null of the labels.
  4. Images beyond session: rho between the image out-of-fold prediction (chain_oof.csv) and the
     resistance left unexplained by the session-only model.
Writes data/derived/session_check.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import collections
import csv
import json
import math
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import kruskal, spearmanr
from sklearn.model_selection import LeaveOneGroupOut

import common as C
import dev_eval
from dev_eval import cv_score, locked, read_labels, sensor_features

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features500x"
G = {}


def serial(kind, n):
    m = dev_eval._orig(kind, n)
    return m.set_params(n_jobs=1) if hasattr(m, "n_jobs") else m


def loso(X, y, groups, kind):
    p = np.zeros(len(y))
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        p[te] = dev_eval.make_model(kind, len(tr)).fit(X[tr], y[tr]).predict(X[te])
    return spearmanr(p, y).statistic


def null_job(k):
    rng = np.random.default_rng(k)
    perm = rng.permutation(len(G["y"]))
    return [loso(G["Xi"], G["y"][perm], G["g"], "img"), loso(G["Xr"], G["y"][perm], G["g"], "params")]


def main():
    dev_eval._orig = dev_eval.make_model
    dev_eval.make_model = serial
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in lab if s not in test)
    days = collections.defaultdict(collections.Counter)
    for r in csv.DictReader(open(FEAT / "images.csv")):
        days[r["sensor"]][r["time"][:10]] += 1
    prim = {s: max(c, key=lambda d: (not d.startswith("2024-08"), c[d], d)) for s, c in days.items()}
    g = np.array([prim[s] for s in dev])
    ra = {r["sensor"]: math.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    y = np.array([ra[s] for s in dev])
    out = dict(n_sessions=int(len(set(g))), sensors_per_session=dict(collections.Counter(g)))
    out["kruskal_r_air_by_session"] = dict(H=round(float(kruskal(*[y[g == d] for d in set(g)]).statistic), 2),
                                          p=round(float(kruskal(*[y[g == d] for d in set(g)]).pvalue), 4))
    S = np.array([[float(gi == d) for d in sorted(set(g))] for gi in g])
    rho_s, _, _, pred_s = cv_score(S, y, "params", 20)
    out["session_only_cv_rho"] = dict(mean=round(float(rho_s.mean()), 3), sd=round(float(rho_s.std()), 3))
    Xi = sensor_features(FEAT / "dinov2_vitb14_n448_pct.npz", FEAT / "images.csv", dev, "mean")
    Xr = np.stack([C.fab_features(s) for s in dev])
    G.update(Xi=Xi, Xr=Xr, y=y, g=g)
    obs = [loso(Xi, y, g, "img"), loso(Xr, y, g, "params")]
    with get_context("fork").Pool(8) as pool:
        null = np.array(pool.map(null_job, range(1000)))
    for j, name in enumerate(("images", "recipe_linear")):
        out[f"leave_one_session_out_{name}"] = dict(rho=round(float(obs[j]), 3),
                                                    p=round(float((1 + (null[:, j] >= obs[j]).sum()) / 1001), 4))
    oof = {r["sensor"]: float(r["img_pred_r_air"]) for r in csv.DictReader(open(ROOT / "data/derived/chain_oof.csv"))}
    b = np.polyfit(pred_s, y, 1)
    res = y - np.polyval(b, pred_s)
    ip = np.array([oof[s] for s in dev])
    r = spearmanr(ip, res).statistic
    rng = np.random.default_rng(0)
    nl = np.array([spearmanr(ip, rng.permutation(res)).statistic for _ in range(10000)])
    out["images_vs_r_air_beyond_session"] = dict(rho=round(float(r), 3), p=round(float((1 + (np.abs(nl) >= abs(r)).sum()) / 10001), 4))
    json.dump(out, open(ROOT / "data/derived/session_check.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
