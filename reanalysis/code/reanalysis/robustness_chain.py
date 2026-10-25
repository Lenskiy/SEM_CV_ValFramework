"""E13: robustness of the chain results (development sensors only; locked sensors never read).

  R1  bootstrap 95% interval (2,000 resamples of sensors, out-of-fold predictions held fixed) for
      the frozen ViT-B prediction of log baseline resistance (20-repeat mean predictions)
  R2  ViT-B -> log R_air without the August-2024 screening-session images (and without those sensors)
  R3  generalisation to unseen deposition conditions: leave-one-level-out CV over ZnCl2 concentration
      (4 levels), current (6) and time (3), for log R_air and log response; ViT-B and recipe
      (the recipe model cannot extrapolate a held-out level of its own variable meaningfully, so it
      is reported for comparison only); pooled held-out Spearman rho
  R4  gap obscuration sensitivity: loss fraction 0.4 / 0.5 / 0.6 x the image's 90th-percentile
      amplitude and window width 8 / 16 / 32 px, Spearman with log R_air (all 72 sensors; no learning)
  R5  resistance screen at other non-responder definitions (response < 0.5%, 1%, 2%): AUC and
      leave-one-out threshold metrics, development sensors
Writes data/derived/robustness_chain.json.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import csv
import json
import math
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import common as C
import dev_eval
from dev_eval import cv_score, locked, make_model, read_labels, sensor_features
from qc_metrics import auc
from screen import best_threshold, metrics

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features500x"
NPZ = FEAT / "dinov2_vitb14_n448_pct.npz"
_make = dev_eval.make_model


def serial(kind, n):
    return _make(kind, n).set_params(n_jobs=1)


def loo_level(X, y, levels):
    pred = np.zeros(len(y))
    for lv in np.unique(levels):
        te = levels == lv
        pred[te] = serial("img" if X.shape[1] > 8 else "params", int((~te).sum())).fit(X[~te], y[~te]).predict(X[te])
    return pred


def tile_means(sensors, drop_month=None):
    d = np.load(NPZ)
    rows = {int(r["img"]): r for r in csv.DictReader(open(FEAT / "images.csv"))}
    f = d["feats"].astype(np.float32)
    s = np.array([rows[int(i)]["sensor"] for i in d["img"]])
    keep = np.ones(len(s), bool) if drop_month is None else np.array(
        [not rows[int(i)]["time"].startswith(drop_month) for i in d["img"]])
    return np.stack([f[(s == x) & keep].mean(0) for x in sensors])


def r3_job(args):
    name, X, y, lv = args
    p = loo_level(X, y, lv)
    return name, round(float(spearmanr(p, y).statistic), 3)


def main():
    dev_eval.make_model = serial
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in lab if s not in test)
    ra = {r["sensor"]: math.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    lr = np.array([ra[s] for s in dev])
    resp = np.array([lab[s]["response"] for s in dev])
    X = sensor_features(NPZ, FEAT / "images.csv", dev, "mean")
    R = np.stack([C.fab_features(s) for s in dev])
    out = {}

    # R1
    _, _, _, pred = cv_score(X, lr, "img", 20)
    rng = np.random.default_rng(0)
    b = [spearmanr(pred[i], lr[i]).statistic for i in (rng.integers(0, len(lr), len(lr)) for _ in range(2000))]
    out["R1_r_air_rho_of_mean_prediction"] = dict(value=round(float(spearmanr(pred, lr).statistic), 3),
                                                  ci95=[round(float(np.percentile(b, q)), 3) for q in (2.5, 97.5)])
    print("R1", out["R1_r_air_rho_of_mean_prediction"], flush=True)

    # R2
    rows = list(csv.DictReader(open(FEAT / "images.csv")))
    aug = sorted({r["sensor"] for r in rows if r["time"].startswith("2024-08")} & set(dev))
    Xa = tile_means(dev, drop_month="2024-08")
    out["R2_without_august_images"] = round(float(cv_score(Xa, lr, "img", 20)[0].mean()), 3)
    keep = np.array([s not in aug for s in dev])
    out["R2_without_august_sensors"] = dict(n=int(keep.sum()),
                                            rho=round(float(cv_score(X[keep], lr[keep], "img", 20)[0].mean()), 3))
    print("R2", out["R2_without_august_images"], out["R2_without_august_sensors"], flush=True)

    # R3
    lvls = {"concentration": np.array([C.parse_sensor_key(s)[0] for s in dev]),
            "current": np.array([C.parse_sensor_key(s)[1] for s in dev]),
            "time": np.array([C.parse_sensor_key(s)[2] for s in dev])}
    jobs = [(f"{t}|{m}|{k}", Z, y, lv) for t, y in (("r_air", lr), ("response", resp))
            for m, Z in (("vitb", X), ("recipe", R)) for k, lv in lvls.items()]
    with get_context("fork").Pool(8) as pool:
        res = dict(pool.map(r3_job, jobs))
    out["R3_leave_one_level_out"] = res
    print("R3", res, flush=True)

    # R4
    import bridging
    reg = {r["sensor"]: float(r["r_air"]) for r in csv.DictReader(open(ROOT / "data/derived/regime_sensors.csv"))}
    sem = sys.argv[1] if len(sys.argv) > 1 else None
    if sem:
        grid = [(fr, w) for fr in (0.4, 0.5, 0.6) for w in (8, 16, 32)]
        per = {g: {} for g in grid}
        for im in C.index_sem(sem):
            a = bridging.load(im.path)
            per_img = {}
            for w in (8, 16, 32):
                bridging.WIN = w
                per_px = bridging.pitch(a)
                amp = bridging.amplitude(a, per_px)
                m = int(2 * per_px)
                core = amp[m:-m, w:-w]
                for fr in (0.4, 0.5, 0.6):
                    per[(fr, w)].setdefault(im.sensor, []).append(float((core < fr * np.percentile(core, 90)).mean()))
        bridging.WIN = 16
        S = sorted(reg)
        out["R4_gap_loss_vs_log_r_air"] = {f"frac{fr}_win{w}": round(float(spearmanr(
            [np.mean(per[(fr, w)][s]) for s in S], np.log10([reg[s] for s in S])).statistic), 3) for fr, w in grid}
        print("R4", out["R4_gap_loss_vs_log_r_air"], flush=True)

    # R5
    x = lr
    r5 = {}
    for thr in (0.005, 0.01, 0.02):
        dead = 10 ** resp < thr
        flag = np.zeros(len(x), bool)
        for k in range(len(x)):
            m = np.arange(len(x)) != k
            flag[k] = x[k] <= best_threshold(x[m], dead[m])
        r5[f"response<{thr:g}"] = dict(n_dead=int(dead.sum()), auc=round(float(auc(x, dead)), 3), loo=metrics(flag, dead))
    out["R5_screen_thresholds"] = r5
    print("R5", json.dumps(r5), flush=True)
    json.dump(out, open(ROOT / "data/derived/robustness_chain.json", "w"), indent=1)


if __name__ == "__main__":
    main()
