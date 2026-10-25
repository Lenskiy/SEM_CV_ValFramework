"""Response permutation null corrected over every configuration compared before the locked test.

The paper's p = 0.03 was corrected over the 6 ViT-B configurations only. The backbone was chosen
after comparing all configurations in dev_eval_500x/dev_results_all.csv (DINOv2 ViT-B and ViT-S,
ResNet-50; three tilings; raw / percentile; mean / mean+s.d.; images, images+params) plus the
recipe-only model and the two img|params variants (dev_results_sep_*): 67 configurations.
Same procedure as perm_test.py: observed rho = mean over 5 repeats of 5-fold CV; each null draw
permutes the labels once (the same permutation for every configuration) and scores one repeat;
the max over configurations gives the corrected p. MicroNet was evaluated after the locked score
and is not part of the selection, so it is not included.
Chunked for a cluster: permutations are generated in full (seed 0) and --start/--stop selects a
slice; --part k writes <out-dir>/part_<k>.npy (observed rho with --observed). --merge combines
the parts into perm_full_response.csv/.npy in --out-dir.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker process
    os.environ[_v] = "1"

import argparse
import csv
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import common as C
import dev_eval
from dev_eval import cv_score, locked, read_labels, sensor_features

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features500x"
DE = ROOT / "data/dev_eval_500x"
_make = dev_eval.make_model
DATA = []


def make_serial(kind, n):
    m = _make(kind, n)
    return m.set_params(n_jobs=1) if hasattr(m, "n_jobs") else m


def configs():
    cfgs = sorted({r["config"] for r in csv.DictReader(open(DE / "dev_results_all.csv")) if r["target"] == "response"})
    for f in sorted(DE.glob("dev_results_sep_*.csv")):
        cfgs += [r["config"] for r in csv.DictReader(open(f)) if r["target"] == "response" and "|" in r["config"]]
    return cfgs


def build(dev, labels):
    params = np.stack([C.fab_features(s) for s in dev])
    y = np.array([labels[s]["response"] for s in dev])
    out = []
    for cfg in configs():
        if cfg == "params_only":
            out.append((cfg, params, "params"))
            continue
        kind = cfg.split("_")[-1]
        agg = cfg.split("_")[-2]
        stem = "_".join(cfg.split("_")[:-2])
        X = sensor_features(FEAT / f"{stem}.npz", FEAT / "images.csv", dev, agg)
        if kind in ("img+params", "img|params"):
            X = np.hstack([X, params])
        out.append((cfg, X, kind))
    return y, out


def one_perm(args):
    k, perm = args
    dev_eval.make_model = make_serial
    y = DATA[0]
    return [cv_score(X, y[perm], kind, 1, seed=k)[0][0] for _, X, kind in DATA[1]]


def observed(args):
    j = args
    dev_eval.make_model = make_serial
    _, X, kind = DATA[1][j]
    return cv_score(X, DATA[0], kind, 5)[0].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "data/features500x"))
    ap.add_argument("--configs-dir", default=str(ROOT / "data/dev_eval_500x"))
    ap.add_argument("--labels", default=str(ROOT / "data/derived/labels_v2.csv"))
    ap.add_argument("--locked", default=str(ROOT / "data/derived/locked_test_sensors.txt"))
    ap.add_argument("--out-dir", default=str(ROOT / "data/dev_eval_500x/perm_full"))
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=None)
    ap.add_argument("--observed", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    globals().update(FEAT=Path(a.features), DE=Path(a.configs_dir))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cfg_names = configs()
    if a.merge:
        return merge(out, cfg_names, a.n_perm)
    labels = read_labels(a.labels)
    test = locked(a.locked)
    have = {r["sensor"] for r in csv.DictReader(open(FEAT / "images.csv"))}
    dev = sorted(s for s in labels if s not in test and s in have)
    y, cf = build(dev, labels)
    DATA.extend([y, cf])
    print(f"{len(cf)} configurations, {len(dev)} sensors", flush=True)
    t0 = time.time()
    with get_context("fork").Pool(a.workers) as pool:
        if a.observed:
            obs = np.array(pool.map(observed, range(len(cf))))
            np.save(out / "observed.npy", obs)
            print(f"observed done in {time.time() - t0:.0f} s; best {cf[int(obs.argmax())][0]} {obs.max():+.3f}", flush=True)
            return
        rng = np.random.default_rng(0)
        perms = [(k, rng.permutation(len(dev))) for k in range(a.n_perm)][a.start:a.stop]
        null = []
        for k, row in enumerate(pool.imap(one_perm, perms)):
            null.append(row)
            print(f"perm {perms[k][0]}  {time.time() - t0:.0f} s", flush=True)
    np.save(out / f"part_{a.start:04d}.npy", np.array(null))


def merge(out, names, n_perm):
    obs = np.load(out / "observed.npy")
    null = np.concatenate([np.load(f) for f in sorted(out.glob("part_*.npy"))])
    assert len(null) == n_perm, (len(null), n_perm)
    mx = null.max(1)
    rows = []
    for j, cfg in enumerate(names):
        rows.append(dict(config=cfg, rho=round(float(obs[j]), 4), null_mean=round(float(null[:, j].mean()), 4),
                         null_sd=round(float(null[:, j].std()), 4),
                         p=round(float((1 + (null[:, j] >= obs[j]).sum()) / (1 + n_perm)), 4),
                         p_max_all=round(float((1 + (mx >= obs[j]).sum()) / (1 + n_perm)), 4)))
    rows.sort(key=lambda r: -r["rho"])
    with open(DE / "perm_full_response.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.save(DE / "perm_full_response.npy", null)
    for r in rows[:8]:
        print("RESULT", r, flush=True)
    print("PRE-REGISTERED CONFIG", next(r for r in rows if r["config"] == "dinov2_vitb14_n448_pct_mean_img"), flush=True)


if __name__ == "__main__":
    main()
