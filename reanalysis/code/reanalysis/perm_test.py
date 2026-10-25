"""Label-permutation null for named dev_eval configs, development sensors only.

Each config is target:stem:agg:kind as in dev_eval --final. The observed rho is
the mean over --repeats repeats of 5-fold CV; each null draw permutes the labels
once and scores one repeat. With several configs, every draw uses the same
permutation for all of them, so the max-over-configs null corrects for having
picked the best one.
"""
import argparse
import csv
from pathlib import Path

import numpy as np

import common as C
from dev_eval import cv_score, locked, read_labels, sensor_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--locked", required=True)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    feat_dir = Path(args.features)
    labels = read_labels(args.labels)
    test = locked(args.locked)
    imgs_csv = feat_dir / "images.csv"
    have = {r["sensor"] for r in csv.DictReader(open(imgs_csv))}
    dev = sorted(s for s in labels if s not in test and s in have)
    params = np.stack([C.fab_features(s) for s in dev])

    data = []
    for cfg in args.configs:
        tname, stem, agg, kind = cfg.split(":")
        X = sensor_features(feat_dir / f"{stem}.npz", imgs_csv, dev, agg)
        if kind == "img+params":
            X = np.hstack([X, params])
        y = np.array([labels[s][tname] for s in dev])
        obs = cv_score(X, y, kind, args.repeats)[0].mean()
        print(f"{cfg:50s} observed rho {obs:+.3f}", flush=True)
        data.append((cfg, X, y, kind, obs))

    rng = np.random.default_rng(0)
    null = np.zeros((args.n_perm, len(data)))
    for k in range(args.n_perm):
        perm = rng.permutation(len(dev))
        for j, (_, X, y, kind, _) in enumerate(data):
            null[k, j] = cv_score(X, y[perm], kind, 1, seed=k)[0][0]
        if (k + 1) % 20 == 0:
            print(f"perm {k + 1}/{args.n_perm}", flush=True)
    mx = null.max(1)
    rows = []
    for j, (cfg, _, _, _, obs) in enumerate(data):
        p = (1 + (null[:, j] >= obs).sum()) / (1 + args.n_perm)
        p_max = (1 + (mx >= obs).sum()) / (1 + args.n_perm)
        rows.append(dict(config=cfg, rho=round(obs, 4), null_mean=round(null[:, j].mean(), 4),
                         null_sd=round(null[:, j].std(), 4), p=round(p, 4), p_maxcorrected=round(p_max, 4)))
        print(f"RESULT {cfg} rho {obs:+.3f} null {null[:, j].mean():+.3f}+/-{null[:, j].std():.3f} "
              f"p {p:.4f} p(max over {len(data)}) {p_max:.4f}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    np.save(out.with_suffix(".npy"), null)


if __name__ == "__main__":
    main()
