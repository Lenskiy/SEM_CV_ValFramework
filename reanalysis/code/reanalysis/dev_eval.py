"""Sensor-level regression from frozen features, development sensors only.

Targets (declared before any image model was run, chosen by run-to-run ICC):
  response  log rel_resp_median  (|R_gas - R_air| / R_air, ICC 0.975 single run)
  snr       log snr_local_median (ICC 0.931)
One row per sensor: tile features averaged over all its images ('mean'), or
mean and s.d. concatenated ('meanstd'). Model: standardise -> PCA (k chosen
by inner CV) -> ridge (alpha by inner CV), or the recipe parameters alone, or both.
Outer loop: 5-fold CV over the 56 development sensors, repeated 20 times.
Metric: Spearman rho of pooled out-of-fold predictions per repeat (mean, s.d.),
plus R^2 and MAE in log10 units; a label-permutation null gives a p-value.

The 15 sensors in data/derived/locked_test_sensors.txt are dropped unless
--final is given; --final fits one named config on all 56 and scores the 15
once, appending the result to data/derived/final_test_log.csv.
"""
import argparse
import csv
import glob
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, KFold, RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C
from qc_metrics import qc_report

TARGETS = {"response": "rel_resp_median", "snr": "snr_local_median"}
ALPHAS = np.logspace(-2, 5, 15)


def read_labels(path):
    out = {}
    for r in csv.DictReader(open(path)):
        if r["excluded_as_outlier"] == "True":
            continue
        out[r["sensor"]] = {k: np.log10(float(r[v])) for k, v in TARGETS.items()}
    return out


def locked(path):
    return [l.strip() for l in open(path) if l.strip() and not l.startswith("#")]


def sensor_features(npz, images_csv, sensors, agg):
    d = np.load(npz)
    img_sensor = {int(r["img"]): r["sensor"] for r in csv.DictReader(open(images_csv))}
    f = d["feats"].astype(np.float32)
    tile_sensor = np.array([img_sensor[int(i)] for i in d["img"]])
    rows = []
    for s in sensors:
        x = f[tile_sensor == s]
        rows.append(np.concatenate([x.mean(0), x.std(0)]) if agg == "meanstd" else x.mean(0))
    return np.stack(rows)


def make_model(kind, n):
    if kind == "params":
        return GridSearchCV(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS},
                            cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error", n_jobs=-1)
    if kind == "linear":       # one fixed measure (edge density), no hyperparameters
        return LinearRegression()
    ks = [k for k in (4, 8, 16, 32) if k < int(0.8 * 0.8 * n)] or [max(1, int(0.8 * 0.8 * n) - 1)]  # tiny n: dry runs
    if kind == "img|params":   # PCA on the image block only; the 4 parameters join after it
        ct = ColumnTransformer([("img", make_pipeline(StandardScaler(), PCA(random_state=0)), slice(0, -4)),
                                ("par", StandardScaler(), slice(-4, None))])
        return GridSearchCV(make_pipeline(ct, Ridge()), {"columntransformer__img__pca__n_components": ks,
                                                         "ridge__alpha": ALPHAS},
                            cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error", n_jobs=-1)
    return GridSearchCV(make_pipeline(StandardScaler(), PCA(random_state=0), Ridge()),
                        {"pca__n_components": ks, "ridge__alpha": ALPHAS},
                        cv=KFold(5, shuffle=True, random_state=0), scoring="neg_mean_squared_error", n_jobs=-1)


def cv_score(X, y, kind, repeats, seed=0):
    rk = RepeatedKFold(n_splits=5, n_repeats=repeats, random_state=seed)
    preds = np.zeros((repeats, len(y)))
    for i, (tr, te) in enumerate(rk.split(X)):
        m = make_model(kind, len(tr)).fit(X[tr], y[tr])
        preds[i // 5, te] = m.predict(X[te])
    rho = np.array([spearmanr(p, y).statistic for p in preds])
    ss = ((y - y.mean()) ** 2).sum()
    r2 = np.array([1 - ((p - y) ** 2).sum() / ss for p in preds])
    mae = np.array([np.abs(p - y).mean() for p in preds])
    return rho, r2, mae, preds.mean(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True, help="directory with *.npz and images.csv")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--locked", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=0, help="permutation null per config (slow; use with --only)")
    ap.add_argument("--only", default="", help="substring filter on feature file names")
    ap.add_argument("--kinds", default="img,img+params", help="comma list from img, img+params, img|params")
    ap.add_argument("--aggs", default="mean,meanstd")
    ap.add_argument("--final", default="", help="config name to score on the locked test set, once")
    args = ap.parse_args()
    feat_dir, out = Path(args.features), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    labels = read_labels(args.labels)
    test = locked(args.locked)
    dev = sorted(s for s in labels if s not in test)
    imgs_csv = feat_dir / "images.csv"
    have = {r["sensor"] for r in csv.DictReader(open(imgs_csv))}
    dev = [s for s in dev if s in have]
    params = np.stack([C.fab_features(s) for s in dev])

    if args.final:
        return final_test(args, labels, dev, test, imgs_csv, params)

    rng = np.random.default_rng(0)
    results = []
    files = sorted(glob.glob(str(feat_dir / "*.npz")))
    files = [f for f in files if args.only in Path(f).name]
    ys = {t: np.array([labels[s][t] for s in dev]) for t in TARGETS}
    for tname, y in ys.items():
        rho, r2, mae, _ = cv_score(params, y, "params", args.repeats)
        results.append(dict(target=tname, config="params_only", rho=rho.mean(), rho_sd=rho.std(),
                            r2=r2.mean(), mae=mae.mean(), perm_p=np.nan))
        print(f"{tname:8s} {'params_only':40s} rho {rho.mean():+.3f} +/- {rho.std():.3f}  R2 {r2.mean():+.3f}  MAE {mae.mean():.3f}", flush=True)
    for f in files:
        for agg in args.aggs.split(","):
            X = sensor_features(f, imgs_csv, dev, agg)
            for kind in args.kinds.split(","):
                Xk = X if kind == "img" else np.hstack([X, params])
                name = f"{Path(f).stem}_{agg}_{kind}"
                for tname, y in ys.items():
                    rho, r2, mae, _ = cv_score(Xk, y, kind, args.repeats)
                    null = [cv_score(Xk, rng.permutation(y), kind, 1, seed=k)[0][0] for k in range(args.n_perm)]
                    p = (1 + sum(n >= rho.mean() for n in null)) / (1 + len(null)) if null else np.nan
                    results.append(dict(target=tname, config=name, rho=rho.mean(), rho_sd=rho.std(),
                                        r2=r2.mean(), mae=mae.mean(), perm_p=p))
                    print(f"{tname:8s} {name:40s} rho {rho.mean():+.3f} +/- {rho.std():.3f}  R2 {r2.mean():+.3f}  "
                          f"MAE {mae.mean():.3f}  perm p {p:.3f}", flush=True)
    with open(out / "dev_results.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(results[0]))
        w.writeheader()
        w.writerows(results)
    print(f"RESULT dev_eval done: {len(results)} configs, {len(dev)} dev sensors")


def final_test(args, labels, dev, test, imgs_csv, params):
    """Fit one config on all development sensors and score the locked test set."""
    lsha = hashlib.sha256(open(args.locked, "rb").read()).hexdigest()[:16]
    log = Path(args.labels).parent / "final_test_log.csv"
    if log.exists() and any(r.get("locked_sha") == lsha for r in csv.DictReader(open(log))):
        raise SystemExit("this locked set has already been scored; it is scored once")
    tname, stem, agg, kind = args.final.split(":")
    f = Path(args.features) / f"{stem}.npz"
    y = np.array([labels[s][tname] for s in dev])
    yt = np.array([labels[s][tname] for s in test])
    pt = np.stack([C.fab_features(s) for s in test])
    X = sensor_features(f, imgs_csv, dev, agg)
    Xt = sensor_features(f, imgs_csv, test, agg)
    if kind in ("img+params", "img|params"):
        X, Xt = np.hstack([X, params]), np.hstack([Xt, pt])
    m = make_model(kind, len(dev)).fit(X, y)
    p = m.predict(Xt)
    rho = spearmanr(p, yt).statistic
    dev_oof = cv_score(X, y, kind, 1)[3]          # repeat 0 out-of-fold scores, for the QC cutoff
    rng = np.random.default_rng(0)
    null = np.array([spearmanr(p, rng.permutation(yt)).statistic for _ in range(10000)])
    res = dict(when=datetime.now().isoformat(timespec="seconds"), config=args.final, n_test=len(test),
               rho=round(float(rho), 4), r2=round(float(1 - ((p - yt) ** 2).sum() / ((yt - yt.mean()) ** 2).sum()), 4),
               mae=round(float(np.abs(p - yt).mean()), 4),
               perm_p=round(float((1 + (null >= rho).sum()) / 10001), 4),
               locked_sha=hashlib.sha256(open(args.locked, "rb").read()).hexdigest()[:16],
               best_params=json.dumps(m.best_params_), **qc_report(dev_oof, y, p, yt))
    log = Path(args.labels).parent / "final_test_log.csv"
    new = not log.exists()
    with open(log, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res))
        if new:
            w.writeheader()
        w.writerow(res)
    print("FINAL", res)


if __name__ == "__main__":
    main()
