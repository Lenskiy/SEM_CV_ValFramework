"""Sensor-level baselines and an acquisition-confound probe (CPU only).

Same sensor-held-out folds as train_cv.py --cv sensor. Random forest, linear
(logistic / ridge on log SNR) and 3-nearest-neighbour models on:
  params      fabrication parameters (log conc, log current, time, log charge)
  image_stats grey-level statistics, Otsu coverage, edge density, GLCM, LBP
              (per image, averaged over each sensor's images)
  image_stats+params
  sem_settings applied brightness/contrast, working distance, dwell time,
              emission current and acquisition date only. If these predict the
              performance category, imaging conditions are a possible shortcut.

Also writes the imaging-session table (sensors per category per acquisition
date). Outputs baselines.json, baseline_predictions.csv, imaging_sessions.csv.
"""
import argparse
import csv
import json
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import common as C


def image_stats(a):
    from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
    from skimage.filters import sobel, threshold_otsu
    f = a.astype(np.float32) / 255.0
    q = np.percentile(f, [5, 25, 50, 75, 95])
    small = a[::4, ::4]
    glcm = graycomatrix((small // 8).astype(np.uint8), distances=[1, 4], angles=[0, np.pi / 2],
                        levels=32, symmetric=True, normed=True)
    tex = [graycoprops(glcm, p).mean(axis=1) for p in ("contrast", "homogeneity", "energy", "correlation")]
    lbp = local_binary_pattern(small, P=8, R=1, method="uniform")
    hist = np.bincount(lbp.astype(int).ravel(), minlength=10)[:10] / lbp.size
    return np.concatenate([[f.mean(), f.std()], q, [(a > threshold_otsu(a)).mean(), sobel(f).mean()],
                           np.concatenate(tex), hist])


def sem_settings(meta):
    t = meta.get("time", "")[:10]
    day = date.fromisoformat(t).toordinal() if t else np.nan

    def num(k):
        try:
            return float(meta.get(k, "nan"))
        except ValueError:
            return np.nan
    return np.array([num("brightness"), num("contrast"), num("working_distance"), num("dwell_time"),
                     num("emission_current"), day])


def make_models(kind, seed):
    """(classifier, regressor, log_target) for a model family."""
    if kind == "rf":
        return (RandomForestClassifier(n_estimators=500, random_state=seed, class_weight="balanced"),
                RandomForestRegressor(n_estimators=500, random_state=seed), False)
    if kind == "linear":
        return (make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)),
                make_pipeline(StandardScaler(), Ridge(alpha=1.0)), True)
    if kind == "knn3":
        return (make_pipeline(StandardScaler(), KNeighborsClassifier(3)),
                make_pipeline(StandardScaler(), KNeighborsRegressor(3)), True)
    raise ValueError(kind)


def evaluate(X, y_snr, y_cat, folds, sensors, seed=0, kind="rf"):
    pred_c, pred_r = {}, {}
    idx = {s: i for i, s in enumerate(sensors)}
    for test in folds:
        te = [idx[s] for s in test]
        tr = [i for i in range(len(sensors)) if sensors[i] not in set(test)]
        clf, reg, log_t = make_models(kind, seed)
        clf.fit(X[tr], y_cat[tr])
        reg.fit(X[tr], np.log(y_snr[tr]) if log_t else y_snr[tr])
        r_pred = reg.predict(X[te])
        r_pred = np.exp(r_pred) if log_t else r_pred
        for i, c, r in zip(te, clf.predict(X[te]), r_pred):
            pred_c[sensors[i]], pred_r[sensors[i]] = int(c), float(r)
    pc = np.array([pred_c[s] for s in sensors])
    pr = np.array([pred_r[s] for s in sensors])
    ss_tot = ((y_snr - y_snr.mean()) ** 2).sum()
    return dict(acc=float((pc == y_cat).mean()),
                balanced_acc=float(np.mean([(pc[y_cat == k] == k).mean() for k in np.unique(y_cat)])),
                mae=float(np.abs(pr - y_snr).mean()), rmse=float(np.sqrt(((pr - y_snr) ** 2).mean())),
                r2=float(1 - ((pr - y_snr) ** 2).sum() / ss_tot)), pc, pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem", required=True)
    ap.add_argument("--snr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--include-outlier", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    labels = C.load_labels(args.snr, include_outlier=args.include_outlier)
    images = [im for im in C.index_sem(args.sem) if im.sensor in labels]
    sensors = sorted(labels)
    y_snr = np.array([labels[s] for s in sensors])
    y_cat = np.array([C.snr_category(v) for v in y_snr])
    folds = C.sensor_folds(sensors, list(y_cat), args.folds, seed=args.seed)

    per_img_stats, per_img_set = {}, {}
    for im in images:
        per_img_stats.setdefault(im.sensor, []).append(image_stats(C.load_image_array(im.path)))
        per_img_set.setdefault(im.sensor, []).append(sem_settings(im.meta))
    feats = dict(
        params=np.stack([C.fab_features(s) for s in sensors]),
        image_stats=np.stack([np.mean(per_img_stats[s], 0) for s in sensors]),
        sem_settings=np.stack([np.nanmean(per_img_set[s], 0) for s in sensors]),
    )
    feats["image_stats+params"] = np.hstack([feats["image_stats"], feats["params"]])

    results, preds = {}, {}
    for name, X in feats.items():
        X = np.nan_to_num(X, nan=np.nanmean(X))
        for kind in ("rf", "linear", "knn3"):
            key = f"{name}|{kind}"
            results[key], pc, pr = evaluate(X, y_snr, y_cat, folds, sensors, args.seed, kind)
            preds[key] = (pc, pr)
            print(f"{name:20s} {kind:7s} acc {results[key]['acc']:.3f} bal {results[key]['balanced_acc']:.3f} "
                  f"MAE {results[key]['mae']:.2f} R2 {results[key]['r2']:.3f}")
    results["majority_class_acc"] = float(np.bincount(y_cat).max() / len(y_cat))
    results["n_sensors"] = len(sensors)
    results["folds"] = folds
    (out / "baselines.json").write_text(json.dumps(results, indent=1))

    with open(out / "baseline_predictions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "true_snr", "true_cat"] + [f"{n}_{k}" for n in preds for k in ("cat", "snr")])
        for i, s in enumerate(sensors):
            w.writerow([s, y_snr[i], y_cat[i]] + [v for n in preds for v in (preds[n][0][i], round(preds[n][1][i], 3))])

    with open(out / "imaging_sessions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "category", "n_images", "acquisition_dates"])
        for s in sensors:
            dates = sorted({im.meta.get("time", "")[:10] for im in images if im.sensor == s})
            w.writerow([s, C.CATEGORIES[C.snr_category(labels[s])], sum(im.sensor == s for im in images), " ".join(dates)])
    print("RESULT baselines done")


if __name__ == "__main__":
    main()
