"""Tile-level explanation of the pre-registered 500x model, and morphology types.

The model (standardise -> PCA -> ridge on the sensor mean of ViT-B tile features)
is affine, so a sensor's prediction is exactly the mean of its tiles' individual
predictions. Each tile's prediction is therefore its exact contribution: no
approximation is involved, unlike gradient or occlusion maps.

1. Tile scores out of fold: 5-fold sensor-held-out split (the first repeat of
   dev_eval's RepeatedKFold), each held-out sensor's tiles scored by a model that
   never saw that sensor.
2. Morphology types: k-means (k = --k) on the development tiles' features, with no
   labels involved. Per sensor: the fraction of its tiles in each type.
3. Per type: mean out-of-fold tile score, and Spearman rho across sensors of the
   type's fraction vs log response and vs each recipe parameter.
4. Montage of the tiles nearest each centroid, so the types can be named.

Locked test sensors are left out of every step. Writes to --out:
tile_scores.csv, type_summary.csv, sensor_types.csv, montage_types.png, types_model.pkl.
"""
import argparse
import csv
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

import common as C
from dev_eval import locked, make_model, read_labels

TILE = 448


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--sem", required=True, help="500x image folder, for the montage")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--locked", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    feat = Path(a.features)

    labels = read_labels(a.labels)
    test = set(locked(a.locked))
    imgs = {int(r["img"]): r for r in csv.DictReader(open(feat / "images.csv"))}
    d = np.load(feat / "dinov2_vitb14_n448_pct.npz")
    F = d["feats"].astype(np.float32)
    t_sensor = np.array([imgs[int(i)]["sensor"] for i in d["img"]])
    dev = sorted(s for s in labels if s not in test and s in set(t_sensor))
    keep = np.isin(t_sensor, dev)
    F, t_sensor, t_img, t_row, t_col = F[keep], t_sensor[keep], d["img"][keep], d["row"][keep], d["col"][keep]
    y = {s: labels[s]["response"] for s in dev}

    # 1. out-of-fold tile scores
    X = np.stack([F[t_sensor == s].mean(0) for s in dev])
    yv = np.array([y[s] for s in dev])
    score = np.full(len(F), np.nan)
    sensor_pred = {}
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=1, random_state=0).split(X):
        m = make_model("img", len(tr)).fit(X[tr], yv[tr]).best_estimator_
        for i in te:
            s = dev[i]
            mask = t_sensor == s
            score[mask] = m.predict(F[mask])
            sensor_pred[s] = float(m.predict(X[i:i + 1])[0])
    gap = max(abs(score[t_sensor == s].mean() - sensor_pred[s]) for s in dev)
    rho = spearmanr([sensor_pred[s] for s in dev], yv).statistic
    print(f"out-of-fold sensor rho {rho:+.3f}; max |mean tile score - sensor prediction| = {gap:.2e} (exact if ~0)")

    # 2. morphology types (unsupervised)
    sc = StandardScaler().fit(F)
    pca = PCA(50, random_state=0).fit(sc.transform(F))
    Z = pca.transform(sc.transform(F))
    km = KMeans(a.k, n_init=10, random_state=0).fit(Z)
    with open(out / "types_model.pkl", "wb") as fh:      # reused by xai_claim2.py (replicates, k checks)
        pickle.dump(dict(scaler=sc, pca=pca, kmeans=km), fh)
    lab = km.labels_
    frac = np.array([[np.mean(lab[t_sensor == s] == k) for k in range(a.k)] for s in dev])
    rec = np.array([C.parse_sensor_key(s) for s in dev])

    # 3. per-type summary
    rows = []
    for k in range(a.k):
        r_resp = spearmanr(frac[:, k], yv)
        rows.append(dict(type=k, n_tiles=int((lab == k).sum()), share=round(float(np.mean(lab == k)), 3),
                         n_sensors_present=int((frac[:, k] > 0.05).sum()),
                         mean_tile_score=round(float(score[lab == k].mean()), 3),
                         rho_fraction_vs_response=round(float(r_resp.statistic), 3), p=round(float(r_resp.pvalue), 4),
                         rho_vs_conc=round(float(spearmanr(frac[:, k], rec[:, 0]).statistic), 2),
                         rho_vs_current=round(float(spearmanr(frac[:, k], rec[:, 1]).statistic), 2),
                         rho_vs_time=round(float(spearmanr(frac[:, k], rec[:, 2]).statistic), 2)))
    with open(out / "type_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{'type':>4s} {'share':>6s} {'sensors':>7s} {'tile score':>10s} {'rho resp':>8s} {'p':>7s}  conc  curr  time")
    for r in sorted(rows, key=lambda r: r["mean_tile_score"]):
        print(f"{r['type']:4d} {r['share']:6.3f} {r['n_sensors_present']:7d} {r['mean_tile_score']:10.2f} "
              f"{r['rho_fraction_vs_response']:+8.2f} {r['p']:7.4f}  {r['rho_vs_conc']:+.2f} {r['rho_vs_current']:+.2f} "
              f"{r['rho_vs_time']:+.2f}")

    with open(out / "tile_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "img", "row", "col", "type", "oof_score"])
        w.writerows(zip(t_sensor, t_img, t_row, t_col, lab, np.round(score, 4)))
    with open(out / "sensor_types.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "log_response", "oof_pred"] + [f"type{k}" for k in range(a.k)])
        for i, s in enumerate(dev):
            w.writerow([s, round(y[s], 4), round(sensor_pred[s], 4)] + list(np.round(frac[i], 4)))

    # 4. montage: 8 tiles nearest each centroid, one row per type (ordered by mean tile score)
    paths = {}
    for im in C.index_sem(a.sem):
        paths[(im.sensor, im.path.name)] = im.path
    cache, th, n_ex = {}, 150, 8
    order = sorted(range(a.k), key=lambda k: rows[k]["mean_tile_score"])
    canvas = Image.new("L", (n_ex * (th + 4) + 4, a.k * (th + 4) + 4), 255)
    for ri, k in enumerate(order):
        idx = np.flatnonzero(lab == k)
        dist = np.linalg.norm(Z[idx] - km.cluster_centers_[k], axis=1)
        picked, used = [], set()
        for j in idx[np.argsort(dist)]:
            if t_sensor[j] not in used:          # one tile per sensor, so the row is not one sensor
                picked.append(j)
                used.add(t_sensor[j])
            if len(picked) == n_ex:
                break
        for ci, j in enumerate(picked):
            r = imgs[int(t_img[j])]
            key = (r["sensor"], r["file"])
            if key not in cache:
                cache[key] = C.load_image_array(paths[key])
            a_ = cache[key][int(t_row[j]) * TILE:(int(t_row[j]) + 1) * TILE, int(t_col[j]) * TILE:(int(t_col[j]) + 1) * TILE]
            lo, hi = np.percentile(a_, [1, 99])
            a_ = np.clip((a_ - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)
            canvas.paste(Image.fromarray(a_).resize((th, th)), (4 + ci * (th + 4), 4 + ri * (th + 4)))
    canvas.save(out / "montage_types.png")
    print("montage rows, top to bottom: types", order, "(lowest to highest mean tile score)")


if __name__ == "__main__":
    main()
