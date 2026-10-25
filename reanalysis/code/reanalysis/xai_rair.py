"""E16: exact tile explanation of the RESISTANCE model (development sensors only).

Same as xai_tiles.py step 1, but the target is log10 baseline resistance (median before each
pulse, data/derived/r_air.csv): standardise -> PCA -> ridge on the sensor mean of frozen ViT-B
tile features (dinov2_vitb14_n448_pct), first repeat of the 5-fold sensor-held-out split, so
each tile's out-of-fold prediction is its exact contribution. The eight morphology types are the
SAME ones as for the response model (type column of data/xai_500x/tile_scores.csv), so the two
explanations can be compared type by type. Also repeats the tile-removal check (E4) for this
model: removing the three overgrowth types (2, 7, 3) vs the same number of random tiles, in the
high-deposition group (PC1 < -30) and the rest.
Writes data/xai_500x/tile_scores_rair.csv and data/xai_500x/xai_rair.json.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

from dev_eval import locked, make_model, read_labels

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features500x"
OVERGROWTH = (2, 7, 3)


def main():
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    imgs = {int(r["img"]): r for r in csv.DictReader(open(FEAT / "images.csv"))}
    d = np.load(FEAT / "dinov2_vitb14_n448_pct.npz")
    F = d["feats"].astype(np.float32)
    ts = np.array([imgs[int(i)]["sensor"] for i in d["img"]])
    dev = sorted(s for s in lab if s not in test and s in set(ts))
    keep = np.isin(ts, dev)
    F, ts, ti, tr_, tc = F[keep], ts[keep], d["img"][keep], d["row"][keep], d["col"][keep]
    old = list(csv.DictReader(open(ROOT / "data/xai_500x/tile_scores.csv")))
    key = lambda s, i, r, c: (s, int(i), int(r), int(c))
    typ_of = {key(o["sensor"], o["img"], o["row"], o["col"]): int(o["type"]) for o in old}
    resp_of = {key(o["sensor"], o["img"], o["row"], o["col"]): float(o["oof_score"]) for o in old}
    typ = np.array([typ_of[key(*k)] for k in zip(ts, ti, tr_, tc)])
    sresp = np.array([resp_of[key(*k)] for k in zip(ts, ti, tr_, tc)])
    ra = {r["sensor"]: math.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    X = np.stack([F[ts == s].mean(0) for s in dev])
    y = np.array([ra[s] for s in dev])
    score, pred = np.full(len(F), np.nan), {}
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=1, random_state=0).split(X):
        m = make_model("img", len(tr)).fit(X[tr], y[tr]).best_estimator_
        for i in te:
            mk = ts == dev[i]
            score[mk] = m.predict(F[mk])
            pred[dev[i]] = float(m.predict(X[i:i + 1])[0])
    gap = max(abs(score[ts == s].mean() - pred[s]) for s in dev)
    out = dict(oof_rho_repeat0=round(float(spearmanr([pred[s] for s in dev], y).statistic), 3),
               max_exactness_gap=float(gap))
    mean_all = float(score.mean())
    out["types"] = {int(k): dict(share=round(float(np.mean(typ == k)), 3),
                                 mean_tile_score_r_air=round(float(score[typ == k].mean()), 3),
                                 mean_tile_score_response=round(float(sresp[typ == k].mean()), 3))
                    for k in range(8)}
    out["mean_tile_score_r_air_all"] = round(mean_all, 3)
    ks = list(range(8))
    out["rank_agreement_types_rho"] = round(float(spearmanr([out["types"][k]["mean_tile_score_r_air"] for k in ks],
                                                            [out["types"][k]["mean_tile_score_response"] for k in ks]).statistic), 3)
    out["tile_level_rho_r_air_vs_response_scores"] = round(float(spearmanr(score, sresp).statistic), 3)
    # tile removal (E4 for the resistance model)
    Z = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    hi = {s for s, z in zip(dev, Z[:, 0]) if z < -30}
    rng = np.random.default_rng(0)
    d_og, d_rand = {}, {}
    for s in dev:
        mk = ts == s
        sc, tp = score[mk], typ[mk]
        m = np.isin(tp, OVERGROWTH)
        if m.sum() == 0 or m.all():
            d_og[s] = d_rand[s] = 0.0
            continue
        d_og[s] = float(sc[~m].mean() - sc.mean())
        d_rand[s] = float(np.mean([np.delete(sc, rng.choice(len(sc), m.sum(), replace=False)).mean() - sc.mean()
                                   for _ in range(100)]))
    for g, ss in (("high_deposition", [s for s in dev if s in hi]), ("rest", [s for s in dev if s not in hi])):
        out[f"removal_{g}"] = dict(n=len(ss), change_overgrowth_removed=round(float(np.mean([d_og[s] for s in ss])), 3),
                                   change_random_removed=round(float(np.mean([d_rand[s] for s in ss])), 3))
    with open(ROOT / "data/xai_500x/tile_scores_rair.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "img", "row", "col", "type", "oof_score"])
        w.writerows(zip(ts, ti, tr_, tc, typ, np.round(score, 4)))
    json.dump(out, open(ROOT / "data/xai_500x/xai_rair.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
