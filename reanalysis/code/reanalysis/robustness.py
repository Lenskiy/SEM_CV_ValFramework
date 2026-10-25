"""Next experiments E1, E3, E4 of the analysis plan (development sensors unless stated).

E1  bootstrap 95% intervals (2,000 resamples of sensors, predictions held fixed):
    development rho / R2 / QC AUC of the frozen model (5 repeats of 5-fold CV, the
    statistic averaged over repeats), the locked-test rho and AUC (15 sensors), and
    the 2,500x edge-density rho (12 sensors)
E3  baseline air resistance per sensor (median over cycles of the mean raw
    resistance in the 10 points before gas-on) against log response, PC1 of the
    sensor feature map, the overgrowth type share, and 2,500x edge density
E4  tile removal: each development sensor's out-of-fold prediction recomputed
    without its tiles of the three lowest-scoring types, vs the same number of
    random tiles (100 draws); high-deposition group (PC1 < -30) vs the rest
Writes data/derived/robustness.json and data/derived/r_air.csv.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import StandardScaler

import common as C
from compute_snr import extract_params_from_filename, find_cycles, load_sheet_data, read_first_sheet
from dev_eval import locked, make_model, read_labels, sensor_features
from qc_metrics import NONRESP_LOG10, auc

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features500x"
CFG = "dinov2_vitb14_n448_pct.npz"
OVERGROWTH = (2, 7, 3)          # rough overgrowth, granular film with flakes, defects (type_names.txt)
B = 2000


def ci(stat, n, rng):
    vals = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        v = stat(idx)
        if np.isfinite(v):
            vals.append(v)
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


def r2(p, y):
    return 1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()


def main():
    rng = np.random.default_rng(0)
    out = {}
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = locked(ROOT / "data/derived/locked_test_sensors.txt")
    dev = sorted(s for s in lab if s not in set(test))
    y = np.array([lab[s]["response"] for s in dev])
    X = sensor_features(FEAT / CFG, FEAT / "images.csv", dev, "mean")

    # ---------------- E1
    P = np.zeros((5, len(y)))
    for i, (tr, te) in enumerate(RepeatedKFold(n_splits=5, n_repeats=5, random_state=0).split(X)):
        P[i // 5, te] = make_model("img", len(tr)).fit(X[tr], y[tr]).predict(X[te])
    nr = y < NONRESP_LOG10
    dev_stats = {"rho": lambda idx: np.mean([spearmanr(p[idx], y[idx]).statistic for p in P]),
                 "r2": lambda idx: np.mean([r2(p[idx], y[idx]) for p in P]),
                 "qc_auc": lambda idx: np.mean([auc(p[idx], nr[idx]) for p in P])}
    e1 = {}
    for k, f in dev_stats.items():
        e1[f"dev_{k}"] = dict(value=round(float(f(np.arange(len(y)))), 3), ci95=ci(f, len(y), rng))
    tst = [s for s in test if s in lab]
    Xt = sensor_features(FEAT / CFG, FEAT / "images.csv", tst, "mean")
    yt = np.array([lab[s]["response"] for s in tst])
    pt = make_model("img", len(dev)).fit(X, y).predict(Xt)
    tn = yt < NONRESP_LOG10
    e1["locked_rho"] = dict(value=round(float(spearmanr(pt, yt).statistic), 3),
                            ci95=ci(lambda i: spearmanr(pt[i], yt[i]).statistic, len(yt), rng))
    e1["locked_qc_auc"] = dict(value=round(auc(pt, tn), 3), ci95=ci(lambda i: auc(pt[i], tn[i]), len(yt), rng))
    alllab = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/derived/labels_v2.csv"))}
    hm = [r for r in csv.DictReader(open(ROOT / "data/derived/hm_morphology.csv")) if r["mag"] == "2500x"]
    e = np.array([float(r["edge_density"]) for r in hm])
    ye = np.log10([float(alllab[r["sensor"]]["rel_resp_median"]) for r in hm])
    e1["edge2500_rho"] = dict(value=round(float(spearmanr(e, ye).statistic), 3),
                              ci95=ci(lambda i: spearmanr(e[i], ye[i]).statistic, len(e), rng))
    out["E1"] = e1
    print("E1 bootstrap 95% intervals")
    for k, v in e1.items():
        print(f"   {k:16s} {v['value']:+.3f}  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")

    # ---------------- E3
    rows = []
    for path in sorted((ROOT / "data/new_sensor_responses").glob("*.xlsx")):
        con, cur, tim = extract_params_from_filename(path.name).split("_")
        s = f"{con[:-1]}M-{cur[:-1]}uA-{tim[:-1]}s"
        sig = load_sheet_data(read_first_sheet(path))
        ra = [np.nanmean(sig[max(0, on - 10):on, 1]) for on, _ in find_cycles(sig)]
        rows.append(dict(sensor=s, r_air_median=float(np.nanmedian(ra)), r_air_first=float(ra[0]), n_cycles=len(ra)))
    with open(ROOT / "data/derived/r_air.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    ra = {r["sensor"]: np.log10(r["r_air_median"]) for r in rows}
    Z = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(X))
    st = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/xai_500x/sensor_types.csv"))}
    og = np.array([sum(float(st[s][f"type{k}"]) for k in OVERGROWTH) for s in dev])
    lr = np.array([ra[s] for s in dev])
    e3 = {}
    for name, v in (("log_response", y), ("PC1", Z[:, 0]), ("overgrowth_share", og)):
        r = spearmanr(lr, v)
        e3[f"vs_{name}"] = dict(rho=round(float(r.statistic), 3), p=round(float(r.pvalue), 4), n=len(dev))
    hs = [r["sensor"] for r in hm]
    r = spearmanr([ra[s] for s in hs], e)
    e3["vs_edge2500"] = dict(rho=round(float(r.statistic), 3), p=round(float(r.pvalue), 4), n=len(hs))
    left = Z[:, 0] < -30
    e3["median_log_r_air"] = dict(high_deposition_group=round(float(np.median(lr[left])), 2),
                                  rest=round(float(np.median(lr[~left])), 2),
                                  bare_10s_250uA=round(float(np.median([ra[s] for s in hs if s.endswith("-10s")])), 2))
    out["E3"] = e3
    print("E3 log10 air resistance (ohm), Spearman")
    for k, v in e3.items():
        print(f"   {k:20s} {v}")

    # ---------------- E4
    T = list(csv.DictReader(open(ROOT / "data/xai_500x/tile_scores.csv")))
    by = {}
    for t in T:
        by.setdefault(t["sensor"], []).append((int(t["type"]), float(t["oof_score"])))
    d_og, d_rand = {}, {}
    for s in dev:
        typ = np.array([a for a, _ in by[s]])
        sc = np.array([b for _, b in by[s]])
        m = np.isin(typ, OVERGROWTH)
        if m.sum() == 0 or m.all():
            d_og[s], d_rand[s] = 0.0, 0.0
            continue
        d_og[s] = float(sc[~m].mean() - sc.mean())
        d_rand[s] = float(np.mean([np.delete(sc, rng.choice(len(sc), m.sum(), replace=False)).mean() - sc.mean()
                                   for _ in range(100)]))
    g = {"high_deposition": [s for s, l in zip(dev, left) if l], "rest": [s for s, l in zip(dev, left) if not l]}
    e4 = {k: dict(n=len(v), mean_change_overgrowth_removed=round(float(np.mean([d_og[s] for s in v])), 3),
                  mean_change_random_removed=round(float(np.mean([d_rand[s] for s in v])), 3),
                  mean_overgrowth_share=round(float(np.mean([og[dev.index(s)] for s in v])), 3)) for k, v in g.items()}
    e4["all"] = dict(rho_change_vs_overgrowth_share=round(float(spearmanr([d_og[s] for s in dev], og).statistic), 3))
    out["E4"] = e4
    print("E4 change in predicted log10 response when tiles are removed")
    for k, v in e4.items():
        print(f"   {k:16s} {v}")
    json.dump(out, open(ROOT / "data/derived/robustness.json", "w"), indent=1)


if __name__ == "__main__":
    main()
