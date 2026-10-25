"""Claim 2, steps 4, 5 and 7 of the analysis plan (500x morphology types).

Loads the k-means types fitted by xai_tiles.py (types_model.pkl; checked
against tile_scores.csv), then:
  4. mediation: repeated sensor-held-out CV rho of recipe parameters alone vs
     recipe + type shares vs type shares alone; ridge weights of the recipe
     with and without the type shares
  5. stability: ICC(1) of each type's share across a sensor's images; the
     same for the 14 sensors imaged in two sessions (session means); and
     persistence of each type at k = 6 and k = 10 (best-overlap match, Jaccard)
  7. replicates: type shares of each replicate vs its original
Locked test sensors are not used. Prints everything; writes claim2.json to --out.
"""
import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.cluster import KMeans

import common as C
from dev_eval import cv_score, locked, make_model, read_labels


def icc1(groups):
    """ICC(1) from a list of 1-D arrays (one per sensor), unequal sizes allowed."""
    k = np.array([len(g) for g in groups], float)
    n, N = len(groups), k.sum()
    gm = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - gm) ** 2 for g in groups) / (n - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - n)
    k0 = (N - (k ** 2).sum() / N) / (n - 1)
    return float((msb - msw) / (msb + (k0 - 1) * msw))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--replicate-features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--replicate-labels", required=True)
    ap.add_argument("--locked", required=True)
    ap.add_argument("--xai", required=True, help="xai_tiles.py output folder")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    feat, xai = Path(a.features), Path(a.xai)
    labels = read_labels(a.labels)
    test = set(locked(a.locked))
    imgs = {int(r["img"]): r for r in csv.DictReader(open(feat / "images.csv"))}
    d = np.load(feat / "dinov2_vitb14_n448_pct.npz")
    F = d["feats"].astype(np.float32)
    t_sensor = np.array([imgs[int(i)]["sensor"] for i in d["img"]])
    dev = sorted(s for s in labels if s not in test and s in set(t_sensor))
    keep = np.isin(t_sensor, dev)
    F, t_sensor, t_img = F[keep], t_sensor[keep], d["img"][keep]
    t_session = np.array([imgs[int(i)]["time"][:7] for i in t_img])

    m = pickle.load(open(xai / "types_model.pkl", "rb"))
    sc, pca, km = m["scaler"], m["pca"], m["kmeans"]
    Z = pca.transform(sc.transform(F))
    lab = km.labels_
    saved = np.array([int(r["type"]) for r in csv.DictReader(open(xai / "tile_scores.csv"))])
    assert (saved == lab).all(), "types differ from xai_tiles.py"
    names = {}
    for line in open(xai / "type_names.txt"):
        if line.strip() and not line.startswith("#"):
            k, n = line.strip().split(",", 1)
            names[int(k)] = n
    y = np.array([labels[s]["response"] for s in dev])
    share = np.array([[np.mean(lab[t_sensor == s] == k) for k in range(a.k)] for s in dev])
    par = np.stack([C.fab_features(s) for s in dev])
    res = {}

    # 4. mediation
    print("4. mediation (repeated sensor-held-out CV, ridge)")
    for name, X in (("recipe", par), ("types", share), ("recipe+types", np.hstack([par, share]))):
        rho = cv_score(X, y, "params", a.repeats)[0]
        res[f"cv_rho_{name}"] = float(rho.mean())
        print(f"   {name:13s} rho {rho.mean():+.3f} +/- {rho.std():.3f}")
    w_par = make_model("params", len(y)).fit(par, y).best_estimator_[-1].coef_
    w_both = make_model("params", len(y)).fit(np.hstack([par, share]), y).best_estimator_[-1].coef_[:4]
    res["recipe_weights_alone"], res["recipe_weights_with_types"] = w_par.tolist(), w_both.tolist()
    print("   recipe weights (log conc, log current, time, log charge), standardised:")
    print("     alone     ", np.round(w_par, 3), " |w| sum", round(float(np.abs(w_par).sum()), 3))
    print("     with types", np.round(w_both, 3), " |w| sum", round(float(np.abs(w_both).sum()), 3))

    # 5. stability
    print("5. stability")
    img_ids = sorted(set(t_img.tolist()))
    img_sensor = {i: t_sensor[t_img == i][0] for i in img_ids}
    stab = {}
    for k in range(a.k):
        per_img = {i: np.mean(lab[t_img == i] == k) for i in img_ids}
        groups = [np.array([per_img[i] for i in img_ids if img_sensor[i] == s]) for s in dev]
        groups = [g for g in groups if len(g) > 1]
        two = [s for s in dev if len(set(t_session[t_sensor == s])) > 1]
        sess = [np.array([np.mean(lab[(t_sensor == s) & (t_session == ss)] == k) for ss in sorted(set(t_session[t_sensor == s]))])
                for s in two]
        stab[k] = dict(icc_images=icc1(groups), icc_sessions=icc1(sess))
    for kk in (6, 10):
        lab2 = KMeans(kk, n_init=10, random_state=0).fit(Z).labels_
        for k in range(a.k):
            jac = max(np.sum((lab == k) & (lab2 == j)) / np.sum((lab == k) | (lab2 == j)) for j in range(kk))
            stab[k][f"jaccard_k{kk}"] = float(jac)
    for k in sorted(stab, key=lambda k: -np.mean(lab == k)):
        v = stab[k]
        print(f"   type {k} {names[k]:42s} ICC images {v['icc_images']:+.2f}  sessions {v['icc_sessions']:+.2f}  "
              f"Jaccard k6 {v['jaccard_k6']:.2f} k10 {v['jaccard_k10']:.2f}")
    res["stability"] = stab

    # 7. replicates
    print("7. replicates: type shares, replicate minus original (percentage points)")
    rf = Path(a.replicate_features)
    rimgs = {int(r["img"]): r for r in csv.DictReader(open(rf / "images.csv"))}
    rd = np.load(rf / "dinov2_vitb14_n448_pct.npz")
    rlab = km.predict(pca.transform(sc.transform(rd["feats"].astype(np.float32))))
    r_sensor = np.array([rimgs[int(i)]["sensor"] for i in rd["img"]])
    rep = {}
    for r in csv.DictReader(open(a.replicate_labels)):
        rep.setdefault(r["sensor"], []).append(float(r["rel_resp_median"]))
    order = sorted(range(a.k), key=lambda k: names[k])
    print("   " + " " * 18 + "  d log resp  " + "  ".join(f"t{k}" for k in order))
    res["replicates"] = {}
    for s in sorted(rep):
        if s in test or s not in dev:
            print(f"   {s:18s}  (original is a locked-test sensor; skipped)")
            continue
        so = np.array([np.mean(lab[t_sensor == s] == k) for k in range(a.k)])
        sr = np.array([np.mean(rlab[r_sensor == s] == k) for k in range(a.k)])
        dy = np.log10(np.mean(rep[s])) - labels[s]["response"]
        res["replicates"][s] = dict(d_log_resp=float(dy), d_share=(sr - so).tolist())
        print(f"   {s:18s}  {dy:+10.2f}  " + "  ".join(f"{100 * (sr[k] - so[k]):+3.0f}" for k in order))
    print("   types:", "; ".join(f"t{k} = {names[k]}" for k in order))
    json.dump(res, open(xai / "claim2.json", "w"), indent=1)


if __name__ == "__main__":
    main()
