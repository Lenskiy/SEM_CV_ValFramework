"""t-SNE of the frozen ViT-B tile features (500x, development sensors), coloured by
sensor response, morphology type and imaging session; plus k-nearest-neighbour
purity: how often a tile's 10 nearest neighbours (in the 50-d PCA space) share its
sensor, its imaging month, its morphology type, or its response tercile.
--space full (default): neighbours and t-SNE on the full 1,536-d standardised tile features;
--space pca50: on the 50-d PCA of the morphology-type model (the earlier version).
--metric euclidean (default) or cosine, for both the neighbour search and t-SNE.
Writes data/xai_500x/tsne[_pca50].png, tsne_coords[_pca50].csv and tsne_purity[_pca50].json."""
import argparse
import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from dev_eval import locked, read_labels

ap = argparse.ArgumentParser()
ap.add_argument("--space", choices=["full", "pca50"], default="full")
ap.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
_a = ap.parse_args()
SPACE, METRIC = _a.space, _a.metric
SUF = ("" if SPACE == "full" else "_pca50") + ("" if METRIC == "euclidean" else "_cosine")
ROOT = Path(__file__).resolve().parents[2]
feat = ROOT / "data/features500x"
labels = read_labels(ROOT / "data/derived/labels_v2.csv")
test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
imgs = {int(r["img"]): r for r in csv.DictReader(open(feat / "images.csv"))}
d = np.load(feat / "dinov2_vitb14_n448_pct.npz")
F = d["feats"].astype(np.float32)
sen = np.array([imgs[int(i)]["sensor"] for i in d["img"]])
dev = sorted(s for s in labels if s not in test and s in set(sen))
keep = np.isin(sen, dev)
F, sen, img = F[keep], sen[keep], d["img"][keep]
month = np.array([imgs[int(i)]["time"][:7] for i in img])
m = pickle.load(open(ROOT / "data/xai_500x/types_model.pkl", "rb"))
Z = StandardScaler().fit_transform(F) if SPACE == "full" else m["pca"].transform(m["scaler"].transform(F))
print(f"space {SPACE}: {Z.shape[1]} dimensions, {len(Z)} tiles", flush=True)
typ = m["kmeans"].labels_
y = np.array([labels[s]["response"] for s in sen])
terc = np.digitize(y, np.quantile([labels[s]["response"] for s in dev], [1 / 3, 2 / 3]))

nn = NearestNeighbors(n_neighbors=200, metric=METRIC).fit(Z)
idx_all = nn.kneighbors(Z, return_distance=False)
idx = idx_all[:, 1:11]
rng = np.random.default_rng(0)
purity = {}
for name, v in (("sensor", sen), ("imaging month", month), ("morphology type", typ), ("response tercile", terc)):
    pur = (v[idx] == v[:, None]).mean()
    base = np.mean([(v[rng.permutation(len(v))[:10]] == v[i]).mean() for i in rng.choice(len(v), 2000)])
    purity[name] = dict(observed=float(pur), chance=float(base))
    print(f"10-NN share with same {name:16s}: {pur:.2f}  (chance {base:.2f})")
pur, base = [], []
for i in range(len(Z)):                 # neighbours from OTHER sensors only
    o = [j for j in idx_all[i, 1:] if sen[j] != sen[i]][:10]
    if len(o) == 10:
        pur.append(np.mean(terc[o] == terc[i]))
        base.append(np.mean(terc[rng.choice(np.flatnonzero(sen != sen[i]), 10)] == terc[i]))
purity["response tercile, other sensors only"] = dict(observed=float(np.mean(pur)), chance=float(np.mean(base)))
print(f"10-NN from other sensors, same response tercile: {np.mean(pur):.2f} (chance {np.mean(base):.2f})")
purity["space"] = f"{SPACE} ({Z.shape[1]} dimensions), {METRIC} distance"
json.dump(purity, open(ROOT / f"data/xai_500x/tsne_purity{SUF}.json", "w"), indent=1)

T = TSNE(2, perplexity=40, init="pca", metric=METRIC, random_state=0).fit_transform(Z)
with open(ROOT / f"data/xai_500x/tsne_coords{SUF}.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["sensor", "img", "x", "y", "type", "month", "log_response", "tercile"])
    w.writerows(zip(sen, img, T[:, 0].round(3), T[:, 1].round(3), typ, month, y.round(3), terc))
fig, ax = plt.subplots(1, 3, figsize=(10, 3.4))
s0 = ax[0].scatter(T[:, 0], T[:, 1], c=y, s=1, cmap="Blues", vmin=-4.7, vmax=0)
ax[0].set_title("sensor log10 response")
plt.colorbar(s0, ax=ax[0], fraction=0.04)
ax[1].scatter(T[:, 0], T[:, 1], c=typ, s=1, cmap="tab10", vmin=0, vmax=9)
ax[1].set_title("morphology type (k = 8)")
mm = sorted(set(month))
ax[2].scatter(T[:, 0], T[:, 1], c=[mm.index(x) for x in month], s=1, cmap="tab10", vmin=0, vmax=9)
ax[2].set_title("imaging month: " + ", ".join(mm))
for a in ax:
    a.set_xticks([])
    a.set_yticks([])
fig.savefig(ROOT / f"data/xai_500x/tsne{SUF}.png", dpi=150, bbox_inches="tight")
