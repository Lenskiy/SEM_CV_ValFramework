"""E6: how many 500x images per sensor does the frozen model need?
For k = 1..4, 20 draws: keep k random images per development sensor (all if it has
fewer), average their tile features, and score one repeat of 5-fold CV (seed = draw).
Writes data/derived/e6_images.json."""
import csv
import json
from pathlib import Path

import numpy as np

from dev_eval import cv_score, locked, read_labels

ROOT = Path(__file__).resolve().parents[2]
feat = ROOT / "data/features500x"
lab = read_labels(ROOT / "data/derived/labels_v2.csv")
test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
imgs = {int(r["img"]): r["sensor"] for r in csv.DictReader(open(feat / "images.csv"))}
d = np.load(feat / "dinov2_vitb14_n448_pct.npz")
F, t_img = d["feats"].astype(np.float32), d["img"]
dev = sorted(s for s in lab if s not in test)
y = np.array([lab[s]["response"] for s in dev])
per = {s: sorted({i for i, ss in imgs.items() if ss == s}) for s in dev}
print("images per sensor:", {n: sum(len(v) == n for v in per.values()) for n in sorted({len(v) for v in per.values()})})
rng = np.random.default_rng(0)
out = {}
for k in (1, 2, 3, 4):
    rhos = []
    for draw in range(20):
        X = np.stack([F[np.isin(t_img, rng.choice(per[s], min(k, len(per[s])), replace=False))].mean(0) for s in dev])
        rhos.append(float(cv_score(X, y, "img", 1, seed=draw)[0][0]))
    out[k] = dict(mean=round(float(np.mean(rhos)), 3), sd=round(float(np.std(rhos)), 3),
                  q10=round(float(np.percentile(rhos, 10)), 3), q90=round(float(np.percentile(rhos, 90)), 3))
    print(f"k={k} images: rho {np.mean(rhos):+.3f} +/- {np.std(rhos):.3f} (10-90%: {out[k]['q10']:+.2f} to {out[k]['q90']:+.2f})", flush=True)
json.dump(out, open(ROOT / "data/derived/e6_images.json", "w"), indent=1)
