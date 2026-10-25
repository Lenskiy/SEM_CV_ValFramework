"""E7b: backbone comparison for predicting log baseline resistance and log response (development sensors;
robustness only).

ViT-B (dinov2_vitb14_n448_pct, fixed before the locked test) stays the model of the paper whatever
this shows. Every frozen backbone and tiling/normalisation choice available, sensor mean, images
only, same pipeline, 20 repeats of sensor-held-out 5-fold CV: DINOv2 ViT-B and ViT-S, ResNet-50
(ImageNet), MicroNet ResNet-50 (micrographs; from scratch and after ImageNet), shallow ImageNet CNNs
(ResNet-18, and ResNet-18 / ResNet-50 truncated after layer2 or layer3), and the recipe.
Writes data/derived/r_air_backbones.csv.
"""

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # one thread per worker process
    os.environ[_v] = "1"

import csv
from multiprocessing import get_context
from pathlib import Path

import numpy as np

import common as C
import dev_eval
from dev_eval import cv_score, locked, read_labels, sensor_features

ROOT = Path(__file__).resolve().parents[2]
_make = dev_eval.make_model
JOBS = []


def serial(kind, n):
    return _make(kind, n).set_params(n_jobs=1)


def run(j):
    dev_eval.make_model = serial
    target, name, X, y, kind = JOBS[j]
    rho, r2, _, _ = cv_score(X, y, kind, 20)
    return dict(target=target, config=name, rho=round(float(rho.mean()), 3), rho_sd=round(float(rho.std()), 3), r2=round(float(r2.mean()), 3))


def main():
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    ra = {r["sensor"]: np.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    dev = [s for s in sorted(lab) if s not in test and s in ra]
    for target, y in (("r_air", np.array([ra[s] for s in dev])), ("response", np.array([lab[s]["response"] for s in dev]))):
        JOBS.append((target, "recipe (C, I, t, charge)", np.stack([C.fab_features(s) for s in dev]), y, "params"))
        for d in ("features500x", "features_micronet", "features_shallow"):
            for f in sorted((ROOT / "data" / d).glob("*.npz")):
                JOBS.append((target, f.stem, sensor_features(f, ROOT / "data" / d / "images.csv", dev, "mean"), y, "img"))
    with get_context("fork").Pool(8) as pool:
        rows = pool.map(run, range(len(JOBS)))
    rows.sort(key=lambda r: (r["target"], -r["rho"]))
    with open(ROOT / "data/derived/r_air_backbones.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"{r['target']:8s} {r['config']:34s} rho {r['rho']:+.3f} +/- {r['rho_sd']:.3f}  R2 {r['r2']:+.3f}")


if __name__ == "__main__":
    main()
