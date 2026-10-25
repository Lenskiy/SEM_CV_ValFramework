"""E2: hand-crafted texture features at 500x through the frozen model's pipeline.

Per native 448 px tile (same grid and 1-99 percentile stretch as the ViT-B features):
  grey-level co-occurrence (64 levels): contrast, homogeneity, energy, correlation at
    distances 1, 2, 4, 8 px, averaged over 4 angles             -> 16
  local binary patterns (uniform, P = 8; radii 1, 2, 3): histograms -> 30
  edge density, dark fraction (Otsu), contrast (hm_morphology definitions) -> 3
Sensor = mean over its tiles; standardise -> PCA -> ridge (dev_eval 'img'), 20 repeats
of 5-fold CV on the development sensors, and a 1,000-permutation null.
Usage: python texture_baseline.py <500x image dir>; writes data/derived/texture_*.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.filters import threshold_otsu

import common as C
from dev_eval import cv_score, locked, read_labels

ROOT = Path(__file__).resolve().parents[2]
TILE = 448


def tile_features(t):
    q = np.clip((t * 64).astype(int), 0, 63).astype(np.uint8)
    g = graycomatrix(q, [1, 2, 4, 8], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=64, symmetric=True, normed=True)
    glcm = np.concatenate([graycoprops(g, p).mean(1) for p in ("contrast", "homogeneity", "energy", "correlation")])
    lbp = np.concatenate([np.bincount(local_binary_pattern(q, 8, r, "uniform").astype(int).ravel(), minlength=10)[:10]
                          / q.size for r in (1, 2, 3)])
    f = ndimage.gaussian_filter(t, 1)
    gm = np.hypot(ndimage.sobel(f, 0), ndimage.sobel(f, 1))
    morph = [gm.mean() / max(f.std(), 1e-6), (f < threshold_otsu(f)).mean(), f.std() / max(f.mean(), 1e-6)]
    return np.concatenate([glcm, lbp, morph])


def main(sem):
    feats, sensors = [], []
    for im in C.index_sem(sem):
        a = C.load_image_array(im.path).astype(np.float32)
        lo, hi = np.percentile(a, [1, 99])
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
        for r in range(a.shape[0] // TILE):
            for c in range(a.shape[1] // TILE):
                feats.append(tile_features(a[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]))
                sensors.append(im.sensor)
    F, sensors = np.array(feats), np.array(sensors)
    np.savez(ROOT / "data/derived/texture_tiles.npz", feats=F, sensor=sensors)
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in lab if s not in test)
    X = np.stack([F[sensors == s].mean(0) for s in dev])
    y = np.array([lab[s]["response"] for s in dev])
    rho = cv_score(X, y, "img", 20)[0]
    rng = np.random.default_rng(0)
    null = np.array([cv_score(X, rng.permutation(y), "img", 1, seed=k)[0][0] for k in range(1000)])
    p = (1 + (null >= rho.mean()).sum()) / 1001
    res = dict(n_tiles=len(F), n_features=F.shape[1], rho=round(float(rho.mean()), 3), rho_sd=round(float(rho.std()), 3),
               null_mean=round(float(null.mean()), 3), null_sd=round(float(null.std()), 3), perm_p=round(float(p), 4))
    json.dump(res, open(ROOT / "data/derived/texture_baseline.json", "w"), indent=1)
    print("E2 texture baseline:", res)


if __name__ == "__main__":
    main(sys.argv[1])
