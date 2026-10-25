"""Morphology measures from the 2,500x / 10,000x / 20,000x SEMs (12 sensors at 250 uA).

Per image (data bar cropped, Gaussian sigma 1 px):
  corr_len_nm   radially averaged autocorrelation length (first lag below 1/e), in nm
  dark_frac     area fraction below the Otsu threshold (voids/pores are dark in SE images)
  edge_density  mean Sobel gradient magnitude / grey-level s.d. (boundary density, scale-free)
  contrast      grey-level s.d. / mean
Per sensor: median over its images, per magnification. These are then compared
(Spearman, n = 12, and n = 11 without the SNR-156 outlier) with the measured log
relative response and with the out-of-fold predictions of the pre-registered
500x model (ViT-B n448 pct, sensor mean, PCA + ridge), so we can see whether the
500x model tracks structure that is resolved at high magnification.

Writes data/derived/hm_morphology.csv (per sensor x magnification) and
data/dev_eval_500x/oof_preregistered.csv; prints the correlation table.
"""
import csv
from pathlib import Path

import numpy as np
from scipy import ndimage
from scipy.stats import spearmanr

import common as C
from dev_eval import cv_score, locked, read_labels, sensor_features

ROOT = Path(__file__).resolve().parents[2]
MEASURES = ("corr_len_nm", "dark_frac", "edge_density", "contrast")


def otsu(a):
    hist = np.bincount(a.ravel(), minlength=256).astype(float)
    p = hist / hist.sum()
    w = np.cumsum(p)
    mu = np.cumsum(p * np.arange(256))
    between = (mu[-1] * w - mu) ** 2 / np.clip(w * (1 - w), 1e-12, None)
    return int(np.argmax(between))


def corr_length(f):
    g = f - f.mean()
    F = np.fft.rfft2(g)
    ac = np.fft.irfft2(np.abs(F) ** 2, s=g.shape)
    ac = np.fft.fftshift(ac) / ac.flat[0]
    cy, cx = np.array(ac.shape) // 2
    y, x = np.indices(ac.shape)
    r = np.hypot(y - cy, x - cx).astype(int)
    prof = np.bincount(r.ravel(), ac.ravel()) / np.bincount(r.ravel())
    below = np.flatnonzero(prof < 1 / np.e)
    return float(below[0]) if len(below) else np.nan


def measures(path, px_nm):
    a = C.load_image_array(path)
    f = ndimage.gaussian_filter(a.astype(np.float32), 1)
    t = otsu(np.clip(f, 0, 255).astype(np.uint8))
    gm = np.hypot(ndimage.sobel(f, 0), ndimage.sobel(f, 1))
    return dict(corr_len_nm=corr_length(f) * px_nm, dark_frac=float((f < t).mean()),
                edge_density=float(gm.mean() / f.std()), contrast=float(f.std() / f.mean()))


def oof_predictions(repeats=5):
    """Out-of-fold predictions of the pre-registered config for all dev sensors."""
    feat = ROOT / "data/features500x"
    labels = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = locked(ROOT / "data/derived/locked_test_sensors.txt")
    have = {r["sensor"] for r in csv.DictReader(open(feat / "images.csv"))}
    dev = sorted(s for s in labels if s not in test and s in have)
    X = sensor_features(feat / "dinov2_vitb14_n448_pct.npz", feat / "images.csv", dev, "mean")
    y = np.array([labels[s]["response"] for s in dev])
    rho, _, _, pred = cv_score(X, y, "img", repeats)
    print(f"pre-registered config, out-of-fold rho {rho.mean():+.3f} over {repeats} repeats")
    out = ROOT / "data/dev_eval_500x/oof_preregistered.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "y_log_rel_resp", "oof_pred"])
        w.writerows(zip(dev, y, pred))
    return dict(zip(dev, pred))


def main():
    rows = []
    for mag in ("2500x", "10000x", "20000x"):
        for im in C.index_sem(ROOT / "data/raw_hm" / mag):
            m = measures(im.path, float(im.meta["pixel_width_nm"]))
            rows.append(dict(sensor=im.sensor, mag=mag, file=im.path.name, **m))
    per = {}
    for r in rows:
        per.setdefault((r["sensor"], r["mag"]), []).append(r)
    table = [dict(sensor=s, mag=mag, n_images=len(v), **{k: float(np.median([x[k] for x in v])) for k in MEASURES})
             for (s, mag), v in sorted(per.items())]
    with open(ROOT / "data/derived/hm_morphology.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)

    lab = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/derived/labels_v2.csv"))}
    oof = oof_predictions()
    outlier = next(s for s, r in lab.items() if r["excluded_as_outlier"] == "True")
    print(f"\nSpearman rho (p) vs measured log rel_resp and vs 500x out-of-fold prediction; "
          f"outlier {outlier} included (n=12) / excluded (n=11)")
    print(f"{'mag':7s} {'measure':13s} {'resp n=12':>16s} {'resp n=11':>16s} {'500x pred n=11':>16s}")
    for mag in ("2500x", "10000x", "20000x"):
        t = [r for r in table if r["mag"] == mag]
        for k in MEASURES:
            x = np.array([r[k] for r in t])
            y = np.log10([float(lab[r["sensor"]]["rel_resp_median"]) for r in t])
            keep = np.array([r["sensor"] != outlier for r in t])
            p = np.array([oof.get(r["sensor"], np.nan) for r in t])
            a, b, c = spearmanr(x, y), spearmanr(x[keep], y[keep]), spearmanr(x[keep], p[keep])
            print(f"{mag:7s} {k:13s} {a.statistic:+.2f} ({a.pvalue:.3f})    {b.statistic:+.2f} ({b.pvalue:.3f})    "
                  f"{c.statistic:+.2f} ({c.pvalue:.3f})")
    keep = [s for s in sorted({r["sensor"] for r in table}) if s != outlier]
    r = spearmanr([oof[s] for s in keep], [np.log10(float(lab[s]["rel_resp_median"])) for s in keep])
    print(f"\n500x out-of-fold prediction vs measured, these 11 sensors: rho {r.statistic:+.2f} (p {r.pvalue:.3f})")


if __name__ == "__main__":
    main()
