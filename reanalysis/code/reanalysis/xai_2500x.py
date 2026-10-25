"""What the 2,500x edge-density signal looks like (12 sensors at 250 uA).

1. Edge density per 448-px tile (the hm_morphology definition applied to tiles),
   for every 2,500x image; within-sensor vs between-sensor spread (ICC).
2. Montage A: sensors ordered by measured response (rows), 6 tiles each at 2,500x.
3. Montage B: one 20,000x crop per sensor, same order, to see the structure that
   the 2,500x edge density summarises.
4. Sensor level: median tile edge density and the share of tiles above the pooled
   median, against log response (Spearman).
Writes to --out: tiles_2500x.csv, montage_2500x.png, montage_20000x.png.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.stats import spearmanr

import common as C

TILE = 448


def tile_edge(f):
    gm = np.hypot(ndimage.sobel(f, 0), ndimage.sobel(f, 1))
    return float(gm.mean() / max(f.std(), 1e-6))


def stretch(a):
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1) * 255, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hm", required=True, help="folder with 2500x/ and 20000x/")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    lab = {r["sensor"]: r for r in csv.DictReader(open(a.labels))}
    resp = {s: np.log10(float(r["rel_resp_median"])) for s, r in lab.items()}

    rows, arrays = [], {}
    for im in C.index_sem(Path(a.hm) / "2500x"):
        f = ndimage.gaussian_filter(C.load_image_array(im.path).astype(np.float32), 1)
        arrays.setdefault(im.sensor, []).append(f)
        for r in range(f.shape[0] // TILE):
            for c in range(f.shape[1] // TILE):
                rows.append(dict(sensor=im.sensor, file=im.path.name, row=r, col=c,
                                 edge=tile_edge(f[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE])))
    with open(out / "tiles_2500x.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sensors = sorted(arrays, key=lambda s: resp[s])
    e = {s: np.array([r["edge"] for r in rows if r["sensor"] == s]) for s in sensors}
    pooled_med = np.median(np.concatenate(list(e.values())))
    within = np.mean([v.var(ddof=1) for v in e.values()])
    between = np.var([v.mean() for v in e.values()], ddof=1)
    n = np.mean([len(v) for v in e.values()])
    icc = (between - within / n) / (between - within / n + within)
    print(f"tile edge density: ICC(1) of single tiles = {icc:.2f} (share of tile variance that is between sensors)")
    med = [np.median(e[s]) for s in sensors]
    hi = [np.mean(e[s] > pooled_med) for s in sensors]
    y = [resp[s] for s in sensors]
    print(f"sensor median tile edge density vs log response: rho {spearmanr(med, y).statistic:+.2f} "
          f"(p {spearmanr(med, y).pvalue:.3f}); share of high-edge tiles: rho {spearmanr(hi, y).statistic:+.2f} "
          f"(p {spearmanr(hi, y).pvalue:.3f})")
    for s, m_, h_ in zip(sensors, med, hi):
        print(f"  {s:18s} log resp {resp[s]:+.2f}  median tile edge {m_:.3f}  high-edge share {h_:.2f}")

    th, n_ex = 170, 6
    rng = np.random.default_rng(0)
    canvas = Image.new("L", (n_ex * (th + 4) + 4, len(sensors) * (th + 4) + 4), 255)
    for ri, s in enumerate(sensors):
        f = arrays[s][0]
        cells = [(r, c) for r in range(f.shape[0] // TILE) for c in range(f.shape[1] // TILE)]
        for ci, k in enumerate(rng.choice(len(cells), n_ex, replace=False)):
            r, c = cells[k]
            canvas.paste(Image.fromarray(stretch(f[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE])).resize((th, th)),
                         (4 + ci * (th + 4), 4 + ri * (th + 4)))
    canvas.save(out / "montage_2500x.png")

    hm20 = {im.sensor: im.path for im in C.index_sem(Path(a.hm) / "20000x")}
    w20, h20 = 360, 240
    canvas = Image.new("L", (3 * (w20 + 4) + 4, 4 * (h20 + 4) + 4), 255)
    for i, s in enumerate(sensors):
        if s not in hm20:
            continue
        f = C.load_image_array(hm20[s]).astype(np.float32)
        crop = f[600:1800, 960:2760]                       # central 1800 x 1200 px = 12 x 8 um
        canvas.paste(Image.fromarray(stretch(crop)).resize((w20, h20)), (4 + (i % 3) * (w20 + 4), 4 + (i // 3) * (h20 + 4)))
    canvas.save(out / "montage_20000x.png")
    print("montage order (low to high response):", sensors)


if __name__ == "__main__":
    main()
