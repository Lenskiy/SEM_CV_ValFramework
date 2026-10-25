"""The hm_morphology measures computed on the 500x images of every sensor.

Each measure is a single fixed number per sensor (median over its images), so it
is scored on the 56 development sensors by plain Spearman correlation with log
relative response, without fitting; the locked test sensors are skipped.
Replicate sensors (March 2026) are measured too, for the replicate comparison.
Writes data/derived/morph_500x.csv. Usage: python morph_500x.py <500x dir> <replicate 500x dir>
"""
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

import common as C
from dev_eval import locked, read_labels
from hm_morphology import MEASURES, ROOT, measures


def per_sensor(root, tag):
    per = {}
    for im in C.index_sem(root):
        per.setdefault(im.sensor, []).append(measures(im.path, float(im.meta["pixel_width_nm"])))
    return [dict(sensor=s, set=tag, n_images=len(v), **{k: float(np.median([x[k] for x in v])) for k in MEASURES})
            for s, v in sorted(per.items())]


def main(sem_dir, rep_dir):
    table = per_sensor(sem_dir, "original") + per_sensor(rep_dir, "replicate")
    with open(ROOT / "data/derived/morph_500x.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader()
        w.writerows(table)
    labels = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = [r for r in table if r["set"] == "original" and r["sensor"] in labels and r["sensor"] not in test]
    y = np.array([labels[r["sensor"]]["response"] for r in dev])
    print(f"500x measures vs log rel_resp, {len(dev)} development sensors")
    for k in MEASURES:
        r = spearmanr([d[k] for d in dev], y)
        print(f"  {k:13s} rho {r.statistic:+.3f}  p {r.pvalue:.4f}")
    at250 = [d for d in dev if C.parse_sensor_key(d["sensor"])[1] == 250]
    r = spearmanr([d["edge_density"] for d in at250], [labels[d["sensor"]]["response"] for d in at250])
    print(f"  edge_density, the {len(at250)} dev sensors at 250 uA: rho {r.statistic:+.3f}  p {r.pvalue:.4f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
