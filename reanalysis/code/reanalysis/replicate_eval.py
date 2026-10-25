"""Does the pre-registered 500x model see the replicate-vs-original differences?

Model: ViT-B n448 pct sensor mean -> PCA + ridge (dev_eval 'img'), fitted on the
development sensors minus the original of the recipe being checked, so neither
sensor of the pair is in training. For the locked-test recipe the model is fitted
on all 56 and only the replicate is predicted (the locked original is not scored).
Observed labels: labels_v2 (original) and labels_replicate (mean over tests).
"""
import csv
from pathlib import Path

import numpy as np

from dev_eval import locked, make_model, read_labels, sensor_features

ROOT = Path(__file__).resolve().parents[2]
CFG = "dinov2_vitb14_n448_pct.npz"


def main():
    f500, frep = ROOT / "data/features500x", ROOT / "data/features_replicate"
    labels = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in labels if s not in test)
    rep = {}
    for r in csv.DictReader(open(ROOT / "data/derived/labels_replicate.csv")):
        rep.setdefault(r["sensor"], []).append(float(r["rel_resp_median"]))
    Xdev = sensor_features(f500 / CFG, f500 / "images.csv", dev, "mean")
    ydev = np.array([labels[s]["response"] for s in dev])
    print(f"{'recipe':18s} {'obs orig':>9s} {'obs rep':>8s} {'pred orig':>9s} {'pred rep':>9s}  direction")
    for s in sorted(rep):
        keep = np.array([d != s for d in dev])
        m = make_model("img", keep.sum()).fit(Xdev[keep], ydev[keep])
        pr = m.predict(sensor_features(frep / CFG, frep / "images.csv", [s], "mean"))[0]
        yr = np.log10(np.mean(rep[s]))
        if s in test:
            print(f"{s:18s} {'(locked)':>9s} {yr:8.2f} {'':>9s} {pr:9.2f}")
            continue
        po = m.predict(Xdev[~keep])[0]
        yo = labels[s]["response"]
        ok = np.sign(pr - po) == np.sign(yr - yo)
        print(f"{s:18s} {yo:9.2f} {yr:8.2f} {po:9.2f} {pr:9.2f}  {'agrees' if ok else 'disagrees'}"
              f" (obs {yr - yo:+.2f}, pred {pr - po:+.2f} log10)")


if __name__ == "__main__":
    main()
