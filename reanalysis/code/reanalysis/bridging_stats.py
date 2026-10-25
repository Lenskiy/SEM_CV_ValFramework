"""E10 tests: do the physical gap descriptors (bridging.py) explain baseline resistance, and the
network's resistance signal?

All sensors (no learning): Spearman of gap_loss and gap_contrast with log R_air, log response
and the metallic-Zn thickness h (regime.py); 10,000-permutation p, Bonferroni over the 2 descriptors.
Development sensors (the network is involved):
  - gap_loss vs the overgrowth type share of the exact explanation (xai_500x types 2, 7, 3)
  - gap_loss vs the out-of-fold ViT-B prediction of R_air (dev_eval, 20 repeats)
  - recipe + gap descriptors -> R_air (CV); does the ViT prediction still track the residual?
Writes data/derived/bridging_stats.json.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from dev_eval import cv_score, locked, read_labels, sensor_features

ROOT = Path(__file__).resolve().parents[2]
OVERGROWTH = (2, 7, 3)
rng = np.random.default_rng(0)


def perm(x, y, n=10000, k=2):
    r = spearmanr(x, y).statistic
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n)])
    p = (1 + (np.abs(null) >= abs(r)).sum()) / (n + 1)
    return dict(rho=round(float(r), 3), p=round(float(p), 4), p_bonf=round(float(min(1, p * k)), 4), n=len(x))


def main():
    br = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/derived/bridging.csv"))}
    reg = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/derived/regime_sensors.csv"))}
    S = [s for s in sorted(reg) if s in br]
    D = {k: np.array([float(br[s][k]) for s in S]) for k in ("gap_loss", "gap_contrast")}
    lr = np.log10([float(reg[s]["r_air"]) for s in S])
    ly = np.log10([max(float(reg[s]["resp"]), 1e-5) for s in S])
    h = np.array([float(reg[s]["h_0.048"]) for s in S])
    out = {"all": {k: dict(log_r_air=perm(v, lr), log_response=perm(v, ly), h=perm(v, h)) for k, v in D.items()}}

    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = [s for s in S if s in lab and s not in test]
    i = [S.index(s) for s in dev]
    st = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/xai_500x/sensor_types.csv"))}
    og = np.array([sum(float(st[s][f"type{k}"]) for k in OVERGROWTH) for s in dev])
    gl, gc, ld = D["gap_loss"][i], D["gap_contrast"][i], lr[i]
    X = sensor_features(ROOT / "data/features500x/dinov2_vitb14_n448_pct.npz", ROOT / "data/features500x/images.csv", dev, "mean")
    _, _, _, p_img = cv_score(X, ld, "img", 20)
    R = np.log10([[float(reg[s][k]) for k in ("C", "I", "t", "h_0.048")] for s in dev])
    rho_r, _, _, p_rec = cv_score(R, ld, "params", 20)
    rho_rg, r2_rg, _, p_recg = cv_score(np.column_stack([R, gl, gc]), ld, "params", 20)
    rho_g, r2_g, _, _ = cv_score(np.column_stack([gl, gc]), ld, "params", 20)
    out["dev"] = dict(
        n=len(dev),
        gap_loss_vs_overgrowth_share=perm(gl, og, k=1),
        gap_loss_vs_vit_pred_r_air=perm(gl, p_img, k=1),
        cv_gap_only_r_air=dict(rho=round(float(rho_g.mean()), 3), r2=round(float(r2_g.mean()), 3)),
        cv_recipe_r_air=dict(rho=round(float(rho_r.mean()), 3)),
        cv_recipe_gap_r_air=dict(rho=round(float(rho_rg.mean()), 3), r2=round(float(r2_rg.mean()), 3)),
        gap_loss_vs_r_air_beyond_recipe=perm(gl, ld - p_rec, k=1),
        vit_vs_r_air_beyond_recipe_and_gap=perm(p_img, ld - p_recg, k=1),
    )
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / "data/derived/bridging_stats.json", "w"), indent=1)


if __name__ == "__main__":
    main()
