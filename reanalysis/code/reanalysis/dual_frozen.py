"""Two-head loss on frozen ViT-B tile features (CPU): does the original paper's CE + lambda x MSE help?

Features: dinov2_vitb14_n448_pct tiles (1,536-d), standardised with training-fold statistics.
Heads: linear regression head and linear 4-class head (ordinal classes = quartiles of the training
sensors' labels), trained on tiles with AdamW (lr 1e-3, weight decay 1e-2, 200 full-batch steps).
Variants: regression only (MSE), CE + 0.001 x MSE (as in the original paper), CE + 1 x MSE.
Split: the 5 folds of repeat 0 of RepeatedKFold(5, random_state=0) over the 56 development sensors
(as train_small.py); 3 seeds. Sensor scores: mean regression output and mean expected class over
its tiles; pooled held-out Spearman rho. Targets: log10 response and log10 baseline resistance.
Note: with frozen features the two heads share no trainable parameters, so the loss separates
into two independent models and lambda has no effect (AdamW is insensitive to the loss scale); the
shared-trunk version of the idea is tested with the small CNNs (train_small.py --dual).
Writes data/derived/dual_frozen.csv.
"""
import csv
import math
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold

from dev_eval import locked, read_labels

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features500x"


def run(F, tile_s, dev, y_s, lam, seed):
    torch.manual_seed(seed)
    reg_out, cls_out = {}, {}
    for tr, te in RepeatedKFold(n_splits=5, n_repeats=1, random_state=0).split(np.zeros(len(dev))):
        train_s = {dev[i] for i in tr}
        m = np.isin(tile_s, list(train_s))
        mu_x, sd_x = F[m].mean(0), F[m].std(0) + 1e-6
        ytr = np.array([y_s[s] for s in sorted(train_s)])
        mu, sd = ytr.mean(), ytr.std()
        edges = np.quantile(ytr, [0.25, 0.5, 0.75])
        X = torch.tensor((F[m] - mu_x) / sd_x, dtype=torch.float32)
        y = torch.tensor([(y_s[s] - mu) / sd for s in tile_s[m]], dtype=torch.float32)
        c = torch.tensor([int(np.digitize(y_s[s], edges)) for s in tile_s[m]])
        reg, cls = torch.nn.Linear(F.shape[1], 1), torch.nn.Linear(F.shape[1], 4)
        opt = torch.optim.AdamW(list(reg.parameters()) + list(cls.parameters()), lr=1e-3, weight_decay=1e-2)
        for _ in range(200):
            mse = torch.nn.functional.mse_loss(reg(X).squeeze(1), y)
            loss = mse if lam is None else torch.nn.functional.cross_entropy(cls(X), c) + lam * mse
            opt.zero_grad()
            loss.backward()
            opt.step()
        with torch.no_grad():
            for i in te:
                s = dev[i]
                Xs = torch.tensor((F[tile_s == s] - mu_x) / sd_x, dtype=torch.float32)
                reg_out[s] = float(reg(Xs).mean()) * sd + mu
                cls_out[s] = float((torch.softmax(cls(Xs), 1) @ torch.arange(4.0)).mean())
    yv = [y_s[s] for s in dev]
    return spearmanr([reg_out[s] for s in dev], yv).statistic, spearmanr([cls_out[s] for s in dev], yv).statistic


def main():
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    d = np.load(FEAT / "dinov2_vitb14_n448_pct.npz")
    img_s = {int(r["img"]): r["sensor"] for r in csv.DictReader(open(FEAT / "images.csv"))}
    tile_s = np.array([img_s[int(i)] for i in d["img"]])
    dev = sorted(s for s in lab if s not in test and s in set(tile_s))
    keep = np.isin(tile_s, dev)
    F, tile_s = d["feats"][keep].astype(np.float32), tile_s[keep]
    ra = {r["sensor"]: math.log10(float(r["r_air_median"])) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    rows = []
    for target, y_s in (("response", {s: lab[s]["response"] for s in dev}), ("r_air", {s: ra[s] for s in dev})):
        for name, lam in (("regression only", None), ("CE + 0.001 MSE", 0.001), ("CE + 1 MSE", 1.0)):
            res = np.array([run(F, tile_s, dev, y_s, lam, seed) for seed in range(3)])
            row = dict(target=target, loss=name, rho_regression=round(float(res[:, 0].mean()), 3),
                       rho_regression_sd=round(float(res[:, 0].std()), 3),
                       rho_expected_class=round(float(res[:, 1].mean()), 3) if lam is not None else "",
                       rho_expected_class_sd=round(float(res[:, 1].std()), 3) if lam is not None else "")
            rows.append(row)
            print(row, flush=True)
    with open(ROOT / "data/derived/dual_frozen.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
