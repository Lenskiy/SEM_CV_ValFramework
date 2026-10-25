"""Small CNNs trained from scratch on 500x tiles (GPU), as comparators for the frozen ViT-B.

Models (single-channel input, regression head on the global-average-pooled features):
  small_resnet  the original paper's backbone: stem conv 3x3/32 stride 2 + max-pool, then three
                residual blocks of 64, 128, 256 filters (two 3x3 conv + BN + ReLU each, stride 2,
                1x1 projection shortcut)
  simple_cnn    four conv 3x3 + BN + ReLU + 2x2 max-pool layers (16, 32, 64, 128 filters), dropout 0.3
Data and split exactly as finetune.py: native 448 px tiles, per-image 1-99 percentile stretch,
fold k of repeat 0 of RepeatedKFold(5, random_state=0) over the 56 development sensors; every
tile carries its sensor's label; sensor prediction = mean over its tiles. Targets: log10 response
or log10 baseline resistance. Fixed settings, chosen before any result: AdamW (lr 1e-3, weight
decay 1e-4), 30 epochs, one-cycle schedule, batch 32, rotation/flip augmentation, no early
stopping; the held-out fold is scored once at the end. Locked sensors are never loaded.
--dual LAMBDA adds the original paper's two-head design: a classification head over 4 ordinal
classes (quartiles of the TRAINING sensors' labels, so no held-out label is used) next to the
regression head, trained with loss = CE + LAMBDA x MSE (the original used LAMBDA = 0.001). Scores
written per sensor: the regression output, the expected class sum_k k p_k (mean over tiles) and the
predicted class (argmax of the mean class probabilities).
Writes <out>/<model>_<target>[_dual<LAMBDA>]_s<seed>/fold<k>/preds.csv and log.json.
"""
import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold

import common as C
from dev_eval import locked, read_labels

TILE = 448


def conv_bn(i, o, s=1):
    return nn.Sequential(nn.Conv2d(i, o, 3, s, 1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Block(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c1 = conv_bn(i, o, 2)
        self.c2 = nn.Sequential(nn.Conv2d(o, o, 3, 1, 1, bias=False), nn.BatchNorm2d(o))
        self.sc = nn.Sequential(nn.Conv2d(i, o, 1, 2, bias=False), nn.BatchNorm2d(o))

    def forward(self, x):
        return torch.relu(self.c2(self.c1(x)) + self.sc(x))


class TwoHead(nn.Module):
    def __init__(self, trunk, dim, drop, n_cls):
        super().__init__()
        self.trunk = nn.Sequential(trunk, nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop))
        self.reg = nn.Linear(dim, 1)
        self.cls = nn.Linear(dim, n_cls) if n_cls else None

    def forward(self, x):
        h = self.trunk(x)
        return self.reg(h).squeeze(1), (self.cls(h) if self.cls is not None else None)


def build(name, n_cls=0):
    if name == "small_resnet":
        feat = nn.Sequential(conv_bn(1, 32, 2), nn.MaxPool2d(2), Block(32, 64), Block(64, 128), Block(128, 256))
        dim, drop = 256, 0.0
    else:
        layers, i = [], 1
        for o in (16, 32, 64, 128):
            layers += [conv_bn(i, o), nn.MaxPool2d(2)]
            i = o
        feat, dim, drop = nn.Sequential(*layers), 128, 0.3
    return TwoHead(feat, dim, drop, n_cls)


def stretch(a):
    f = a.astype(np.float32)
    lo, hi = np.percentile(f, [1, 99])
    return (np.clip((f - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def batch_tensor(arrays, idx, device, augment, rng):
    ts = []
    for i, r, c in idx:
        t = arrays[i][r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]
        if augment:
            t = np.rot90(t, rng.integers(4))
            if rng.random() < 0.5:
                t = t[:, ::-1]
        ts.append(np.ascontiguousarray(t))
    x = torch.from_numpy(np.stack(ts)).float().div(255).unsqueeze(1)
    return ((x - 0.5) / 0.25).to(device, non_blocking=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--r-air", required=True)
    ap.add_argument("--locked", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", choices=["small_resnet", "simple_cnn"], required=True)
    ap.add_argument("--target", choices=["response", "r_air"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--dual", type=float, default=None, help="two-head: loss = CE + DUAL x MSE")
    ap.add_argument("--smoke", action="store_true", help="1 image per sensor, 1 epoch")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda")
    if args.smoke:
        args.epochs = 1

    labels = read_labels(args.labels)
    test_locked = set(locked(args.locked))
    images = [im for im in C.index_sem(args.sem) if im.sensor in labels and im.sensor not in test_locked]
    dev = sorted({im.sensor for im in images})
    assert len(dev) == 56, len(dev)
    if args.target == "response":
        y_s = {s: labels[s]["response"] for s in dev}
    else:
        ra = {r["sensor"]: math.log10(float(r["r_air_median"])) for r in csv.DictReader(open(args.r_air))}
        y_s = {s: ra[s] for s in dev}
    tr_idx, _ = list(RepeatedKFold(n_splits=5, n_repeats=1, random_state=0).split(np.zeros(len(dev))))[args.fold]
    train_s = {dev[i] for i in tr_idx}
    if args.smoke:
        seen, keep = set(), []
        for im in images:
            if im.sensor not in seen:
                seen.add(im.sensor)
                keep.append(im)
        images = keep

    arrays = [stretch(C.load_image_array(im.path)) for im in images]
    img_sensor = [im.sensor for im in images]
    grid = [(r, c) for r in range(C.CROP_H // TILE) for c in range(C.CROP_W // TILE)]
    tiles = [(i, r, c) for i in range(len(images)) for r, c in grid]
    train_tiles = [t for t in tiles if img_sensor[t[0]] in train_s]
    ytr = np.array([y_s[s] for s in sorted(train_s)])
    mu, sd = ytr.mean(), ytr.std()
    edges = np.quantile(ytr, [0.25, 0.5, 0.75])                     # training sensors only
    c_s = {s: int(np.digitize(y_s[s], edges)) for s in dev}
    print(f"{args.model} {args.target} fold {args.fold} seed {args.seed}: {len(train_s)} train sensors, "
          f"{len(train_tiles)} train tiles", flush=True)

    model = build(args.model, 4 if args.dual is not None else 0).to(device).to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps = args.epochs * math.ceil(len(train_tiles) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=steps, pct_start=0.1)
    hist = []
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train_tiles))
        tot, n, t0 = 0.0, 0, time.time()
        for b in range(0, len(order), args.batch):
            idx = [train_tiles[k] for k in order[b:b + args.batch]]
            x = batch_tensor(arrays, idx, device, True, rng)
            y = torch.tensor([(y_s[img_sensor[i]] - mu) / sd for i, _, _ in idx], device=device, dtype=torch.float32)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                reg, logit = model(x)
                loss = nn.functional.mse_loss(reg.float(), y)
                if args.dual is not None:
                    c = torch.tensor([c_s[img_sensor[i]] for i, _, _ in idx], device=device)
                    loss = nn.functional.cross_entropy(logit.float(), c) + args.dual * loss
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        hist.append(tot / n)
        print(f"epoch {ep + 1}/{args.epochs} train mse {tot / n:.4f} ({time.time() - t0:.0f} s)", flush=True)

    model.eval()
    preds, probs = {s: [] for s in dev}, {s: [] for s in dev}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, len(tiles), 128):
            idx = tiles[b:b + 128]
            reg, logit = model(batch_tensor(arrays, idx, device, False, rng))
            p = reg.float().cpu().numpy() * sd + mu
            q = torch.softmax(logit.float(), 1).cpu().numpy() if logit is not None else np.zeros((len(idx), 4))
            for (i, _, _), pi, qi in zip(idx, p, q):
                preds[img_sensor[i]].append(pi)
                probs[img_sensor[i]].append(qi)
    held = [s for s in dev if s not in train_s]
    rho = spearmanr([np.mean(preds[s]) for s in held], [y_s[s] for s in held]).statistic
    tag = "" if args.dual is None else f"_dual{args.dual:g}"
    out = Path(args.out) / f"{args.model}_{args.target}{tag}_s{args.seed}" / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "preds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "y", "pred", "heldout", "true_class", "exp_class", "pred_class"])
        for s in dev:
            q = np.mean(probs[s], 0)
            w.writerow([s, y_s[s], float(np.mean(preds[s])), s not in train_s, c_s[s],
                        float(np.mean([qi @ np.arange(4) for qi in probs[s]])), int(q.argmax())])
    prov = {f: hashlib.md5(open(f, "rb").read()).hexdigest()[:12] for f in ("common.py", "train_small.py", "dev_eval.py")}
    json.dump(dict(vars(args), train_mse=hist, heldout_rho=float(rho), n_heldout=len(held), provenance=prov,
                   n_params=sum(p.numel() for p in model.parameters())), open(out / "log.json", "w"), indent=1)
    if args.dual is not None:
        rho_c = spearmanr([np.mean([qi @ np.arange(4) for qi in probs[s]]) for s in held], [y_s[s] for s in held]).statistic
        print(f"RESULT expected-class rho {rho_c:+.3f}", flush=True)
    print(f"RESULT {args.model} {args.target} fold {args.fold} seed {args.seed} held-out rho {rho:+.3f}", flush=True)


if __name__ == "__main__":
    main()
