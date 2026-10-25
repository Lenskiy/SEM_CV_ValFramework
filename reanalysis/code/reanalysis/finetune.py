"""Fine-tune DINOv2 ViT-B/14 on SEM tiles to predict log relative response (GPU).

One job = one outer fold of the first repeat of dev_eval's RepeatedKFold(5,
random_state=0) over the 56 development sensors, so results line up with the
frozen-feature screen. The 15 locked test sensors are never loaded.

Input: native 448 px tiles (40 per image), per-image 1-99 percentile stretch
(the pre-registered frozen config). Every tile carries its sensor's label.
Model: last --unfreeze transformer blocks and the final norm are trained,
the rest stays frozen; a linear head on [CLS, mean patch token] predicts the
standardised label. All settings are fixed in advance; no epoch or setting is
chosen on the held-out fold, which is scored only once, at the end.

Writes <out>/ft_u<unfreeze>_s<seed>/fold<k>/:
  preds.csv  sensor, y, pred (mean over the sensor's tiles)
  emb.npz    sensor-mean embeddings of all dev sensors from the tuned model,
             with a train/test mask, for ridge (+ parameters) downstream
  log.json   settings, provenance, per-epoch train loss, held-out rho
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
from scipy.stats import spearmanr
from sklearn.model_selection import RepeatedKFold

import common as C
from dev_eval import locked, read_labels

DINO_REPO = Path.home() / ".cache/torch/hub/facebookresearch_dinov2_main"
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
TILE = 448
TARGET = "response"


class Regressor(torch.nn.Module):
    def __init__(self, unfreeze):
        super().__init__()
        self.backbone = torch.hub.load(str(DINO_REPO), "dinov2_vitb14", source="local")
        for p in self.backbone.parameters():
            p.requires_grad = False
        for blk in self.backbone.blocks[-unfreeze:] if unfreeze else []:
            for p in blk.parameters():
                p.requires_grad = True
        for p in self.backbone.norm.parameters():
            p.requires_grad = unfreeze > 0
        self.head = torch.nn.Linear(2 * self.backbone.embed_dim, 1)

    def embed(self, x):
        f = self.backbone.forward_features(x)
        return torch.cat([f["x_norm_clstoken"], f["x_norm_patchtokens"].mean(1)], 1)

    def forward(self, x):
        return self.head(self.embed(x)).squeeze(1)


def stretch(a):
    """Per-image 1-99 percentile stretch to uint8, as extract_features 'pct'."""
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
    return ((x.expand(-1, 3, -1, -1) - MEAN) / STD).to(device, non_blocking=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--locked", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fold", type=int, required=True, help="0-4; -1 = final: train on all 56, predict the locked sensors")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unfreeze", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5, help="backbone blocks")
    ap.add_argument("--head-lr", type=float, default=1e-3)
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
    if args.fold == -1:
        train_s = set(dev)
    else:
        tr_idx, te_idx = list(RepeatedKFold(n_splits=5, n_repeats=1, random_state=0).split(np.zeros(len(dev))))[args.fold]
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
    y_s = {s: labels[s][TARGET] for s in dev}
    ytr = np.array([y_s[s] for s in sorted(train_s)])
    mu, sd = ytr.mean(), ytr.std()
    print(f"fold {args.fold}: {len(train_s)} train / {len(dev) - len(train_s)} held-out sensors, "
          f"{len(images)} images, {len(train_tiles)} train tiles", flush=True)

    model = Regressor(args.unfreeze).to(device)
    bb = [p for p in model.backbone.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr}, {"params": model.head.parameters(), "lr": args.head_lr}],
                            weight_decay=0.05)
    steps = args.epochs * math.ceil(len(train_tiles) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr], total_steps=steps, pct_start=0.1)

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
                loss = torch.nn.functional.mse_loss(model(x).float(), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(idx)
            n += len(idx)
        hist.append(tot / n)
        print(f"epoch {ep + 1}/{args.epochs} train mse {tot / n:.4f} ({time.time() - t0:.0f} s)", flush=True)

    model.eval()
    out = Path(args.out) / f"ft_u{args.unfreeze}_s{args.seed}" / ("final" if args.fold == -1 else f"fold{args.fold}")
    out.mkdir(parents=True, exist_ok=True)
    if args.fold == -1:
        return predict_locked(args, model, test_locked, mu, sd, rng, device, out, hist)

    # one pass over every dev tile: predictions and embeddings
    preds, embs = {s: [] for s in dev}, {s: [] for s in dev}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, len(tiles), 64):
            idx = tiles[b:b + 64]
            e = model.embed(batch_tensor(arrays, idx, device, False, rng))
            p = model.head(e).squeeze(1).float().cpu().numpy() * sd + mu
            for (i, _, _), pi, ei in zip(idx, p, e.float().cpu().numpy()):
                preds[img_sensor[i]].append(pi)
                embs[img_sensor[i]].append(ei)
    held = [s for s in dev if s not in train_s]
    yh = np.array([y_s[s] for s in held])
    ph = np.array([np.mean(preds[s]) for s in held])
    rho = spearmanr(ph, yh).statistic

    with open(out / "preds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "y", "pred", "heldout"])
        for s in dev:
            w.writerow([s, y_s[s], float(np.mean(preds[s])), s not in train_s])
    np.savez(out / "emb.npz", sensors=np.array(dev), emb=np.stack([np.mean(embs[s], 0) for s in dev]),
             heldout=np.array([s not in train_s for s in dev]))
    prov = {f: hashlib.md5(open(f, "rb").read()).hexdigest()[:12] for f in ("common.py", "finetune.py", "dev_eval.py")}
    json.dump(dict(vars(args), train_mse=hist, heldout_rho=float(rho), n_heldout=len(held), provenance=prov,
                   locked_sha=hashlib.sha256(open(args.locked, "rb").read()).hexdigest()[:16]),
              open(out / "log.json", "w"), indent=1)
    print(f"RESULT fold {args.fold} held-out rho {rho:+.3f} (n={len(held)})", flush=True)


def predict_locked(args, model, test_locked, mu, sd, rng, device, out, hist):
    """Final model: predictions for the locked sensors' images. Their labels are not read here."""
    images = [im for im in C.index_sem(args.sem) if im.sensor in test_locked]
    arrays = [stretch(C.load_image_array(im.path)) for im in images]
    grid = [(r, c) for r in range(C.CROP_H // TILE) for c in range(C.CROP_W // TILE)]
    tiles = [(i, r, c) for i in range(len(images)) for r, c in grid]
    preds = {}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, len(tiles), 64):
            idx = tiles[b:b + 64]
            p = model(batch_tensor(arrays, idx, device, False, rng)).float().cpu().numpy() * sd + mu
            for (i, _, _), pi in zip(idx, p):
                preds.setdefault(images[i].sensor, []).append(pi)
    with open(out / "locked_preds.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "pred"])
        for s in sorted(preds):
            w.writerow([s, float(np.mean(preds[s]))])
    prov = {f: hashlib.md5(open(f, "rb").read()).hexdigest()[:12] for f in ("common.py", "finetune.py", "dev_eval.py")}
    json.dump(dict(vars(args), train_mse=hist, n_locked=len(preds), provenance=prov,
                   locked_sha=hashlib.sha256(open(args.locked, "rb").read()).hexdigest()[:16]),
              open(out / "log.json", "w"), indent=1)
    print(f"RESULT final model: predictions for {len(preds)} locked sensors written (labels not read)", flush=True)


if __name__ == "__main__":
    main()
