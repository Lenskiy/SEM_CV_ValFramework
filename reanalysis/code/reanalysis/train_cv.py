"""Train and evaluate one cross-validation fold of the dual-head ResNet.

Schemes (--cv):
  image   original notebook: 4 folds splitting each sensor's images, so every
          sensor appears in train and test (reproduces the paper's numbers)
  sensor  whole sensors held out: 5 folds stratified by category; early
          stopping on 10% of the TRAINING sensors

--params adds the encoded fabrication parameters to the image features
(image + parameters model). --faithful keeps the original softmax-before-
cross-entropy and random-patch validation split.

Outputs in <out>/<run>/fold<k>/: best.pt, history.csv, test_predictions.csv,
summary.json. Prints a final "RESULT" line.
"""
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import common as C


class PatchSet(Dataset):
    def __init__(self, arrays, items, labels, sensors, use_params):
        self.arrays, self.items, self.labels = arrays, items, labels
        self.sensors, self.use_params = sensors, use_params
        self.grid = C.patch_grid()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        img, p = self.items[i]
        r, c = self.grid[p]
        a = self.arrays[img][r * C.PATCH:(r + 1) * C.PATCH, c * C.PATCH:(c + 1) * C.PATCH]
        x = torch.from_numpy(a).float().div_(255.0).unsqueeze(0).expand(3, -1, -1).contiguous()
        snr = self.labels[self.sensors[img]]
        params = torch.from_numpy(C.fab_features(self.sensors[img])) if self.use_params else torch.zeros(0)
        return x, params, C.snr_category(snr), np.float32(snr), img, p


def run_epoch(model, loader, device, opt=None, temp=1e-3):
    train = opt is not None
    model.train(train)
    ce, mse = nn.CrossEntropyLoss(), nn.MSELoss()
    tot, n, correct = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, params, cat, snr, _, _ in loader:
            x, params = x.to(device, non_blocking=True), params.to(device)
            cat, snr = cat.to(device), snr.to(device)
            out_c, out_r = model(x, params)
            loss = ce(out_c, cat) + temp * mse(out_r, snr)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            tot += loss.item() * len(x)
            n += len(x)
            correct += (out_c.argmax(1) == cat).sum().item()
    return tot / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    rows = []
    for x, params, cat, snr, img, p in loader:
        out_c, out_r = model(x.to(device), params.to(device))
        prob = torch.softmax(out_c, 1) if model.return_logits else out_c
        for k in range(len(x)):
            rows.append((int(img[k]), int(p[k]), int(cat[k]), float(snr[k]),
                         *[float(v) for v in prob[k].cpu()], float(out_r[k].cpu())))
    return rows


def metrics(true_snr, pred_snr, true_cat, pred_cat):
    true_snr, pred_snr = np.asarray(true_snr), np.asarray(pred_snr)
    ss_res = ((true_snr - pred_snr) ** 2).sum()
    ss_tot = ((true_snr - true_snr.mean()) ** 2).sum()
    return dict(n=len(true_snr), acc=float(np.mean(np.asarray(true_cat) == np.asarray(pred_cat))),
                mae=float(np.abs(true_snr - pred_snr).mean()),
                rmse=float(np.sqrt(((true_snr - pred_snr) ** 2).mean())),
                r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))


def aggregate(rows, key):
    """Average class probabilities and SNR predictions over patches per key."""
    groups = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out = {}
    for k, rs in groups.items():
        a = np.array([r[4:9] for r in rs])
        out[k] = dict(true_cat=rs[0][2], true_snr=rs[0][3], prob=a[:, :4].mean(0), pred_snr=a[:, 4].mean())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem", required=True)
    ap.add_argument("--snr", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cv", choices=["image", "sensor"], required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--folds", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--params", action="store_true")
    ap.add_argument("--include-outlier", action="store_true")
    ap.add_argument("--faithful", action="store_true")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=40)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-train-patches", type=int, default=0, help="smoke tests only")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run = f"{args.cv}cv{'_params' if args.params else ''}{'_withoutlier' if args.include_outlier else ''}" \
          f"{'_faithful' if args.faithful else ''}_s{args.seed}"
    out = Path(args.out) / run / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    for f in ("common.py", "train_cv.py"):
        print("PROV", hashlib.md5(Path(f).read_bytes()).hexdigest()[:12], f)
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    labels = C.load_labels(args.snr, include_outlier=args.include_outlier)
    images = [im for im in C.index_sem(args.sem) if im.sensor in labels]
    missing = sorted(set(labels) - {im.sensor for im in images})
    print(f"images {len(images)} of {len(set(im.sensor for im in images))} sensors; labelled without images: {missing}")
    sensors = [im.sensor for im in images]

    # ----- folds
    if args.cv == "image":
        n_folds = args.folds or 4
        test_imgs = set(C.image_folds(images, n_folds, seed=42)[args.fold])
        train_imgs = [i for i in range(len(images)) if i not in test_imgs]
    else:
        n_folds = args.folds or 5
        uniq = sorted(labels)
        folds = C.sensor_folds(uniq, [C.snr_category(labels[s]) for s in uniq], n_folds, seed=args.seed)
        test_sensors = set(folds[args.fold])
        test_imgs = {i for i, s in enumerate(sensors) if s in test_sensors}
        train_imgs = [i for i in range(len(images)) if i not in test_imgs]
    test_imgs = sorted(test_imgs)

    rng = np.random.default_rng(args.seed)
    if args.cv == "sensor" and not args.faithful:
        # validation = 10% of training sensors, stratified by category
        tr_s = sorted({sensors[i] for i in train_imgs})
        val_s = set()
        for c in range(4):
            members = [s for s in tr_s if C.snr_category(labels[s]) == c]
            rng.shuffle(members)
            val_s.update(members[:max(1, round(0.1 * len(members)))])
        val_items = [(i, p) for i in train_imgs if sensors[i] in val_s for p in range(40)]
        fit_items = [(i, p) for i in train_imgs if sensors[i] not in val_s for p in range(40)]
    else:
        # original: random 10% of training patches
        items = [(i, p) for i in train_imgs for p in range(40)]
        perm = rng.permutation(len(items))
        n_val = int(0.1 * len(items))
        val_items = [items[j] for j in perm[:n_val]]
        fit_items = [items[j] for j in perm[n_val:]]
    if args.max_train_patches:
        fit_items = [fit_items[j] for j in rng.permutation(len(fit_items))[:args.max_train_patches]]
        val_items = val_items[:max(64, args.max_train_patches // 8)]
    test_items = [(i, p) for i in test_imgs for p in range(40)]
    print(f"fold {args.fold}/{n_folds}: train patches {len(fit_items)}, val {len(val_items)}, test {len(test_items)} "
          f"({len({sensors[i] for i in test_imgs})} test sensors)")

    # ----- data in memory
    t0 = time.time()
    need = sorted({i for i, _ in fit_items + val_items + test_items})
    arrays = {i: C.load_image_array(images[i].path) for i in need}
    print(f"loaded {len(arrays)} images in {time.time() - t0:.0f}s")

    def loader(items, shuffle):
        return DataLoader(PatchSet(arrays, items, labels, sensors, args.params), batch_size=args.batch,
                          shuffle=shuffle, num_workers=args.workers, pin_memory=True,
                          persistent_workers=args.workers > 0)

    fit_dl, val_dl, test_dl = loader(fit_items, True), loader(val_items, False), loader(test_items, False)

    model = C.build_model(n_params=4 if args.params else 0, return_logits=not args.faithful).to(device)
    model.return_logits = not args.faithful
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, "min")
    best, wait, hist = float("inf"), 0, []
    for ep in range(args.epochs):
        t1 = time.time()
        tr_loss, tr_acc = run_epoch(model, fit_dl, device, opt)
        va_loss, va_acc = run_epoch(model, val_dl, device)
        sched.step(va_loss)
        hist.append((ep, tr_loss, tr_acc, va_loss, va_acc, opt.param_groups[0]["lr"], time.time() - t1))
        improved = va_loss < best
        if improved:
            best, wait = va_loss, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            wait += 1
        print(f"ep {ep} train {tr_loss:.4f}/{tr_acc:.3f} val {va_loss:.4f}/{va_acc:.3f} "
              f"lr {opt.param_groups[0]['lr']:.1e} {time.time() - t1:.0f}s{' *' if improved else ''}", flush=True)
        if wait >= args.patience:
            break
    with open(out / "history.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr", "seconds"])
        w.writerows(hist)

    # ----- test predictions
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    rows = predict(model, test_dl, device)
    with open(out / "test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sensor", "image", "image_time", "patch", "true_cat", "true_snr",
                    "p_low", "p_mid", "p_high", "p_ultra", "pred_snr"])
        for r in rows:
            im = images[r[0]]
            w.writerow([im.sensor, im.path.name, im.meta.get("time", ""), r[1], r[2], r[3], *r[4:]])

    patch = metrics([r[3] for r in rows], [r[8] for r in rows], [r[2] for r in rows],
                    [int(np.argmax(r[4:8])) for r in rows])
    img_agg = aggregate(rows, key=lambda r: r[0])
    sen_agg = aggregate(rows, key=lambda r: sensors[r[0]])

    def agg_metrics(a):
        v = list(a.values())
        return metrics([g["true_snr"] for g in v], [g["pred_snr"] for g in v],
                       [g["true_cat"] for g in v], [int(np.argmax(g["prob"])) for g in v])

    summary = dict(run=run, fold=args.fold, n_folds=n_folds, epochs_run=len(hist), best_val_loss=best,
                   patch=patch, image=agg_metrics(img_agg), sensor=agg_metrics(sen_agg),
                   test_sensors=sorted({sensors[i] for i in test_imgs}), args=vars(args))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"RESULT run={run} fold={args.fold} patch_acc={patch['acc']:.4f} patch_mae={patch['mae']:.3f} "
          f"image_acc={summary['image']['acc']:.4f} sensor_acc={summary['sensor']['acc']:.4f} "
          f"sensor_mae={summary['sensor']['mae']:.3f} sensor_r2={summary['sensor']['r2']:.3f}")


if __name__ == "__main__":
    main()
