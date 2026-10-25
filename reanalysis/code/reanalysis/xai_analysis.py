"""XAI and feature analyses for one trained fold (held-out sensors only).

Run after train_cv.py. For fold k of a run it loads best.pt and the fold's
test images, then writes to <run>/fold<k>/xai/:

  cosine.json            category-pair cosine similarity of GAP features, both
                         as in the paper (all pairs, incl. same sensor and self)
                         and corrected (pairs from different sensors only), with a
                         sensor-level label-permutation test of within - between
  iou_between.csv        per sensor: between-category LayerCAM IoU (paper method:
                         per-map min-max, threshold 0.6, pooled over the sensor's
                         patches), mask area fractions, analytic chance IoU and a
                         patch-shuffle null; for per-patch and per-image normalisation
  iou_within.csv         per sensor: mean/std IoU between pairs of patches'
                         true-class maps (the Fig. 4 statistic)
  sanity.json            model-randomisation check and deletion test
  patch_outputs.csv      per test patch: predicted SNR, class probabilities
  tsne_points.csv        fold 0 only: t-SNE of train + test (+ replicate) patches,
                         columns x, y, snr, split, sensor (input to make_tsne_layer.py)

Prints a final "RESULT" line.
"""
import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import common as C
from train_cv import PatchSet

THR = 0.6


def load_fold(run_dir, fold):
    s = json.loads((Path(run_dir) / f"fold{fold}" / "summary.json").read_text())
    return s, s["args"]


def minmax(a, axes):
    lo = a.min(axis=axes, keepdims=True)
    hi = a.max(axis=axes, keepdims=True)
    return (a - lo) / np.where(hi > lo, hi - lo, 1.0)


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum()) / float(u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="<out>/<run> directory from train_cv.py")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--replicate-sem", default="", help="optional replicate SEM root for t-SNE")
    ap.add_argument("--replicate-snr", default="", help="optional replicate snr_per_sensor.csv")
    ap.add_argument("--n-perm", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=20)
    args = ap.parse_args()
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summ, targs = load_fold(args.run, args.fold)
    fold_dir = Path(args.run) / f"fold{args.fold}"
    out = fold_dir / "xai"
    out.mkdir(exist_ok=True)
    use_params = bool(targs["params"])
    labels = C.load_labels(targs["snr"], include_outlier=targs["include_outlier"])
    images = [im for im in C.index_sem(targs["sem"]) if im.sensor in labels]
    sensors = [im.sensor for im in images]
    test_s = set(summ["test_sensors"])
    test_imgs = [i for i, s in enumerate(sensors) if s in test_s]

    model = C.build_model(n_params=4 if use_params else 0, return_logits=not targs["faithful"]).to(device)
    model.load_state_dict(torch.load(fold_dir / "best.pt", map_location=device))
    model.eval()

    arrays = {i: C.load_image_array(images[i].path) for i in test_imgs}
    ds = PatchSet(arrays, [(i, p) for i in test_imgs for p in range(40)], labels, sensors, use_params)

    # ---------------------------------------------------------- per-patch pass
    feats, preds, probs, maps, metas = [], [], [], [], []
    for start in range(0, len(ds), args.batch):
        batch = [ds[j] for j in range(start, min(start + args.batch, len(ds)))]
        x = torch.stack([b[0] for b in batch]).to(device)
        prm = torch.stack([b[1] for b in batch]).to(device)
        with torch.no_grad():
            f = model.features(x)
            oc, orr = model(x, prm)
        feats.append(f.cpu().numpy())
        preds.append(orr.cpu().numpy())
        probs.append((torch.softmax(oc, 1) if not targs["faithful"] else oc).cpu().numpy())
        with torch.enable_grad():
            store = {}
            h = model.layer3.register_forward_hook(lambda m, i, o: store.__setitem__("A", o))
            oc2, _ = model(x, prm)
            A = store["A"]
            m4 = []
            for c in range(4):
                g, = torch.autograd.grad(oc2[:, c].sum(), A, retain_graph=c < 3)
                m4.append(F.relu((F.relu(g) * A).sum(1)).detach().cpu().numpy())
            h.remove()
        maps.append(np.stack(m4, 1))                    # (B, 4, 30, 30)
        metas += [(b[4], b[5], b[2], float(b[3])) for b in batch]
    feats, preds, probs, maps = map(np.concatenate, (feats, preds, probs, maps))
    img_ids = np.array([m[0] for m in metas])
    patch_ids = np.array([m[1] for m in metas])
    cats = np.array([m[2] for m in metas])
    snrs = np.array([m[3] for m in metas])
    sen = np.array([sensors[i] for i in img_ids])

    with open(out / "patch_outputs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sensor", "image", "patch", "true_cat", "true_snr", "pred_snr", "p_low", "p_mid", "p_high", "p_ultra"])
        for k in range(len(metas)):
            w.writerow([sen[k], images[img_ids[k]].path.name, patch_ids[k], cats[k], snrs[k], preds[k], *probs[k]])

    # ---------------------------------------------------------- cosine similarity
    fn = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    S = fn @ fn.T
    same_sensor = sen[:, None] == sen[None, :]
    eye = np.eye(len(S), dtype=bool)

    def cos_table(cat_vec, exclude_same):
        t = {}
        for a in range(4):
            for b in range(a, 4):
                m = (cat_vec[:, None] == a) & (cat_vec[None, :] == b)
                m &= ~eye
                if exclude_same:
                    m &= ~same_sensor
                if m.any():
                    t[f"{C.CATEGORIES[a]}-{C.CATEGORIES[b]}"] = (float(S[m].mean()), float(S[m].std()), int(m.sum()))
        return t

    def gap(cat_vec):
        within = (cat_vec[:, None] == cat_vec[None, :]) & ~same_sensor
        between = cat_vec[:, None] != cat_vec[None, :]
        return float(S[within].mean() - S[between].mean())

    uniq_s = sorted(set(sen))
    s_cat = {s: C.snr_category(labels[s]) for s in uniq_s}
    obs = gap(cats)
    null = []
    for _ in range(args.n_perm):
        perm = dict(zip(uniq_s, rng.permutation([s_cat[s] for s in uniq_s])))
        null.append(gap(np.array([perm[s] for s in sen])))
    null = np.array(null)
    cosine = dict(paper_style=cos_table(cats, exclude_same=False), different_sensors_only=cos_table(cats, True),
                  gap_within_minus_between=obs, perm_null_mean=float(null.mean()), perm_null_sd=float(null.std()),
                  perm_p=float((1 + (null >= obs).sum()) / (1 + len(null))), n_perm=args.n_perm,
                  n_test_sensors=len(uniq_s))
    (out / "cosine.json").write_text(json.dumps(cosine, indent=1))

    # ---------------------------------------------------------- LayerCAM IoU
    rows_b, rows_w = [], []
    for norm in ("patch", "image"):
        if norm == "patch":
            nm = minmax(maps, (2, 3))
        else:
            nm = np.empty_like(maps)
            for i in np.unique(img_ids):
                sel = img_ids == i
                nm[sel] = minmax(maps[sel], (0, 2, 3))   # per class, over the image's 40 patches
        masks = nm >= THR
        for s in uniq_s:
            sel = sen == s
            M = masks[sel]                                       # (n, 4, 30, 30)
            row = dict(sensor=s, norm=norm, category=C.CATEGORIES[s_cat[s]], n_patches=int(sel.sum()))
            for a, b in combinations(range(4), 2):
                key = f"{C.CATEGORIES[a]}-{C.CATEGORIES[b]}"
                row[key] = iou(M[:, a], M[:, b])
                pa, pb = M[:, a].mean(), M[:, b].mean()
                row[key + "_chance"] = float(pa * pb / (pa + pb - pa * pb)) if pa + pb > 0 else float("nan")
                shuf = rng.permutation(len(M))
                row[key + "_shuffle"] = iou(M[:, a], M[shuf, b])
            for c in range(4):
                row[f"area_{C.CATEGORIES[c]}"] = float(M[:, c].mean())
            rows_b.append(row)
            if norm == "patch":
                tc = s_cat[s]
                pairs = list(combinations(range(len(M)), 2))
                vals = np.array([iou(M[i, tc], M[j, tc]) for i, j in pairs])
                rows_w.append(dict(sensor=s, category=C.CATEGORIES[tc], mean_iou=float(vals.mean()),
                                   std_iou=float(vals.std()), n_pairs=len(vals)))
    for name, rows in (("iou_between.csv", rows_b), ("iou_within.csv", rows_w)):
        with open(out / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # ---------------------------------------------------------- sanity checks
    sub = rng.choice(len(ds), size=min(400, len(ds)), replace=False)
    xs = torch.stack([ds[j][0] for j in sub]).to(device)
    ps = torch.stack([ds[j][1] for j in sub]).to(device)
    true_c = cats[sub]

    def cam_true(mdl):
        out_maps = []
        for start in range(0, len(sub), args.batch):
            x = xs[start:start + args.batch]
            store = {}
            h = mdl.layer3.register_forward_hook(lambda m, i, o: store.__setitem__("A", o))
            oc, _ = mdl(x, ps[start:start + args.batch])
            A = store["A"]
            tc = torch.as_tensor(true_c[start:start + args.batch], device=device)
            g, = torch.autograd.grad(oc.gather(1, tc[:, None]).sum(), A)
            out_maps.append(F.relu((F.relu(g) * A).sum(1)).detach().cpu().numpy())
            h.remove()
        return np.concatenate(out_maps)

    from scipy.stats import spearmanr
    base_maps = cam_true(model)
    sanity = {}
    for level in ("heads", "heads+layer3", "all"):
        rnd = C.build_model(n_params=4 if use_params else 0, return_logits=not targs["faithful"]).to(device)
        rnd.load_state_dict(model.state_dict())
        rnd.eval()
        torch.manual_seed(1)
        mods = {"heads": [rnd.fcC, rnd.fcR],
                "heads+layer3": [rnd.fcC, rnd.fcR, rnd.layer3],
                "all": [rnd]}[level]
        for m in mods:
            for mm in m.modules():
                if hasattr(mm, "reset_parameters"):
                    mm.reset_parameters()
        rm = cam_true(rnd)
        rho = [spearmanr(a.ravel(), b.ravel()).statistic for a, b in zip(base_maps, rm)]
        sanity[f"randomise_{level}_spearman_mean"] = float(np.nanmean(rho))

    # deletion: replace top-20% CAM cells (of the 30x30 grid) vs random 20% with the patch mean
    k = int(0.2 * 900)
    cell = C.PATCH // 30
    tc_all = torch.as_tensor(true_c, device=device)[:, None]

    def p_true(x):   # batched, the full 400-patch stack does not fit on the GPU
        out_p = []
        for i in range(0, len(x), args.batch):
            oc = model(x[i:i + args.batch], ps[i:i + args.batch])[0]
            oc = oc if targs["faithful"] else torch.softmax(oc, 1)
            out_p.append(oc.gather(1, tc_all[i:i + args.batch]).squeeze(1))
        return torch.cat(out_p)

    with torch.no_grad():
        p0 = p_true(xs)

        def delete(order_fn):
            xd = xs.clone()
            for n in range(len(xd)):
                idx = order_fn(n)
                mean = xd[n].mean()
                for flat in idx:
                    r, c = divmod(int(flat), 30)
                    xd[n, :, r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = mean
            return p_true(xd)

        p_top = delete(lambda n: np.argsort(base_maps[n].ravel())[::-1][:k])
        p_rnd = delete(lambda n: rng.choice(900, k, replace=False))
    sanity.update(deletion_fraction=0.2, n_patches=len(sub),
                  p_true_original=float(p0.mean()), p_true_after_top_deletion=float(p_top.mean()),
                  p_true_after_random_deletion=float(p_rnd.mean()))
    (out / "sanity.json").write_text(json.dumps(sanity, indent=1))

    # ---------------------------------------------------------- t-SNE export (fold 0)
    if args.fold == 0:
        from sklearn.manifold import TSNE
        train_imgs = [i for i, s in enumerate(sensors) if s not in test_s]
        tr_arrays = {i: C.load_image_array(images[i].path) for i in train_imgs}
        tr_ds = PatchSet(tr_arrays, [(i, p) for i in train_imgs for p in range(40)], labels, sensors, use_params)
        tr_feats = []
        with torch.no_grad():
            for start in range(0, len(tr_ds), 64):
                x = torch.stack([tr_ds[j][0] for j in range(start, min(start + 64, len(tr_ds)))]).to(device)
                tr_feats.append(model.features(x).cpu().numpy())
        tr_feats = np.concatenate(tr_feats)
        all_f = [tr_feats, feats]
        split = ["train"] * len(tr_feats) + ["test"] * len(feats)
        all_sen = [sensors[i] for i in train_imgs for _ in range(40)] + list(sen)
        all_snr = [labels[s] for s in all_sen]
        if args.replicate_sem and args.replicate_snr:
            rep_labels = C.load_labels(args.replicate_snr, include_outlier=True)
            rep_imgs = [im for im in C.index_sem(args.replicate_sem) if im.sensor in rep_labels]
            rep_arrays = {i: C.load_image_array(im.path) for i, im in enumerate(rep_imgs)}
            rep_ds = PatchSet(rep_arrays, [(i, p) for i in rep_arrays for p in range(40)],
                              rep_labels, [im.sensor for im in rep_imgs], use_params)
            with torch.no_grad():
                rf = np.concatenate([model.features(torch.stack([rep_ds[j][0] for j in range(s, min(s + 64, len(rep_ds)))]).to(device)).cpu().numpy()
                                     for s in range(0, len(rep_ds), 64)])
            all_f.append(rf)
            split += ["replicate"] * len(rf)
            all_sen += [rep_imgs[i].sensor for i in rep_arrays for _ in range(40)]
            all_snr += [rep_labels[rep_imgs[i].sensor] for i in rep_arrays for _ in range(40)]
        emb = TSNE(n_components=2, random_state=42, perplexity=30, init="pca",
                   learning_rate="auto").fit_transform(np.concatenate(all_f))
        with open(out / "tsne_points.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["x", "y", "snr", "split", "sensor"])
            for (xv, yv), snr, sp, s in zip(emb, all_snr, split, all_sen):
                w.writerow([f"{xv:.4f}", f"{yv:.4f}", snr, sp, s])

    pb = [r for r in rows_b if r["norm"] == "patch"]
    mean_iou = np.mean([np.mean([r[f"{C.CATEGORIES[a]}-{C.CATEGORIES[b]}"] for a, b in combinations(range(4), 2)]) for r in pb])
    mean_chance = np.nanmean([np.mean([r[f"{C.CATEGORIES[a]}-{C.CATEGORIES[b]}_chance"] for a, b in combinations(range(4), 2)]) for r in pb])
    print(f"RESULT xai fold={args.fold} cos_gap={obs:.4f} perm_p={cosine['perm_p']:.4f} "
          f"iou_between={mean_iou:.3f} iou_chance={mean_chance:.3f} "
          f"rand_all_rho={sanity['randomise_all_spearman_mean']:.3f} "
          f"p_true={sanity['p_true_original']:.3f}/{sanity['p_true_after_top_deletion']:.3f}/{sanity['p_true_after_random_deletion']:.3f}")


if __name__ == "__main__":
    main()
