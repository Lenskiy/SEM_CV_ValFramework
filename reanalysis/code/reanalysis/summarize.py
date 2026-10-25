"""Pool the folds of each run into the numbers the paper reports.

Usage:
    python summarize.py <runs dir> [--compare RUN_A RUN_B] [--baselines baselines.json]

For every run directory with fold*/summary.json:
  - fold mean +/- sample s.d. of patch/image/sensor accuracy and MAE
  - pooled sensor-level metrics over all held-out sensors, bootstrap 95% CIs
    (resampling sensors), confusion matrix
  - nested variance decomposition of patch-level predicted SNR:
    category / sensor-within-category / image-within-sensor / patch-within-image
  - pooled XAI results if fold*/xai exists
--compare does a paired sensor-level comparison of two runs (McNemar exact
test on correctness, bootstrap of the accuracy and MAE differences).
Writes <runs dir>/summary_<run>.json and prints a readable report.
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

CATS = ["Low", "Mid", "High", "Ultra-high"]
B = 10000


def read_preds(run):
    rows = []
    for f in sorted(run.glob("fold*/test_predictions.csv")):
        rows += list(csv.DictReader(open(f)))
    return rows


def sensor_table(rows):
    g = defaultdict(list)
    for r in rows:
        g[r["sensor"]].append(r)
    t = {}
    for s, rs in g.items():
        p = np.array([[float(r[k]) for k in ("p_low", "p_mid", "p_high", "p_ultra")] for r in rs]).mean(0)
        t[s] = dict(true_cat=int(rs[0]["true_cat"]), true_snr=float(rs[0]["true_snr"]),
                    pred_cat=int(p.argmax()), pred_snr=float(np.mean([float(r["pred_snr"]) for r in rs])))
    return t


def boot_ci(fn, n, rng):
    vals = [fn(rng.integers(0, n, n)) for _ in range(B)]
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def pooled_metrics(t, rng):
    s = sorted(t)
    tc = np.array([t[k]["true_cat"] for k in s]); pc = np.array([t[k]["pred_cat"] for k in s])
    ts = np.array([t[k]["true_snr"] for k in s]); ps = np.array([t[k]["pred_snr"] for k in s])

    def r2(i):
        return 1 - ((ts[i] - ps[i]) ** 2).sum() / ((ts[i] - ts[i].mean()) ** 2).sum()
    conf = np.zeros((4, 4), int)
    for a, b in zip(tc, pc):
        conf[a, b] += 1
    return dict(n_sensors=len(s),
                acc=float((tc == pc).mean()), acc_ci=boot_ci(lambda i: (tc[i] == pc[i]).mean(), len(s), rng),
                balanced_acc=float(np.mean([(pc[tc == k] == k).mean() for k in np.unique(tc)])),
                within_one_category=float((np.abs(tc - pc) <= 1).mean()),
                mae=float(np.abs(ts - ps).mean()), mae_ci=boot_ci(lambda i: np.abs(ts[i] - ps[i]).mean(), len(s), rng),
                rmse=float(np.sqrt(((ts - ps) ** 2).mean())), r2=float(r2(np.arange(len(s)))),
                r2_ci=boot_ci(r2, len(s), rng), confusion_true_rows=conf.tolist())


def variance_decomposition(rows):
    """Nested sums of squares of patch-level predicted SNR."""
    y = np.array([float(r["pred_snr"]) for r in rows])
    cat = np.array([r["true_cat"] for r in rows]); sen = np.array([r["sensor"] for r in rows])
    img = np.array([r["sensor"] + "|" + r["image"] for r in rows])
    grand = y.mean()

    mean_of = lambda labels: {g: y[labels == g].mean() for g in np.unique(labels)}
    cat_mean, sen_mean, img_mean = mean_of(cat), mean_of(sen), mean_of(img)
    ss_cat = sum((cat == c).sum() * (cat_mean[c] - grand) ** 2 for c in cat_mean)
    ss_sen = sum((sen == s).sum() * (sen_mean[s] - cat_mean[cat[sen == s][0]]) ** 2 for s in sen_mean)
    ss_img = sum((img == i).sum() * (img_mean[i] - sen_mean[sen[img == i][0]]) ** 2 for i in img_mean)
    ss_patch = float(((y - np.array([img_mean[i] for i in img])) ** 2).sum())
    tot = float(((y - grand) ** 2).sum())
    return {k: round(v / tot, 4) for k, v in
            dict(category=ss_cat, sensor_within_category=ss_sen, image_within_sensor=ss_img,
                 patch_within_image=ss_patch).items()}


def fold_stats(run):
    out = {}
    for level in ("patch", "image", "sensor"):
        for m in ("acc", "mae"):
            v = [json.loads(f.read_text())[level][m] for f in sorted(run.glob("fold*/summary.json"))]
            out[f"{level}_{m}"] = [float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, v]
    return out


def xai_pool(run):
    folds = sorted(run.glob("fold*/xai"))
    if not folds:
        return None
    res = {}
    cos = [json.loads((f / "cosine.json").read_text()) for f in folds if (f / "cosine.json").exists()]
    if cos:
        for style in ("paper_style", "different_sensors_only"):
            keys = sorted({k for c in cos for k in c[style]})
            res[f"cosine_{style}"] = {k: float(np.mean([c[style][k][0] for c in cos if k in c[style]])) for k in keys}
        res["cosine_gap_per_fold"] = [c["gap_within_minus_between"] for c in cos]
        res["cosine_perm_p_per_fold"] = [c["perm_p"] for c in cos]
    rows = []
    for f in folds:
        if (f / "iou_between.csv").exists():
            rows += list(csv.DictReader(open(f / "iou_between.csv")))
    pairs = [f"{a}-{b}" for i, a in enumerate(CATS) for b in CATS[i + 1:]]
    for norm in ("patch", "image"):
        rs = [r for r in rows if r["norm"] == norm]
        if rs:
            res[f"iou_between_{norm}norm"] = {p: [float(np.mean([float(r[p]) for r in rs])),
                                                  float(np.std([float(r[p]) for r in rs], ddof=1)),
                                                  float(np.nanmean([float(r[p + '_chance']) for r in rs])),
                                                  float(np.mean([float(r[p + '_shuffle']) for r in rs]))] for p in pairs}
            res[f"iou_between_{norm}norm_overall"] = dict(
                mean=float(np.mean([float(r[p]) for r in rs for p in pairs])),
                chance=float(np.nanmean([float(r[p + "_chance"]) for r in rs for p in pairs])),
                shuffle=float(np.mean([float(r[p + "_shuffle"]) for r in rs for p in pairs])))
    within = []
    for f in folds:
        if (f / "iou_within.csv").exists():
            within += list(csv.DictReader(open(f / "iou_within.csv")))
    if within:
        m = np.array([float(r["mean_iou"]) for r in within])
        res["iou_within_sensor"] = dict(n=len(m), mean=float(m.mean()), below_7_5=int((m < 0.075).sum()),
                                        between=int(((m >= 0.075) & (m <= 0.10)).sum()), above_10=int((m > 0.10).sum()))
        with open(run / "iou_within_all.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(within[0]))
            w.writeheader()
            w.writerows(within)
    san = [json.loads((f / "sanity.json").read_text()) for f in folds if (f / "sanity.json").exists()]
    if san:
        res["sanity_mean"] = {k: float(np.mean([s[k] for s in san])) for k in san[0] if isinstance(san[0][k], float)}
    return res


def mcnemar(ta, tb):
    common = sorted(set(ta) & set(tb))
    ca = np.array([ta[s]["true_cat"] == ta[s]["pred_cat"] for s in common])
    cb = np.array([tb[s]["true_cat"] == tb[s]["pred_cat"] for s in common])
    b01, b10 = int((ca & ~cb).sum()), int((~ca & cb).sum())
    p = binomtest(b01, b01 + b10, 0.5).pvalue if b01 + b10 else 1.0
    ea = np.array([abs(ta[s]["true_snr"] - ta[s]["pred_snr"]) for s in common])
    eb = np.array([abs(tb[s]["true_snr"] - tb[s]["pred_snr"]) for s in common])
    rng = np.random.default_rng(1)
    return dict(n=len(common), a_only_correct=b01, b_only_correct=b10, mcnemar_p=float(p),
                acc_diff=float(ca.mean() - cb.mean()),
                acc_diff_ci=boot_ci(lambda i: ca[i].mean() - cb[i].mean(), len(common), rng),
                mae_diff=float(ea.mean() - eb.mean()),
                mae_diff_ci=boot_ci(lambda i: ea[i].mean() - eb[i].mean(), len(common), rng))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--compare", nargs=2, action="append", default=[])
    args = ap.parse_args()
    root = Path(args.runs)
    tables = {}
    for run in sorted(p for p in root.iterdir() if p.is_dir() and list(p.glob("fold*/summary.json"))):
        rng = np.random.default_rng(0)
        rows = read_preds(run)
        t = sensor_table(rows)
        tables[run.name] = t
        rep = dict(run=run.name, n_folds=len(list(run.glob("fold*/summary.json"))),
                   folds=fold_stats(run), pooled_sensor=pooled_metrics(t, rng),
                   variance_decomposition=variance_decomposition(rows), xai=xai_pool(run))
        (root / f"summary_{run.name}.json").write_text(json.dumps(rep, indent=1))
        f, p = rep["folds"], rep["pooled_sensor"]
        print(f"\n== {run.name} ({rep['n_folds']} folds)")
        for k in ("patch_acc", "image_acc", "sensor_acc", "patch_mae", "sensor_mae"):
            print(f"  {k:11s} {f[k][0]:.4f} +/- {f[k][1]:.4f}")
        print(f"  pooled sensors n={p['n_sensors']} acc {p['acc']:.3f} {p['acc_ci']} bal {p['balanced_acc']:.3f} "
              f"within-1 {p['within_one_category']:.3f} MAE {p['mae']:.2f} {p['mae_ci']} R2 {p['r2']:.3f}")
        print(f"  confusion (rows true L/M/H/U): {p['confusion_true_rows']}")
        print(f"  variance of predicted SNR: {rep['variance_decomposition']}")
        if rep["xai"]:
            print(f"  xai: {json.dumps(rep['xai'], default=float)[:900]}")
    for a, b in args.compare:
        c = mcnemar(tables[a], tables[b])
        (root / f"compare_{a}__{b}.json").write_text(json.dumps(c, indent=1))
        print(f"\n== compare {a} vs {b}: {c}")
    print("RESULT summarize done")


if __name__ == "__main__":
    main()
