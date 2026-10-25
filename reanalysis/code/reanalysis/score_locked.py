"""One-shot locked-test score for prediction files made without the labels.

Averages the 'pred' column of the given locked_preds.csv files (e.g. one per
fine-tuning seed), reads the locked labels (here and only here), and appends
rho, a permutation p, MAE and checksums to data/derived/final_test_log.csv.
Refuses if that log already holds a score for this locked file.
Usage: python score_locked.py --name NAME --labels L --locked LOCKED --dev-oof fold preds.csv ... --preds seed locked_preds.csv ...
"""
import argparse
import csv
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from dev_eval import locked
from qc_metrics import qc_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--locked", required=True)
    ap.add_argument("--dev-oof", nargs="+", required=True,
                    help="finetune preds.csv files (heldout column) for the QC cutoff, all folds and seeds")
    ap.add_argument("--preds", nargs="+", required=True, help="locked_preds.csv files, one per seed")
    a = ap.parse_args()
    lsha = hashlib.sha256(open(a.locked, "rb").read()).hexdigest()[:16]
    log = Path(a.labels).parent / "final_test_log.csv"
    if log.exists() and any(r.get("locked_sha") == lsha for r in csv.DictReader(open(log))):
        raise SystemExit("this locked set has already been scored; it is scored once")
    test = locked(a.locked)
    P = [{r["sensor"]: float(r["pred"]) for r in csv.DictReader(open(f))} for f in a.preds]
    sensors = [s for s in test if all(s in p for p in P)]
    pred = np.array([np.mean([p[s] for p in P]) for s in sensors])
    lab = {r["sensor"]: r for r in csv.DictReader(open(a.labels))}
    keep = [i for i, s in enumerate(sensors) if lab[s]["excluded_as_outlier"] != "True"]
    sensors, pred = [sensors[i] for i in keep], pred[keep]
    y = np.log10([float(lab[s]["rel_resp_median"]) for s in sensors])
    rho = spearmanr(pred, y).statistic
    rng = np.random.default_rng(0)
    null = np.array([spearmanr(pred, rng.permutation(y)).statistic for _ in range(10000)])
    res = dict(when=datetime.now().isoformat(timespec="seconds"), config=a.name, n_test=len(sensors),
               rho=round(float(rho), 4), perm_p=round(float((1 + (null >= rho).sum()) / 10001), 4),
               r2=round(float(1 - ((pred - y) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4),
               mae=round(float(np.abs(pred - y).mean()), 4), locked_sha=lsha,
               best_params="files: " + " ".join(hashlib.md5(open(f, "rb").read()).hexdigest()[:12] for f in a.preds))
    oof = {}
    for f in a.dev_oof:
        for r in csv.DictReader(open(f)):
            if r["heldout"] == "True":
                oof.setdefault(r["sensor"], []).append(float(r["pred"]))
    dev = sorted(oof)
    dev_y = np.log10([float(lab[s]["rel_resp_median"]) for s in dev])
    res.update(qc_report(np.array([np.mean(oof[s]) for s in dev]), dev_y, pred, y))
    new = not log.exists()
    with open(log, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res))
        if new:
            w.writeheader()
        w.writerow(res)
    print("FINAL", res)


if __name__ == "__main__":
    main()
