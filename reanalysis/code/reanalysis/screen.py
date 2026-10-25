"""E12: baseline resistance as a pass/fail screen for non-responders (development sensors only).

Non-responder: median relative response < 1% (as in qc_metrics). Screen: flag a sensor when its
baseline resistance in air (r_air.csv, measured at 300 C after stabilisation, before the first
acetone pulse) is below a threshold.
  - AUC with a 2,000-resample bootstrap interval over sensors
  - threshold chosen by leave-one-out (the held-out sensor is classified with the threshold that
    maximises balanced accuracy on the other 55): sensitivity, specificity, balanced accuracy,
    and the share of gas tests that would be skipped
  - the threshold chosen on all 56, reported for practitioners, with the same metrics in-sample
  - for comparison, the frozen image model's response predictions (out-of-fold, chain_oof.csv)
The 15 locked sensors are not used. Writes data/derived/screen.json.
"""
import csv
import json
from pathlib import Path

import numpy as np

from dev_eval import locked, read_labels
from qc_metrics import NONRESP_LOG10, auc

ROOT = Path(__file__).resolve().parents[2]


def best_threshold(x, dead):
    best, thr = -1, None
    for t in np.unique(x):
        f = x <= t
        ba = (f[dead].mean() + (~f[~dead]).mean()) / 2
        if ba > best:
            best, thr = ba, t
    return thr


def metrics(flag, dead):
    sens, spec = flag[dead].mean(), (~flag[~dead]).mean()
    return dict(sensitivity=round(float(sens), 3), specificity=round(float(spec), 3),
                balanced_accuracy=round(float((sens + spec) / 2), 3), flagged_share=round(float(flag.mean()), 3),
                good_sensors_rejected=int((flag & ~dead).sum()), dead_sensors_missed=int((~flag & dead).sum()))


def main():
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in lab if s not in test)
    ra = {r["sensor"]: float(r["r_air_median"]) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    x = np.log10([ra[s] for s in dev])
    dead = np.array([lab[s]["response"] < NONRESP_LOG10 for s in dev])
    rng = np.random.default_rng(0)
    a = auc(x, dead)
    boot = []
    for _ in range(2000):
        i = rng.integers(0, len(x), len(x))
        if dead[i].any() and (~dead[i]).any():
            boot.append(auc(x[i], dead[i]))
    out = dict(n=len(dev), n_nonresponders=int(dead.sum()),
               auc=dict(value=round(float(a), 3), ci95=[round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)]))
    flag, thr_loo = np.zeros(len(x), bool), []
    for k in range(len(x)):
        m = np.arange(len(x)) != k
        t = best_threshold(x[m], dead[m])
        thr_loo.append(t)
        flag[k] = x[k] <= t
    out["loo"] = metrics(flag, dead)
    out["loo"]["threshold_ohm_range"] = [round(float(10 ** min(thr_loo))), round(float(10 ** max(thr_loo)))]
    t = best_threshold(x, dead)
    out["all56"] = dict(threshold_ohm=round(float(10 ** t)), **metrics(x <= t, dead))
    for name, tt in (("fixed_10kohm", 4.0), ("fixed_100kohm", 5.0)):
        out[name] = metrics(x < tt, dead)
    oof = {r["sensor"]: float(r["img_pred_response"]) for r in csv.DictReader(open(ROOT / "data/derived/chain_oof.csv"))}
    out["image_model_auc"] = round(float(auc(np.array([oof[s] for s in dev]), dead)), 3)
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / "data/derived/screen.json", "w"), indent=1)


if __name__ == "__main__":
    main()
