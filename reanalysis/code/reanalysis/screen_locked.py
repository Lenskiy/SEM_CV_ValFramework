"""E12b: one-shot locked check of the baseline-resistance screen (rule: data/derived/screen_locked_rule.txt).

Refuses to run if data/derived/screen_locked_log.json exists. Records the rule file's sha256.
"""
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from dev_eval import locked
from qc_metrics import auc

ROOT = Path(__file__).resolve().parents[2]
RULE = ROOT / "data/derived/screen_locked_rule.txt"
LOG = ROOT / "data/derived/screen_locked_log.json"
THRESHOLD = 1e4


def main():
    if LOG.exists():
        sys.exit(f"{LOG} exists: the locked screen check has already been run once.")
    sha = hashlib.sha256(RULE.read_bytes()).hexdigest()
    lab = {r["sensor"]: float(r["rel_resp_median"]) for r in csv.DictReader(open(ROOT / "data/derived/labels_v2.csv"))}
    ra = {r["sensor"]: float(r["r_air_median"]) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    S = [s for s in locked(ROOT / "data/derived/locked_test_sensors.txt") if s in lab]
    r = np.array([ra[s] for s in S])
    dead = np.array([lab[s] < 0.01 for s in S])
    flag = r < THRESHOLD
    sens, spec = flag[dead].mean(), (~flag[~dead]).mean()
    x = np.log10(r)
    a = auc(x, dead)
    rng = np.random.default_rng(0)
    boot = [auc(x[i], dead[i]) for i in (rng.integers(0, len(x), len(x)) for _ in range(2000))
            if dead[i].any() and (~dead[i]).any()]
    null = np.array([auc(x, rng.permutation(dead)) for _ in range(10000)])
    p = (1 + (null >= a).sum()) / 10001
    out = dict(run_at=datetime.now().isoformat(timespec="seconds"), rule_sha256=sha, n=len(S),
               n_nonresponders=int(dead.sum()), threshold_ohm=THRESHOLD,
               caught=f"{int((flag & dead).sum())}/{int(dead.sum())}",
               good_rejected=f"{int((flag & ~dead).sum())}/{int((~dead).sum())}",
               sensitivity=round(float(sens), 3), specificity=round(float(spec), 3),
               balanced_accuracy=round(float((sens + spec) / 2), 3),
               auc=round(float(a), 3), auc_ci95=[round(float(np.percentile(boot, q)), 3) for q in (2.5, 97.5)],
               auc_perm_p=round(float(p), 4),
               sensors=[dict(sensor=s, r_air_ohm=round(float(v)), rel_resp=lab[s], flagged=bool(f))
                        for s, v, f in zip(S, r, flag)])
    out["confirmed"] = bool(out["auc_perm_p"] < 0.05 and out["balanced_accuracy"] >= 0.70)
    json.dump(out, open(LOG, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "sensors"}, indent=1))


if __name__ == "__main__":
    main()
