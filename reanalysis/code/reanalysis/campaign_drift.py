"""E19: does the first batch drift over the fabrication campaign? (all 72 sensors; descriptive)

The workbooks carry no test dates (their file stamps are later re-saves), but every
sensor was imaged at 500x after its gas test, so the imaging date bounds the test date.
Each sensor gets its primary imaging day as in session_check.py (most images, preferring
days outside the August 2024 magnification screening). Checks, against that date:
Spearman rho with log10 baseline resistance and log10 relative response; and the share
of sensors below 10 kOhm and of non-responders (< 1%) in the earlier and later half.
"""
import collections
import csv
import json
import math
from datetime import date
from pathlib import Path

import numpy as np
from scipy.stats import fisher_exact, spearmanr

ROOT = Path(__file__).resolve().parents[2]
FEAT = ROOT / "data/features500x"

days = collections.defaultdict(collections.Counter)
for r in csv.DictReader(open(FEAT / "images.csv")):
    days[r["sensor"]][r["time"][:10]] += 1
prim = {s: max(c, key=lambda d: (not d.startswith("2024-08"), c[d], d)) for s, c in days.items()}
ra = {r["sensor"]: float(r["r_air_median"]) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
resp = {r["sensor"]: float(r["rel_resp_median"]) for r in csv.DictReader(open(ROOT / "data/derived/labels_v2.csv"))}
sens = sorted(s for s in prim if s in ra and s in resp)
t = np.array([date.fromisoformat(prim[s]).toordinal() for s in sens], float)
lr = np.array([math.log10(ra[s]) for s in sens])
ly = np.array([math.log10(resp[s]) for s in sens])
late = t > np.median(t)
out = dict(n=len(sens), first_day=prim[sens[int(t.argmin())]], last_day=prim[sens[int(t.argmax())]],
           n_days=len(set(prim[s] for s in sens)))
for name, v in (("log_r_air", lr), ("log_response", ly)):
    r = spearmanr(t, v)
    out[f"date_vs_{name}"] = dict(rho=round(float(r.statistic), 3), p=round(float(r.pvalue), 3))
for name, flag in (("below_10kohm", 10 ** lr < 1e4), ("non_responder", 10 ** ly < 0.01)):
    a, b = int(flag[~late].sum()), int(flag[late].sum())
    out[f"{name}_early_late"] = dict(early=f"{a}/{int((~late).sum())}", late=f"{b}/{int(late.sum())}",
                                    fisher_p=round(float(fisher_exact([[a, (~late).sum() - a], [b, late.sum() - b]]).pvalue), 3))
json.dump(out, open(ROOT / "data/derived/campaign_drift.json", "w"), indent=1)
print(json.dumps(out, indent=1))
