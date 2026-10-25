"""E17: direction of the resistance change (review check; development sensors for statistics).

The label is |R_gas - R_air| / R_air, which discards the sign. For ZnO (n-type) and acetone
(reducing) the resistance should fall. Per cycle (same windows as labels_v2.py, raw resistance,
10-point means before gas-on and before gas-off): signed change s = (R_air - R_gas) / R_air
(> 0: resistance fell). Per sensor: fraction of the 30 cycles with s > 0, and median s.
Classes: 'consistent' (>= 80% of cycles fall), 'reversed' (<= 20%), 'inconsistent' (otherwise).
'No detectable response': inconsistent sign, or median local SNR < 3 (labels_v2.csv).
Checks on the 56 development sensors:
  rho(log R_air, log label) as in the paper; rho(log R_air, signed median change) (ranks keep the
  sign); rho with every 'no detectable response' sensor tied at the lowest label value; screen AUC
  for non-responders defined as response < 1% OR no detectable response; and whether non-responders
  with a consistent sign (a small but real, diluted response) also have low R_air.
Writes data/derived/sign_label.csv and sign_label.json.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu, spearmanr

from compute_snr import extract_params_from_filename, find_cycles, load_sheet_data, read_first_sheet
from dev_eval import locked
from qc_metrics import auc

ROOT = Path(__file__).resolve().parents[2]


def main():
    lab = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/derived/labels_v2.csv"))}
    ra = {r["sensor"]: float(r["r_air_median"]) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    rows = []
    for path in sorted((ROOT / "data/new_sensor_responses").glob("*.xlsx")):
        con, cur, tim = extract_params_from_filename(path.name).split("_")
        s = f"{con[:-1]}M-{cur[:-1]}uA-{tim[:-1]}s"
        sig = load_sheet_data(read_first_sheet(path))
        raw = sig[:, 1]
        ch = []
        for on, off in find_cycles(sig):
            r_air, r_gas = raw[max(0, on - 10):on].mean(), raw[off - 10:off].mean()
            ch.append((r_air - r_gas) / r_air)
        ch = np.array(ch)
        frac = float(np.mean(ch > 0))
        cls = "consistent" if frac >= 0.8 else "reversed" if frac <= 0.2 else "inconsistent"
        snr = float(lab[s]["snr_local_median"])
        rows.append(dict(sensor=s, split="locked" if s in test else "excluded" if lab[s]["excluded_as_outlier"] == "True" else "development",
                         frac_falls=round(frac, 3), median_signed_change=float(np.median(ch)), sign_class=cls,
                         snr_local_median=snr, no_detectable=cls != "consistent" or snr < 3,
                         rel_resp=float(lab[s]["rel_resp_median"]), r_air=ra[s]))
    with open(ROOT / "data/derived/sign_label.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    dev = [r for r in rows if r["split"] == "development"]
    lr = np.log10([r["r_air"] for r in dev])
    y = np.log10([r["rel_resp"] for r in dev])
    nd = np.array([r["no_detectable"] for r in dev])
    low = np.array([r["rel_resp"] < 0.01 for r in dev])
    cons = np.array([r["sign_class"] == "consistent" for r in dev])
    out = dict(n_dev=len(dev))
    out["classes_all_dev"] = {c: int(sum(r["sign_class"] == c for r in dev)) for c in ("consistent", "reversed", "inconsistent")}
    out["among_response_below_1pct"] = dict(n=int(low.sum()), consistent=int((low & cons).sum()),
                                            no_detectable=int((low & nd).sum()))
    out["among_response_at_least_1pct"] = dict(n=int((~low).sum()), consistent=int((~low & cons).sum()),
                                               no_detectable=int((~low & nd).sum()))
    out["rho_r_air_vs_label_paper"] = round(float(spearmanr(lr, y).statistic), 3)
    out["rho_r_air_vs_signed_median_change"] = round(float(spearmanr(lr, [r["median_signed_change"] for r in dev]).statistic), 3)
    yt = y.copy()
    yt[nd] = y.min()
    out["rho_r_air_vs_label_no_detectable_tied"] = round(float(spearmanr(lr, yt).statistic), 3)
    out["rho_r_air_vs_label_detectable_only"] = dict(n=int((~nd).sum()), rho=round(float(spearmanr(lr[~nd], y[~nd]).statistic), 3))
    dead2 = low | nd
    out["screen_auc_nonresponder_or_no_detectable"] = dict(n_flag=int(dead2.sum()), auc=round(float(auc(lr, dead2)), 3),
                                                           caught_at_10k=int(((lr < 4) & dead2).sum()),
                                                           rejected_good_at_10k=int(((lr < 4) & ~dead2).sum()))
    a, b = lr[low & cons], lr[low & ~cons]
    out["below_1pct_consistent_vs_not_log_r_air"] = dict(median_consistent=round(float(np.median(a)), 2) if len(a) else None,
                                                         median_not=round(float(np.median(b)), 2) if len(b) else None,
                                                         n=[int(len(a)), int(len(b))],
                                                         mannwhitney_p=round(float(mannwhitneyu(a, b).pvalue), 3) if len(a) and len(b) else None)
    json.dump(out, open(ROOT / "data/derived/sign_label.json", "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
