"""E15: conduction type from the heating logs (existing records; descriptive, no model, no scoring).

Some original workbooks (data/new_sensor_responses/*.xlsx) hold a 'Heating process 1' sheet:
resistance (column C) and temperature (column G) logged while the sensor is first heated from
room temperature to 350 C, before any gas test. A semiconducting oxide loses resistance on heating
(negative temperature coefficient); a metal or degenerate conductor gains it (positive).
Per sensor, on the heating ramp (from the start to the first time the maximum temperature is
reached), between 100 and 290 C (above water desorption, below the plateau):
  rho_TR      Spearman rho between temperature and resistance
  ratio       R(150 C) / R(280 C), medians within +-5 C (> 1: falls on heating)
  type        'rises' (rho_TR > 0.5), 'falls' (rho_TR < -0.5), otherwise 'mixed'
Over-range readings (resistance >= 1e11 or <= 0) are dropped.
Sensors are listed with baseline resistance and response; locked sensors are included (this is a
physical description, not a prediction) and flagged. Writes data/derived/heating_rt.csv and
heating_rt.json (counts by resistance band and by response class).
"""
import csv
import json
import re
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from dev_eval import locked
from replicate_labels import sheets

ROOT = Path(__file__).resolve().parents[2]


def sensor_of(path):
    s = path.stem.replace("ZnCl2", "")
    m = re.match(r"([\d.]+)M-([\d.]+)(mA|uA)-(\d+)s", s)
    cur = float(m.group(2)) * (1000 if m.group(3) == "mA" else 1)
    return f"{float(m.group(1)):g}M-{cur:g}uA-{m.group(4)}s"


def curve(path):
    sh = sheets(path)
    if "Heating process 1" not in sh:
        return None
    rows = [r for r in sh["Heating process 1"][1:] if 2 in r and 6 in r and isinstance(r[2], float)
            and 0 < r[2] < 1e11]          # drop over-range readings of the meter (logged as ~1e30 or <= 0)
    T, R = np.array([r[6] for r in rows]), np.array([r[2] for r in rows])
    end = int(np.argmax(T >= T.max() - 0.5)) + 1
    return T[:end], R[:end]


def main():
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    lab = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/derived/labels_v2.csv"))}
    ra = {r["sensor"]: float(r["r_air_median"]) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    out, curves = [], {}
    for f in sorted((ROOT / "data/new_sensor_responses").glob("*.xlsx")):
        c = curve(f)
        if c is None:
            continue
        s = sensor_of(f)
        T, R = c
        m = (T > 100) & (T < 290)
        if m.sum() < 20:
            continue
        rho = spearmanr(T[m], R[m]).statistic
        med = lambda t0: np.median(R[np.abs(T - t0) < 5])
        typ = "rises" if rho > 0.5 else "falls" if rho < -0.5 else "mixed"
        resp = float(lab[s]["rel_resp_median"])
        out.append(dict(sensor=s, split="locked" if s in test else "excluded" if lab[s]["excluded_as_outlier"] == "True" else "development",
                        n_points=int(m.sum()), rho_TR=round(float(rho), 3), ratio_150_280=round(float(med(150) / med(280)), 3),
                        type=typ, r_air=ra[s], response=resp))
        curves[s] = (T, R)
    with open(ROOT / "data/derived/heating_rt.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    summ = {"n_sensors": len(out), "n_workbooks": len(list((ROOT / "data/new_sensor_responses").glob("*.xlsx")))}
    for name, sel in (("R_air < 1 kOhm", lambda o: o["r_air"] < 1e3), ("1-10 kOhm", lambda o: 1e3 <= o["r_air"] < 1e4),
                      ("10 kOhm - 1 MOhm", lambda o: 1e4 <= o["r_air"] < 1e6), (">= 1 MOhm", lambda o: o["r_air"] >= 1e6),
                      ("response < 1%", lambda o: o["response"] < 0.01), ("response >= 1%", lambda o: o["response"] >= 0.01)):
        g = [o for o in out if sel(o)]
        summ[name] = {t: sum(o["type"] == t for o in g) for t in ("rises", "falls", "mixed")}
    x = np.log10([o["r_air"] for o in out])
    from scipy.stats import fisher_exact
    r = np.array([o["rho_TR"] for o in out])
    obs = spearmanr(r, x).statistic
    rng = np.random.default_rng(0)
    null = np.array([spearmanr(r, rng.permutation(x)).statistic for _ in range(10000)])
    summ["rho_TR_vs_log_r_air"] = dict(rho=round(float(obs), 3), p_two_sided=round(float((1 + (np.abs(null) >= abs(obs)).sum()) / 10001), 4))
    dead = np.array([o["response"] < 0.01 for o in out])
    rises = np.array([o["type"] == "rises" for o in out])
    summ["rises_vs_nonresponder_fisher_p"] = round(float(fisher_exact([[int((rises & dead).sum()), int((rises & ~dead).sum())],
                                                                       [int((~rises & dead).sum()), int((~rises & ~dead).sum())]]).pvalue), 4)
    json.dump(summ, open(ROOT / "data/derived/heating_rt.json", "w"), indent=1)
    print(json.dumps(summ, indent=1))
    np.save(ROOT / "data/derived/heating_rt_curves.npy", {k: np.stack(v) for k, v in curves.items()}, allow_pickle=True)


if __name__ == "__main__":
    main()
