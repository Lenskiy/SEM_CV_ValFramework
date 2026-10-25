"""E11 gate: are response kinetics and the dose-response exponent measurable on this rig?

Protocol in every file: three repeats of a 0.1-1.0 ppm acetone staircase (10 steps, 100 s on,
200 s off, ~1.2 s sampling, 0.5 L/min total flow). Resistance is smoothed with a 7-sample
running median (removes the few-second sawtooth). Per cycle:
  R0     mean of the 10 s before gas-on;  R1 mean of the last 10 s of exposure; dR = R1 - R0
  t50    time from gas-on until the signal first passes R0 + dR/2
  rec50  time from gas-off until the signal is back within dR/2 of R0 (NaN if not within 200 s)
  rec_end  fraction of dR recovered by the next gas-on
A cycle counts only if |dR| > 5 x the s.d. of the smoothed baseline window.
Per staircase repeat: median over its cycles of t50, rec50, rec_end, and the dose-response
exponent beta (slope of log |dR|/R0 on log concentration, >= 6 valid steps).
Working sensors only (rel_resp_median >= 1%).
Gates, fixed before looking:
  G1 repeatability: ICC(1) over the 3 staircase repeats (log scale for times) >= 0.8
  G2 not rig-limited (times only): median / 10th percentile across sensors >= 2 and the
     10th percentile >= 5 samples (6 s)
Writes data/derived/kinetics_sensors.csv and data/derived/kinetics_gate.json.
"""
import csv
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from compute_snr import extract_params_from_filename, find_cycles, load_sheet_data, read_first_sheet

ROOT = Path(__file__).resolve().parents[2]
MIN_RESP = 0.01


def cycle_metrics(t, r, conc, cycles):
    out = []
    for k, (on, off) in enumerate(cycles):
        nxt = cycles[k + 1][0] if k + 1 < len(cycles) else len(t)
        b = (t >= t[on] - 10) & (t < t[on])
        e = (t > t[off] - 10) & (t <= t[off])
        if b.sum() < 3 or e.sum() < 3:
            continue
        r0, r1 = r[b].mean(), r[e].mean()
        dr = r1 - r0
        ok = abs(dr) > 5 * max(r[b].std(), 1e-12)
        sgn = np.sign(dr)
        seg = sgn * (r[on:off + 1] - r0)
        i50 = np.argmax(seg >= abs(dr) / 2) if (seg >= abs(dr) / 2).any() else None
        rseg = sgn * (r[off:nxt] - r0)
        j50 = np.argmax(rseg <= abs(dr) / 2) if (rseg <= abs(dr) / 2).any() else None
        last = r[max(off, nxt - 8):nxt].mean()
        out.append(dict(k=k, conc=float(np.median(conc[on + 10:off])), ok=bool(ok), rel=abs(dr) / r0,
                        t50=t[on + i50] - t[on] if ok and i50 is not None else np.nan,
                        rec50=t[off + j50] - t[off] if ok and j50 is not None else np.nan,
                        rec_end=1 - sgn * (last - r0) / abs(dr) if ok else np.nan))
    return out


def icc1(M):
    M = M[~np.isnan(M).any(1)]
    n, k = M.shape
    msb = k * ((M.mean(1) - M.mean()) ** 2).sum() / (n - 1)
    msw = ((M - M.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return float((msb - msw) / (msb + (k - 1) * msw)), n


def main():
    lab = {r["sensor"]: float(r["rel_resp_median"]) for r in csv.DictReader(open(ROOT / "data/derived/labels_v2.csv"))}
    rows = []
    for path in sorted((ROOT / "data/new_sensor_responses").glob("*.xlsx")):
        con, cur, tim = extract_params_from_filename(path.name).split("_")
        s = f"{con[:-1]}M-{cur[:-1]}uA-{tim[:-1]}s"
        if lab.get(s, 0) < MIN_RESP:
            continue
        d = load_sheet_data(read_first_sheet(path))
        d = d[~np.isnan(d[:, :2]).any(1)]
        r = median_filter(d[:, 1], 7, mode="nearest")
        cyc = cycle_metrics(d[:, 0], r, np.nan_to_num(d[:, 2]), find_cycles(d))
        row = dict(sensor=s, n_valid=sum(c["ok"] for c in cyc))
        for rep in range(3):
            cs = [c for c in cyc if rep * 10 <= c["k"] < rep * 10 + 10]
            for m in ("t50", "rec50", "rec_end"):
                row[f"{m}_{rep}"] = float(np.nanmedian([c[m] for c in cs])) if any(np.isfinite(c[m]) for c in cs) else np.nan
            v = [c for c in cs if c["ok"] and c["conc"] > 0]
            row[f"beta_{rep}"] = float(np.polyfit(np.log10([c["conc"] for c in v]), np.log10([c["rel"] for c in v]), 1)[0]) if len(v) >= 6 else np.nan
        rows.append(row)
    with open(ROOT / "data/derived/kinetics_sensors.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    gate = dict(n_working=len(rows))
    for m, log in (("t50", True), ("rec50", True), ("rec_end", False), ("beta", False)):
        M = np.array([[r[f"{m}_{k}"] for k in range(3)] for r in rows], float)
        icc, n = icc1(np.log10(M) if log else M)
        sens = np.nanmedian(M, 1)
        g = dict(icc=round(icc, 3), n=n, median=round(float(np.nanmedian(sens)), 3),
                 p10=round(float(np.nanpercentile(sens, 10)), 3), p90=round(float(np.nanpercentile(sens, 90)), 3),
                 G1=icc >= 0.8)
        if m in ("t50", "rec50"):
            g["G2"] = bool(g["median"] / g["p10"] >= 2 and g["p10"] >= 6)
        gate[m] = g
        print(m, g)
    json.dump(gate, open(ROOT / "data/derived/kinetics_gate.json", "w"), indent=1)


if __name__ == "__main__":
    main()
