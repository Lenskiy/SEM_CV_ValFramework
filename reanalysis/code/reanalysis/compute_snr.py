"""Recompute per-cycle and per-sensor SNR from the sensor-response workbooks.

Faithful port of the original notebook (SEM_CV_ValFramework,
Interpreting_morphology.ipynb, cells 4-9): same cycle detection, AsLS
baseline removal, response/baseline windows and per-sensor median. Workbooks
are read with the standard library instead of pandas.

Usage:
    python compute_snr.py <dir with *.xlsx> <output dir>

Writes snr_per_cycle.csv and snr_per_sensor.csv to the output dir.
"""
import csv
import math
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np

from remove_baseline import remove_baseline

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def _col(ref):
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_first_sheet(path):
    """Rows of the first worksheet as dicts {column index: value}.

    Numbers become floats, shared/inline strings stay strings; empty cells are
    absent (pandas would give NaN).
    """
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    rows = []
    for r in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "row"):
        d = {}
        for c in r.findall(NS + "c"):
            v = c.find(NS + "v")
            if v is None:
                continue
            t = c.get("t")
            if t == "s":
                d[_col(c.get("r"))] = shared[int(v.text)]
            elif t in ("str", "inlineStr"):
                d[_col(c.get("r"))] = v.text
            else:
                d[_col(c.get("r"))] = float(v.text)
        rows.append(d)
    return rows


def extract_params_from_filename(filename):
    """Notebook logic: '0.1MZnCl2-1mA-60s.xlsx' -> '0.1M_1000u_60s'."""
    numbers = re.findall(r"\d+\.?\d*[a-zA-Z]", filename)
    numbers = [str(int(i[0]) * 1000) + "u" if i.endswith("m") else i for i in numbers]
    grouped = defaultdict(list)
    for m in numbers:
        grouped[m[-1]].append(m)
    return f"{grouped['M'][0]}_{grouped['u'][0]}_{grouped['s'][0]}"


def load_sheet_data(rows):
    """(time, resistance, concentration) per data row, as the notebook builds it.

    pandas consumes the first row as the header. A row is kept unless its
    concentration cell holds text; empty cells become NaN.
    """
    data = []
    for d in rows[1:]:
        conc = d.get(6, math.nan)
        if isinstance(conc, str):
            continue
        data.append((d.get(1, math.nan), d.get(2, math.nan), conc))
    return np.array(data, dtype=float)


def find_cycles(data, min_len=50):
    transitions = []
    for i in range(1, len(data)):
        pc, cc = data[i - 1][2], data[i][2]
        if pc == 0 and cc > 0:
            transitions.append(("ON", i))
        elif pc > 0 and cc == 0:
            transitions.append(("OFF", i))
    cycles, i = [], 0
    while i < len(transitions) - 1:
        if transitions[i][0] == "ON" and transitions[i + 1][0] == "OFF":
            on, off = transitions[i][1], transitions[i + 1][1]
            if off - on >= min_len:
                cycles.append((on, off))
            i += 2
        else:
            i += 1
    return cycles


def sensor_cycles(signal):
    """Per-cycle values exactly as notebook cell 8."""
    cycles = find_cycles(signal)
    flat_signal, _ = remove_baseline(signal[:, 1], method="als")
    out = []
    baseline_R = []
    for c_num, (on, off) in enumerate(cycles, 1):
        if c_num == 1:
            bl_start = max(0, on - 40)
            baseline_R = [signal[j][1] for j in range(10, bl_start) if signal[j][2] == 0]
            baseline_C = [flat_signal[j] for j in range(10, bl_start) if signal[j][2] == 0]
        else:
            baseline_C = [flat_signal[j] for j in range(on, on + 40)]
        exp_C = [flat_signal[j] for j in range(on, off)]
        mean_conc = sum(signal[j][2] for j in range(on, off)) / (off - on)
        R_base = sum(baseline_C[:10]) / 10 if baseline_C else 0.0
        sigma_base = float(np.array(baseline_R).std()) if baseline_R else 0.0
        R_exp = sum(exp_C[-10:]) / 10
        delta_R = abs(R_exp - R_base)
        snr = delta_R / sigma_base if sigma_base > 0 else float("inf")
        out.append(dict(cycle_no=c_num, mean_conc=round(mean_conc, 4),
                        base_r=round(R_base, 4), response_r=round(R_exp, 4),
                        delta_r=round(delta_R, 4), sigma_base=round(sigma_base, 4),
                        snr_lin=round(snr, 4)))
    return out


def snr_class(snr):
    """Notebook snrToClass, including its gap at exactly 5, 10 and 20."""
    if snr < 5.0:
        return "Low"
    if 5.0 < snr < 10.0:
        return "Mid"
    if 10.0 < snr < 20.0:
        return "High"
    if snr > 20.0:
        return "Ultra-high"
    return "UNASSIGNED"


def main(in_dir, out_dir):
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cycle, per_sensor = [], []
    for path in sorted(in_dir.glob("*.xlsx")):
        key = extract_params_from_filename(path.name)
        con, cur, tim = key.split("_")
        signal = load_sheet_data(read_first_sheet(path))
        nan_rows = int(np.isnan(signal).any(axis=1).sum())
        cyc = sensor_cycles(signal)
        for c in cyc:
            per_cycle.append(dict(file=path.name, sol_con=con[:-1], current=cur[:-1], time=tim[:-1], **c))
        snrs = np.array([c["snr_lin"] for c in cyc])
        per_sensor.append(dict(file=path.name, sol_con=con[:-1], current=cur[:-1], time=tim[:-1],
                               sensor=f"{con[:-1]}M-{cur[:-1]}uA-{tim[:-1]}s",
                               n_cycles=len(cyc), nan_rows=nan_rows,
                               snr_median=round(float(np.median(snrs)), 5),
                               snr_mean=round(float(np.mean(snrs)), 5)))
    # The notebook drops the sensor with the largest median SNR as an outlier.
    top = max(per_sensor, key=lambda s: s["snr_median"])
    for s in per_sensor:
        s["category"] = snr_class(s["snr_median"])
        s["excluded_as_outlier"] = s is top

    for name, rows in (("snr_per_cycle.csv", per_cycle), ("snr_per_sensor.csv", per_sensor)):
        with open(out_dir / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    kept = [s for s in per_sensor if not s["excluded_as_outlier"]]
    counts = defaultdict(int)
    for s in kept:
        counts[s["category"]] += 1
    print(f"sensors {len(per_sensor)}, cycles {len(per_cycle)}, "
          f"cycles per sensor {sorted({s['n_cycles'] for s in per_sensor})}")
    print(f"excluded outlier: {top['sensor']} median SNR {top['snr_median']}")
    print("categories (71 kept):", dict(counts))
    print("RESULT compute_snr done")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
