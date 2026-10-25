"""Label variants and their run-to-run repeatability (choose the target before any image model).

Each sensor was tested in 3 runs x 10 gas concentrations (cycles 1-10, 11-20, 21-30).
Variants per cycle:
  snr_pub     notebook SNR: ALS-flattened response / sigma of the raw pre-test baseline
  snr_local   same response / sigma of the flattened signal in the 40 points before
              that cycle's gas-on (noise measured next to each response)
  rel_resp    |R_gas - R_air| / R_air from the raw resistance (10-point means)
Per sensor and run: median over the 10 concentrations, plus the slope of the
response against concentration. Repeatability = ICC(1) of the log value across
the 3 runs, for one run and for the 3-run mean.

Writes data/derived/labels_v2.csv (one row per sensor) and prints the ICC table.
Usage: python labels_v2.py <xlsx dir> <out dir>
"""
import csv
import sys
from pathlib import Path

import numpy as np

from compute_snr import extract_params_from_filename, find_cycles, load_sheet_data, read_first_sheet
from remove_baseline import remove_baseline


def cycle_values(signal):
    cycles = find_cycles(signal)
    flat, _ = remove_baseline(signal[:, 1], method="als")
    raw = signal[:, 1]
    first_on = cycles[0][0]
    pre = [j for j in range(10, max(0, first_on - 40)) if signal[j][2] == 0]
    sigma_pub = float(np.std(raw[pre]))
    out = []
    for on, off in cycles:
        conc = float(signal[on:off, 2].mean())
        base_c = flat[on:on + 40]
        d_flat = abs(flat[off - 10:off].mean() - base_c[:10].mean())
        noise_local = float(np.std(flat[max(0, on - 40):on]))
        r_air = float(raw[max(0, on - 10):on].mean())
        r_gas = float(raw[off - 10:off].mean())
        out.append(dict(conc=conc,
                        snr_pub=d_flat / sigma_pub if sigma_pub > 0 else np.nan,
                        snr_local=d_flat / noise_local if noise_local > 0 else np.nan,
                        rel_resp=abs(r_gas - r_air) / abs(r_air) if r_air != 0 else np.nan))
    return out


def icc1(L):
    """ICC(1) for an (n subjects x k runs) array; returns (single run, mean of k)."""
    L = L[~np.isnan(L).any(1)]
    n, k = L.shape
    gm = L.mean()
    msb = k * ((L.mean(1) - gm) ** 2).sum() / (n - 1)
    msw = ((L - L.mean(1, keepdims=True)) ** 2).sum() / (n * (k - 1))
    return (msb - msw) / (msb + (k - 1) * msw), (msb - msw) / msb


def main(in_dir, out_dir):
    rows, per_run = [], {}
    for path in sorted(Path(in_dir).glob("*.xlsx")):
        con, cur, tim = extract_params_from_filename(path.name).split("_")
        sensor = f"{con[:-1]}M-{cur[:-1]}uA-{tim[:-1]}s"
        cyc = cycle_values(load_sheet_data(read_first_sheet(path)))
        runs = [cyc[k * 10:(k + 1) * 10] for k in range(3)]
        rec = dict(sensor=sensor, n_cycles=len(cyc))
        for v in ("snr_pub", "snr_local", "rel_resp"):
            med = [np.nanmedian([c[v] for c in r]) for r in runs]
            slope = [np.polyfit([c["conc"] for c in r], [c[v] for c in r], 1)[0] for r in runs]
            per_run.setdefault(f"{v}_median", []).append(med)
            per_run.setdefault(f"{v}_slope", []).append(slope)
            rec[f"{v}_median"] = float(np.nanmedian([c[v] for c in cyc]))
            rec[f"{v}_slope"] = float(np.mean(slope))
        rows.append(rec)

    # the notebook's outlier (largest published median SNR) is excluded from the ICC
    top = max(range(len(rows)), key=lambda i: rows[i]["snr_pub_median"])
    print(f"{'label':22s} ICC single run   ICC 3-run mean   (n = {len(rows) - 1}, outlier {rows[top]['sensor']} excluded)")
    for key, vals in per_run.items():
        A = np.delete(np.array(vals, dtype=float), top, axis=0)
        A = np.log(np.clip(A, 1e-9, None))
        s, m = icc1(A)
        print(f"{key:22s} {s:14.3f} {m:16.3f}")
    out = Path(out_dir) / "labels_v2.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]) + ["excluded_as_outlier"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**r, "excluded_as_outlier": i == top})
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
