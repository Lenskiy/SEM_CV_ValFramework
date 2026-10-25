"""Labels for the replicate sensors (Sensor-optimize-new-replicate, tested March 2026).

Their workbooks hold a heating sheet plus a test sheet per sensor; the test sheet
has the original layout (loop, time, resistance, ..., acetone ppm in column 6),
so the same per-cycle code as labels_v2 applies. Writes
data/derived/labels_replicate.csv and prints the replicate next to the original.
Usage: python replicate_labels.py <Sensor Responses dir> <labels_v2.csv> <out dir>
"""
import csv
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import numpy as np

from compute_snr import NS, load_sheet_data
from labels_v2 import cycle_values


def sheets(path):
    """{sheet name: rows as {column index: value}} for every worksheet."""
    z = zipfile.ZipFile(path)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(NS + "si"):
            shared.append("".join(t.text or "" for t in si.iter(NS + "t")))
    names = [s.get("name") for s in ET.fromstring(z.read("xl/workbook.xml")).iter(NS + "sheet")]
    out = {}
    for k, name in enumerate(names, 1):
        rows = []
        for r in ET.fromstring(z.read(f"xl/worksheets/sheet{k}.xml")).iter(NS + "row"):
            d = {}
            for c in r.findall(NS + "c"):
                v = c.find(NS + "v")
                if v is None:
                    continue
                letters = re.match(r"[A-Z]+", c.get("r")).group()
                col = sum((ord(ch) - 64) * 26 ** i for i, ch in enumerate(reversed(letters))) - 1
                d[col] = shared[int(v.text)] if c.get("t") == "s" else float(v.text)
            rows.append(d)
        out[name] = rows
    return out


def sensor_key(sheet_name):
    """'0.2ZnO-5mA-60s' -> '0.2M-5000uA-60s'."""
    m = re.match(r"([\d.]+)ZnO-([\d.]+)([mu])A-(\d+)s", sheet_name)
    cur = float(m.group(2)) * (1000 if m.group(3) == "m" else 1)
    return f"{m.group(1)}M-{cur:g}uA-{m.group(4)}s"


def main(in_dir, labels_csv, out_dir):
    orig = {r["sensor"]: r for r in csv.DictReader(open(labels_csv))}
    rows = []
    for path in sorted(Path(in_dir).glob("*.xlsx")):
        test = "2nd" if "2nd" in path.name.lower() else "1st"
        for name, data in sheets(path).items():
            if not str(data[0].get(0, "")).lower().startswith("loop"):
                continue
            cyc = cycle_values(load_sheet_data(data))
            s = sensor_key(name)
            rec = dict(sensor=s, file=path.name, test=test, n_cycles=len(cyc))
            for v in ("snr_pub", "snr_local", "rel_resp"):
                rec[f"{v}_median"] = float(np.nanmedian([c[v] for c in cyc]))
            rows.append(rec)
            o = orig.get(s, {})
            print(f"{s:18s} {test} cycles {len(cyc):2d}  rel_resp {rec['rel_resp_median']:.4f} (original "
                  f"{float(o.get('rel_resp_median', 'nan')):.4f})  snr_local {rec['snr_local_median']:6.2f} "
                  f"(original {float(o.get('snr_local_median', 'nan')):6.2f})")
    out = Path(out_dir) / "labels_replicate.csv"
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:4])
