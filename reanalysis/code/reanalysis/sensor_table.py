"""Per-sensor table for the Supplementary Information (Table S2) and as a data file.

One row per sensor (72): recipe, split (development / locked / excluded), median relative response,
baseline resistance (median before each pulse), metallic-zinc equivalent thickness h, gap obscuration,
direction of the resistance change (sign_label.py), resistance trend on first heating where logged
(heating_rt.py), number of 500x images. Writes data/derived/sensor_table.csv and
paper_v2/table_s2.tex (the longtable, input by supplementary.tex).
"""
import csv
from pathlib import Path

import common as C
from dev_eval import locked

ROOT = Path(__file__).resolve().parents[2]
HEAD = r"""\setlength{\tabcolsep}{4pt}
\begin{footnotesize}
\begin{longtable}{lllllllrrl}
\caption{\textbf{Per-sensor data.} Recipe (ZnCl$_2$ concentration in M, current, time in s); split (dev: development, locked: locked test, excl.: excluded); median relative response over 30 cycles (\%); baseline resistance (median before each acetone pulse); metallic-zinc equivalent thickness $h$ ($\mu$m, an upper bound); gap obscuration (\% of image area); share of cycles in which the resistance fell (\%); resistance trend on first heating where logged (Fig.~S3). The same data are in \texttt{sensor\_table.csv} with the code.}\label{tab:sensors}\\
\toprule
M & current & s & split & resp.\ \% & resistance & $h$ & gaps \% & falls \% & heating \\
\midrule
\endfirsthead
\toprule
M & current & s & split & resp.\ \% & resistance & $h$ & gaps \% & falls \% & heating \\
\midrule
\endhead
"""
TAIL = r"""
\bottomrule
\end{longtable}
\end{footnotesize}"""


def rd(name, key="sensor"):
    return {r[key]: r for r in csv.DictReader(open(ROOT / name))}


def ohm(v):
    return f"{v / 1e6:.3g}~M$\\Omega$" if v >= 1e6 else f"{v / 1e3:.3g}~k$\\Omega$" if v >= 1e3 else f"{v:.3g}~$\\Omega$"


def main():
    lab, ra = rd("data/derived/labels_v2.csv"), rd("data/derived/r_air.csv")
    reg, br = rd("data/derived/regime_sensors.csv"), rd("data/derived/bridging.csv")
    sg, ht = rd("data/derived/sign_label.csv"), rd("data/derived/heating_rt.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    rows = []
    for s in sorted(lab, key=lambda s: C.parse_sensor_key(s)):
        c, i, t = C.parse_sensor_key(s)
        split = "locked" if s in test else "excluded" if lab[s]["excluded_as_outlier"] == "True" else "development"
        rows.append(dict(sensor=s, zncl2_M=c, current_uA=i, time_s=t, split=split,
                         response_pct=round(100 * float(lab[s]["rel_resp_median"]), 4),
                         r_air_ohm=round(float(ra[s]["r_air_median"]), 1), h_um=round(float(reg[s]["h_0.048"]), 4),
                         gap_obscured_pct=round(100 * float(br[s]["gap_loss"]), 2),
                         falls_in_cycles_pct=round(100 * float(sg[s]["frac_falls"]), 0),
                         heating_trend=ht[s]["type"] if s in ht else "", n_images_500x=int(br[s]["n_images"])))
    with open(ROOT / "data/derived/sensor_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    sp = {"development": "dev", "locked": "locked", "excluded": "excl."}
    lines = []
    for r in rows:
        cur = f"{r['current_uA'] / 1000:g}~mA" if r["current_uA"] >= 1000 else f"{r['current_uA']:g}~$\\mu$A"
        lines.append(f"{r['zncl2_M']:g} & {cur} & {r['time_s']:g} & {sp[r['split']]} & {r['response_pct']:.3g} & "
                     f"{ohm(r['r_air_ohm'])} & {r['h_um']:.3g} & {r['gap_obscured_pct']:.1f} & "
                     f"{r['falls_in_cycles_pct']:.0f} & {r['heating_trend'] or '--'} \\\\")
    (ROOT / "paper_v2/table_s2.tex").write_text(HEAD + "\n".join(lines) + "\n" + TAIL)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
