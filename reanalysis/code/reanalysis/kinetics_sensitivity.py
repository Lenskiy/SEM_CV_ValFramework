"""E11 robustness: do the kinetics gates fail whatever reasonable choices are made?

Grid over the analysis choices of kinetics.py:
  smoothing   running median of 1 (none), 3, 7, 15 samples
  window      R0 / R1 averaging window of 5, 10, 20 s
  definition  'half': time to 50% of the change (response) / back within 50% (recovery)
              '10-90': time from 10% to 90% of the change (response) / 90% to 10% remaining
                       (recovery); NaN if a level is not reached
Same working sensors, validity rule and gates (G1 ICC >= 0.8; G2 median / p10 >= 2, p10 >= 6 s).
Also plots one 0.5 ppm pulse (second staircase) for the sensors at the 10th, 50th and 90th
percentile of the default response half-time, with the measured times marked.
Steady state: per pulse, the share of the total change that happens in the last 20 s of the
100 s exposure (default choices); a sensor whose median share exceeds 10% has not settled.
Writes data/derived/kinetics_sensitivity.csv and paper_v2/figures/kinetics_pulses.{pdf,png}.
"""
import csv
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter

from compute_snr import extract_params_from_filename, find_cycles, load_sheet_data, read_first_sheet
from kinetics import MIN_RESP, icc1

ROOT = Path(__file__).resolve().parents[2]
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9aa0a6"


def crossing(seg, level, below=False):
    m = seg <= level if below else seg >= level
    return int(np.argmax(m)) if m.any() else None


def pulse(t, r, on, off, nxt, win, defn):
    b = (t >= t[on] - win) & (t < t[on])
    e = (t > t[off] - win) & (t <= t[off])
    if b.sum() < 3 or e.sum() < 3:
        return None
    r0, r1 = r[b].mean(), r[e].mean()
    dr = r1 - r0
    if abs(dr) <= 5 * max(r[b].std(), 1e-12):
        return dict(ok=False)
    s = np.sign(dr) * (r - r0) / abs(dr)          # 0 at baseline, 1 at end of exposure
    up, down = s[on:off + 1], s[off:nxt]
    if defn == "half":
        i, j = crossing(up, 0.5), crossing(down, 0.5, True)
        resp = t[on + i] - t[on] if i is not None else np.nan
        rec = t[off + j] - t[off] if j is not None else np.nan
    else:
        i1, i9 = crossing(up, 0.1), crossing(up, 0.9)
        j9, j1 = crossing(down, 0.9, True), crossing(down, 0.1, True)
        resp = t[on + i9] - t[on + i1] if None not in (i1, i9) else np.nan
        rec = t[off + j1] - t[off + j9] if None not in (j9, j1) else np.nan
    return dict(ok=True, resp=resp, rec=rec, r0=r0, r1=r1)


def load():
    lab = {r["sensor"]: float(r["rel_resp_median"]) for r in csv.DictReader(open(ROOT / "data/derived/labels_v2.csv"))}
    data = {}
    for path in sorted((ROOT / "data/new_sensor_responses").glob("*.xlsx")):
        con, cur, tim = extract_params_from_filename(path.name).split("_")
        s = f"{con[:-1]}M-{cur[:-1]}uA-{tim[:-1]}s"
        if lab.get(s, 0) >= MIN_RESP:
            d = load_sheet_data(read_first_sheet(path))
            d = d[~np.isnan(d[:, :2]).any(1)]
            data[s] = (d, find_cycles(d))
    return data


def per_repeat(data, smooth, win, defn):
    out = {}
    for s, (d, cyc) in data.items():
        t, r = d[:, 0], median_filter(d[:, 1], smooth, mode="nearest") if smooth > 1 else d[:, 1]
        P = [pulse(t, r, on, off, cyc[k + 1][0] if k + 1 < len(cyc) else len(t), win, defn)
             for k, (on, off) in enumerate(cyc)]
        out[s] = {m: [np.nanmedian([p[m] for p in P[rep * 10:rep * 10 + 10] if p and p["ok"]] or [np.nan])
                      for rep in range(3)] for m in ("resp", "rec")}
    return out


def gates(rep):
    g = {}
    for m in ("resp", "rec"):
        M = np.array([rep[s][m] for s in rep], float)
        icc, n = icc1(np.log10(M))
        sens = np.nanmedian(M, 1)
        med, p10 = np.nanmedian(sens), np.nanpercentile(sens, 10)
        g[m] = dict(icc=round(icc, 3), n=n, p10=round(float(p10), 1), median=round(float(med), 1),
                    p90=round(float(np.nanpercentile(sens, 90)), 1), spread=round(float(med / p10), 2),
                    G1=icc >= 0.8, G2=bool(med / p10 >= 2 and p10 >= 6))
    return g


def plot(data, rep):
    t50 = {s: np.nanmedian(v["resp"]) for s, v in rep.items()}
    order = sorted(t50, key=t50.get)
    picks = [order[int(q * (len(order) - 1))] for q in (0.1, 0.5, 0.9)]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.4), sharey=True)
    for ax, s, lab in zip(axes, picks, ("10th percentile", "median", "90th percentile")):
        d, cyc = data[s]
        t, raw = d[:, 0], d[:, 1]
        r = median_filter(raw, 7, mode="nearest")
        k = 14                                                        # 0.5 ppm, second staircase
        on, off, nxt = cyc[k][0], cyc[k][1], cyc[k + 1][0]
        p = pulse(t, r, on, off, nxt, 10, "half")
        sl = slice(np.searchsorted(t, t[on] - 40), nxt)
        tt = t[sl] - t[on]
        norm = lambda x: (x - p["r0"]) / p["r0"] * 100
        ax.axvspan(0, t[off] - t[on], color=BLUE, alpha=0.08, lw=0)
        ax.plot(tt, norm(raw[sl]), color=GREY, lw=0.6, label="raw")
        ax.plot(tt, norm(r[sl]), color=BLUE, lw=1.5, label="smoothed (7-sample median)")
        half = norm((p["r0"] + p["r1"]) / 2)
        ax.axhline(half, color=GREY, lw=0.6, ls="--")
        for x in (p["resp"], t[off] - t[on] + p["rec"]):
            if np.isfinite(x):
                ax.plot([x], [half], "o", ms=5, color=ORANGE, mec="white", mew=1, zorder=5)
        ax.set_title(f"{lab}: {s}\nresponse {p['resp']:.0f} s, recovery {p['rec']:.0f} s", fontsize=7)
        ax.set_xlabel("time from gas-on (s)", fontsize=7)
        ax.tick_params(labelsize=6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("resistance change (%)", fontsize=7)
    axes[2].legend(*axes[0].get_legend_handles_labels(), fontsize=6, frameon=False, loc="lower right")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(ROOT / f"paper_v2/figures/kinetics_pulses.{ext}", dpi=200)


def settling(data):
    share = []
    for s, (d, cyc) in data.items():
        t, r = d[:, 0], median_filter(d[:, 1], 7, mode="nearest")
        v = []
        for on, off in cyc:
            p = pulse(t, r, on, off, off + 1, 10, "half")
            if p and p["ok"]:
                late = r[(t >= t[off] - 25) & (t < t[off] - 15)].mean()
                v.append((p["r1"] - late) / (p["r1"] - p["r0"]))
        share.append(np.median(v))
    share = np.array(share)
    print(f"steady state: median late-change share {np.median(share):.2f}; "
          f"{(share > 0.1).sum()}/{len(share)} sensors still changing (> 10% in the last 20 s)")


def main():
    data = load()
    settling(data)
    rows = []
    for smooth, win, defn in itertools.product((1, 3, 7, 15), (5, 10, 20), ("half", "10-90")):
        g = gates(per_repeat(data, smooth, win, defn))
        for m, v in g.items():
            rows.append(dict(smooth=smooth, window_s=win, definition=defn, measure=m, **v))
            print(f"smooth {smooth:2d} win {win:2d} {defn:5s} {m:4s} ICC {v['icc']:.2f} "
                  f"p10/median/p90 {v['p10']}/{v['median']}/{v['p90']} spread {v['spread']} G1 {v['G1']} G2 {v['G2']}")
    with open(ROOT / "data/derived/kinetics_sensitivity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    plot(data, per_repeat(data, 7, 10, "half"))


if __name__ == "__main__":
    main()
