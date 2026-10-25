"""E8: electrochemical regime of the recipe grid (no images; all sensors with baseline resistance).

Hypothesis: without O2 bubbling, the O2-reduction current that precipitates ZnO is capped at
tens of uA/cm2, far below every applied current, so the excess current reduces Zn2+ to metal
(or water to H2 when Zn2+ is depleted). Thick metallic deposits that survive the 400 C anneal
(Zn melts at 420 C) bridge the 5 um gaps and give near-metallic baseline resistance.

Physical variable, fixed before looking at its fit: the metallic-Zn equivalent thickness
  h = Q_Zn * M_Zn / (2 F rho_Zn A)
with Q_Zn the Zn2+-reduction charge, all of the applied current until the Sand time of Zn2+
(planar semi-infinite diffusion, unstirred) and the diffusion-limited current after it:
  tau = pi D (2 F C)^2 / (4 j^2);  Q_Zn/A = j min(t, tau) + 4 F C sqrt(D/pi) (sqrt t - sqrt tau)_+
Constants at 70 C: D_Zn2+ 1.8e-5 cm2/s (0.70e-5 at 25 C scaled by T/viscosity), D_O2 5e-5 cm2/s,
O2 at air saturation 0.12 mM. Area A: Pt half of the 3.5 mm-diameter cell (0.048 cm2);
a range 0.02-0.1 cm2 is reported.

Tests
  1  Spearman rho of log R_air (and log response) with current, concentration, time, charge Q
     and h; 10,000-permutation p for each; bootstrap 95% interval of rho(h) - rho(Q)
  2  linear model log R_air ~ log C + log I + log t + log C x log I; permutation p of the
     interaction (P1: the current effect is stronger at higher concentration)
  3  non-responders (response < 1%) against a single threshold on h, leave-one-out
     (threshold chosen on the rest), balanced accuracy; same for Q
Writes data/derived/regime.json and data/derived/regime_sensors.csv.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
F, M_ZN, RHO_ZN = 96485.0, 65.38, 7.14
D_ZN, D_O2, C_O2 = 1.8e-5, 5e-5, 0.12e-6          # cm2/s, cm2/s, mol/cm3
A0, AREAS = 0.048, (0.02, 0.048, 0.1)
NONRESP = 0.01
rng = np.random.default_rng(0)


def h_zn(c_molar, i_amp, t, area):
    c, j = c_molar * 1e-3, i_amp / area                # mol/cm3, A/cm2
    tau = math.pi * D_ZN * (2 * F * c) ** 2 / (4 * j * j)
    q = j * min(t, tau) + (4 * F * c * math.sqrt(D_ZN / math.pi) * (math.sqrt(t) - math.sqrt(tau)) if t > tau else 0)
    return q * M_ZN / (2 * F * RHO_ZN) * 1e4          # um


def perm_rho(x, y, n=10000):
    r = spearmanr(x, y).statistic
    null = np.array([spearmanr(x, rng.permutation(y)).statistic for _ in range(n)])
    return round(float(r), 3), round(float((1 + (np.abs(null) >= abs(r)).sum()) / (n + 1)), 4)


def loo_threshold(x, dead):
    hit = []
    for k in range(len(x)):
        m = np.arange(len(x)) != k
        best, thr = -1, None
        for t in np.unique(x[m]):
            p = x[m] >= t
            ba = (p[dead[m]].mean() + (~p[~dead[m]]).mean()) / 2
            if ba > best:
                best, thr = ba, t
        hit.append(x[k] >= thr)
    hit = np.array(hit)
    return round(float((hit[dead].mean() + (~hit[~dead]).mean()) / 2), 3)


def main():
    ra = {r["sensor"]: float(r["r_air_median"]) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    rows = []
    for r in csv.DictReader(open(ROOT / "data/derived/labels_v2.csv")):
        s = r["sensor"]
        if s not in ra:
            continue
        c, i, t = s.split("-")
        c, i, t = float(c[:-1]), float(i[:-2]) * 1e-6, float(t[:-1])
        rows.append(dict(sensor=s, C=c, I=i, t=t, Q=i * t, resp=float(r["rel_resp_median"]), r_air=ra[s],
                         **{f"h_{a}": h_zn(c, i, t, a) for a in AREAS}))
    lr = np.log10([r["r_air"] for r in rows])
    ly = np.log10([max(r["resp"], 1e-5) for r in rows])
    dead = np.array([r["resp"] < NONRESP for r in rows])
    V = {k: np.array([r[k] for r in rows]) for k in ("I", "C", "t", "Q", *[f"h_{a}" for a in AREAS])}
    out = dict(n=len(rows), j_lim_O2_uA_cm2_at_30s=round(4 * F * C_O2 * math.sqrt(D_O2 / (math.pi * 30)) * 1e6, 1),
               applied_j_mA_cm2=[round(i / A0 * 1e3, 1) for i in (100e-6, 5e-3)])
    out["1_spearman"] = {k: dict(log_r_air=perm_rho(v, lr), log_response=perm_rho(v, ly)) for k, v in V.items()}
    hq = V[f"h_{A0}"]
    d = []
    for _ in range(2000):
        b = rng.integers(0, len(lr), len(lr))
        d.append(spearmanr(hq[b], lr[b]).statistic - spearmanr(V["Q"][b], lr[b]).statistic)
    out["1_rho_h_minus_rho_Q_ci95"] = [round(float(np.percentile(d, q)), 3) for q in (2.5, 97.5)]

    Xd = np.column_stack([np.ones(len(lr)), np.log10(V["C"]), np.log10(V["I"]), np.log10(V["t"])])
    inter = np.log10(V["C"]) * np.log10(V["I"])
    inter = inter - Xd @ np.linalg.lstsq(Xd, inter, rcond=None)[0]
    X = np.column_stack([Xd, inter])
    b = np.linalg.lstsq(X, lr, rcond=None)[0]
    null = [np.linalg.lstsq(X, np.concatenate([Xd @ np.linalg.lstsq(Xd, lr, rcond=None)[0]]) +
                            rng.permutation(lr - Xd @ np.linalg.lstsq(Xd, lr, rcond=None)[0]), rcond=None)[0][4]
            for _ in range(10000)]
    out["2_interaction"] = dict(coef_logC=round(float(b[1]), 3), coef_logI=round(float(b[2]), 3),
                                coef_logt=round(float(b[3]), 3), coef_logCxlogI=round(float(b[4]), 3),
                                perm_p=round(float((1 + (np.abs(null) >= abs(b[4])).sum()) / 10001), 4))
    out["3_nonresponder_loo_balanced_accuracy"] = {k: loo_threshold(V[k], dead) for k in ("Q", "I", f"h_{A0}")}
    out["3_n_nonresponders"] = int(dead.sum())
    print(json.dumps(out, indent=1))
    json.dump(out, open(ROOT / "data/derived/regime.json", "w"), indent=1)
    with open(ROOT / "data/derived/regime_sensors.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
