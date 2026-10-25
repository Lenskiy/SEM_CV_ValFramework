"""The 2,500x analysis plan (Claude Doc "ZnO sensors: 2,500x analysis plan"), stage by stage.

Stages, run in this order; each reads what the previous one wrote in --out:
  qc       per-image acceptance: acquisition settings and manual exclusions; low
           focus scores are only flagged for review (a Laplacian score also falls
           with sparse morphology, so it cannot drop images by itself)
           -> qc_images.csv, qc_review.csv (labels are not read)
  measure  hm_morphology measures of every accepted image -> morph_images.csv,
           per-sensor medians -> morph_sensors.csv (labels are not read)
  session  edge-density shift for sensors imaged in two sessions -> session_check.json
  h1       H1 (one-sided Spearman) and partial Spearman given the recipe, on the
           development sensors outside the discovery set -> h1.json
  cv       M0-M4, repeated 5-fold sensor-held-out CV on all development sensors,
           permutation nulls with the max-over-models correction -> cv_results.csv, cv.json
  choose   the pre-registered selection rule -> preregistered_final.txt (+ sha256)
  final    one-shot locked-test score of the chosen model, plus the H1 replication
           -> appended to final_test_log.csv; refuses to run twice for one locked set

Models (log10 relative response): M0 recipe parameters, M1 edge density alone,
M2 recipe + edge density, M3 frozen ViT-B n448/pct sensor mean (PCA + ridge),
M4 = M3 with the recipe added after PCA.

The locked sensors are never read before 'final'. A dry run on other data uses
its own --locked / --discovery files and a scratch --out, so the real files are
not touched.
"""
import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import ndimage, stats

import common as C
from dev_eval import cv_score, locked, make_model, read_labels
from hm_morphology import MEASURES, measures

REF = dict(detector="sed", pixel_width_nm=53.958, high_voltage=-5.3)   # the Aug 2024 2,500x set
FOCUS_RATIO = 1 / 3            # flag for review an image whose Laplacian variance is below this x its sensor's median
SESSION_LIMIT = 0.25           # session shift allowed, as a fraction of the between-sensor s.d. ...
SESSION_ALPHA = 0.05           # ... unless it is also not distinguishable from image-sampling noise
NEAR = 0.05                    # selection rule: a simpler model within this rho of the best wins
ORDER = ["M1", "M2", "M3", "M4"]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def read_csv(path):
    return list(csv.DictReader(open(path)))


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def sensor_list(path):
    return locked(path) if path else []


# ------------------------------------------------------------------ stage: qc
def stage_qc(a):
    manual = {}
    if a.manual_exclusions:
        manual = {r["file"]: r["reason"] for r in read_csv(a.manual_exclusions)}
    rows = []
    for im in C.index_sem(a.images):
        m = im.meta
        problems = []
        if m["detector"].lower() != REF["detector"]:
            problems.append(f"detector {m['detector']}")
        px = float(m["pixel_width_nm"] or "nan")
        if not abs(px - REF["pixel_width_nm"]) <= 0.03 * REF["pixel_width_nm"]:
            problems.append(f"pixel {px:.2f} nm")
        hv = float(m["high_voltage"] or "nan")
        if not abs(hv - REF["high_voltage"]) <= 0.2:
            problems.append(f"voltage {hv}")
        wd = float(m["working_distance"] or "nan")
        if not 6 <= wd <= 8:
            problems.append(f"working distance {wd:.2f} mm")
        a_ = C.load_image_array(im.path).astype(np.float32)
        rows.append(dict(sensor=im.sensor, file=im.path.name, session=m["time"][:10],
                         focus=float(ndimage.laplace(ndimage.gaussian_filter(a_, 1)).var()),
                         settings="; ".join(problems), manual=manual.get(im.path.name, "")))
    for s in {r["sensor"] for r in rows}:
        mine = [r for r in rows if r["sensor"] == s]
        med = np.median([r["focus"] for r in mine])
        for r in mine:
            r["focus_ratio"] = round(r["focus"] / med, 3)
    per_sensor = {}
    for r in rows:
        reasons = [x for x in (r["settings"], r["manual"]) if x]
        r["review"] = r["focus_ratio"] < FOCUS_RATIO and not r["manual"]
        r["accepted"] = not reasons
        r["reason"] = "; ".join(reasons)
        per_sensor.setdefault(r["sensor"], []).append(r["accepted"])
    for r in rows:
        if sum(per_sensor[r["sensor"]]) < 2 and r["accepted"]:
            r["accepted"], r["reason"] = False, "sensor has fewer than 2 accepted images"
    write_csv(a.out / "qc_images.csv", rows)
    review = [dict(file=r["file"], sensor=r["sensor"], focus_ratio=r["focus_ratio"]) for r in rows if r["review"]]
    if review:
        write_csv(a.out / "qc_review.csv", review)
        print(f"qc: {len(review)} images with low focus scores; look at them and add any to be dropped "
              f"to the manual-exclusions file (file, reason), then rerun qc:")
        for r in review:
            print(f"  review {r['file']} (focus {r['focus_ratio']:.2f} of sensor median)")
    n_ok = sum(r["accepted"] for r in rows)
    print(f"qc: {n_ok}/{len(rows)} images accepted; "
          f"{sum(sum(v) >= 2 for v in per_sensor.values())}/{len(per_sensor)} sensors usable")
    for r in rows:
        if not r["accepted"]:
            print(f"  dropped {r['file']}: {r['reason']}")


# ------------------------------------------------------------- stage: measure
def stage_measure(a):
    qc = [r for r in read_csv(a.out / "qc_images.csv") if r["accepted"] == "True"]
    by_name = {im.path.name: im for im in C.index_sem(a.images)}
    rows = []
    for r in qc:
        im = by_name[r["file"]]
        rows.append(dict(sensor=r["sensor"], file=r["file"], session=r["session"],
                         **measures(im.path, float(im.meta["pixel_width_nm"]))))
    write_csv(a.out / "morph_images.csv", rows)
    write_csv(a.out / "morph_sensors.csv", per_sensor_medians(rows))
    print(f"measure: {len(rows)} images, {len({r['sensor'] for r in rows})} sensors")


def per_sensor_medians(rows, sessions=None):
    out = {}
    for r in rows:
        if sessions is None or r["session"] in sessions:
            out.setdefault(r["sensor"], []).append(r)
    return [dict(sensor=s, n_images=len(v), **{k: float(np.median([float(x[k]) for x in v])) for k in MEASURES})
            for s, v in sorted(out.items())]


# ------------------------------------------------------------- stage: session
def stage_session(a):
    rows = read_csv(a.out / "morph_images.csv")
    by = {}
    for r in rows:
        by.setdefault(r["sensor"], {}).setdefault(r["session"], []).append(float(r["edge_density"]))
    twice = {s: v for s, v in by.items() if len(v) > 1}
    latest = max(r["session"] for r in rows)
    sd = np.std([np.median(v[max(v)]) for v in by.values()], ddof=1)
    shifts = {s: float(np.median(v[max(v)]) - np.median(v[min(v)])) for s, v in twice.items()}
    res = dict(n_sensors_twice=len(twice), between_sensor_sd=float(sd), latest_session=latest, shifts=shifts)
    if twice:
        mean_shift = float(np.mean(list(shifts.values())))
        p = float(stats.ttest_1samp(list(shifts.values()), 0).pvalue) if len(twice) > 1 else np.nan
        exclude = bool(abs(mean_shift) / sd > SESSION_LIMIT and p < SESSION_ALPHA)
        res.update(mean_shift=mean_shift, ratio=abs(mean_shift) / sd, t_test_p=p, exclude_earlier_sessions=exclude)
        print(f"session: {len(twice)} sensors imaged twice, mean edge-density shift {mean_shift:+.3f} "
              f"= {abs(mean_shift) / sd:.2f} x between-sensor s.d. (limit {SESSION_LIMIT}), t-test p {p:.3f} "
              f"(limit {SESSION_ALPHA}); exclude earlier sessions: {exclude}")
    else:
        res.update(mean_shift=None, ratio=None, exclude_earlier_sessions=False)
        print("session: no sensor imaged in two sessions; check not possible")
    json.dump(res, open(a.out / "session_check.json", "w"), indent=1)


def sensor_measures(a):
    """Per-sensor medians; for a sensor imaged twice, only its latest session counts.
    If the session check failed, sensors with no image from the latest session are dropped."""
    rows = read_csv(a.out / "morph_images.csv")
    chk = json.load(open(a.out / "session_check.json"))
    latest = {}
    for r in rows:
        latest[r["sensor"]] = max(latest.get(r["sensor"], ""), r["session"])
    keep = [r for r in rows if r["session"] == latest[r["sensor"]]
            and (not chk["exclude_earlier_sessions"] or r["session"] == chk["latest_session"])]
    return {r["sensor"]: r for r in per_sensor_medians(keep)}


# ------------------------------------------------------------------ stage: h1
def partial_spearman(x, y, Z):
    rx, ry = stats.rankdata(x), stats.rankdata(y)
    Zr = np.column_stack([np.ones(len(x))] + [stats.rankdata(z) for z in Z.T])
    ex = rx - Zr @ np.linalg.lstsq(Zr, rx, rcond=None)[0]
    ey = ry - Zr @ np.linalg.lstsq(Zr, ry, rcond=None)[0]
    r = np.corrcoef(ex, ey)[0, 1]
    df = len(x) - 2 - Z.shape[1]
    t = r * np.sqrt(df / max(1e-12, 1 - r ** 2))
    return float(r), float(stats.t.sf(t, df))          # one-sided, r > 0


def recipe(sensors):
    return np.array([[np.log10(c), np.log10(i), t] for c, i, t in map(C.parse_sensor_key, sensors)])


def development(a):
    labels = read_labels(a.labels)
    test = set(sensor_list(a.locked))
    return labels, test, sorted(s for s in labels if s not in test)


def stage_h1(a):
    meas = sensor_measures(a)
    labels, test, dev = development(a)
    disc = set(sensor_list(a.discovery))
    new = [s for s in dev if s in meas and s not in disc]
    missing = [s for s in dev if s not in meas and s not in disc]
    x = np.array([meas[s]["edge_density"] for s in new])
    y = np.array([labels[s]["response"] for s in new])
    r = stats.spearmanr(x, y, alternative="greater")
    pr, pp = partial_spearman(x, y, recipe(new)) if len(new) > 6 else (np.nan, np.nan)
    res = dict(n=len(new), missing=missing, rho=float(r.statistic), p_one_sided=float(r.pvalue),
               partial_rho=pr, partial_p_one_sided=pp, underpowered=len(new) < 35,
               secondary={k: float(stats.spearmanr([meas[s][k] for s in new], y).statistic) for k in MEASURES})
    json.dump(res, open(a.out / "h1.json", "w"), indent=1)
    print(f"H1 on {len(new)} development sensors outside the discovery set: rho {r.statistic:+.3f}, "
          f"one-sided p {r.pvalue:.4f}; partial rho given recipe {pr:+.3f} (p {pp:.4f})"
          + ("  [UNDERPOWERED: fewer than 35 sensors]" if len(new) < 35 else ""))


# ------------------------------------------------------------------ stage: cv
def image_features(a, sensors):
    """Sensor mean of the frozen tile features, accepted images only."""
    feat = Path(a.features)
    ok = {r["file"] for r in read_csv(a.out / "qc_images.csv") if r["accepted"] == "True"}
    img_rows = {int(r["img"]): r for r in read_csv(feat / "images.csv")}
    d = np.load(feat / "dinov2_vitb14_n448_pct.npz")
    f = d["feats"].astype(np.float32)
    tile_sensor = np.array([img_rows[int(i)]["sensor"] if img_rows[int(i)]["file"] in ok else "" for i in d["img"]])
    return np.stack([f[tile_sensor == s].mean(0) for s in sensors])


def design(a, sensors, meas):
    par = np.stack([C.fab_features(s) for s in sensors])
    edge = np.array([[meas[s]["edge_density"]] for s in sensors])
    img = image_features(a, sensors)
    return {"M0": (par, "params"), "M1": (edge, "linear"), "M2": (np.hstack([par, edge]), "params"),
            "M3": (img, "img"), "M4": (np.hstack([img, par]), "img|params")}


def stage_cv(a):
    meas = sensor_measures(a)
    labels, test, dev = development(a)
    dev = [s for s in dev if s in meas]
    y = np.array([labels[s]["response"] for s in dev])
    models = design(a, dev, meas)
    rows, per_repeat = [], {}
    for name, (X, kind) in models.items():
        rho, r2, mae, _ = cv_score(X, y, kind, a.repeats)
        per_repeat[name] = rho
        rows.append(dict(model=name, kind=kind, n=len(dev), rho=rho.mean(), rho_sd=rho.std(), r2=r2.mean(), mae=mae.mean()))
        print(f"{name} ({kind:10s}) rho {rho.mean():+.3f} +/- {rho.std():.3f}  R2 {r2.mean():+.3f}  MAE {mae.mean():.3f}", flush=True)
    rng = np.random.default_rng(0)
    null = np.zeros((a.n_perm, len(ORDER)))
    for k in range(a.n_perm):
        yp = y[rng.permutation(len(y))]
        for j, name in enumerate(ORDER):
            X, kind = models[name]
            null[k, j] = cv_score(X, yp, kind, 1, seed=k)[0][0]
        if (k + 1) % 50 == 0:
            print(f"  permutation {k + 1}/{a.n_perm}", flush=True)
    mx = null.max(1)
    for r in rows:
        if r["model"] in ORDER and a.n_perm:
            j = ORDER.index(r["model"])
            r["perm_p"] = (1 + (null[:, j] >= r["rho"]).sum()) / (1 + a.n_perm)
            r["perm_p_max"] = (1 + (mx >= r["rho"]).sum()) / (1 + a.n_perm)
        else:
            r["perm_p"] = r["perm_p_max"] = np.nan
    write_csv(a.out / "cv_results.csv", rows)
    m2_beats_m0 = float(np.mean(per_repeat["M2"] > per_repeat["M0"]))   # same RepeatedKFold seed: paired splits
    json.dump(dict(n_dev=len(dev), repeats=a.repeats, n_perm=a.n_perm, m2_beats_m0_fraction=m2_beats_m0,
                   per_repeat={k: v.tolist() for k, v in per_repeat.items()}), open(a.out / "cv.json", "w"), indent=1)
    print(f"M2 beats M0 in {m2_beats_m0:.0%} of paired repeats (H2 needs > 95%)")
    for r in rows:
        print(f"  {r['model']} perm p {r['perm_p']:.4f}  max-corrected {r['perm_p_max']:.4f}")


# -------------------------------------------------------------- stage: choose
def stage_choose(a):
    res = {r["model"]: float(r["rho"]) for r in read_csv(a.out / "cv_results.csv")}
    best = max(ORDER, key=lambda m: res[m])
    chosen = next(m for m in ORDER if res[m] >= res[best] - NEAR)
    path = a.out / "preregistered_final.txt"
    if path.exists():
        raise SystemExit(f"{path} exists; the choice is made once")
    path.write_text(f"# Chosen {datetime.now().isoformat(timespec='seconds')} by the rule: best of M1-M4 by mean dev rho, "
                    f"simplest within {NEAR} wins.\n# dev rho: " + ", ".join(f"{m} {res[m]:+.3f}" for m in ORDER) +
                    f"\n# cv_results.csv sha256 {sha(a.out / 'cv_results.csv')}; locked file sha256 {sha(a.locked)}\n{chosen}\n")
    print(f"choose: best {best} ({res[best]:+.3f}); chosen {chosen} ({res[chosen]:+.3f}); "
          f"written to {path} (sha256 {sha(path)})")


# --------------------------------------------------------------- stage: final
def stage_final(a):
    pre = a.out / "preregistered_final.txt"
    chosen = [l.strip() for l in open(pre) if l.strip() and not l.startswith("#")][0]
    log = a.out / "final_test_log.csv"
    if log.exists() and any(r["locked_sha"] == sha(a.locked) for r in read_csv(log)):
        raise SystemExit("this locked set has already been scored; it is scored once")
    meas = sensor_measures(a)
    labels, test, dev = development(a)
    all_labels = {}
    for r in read_csv(a.labels):                       # the locked labels are read here, and only here
        if r["excluded_as_outlier"] != "True":
            all_labels[r["sensor"]] = np.log10(float(r["rel_resp_median"]))
    dev = [s for s in dev if s in meas]
    tst = [s for s in sorted(test) if s in meas and s in all_labels]
    y, yt = np.array([labels[s]["response"] for s in dev]), np.array([all_labels[s] for s in tst])
    X, kind = design(a, dev, meas)[chosen]
    Xt, _ = design(a, tst, meas)[chosen]
    p = make_model(kind, len(dev)).fit(X, y).predict(Xt)
    rho = stats.spearmanr(p, yt).statistic
    rng = np.random.default_rng(0)
    null = np.array([stats.spearmanr(p, rng.permutation(yt)).statistic for _ in range(10000)])
    e = np.array([meas[s]["edge_density"] for s in tst])
    edge_rho = stats.spearmanr(e, yt).statistic
    edge_null = np.array([stats.spearmanr(e, rng.permutation(yt)).statistic for _ in range(10000)])
    res = dict(when=datetime.now().isoformat(timespec="seconds"), model=chosen, n_test=len(tst),
               missing=len(test) - len(tst), rho=round(float(rho), 4), perm_p=round(float((1 + (null >= rho).sum()) / 10001), 4),
               mae=round(float(np.abs(p - yt).mean()), 4), edge_rho=round(float(edge_rho), 4),
               edge_perm_p_one_sided=round(float((1 + (edge_null >= edge_rho).sum()) / 10001), 4), locked_sha=sha(a.locked), prereg_sha=sha(pre))
    new = not log.exists()
    with open(log, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res))
        if new:
            w.writeheader()
        w.writerow(res)
    print("FINAL", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["qc", "measure", "session", "h1", "cv", "choose", "final"])
    ap.add_argument("--images", required=True, help="folder of <sensor>-2500x/ subfolders")
    ap.add_argument("--features", help="folder with dinov2_vitb14_n448_pct.npz and images.csv (cv, final)")
    ap.add_argument("--labels", required=True)
    ap.add_argument("--locked", required=True)
    ap.add_argument("--discovery", default="", help="sensors that generated H1, one per line")
    ap.add_argument("--manual-exclusions", default="", help="csv with columns file, reason")
    ap.add_argument("--out", required=True)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--n-perm", type=int, default=1000)
    a = ap.parse_args()
    a.out = Path(a.out)
    a.out.mkdir(parents=True, exist_ok=True)
    globals()[f"stage_{a.stage}"](a)


if __name__ == "__main__":
    main()
