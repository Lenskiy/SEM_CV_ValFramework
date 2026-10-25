"""Paper figures (main text and Supplementary).

  fig_recipes.pdf     Fig. 1: recipe grid of log10 relative response for all 72 sensors (locked
                      sensors marked by an inner white square, outlier by *), and three 500x images
  fig_rair.pdf        Fig. 2: response against baseline resistance (56 development sensors)
  fig_chain.pdf       Fig. 3: recipe -> microstructure -> baseline resistance -> response (E8-E10)
  fig_models.pdf      Fig. 4: all model families for response and resistance; locked score
  fig_replicates.pdf  Fig. 5: replicate minus original, measured vs predicted; type-share change
  fig_tilemaps.pdf    Fig. 6: exact tile scores over four example images
  fig_types.pdf       Fig. 7: the 8 morphology types (500x) and their mean tile score
  fig_tsne.pdf        Supplementary Fig. S1: t-SNE of tiles and nearest-neighbour agreement
  fig_pca.pdf         Supplementary Fig. S2: sensor-level PCA of the image features
  fig_heating.pdf     Supplementary Fig. S3: resistance on first heating vs baseline resistance
  fig_2500x.pdf       not used in the paper (2,500x edge density, 12 sensors)
Writes to paper_v2/figures/. Colours: reference palette of the dataviz method
(sequential blue; categorical blue / orange / aqua).
"""
import csv
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from PIL import Image

import common as C
from dev_eval import locked

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper_v2/figures"
SEM500 = Path(os.environ.get("ZNO_SEM500", ROOT / "data/SEM_500x"))   # 500x images, one folder per sensor
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
CONC, CURR, TIME = [0.01, 0.05, 0.1, 0.2], [100, 250, 500, 1000, 3000, 5000], [10, 30, 60]

plt.rcParams.update({"hatch.color": "#b5b4ae", "hatch.linewidth": 0.5, "font.family": "sans-serif", "font.size": 7, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.dpi": 300})


def labels():
    return {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/derived/labels_v2.csv"))}


def key(c, i, t):
    return f"{c:g}M-{i}uA-{t}s"


def stretch(a):
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)


def pretty(s):
    c, i, t = C.parse_sensor_key(s)
    cur = f"{i / 1000:g} mA" if i >= 1000 else f"{i:g} µA"
    return f"{c:g} M, {cur}, {t:g} s"


def panel_label(ax, s, x=-0.02, y=1.02):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom", ha="right")


# ---------------------------------------------------------------- figure: recipes
def fig_recipes():
    lab = labels()
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    vals = [np.log10(float(r["rel_resp_median"])) for s, r in lab.items() if r["excluded_as_outlier"] != "True"]
    vmin, vmax = np.floor(min(vals)), 0
    fig = plt.figure(figsize=(7.1, 4.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1, 1.15], width_ratios=[1, 1, 1, 0.06], hspace=0.45, wspace=0.12)
    for j, t in enumerate(TIME):
        ax = fig.add_subplot(gs[0, j])
        for yi, c in enumerate(CONC):
            for xi, i in enumerate(CURR):
                s = key(c, i, t)
                r = lab.get(s)
                if r is None:
                    ax.add_patch(Rectangle((xi, yi), 1, 1, fc="#d9d8d3", ec="white", lw=1.5))
                    continue
                v = np.log10(float(r["rel_resp_median"]))
                ax.add_patch(Rectangle((xi, yi), 1, 1, fc=SEQ((v - vmin) / (vmax - vmin)), ec="white", lw=1.5))
                if s in test:
                    ax.add_patch(Rectangle((xi + 0.12, yi + 0.12), 0.76, 0.76, fc="none", ec="white", lw=1.0))
                if r["excluded_as_outlier"] == "True":
                    ax.text(xi + 0.5, yi + 0.5, "*", ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 4)
        ax.set_aspect("equal")
        ax.set_xticks(np.arange(6) + 0.5, ["0.1", "0.25", "0.5", "1", "3", "5"])
        ax.set_yticks(np.arange(4) + 0.5, [f"{c:g}" for c in CONC] if j == 0 else [])
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{t} s deposition", fontsize=7, color=INK)
        ax.set_xlabel("Current (mA)")
        if j == 0:
            ax.set_ylabel("ZnCl$_2$ (M)")
            panel_label(ax, "a")
    cax = fig.add_subplot(gs[0, 3])
    cb = plt.colorbar(plt.cm.ScalarMappable(cmap=SEQ, norm=plt.Normalize(vmin, vmax)), cax=cax)
    cb.set_label("log$_{10}$ relative response", fontsize=7)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2)

    # b: three 500x examples: good, overgrown dead, dead 10 s deposit (all development sensors)
    ex = [("0.05M-250uA-30s", "good responder"), ("0.1M-5000uA-30s", "overgrown, non-responding"),
          ("0.05M-250uA-10s", "10 s deposit, non-responding")]
    idx500 = {}
    for im in C.index_sem(SEM500):
        idx500.setdefault(im.sensor, im.path)
    sub = gs[1, :3].subgridspec(1, 3, wspace=0.05)
    for k, (s, tag) in enumerate(ex):
        ax = fig.add_subplot(sub[0, k])
        a = C.load_image_array(idx500[s])[300:1500, 1000:2800]
        ax.imshow(stretch(a), cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        L = 50 / 0.2698
        ax.add_patch(Rectangle((1800 - L - 60, 1200 - 110), L, 30, color="white"))
        ax.text(1800 - L / 2 - 60, 1200 - 130, "50 µm", color="white", ha="center", va="bottom", fontsize=6)
        ax.set_title(f"{tag}\n{pretty(s)}, response {100 * float(lab[s]['rel_resp_median']):.2g}%", fontsize=6.5, color=INK)
        if k == 0:
            panel_label(ax, "b")
    fig.savefig(OUT / "fig_recipes.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_recipes.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# -------------------------------------------------------------- figure: replicates
def fig_replicates():
    pairs = [("0.1M-1000uA-60s", 2.94, 1.45, -86), ("0.2M-250uA-60s", 0.56, 0.71, -56), ("0.2M-5000uA-60s", 0.32, 0.28, -38)]
    names = ["0.1 M, 1 mA, 60 s", "0.2 M, 250 µA, 60 s", "0.2 M, 5 mA, 60 s"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.4), gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.55))
    y = np.arange(len(pairs))[::-1]
    a1.axvspan(-0.45, 0.45, color="#f0efec", lw=0, zorder=0)
    h = 0.32
    a1.barh(y + h / 2 + 0.02, [p[1] for p in pairs], height=h, color=BLUE, label="measured")
    a1.barh(y - h / 2 - 0.02, [p[2] for p in pairs], height=h, color=ORANGE, label="predicted (500× model)")
    for yi, p in zip(y, pairs):
        a1.text(p[1] + 0.05, yi + h / 2 + 0.02, f"{p[1]:+.2f}", va="center", fontsize=6, color=INK)
        a1.text(p[2] + 0.05, yi - h / 2 - 0.02, f"{p[2]:+.2f}", va="center", fontsize=6, color=INK)
    a1.set_yticks(y, names)
    a1.tick_params(axis="y", length=0)
    a1.axvline(0, color=INK2, lw=0.6)
    a1.set_xlabel("Replicate minus original, log$_{10}$ relative response")
    a1.set_xlim(-0.6, 3.4)
    a1.set_ylim(-0.6, len(pairs) - 0.1)
    a1.legend(frameon=False, loc="lower right", fontsize=6)
    a1.spines["left"].set_visible(False)
    panel_label(a1, "a")
    a2.barh(y, [p[3] for p in pairs], height=0.5, color=BLUE)
    a2.axvspan(-19, 11, color="#f0efec", lw=0, zorder=0)
    for yi, p in zip(y, pairs):
        a2.text(p[3] - 2, yi, f"{p[3]:+d}", va="center", ha="right", fontsize=6, color=INK)
    a2.set_yticks(y, [])
    a2.tick_params(axis="y", length=0)
    a2.axvline(0, color=INK2, lw=0.6)
    a2.set_xlim(-105, 20)
    a2.set_ylim(-0.6, len(pairs) - 0.1)
    a2.set_xlabel("Change in share of the lowest-scoring\nmorphology type (percentage points)")
    a2.spines["left"].set_visible(False)
    panel_label(a2, "b")
    fig.savefig(OUT / "fig_replicates.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_replicates.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure: types
def fig_types():
    summ = {int(r["type"]): r for r in csv.DictReader(open(ROOT / "data/xai_500x/type_summary.csv"))}
    names = {}
    for line in open(ROOT / "data/xai_500x/type_names.txt"):
        if line.strip() and not line.startswith("#"):
            k, n = line.strip().split(",", 1)
            names[int(k)] = n
    order = sorted(summ, key=lambda k: float(summ[k]["mean_tile_score"]))
    mont = np.asarray(Image.open(ROOT / "data/xai_500x/montage_types.png"))
    n = len(order)
    fig = plt.figure(figsize=(6.6, 4.6))
    ax = fig.add_axes([0.0, 0.08, 0.5, 0.9])
    ax.imshow(mont, cmap="gray", vmin=0, vmax=255, extent=(0, 8, 0, n), aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    panel_label(ax, "a")
    ax2 = fig.add_axes([0.52, 0.08, 0.24, 0.9], sharey=ax)
    sc = [float(summ[k]["mean_tile_score"]) for k in order]
    yy = np.arange(n)[::-1] + 0.5
    ref = sum(float(summ[k]["mean_tile_score"]) * float(summ[k]["share"]) for k in order)  # mean over all tiles
    lo = min(sc) - 0.15
    ax2.axvline(ref, color=INK2, lw=0.6)
    for y_, k, v in zip(yy, order, sc):
        ax2.plot([ref, v], [y_ - 0.12] * 2, color=BLUE, lw=1.2, zorder=2)
        ax2.scatter(v, y_ - 0.12, s=16, color=BLUE, zorder=3)
        ax2.text(lo + 0.02, y_ + 0.08, f"{names[k]} ({100 * float(summ[k]['share']):.0f}% of tiles)",
                 fontsize=6, color=INK, va="bottom", clip_on=False, bbox=dict(fc="white", ec="none", pad=0.3, alpha=0.85))
        ax2.text(v + (0.04 if v >= ref else -0.04), y_ - 0.12, f"{v:.2f}".replace("-", "−"), fontsize=6, color=INK,
                 va="center", ha="left" if v >= ref else "right")
    ax2.set_ylim(0, n)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    ax2.spines["left"].set_visible(False)
    ax2.set_xlabel("Mean tile score, response model\n(predicted log$_{10}$ response)")
    ax2.set_xlim(lo, max(sc) + 0.35)
    panel_label(ax2, "b")
    # c: the same types under the resistance model (xai_rair.py)
    import json
    xr = json.load(open(ROOT / "data/xai_500x/xai_rair.json"))
    sr = [xr["types"][str(k)]["mean_tile_score_r_air"] for k in order]
    ref3 = xr["mean_tile_score_r_air_all"]
    ax3 = fig.add_axes([0.79, 0.08, 0.19, 0.9], sharey=ax)
    ax2.set_zorder(5)                  # type names (in b) are drawn over panel c
    ax2.patch.set_alpha(0)
    ax3.axvline(ref3, color=INK2, lw=0.6)
    for y_, v in zip(yy, sr):
        ax3.plot([ref3, v], [y_ - 0.12] * 2, color=AQUA, lw=1.2, zorder=2)
        ax3.scatter(v, y_ - 0.12, s=16, color=AQUA, zorder=3)
        ax3.text(v + (0.15 if v >= ref3 else -0.15), y_ - 0.12, f"{v:.1f}", fontsize=6, color=INK,
                 va="center", ha="left" if v >= ref3 else "right")
    ax3.set_ylim(0, n)
    ax3.set_xlim(min(sr) - 0.7, max(sr) + 0.6)
    ax3.tick_params(axis="y", left=False, labelleft=False)
    ax3.spines["left"].set_visible(False)
    ax3.set_xlabel("Mean tile score, resistance model\n(predicted log$_{10}$ Ω)")
    panel_label(ax3, "c")
    fig.savefig(OUT / "fig_types.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_types.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure: 2500x
def fig_2500x():
    lab = labels()
    m = [r for r in csv.DictReader(open(ROOT / "data/derived/hm_morphology.csv")) if r["mag"] == "2500x"]
    fig = plt.figure(figsize=(7.1, 3.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6], wspace=0.15)
    ax = fig.add_subplot(gs[0, 0])
    cols = {10: BLUE, 30: ORANGE, 60: AQUA}
    for t in TIME:
        pts = [r for r in m if C.parse_sensor_key(r["sensor"])[2] == t]
        for r in pts:
            out = lab[r["sensor"]]["excluded_as_outlier"] == "True"
            ax.scatter(float(r["edge_density"]), np.log10(float(lab[r["sensor"]]["rel_resp_median"])), s=26,
                       facecolor="white" if out else cols[t], edgecolor=cols[t], lw=1.2, zorder=3)
    for t in TIME:
        ax.scatter([], [], s=26, color=cols[t], label=f"{t} s")
    ax.scatter([], [], s=26, facecolor="white", edgecolor=INK2, lw=1.2, label="excluded outlier")
    ax.legend(frameon=False, fontsize=6, loc="lower right", title="deposition time", title_fontsize=6)
    ax.set_xlabel("Edge density at 2,500× (gradient / grey-level s.d.)")
    ax.set_ylabel("log$_{10}$ relative response")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    panel_label(ax, "a", x=-0.16, y=1.0)
    # b: tiles of the two lowest and two highest responders
    sens = sorted({r["sensor"] for r in m}, key=lambda s: float(lab[s]["rel_resp_median"]))
    pick = sens[:2] + sens[-2:]
    idx = {}
    for im in C.index_sem(ROOT / "data/raw_hm/2500x"):
        idx.setdefault(im.sensor, im.path)
    sub = gs[0, 1].subgridspec(2, 2, wspace=0.04, hspace=0.25)
    for k, s in enumerate(pick):
        a_ = fig.add_subplot(sub[k // 2, k % 2])
        a = C.load_image_array(idx[s])[400:1300, 1200:2550]
        a_.imshow(stretch(a), cmap="gray", vmin=0, vmax=1)
        a_.set_xticks([])
        a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_visible(False)
        L = 10 / 0.05396
        a_.add_patch(Rectangle((1350 - L - 40, 900 - 80), L, 22, color="white"))
        a_.text(1350 - L / 2 - 40, 900 - 95, "10 µm", color="white", ha="center", va="bottom", fontsize=6)
        tag = "low" if k < 2 else "high"
        c, _, t = C.parse_sensor_key(s)
        a_.set_title(f"{tag}: {c:g} M, {t:g} s ({100 * float(lab[s]['rel_resp_median']):.2g}%)", fontsize=6, color=INK)
        if k == 0:
            panel_label(a_, "b", x=-0.06, y=1.0)
    fig.savefig(OUT / "fig_2500x.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_2500x.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# --------------------------------------------------------------- figure: models
def fig_models():
    import json
    from scipy.stats import spearmanr
    from dev_eval import make_model, read_labels, sensor_features
    bb = list(csv.DictReader(open(ROOT / "data/derived/r_air_backbones.csv")))
    small = list(csv.DictReader(open(ROOT / "data/derived/compare_small.csv")))
    tex = {"response": json.load(open(ROOT / "data/derived/texture_baseline.json"))["rho"],
           "r_air": json.load(open(ROOT / "data/derived/r_air_predict.json"))["B_texture_r_air"]["rho"]}
    pf = {r["config"]: r for r in csv.DictReader(open(ROOT / "data/dev_eval_500x/perm_full_response.csv"))}
    rp = json.load(open(ROOT / "data/derived/r_air_predict.json"))["A_vitb_r_air"]
    nulls = {"response": (float(pf["dinov2_vitb14_n448_pct_mean_img"]["null_mean"]), float(pf["dinov2_vitb14_n448_pct_mean_img"]["null_sd"])),
             "r_air": (rp["null_mean"], rp["null_sd"])}
    rn = json.load(open(ROOT / "data/derived/recipe_nonlinear.json"))["models"]
    groups = [("Recipe, linear", lambda c: c.startswith("recipe")),
              ("Recipe, nearest neighbours", lambda c: c.startswith("knn|")),
              ("Texture statistics", lambda c: c == "texture"),
              ("Four-layer CNN, from scratch", lambda c: c == "simple_cnn"),
              ("Three-block ResNet, from scratch", lambda c: c == "small_resnet"),
              ("ResNet-18 to stage 2", lambda c: c.startswith("resnet18_l2")),
              ("ResNet-18", lambda c: c.startswith("resnet18_") and not c.startswith("resnet18_l2")),
              ("ResNet-50 to stage 2/3", lambda c: c.startswith("resnet50_l")),
              ("ResNet-50", lambda c: c.startswith("resnet50_") and not c.startswith("resnet50_l")),
              ("ResNet-50, MicroNet", lambda c: "micronet" in c),
              ("DINOv2 ViT-S", lambda c: c.startswith("dinov2_vits14")),
              ("DINOv2 ViT-B", lambda c: c.startswith("dinov2_vitb14")),
              ("ViT-B, fine-tuned", lambda c: c == "finetuned")]
    chosen = "dinov2_vitb14_n448_pct"
    fig = plt.figure(figsize=(5.5, 3.8))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 0.72, 1.0], wspace=0.12)
    axes = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    for ax, target, title in ((axes[0], "response", "Response"), (axes[1], "r_air", "Baseline resistance")):
        pts = [(r["config"], float(r["rho"]), float(r["rho_sd"]), "o") for r in bb if r["target"] == target]
        pts.append(("texture", tex[target], 0.0, "o"))
        pts += [(k, v["rho_mean"], v["rho_sd"], "o") for k, v in rn.items() if k.startswith("knn|") and k.endswith("|" + target)]
        for r in small:
            if r["target"] == target and r["model"] in ("simple_cnn", "small_resnet"):
                pts += [(r["model"], float(v), 0.0, "D") for v in r["rho_seeds"].split()]
        if target == "response":    # per-seed pooled rho, data/dev_eval_500x/selection_outcome_500x.txt
            sel = open(ROOT / "data/dev_eval_500x/selection_outcome_500x.txt").read()
            seeds = sel.split("(seeds ")[1].split(")")[0].split(",")
            pts += [("finetuned", float(v), 0.0, "D") for v in seeds]
        nm, ns = nulls[target]
        ax.axvspan(nm - 2 * ns, nm + 2 * ns, color="#f0efec", lw=0, zorder=0)
        ax.axvline(nm, color="#b5b4ae", lw=0.8, zorder=0)
        ax.axvline(0, color=INK2, lw=0.6)
        for gi, (name, sel) in enumerate(groups):
            y0 = len(groups) - gi
            rs = [p for p in pts if sel(p[0])]
            for k, (c, rho, sd, mk) in enumerate(rs):
                jit = (k - (len(rs) - 1) / 2) * 0.06
                ch = c == chosen
                ax.errorbar(rho, y0 + jit, xerr=sd, fmt=mk, ms=4.5 if ch else 2.8, color=ORANGE if ch else BLUE,
                            ecolor="#9b9a96", elinewidth=0.8, capsize=0, zorder=3 if ch else 2)
        ax.set_xlim(-0.55, 0.7)
        ax.set_ylim(0.4, len(groups) + 0.8)
        ax.set_xlabel("Spearman ρ, held-out sensors" if target == "response" else "Spearman ρ")
        ax.set_title(title, fontsize=7, color=INK)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_yticks(range(len(groups), 0, -1), [g[0] for g in groups])
    axes[1].set_yticks(range(len(groups), 0, -1), [""] * len(groups))
    panel_label(axes[0], "a", x=-0.62, y=1.0)
    panel_label(axes[1], "b", x=-0.05, y=1.0)
    a2 = fig.add_subplot(gs[2])
    pos = a2.get_position()
    a2.set_position([pos.x0 + 0.07, pos.y0, pos.width - 0.07, pos.height])
    # b: locked test (scored once); predictions re-derived for plotting with the same deterministic fit
    feat = ROOT / "data/features500x"
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = locked(ROOT / "data/derived/locked_test_sensors.txt")
    dev = sorted(s_ for s_ in lab if s_ not in set(test))
    X = sensor_features(feat / "dinov2_vitb14_n448_pct.npz", feat / "images.csv", dev, "mean")
    y = np.array([lab[s_]["response"] for s_ in dev])
    tst = [s_ for s_ in test if s_ in lab]
    Xt = sensor_features(feat / "dinov2_vitb14_n448_pct.npz", feat / "images.csv", tst, "mean")
    yt = np.array([lab[s_]["response"] for s_ in tst])
    pt = make_model("img", len(dev)).fit(X, y).predict(Xt)
    oof = {r["sensor"]: float(r["oof_pred"]) for r in csv.DictReader(open(ROOT / "data/dev_eval_500x/oof_preregistered.csv"))}
    a2.scatter([oof[s_] for s_ in dev], y, s=10, color="#a3a29d", lw=0, label="development (out of fold)")
    a2.scatter(pt, yt, s=22, color=ORANGE, edgecolor="white", lw=0.6, label="locked test (scored once)", zorder=3)
    a2.axhline(-2, color=INK2, lw=0.5, ls=(0, (3, 2)))
    a2.set_xlabel("Predicted log$_{10}$ relative response")
    a2.set_ylabel("Measured log$_{10}$ relative response")
    a2.legend(frameon=False, fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    rho = spearmanr(pt, yt).statistic
    logged = [r for r in csv.DictReader(open(ROOT / "data/derived/final_test_log.csv"))][0]
    assert abs(rho - float(logged["rho"])) < 1e-3, (rho, logged["rho"])     # same fit as the one-shot score
    a2.set_title(f"Locked test (15 sensors)\nρ = {float(logged["rho"]) + 1e-9:.2f}, p = {float(logged['perm_p']):.2f}",
                 fontsize=7, color=INK)
    a2.grid(color=GRID, lw=0.5)
    a2.set_axisbelow(True)
    panel_label(a2, "c", x=-0.22, y=1.04)
    fig.savefig(OUT / "fig_models.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_models.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure: air resistance
def replicate_r_air():
    from compute_snr import find_cycles, load_sheet_data
    from replicate_labels import sensor_key, sheets
    out = {}
    for path in sorted((ROOT / "data/raw_replicate/Sensor Responses").glob("*.xlsx")):
        for name, data in sheets(path).items():
            if str(data[0].get(0, "")).lower().startswith("loop"):
                sig = load_sheet_data(data)
                ra = [np.nanmean(sig[max(0, on - 10):on, 1]) for on, _ in find_cycles(sig)]
                out.setdefault(sensor_key(name), []).append(float(np.nanmedian(ra)))
    return {k: float(np.median(v)) for k, v in out.items()}


def fig_rair():
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from dev_eval import read_labels, sensor_features
    feat = ROOT / "data/features500x"
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s_ for s_ in lab if s_ not in test)
    y = np.array([lab[s_]["response"] for s_ in dev])
    ra = {r["sensor"]: float(r["r_air_median"]) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    x = np.log10([ra[s_] for s_ in dev])
    Z = PCA(2, random_state=0).fit_transform(StandardScaler().fit_transform(
        sensor_features(feat / "dinov2_vitb14_n448_pct.npz", feat / "images.csv", dev, "mean")))
    hi = Z[:, 0] < -30
    t10 = np.array([C.parse_sensor_key(s_)[2] == 10 for s_ in dev]) & ~hi
    fig, ax = plt.subplots(figsize=(3.5, 2.9))
    for m, col, name in ((~hi & ~t10, "#86b6ef", "other sensors"), (t10, AQUA, "10 s deposits"),
                         (hi, ORANGE, "high-deposition group")):
        ax.scatter(x[m], y[m], s=20, color=col, edgecolor="white", lw=0.5, label=name, zorder=3)
    rep = replicate_r_air()
    reps = {r["sensor"]: [] for r in csv.DictReader(open(ROOT / "data/derived/labels_replicate.csv"))}
    for r in csv.DictReader(open(ROOT / "data/derived/labels_replicate.csv")):
        reps[r["sensor"]].append(float(r["rel_resp_median"]))
    for s_, v in reps.items():
        if s_ in dev:
            i = dev.index(s_)
            ax.annotate("", xy=(np.log10(rep[s_]), np.log10(np.mean(v))), xytext=(x[i], y[i]),
                        arrowprops=dict(arrowstyle="->", color=INK2, lw=0.7), zorder=2)
    ax.plot([], [], color=INK2, lw=0.7, label="original → replicate")
    r = spearmanr(x, y)
    ax.set_title(f"Baseline resistance vs response: ρ = {r.statistic:.2f} (56 sensors)", fontsize=7, color=INK)
    ax.axhline(-2, color=INK2, lw=0.5, ls=(0, (3, 2)))
    ax.set_xlabel("log$_{10}$ baseline resistance in air (Ω)")
    ax.set_ylabel("log$_{10}$ relative response")
    ax.legend(frameon=False, fontsize=6, loc="lower right")
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    fig.savefig(OUT / "fig_rair.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_rair.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure: tile maps
def fig_tilemaps():
    from matplotlib.colors import LinearSegmentedColormap as LSC
    T = list(csv.DictReader(open(ROOT / "data/xai_500x/tile_scores_rair.csv")))     # resistance model (xai_rair.py)
    imgs = {int(r["img"]): r for r in csv.DictReader(open(ROOT / "data/features500x/images.csv"))}
    lab = labels()
    ra = {r["sensor"]: float(r["r_air_median"]) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    mean_score = np.mean([float(t["oof_score"]) for t in T])
    picks = [("0.05M-250uA-30s", "good responder"), ("0.1M-5000uA-30s", "overgrown, non-responding"),
             ("0.05M-250uA-10s", "10 s deposit, non-responding"), ("0.1M-1000uA-60s", "replicated recipe (original)")]
    paths = {}
    for im in C.index_sem(SEM500):
        paths[(im.sensor, im.path.name)] = im.path
    div = LSC.from_list("div", ["#c0392b", "#e34948", "#f0efec", "#3987e5", "#1c5cab"])
    fig, axs = plt.subplots(1, 4, figsize=(7.1, 1.95), gridspec_kw=dict(wspace=0.05))
    span = 2.0
    for ax, (s, tag) in zip(axs, picks):
        ts = [t for t in T if t["sensor"] == s]
        img = int(ts[0]["img"])
        ts = [t for t in ts if int(t["img"]) == img]
        a = C.load_image_array(paths[(s, imgs[img]["file"])])[:2240, :3584]
        ax.imshow(stretch(a), cmap="gray", vmin=0, vmax=1)
        grid = np.full((5, 8), np.nan)
        for t in ts:
            grid[int(t["row"]), int(t["col"])] = float(t["oof_score"]) - mean_score
        ax.imshow(grid, cmap=div, vmin=-span, vmax=span, alpha=0.45, extent=(0, 3584, 2240, 0), interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        r_ = ra[s]
        rs = f"{r_ / 1e6:.2g} MΩ" if r_ >= 1e6 else f"{r_ / 1e3:.2g} kΩ" if r_ >= 1e3 else f"{r_:.3g} Ω"
        ax.set_title(f"{tag}\n{pretty(s)}\n{100 * float(lab[s]['rel_resp_median']):.2g}% response, {rs}",
                     fontsize=6, color=INK)
    sm = plt.cm.ScalarMappable(cmap=div, norm=plt.Normalize(-span, span))
    cb = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.05, pad=0.04, aspect=50, extend="both")
    cb.set_label("Tile score minus development mean (predicted log$_{10}$ baseline resistance, Ω); red lowers the prediction",
                 fontsize=6)
    cb.outline.set_visible(False)
    fig.savefig(OUT / "fig_tilemaps.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_tilemaps.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure: t-SNE
def fig_tsne():
    import json
    rows = list(csv.DictReader(open(ROOT / "data/xai_500x/tsne_coords.csv")))
    pur = json.load(open(ROOT / "data/xai_500x/tsne_purity.json"))   # full 1,536-d space
    X = np.array([[float(r["x"]), float(r["y"])] for r in rows])
    sen = np.array([r["sensor"] for r in rows])
    terc = np.array([int(r["tercile"]) for r in rows])
    resp = {r["sensor"]: float(r["log_response"]) for r in rows}
    seq3 = ["#86b6ef", "#2a78d6", "#104281"]
    names3 = ["lowest third", "middle third", "highest third"]
    fig = plt.figure(figsize=(7.1, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1], wspace=0.08)
    # a: three example sensors, one per response third (the median sensor of each third)
    ax = fig.add_subplot(gs[0, 0])
    ax.scatter(X[:, 0], X[:, 1], s=0.6, color="#e4e3df", lw=0, rasterized=True)
    xmin, xmax = X[:, 0].min(), X[:, 0].max()
    for t in range(3):
        ss = sorted({s for s in set(sen) if terc[sen == s][0] == t}, key=lambda s: resp[s])
        s = ss[len(ss) // 2]
        m = sen == s
        ax.scatter(X[m, 0], X[m, 1], s=3, color=seq3[t], lw=0, rasterized=True)
        cx, cy = np.median(X[m], 0)
        right = cx < (xmin + xmax) / 2
        ax.annotate(f"{pretty(s)}\n({names3[t]})", (cx, cy), xytext=(cx + (25 if right else -25), cy + 12),
                    fontsize=6, color=INK, ha="left" if right else "right",
                    arrowprops=dict(arrowstyle="-", color=INK2, lw=0.5))
    ax.set_title("Tiles of three example sensors", fontsize=7, color=INK)
    # b: every tile coloured by its sensor's log10 relative response (continuous)
    ax2 = fig.add_subplot(gs[0, 1])
    yv = np.array([float(r["log_response"]) for r in rows])
    o = np.argsort(yv)
    sc = ax2.scatter(X[o, 0], X[o, 1], c=yv[o], s=0.6, cmap=SEQ, vmin=-4, vmax=0, lw=0, rasterized=True)
    cb = fig.colorbar(sc, ax=ax2, orientation="horizontal", fraction=0.05, pad=0.03, aspect=30, extend="min")
    cb.set_label("sensor log$_{10}$ relative response", fontsize=6)
    cb.ax.tick_params(labelsize=5.5, length=2)
    cb.outline.set_visible(False)
    ax2.set_title("All tiles, by their sensor's response", fontsize=7, color=INK)
    for a_, lab_ in ((ax, "a"), (ax2, "b")):
        a_.set_xticks([])
        a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_visible(False)
        panel_label(a_, lab_, x=0.0, y=1.0)
    # c: neighbour agreement vs chance, row labels above each row
    ax3 = fig.add_subplot(gs[0, 2])
    items = [("same sensor", "sensor"), ("same response third", "response tercile"),
             ("same response third, other sensors only", "response tercile, other sensors only")]
    yy = np.arange(len(items))[::-1]
    for y_, (lab_, k) in zip(yy, items):
        o, c = 100 * pur[k]["observed"], 100 * pur[k]["chance"]
        ax3.text(0, y_ + 0.3, lab_, fontsize=6, color=INK, va="bottom")
        ax3.plot([c, o], [y_, y_], color=GRID, lw=2, zorder=1)
        ax3.scatter(c, y_, s=22, facecolor="white", edgecolor=INK2, lw=1, zorder=2)
        ax3.scatter(o, y_, s=22, color=BLUE, zorder=3)
        ax3.text(o + 3, y_, f"{o:.0f}%", fontsize=6, color=INK, va="center")
    ax3.scatter([], [], s=22, color=BLUE, label="observed")
    ax3.scatter([], [], s=22, facecolor="white", edgecolor=INK2, lw=1, label="chance")
    ax3.legend(frameon=False, fontsize=6, loc="lower right", handletextpad=0.2)
    ax3.set_yticks([])
    ax3.spines["left"].set_visible(False)
    ax3.set_xlim(0, 105)
    ax3.set_ylim(-0.75, len(items) - 0.1)
    ax3.set_xlabel("Share of 10 nearest neighbours (%)")
    ax3.set_title("What a tile's neighbours share", fontsize=7, color=INK)
    panel_label(ax3, "c", x=0.0, y=1.0)
    fig.savefig(OUT / "fig_tsne.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_tsne.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure: sensor PCA
def fig_pca():
    from scipy.spatial.distance import pdist
    from scipy.stats import spearmanr
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from dev_eval import read_labels, sensor_features
    feat = ROOT / "data/features500x"
    lab = read_labels(ROOT / "data/derived/labels_v2.csv")
    test = set(locked(ROOT / "data/derived/locked_test_sensors.txt"))
    dev = sorted(s for s in lab if s not in test)
    y = np.array([lab[s]["response"] for s in dev])
    X = StandardScaler().fit_transform(sensor_features(feat / "dinov2_vitb14_n448_pct.npz", feat / "images.csv", dev, "mean"))
    pca = PCA(2, random_state=0).fit(X)
    Z = pca.transform(X)
    ev = 100 * pca.explained_variance_ratio_
    r1, r2 = spearmanr(Z[:, 0], y).statistic, spearmanr(Z[:, 1], y).statistic
    rng = np.random.default_rng(0)
    dX = pdist(X)
    mant = spearmanr(dX, pdist(y[:, None])).statistic
    null = [spearmanr(dX, pdist(rng.permutation(y)[:, None])).statistic for _ in range(2000)]
    pm = (1 + np.sum(np.array(null) >= mant)) / 2001
    rec = np.array([C.parse_sensor_key(s_) for s_ in dev])
    fig, axs = plt.subplots(3, 2, figsize=(6.3, 7.6), gridspec_kw=dict(wspace=0.55, hspace=0.45))
    a1 = axs[0, 0]
    sc = a1.scatter(Z[:, 0], Z[:, 1], c=y, cmap=SEQ, vmin=np.floor(y.min()), vmax=0, s=26, edgecolor="white", lw=0.6)
    cb = plt.colorbar(sc, ax=a1, fraction=0.05, pad=0.02)
    cb.set_label("log$_{10}$ relative response", fontsize=6.5)
    cb.outline.set_visible(False)
    a1.set_title(f"Response (PC1 ρ = {r1:+.2f}, PC2 ρ = {r2:+.2f})", fontsize=7, color=INK)
    a1.axvline(-30, color=INK2, lw=0.6, ls=(0, (3, 2)))
    a1.set_ylim(Z[:, 1].min() - 3, Z[:, 1].max() + 3)
    # b, c: baseline resistance and zinc-thickness estimate (data/derived/r_air.csv, regime_sensors.csv)
    ra = {r["sensor"]: float(r["r_air_median"]) for r in csv.DictReader(open(ROOT / "data/derived/r_air.csv"))}
    hh = {r["sensor"]: float(r["h_0.048"]) for r in csv.DictReader(open(ROOT / "data/derived/regime_sensors.csv"))}
    for ax_, v, lab_, ttl in ((axs[0, 1], np.log10([ra[s] for s in dev]), "log$_{10}$ baseline resistance (Ω)",
                              "Baseline resistance"),
                             (axs[1, 0], np.log10([hh[s] for s in dev]), "log$_{10}$ zinc thickness h (µm)",
                              "Metallic-zinc equivalent thickness h")):
        sc_ = ax_.scatter(Z[:, 0], Z[:, 1], c=v, cmap=SEQ, s=26, edgecolor="white", lw=0.6)
        cb_ = plt.colorbar(sc_, ax=ax_, fraction=0.05, pad=0.02)
        cb_.set_label(lab_, fontsize=6.5)
        cb_.outline.set_visible(False)
        ax_.set_title(f"{ttl} (PC1 ρ = " + f"{spearmanr(Z[:, 0], v).statistic:+.2f})".replace("-", "−"), fontsize=7, color=INK)
    oranges = ["#f6c2a6", "#f6b08f", "#f0875a", "#eb6834", "#c4501f", "#8f3612"]
    specs = [(axs[1, 1], 1, "Deposition current", lambda v: f"{v / 1000:g} mA" if v >= 1000 else f"{v:g} µA"),
             (axs[2, 0], 0, "ZnCl$_2$ concentration", lambda v: f"{v:g} M"),
             (axs[2, 1], 2, "Deposition time", lambda v: f"{v:g} s")]
    for ax_, j, title, fmt in specs:
        lv = sorted(set(rec[:, j]))
        cols = [oranges[round(i * (len(oranges) - 1) / max(1, len(lv) - 1))] for i in range(len(lv))]
        for v, col in zip(lv, cols):
            m = rec[:, j] == v
            ax_.scatter(Z[m, 0], Z[m, 1], color=col, s=26, edgecolor="white", lw=0.6, label=fmt(v))
        ax_.legend(frameon=False, fontsize=6, loc="center left", bbox_to_anchor=(1.0, 0.5), handletextpad=0.1)
        rr = spearmanr(Z[:, 0], rec[:, j]).statistic
        ax_.set_title(f"{title} (PC1 ρ = {rr:+.2f})", fontsize=7, color=INK)
    for a_, l_ in zip(axs.ravel(), "abcdef"):
        a_.set_xlabel(f"PC1 ({ev[0]:.0f}% of variance)")
        a_.set_ylabel(f"PC2 ({ev[1]:.0f}%)")
        a_.grid(color=GRID, lw=0.5)
        a_.set_axisbelow(True)
        panel_label(a_, l_, x=-0.2, y=1.0)
    fig.savefig(OUT / "fig_pca.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_pca.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------- figure: the chain
def fig_chain():
    """Recipe -> microstructure -> baseline resistance -> response (E8-E10)."""
    import json
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from scipy.stats import spearmanr
    import bridging
    J = lambda n: json.load(open(ROOT / f"data/derived/{n}.json"))
    ch, bs, rg = J("chain"), J("bridging_stats"), J("regime")
    reg = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/derived/regime_sensors.csv"))}
    br = {r["sensor"]: r for r in csv.DictReader(open(ROOT / "data/derived/bridging.csv"))}
    S = sorted(reg)
    lr = np.log10([float(reg[s]["r_air"]) for s in S])
    dead = np.array([float(reg[s]["resp"]) < 0.01 for s in S])
    h = np.array([float(reg[s]["h_0.048"]) for s in S])
    gl = np.array([float(br[s]["gap_loss"]) * 100 for s in S])
    oof = list(csv.DictReader(open(ROOT / "data/derived/chain_oof.csv")))

    fig = plt.figure(figsize=(7.1, 4.8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.05], hspace=0.3, wspace=0.38)
    ax = fig.add_subplot(gs[0, :2])
    ax.set_xlim(0, 10.15)
    ax.set_ylim(0, 4.3)
    ax.axis("off")
    import matplotlib.patheffects as pe
    f = lambda v: f"{v:.2f}".replace("-", "−")
    rn = json.load(open(ROOT / "data/derived/recipe_nonlinear.json"))["models"]
    xs, w, y0, hgt = [1.075, 3.725, 6.375, 9.025], 2.15, 2.3, 0.8
    boxes = [("Recipe", "C, I, t"), ("Microstructure", "500× SEM"), ("Resistance", "baseline, in air"),
             ("Response", "to acetone")]
    fills = ["#3987e5", "#256abf", "#104281", "#0b2a52"]           # one hue, darkening along the chain
    for x, (t1, t2), fc in zip(xs, boxes, fills):
        ax.add_patch(Rectangle((x - w / 2, y0), w, hgt, fc=fc, ec="none", zorder=3))
        ax.text(x, y0 + 0.52, t1, ha="center", va="center", fontsize=6.2, color="white", fontweight="bold", zorder=4)
        ax.text(x, y0 + 0.24, t2, ha="center", va="center", fontsize=6, color="#dbe8f8", zorder=4)
    gi = bs["all"]["gap_loss"]
    links = [(f"ρ = {f(gi['h']['rho'])}", "gap obscuration\nvs zinc thickness"),
             (f"ρ = {f(ch['images->R_air']['rho'])}", f"network; {f(ch['a_images_vs_R_air_beyond_recipe']['rho'])}\nbeyond recipe"),
             (f"ρ = {f(ch['R_air_vs_response_rho'])}", f"{f(ch['b_R_air_vs_response_beyond_recipe']['rho'])}\nbeyond recipe")]
    for k, (t1, t2) in enumerate(links):
        xa, xb = xs[k] + w / 2 + 0.06, xs[k + 1] - w / 2 - 0.06
        ax.add_patch(FancyArrowPatch((xa, y0 + hgt / 2), (xb, y0 + hgt / 2), arrowstyle="-|>", mutation_scale=8,
                                     color=INK2, lw=1.1, shrinkA=0, shrinkB=0))
        xm = (xs[k] + xs[k + 1]) / 2
        ax.text(xm, y0 + hgt + 0.55, t1, ha="center", va="bottom", fontsize=6.6, color=INK, fontweight="bold")
        ax.text(xm, y0 + hgt + 0.52, t2, ha="center", va="top", fontsize=6, color=INK, linespacing=1.1)
    # links that bypass a step: square connectors below the chain (white halo where lines cross)
    halo = [pe.Stroke(linewidth=3.2, foreground="white"), pe.Normal()]
    byp = [(xs[1] + 0.4, xs[3] - 0.4, 1.75,
            f"network → response  ρ = {f(ch['images->response']['rho'])};  after resistance {f(ch['c_images_vs_response_beyond_R_air']['rho'])}"),
           (xs[0] + 0.4, xs[2] - 0.4, 1.15,
            f"recipe → resistance  ρ = {f(ch['recipe->R_air']['rho'])} (linear) to {f(rn['knn|C,I,t,h|r_air']['rho_mean'])} (nearest neighbours)"),
           (xs[0] - 0.4, xs[3] + 0.4, 0.5,
            f"recipe → response  ρ = {f(ch['recipe->response']['rho'])} (linear) to {f(rn['knn|C,I,t,h|response']['rho_mean'])} (nearest neighbours)")]
    for xa, xb, yy, txt in byp:
        ax.plot([xa, xa, xb, xb], [y0, yy, yy, y0 - 0.1], color="#9b9a96", lw=1.0, solid_joinstyle="miter",
                path_effects=halo, zorder=2)
        ax.add_patch(FancyArrowPatch((xb, y0 - 0.12), (xb, y0), arrowstyle="-|>", mutation_scale=7, color="#9b9a96",
                                     lw=1.0, shrinkA=0, shrinkB=0, zorder=2))
        ax.text((xa + xb) / 2, yy, txt, ha="center", va="center", fontsize=6, color=INK, zorder=3,
                bbox=dict(fc="white", ec="none", pad=1.2))
    panel_label(ax, "a", x=0.0, y=1.0)

    sub = gs[0, 2].subgridspec(2, 1, hspace=0.45)
    for k, (s, title) in enumerate((("0.05M-250uA-30s", "good (0.05 M, 250 µA, 30 s)"),
                                    ("0.1M-5000uA-30s", "non-responding (0.1 M, 5 mA, 30 s)"))):
        im = next(i for i in C.index_sem(SEM500) if i.sensor == s)
        a = bridging.load(im.path)
        d, lost = bridging.describe(a)
        m = int(2 * d["pitch_px"])
        L = np.zeros(a.shape, bool)
        L[m:-m, bridging.WIN:-bridging.WIN] = lost
        y0, x0 = 300, 300
        crop, Lc = a[y0:y0 + 500, x0:x0 + 750], L[y0:y0 + 500, x0:x0 + 750]
        rgb = np.stack([crop] * 3, -1)
        rgb[Lc] = 0.45 * rgb[Lc] + 0.55 * np.array([0.106, 0.686, 0.478])
        axi = fig.add_subplot(sub[k])
        axi.imshow(rgb)
        axi.set_xticks([])
        axi.set_yticks([])
        axi.set_title(f"{title}: {d['gap_loss'] * 100:.1f}% obscured", fontsize=6, color=INK, pad=2)
        if k == 0:
            panel_label(axi, "b", x=-0.3, y=1.2)
        axi.plot([30, 30 + 50 / 0.27], [470, 470], color="white", lw=1.5)
        axi.text(30 + 25 / 0.27, 455, "50 µm", color="white", fontsize=6, ha="center", va="bottom")

    def scatter(axs, x, y, d, xl, yl, title, logx=False):
        for msk, col, nm in ((~d, BLUE, "response ≥ 1%"), (d, ORANGE, "response < 1%")):
            axs.scatter(x[msk], y[msk], s=14, color=col, edgecolor="white", lw=0.4, label=nm, zorder=3)
        if logx:
            axs.set_xscale("log")
        axs.set_xlabel(xl)
        axs.set_ylabel(yl)
        axs.set_title(title, fontsize=6.4, color=INK)
        axs.grid(color=GRID, lw=0.5)
        axs.set_axisbelow(True)

    ylab = "log$_{10}$ baseline resistance (Ω)"
    a3 = fig.add_subplot(gs[1, 0])
    r = rg["1_spearman"]["h_0.048"]["log_r_air"]
    scatter(a3, h, lr, dead, "metallic-Zn equivalent thickness h (µm)", ylab,
            f"Deposition: ρ = {f(r[0])} (72 sensors)", logx=True)
    panel_label(a3, "c", x=-0.22, y=1.02)
    a3.legend(frameon=False, fontsize=6, loc="lower left", handletextpad=0.1)
    a4 = fig.add_subplot(gs[1, 1])
    scatter(a4, gl, lr, dead, "gaps obscured (% of image area)", ylab,
            f"Physical descriptor: ρ = {f(gi['log_r_air']['rho'])} (72)")
    panel_label(a4, "d", x=-0.22, y=1.02)
    a5 = fig.add_subplot(gs[1, 2])
    po = np.array([float(o["img_pred_r_air"]) for o in oof])
    mo = np.array([float(o["log_r_air"]) for o in oof])
    do = np.array([10 ** float(o["log_response"]) < 0.01 for o in oof])
    scatter(a5, po, mo, do, "held-out prediction (log$_{10}$ Ω)", ylab,
            f"Network: ρ = {f(ch['images->R_air']['rho'])} (56, p = 0.001)")
    lim = [min(po.min(), mo.min()) - 0.2, max(po.max(), mo.max()) + 0.2]
    a5.plot(lim, lim, color=INK2, lw=0.5, ls=(0, (3, 2)))
    panel_label(a5, "e", x=-0.22, y=1.02)
    fig.savefig(OUT / "fig_chain.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_chain.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------ figure: heating curves (Supp.)
def fig_heating():
    """Resistance on first heating (existing records, 29 sensors), against baseline resistance."""
    import json
    rows = list(csv.DictReader(open(ROOT / "data/derived/heating_rt.csv")))
    cur = np.load(ROOT / "data/derived/heating_rt_curves.npy", allow_pickle=True).item()
    summ = json.load(open(ROOT / "data/derived/heating_rt.json"))
    from scipy.ndimage import median_filter
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.1, 2.9), gridspec_kw=dict(wspace=0.4))
    for r in rows:
        T, R = cur[r["sensor"]]
        m = (T > 100) & (T < 290)
        lr_ = median_filter(np.log10(R[m]), size=31, mode="nearest")          # removes single-reading spikes
        ref = np.median(lr_[np.abs(T[m] - 150) < 5]) if np.any(np.abs(T[m] - 150) < 5) else lr_[0]
        dead_ = float(r["response"]) < 0.01
        a1.plot(T[m], lr_ - ref, color=ORANGE if dead_ else BLUE, lw=0.8, alpha=0.9, zorder=3 if dead_ else 2)
    a1.plot([], [], color=BLUE, lw=0.9, label="response ≥ 1%")
    a1.plot([], [], color=ORANGE, lw=0.9, label="response < 1%")
    a1.legend(frameon=False, fontsize=6, loc="lower left")
    a1.axhline(0, color=INK2, lw=0.5)
    a1.set_xlabel("Temperature on first heating (°C)")
    a1.set_ylabel("log$_{10}$ R / R(150 °C), running median")
    a1.grid(color=GRID, lw=0.5)
    a1.set_axisbelow(True)
    panel_label(a1, "a", x=-0.12, y=1.0)
    x = np.log10([float(r["r_air"]) for r in rows])
    y = np.array([float(r["rho_TR"]) for r in rows])
    d = np.array([float(r["response"]) < 0.01 for r in rows])
    lk = np.array([r["split"] == "locked" for r in rows])
    for msk, col, nm in ((~d, BLUE, "response ≥ 1%"), (d, ORANGE, "response < 1%")):
        a2.scatter(x[msk & ~lk], y[msk & ~lk], s=18, color=col, edgecolor="white", lw=0.5, label=nm, zorder=3)
        a2.scatter(x[msk & lk], y[msk & lk], s=18, marker="s", color=col, edgecolor="white", lw=0.5, zorder=3)
    a2.scatter([], [], s=18, marker="s", color=INK2, label="locked test sensor")
    a2.axhline(0, color=INK2, lw=0.5)
    a2.set_xlabel("log$_{10}$ baseline resistance (Ω)")
    a2.set_ylabel("Spearman ρ, temperature vs resistance\n(100–290 °C on first heating)")
    st = summ["rho_TR_vs_log_r_air"]
    a2.set_title(f"ρ = {st['rho']:.2f}, p = {st['p_two_sided']:.4f} ({len(rows)} sensors)".replace("-", "−"), fontsize=7, color=INK)
    a2.legend(frameon=False, fontsize=6, loc="center right")
    a2.grid(color=GRID, lw=0.5)
    a2.set_axisbelow(True)
    panel_label(a2, "b", x=-0.2, y=1.0)
    fig.savefig(OUT / "fig_heating.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_heating.png", bbox_inches="tight", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_recipes()
    fig_models()
    fig_replicates()
    fig_types()
    fig_2500x()
    fig_tsne()
    fig_pca()
    fig_tilemaps()
    fig_rair()
    fig_chain()
    fig_heating()
    print("figures written to", OUT)
