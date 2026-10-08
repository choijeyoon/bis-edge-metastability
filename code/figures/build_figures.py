"""Figures 1-4 and S1-S4 of "Pathogen-related images reduce cortical edge metastability" (Network Neuroscience).
Every number is read from results/tables/*.csv or data/; nothing is typed in. Canvas 174 mm, 7-pt floor, one typeface.
Surface renders (figures/brains/*.png) come from code/figures/render_brains.py (brainspace + surfplot).
usage: python code/figures/build_figures.py [1 2 3 4 S1 S2 S3 S4]   (default: all)
"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gamma as gam
from PIL import Image, ImageChops
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap, to_rgb
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
T = ROOT / "results" / "tables"; DATA = ROOT / "data"; OUT = ROOT / "figures"; BR = OUT / "brains"
SEED = 20260911   # all resampling (null displays, bootstrap intervals) uses this seed
OUT.mkdir(parents=True, exist_ok=True)
MM = 1 / 25.4
plt.rcParams.update({"pdf.fonttype": 42, "savefig.dpi": 300, "hatch.linewidth": 0.4, "axes.spines.top": False, "axes.spines.right": False})


def fix_minus(fig):
    from matplotlib.text import Text
    import re
    for t in fig.findobj(Text):
        s = t.get_text()
        if "-" in s: t.set_text(re.sub(r"(?<![\w.$])-(?=\.?\d)", "\u2212", s))


def audit(fig, tol_mm=0.3):
    """Report overlapping text/axes boxes and out-of-canvas elements, in mm."""
    fix_minus(fig)
    fig.canvas.draw(); r = fig.canvas.get_renderer(); dpi = fig.dpi
    boxes = []
    for ax in fig.axes:
        boxes.append(("axes", ax.get_tightbbox(r)))
        for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels():
            if t.get_text().strip():
                boxes.append((t.get_text()[:25], t.get_window_extent(r)))
    for t in fig.texts:
        boxes.append((t.get_text()[:25], t.get_window_extent(r)))
    Wpx, Hpx = fig.get_size_inches() * dpi
    problems = 0
    for name, bb in boxes:
        if bb.x0 < -1 or bb.y0 < -1 or bb.x1 > Wpx + 1 or bb.y1 > Hpx + 1:
            print(f"  EDGE: {name!r} x0={bb.x0:.0f} y0={bb.y0:.0f} x1={bb.x1:.0f} y1={bb.y1:.0f} W={Wpx:.0f} H={Hpx:.0f}"); problems += 1
    texts = [(n, b) for n, b in boxes if n != "axes"]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            a, b = texts[i][1], texts[j][1]
            ox = min(a.x1, b.x1) - max(a.x0, b.x0); oy = min(a.y1, b.y1) - max(a.y0, b.y0)
            if ox > tol_mm / 25.4 * dpi and oy > tol_mm / 25.4 * dpi:
                print(f"  OVERLAP: {texts[i][0]!r} x {texts[j][0]!r} ({ox/dpi*25.4:.1f} x {oy/dpi*25.4:.1f} mm)"); problems += 1
    print(f"problems {problems}")
    return problems


def cohen_d(x): return x.mean() / x.std(ddof=1)


def d_ci(x, B=5000):
    rng = np.random.default_rng(SEED)
    bs = [cohen_d(x[rng.integers(0, len(x), len(x))]) for _ in range(B)]
    return np.percentile(bs, [2.5, 97.5])


def panel_label(ax, s, dx=-0.18, dy=1.05):
    ax.text(dx, dy, s, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom", ha="left")


def sel(lag, em, gsr, parc="schaefer400", ztime="cond", blockset="all"):
    q = deltas[(deltas.lag == lag) & (deltas.em == em) & (deltas.gsr == gsr) & (deltas.parc == parc) & (deltas.ztime == ztime) & (deltas.blockset == blockset)]
    return q.set_index("sid").loc[ids, "delta"].values

g1 = pd.read_csv(DATA / "edge_metastability" / "global_S1.csv"); g2 = pd.read_csv(DATA / "edge_metastability" / "global_S2.csv")
demo = pd.read_csv(DATA / "participants.csv").set_index("SubjectID")
ids = g1.SubjectID.tolist(); sex = demo.loc[ids, "Sex"].values
summ = pd.read_csv(T / "spec_curve_summary_S1.csv"); joint = pd.read_csv(T / "spec_curve_joint_S1.csv")
deltas = pd.read_csv(DATA / "edge_metastability" / "spec_curve_subject_deltas_S1.csv")
grid = pd.read_csv(T / "gsr_grid_sex_HI.csv"); prim = pd.read_csv(T / "primary_H1.csv").iloc[0]

ORANGE, TEAL, NEU = "#D8543F", "#2E7EB8", "#7f7f7f"
INF_L, NEU_L = "#F4B8AE", "#cfcfcf"
YEO = {"Visual": "#7A1E7A", "Somatomotor": "#4A9BD5", "DorsalAttention": "#1B7B3A", "VentralAttention": "#C8A2C8",
       "Limbic": "#D9D67E", "Control": "#E69422", "DefaultMode": "#C64B4B"}
NAMES = list(YEO); SHORT = ["VIS", "SOM", "DAN", "VAN", "LIM", "CON", "DMN"]
blues = LinearSegmentedColormap.from_list("red_blue", ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f2f2f2"])
netstats = pd.read_csv(T / "network_stats_sync_S1.csv"); nodestats = pd.read_csv(T / "node_stats_edgewise_S1.csv")
nd = {g: pd.read_csv(DATA / "edge_metastability" / f"network_delta_sync_{g}_lag0_S1.csv", index_col=0).loc[ids] for g in ("none", "gsr")}
yeo = np.array([{"Vis": 1, "SomMot": 2, "DorsAttn": 3, "SalVentAttn": 4, "Limbic": 5, "Cont": 6, "Default": 7}[s.split("_")[2]] for s in pd.read_csv(DATA / "atlases" / "schaefer400_7net_labels.tsv", sep="\t").name])
TR, DUMMY, BLK = 2.0, 4, 9


def trim(path, pad=6):
    im = Image.open(path).convert("RGB"); bg = Image.new("RGB", im.size, (255, 255, 255))
    b = ImageChops.difference(im, bg).getbbox(); return im.crop((b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad))


def img_panel(ax, path):
    ax.imshow(trim(path)); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_visible(False)


def violin(ax, x, data, color, width=0.7, side="both", pts=True, box=True, jitter=0.08, rng=np.random.default_rng(0), ms=7):
    kde = stats.gaussian_kde(data); yy = np.linspace(data.min() - 0.1 * np.ptp(data), data.max() + 0.1 * np.ptp(data), 200)
    dens = kde(yy); dens = dens / dens.max() * width / 2
    if side in ("both", "left"): ax.fill_betweenx(yy, x - dens, x, color=color, alpha=0.35, lw=0)
    if side in ("both", "right"): ax.fill_betweenx(yy, x, x + dens, color=color, alpha=0.35, lw=0)
    if pts:
        xs = x + rng.uniform(-jitter, jitter, len(data)) if side == "both" else x + (0.06 + rng.uniform(0, jitter, len(data))) * (1 if side == "left" else -1)
        ax.scatter(xs, data, s=ms, color=color, lw=0, zorder=3, alpha=0.9)
    if box:
        q1, med, q3 = np.percentile(data, [25, 50, 75])
        ax.plot([x, x], [q1, q3], color="k", lw=1.6, zorder=4, solid_capstyle="butt"); ax.plot([x - 0.08, x + 0.08], [med, med], color="k", lw=1.2, zorder=5)


def zscore(ts):
    mu = ts.mean(1, keepdims=True); sd = ts.std(1, keepdims=True, ddof=1); sd[sd == 0] = 1; return (ts - mu) / sd


def rts(z):
    n = z.shape[0]; s = z.sum(0); s2 = (z * z).sum(0); return 0.5 * (s ** 2 - s2) / (n * (n - 1) / 2)

FONT = "Liberation Sans"; FS, FS_T, FS_L = 7, 8, 9
plt.rcParams.update({"font.family": FONT, "font.size": FS, "axes.labelsize": FS, "axes.titlesize": FS, "xtick.labelsize": FS, "ytick.labelsize": FS,
                     "legend.fontsize": FS, "svg.fonttype": "none", "axes.linewidth": 0.5,
                     "mathtext.fontset": "custom", "mathtext.rm": FONT, "mathtext.it": f"{FONT}:italic", "mathtext.bf": f"{FONT}:bold",
                     "xtick.major.size": 2, "ytick.major.size": 2, "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.pad": 2, "ytick.major.pad": 2})
DARK = "#222222"; GRID = "#cccccc"; FW = 174.0
FH = 100.0


def mm(x, y, w, h): return [x / FW, 1 - (y + h) / FH, w / FW, h / FH]
def fx(x): return x / FW
def fy(y): return 1 - y / FH


def head(fig, x, y, letter, text=None, color=DARK):
    fig.text(fx(x), fy(y), letter, fontsize=FS_L, fontweight="bold", va="baseline", ha="left")
    if text: fig.text(fx(x + 5), fy(y), text, fontsize=FS_T, fontweight="bold", va="baseline", ha="left", color=color)


def zero(ax, axis="y"): (ax.axhline if axis == "y" else ax.axvline)(0, color=DARK, lw=0.5, ls=(0, (2, 2)), zorder=1)


def font_report(fig):
    used = sorted({round(tx.get_fontsize(), 1) for ax in fig.axes for tx in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() if tx.get_text().strip()}
                  | {round(tx.get_fontsize(), 1) for tx in fig.texts} | {round(t.get_fontsize(), 1) for ax in fig.axes for lg in [ax.get_legend()] if lg for t in lg.get_texts()})
    print("font sizes used:", used)


def save(fig, name):
    audit(fig); font_report(fig)
    for ext in ("pdf", "png", "svg"): fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)


def dbar(ax, x, d, lo, hi, color, mk="o", mfc=None):
    ax.errorbar(x, d, yerr=[[d - lo], [hi - d]], fmt=mk, color=color, mfc=color if mfc is None else mfc, ms=3.5, capsize=1.5, lw=0.7, zorder=3)


# ============================================================ Figure 1
PI, PJ = 79, 174
DARK = "#222222"; ARROW = "#555555"; LINK = "#9a9a9a"; GRID = "#cccccc"; HATCH = "#b5b5b5"
FW = 174.0
ZOOM_BLOCK = 30


def arrow(fig, p0, p1, text=None, tpos=None, tha="center", tva="bottom", rot=0):
    fig.add_artist(FancyArrowPatch((fx(p0[0]), fy(p0[1])), (fx(p1[0]), fy(p1[1])), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=6, lw=0.7, color=ARROW, shrinkA=0, shrinkB=0))
    if text: fig.text(fx(tpos[0]), fy(tpos[1]), text, fontsize=FS, color=ARROW, ha=tha, va=tva, rotation=rot)


def clean(ax, left=True, bottom=True):
    ax.spines["left"].set_visible(left); ax.spines["bottom"].set_visible(bottom)
    if not left: ax.set_yticks([])
    if not bottom: ax.set_xticks([])


def hatch_excluded(ax, y0, y1):
    ax.add_patch(Rectangle((0, y0), 18, y1 - y0, facecolor="white", edgecolor=HATCH, hatch="////", lw=0, zorder=2.5, alpha=0.8))


def parcel_pixels(img_shape, pids):
    from brainspace.datasets import load_conte69, load_parcellation
    from brainspace.mesh.mesh_elements import get_points
    lh, _ = load_conte69(); lab = load_parcellation("schaefer", scale=400, join=True)[:lh.n_points]; P = get_points(lh)
    H, W = img_shape[:2]; y0, y1 = P[:, 1].min(), P[:, 1].max(); z0, z1 = P[:, 2].min(), P[:, 2].max()
    return {pid: ((y1 - c[1]) / (y1 - y0) * W, (z1 - c[2]) / (z1 - z0) * H) for pid in pids for c in [P[lab == pid + 1].mean(0)]}


def figure1():
    global FH; FH = 162.0
    sid = 1
    ts = np.load(DATA / "example" / f"sub-{sid:02d}_S1_schaefer400.npy").astype(float)[:, DUMMY:]
    ev = pd.read_csv(DATA / "example" / f"sub-{sid:02d}_ses-S1_task-picrate_events.tsv", sep="\t")
    T_ = ts.shape[1]; t = np.arange(T_) * TR; TMAX = T_ * TR
    z = zscore(ts); R = rts(z); m2 = z.mean(0) ** 2
    onsets = ev.onset.values - DUMMY * TR; types = ev.trial_type.values
    win = [(o, ty) for o, ty in zip(onsets, types) if o >= 0]
    blocks = {"neutral": [], "infection": []}
    for o, ty in win:
        s0 = int(np.floor(o / TR))
        if s0 + BLK <= T_: blocks[ty].append(ts[:, s0:s0 + BLK])
    Rc = {ty: rts(zscore(np.concatenate(b, 1))) for ty, b in blocks.items()}
    em = {ty: v.std(ddof=1) for ty, v in Rc.items()}

    fig = plt.figure(figsize=(FW * MM, FH * MM))
    XL, XR = 20.0, 160.0
    # ================================================ a: run, stimuli, one block zoomed
    head(fig, 5, 5, "a", "Task")
    axa = fig.add_axes(mm(10, 11, 90, 3.6))
    for i, (o, ty) in enumerate(zip(ev.onset.values, types)):
        c = ORANGE if ty == "infection" else NEU
        axa.add_patch(Rectangle((o, 0), 18, 1, facecolor=c if i > 0 else "white", edgecolor="white" if i > 0 else HATCH, hatch="////" if i == 0 else None, lw=0.5))
    for i in range(6):   # opening sequence N N N I I I, then alternation
        axa.text(ev.onset.values[i] + 9, 1.25, "I" if types[i] == "infection" else "N", ha="center", va="bottom", fontsize=FS, color=ORANGE if types[i] == "infection" else NEU)
    axa.text(ev.onset.values[6] + 9, 1.25, "N I N I … alternating", ha="left", va="bottom", fontsize=FS, color=DARK)
    zb = ev.onset.values[ZOOM_BLOCK]
    axa.add_patch(Rectangle((zb, 0), 18, 1, facecolor="none", edgecolor=DARK, lw=0.8, zorder=5, clip_on=False))
    axa.set_xlim(0, 720); axa.set_ylim(0, 1); clean(axa, left=False); axa.set_xticks([0, 360, 720])
    axa.tick_params(axis="x", length=1.5, pad=1.5); axa.set_xlabel("Time in run (s), 40 blocks × 18 s", labelpad=1)
    for k, (f, lab, c) in enumerate([("img_neutral_c.png", "neutral", NEU), ("img_infection_c.png", "infection", ORANGE)]):
        iax = fig.add_axes(mm(10 + k * 24, 24, 22, 15))
        if (DATA / "stimuli" / f).exists(): img_panel(iax, DATA / "stimuli" / f)
        else:   # the stimulus images are not redistributed; draw a labelled placeholder of the same proportions
            iax.add_patch(Rectangle((0.01, 0.01), 0.98, 0.98, facecolor="#f4f4f4", edgecolor=c, lw=0.9)); iax.set_xlim(0, 1); iax.set_ylim(0, 1)
            iax.text(0.5, 0.5, f"{lab} image\n+ ratings, 9 s", ha="center", va="center", fontsize=FS, color="#555555"); iax.set_xticks([]); iax.set_yticks([])
            for sp in iax.spines.values(): sp.set_visible(False)
        fig.add_artist(Rectangle((fx(10 + k * 24), fy(43.6)), fx(2.4), 2.4 / FH, transform=fig.transFigure, facecolor=c, edgecolor="none"))
        fig.text(fx(13.4 + k * 24), fy(42.4), lab, fontsize=FS, color=DARK, va="center", ha="left")
    axb = fig.add_axes(mm(60, 24, 36, 15))
    tt = np.linspace(0, 36, 500); hrf = gam.pdf(tt, 6) - gam.pdf(tt, 16) / 6; dt = tt[1] - tt[0]
    box = ((tt >= 0) & (tt < 9)).astype(float); pred = np.convolve(box, hrf, mode="full")[:len(tt)] * dt; pred /= pred.max()
    axb.fill_between(tt, 0, box * 1.08, color="#E8DFC4", lw=0); axb.plot(tt, pred, color=DARK, lw=0.8)
    axb.text(4.5, 1.14, "image", ha="center", va="bottom", fontsize=FS, color=DARK)
    axb.text(15, 1.14, "fixation", ha="center", va="bottom", fontsize=FS, color=DARK)
    axb.text(22, 0.7, "predicted\nBOLD", fontsize=FS, color=DARK, va="center")
    for lag, yy, lab, fc in [(0, -0.6, "9-TR window, at onset", DARK), (2, -1.05, "shifted +2 TR", "white")]:
        axb.add_patch(Rectangle((lag * TR, yy), 18, 0.22, facecolor=fc, edgecolor=DARK, lw=0.5))
        for k in range(1, BLK): axb.plot([lag * TR + k * TR] * 2, [yy, yy + 0.22], color="white" if fc == DARK else DARK, lw=0.4)
        axb.text(lag * TR + 18.8, yy + 0.11, lab, fontsize=FS, va="center", color=DARK)
    axb.set_xlim(0, 36); axb.set_xticks([0, 9, 18, 27, 36]); axb.set_xlabel("Time from block onset (s)", labelpad=1)
    clean(axb, left=False); axb.set_ylim(-1.15, 1.75)
    for xa, xb_ in [(zb, 0), (zb + 18, 18)]:
        fig.add_artist(ConnectionPatch(xyA=(xa, 0), coordsA=axa.transData, xyB=(xb_, 1.75), coordsB=axb.transData, color=LINK, lw=0.5, ls=(0, (2, 1.5)), zorder=0))
    # ================================================ b: parcellation, two parcels, one edge
    head(fig, 105, 5, "b", "Parcel time series")
    bax = fig.add_axes(mm(105, 6.5, 30, 22)); brain = Image.open(BR / "yeo400_lh_lat.png").convert("RGB"); bax.imshow(brain); bax.set_xticks([]); bax.set_yticks([])
    for sp in bax.spines.values(): sp.set_visible(False)
    max_ = fig.add_axes(mm(105, 27.5, 30, 22)); max_.imshow(Image.open(BR / "yeo400_lh_med.png").convert("RGB")); max_.set_xticks([]); max_.set_yticks([])
    for sp in max_.spines.values(): sp.set_visible(False)
    fig.text(fx(120), fy(49.5), "Schaefer 400 parcels, Yeo 7 networks", ha="center", va="center", fontsize=FS, color=DARK)
    bb = ImageChops.difference(brain, Image.new("RGB", brain.size, (255, 255, 255))).getbbox()
    pix = {k: (bb[0] + u, bb[1] + v) for k, (u, v) in parcel_pixels((bb[3] - bb[1], bb[2] - bb[0]), [PI, PJ]).items()}
    seg = slice(0, 60); tseg = t[seg]
    rows = [(z[PI, seg], YEO["DorsalAttention"], "$z_i$", PI), (z[PJ, seg], YEO["DefaultMode"], "$z_j$", PJ), (z[PI, seg] * z[PJ, seg], DARK, "$z_i z_j$", None)]
    for r_, (y, c, lab, pid) in enumerate(rows):
        ax = fig.add_axes(mm(140, 8.5 + r_ * 10, 20, 7.5))
        ax.plot(tseg, y, color=c, lw=0.7); ax.axhline(0, color=GRID, lw=0.4, zorder=0)
        ax.set_xlim(0, 120); clean(ax, left=False, bottom=(r_ == 2)); ax.set_ylabel(lab, rotation=0, ha="right", va="center", labelpad=3, color=c, fontsize=FS)
        if r_ == 2: ax.set_xticks([0, 60, 120]); ax.set_xlabel("Time (s)", labelpad=1)
        if r_ == 1: ax.text(-0.2, 1.2, "×", transform=ax.transAxes, fontsize=FS_T, ha="center", va="center", color=DARK)
        if r_ == 2: ax.text(-0.2, 1.2, "=", transform=ax.transAxes, fontsize=FS_T, ha="center", va="center", color=DARK)
        if pid is not None:
            bax.plot(*pix[pid], "o", ms=2.8, mfc="white", mec=c, mew=0.8, zorder=6)
            fig.add_artist(ConnectionPatch(xyA=pix[pid], coordsA=bax.transData, xyB=(-0.33, 0.5), coordsB=ax.transAxes, color=c, lw=0.6, zorder=0))
    # ================================================ c: edge time series, full width
    head(fig, 5, 55.5, "c", "Edge time series, all 79,800 pairs (1 in 25 shown)")
    axd = fig.add_axes(mm(XL, 61.5, XR - XL, 26))
    iu, ju = np.triu_indices(400, 1); order = np.lexsort((yeo[ju], yeo[iu])); iu, ju = iu[order], ju[order]
    step = 25; iu_s, ju_s = iu[::step], ju[::step]; E = z[iu_s] * z[ju_s]; nE = E.shape[0]
    axd.imshow(E, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, extent=[0, TMAX, nE, 0], interpolation="nearest")
    for o, ty in win:
        axd.add_patch(Rectangle((o, -nE * 0.10), 18, nE * 0.06, facecolor=ORANGE if ty == "infection" else NEU, edgecolor="none", clip_on=False))
    hatch_excluded(axd, 0, nE)
    axd.set_ylim(nE, 0); axd.set_xlim(0, TMAX); axd.set_yticks([]); axd.set_xticks([])
    for s in axd.spines.values(): s.set_visible(True); s.set_linewidth(0.5)
    cols = np.array([to_rgb(YEO[n]) for n in NAMES]); strip = np.stack([cols[yeo[iu_s] - 1], cols[yeo[ju_s] - 1]], 1)
    sax = fig.add_axes(mm(XL - 3.4, 61.5, 2.6, 26)); sax.imshow(strip, aspect="auto", interpolation="nearest")
    sax.set_xticks([]); sax.set_yticks([]); [sp.set_visible(False) for sp in sax.spines.values()]
    sax.text(0.25, 1.03, "i", transform=sax.transAxes, ha="center", va="bottom", fontsize=FS, style="italic"); sax.text(0.75, 1.03, "j", transform=sax.transAxes, ha="center", va="bottom", fontsize=FS, style="italic")
    sax.text(-0.8, 0.5, "edges, by network pair", transform=sax.transAxes, rotation=90, ha="right", va="center", fontsize=FS)
    cax = fig.add_axes(mm(136, 52.6, 24, 1.8)); cb = fig.colorbar(plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-3, 3)), cax=cax, orientation="horizontal")
    cb.set_ticks([-3, 0, 3]); cb.ax.tick_params(labelsize=FS, length=1.5, pad=1); cb.outline.set_linewidth(0.4)
    cax.text(-0.08, 0.5, "$z_i z_j$", transform=cax.transAxes, ha="right", va="center", fontsize=FS)
    # ================================================ d: R(t) on the same time axis, inset R vs m^2
    head(fig, 5, 93, "d", "Global cofluctuation R(t), mean over edges, same time axis")
    axe = fig.add_axes(mm(XL, 95.5, XR - XL, 22))
    for o, ty in win: axe.axvspan(o, o + 18, color=ORANGE if ty == "infection" else NEU, alpha=0.13, lw=0)
    axe.plot(t, R, color=ORANGE, lw=0.7); hatch_excluded(axe, -0.1, 5.4)
    axe.set_xlim(0, TMAX); axe.set_xticks([0, 180, 360, 540, 720]); axe.set_xlabel("Time in run (s)", labelpad=1.5); axe.set_ylabel("R(t)", labelpad=2)
    axe.set_ylim(-0.1, 5.4); axe.set_yticks([0, 2, 4])
    # ================================================ e: EM from the concatenated windows
    head(fig, 5, 128.5, "e", "Edge metastability: windows of d, concatenated by condition; EM = SD of R(t)")
    for k, (ty, c, lab) in enumerate([("neutral", NEU, "neutral"), ("infection", ORANGE, "infection")]):
        ax = fig.add_axes(mm(XL + k * 73, 132, 60, 20))
        v = Rc[ty]; n = len(v); x = np.arange(n) * TR; mu, sd = v.mean(), em[ty]
        for b in range(1, n // BLK): ax.axvline(b * BLK * TR, color=GRID, lw=0.3, zorder=0)
        ax.plot(x, v, color=c, lw=0.7); ax.axhline(mu, color=DARK, lw=0.5, ls=(0, (3, 2)))
        xs = n * TR + 8
        ax.plot([xs, xs], [mu - sd, mu + sd], color=DARK, lw=1.0, solid_capstyle="butt", clip_on=False)
        for yy in (mu - sd, mu + sd): ax.plot([xs - 3, xs + 3], [yy, yy], color=DARK, lw=0.6, clip_on=False)
        ax.text(xs + 8, mu, f"SD\n{sd:.2f}", fontsize=FS, va="center", ha="left", color=DARK)
        ax.set_xlim(0, n * TR); ax.set_ylim(-0.1, 4.2); ax.set_yticks([0, 1, 2, 3]); clean(ax)
        ax.text(0.015, 0.97, f"{lab} windows, {n // BLK} × 9 TR", transform=ax.transAxes, ha="left", va="top", fontsize=FS, color=c)
        ax.text(0.015, 0.80, f"EM = {sd:.2f}", transform=ax.transAxes, ha="left", va="top", fontsize=FS, color=DARK)
        ax.set_xticks([0, 90, 180, 270, 360]); ax.set_xlabel("Concatenated time (s)", labelpad=1.5)
        if k == 0: ax.set_ylabel("R(t)", labelpad=2)
    # ================================================ flow arrows, right margin
    arrow(fig, (164, 9), (164, 59.5), "all pairs", tpos=(167.5, 34), tha="center", tva="center", rot=90)
    arrow(fig, (164, 62), (164, 94), "mean over edges", tpos=(167.5, 78), tha="center", tva="center", rot=90)
    arrow(fig, (164, 99), (164, 131), "windows by condition", tpos=(167.5, 115), tha="center", tva="center", rot=90)
    print("Figure 1"); audit(fig)
    # type-size floor check
    small = sorted({round(tx.get_fontsize(), 1) for ax in fig.axes for tx in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels() if tx.get_text().strip()} | {round(tx.get_fontsize(), 1) for tx in fig.texts})
    print("font sizes used:", small)
    for ext in ("pdf", "png", "svg"): fig.savefig(OUT / f"figure1.{ext}")
    plt.close(fig); print({k: round(v, 4) for k, v in em.items()})


# ============================================================ Figure 2: primary result and parcel maps
def figure2():
    global FH; FH = 138.0
    fig = plt.figure(figsize=(FW * MM, FH * MM))
    neu, inf, y = g1.EM_Neu.values, g1.EM_Inf.values, g1.Delta.values
    head(fig, 5, 5, "a", "Pre-specified test, session 1 (n = 44)")
    ax = fig.add_axes(mm(14, 12, 40, 37))
    rng0 = np.random.default_rng(0); jx = rng0.uniform(-0.08, 0.08, 44)
    violin(ax, 0, neu, NEU, width=0.8, pts=False, ms=8); violin(ax, 1, inf, ORANGE, width=0.8, pts=False, ms=8)
    for i in range(44): ax.plot([jx[i], 1 + jx[i]], [neu[i], inf[i]], color="#c4c4c4", lw=0.35, zorder=2)
    ax.scatter(jx, neu, s=8, color=NEU, lw=0, zorder=3, alpha=0.9); ax.scatter(1 + jx, inf, s=8, color=ORANGE, lw=0, zorder=3, alpha=0.9)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["neutral", "infection"]); ax.set_xlim(-0.65, 1.65); ax.set_ylabel("Edge metastability (EM)")
    top = max(neu.max() + 0.1 * np.ptp(neu), inf.max() + 0.1 * np.ptp(inf))          # violin extent, not just the points
    yb = top * 1.06; ax.plot([0, 0, 1, 1], [yb, yb * 1.03, yb * 1.03, yb], color=DARK, lw=0.6)
    ax.text(0.5, yb * 1.05, f"p = {prim.p_perm:.3f}", ha="center", va="bottom", fontsize=FS, color=DARK); ax.set_ylim(0, yb * 1.24); ax.set_yticks([0, 0.5, 1.0, 1.5])
    for lab_, c in zip(ax.get_xticklabels(), (NEU, ORANGE)): lab_.set_color(c)
    head(fig, 62, 5, "b", "Difference")
    ax = fig.add_axes(mm(70, 11, 18, 38))
    violin(ax, 0, y, ORANGE, width=0.9, jitter=0.15, ms=8); zero(ax)
    ax.set_xticks([]); ax.spines["bottom"].set_visible(False); ax.set_xlim(-0.6, 0.6); ax.set_ylabel("ΔEM (infection − neutral)")
    yl = ax.get_ylim(); ax.set_ylim(yl[0], yl[1] + 0.32 * (yl[1] - yl[0]))
    ax.text(0.5, 0.99, f"d = {prim.d:.2f}\n{int(prim.n_down)} of 44 < 0", transform=ax.transAxes, ha="center", va="top", fontsize=FS, color=DARK)
    head(fig, 98, 5, "c", "Sign-flip permutation, two-tailed")
    ax = fig.add_axes(mm(106, 11, 56, 38))
    X = np.random.default_rng(SEED).choice([-1.0, 1.0], size=(10000, len(y))) * y[None, :]   # same draws as the test
    null = X.mean(1) / X.std(1, ddof=1)
    ax.hist(null, bins=40, color="#dddddd", lw=0); ax.axvline(prim.d, color=ORANGE, lw=1.4); zero(ax, "x")
    ax.set_xlabel("Cohen's d under the null"); ax.set_ylabel("Draws (10,000)"); ax.set_yticks([]); ax.set_xticks([-0.6, -0.3, 0, 0.3, 0.6])
    ax.text(prim.d - 0.015, ax.get_ylim()[1] * 0.97, f"observed\np = {prim.p_perm:.3f}", color=ORANGE, ha="right", va="top", fontsize=FS)
    for r, (gsr, ttl, y0) in enumerate([("none", "Parcel-wise d, edge-wise definition, no GSR", 60), ("gsr", "Parcel-wise d, edge-wise definition, GSR", 99)]):
        n_sig = int(((nodestats.gsr == gsr) & (nodestats.lag == 0) & (nodestats.q < .05)).sum())
        head(fig, 5, y0, "de"[r], f"{ttl}; outlined: q < .05, {n_sig} of 400 parcels")
        ax = fig.add_axes(mm(10, y0 + 3, 142, 33)); img_panel(ax, BR / f"map_{gsr}.png")
        for k, lab_ in enumerate(["L lateral", "L medial", "R lateral", "R medial"]):
            ax.text(0.125 + k * 0.25, -0.02, lab_, transform=ax.transAxes, ha="center", va="top", fontsize=FS, color="#666666")
    cax = fig.add_axes(mm(156, 78, 2.6, 40)); cb = fig.colorbar(plt.cm.ScalarMappable(cmap=blues, norm=plt.Normalize(-0.7, 0)), cax=cax)
    cb.set_ticks([-0.7, -0.35, 0]); cb.ax.tick_params(labelsize=FS, length=2, pad=1.5); cb.outline.set_linewidth(0.4); cb.set_label("Cohen's d", fontsize=FS, labelpad=2)
    print("Figure 2"); save(fig, "figure2")


# ============================================================ Figure 4: networks, two conventions
def figure4():  # networks
    global FH; FH = 120.0
    fig = plt.figure(figsize=(FW * MM, FH * MM))
    cw, x0 = 20.0, 14.0
    for r, (gsr, ttl, c, y0) in enumerate([("none", "Within-network ΔEM, no GSR", ORANGE, 5), ("gsr", "Within-network ΔEM, GSR", TEAL, 63)]):
        head(fig, 5, y0, "ab"[r], ttl)
        vals = nd[gsr]; ymin, ymax = vals.values.min() - 0.05, vals.values.max() + 0.45 * np.ptp(vals.values)
        for k, name in enumerate(NAMES):
            q = netstats[(netstats.gsr == gsr) & (netstats.lag == 0) & (netstats.network == name)].q.iloc[0]
            tax = fig.add_axes(mm(x0 + k * cw, y0 + 6.5, cw - 1.5, 13)); img_panel(tax, BR / f"net_{gsr}_{name}.png")
            tax.set_title(SHORT[k] + (" *" if q < .05 else ""), fontsize=FS, color=DARK, fontweight="bold", pad=1.5)
            ax = fig.add_axes(mm(x0 + k * cw, y0 + 22, cw - 1.5, 30)); x = vals[name].values
            violin(ax, 0, x, c, width=0.9, jitter=0.12, ms=6); zero(ax)
            ax.set_xlim(-0.6, 0.6); ax.set_xticks([]); ax.set_ylim(ymin, ymax); ax.set_yticks([-0.5, 0.0, 0.5]); ax.spines["bottom"].set_visible(False)
            ax.text(0.5, 0.98, f"d = {cohen_d(x):.2f}", transform=ax.transAxes, ha="center", va="top", fontsize=FS)
            if k == 0: ax.set_ylabel("ΔEM within network")
            else: ax.set_yticklabels([])
    cax = fig.add_axes(mm(157, 40, 2.6, 40)); cb = fig.colorbar(plt.cm.ScalarMappable(cmap=blues, norm=plt.Normalize(-0.7, 0)), cax=cax)
    cb.set_ticks([-0.7, -0.35, 0]); cb.ax.tick_params(labelsize=FS, length=2, pad=1.5); cb.outline.set_linewidth(0.4); cb.set_label("Cohen's d, network", fontsize=FS, labelpad=2)
    print("Figure 4"); save(fig, "figure4")


# ============================================================ Figure 3: specification curve (480), dimension summary, R = m^2
def figure3():  # specification curve
    global FH; FH = 166.0
    fig = plt.figure(figsize=(FW * MM, FH * MM))
    s = summ.sort_values("d").reset_index(drop=True); n = len(s)
    keys = ["lag", "em", "ztime", "parc", "gsr", "blockset"]; piv = deltas.pivot_table(index="sid", columns=keys, values="delta")
    head(fig, 5, 5, "a", f"{n} specifications, sorted by effect size")
    ax = fig.add_axes(mm(20, 10, 146, 48))
    idx = np.random.default_rng(SEED).integers(0, 44, (2000, 44)); lo, hi = [], []
    for _, r in s.iterrows():
        x = piv[tuple(r[k] for k in keys)].values; xb = x[idx]; bs = xb.mean(1) / xb.std(1, ddof=1); l, h = np.percentile(bs, [2.5, 97.5]); lo.append(l); hi.append(h)
    sig = (s.p_perm < .05).values; col = np.where(s.gsr == "gsr", TEAL, ORANGE)
    ax.vlines(range(n), lo, hi, color="#dedede", lw=0.35, zorder=1)
    ax.scatter(np.arange(n)[sig], s.d[sig], s=5, c=col[sig], lw=0, zorder=3)
    ax.scatter(np.arange(n)[~sig], s.d[~sig], s=5, facecolor="white", edgecolor=col[~sig], lw=0.5, zorder=3)
    p0 = s[(s.lag == 0) & (s.em == "concat") & (s.ztime == "cond") & (s.parc == "schaefer400") & (s.gsr == "none") & (s.blockset == "all")].index[0]
    ax.scatter([p0], [s.d[p0]], s=70, facecolor="none", edgecolor="k", lw=1.0, zorder=4)
    ax.annotate("pre-specified", (p0, s.d[p0]), xytext=(p0 - 90, s.d[p0] - 0.3), fontsize=FS, arrowprops=dict(arrowstyle="-", lw=0.6, color="k"))
    zero(ax); ax.set_xlim(-2, n + 1); ax.set_xticks([0, 100, 200, 300, 400, 480]); ax.set_xlabel("Specification rank"); ax.set_ylabel("Cohen's d (infection − neutral)"); ax.set_ylim(min(-1.1, np.floor((min(lo) - 0.05) * 10) / 10), 0.45); ax.set_yticks([-1.0, -0.5, 0.0])
    j = joint.set_index("statistic"); pf = lambda p: f"{p:.4f}" if p > 0 else "< 0.0001"
    ax.text(0.99, 0.04, f"median d = {j.loc['median_d','observed']:.2f}; {int(j.loc['n_sig_negative','observed'])} of {n} significant, all negative\n"
                        f"joint permutation p: median {pf(j.loc['median_d','p'])}, count {pf(j.loc['n_sig_negative','p'])}, sum {pf(j.loc['sum_sig_d','p'])}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FS)
    h = [plt.Line2D([], [], marker="o", ls="", color=ORANGE, ms=4, label="no GSR, p < .05"), plt.Line2D([], [], marker="o", ls="", mfc="white", mec=ORANGE, ms=4, label="no GSR, p ≥ .05"),
         plt.Line2D([], [], marker="o", ls="", color=TEAL, ms=4, label="GSR, p < .05"), plt.Line2D([], [], marker="o", ls="", mfc="white", mec=TEAL, ms=4, label="GSR, p ≥ .05")]
    ax.legend(handles=h, frameon=False, loc="upper left", ncol=2, columnspacing=1.2, handletextpad=0.4)
    head(fig, 5, 71, "b", "Effect by analysis choice")
    ax = fig.add_axes(mm(38, 75, 48, 82)); axb = fig.add_axes(mm(90, 75, 22, 82), sharey=ax)
    rowspec = [("gsr", ["none", "gsr"], ["no GSR", "GSR"]), ("lag", [0, 1, 2, 3], ["shift 0", "shift 1 TR", "shift 2 TR", "shift 3 TR"]),
               ("em", ["concat", "within", "between", "edgewise"], ["synchrony-based", "within-block", "between-block", "edge-wise"]),
               ("parc", [f"schaefer{k}" for k in (100, 200, 400, 600, 800, 1000)], [f"Schaefer {k}" for k in (100, 200, 400, 600, 800, 1000)]),
               ("ztime", ["cond", "run"], ["z within condition", "z within run"]), ("blockset", ["all", "common"], ["all blocks", "19/19 blocks"])]
    yy = 0; ypos = []; ylab = []
    for key, levels, labs in rowspec:
        for lv, lb in zip(levels, labs):
            m = (s[key] == lv).values; dd = s.d[m]; frac = ((s.p_perm < .05) & m).sum() / m.sum()
            c = TEAL if (key == "gsr" and lv == "gsr") else (ORANGE if key == "gsr" else "#444444")
            ax.plot([dd.quantile(.25), dd.quantile(.75)], [yy, yy], color=c, lw=1.0, zorder=2); ax.scatter(dd.median(), yy, s=18, color=c, zorder=3)
            axb.barh(yy, frac, height=0.6, color=c, lw=0, alpha=0.85); axb.text(frac + 0.03, yy, f"{frac:.2f}", va="center", fontsize=FS)
            ypos.append(yy); ylab.append(lb); yy -= 1
        yy -= 0.7
    ax.set_yticks(ypos); ax.set_yticklabels(ylab); ax.set_ylim(yy + 0.4, 0.8); zero(ax, "x"); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.7, 0.1); ax.set_xticks([-0.6, -0.4, -0.2, 0.0]); ax.set_xlabel("Median d (IQR)"); ax.spines["left"].set_visible(False)
    axb.set_xlim(0, 1.25); axb.set_xticks([0, 0.5, 1]); axb.set_xlabel("Fraction p < .05"); axb.tick_params(axis="y", length=0, labelleft=False); axb.spines["left"].set_visible(False)
    head(fig, 118, 71, "c", "Shift × definition")
    ax = fig.add_axes(mm(131, 75, 33, 28))
    ems = ["concat", "within", "between", "edgewise"]; mks = ["o", "s", "^", "D"]; labels = ["synchrony-based", "within-block", "between-block", "edge-wise"]
    for gsr, c in [("none", ORANGE), ("gsr", TEAL)]:
        for em, mk in zip(ems, mks):
            ds = [cohen_d(sel(lag, em, gsr)) for lag in range(4)]
            ax.plot(range(4), ds, marker=mk, ms=3, lw=0.8, color=c, mfc=c if em in ("concat", "edgewise") else "white")
    zero(ax); ax.set_xticks(range(4)); ax.set_xlabel("Window shift (TR)"); ax.set_ylabel("Cohen's d"); ax.set_ylim(-0.85, 0.85); ax.set_yticks([-0.8, -0.4, 0.0])
    from matplotlib.lines import Line2D
    hd = [Line2D([], [], marker=mk, ms=3, lw=0, color=DARK, mfc=DARK if em in ("concat", "edgewise") else "white", label=lb) for em, mk, lb in zip(ems, mks, labels)]
    hd += [Line2D([], [], lw=1.2, color=ORANGE, label="no GSR"), Line2D([], [], lw=1.2, color=TEAL, label="GSR")]
    ax.legend(handles=hd, frameon=False, loc="upper left", ncol=2, handlelength=1.2, handletextpad=0.3, columnspacing=0.6, labelspacing=0.15, borderaxespad=0.0)
    head(fig, 120, 118, "d", "R(t) ≈ m(t)² without GSR")
    ax = fig.add_axes(mm(133, 122, 26, 26))
    ts = np.load(DATA / "example" / "sub-01_S1_schaefer400.npy").astype(float)[:, DUMMY:]
    z = zscore(ts); R = rts(z); m2 = z.mean(0) ** 2
    ax.plot([0, 5.4], [0, 5.4], color=GRID, lw=0.6, zorder=0); ax.scatter(m2, R, s=3, color=DARK, lw=0, alpha=0.7)
    ax.set_xlim(-0.1, 5.4); ax.set_ylim(-0.1, 5.4); ax.set_xticks([0, 2.5, 5]); ax.set_yticks([0, 2.5, 5])
    ax.set_xlabel("m(t)², squared mean z"); ax.set_ylabel("R(t), no GSR")
    ax.text(0.04, 0.96, f"one run, {ts.shape[1]} volumes\nr = {np.corrcoef(R, m2)[0, 1]:.3f}", transform=ax.transAxes, fontsize=FS, va="top")
    print("Figure 3"); save(fig, "figure3")


# ============================================================ Figure S3: sex, HI, six conventions (old Figure 5)
def figureS3():
    global FH; FH = 125.0
    fig = plt.figure(figsize=(FW * MM, FH * MM))
    q = pd.read_csv(DATA / "questionnaires" / "scores.csv").set_index("SubjectID").loc[ids]
    hi = ((q.HI_pre + q.HI_post) / 2).values.astype(float)
    convs = [("none", "concat", 0, "1, pre-specified"), ("gsr", "edgewise", 0, "4, edge-wise, GSR")]
    for k, (gsr, em, lag, lab) in enumerate(convs):
        c = ORANGE if gsr == "none" else TEAL
        head(fig, 5 + k * 46, 5, "ab"[k], f"Convention {lab}", color=c)
        ax = fig.add_axes(mm(14 + k * 46, 11, 34, 38)); x = sel(lag, em, gsr)
        violin(ax, 0, x[sex == "F"], c, width=0.8, jitter=0.12, ms=7); violin(ax, 1, x[sex == "M"], c, width=0.8, jitter=0.12, ms=7)
        r = grid[(grid.parc == "schaefer400") & (grid.ses == 1) & (grid.gsr == gsr) & (grid.em == em) & (grid.lag == lag)].iloc[0]
        zero(ax); ax.set_xticks([0, 1]); ax.set_xticklabels([f"women\nd = {r.d_F:.2f}".replace("-", "\u2212"), f"men\nd = {r.d_M:.2f}".replace("-", "\u2212")]); ax.set_xlim(-0.6, 1.6)
        ax.text(0.5, 0.99, f"sex difference p = {r.p_sex:.3f}", transform=ax.transAxes, ha="center", va="top", fontsize=FS)
        if k == 0: ax.set_ylabel("ΔEM (infection − neutral)")
    for k, (gsr, em, lag, lab) in enumerate(convs):
        c = ORANGE if gsr == "none" else TEAL
        head(fig, 101 + k * 37, 5, "cd"[k], f"Trait HI, convention {lab.split(',')[0]}", color=c)
        ax = fig.add_axes(mm(110 + k * 37, 11, 25, 38)); x = sel(lag, em, gsr)
        ax.scatter(hi, x, s=9, color=c, lw=0, alpha=0.85)
        Xd = np.column_stack([np.ones(44), hi]); beta = np.linalg.lstsq(Xd, x, rcond=None)[0]; xx = np.linspace(hi.min(), hi.max(), 50)
        s2 = ((x - Xd @ beta) ** 2).sum() / 42; cov = s2 * np.linalg.inv(Xd.T @ Xd); XX = np.column_stack([np.ones(50), xx]); se = np.sqrt(np.einsum("ij,jk,ik->i", XX, cov, XX))
        ax.plot(xx, XX @ beta, color="k", lw=0.9); ax.fill_between(xx, XX @ beta - 2.02 * se, XX @ beta + 2.02 * se, color="k", alpha=0.1, lw=0)
        r, p = stats.pearsonr(hi, x); rho, prho = stats.spearmanr(hi, x)
        ax.text(0.98, 0.02, f"r = {r:+.2f}, p = {p:.3f}\nρ = {rho:+.2f}, p = {prho:.3f}", transform=ax.transAxes, va="bottom", ha="right", fontsize=FS)
        ax.set_xlabel("Trait HI"); zero(ax); ax.set_xticks([5, 15, 25]); yl = ax.get_ylim(); ax.set_ylim(yl[0] - 0.4 * (yl[1] - yl[0]), yl[1])
        if k == 0: ax.set_ylabel("ΔEM")
    head(fig, 5, 64, "e", "Across six conventions")
    ax = fig.add_axes(mm(14, 68, 150, 42))
    convs6 = [("none", "concat", 0), ("none", "edgewise", 0), ("gsr", "concat", 0), ("gsr", "edgewise", 0), ("gsr", "concat", 2), ("gsr", "edgewise", 2)]
    s2df = pd.read_csv(T / "session2_reduced_grid.csv"); g4 = grid[grid.parc == "schaefer400"]
    for k, (gsr, em, lag) in enumerate(convs6):
        c = ORANGE if gsr == "none" else TEAL; r = g4[(g4.ses == 1) & (g4.gsr == gsr) & (g4.em == em) & (g4.lag == lag)].iloc[0]
        x = sel(lag, em, gsr); xF, xM = x[sex == "F"], x[sex == "M"]
        for dx, v, mk in [(-0.27, xF, "o"), (-0.09, xM, "s")]:
            d = cohen_d(v); lo, hi_ = d_ci(v); dbar(ax, k + dx, d, lo, hi_, c, mk, mfc=c if mk == "o" else "white")
        zf = np.arctanh(r.r_HI); se = 1 / np.sqrt(41)
        dbar(ax, k + 0.09, r.r_HI, np.tanh(zf - 1.96 * se), np.tanh(zf + 1.96 * se), c, "D", mfc="white")
        x2 = s2df[(s2df.parc == "schaefer400") & (s2df.gsr == gsr) & (s2df.em == em) & (s2df.lag == lag)].set_index("sid").loc[ids, "delta"].values
        d = cohen_d(x2); lo, hi_ = d_ci(x2); dbar(ax, k + 0.27, d, lo, hi_, c, "^", mfc="white")
    zero(ax); ax.set_xticks(range(6))
    ax.set_xticklabels(["1 synchrony\nno GSR", "2 edge-wise\nno GSR", "3 synchrony\nGSR", "4 edge-wise\nGSR", "5 synchrony\nGSR, +2 TR", "6 edge-wise\nGSR, +2 TR"])
    ax.set_ylim(-1.4, 1.0); ax.set_ylabel("Cohen's d or r")
    h = [plt.Line2D([], [], marker="o", ls="", color="k", ms=3.5, label="d, women (S1)"), plt.Line2D([], [], marker="s", ls="", mfc="white", mec="k", ms=3.5, label="d, men (S1)"),
         plt.Line2D([], [], marker="D", ls="", mfc="white", mec="k", ms=3.5, label="r (trait HI, ΔEM)"), plt.Line2D([], [], marker="^", ls="", mfc="white", mec="k", ms=3.5, label="d, session 2")]
    ax.legend(handles=h, frameon=False, loc="lower left", ncol=4, columnspacing=1.0, handletextpad=0.3)
    print("Figure S3"); save(fig, "figure_S3")


# ============================================================ Figure S4: individual-difference validity (retest, in-scanner ratings)
def figureS4():
    global FH; FH = 62.0
    fig = plt.figure(figsize=(FW * MM, FH * MM))
    d2 = pd.read_csv(DATA / "edge_metastability" / "spec_curve_subject_deltas_S2.csv")
    def sel2(lag, em, gsr):
        q = d2[(d2.lag == lag) & (d2.em == em) & (d2.gsr == gsr) & (d2.parc == "schaefer400") & (d2.ztime == "cond") & (d2.blockset == "all")]
        return q.set_index("sid").loc[ids, "delta"].values
    for k, (gsr, em, lab, c) in enumerate([("none", "concat", "1, pre-specified", ORANGE), ("gsr", "edgewise", "4, edge-wise, GSR", TEAL)]):
        head(fig, 5 + k * 47, 5, "ab"[k], f"Retest, convention {lab.split(',')[0]}", color=c)
        ax = fig.add_axes(mm(14 + k * 47, 11, 30, 40)); x1, x2 = sel(0, em, gsr), sel2(0, em, gsr)
        ax.scatter(x1, x2, s=9, color=c, lw=0, alpha=0.85); zero(ax); zero(ax, "x")
        r, p = stats.pearsonr(x1, x2); ax.text(0.98, 0.98, f"r = {r:+.2f}\np = {p:.3f}", transform=ax.transAxes, ha="right", va="top", fontsize=FS)
        ax.set_xlabel("ΔEM, session 1"); ax.set_ylabel("ΔEM, session 2"); ax.set_title(["pre-specified", "edge-wise, GSR"][k], fontsize=FS, color=c, pad=2)
        lim = max(np.abs(np.r_[x1, x2])) * 1.1; ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        tk = [-0.5, 0, 0.5] if lim > 0.5 else [-0.2, 0, 0.2]; ax.set_xticks(tk); ax.set_yticks(tk)
    # c: block-level EM vs within-condition rating, session 1
    head(fig, 101, 5, "c", "Block EM vs in-scanner rating, session 1")
    ax = fig.add_axes(mm(114, 11, 50, 40))
    bl = pd.read_csv(DATA / "edge_metastability" / "block_level_EM.csv"); rt = pd.read_csv(DATA / "ratings" / "block_ratings.csv")
    m = bl.merge(rt, on=["sid", "session", "order"]); m = m[m.session == 1].dropna(subset=["mean3"]).copy()
    m["cond"] = (m.trial_type == "infection").astype(int); m["ord_c"] = (m.order - 20) / 10
    m["rat_wc"] = m.mean3 - m.groupby(["sid", "cond"]).mean3.transform("mean"); m["em_ws"] = m.em - m.groupby(["sid", "cond"]).em.transform("mean")
    fit = smf.mixedlm("em ~ cond + ord_c + rat_wc", m, groups=m.sid, re_formula="~cond").fit(reml=False)
    for cond, c, lab in [(0, NEU, "neutral"), (1, ORANGE, "infection")]:
        q = m[m.cond == cond]; ax.scatter(q.rat_wc + np.random.default_rng(0).uniform(-0.04, 0.04, len(q)), q.em_ws, s=3, color=c, lw=0, alpha=0.4, label=lab)
    bins = np.arange(-1.5, 1.51, 0.5); mid = (bins[:-1] + bins[1:]) / 2
    for cond, c in [(0, NEU), (1, ORANGE)]:
        q = m[m.cond == cond]; cut = pd.cut(q.rat_wc, bins); g = q.groupby(cut, observed=True).em_ws.agg(["mean", "sem", "size"])
        ok = g["size"] >= 5; xs = [iv.mid for iv in g.index[ok]]
        ax.errorbar(xs, g["mean"][ok], yerr=g["sem"][ok], fmt="o-", color=c, ms=3, lw=0.8, capsize=1.5, zorder=4)
    zero(ax); ax.set_xlabel("Rating, centred within participant × condition"); ax.set_ylabel("Block EM, centred")
    yl = ax.get_ylim(); ax.set_ylim(yl[0], yl[1] + 0.18 * (yl[1] - yl[0])); ax.set_yticks([-0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    ax.text(0.02, 0.98, f"mixed-model rating slope b = {fit.params['rat_wc']:+.3f}, p = {fit.pvalues['rat_wc']:.2f}", transform=ax.transAxes, va="top", fontsize=FS)
    ax.legend(frameon=False, loc="lower right", handletextpad=0.3, markerscale=2.5)
    print("Figure S4"); save(fig, "figure_S4")


def figureS1():
    global FH; FH = 125.0
    # S1: indicator matrix (full specification curve)
    s = summ.sort_values("d").reset_index(drop=True); n=len(s); sig=(s.p_perm<.05).values; col=np.where(s.gsr=="gsr",TEAL,ORANGE)
    fig=plt.figure(figsize=(174*MM,125*MM)); gs_=fig.add_gridspec(2,1,left=0.14,right=0.98,top=0.95,bottom=0.06,hspace=0.08,height_ratios=[1,1.1])
    ax=fig.add_subplot(gs_[0]); ax.scatter(np.arange(n)[sig],s.d[sig],s=9,c=col[sig],lw=0); ax.scatter(np.arange(n)[~sig],s.d[~sig],s=9,facecolor="white",edgecolor=col[~sig],lw=0.6)
    ax.axhline(0,color="k",lw=0.6,ls=":"); ax.set_xticks([]); ax.set_ylabel("Cohen's d"); ax.set_xlim(-1,n); ax.set_yticks([-0.8,-0.4,0]); panel_label(ax,"a",dx=-0.1,dy=1.0)
    ax2=fig.add_subplot(gs_[1],sharex=ax); panel_label(ax2,"b",dx=-0.1,dy=1.0)
    rowspec=[("gsr",["none","gsr"],["no GSR","GSR"]),("lag",[0,1,2,3],["shift 0","shift 1 TR","shift 2 TR","shift 3 TR"]),("em",["concat","within","between","edgewise"],["synchrony-based","within-block","between-block","edge-wise"]),("parc",[f"schaefer{k}" for k in (100,200,400,600,800,1000)],[f"Schaefer {k}" for k in (100,200,400,600,800,1000)]),("ztime",["cond","run"],["z within condition","z within run"]),("blockset",["all","common"],["all blocks","19/19 blocks"])]
    yy=0; yp=[]; yl=[]
    for key,levels,labs in rowspec:
        for lv,lb in zip(levels,labs):
            m=(s[key]==lv).values; ax2.scatter(np.arange(n)[m&sig],np.full((m&sig).sum(),yy),s=5,c=col[m&sig],lw=0); ax2.scatter(np.arange(n)[m&~sig],np.full((m&~sig).sum(),yy),s=5,facecolor="white",edgecolor=col[m&~sig],lw=0.4)
            yp.append(yy); yl.append(lb); yy-=1
        yy-=0.7
    ax2.set_yticks(yp); ax2.set_yticklabels(yl,fontsize=FS); ax2.set_xticks([]); ax2.set_ylim(yy+0.3,0.8); ax2.spines["bottom"].set_visible(False); ax2.spines["left"].set_visible(False); ax2.tick_params(axis="y",length=0)
    ax2.set_xlabel("Specifications, sorted by effect size")
    print("Figure S1"); audit(fig); fig.savefig(OUT / "figure_S1.pdf"); fig.savefig(OUT / "figure_S1.png"); plt.close(fig)


def figureS2():
    global FH; FH = 80.0
    # S2: between-subject correlation of delta EM across conventions
    fig=plt.figure(figsize=(100*MM,100*MM)); ax=fig.add_axes([0.33,0.33,0.64,0.64])
    names=[("concat","none"),("within","none"),("between","none"),("edgewise","none"),("concat","gsr"),("edgewise","gsr")]
    M=np.column_stack([sel(0,em,g) for em,g in names]); C=np.corrcoef(M.T); ax.imshow(C,cmap="RdBu_r",vmin=-1,vmax=1)
    short=["synchrony-based","within-block","between-block","edge-wise","synchrony-based, GSR","edge-wise, GSR"]; ax.set_xticks(range(6)); ax.set_xticklabels(short,rotation=90,fontsize=FS); ax.set_yticks(range(6)); ax.set_yticklabels(short,fontsize=FS)
    for i in range(6):
        for j in range(6): ax.text(j,i,f"{C[i,j]:.2f}",ha="center",va="center",fontsize=FS,color="w" if abs(C[i,j])>0.6 else "k")
    ax.spines["top"].set_visible(True); ax.spines["right"].set_visible(True)
    print("Figure S2"); audit(fig); fig.savefig(OUT / "figure_S2.pdf"); fig.savefig(OUT / "figure_S2.png"); plt.close(fig)

if __name__ == "__main__":
    for f in sys.argv[1:] or ["1", "2", "3", "4", "S1", "S2", "S3", "S4"]:
        globals()[f"figure{f}"]()
