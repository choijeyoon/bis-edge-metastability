"""Surface renders for Figures 2 and 4 (run headless, e.g. under xvfb-run). Requires brainspace and surfplot.
Parcel maps: sequential blue for reductions, outlines for q<.05 parcels, lateral+medial both hemispheres.
Network thumbnails: left hemisphere, lateral above medial, network parcels outlined and filled by within-network d.
usage: python code/figures/render_brains.py [maps] [networks]   (default: both)"""
import sys
import numpy as np, pandas as pd
from pathlib import Path
from brainspace.datasets import load_conte69, load_parcellation
from surfplot import Plot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

ROOT = Path(__file__).resolve().parents[2]; T = ROOT / "results" / "tables"
OUT = ROOT / "figures" / "brains"; OUT.mkdir(parents=True, exist_ok=True)
lh, rh = load_conte69(); lab = load_parcellation("schaefer", scale=400, join=True)
n_lh = lh.n_points
node = pd.read_csv(T / "node_stats_edgewise_S1.csv"); net = pd.read_csv(T / "network_stats_sync_S1.csv")
yeo = np.array([{"Vis": 1, "SomMot": 2, "DorsAttn": 3, "SalVentAttn": 4, "Limbic": 5, "Cont": 6, "Default": 7}[s.split("_")[2]] for s in pd.read_csv(ROOT / "data" / "atlases" / "schaefer400_7net_labels.tsv", sep="\t").name])
NAMES = ["Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Limbic", "Control", "DefaultMode"]
# sequential ramps for d from 0 (near white) to -0.7; positive values clipped to the light end.
# Hue follows the convention (purple without GSR, blue with GSR); the purple stops are the blue stops
# with OKLCH lightness and chroma held fixed and only the hue changed, so equal d gives equal lightness.
RAMP = {"gsr": ["#08306b", "#2171b5", "#6baed6", "#c6dbef", "#f2f2f2"],
        "none": ["#3b2264", "#765baf", "#ab99d7", "#dad4ee", "#f2f2f2"]}
CMAP = {g: LinearSegmentedColormap.from_list(f"d_{g}", c) for g, c in RAMP.items()}
VMIN, VMAX = -0.7, 0.0

WHAT = set(sys.argv[1:]) or {"maps", "networks"}
for gsr in ("none", "gsr"):
  if "maps" in WHAT:
    s = node[(node.gsr == gsr) & (node.lag == 0)].sort_values("parcel")
    vals = np.full(401, np.nan); vals[1:] = np.clip(s.d.values, VMIN, VMAX)
    sig = np.zeros(401); sig[1:] = (s.q.values < .05).astype(float)
    data = vals[lab]; outline = sig[lab]
    p = Plot(lh, rh, views=["lateral", "medial"], size=(2400, 600), zoom=1.15, brightness=0.8, layout="row")
    p.add_layer(data, cmap=CMAP[gsr], color_range=(VMIN, VMAX), cbar=False)
    p.add_layer(outline, as_outline=True, cmap=ListedColormap(["#1a1a1a"]), cbar=False, color_range=(0.5, 1.5))
    fig = p.build(); fig.savefig(OUT / f"map_{gsr}.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
  if "networks" in WHAT:
    # network thumbnails: lh lateral above medial; the outline keeps the network visible when d is near zero
    for k, name in enumerate(NAMES, 1):
        d = net[(net.gsr == gsr) & (net.lag == 0) & (net.network == name)].d.iloc[0]
        v = np.full(401, np.nan); v[1:][yeo == k] = np.clip(d, VMIN, VMAX)
        mask = np.zeros(401); mask[1:][yeo == k] = 1
        p = Plot(lh, views=["lateral", "medial"], layout="column", size=(600, 900), zoom=1.5, brightness=0.85)
        p.add_layer(v[lab][:n_lh], cmap=CMAP[gsr], color_range=(VMIN, VMAX), cbar=False)
        p.add_layer(mask[lab][:n_lh], as_outline=True, cmap=ListedColormap(["#333333"]), cbar=False, color_range=(0.5, 1.5))
        fig = p.build(); fig.savefig(OUT / f"net_{gsr}_{name}.png", dpi=200, bbox_inches="tight", facecolor="white"); plt.close(fig)
  print(gsr, "done", flush=True)
