"""Left hemisphere, lateral and medial, Yeo-7 colours with all Schaefer-400 parcel boundaries drawn,
and the two example parcels outlined in black. Run under xvfb-run."""
import sys, numpy as np
from brainspace.datasets import load_conte69, load_parcellation
from surfplot import Plot
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

pi, pj = int(sys.argv[1]), int(sys.argv[2])
YEO = ["#7A1E7A", "#4A9BD5", "#1B7B3A", "#C8A2C8", "#D9D67E", "#E69422", "#C64B4B"]
lh, rh = load_conte69(); lab = load_parcellation("schaefer", scale=400, join=True)[:lh.n_points]
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parents[2]; OUT = ROOT / "figures" / "brains"; OUT.mkdir(parents=True, exist_ok=True)
yeo = np.array([{"Vis": 1, "SomMot": 2, "DorsAttn": 3, "SalVentAttn": 4, "Limbic": 5, "Cont": 6, "Default": 7}[s.split("_")[2]] for s in pd.read_csv(ROOT / "data" / "atlases" / "schaefer400_7net_labels.tsv", sep="\t").name])
v = np.full(401, np.nan); v[1:] = yeo
o = np.zeros(401); o[pi + 1] = 1; o[pj + 1] = 1
for view, name in [("lateral", "lat"), ("medial", "med")]:
    p = Plot(lh, views=view, size=(900, 660), zoom=1.55, brightness=0.85)
    p.add_layer(v[lab], cmap=ListedColormap(YEO), color_range=(1, 7), cbar=False)
    p.add_layer(lab.astype(float), as_outline=True, cmap=ListedColormap(["#4a4a4a"]), cbar=False, color_range=(0, 1))  # parcel boundaries
    p.add_layer(o[lab], as_outline=True, cmap=ListedColormap(["#000000"]), color_range=(0.5, 1.5), cbar=False)
    fig = p.build(); fig.savefig(OUT / f"yeo400_lh_{name}.png", dpi=300, bbox_inches="tight", facecolor="white"); plt.close(fig)
print("ok")
