"""Individual-difference validity checks on the corrected data: session-to-session correlation of dEM (and of the EM
levels) under eight conventions, and block-level mixed models of EM on the in-scanner ratings.
Writes results/tables/retest_reliability.csv and rating_lmm.csv.
usage: python code/analysis/validity_checks.py"""
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]; EM = ROOT / "data" / "edge_metastability"; T = ROOT / "results" / "tables"
s1 = pd.read_csv(EM / "spec_curve_subject_deltas_S1.csv"); s2 = pd.read_csv(EM / "spec_curve_subject_deltas_S2.csv")
rows = []
for gsr in ("none", "gsr"):
    for em in ("concat", "edgewise"):
        for lag in (0, 2):
            f = lambda d: d[(d.parc == "schaefer400") & (d.ztime == "cond") & (d.blockset == "all") & (d.gsr == gsr) & (d.em == em) & (d.lag == lag)].set_index("sid")
            a, b = f(s1), f(s2).loc[f(s1).index]
            r, p = stats.pearsonr(a.delta, b.delta); rho = stats.spearmanr(a.delta, b.delta)[0]
            rows.append(dict(GSR="no" if gsr == "none" else "yes", definition={"concat": "concatenated", "edgewise": "edge-wise"}[em], shift=lag, n=len(a),
                             r_delta=r, p_delta=p, rho_delta=rho, r_EM_infection=stats.pearsonr(a.em_inf, b.em_inf)[0], r_EM_neutral=stats.pearsonr(a.em_neu, b.em_neu)[0]))
pd.DataFrame(rows).to_csv(T / "retest_reliability.csv", index=False)

bl = pd.read_csv(EM / "block_level_EM.csv"); rt = pd.read_csv(ROOT / "data" / "ratings" / "block_ratings.csv")
m = bl.merge(rt, on=["sid", "session", "order"]); m["cond"] = (m.trial_type == "infection").astype(int); m["ord_c"] = (m.order - 20) / 10
rows = []
for ses in (1, 2):
    for v, name in [("disgust", "disgust"), ("fear", "fear"), ("infect", "infection threat"), ("mean3", "mean of three")]:
        x = m[m.session == ses].dropna(subset=[v]).copy()
        x["wc"] = x[v] - x.groupby(["sid", "cond"])[v].transform("mean")
        fit = smf.mixedlm("em ~ cond + ord_c + wc", x, groups=x.sid, re_formula="~cond").fit(reml=False)
        g = pd.read_csv(EM / f"global_S{ses}.csv").set_index("SubjectID")
        rd = x.groupby(["sid", "cond"])[v].mean().unstack(); rd = (rd[1] - rd[0]).reindex(g.index); ok = rd.notna()
        r, p = stats.pearsonr(rd[ok], g.Delta[ok])
        rows.append(dict(session=ses, rating=name, n_blocks=len(x), b_rating=fit.params["wc"], z_rating=fit.tvalues["wc"], p_rating=fit.pvalues["wc"],
                         b_condition=fit.params["cond"], p_condition=fit.pvalues["cond"], r_subject=r, p_subject=p, n_subjects=int(ok.sum())))
out = pd.DataFrame(rows); out.to_csv(T / "rating_lmm.csv", index=False)
pd.set_option("display.width", 200); print(pd.read_csv(T / "retest_reliability.csv").round(3).to_string()); print(out.round(4).to_string())
