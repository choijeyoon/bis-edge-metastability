"""Secondary and exploratory analyses (Results sections on sex, session, decomposition, motion, traits, and the
GSR x parcellation x shift grid). Inputs are the per-participant tables in data/ (written by recompute_schaefer400.py
and spec_curve.py from the imaging derivatives); outputs go to results/tables/.

Outputs (results/tables/):
  primary_H1.csv, secondary_interactions.csv, decomposition_by_sex.csv, decomposition_component_correlations.csv,
  block_lmm_fd.csv, trait_family.csv, hi_robustness.csv, change_reliability.csv, power_sensitivity.csv,
  gsr_grid_sex_HI.csv, session2_reduced_grid.csv, individual_difference_correlations_S1_lag0.csv, gsr_fd_checks.csv
usage: python code/analysis/secondary_analyses.py
"""
from pathlib import Path
import numpy as np, pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy import stats
import warnings; warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "data"; EM = DATA / "edge_metastability"
OUT = ROOT / "results" / "tables"; OUT.mkdir(parents=True, exist_ok=True)
# Every resampling procedure starts from a fresh generator with this seed, so a given test returns the
# same p in every table and matches spec_curve.py.
SEED = 20260911

g1 = pd.read_csv(EM / "global_S1.csv"); ids = g1.SubjectID.tolist(); n = len(ids)
demo = pd.read_csv(DATA / "participants.csv")[["SubjectID", "Sex", "Age"]]
g2 = pd.read_csv(EM / "global_S2.csv").set_index("SubjectID").loc[ids].reset_index()
sex = demo.set_index("SubjectID").loc[ids, "Sex"].values
fem = (sex == "F").astype(int)
y = g1.Delta.values


def cohen_d(x): return x.mean() / x.std(ddof=1)


def signflip_p(x, B=10000):
    d = cohen_d(x)
    flips = np.random.default_rng(SEED).choice([-1.0, 1.0], size=(B, len(x)))
    X = flips * x[None, :]
    nd = X.mean(1) / X.std(1, ddof=1)
    return d, (np.abs(nd) >= abs(d)).mean()


# ---------------- primary ----------------
d1, p1 = signflip_p(y)
t1, pt1 = stats.ttest_rel(g1.EM_Inf, g1.EM_Neu)
pd.DataFrame([dict(test="H1 S1 global EM, concat lag0, sign-flip permutation two-tailed", n=n, mean_delta=y.mean(),
                   d=d1, p_perm=p1, t=t1, p_t=pt1, n_down=int((y < 0).sum()))]).to_csv(OUT / "primary_H1.csv", index=False)

# ---------------- secondary: interactions ----------------
def long_df(g, ses):
    a = g[["SubjectID"]].copy(); a["cond"] = 1; a["EM"] = g.EM_Inf.values; a["ses"] = ses
    b = g[["SubjectID"]].copy(); b["cond"] = 0; b["EM"] = g.EM_Neu.values; b["ses"] = ses
    return pd.concat([a, b])
L = pd.concat([long_df(g1, 1), long_df(g2, 2)]).merge(demo, on="SubjectID")
L["fem"] = (L.Sex == "F").astype(int); L["S2"] = (L.ses == 2).astype(int)
L1 = L[L.ses == 1]
m_sex = smf.mixedlm("EM ~ cond * fem", L1, groups=L1.SubjectID).fit(reml=False)
m_sex0 = smf.mixedlm("EM ~ cond + fem", L1, groups=L1.SubjectID).fit(reml=False)
m_ses = smf.mixedlm("EM ~ cond * S2", L, groups=L.SubjectID).fit(reml=False)
m_ses0 = smf.mixedlm("EM ~ cond + S2", L, groups=L.SubjectID).fit(reml=False)
rows = []
for name, m, m0, term in [("cond x sex (S1)", m_sex, m_sex0, "cond:fem"), ("cond x session", m_ses, m_ses0, "cond:S2")]:
    bf01 = np.exp((m.bic - m0.bic) / 2)
    rows.append(dict(test=name, beta=m.params[term], z=m.tvalues[term], p=m.pvalues[term], BF01_bic=bf01))
sec = pd.DataFrame(rows)
order = np.argsort(sec.p.values); adj = np.empty(2)
adj[order[0]] = min(1, 2 * sec.p.values[order[0]]); adj[order[1]] = min(1, max(adj[order[0]], sec.p.values[order[1]]))
sec["p_holm"] = adj
for sv in "FM":
    x = y[sex == sv]; d, p = signflip_p(x)
    sec = pd.concat([sec, pd.DataFrame([dict(test=f"H1 S1 {sv} only (descriptive)", beta=x.mean(), z=np.nan, p=p, BF01_bic=np.nan, p_holm=np.nan, d=d, n=len(x))])])
d2, p2 = signflip_p(g2.Delta.values)
sec = pd.concat([sec, pd.DataFrame([dict(test="H1 S2 (descriptive)", beta=g2.Delta.mean(), z=np.nan, p=p2, BF01_bic=np.nan, p_holm=np.nan, d=d2, n=n)])])
sec.to_csv(OUT / "secondary_interactions.csv", index=False)

# ---------------- decomposition by sex ----------------
sc = pd.read_csv(EM / "spec_curve_subject_deltas_S1.csv")
base = sc[(sc.ztime == "cond") & (sc.parc == "schaefer400") & (sc.gsr == "none") & (sc.blockset == "all")]
assert np.allclose(base[(base.lag == 0) & (base.em == "concat")].set_index("sid").loc[ids, "delta"].values, y, atol=1e-6), "spec curve and recompute disagree"
rows = []
for lag in (0, 2):
    for em in ["concat", "within", "between", "edgewise"]:
        d_ = base[(base.lag == lag) & (base.em == em)].set_index("sid").loc[ids, "delta"].values
        dF, dM = d_[sex == "F"], d_[sex == "M"]
        ti, pi = stats.ttest_ind(dF, dM)
        rows.append(dict(lag=lag, em=em, d_all=cohen_d(d_), p_perm_all=signflip_p(d_)[1], d_F=cohen_d(dF), d_M=cohen_d(dM), t_sexdiff=ti, p_sexdiff=pi))
w = base[(base.lag == 0) & (base.em == "within")].set_index("sid").loc[ids, "delta"].values
b = base[(base.lag == 0) & (base.em == "between")].set_index("sid").loc[ids, "delta"].values
c = base[(base.lag == 0) & (base.em == "concat")].set_index("sid").loc[ids, "delta"].values
pd.DataFrame(rows).to_csv(OUT / "decomposition_by_sex.csv", index=False)
pd.DataFrame([dict(pair="within vs between", r=np.corrcoef(w, b)[0, 1]), dict(pair="within vs concat", r=np.corrcoef(w, c)[0, 1]),
              dict(pair="between vs concat", r=np.corrcoef(b, c)[0, 1])]).to_csv(OUT / "decomposition_component_correlations.csv", index=False)

# ---------------- block-level LMM with FD ----------------
bl = pd.read_csv(EM / "block_level_EM.csv")
fd = pd.read_csv(DATA / "motion" / "fd_per_block.csv")
fd["sid"] = fd.subject.astype(int); fd["session"] = fd.session.str[1].astype(int)
bl["onset_r"] = bl.onset.round(3); fd["onset_r"] = fd.block_onset.round(3)
bl = bl.merge(fd[["sid", "session", "onset_r", "block_mean_fd"]], on=["sid", "session", "onset_r"], how="left")
assert bl.block_mean_fd.notna().all(), "FD merge failed"
bl = bl.merge(demo, left_on="sid", right_on="SubjectID")
bl["cond"] = (bl.trial_type == "infection").astype(int); bl["fem"] = (bl.Sex == "F").astype(int)
bl["ord_c"] = (bl.order - 20) / 10; bl["fd_c"] = bl.block_mean_fd - bl.block_mean_fd.mean()
b1 = bl[bl.session == 1]
rows = []
for name, f in [("cond + order", "em ~ cond + ord_c"), ("cond + order + FD", "em ~ cond + ord_c + fd_c"),
                ("cond*sex + order + FD", "em ~ cond*fem + ord_c + fd_c")]:
    m = smf.mixedlm(f, b1, groups=b1.sid, re_formula="~cond").fit(reml=False)
    for term in [k for k in m.params.index if k not in ("Intercept", "Group Var", "Group x cond Cov", "cond Var")]:
        rows.append(dict(model=name, term=term, beta=m.params[term], z=m.tvalues[term], p=m.pvalues[term]))
fdc = b1.groupby(["sid", "cond"]).block_mean_fd.mean().unstack().loc[ids]
dfd = (fdc[1] - fdc[0]).values; tfd, pfd = stats.ttest_rel(fdc[1], fdc[0])
rows.append(dict(model="FD check", term="mean FD infection - neutral (paired t)", beta=dfd.mean(), z=tfd, p=pfd))
r, p = stats.pearsonr(dfd, y); rows.append(dict(model="FD check", term="r(delta FD, delta EM)", beta=r, z=np.nan, p=p))
mfd = b1.groupby("sid").block_mean_fd.mean().loc[ids].values
r, p = stats.pearsonr(mfd, y); rows.append(dict(model="FD check", term="r(mean FD S1, delta EM)", beta=r, z=np.nan, p=p))
keep = mfd <= np.percentile(mfd, 75); d_k, p_k = signflip_p(y[keep])
rows.append(dict(model="FD check", term=f"H1 excluding top-quartile FD (n={keep.sum()})", beta=d_k, z=np.nan, p=p_k))
pd.DataFrame(rows).to_csv(OUT / "block_lmm_fd.csv", index=False)

# ---------------- trait family ----------------
pre = pd.read_csv(DATA / "questionnaires" / "items_pre.csv").set_index("SubjectID").loc[ids]
post = pd.read_csv(DATA / "questionnaires" / "items_post.csv").set_index("SubjectID").loc[ids]
scales = {"IND": [f"IND_{i}" for i in range(1, 9)], "HI": [f"IND_{i}" for i in range(1, 5)], "VI": [f"IND_{i}" for i in range(5, 9)],
          "COL": [f"COL_{i}" for i in range(1, 9)], "HC": [f"COL_{i}" for i in range(1, 5)], "VC": [f"COL_{i}" for i in range(5, 9)]}
def alpha(X):
    X = np.asarray(X, float); k = X.shape[1]; return k / (k - 1) * (1 - X.var(0, ddof=1).sum() / X.sum(1).var(ddof=1))
rel = []; X = {}
for s, cols in scales.items():
    sp, so = pre[cols].sum(1).values.astype(float), post[cols].sum(1).values.astype(float)
    X[f"{s}_pre"], X[f"{s}_post"] = sp, so
    a1, a2, r = alpha(pre[cols]), alpha(post[cols]), np.corrcoef(sp, so)[0, 1]
    rel.append(dict(scale=s, k_items=len(cols), alpha_pre=a1, alpha_post=a2, r_prepost=r, rel_change=(a1 + a2 - 2 * r) / (2 - 2 * r),
                    sd_change=(so - sp).std(ddof=1), r_change_deltaEM=stats.pearsonr(so - sp, y)[0], p_change_deltaEM=stats.pearsonr(so - sp, y)[1]))
pd.DataFrame(rel).to_csv(OUT / "change_reliability.csv", index=False)
names = list(X); Xm = np.column_stack([X[k] for k in names])
Xz = (Xm - Xm.mean(0)) / Xm.std(0, ddof=1)
def corr_all(v):
    vz = (v - v.mean()) / v.std(ddof=1); return Xz.T @ vz / (n - 1)
robs = corr_all(y)
B = 20000; mx = np.empty(B); rng = np.random.default_rng(SEED)
for bb in range(B):
    mx[bb] = np.abs(corr_all(rng.permutation(y))).max()
fam = pd.DataFrame([dict(predictor=k, r=r, p_raw=stats.pearsonr(X[k], y)[1], p_fwer_maxstat=(mx >= abs(r)).mean(),
                         rho=stats.spearmanr(X[k], y)[0], p_spearman=stats.spearmanr(X[k], y)[1]) for k, r in zip(names, robs)])
fam.to_csv(OUT / "trait_family.csv", index=False)

# ---------------- HI robustness ----------------
hi = (X["HI_pre"] + X["HI_post"]) / 2
rows = []
def add(label, x, yy, **kw):
    r, p = stats.pearsonr(x, yy); rows.append(dict(analysis=label, r=r, p=p, **kw))
add("HI mean(pre,post) vs Delta EM (concat lag0)", hi, y)
add("HI_pre vs Delta EM", X["HI_pre"], y); add("HI_post vs Delta EM", X["HI_post"], y)
rho, pr = stats.spearmanr(hi, y); rows.append(dict(analysis="HI mean: Spearman", r=rho, p=pr))
rng = np.random.default_rng(SEED)
bs = [stats.pearsonr(hi[i], y[i])[0] for i in (rng.integers(0, n, n) for _ in range(10000))]
lo, hiC = np.percentile(bs, [2.5, 97.5]); rows.append(dict(analysis="HI mean: bootstrap 95% CI", r=np.nan, p=np.nan, ci_low=lo, ci_high=hiC))
loo = [stats.pearsonr(np.delete(hi, i), np.delete(y, i)) for i in range(n)]
rows.append(dict(analysis="HI mean: leave-one-out", r=min(r for r, p in loo), p=max(p for r, p in loo), r_max=max(r for r, p in loo)))
def partial(x, yy, z):
    rx = x - np.polyval(np.polyfit(z, x, 1), z); ry = yy - np.polyval(np.polyfit(z, yy, 1), z); return stats.pearsonr(rx, ry)
r, p = partial(hi, y, mfd); rows.append(dict(analysis="HI mean: partial, controlling mean FD", r=r, p=p))
r, p = partial(hi, y, fem.astype(float)); rows.append(dict(analysis="HI mean: partial, controlling sex", r=r, p=p))
for sv in "FM":
    m = sex == sv; add(f"HI mean, {sv} only", hi[m], y[m], n=int(m.sum()))
for lag in (0, 2):
    for em in ["concat", "within", "between", "edgewise"]:
        d_ = base[(base.lag == lag) & (base.em == em)].set_index("sid").loc[ids, "delta"].values
        add(f"HI mean vs Delta EM: {em}, lag {lag}", hi, d_)
add("HI mean vs Delta EM session 2", hi, g2.Delta.values)
pd.DataFrame(rows).to_csv(OUT / "hi_robustness.csv", index=False)

# ---------------- power sensitivity ----------------
from statsmodels.stats.power import TTestPower
tp = TTestPower()
pd.DataFrame([dict(quantity="minimum detectable d, n=44, alpha=.05 two-tailed, power=.80", value=tp.solve_power(nobs=44, alpha=.05, power=.8, alternative="two-sided")),
              dict(quantity="n required for d=0.30, alpha=.05 two-tailed, power=.80", value=tp.solve_power(effect_size=.30, alpha=.05, power=.8, alternative="two-sided")),
              dict(quantity="achieved power for d=0.30 at n=44", value=tp.power(effect_size=.30, nobs=44, alpha=.05, alternative="two-sided"))]).to_csv(OUT / "power_sensitivity.csv", index=False)

# ---------------- GSR x parcellation x lag grid, both sessions, with sex and HI ----------------
sc2 = pd.read_csv(EM / "spec_curve_subject_deltas_S2.csv")
def grid(df, ses):
    q = df[(df.ztime == "cond") & (df.blockset == "all") & df.em.isin(["concat", "edgewise"]) & df.lag.isin([0, 2]) & df.parc.isin(["schaefer200", "schaefer400"])]
    return q[["sid", "parc", "gsr", "lag", "em", "delta"]].assign(ses=ses)
s1, s2 = grid(sc, 1), grid(sc2, 2); s2.to_csv(OUT / "session2_reduced_grid.csv", index=False)
both = pd.concat([s1, s2]); out = []
for (ses, parc, gsr, lag, em), g in both.groupby(["ses", "parc", "gsr", "lag", "em"]):
    d = g.set_index("sid").loc[ids, "delta"].values
    dF, dM = d[sex == "F"], d[sex == "M"]; ti, pi = stats.ttest_ind(dF, dM); t, p = stats.ttest_1samp(d, 0)
    r, pr = stats.pearsonr(hi, d); rho, prho = stats.spearmanr(hi, d)
    out.append(dict(ses=ses, parc=parc, gsr=gsr, lag=lag, em=em, d=cohen_d(d), p=p, p_perm=signflip_p(d)[1], n_down=int((d < 0).sum()), d_F=cohen_d(dF), d_M=cohen_d(dM), p_sex=pi, r_HI=r, p_HI=pr, rho_HI=rho, p_rhoHI=prho))
pd.DataFrame(out).to_csv(OUT / "gsr_grid_sex_HI.csv", index=False)
piv = s1[s1.lag == 0].pivot_table(index="sid", columns=["parc", "gsr", "em"], values="delta").loc[ids]
piv.corr().to_csv(OUT / "individual_difference_correlations_S1_lag0.csv")

# ---------------- GSR results against head motion ----------------
rows = []
for (parc, em, lag), g in s1[s1.gsr == "gsr"].groupby(["parc", "em", "lag"]):
    d = g.set_index("sid").loc[ids, "delta"].values; dd, pp = signflip_p(d)
    r1, p1 = stats.pearsonr(dfd, d); r2, p2 = stats.pearsonr(mfd, d); dk, pk = signflip_p(d[keep])
    ols = sm.OLS(d, sm.add_constant(dfd)).fit()
    rows.append(dict(parc=parc, em=em, lag=lag, d=dd, p=pp, r_dFD=r1, p_dFD=p1, r_meanFD=r2, p_meanFD=p2, d_lowFD=dk, p_lowFD=pk, intercept_p_adj_dFD=ols.pvalues[0]))
pd.DataFrame(rows).to_csv(OUT / "gsr_fd_checks.csv", index=False)

print("done")
pd.set_option("display.width", 250)
for f in ["primary_H1", "secondary_interactions", "decomposition_by_sex", "decomposition_component_correlations", "block_lmm_fd", "trait_family", "hi_robustness", "change_reliability", "gsr_grid_sex_HI", "gsr_fd_checks"]:
    print(f"\n=== {f}"); print(pd.read_csv(OUT / f"{f}.csv").round(3).to_string())
