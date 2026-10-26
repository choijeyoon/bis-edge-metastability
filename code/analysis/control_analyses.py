"""Control analyses: task-evoked responses and nuisance signals, button presses, network-level effects by hemisphere,
and the across-run covariance structure of the condition contrast (Supplementary Tables S17-S20); replication of the
control variants in run 2, condition-specific global-signal coefficients and the strictly alternating blocks 7-40
(control_replication.csv, Supplementary Table S21).
Schaefer 400, normalization within condition, all blocks. usage: python3 control_analyses.py <BIS_edge root>"""
import sys, re
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(sys.argv[1]); BIDS = ROOT / "fMRIPrep_BIDS"; PM = BIDS / "parcellation_multires"
REL = ROOT / "bis_edge_metastability/github/bis-edge-metastability"
OUT = ROOT / "bis_edge_metastability/results/tables/netneuro_corrected"; SEED = 20260911
TR, DUMMY, BLK, N = 2.0, 4, 9, 400
# sub-01 run 2: the exported BOLD file is a copy of run 1 (identical NIfTI; the DICOM export holds series 201, EPI_S1); excluded
INVALID = {(1, 2)}   # (participant, run)
MOTION = [f"{a}_{x}" for a in ("trans", "rot") for x in "xyz"]
NUIS = MOTION + [m + "_derivative1" for m in MOTION] + [m + "_power2" for m in MOTION] + [m + "_derivative1_power2" for m in MOTION] + ["white_matter", "csf"]


def zscore(ts):
    mu = ts.mean(1, keepdims=True); sd = ts.std(1, keepdims=True, ddof=1); sd[sd == 0] = 1; return (ts - mu) / sd


def regress(ts, X):
    X = np.column_stack([np.ones(X.shape[0]), X]); beta, *_ = np.linalg.lstsq(X, ts.T, rcond=None); return ts - (X @ beta).T


def regress_gs(ts, gs):
    gs = gs - gs.mean(); beta = (ts * gs).sum(1) / (gs @ gs); return ts - beta[:, None] * gs[None, :]


def rts(z):
    n = z.shape[0]; s = z.sum(0); s2 = (z * z).sum(0); return 0.5 * (s ** 2 - s2) / (n * (n - 1) / 2)


def edgewise_sd(z):
    T = z.shape[1]; z2 = z * z; m1 = z @ z.T / T; m2 = z2 @ z2.T / T
    var = (m2 - m1 ** 2) * T / (T - 1); iu = np.triu_indices(z.shape[0], 1); return float(np.sqrt(np.clip(var[iu], 0, None)).mean())


def windows(ts, ev, lag, first=0):
    T = ts.shape[1]; out = {"infection": [], "neutral": []}
    for i, (_, e) in enumerate(ev.iterrows()):
        s0 = int(np.floor((e.onset - DUMMY * TR) / TR)) + lag
        if i < first or s0 < 0 or s0 + BLK > T: continue
        out[e.trial_type].append(ts[:, s0:s0 + BLK])
    return {k: np.stack(v) for k, v in out.items()}          # blocks x parcels x time


def cond_gs(gs, ev, lag, T):
    """global signal split into three regressors: samples in infection windows, in neutral windows, elsewhere"""
    M = np.zeros((3, T))
    for c, w in windows(np.arange(T)[None, :].astype(float), ev, lag).items():
        for idx in w[:, 0, :].astype(int): M[0 if c == "infection" else 1, idx] = 1
    M[2] = 1 - M[0] - M[1]; return (M * (gs - gs.mean())[None, :]).T


ALT = 6   # strictly alternating segment: blocks 7-40 (0-based index 6 onward)


def signflip(x, B=10000):
    d = x.mean() / x.std(ddof=1); rng = np.random.default_rng(SEED)
    X = rng.choice([-1.0, 1.0], size=(B, len(x))) * x[None, :]; nd = X.mean(1) / X.std(1, ddof=1)
    return float(d), float((np.abs(nd) >= abs(d)).mean())


lab = pd.read_csv(REL / "data/atlases/schaefer400_7net_labels.tsv", sep="\t").name.astype(str)
som_l = np.array([("_SomMot_" in s) and ("LH_" in s) for s in lab]); som_r = np.array([("_SomMot_" in s) and ("RH_" in s) for s in lab])
vis_l = np.array([("_Vis_" in s) and ("LH_" in s) for s in lab]); vis_r = np.array([("_Vis_" in s) and ("RH_" in s) for s in lab])
NETS = {k: np.array([f"_{k}_" in s for s in lab]) for k in ("Vis", "SomMot", "DorsAttn", "SalVentAttn", "Limbic", "Cont", "Default")}
assert sum(v.sum() for v in NETS.values()) == N
assert som_l.sum() + som_r.sum() == 77 and vis_l.sum() + vis_r.sum() == 61, (som_l.sum(), som_r.sum(), vis_l.sum(), vis_r.sum())

subs = sorted(int(m.group(1)) for p in PM.iterdir() if (m := re.match(r"sub-(\d+)$", p.name)) and (p / f"sub-{int(m.group(1)):02d}_S1_schaefer400.npy").exists())
rows, net = [], []
for sid in subs:
    for ses in ("S1", "S2"):
        f = PM / f"sub-{sid:02d}/sub-{sid:02d}_{ses}_schaefer400.npy"
        if not f.exists() or (sid, int(ses[1])) in INVALID: continue
        conf = pd.read_csv(BIDS / f"results/sub-{sid:02d}/ses-{ses}/func/sub-{sid:02d}_ses-{ses}_task-picrate_desc-confounds_timeseries.tsv", sep="\t").iloc[DUMMY:]
        ev = pd.read_csv(BIDS / f"sub-{sid:02d}/func/sub-{sid:02d}_ses-{ses}_task-picrate_events.tsv", sep="\t")
        raw = np.load(f).astype(float)[:, DUMMY:]; T = raw.shape[1]
        t = np.linspace(-1, 1, T); trend = np.column_stack([t, t ** 2])
        nuis = conf[NUIS].fillna(0).to_numpy(); gs = conf["global_signal"].to_numpy()
        for gsr in ("none", "gsr", "gsr_cond"):
            for lag in (0, 2):
                if gsr == "gsr_cond":    # one global-signal coefficient per parcel and condition
                    gc = cond_gs(gs, ev, lag, T); base_ts = regress(raw, gc)
                    cleaned = {"nuisance": regress(raw, np.column_stack([trend, nuis, gc]))}
                    variants = ("base", "nuisance")
                else:
                    base_ts = regress_gs(raw, gs) if gsr == "gsr" else raw
                    g_ = [gs] if gsr == "gsr" else []
                    parts = {"nuisance": [trend, nuis] + g_, "trend": [trend] + g_, "motion24": [nuis[:, :24]] + g_, "wmcsf": [nuis[:, 24:]] + g_}
                    cleaned = {k: regress(raw, np.column_stack(v)) for k, v in parts.items()}
                    variants = ("base", "evoked", "nuisance", "both", "trend", "motion24", "wmcsf")
                for var in variants:
                  for bs, first in (("all", 0), ("alt", ALT)):
                    if bs == "alt" and var not in ("base", "nuisance"): continue
                    W = windows(cleaned["nuisance"] if var == "both" else cleaned.get(var, base_ts), ev, lag, first)
                    for cond, w in W.items():
                        if var in ("evoked", "both"): w = w - w.mean(0, keepdims=True)   # condition-mean stimulus-locked response
                        cat = zscore(np.concatenate(list(w), 1))
                        rows.append(dict(sid=sid, run=ses, gsr=gsr, lag=lag, variant=var, blockset=bs, cond=cond, n_blocks=w.shape[0],
                                         em_sync=rts(cat).std(ddof=1), em_edge=edgewise_sd(cat)))
                        if var in ("base", "nuisance") and lag == 0 and ses == "S1" and bs == "all" and gsr != "gsr_cond":
                            r = dict(sid=sid, gsr=gsr, variant=var, cond=cond)
                            for name, mask in [("som_lh", som_l), ("som_rh", som_r), ("som", som_l | som_r), ("vis_lh", vis_l), ("vis_rh", vis_r)] + list(NETS.items()):
                                r[name] = rts(cat[mask]).std(ddof=1)
                            net.append(r)
    print(f"sub-{sid:02d}", flush=True)

df = pd.DataFrame(rows); df.to_csv(OUT / "control_analyses_subject.csv", index=False)
res = []
main = (df.run == "S1") & (df.blockset == "all") & (df.gsr != "gsr_cond")
for (gsr, lag, var), g in df[main].groupby(["gsr", "lag", "variant"], sort=False):
    g = g.set_index(["sid", "cond"])
    for m in ("em_sync", "em_edge"):
        x = (g[m].xs("infection", level="cond") - g[m].xs("neutral", level="cond")).values; d, p = signflip(x)
        res.append(dict(gsr=gsr, lag=lag, variant=var, measure=m, n=len(x), mean_inf=g[m].xs("infection", level="cond").mean(),
                        mean_neu=g[m].xs("neutral", level="cond").mean(), mean_delta=x.mean(), d=d, p_perm=p))
pd.DataFrame(res).to_csv(OUT / "control_analyses.csv", index=False)

# replication in run 2, condition-specific GS coefficients, strictly alternating blocks; runs pooled as the mean of the two contrasts
rep = []; keep = df.variant.isin(["base", "nuisance"])
for (bs, gsr, lag, var), g in df[keep].groupby(["blockset", "gsr", "lag", "variant"], sort=False):
    p = g.pivot_table(index="sid", columns=["run", "cond"], values=["em_sync", "em_edge"])
    for m in ("em_sync", "em_edge"):
        dl = {r: p[(m, r, "infection")] - p[(m, r, "neutral")] for r in ("S1", "S2")}
        for run, x in (("S1", dl["S1"]), ("S2", dl["S2"]), ("pooled", ((dl["S1"] + dl["S2"]) / 2))):
            x = x.dropna().values; d, pv = signflip(x)
            rep.append(dict(run=run, blockset=bs, gsr=gsr, lag=lag, variant=var, measure=m, n=len(x), mean_delta=x.mean(), d=d, p_perm=pv))
pd.DataFrame(rep).to_csv(OUT / "control_replication.csv", index=False)

# somatomotor and visual networks by hemisphere (synchrony-based, run 1, onset)
nd = pd.DataFrame(net); hres = []
for (gsr, var), g in nd.groupby(["gsr", "variant"], sort=False):
    g = g.set_index(["sid", "cond"]); dl = {}
    for name in ("som", "som_lh", "som_rh", "vis_lh", "vis_rh") + tuple(NETS):
        x = (g[name].xs("infection", level="cond") - g[name].xs("neutral", level="cond")).values; dl[name] = x; d, p = signflip(x)
        hres.append(dict(gsr=gsr, variant=var, quantity=name, d=d, p_perm=p, p_t=stats.ttest_1samp(x, 0).pvalue, mean_delta=x.mean()))
    for a, b in [("som_lh", "som_rh"), ("vis_lh", "vis_rh")]:
        x = dl[a] - dl[b]; d, p = signflip(x)
        hres.append(dict(gsr=gsr, variant=var, quantity=f"{a} minus {b}", d=d, p_perm=p, p_t=stats.ttest_1samp(x, 0).pvalue, mean_delta=x.mean()))
pd.DataFrame(hres).to_csv(OUT / "control_networks.csv", index=False)

# button presses: responses per block from the in-scanner rating logs, by condition
rt = pd.read_csv(REL / "data/ratings/block_ratings.csv"); rt = rt[rt.sid.isin(subs)]; bres = []
for sid in rt.sid.unique():
    for ses in (1, 2):
        evf = BIDS / f"sub-{sid:02d}/func/sub-{sid:02d}_ses-S{ses}_task-picrate_events.tsv"
        r = rt[(rt.sid == sid) & (rt.session == ses)]
        if r.empty or not evf.exists(): continue
        tt = pd.read_csv(evf, sep="\t").trial_type.to_numpy()
        r = r.assign(cond=tt[r.order.to_numpy()], n_resp=r[["disgust", "fear", "infect"]].notna().sum(1))
        for cond, g in r.groupby("cond"):
            bres.append(dict(sid=sid, run=ses, cond=cond, presses_per_block=g.n_resp.mean(), mean_rating=g.mean3.mean(), n_blocks=len(g)))
bd = pd.DataFrame(bres); bout = []
for ses, g in bd.groupby("run"):
    p = g.pivot(index="sid", columns="cond", values=["presses_per_block", "mean_rating"])
    for q in ("presses_per_block", "mean_rating"):
        x = (p[q]["infection"] - p[q]["neutral"]).dropna().values; tt = stats.ttest_1samp(x, 0)
        bout.append(dict(run=ses, quantity=q, n=len(x), infection=p[q]["infection"].mean(), neutral=p[q]["neutral"].mean(),
                         difference=x.mean(), t=tt.statistic, p=tt.pvalue))
pd.DataFrame(bout).to_csv(OUT / "control_buttons.csv", index=False)

# across-run covariance of the contrast, decomposed into the four level covariances
cres = []
lv = df[(df.variant == "base") & (df.lag == 0) & (df.blockset == "all")]
for gsr in ("none", "gsr"):
    for m in ("em_sync", "em_edge"):
        p = lv[lv.gsr == gsr].pivot_table(index="sid", columns=["run", "cond"], values=m).dropna()
        I1, N1, I2, N2 = (p[("S1", "infection")], p[("S1", "neutral")], p[("S2", "infection")], p[("S2", "neutral")])
        c = lambda a, b: float(np.cov(a, b)[0, 1])
        cres.append(dict(gsr=gsr, measure=m, n=len(p), r_delta=stats.pearsonr(I1 - N1, I2 - N2)[0],
                         cov_delta=c(I1 - N1, I2 - N2), cov_I1_I2=c(I1, I2), cov_N1_N2=c(N1, N2), cov_I1_N2=c(I1, N2), cov_N1_I2=c(N1, I2),
                         r_I1_I2=stats.pearsonr(I1, I2)[0], r_N1_N2=stats.pearsonr(N1, N2)[0], r_I1_N2=stats.pearsonr(I1, N2)[0],
                         r_N1_I2=stats.pearsonr(N1, I2)[0], r_I1_N1=stats.pearsonr(I1, N1)[0], r_I2_N2=stats.pearsonr(I2, N2)[0],
                         var_I1=float(np.var(I1, ddof=1)), var_N1=float(np.var(N1, ddof=1)), var_I2=float(np.var(I2, ddof=1)), var_N2=float(np.var(N2, ddof=1))))
pd.DataFrame(cres).to_csv(OUT / "retest_covariance.csv", index=False)
for f in ("control_analyses.csv", "control_replication.csv", "control_networks.csv", "control_buttons.csv", "retest_covariance.csv"):
    print(f); print(pd.read_csv(OUT / f).round(4).to_string())
