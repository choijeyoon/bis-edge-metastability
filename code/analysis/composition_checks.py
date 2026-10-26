"""Composition of the condition contrast (Supplementary Table S16): static-connectivity surrogates, R(t) composition, amplitude checks,
order-confound subset (strictly alternating blocks 7-40) and the GSR effect without visual parcels.
Schaefer 400, run 1, window at onset, normalization within condition, all blocks. usage: python3 composition_checks.py <BIS_edge root>"""
import sys, re, json
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
ROOT = Path(sys.argv[1]); BIDS = ROOT / "fMRIPrep_BIDS"; PM = BIDS / "parcellation_multires"
OUT = ROOT / "bis_edge_metastability/results/tables/netneuro_corrected"; SEED = 20260911
TR, DUMMY, BLK, N = 2.0, 4, 9, 400

def zscore(ts):
    mu = ts.mean(1, keepdims=True); sd = ts.std(1, keepdims=True, ddof=1); sd[sd == 0] = 1; return (ts - mu) / sd
def regress_gs(ts, gs):
    gs = gs - gs.mean(); beta = (ts * gs).sum(1) / (gs @ gs); return ts - beta[:, None] * gs[None, :]
def rts(z):
    n = z.shape[0]; s = z.sum(0); s2 = (z * z).sum(0); return 0.5 * (s ** 2 - s2) / (n * (n - 1) / 2)
def edgewise_sd(z):
    T = z.shape[1]; z2 = z * z; m1 = z @ z.T / T; m2 = z2 @ z2.T / T
    var = (m2 - m1 ** 2) * T / (T - 1); iu = np.triu_indices(z.shape[0], 1); return float(np.sqrt(np.clip(var[iu], 0, None)).mean())
def extract(ts, ev, first_block=0):
    T = ts.shape[1]; out = {"infection": [], "neutral": []}
    for i, e in ev.iterrows():
        if i < first_block: continue
        s0 = int(np.floor((e.onset - DUMMY * TR) / TR))
        if s0 < 0 or s0 + BLK > T: continue
        out[e.trial_type].append(ts[:, s0:s0 + BLK])
    return out
def measures(cat):
    """cat: parcels x time, already z-scored within condition."""
    R = rts(cat); m = cat.mean(0); q = (cat ** 2).mean(0)
    C = np.corrcoef(cat); iu = np.triu_indices(N, 1); rbar = C[iu].mean()
    return dict(em_sync=R.std(ddof=1), em_edge=edgewise_sd(cat), rbar=rbar, N_var_m=N * m.var(ddof=1), var_q=q.var(ddof=1),
                sd_q=q.std(ddof=1), cov_term=(N * m.var(ddof=1) - 1) / (N - 1))
def signflip(x, B=10000):
    d = x.mean() / x.std(ddof=1); rng = np.random.default_rng(SEED)
    X = rng.choice([-1.0, 1.0], size=(B, len(x))) * x[None, :]; nd = X.mean(1) / X.std(1, ddof=1)
    return float(d), float((np.abs(nd) >= abs(d)).mean())

subs = sorted(int(m.group(1)) for p in PM.iterdir() if (m := re.match(r"sub-(\d+)$", p.name)) and (p / f"sub-{int(m.group(1)):02d}_S1_schaefer400.npy").exists())
lab = pd.read_csv(ROOT / "bis_edge_metastability/github/bis-edge-metastability/data/atlases/schaefer400_7net_labels.tsv", sep="\t").name
vis = np.array(["_Vis_" in s for s in lab])
K = 30; rows = []; rng = np.random.default_rng(SEED)
for sid in subs:
    conf = pd.read_csv(BIDS / f"results/sub-{sid:02d}/ses-S1/func/sub-{sid:02d}_ses-S1_task-picrate_desc-confounds_timeseries.tsv", sep="\t")
    gs = conf["global_signal"].to_numpy()[DUMMY:]
    ev = pd.read_csv(BIDS / f"sub-{sid:02d}/func/sub-{sid:02d}_ses-S1_task-picrate_events.tsv", sep="\t")
    raw = np.load(PM / f"sub-{sid:02d}/sub-{sid:02d}_S1_schaefer400.npy").astype(float)[:, DUMMY:]
    for gsr in ("none", "gsr"):
        ts = regress_gs(raw, gs) if gsr == "gsr" else raw
        bl = extract(ts, ev); bl7 = extract(ts, ev, first_block=6)
        for cond in ("infection", "neutral"):
            cat = zscore(np.concatenate(bl[cond], 1)); T = cat.shape[1]
            r = dict(sid=sid, gsr=gsr, cond=cond, **measures(cat))
            # covariance-matched Gaussian surrogates of the concatenated windows (same static covariance, no temporal structure)
            C = np.cov(cat); L = np.linalg.cholesky(C + 1e-8 * np.eye(N)); ss, se = [], []
            for _ in range(K):
                sur = zscore(L @ rng.standard_normal((N, T))); ss.append(rts(sur).std(ddof=1)); se.append(edgewise_sd(sur))
            r.update(sur_sync=float(np.mean(ss)), sur_edge=float(np.mean(se)))
            # extended amplitude checks on run-normalized signals (variance over the concatenated windows)
            catr = np.concatenate(extract(zscore(ts), ev)[cond], 1); pv = catr.var(1, ddof=1)
            r.update(amp_mean_var=float(pv.mean()), amp_sd_var=float(pv.std(ddof=1)), amp_sd_q=float((catr ** 2).mean(0).std(ddof=1)))
            # strictly alternating segment (blocks 7-40)
            cat7 = zscore(np.concatenate(bl7[cond], 1)); R7 = rts(cat7); Rb = R7.reshape(-1, BLK)
            r.update(alt_sync=R7.std(ddof=1), alt_between=Rb.mean(1).std(ddof=1), alt_edge=edgewise_sd(cat7), n_alt=len(bl7[cond]))
            # without visual-network parcels
            catv = zscore(np.concatenate(bl[cond], 1)[~vis]); r.update(novis_sync=rts(catv).std(ddof=1), novis_edge=edgewise_sd(catv))
            rows.append(r)
    print(f"sub-{sid:02d}", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / "composition_checks_subject.csv", index=False)
res = []
for gsr in ("none", "gsr"):
    g = df[df.gsr == gsr].set_index(["sid", "cond"])
    inf, neu = g.xs("infection", level="cond"), g.xs("neutral", level="cond")
    for col in ["em_sync", "em_edge", "rbar", "N_var_m", "var_q", "sd_q", "cov_term", "sur_sync", "sur_edge", "amp_mean_var", "amp_sd_var", "amp_sd_q", "alt_sync", "alt_between", "alt_edge", "novis_sync", "novis_edge"]:
        x = (inf[col] - neu[col]).values; d, p = signflip(x)
        res.append(dict(gsr=gsr, quantity=col, mean_inf=inf[col].mean(), mean_neu=neu[col].mean(), mean_delta=x.mean(), d=d, p_perm=p))
    for obs, sur in [("em_sync", "sur_sync"), ("em_edge", "sur_edge")]:
        x = ((inf[obs] - neu[obs]) - (inf[sur] - neu[sur])).values; d, p = signflip(x)
        res.append(dict(gsr=gsr, quantity=f"{obs} minus surrogate", mean_inf=np.nan, mean_neu=np.nan, mean_delta=x.mean(), d=d, p_perm=p))
        r_os = stats.pearsonr((inf[obs] - neu[obs]).values, (inf[sur] - neu[sur]).values)[0]
        res.append(dict(gsr=gsr, quantity=f"corr(delta {obs}, delta {sur}) across participants", mean_inf=np.nan, mean_neu=np.nan, mean_delta=r_os, d=np.nan, p_perm=np.nan))
out = pd.DataFrame(res); out.to_csv(OUT / "composition_checks.csv", index=False); print(out.round(4).to_string())
