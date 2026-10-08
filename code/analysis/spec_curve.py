"""Specification curve on correctly parcellated BIS time series (fMRIPrep MNI152NLin2009cAsym res-2 BOLD, templateflow
Schaefer 2018 atlases in the same space). Identical definitions to src/supplementary/spec_curve.py; the parcellation
dimension now has six levels (100, 200, 400, 600, 800, 1000) -> 480 specifications.
usage: python3 spec_curve_corrected.py <BIS_edge root> [session]
writes bis_edge_metastability/results/tables/netneuro_corrected/{spec_curve_subject_deltas,spec_curve_summary,spec_curve_joint}_S{ses}.csv
"""
import sys, re, itertools, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(sys.argv[1]); SES = int(sys.argv[2]) if len(sys.argv) > 2 else 1
BIDS = ROOT / "fMRIPrep_BIDS"; PM = BIDS / "parcellation_multires"; OUT = ROOT / "bis_edge_metastability/results/tables/netneuro_corrected"; OUT.mkdir(parents=True, exist_ok=True)
TR, DUMMY, BLK = 2.0, 4, 9
LAGS = [0, 1, 2, 3]; EMS = ["concat", "within", "between", "edgewise"]; ZTIMES = ["run", "cond"]; PARCS = [100, 200, 400, 600, 800, 1000]; GSRS = ["none", "gsr"]; BLOCKSETS = ["all", "common"]


def zscore(ts):
    mu = ts.mean(1, keepdims=True); sd = ts.std(1, keepdims=True, ddof=1); sd[sd == 0] = 1; return (ts - mu) / sd


def regress_gs(ts, gs):
    gs = gs - gs.mean(); beta = (ts * gs).sum(1) / (gs @ gs); return ts - beta[:, None] * gs[None, :]


def rts(z):
    n = z.shape[0]; s = z.sum(0); s2 = (z * z).sum(0); return 0.5 * (s ** 2 - s2) / (n * (n - 1) / 2)


def edgewise_sd(z):
    T = z.shape[1]; z2 = z * z; m1 = z @ z.T / T; m2 = z2 @ z2.T / T
    var = (m2 - m1 ** 2) * T / (T - 1); iu = np.triu_indices(z.shape[0], 1); return float(np.sqrt(np.clip(var[iu], 0, None)).mean())


def em_value(blocks, em, ztime):
    cat = np.concatenate(blocks, 1)
    if ztime == "cond": cat = zscore(cat)
    if em == "edgewise": return edgewise_sd(cat)
    R = rts(cat)
    if em == "concat": return float(R.std(ddof=1))
    Rb = R.reshape(len(blocks), BLK)
    return float(Rb.std(1, ddof=1).mean()) if em == "within" else float(Rb.mean(1).std(ddof=1))


def extract(ts, ev, lag, blockset):
    T = ts.shape[1]; out = {"infection": [], "neutral": []}; events = ev.iloc[:-1] if blockset == "common" else ev
    for _, e in events.iterrows():
        s0 = int(np.floor((e.onset - DUMMY * TR) / TR)); s = s0 + lag
        if s0 < 0 or s + BLK > T: continue
        out[e.trial_type].append(ts[:, s:s + BLK])
    return out


subs = sorted(int(m.group(1)) for p in PM.iterdir() if (m := re.match(r"sub-(\d+)$", p.name)) and (p / f"sub-{int(m.group(1)):02d}_S{SES}_schaefer1000.npy").exists())
print(len(subs), "subjects, session", SES, flush=True)
rows = []; t0 = time.time()
for sid in subs:
    conf = pd.read_csv(BIDS / f"results/sub-{sid:02d}/ses-S{SES}/func/sub-{sid:02d}_ses-S{SES}_task-picrate_desc-confounds_timeseries.tsv", sep="\t")
    gs = conf["global_signal"].to_numpy()[DUMMY:]
    ev = pd.read_csv(BIDS / f"sub-{sid:02d}/func/sub-{sid:02d}_ses-S{SES}_task-picrate_events.tsv", sep="\t")
    for parc in PARCS:
        raw = np.load(PM / f"sub-{sid:02d}/sub-{sid:02d}_S{SES}_schaefer{parc}.npy").astype(float)[:, DUMMY:]
        for gsr in GSRS:
            ts = regress_gs(raw, gs) if gsr == "gsr" else raw; ts_runz = zscore(ts)
            for lag in LAGS:
                for bs in BLOCKSETS:
                    if lag >= 1 and bs == "common": continue
                    for zt in ZTIMES:
                        bl = extract(ts_runz if zt == "run" else ts, ev, lag, bs)
                        for em in EMS:
                            ei = em_value(bl["infection"], em, zt); en = em_value(bl["neutral"], em, zt)
                            rows.append(dict(sid=sid, lag=lag, em=em, ztime=zt, parc=f"schaefer{parc}", gsr=gsr, blockset=bs, n_inf=len(bl["infection"]), n_neu=len(bl["neutral"]), em_inf=ei, em_neu=en, delta=ei - en))
    print(f"  sub-{sid:02d} done ({time.time() - t0:.0f}s)", flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / f"spec_curve_subject_deltas_S{SES}.csv", index=False)
keys = ["lag", "em", "ztime", "parc", "gsr", "blockset"]; rng = np.random.default_rng(20260911)
piv = df.pivot_table(index="sid", columns=keys, values="delta"); D = piv.values; n = D.shape[0]
d_obs = D.mean(0) / D.std(0, ddof=1); t_obs = d_obs * np.sqrt(n); p_t = 2 * stats.t.sf(np.abs(t_obs), n - 1); B = 10000
null_d = np.empty((B, D.shape[1]))
for b0 in range(0, B, 250):
    flips = rng.choice([-1.0, 1.0], size=(min(250, B - b0), n)); X = flips[:, :, None] * D[None, :, :]; null_d[b0:b0 + X.shape[0]] = X.mean(1) / X.std(1, ddof=1)
null_p = 2 * stats.t.sf(np.abs(null_d * np.sqrt(n)), n - 1); p_perm = (np.abs(null_d) >= np.abs(d_obs)[None, :]).mean(0)
summ = pd.DataFrame([dict(zip(keys, col)) | dict(n=n, mean_delta=D[:, j].mean(), d=d_obs[j], t=t_obs[j], p_t=p_t[j], p_perm=p_perm[j], n_down=int((D[:, j] < 0).sum())) for j, col in enumerate(piv.columns)]).sort_values("d")
summ.to_csv(OUT / f"spec_curve_summary_S{SES}.csv", index=False)
sig = p_t < .05; null_sig = null_p < .05
joint = pd.DataFrame([dict(statistic="median_d", observed=np.median(d_obs), p=(np.median(null_d, 1) <= np.median(d_obs)).mean()),
                      dict(statistic="n_sig", observed=int(sig.sum()), p=(null_sig.sum(1) >= sig.sum()).mean()),
                      dict(statistic="n_sig_negative", observed=int((sig & (d_obs < 0)).sum()), p=((null_sig & (null_d < 0)).sum(1) >= (sig & (d_obs < 0)).sum()).mean()),
                      dict(statistic="sum_sig_d", observed=float(d_obs[sig].sum()), p=((null_d * null_sig).sum(1) <= d_obs[sig].sum()).mean())])
joint["n_specs"] = D.shape[1]; joint["B"] = B; joint.to_csv(OUT / f"spec_curve_joint_S{SES}.csv", index=False)
print(joint.to_string()); print(summ.groupby("parc").d.median().round(3).to_string())
