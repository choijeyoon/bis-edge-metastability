"""Recompute the Schaefer-400 analyses of the Network Neuroscience revision on the correctly parcellated time series
(fMRIPrep_BIDS/parcellation_multires, templateflow Schaefer 2018 atlases in MNI152NLin2009cAsym). Definitions are the
canonical ones (z within condition after concatenating blocks, 9-TR windows at onset, first block excluded).

Per subject x session (cached in results/tables/netneuro_corrected/_cache_maps_S{ses}/sub-XX.npz):
  global synchrony-based EM (concatenated windows) infection/neutral, no GSR and GSR, lag 0        -> global_corrected_S{ses}.csv (EM_Inf, EM_Neu, Delta)
  network synchrony-based EM per Yeo network, infection/neutral, GSR x lag 0/2       -> network_delta_sync_{gsr}_lag{lag}_S{ses}.csv, network_stats_sync_S{ses}.csv
  parcel edge-wise EM difference, GSR x lag 0/2                            -> node_delta_edgewise_{gsr}_lag{lag}_S{ses}.csv, node_stats_edgewise_S{ses}.csv
  block-level EM (z within block, SD of R(t)), no GSR, lag 0               -> block_level_EM_corrected.csv
  within-block amplitude by condition and edge-wise dEM (run-z / cond-z)   -> amplitude_check.csv, amplitude_check_subjects.csv
Also writes a .mat mirror (parcellation_corrected/sub-XX/Schaefer400_S{ses}/SubjectXX_Schaefer400.mat, 400 x 360 incl.
dummies) so the repository pipeline runs unchanged with BIS_RAW_TS_ROOT pointing to it.
usage: python3 recompute_corrected.py <BIS_edge root> [--budget 160]   (rerun until it prints "all cached")
"""
import argparse, re, time
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from scipy.io import savemat
from statsmodels.stats.multitest import multipletests

TR, DUMMY, BLK = 2.0, 4, 9
NAMES = ["Visual", "Somatomotor", "DorsalAttention", "VentralAttention", "Limbic", "Control", "DefaultMode"]
KEY = {"Vis": 1, "SomMot": 2, "DorsAttn": 3, "SalVentAttn": 4, "Limbic": 5, "Cont": 6, "Default": 7}
GSRS = ("none", "gsr"); LAGS = (0, 2)


def zscore(ts):
    mu = ts.mean(1, keepdims=True); sd = ts.std(1, keepdims=True, ddof=1); sd[sd == 0] = 1; return (ts - mu) / sd


def regress_gs(ts, gs):
    gs = gs - gs.mean(); beta = (ts * gs).sum(1) / (gs @ gs); return ts - beta[:, None] * gs[None, :]


def rts(z):
    n = z.shape[0]; s = z.sum(0); s2 = (z * z).sum(0); return 0.5 * (s ** 2 - s2) / (n * (n - 1) / 2)


def em_sync(z):
    return float(rts(z).std(ddof=1))


def edgewise_sd(z):
    T = z.shape[1]; z2 = z * z; m1 = z @ z.T / T; m2 = z2 @ z2.T / T
    var = (m2 - m1 ** 2) * T / (T - 1); iu = np.triu_indices(z.shape[0], 1); return float(np.sqrt(np.clip(var[iu], 0, None)).mean())


def node_edgewise(z):
    """mean over j != i of SD_t[z_i z_j], from the N x N moment matrices."""
    T = z.shape[1]; z2 = z * z; m1 = z @ z.T / T; m2 = z2 @ z2.T / T
    sd = np.sqrt(np.clip((m2 - m1 ** 2) * T / (T - 1), 0, None)); np.fill_diagonal(sd, 0.0)
    return sd.sum(1) / (z.shape[0] - 1)


def net_sync(z, yeo):
    return np.array([em_sync(z[yeo == k]) for k in range(1, 8)])


def extract(ts, ev, lag):
    T = ts.shape[1]; out = {"infection": [], "neutral": []}
    for _, e in ev.iterrows():
        s0 = int(np.floor((e.onset - DUMMY * TR) / TR)); s = s0 + lag
        if s0 < 0 or s + BLK > T: continue
        out[e.trial_type].append(ts[:, s:s + BLK])
    return out


def per_subject(sid, ses, root, yeo):
    bids = root / "fMRIPrep_BIDS"
    raw = np.load(bids / f"parcellation_multires/sub-{sid:02d}/sub-{sid:02d}_S{ses}_schaefer400.npy").astype(float)
    mirror = bids / f"parcellation_corrected/sub-{sid:02d}/Schaefer400_S{ses}/Subject{sid:02d}_Schaefer400.mat"
    if not mirror.exists():
        mirror.parent.mkdir(parents=True, exist_ok=True); savemat(str(mirror), {"schaeferts": raw})
    ts = raw[:, DUMMY:]
    conf = pd.read_csv(bids / f"results/sub-{sid:02d}/ses-S{ses}/func/sub-{sid:02d}_ses-S{ses}_task-picrate_desc-confounds_timeseries.tsv", sep="\t")
    gs = conf["global_signal"].to_numpy()[DUMMY:]
    ev = pd.read_csv(bids / f"sub-{sid:02d}/func/sub-{sid:02d}_ses-S{ses}_task-picrate_events.tsv", sep="\t")
    out = {}
    for gi, gsr in enumerate(GSRS):
        src = regress_gs(ts, gs) if gsr == "gsr" else ts
        for li, lag in enumerate(LAGS):
            bl = extract(src, ev, lag)
            zi, zn = zscore(np.concatenate(bl["infection"], 1)), zscore(np.concatenate(bl["neutral"], 1))
            out[f"node_{gsr}_{lag}"] = node_edgewise(zi) - node_edgewise(zn)
            out[f"net_inf_{gsr}_{lag}"], out[f"net_neu_{gsr}_{lag}"] = net_sync(zi, yeo), net_sync(zn, yeo)
            if lag == 0:
                out[f"global_{gsr}"] = np.array([em_sync(zi), em_sync(zn)])
                # amplitude check: run-z within-block variance; edge-wise dEM under run-z and cond-z
                zr = zscore(src); blr = extract(zr, ev, 0)
                amp = {c: np.mean([b.var(1, ddof=1).mean() for b in blr[c]]) for c in blr}
                em_run = {c: edgewise_sd(np.concatenate(blr[c], 1)) for c in blr}
                em_cond = {c: edgewise_sd(zscore(np.concatenate(blr[c], 1))) for c in blr}
                out[f"amp_{gsr}"] = np.array([amp["infection"], amp["neutral"], em_run["infection"] - em_run["neutral"], em_cond["infection"] - em_cond["neutral"]])
    # block-level EM (no GSR): z within block, SD of R(t); first block dropped by the onset rule
    rows = []
    for i, (_, e) in enumerate(ev.iterrows()):
        s = int(np.floor((e.onset - DUMMY * TR) / TR))
        if s < 0 or s + BLK > ts.shape[1]: continue
        rows.append((i, e.onset, 1 if e.trial_type == "infection" else 0, em_sync(zscore(ts[:, s:s + BLK]))))
    out["blocks"] = np.array(rows)
    return out


def aggregate(root, out, yeo, subs):
    glob_rows = {1: [], 2: []}; blocks = []; amp = []
    for ses in (1, 2):
        cache = out / f"_cache_maps_S{ses}"; node = {}; net = {}; net_inf = {}; net_neu = {}; ids = []
        for sid in subs:
            f = cache / f"sub-{sid:02d}.npz"
            if not f.exists(): continue
            d = np.load(f); ids.append(sid)
            for gsr in GSRS:
                for lag in LAGS:
                    node.setdefault((gsr, lag), []).append(d[f"node_{gsr}_{lag}"])
                    net_inf.setdefault((gsr, lag), []).append(d[f"net_inf_{gsr}_{lag}"]); net_neu.setdefault((gsr, lag), []).append(d[f"net_neu_{gsr}_{lag}"])
                g = d[f"global_{gsr}"]; glob_rows[ses].append(dict(SubjectID=sid, gsr=gsr, EM_Inf=g[0], EM_Neu=g[1], Delta=g[0] - g[1]))
                a = d[f"amp_{gsr}"]; amp.append(dict(session=ses, sid=sid, gsr=gsr, amp_inf=a[0], amp_neu=a[1], dEM_edge_run=a[2], dEM_edge_cond=a[3]))
            for order, onset, cond, em in d["blocks"]:
                blocks.append(dict(order=int(order), trial_type="infection" if cond else "neutral", em=em, onset=onset, sid=sid, session=ses))
        if not ids: continue
        rows = []
        for (gsr, lag), L in node.items():
            D = np.array(L); pd.DataFrame(D, index=ids, columns=[f"p{i + 1}" for i in range(400)]).to_csv(out / f"node_delta_edgewise_{gsr}_lag{lag}_S{ses}.csv")
            t, p = stats.ttest_1samp(D, 0); q = multipletests(p, method="fdr_bh")[1]
            for i in range(400): rows.append(dict(gsr=gsr, lag=lag, parcel=i + 1, yeo=NAMES[yeo[i] - 1], mean_delta=D[:, i].mean(), d=D[:, i].mean() / D[:, i].std(ddof=1), t=t[i], p=p[i], q=q[i]))
        pd.DataFrame(rows).to_csv(out / f"node_stats_edgewise_S{ses}.csv", index=False)
        rows = []
        for (gsr, lag) in node:
            I, N = np.array(net_inf[(gsr, lag)]), np.array(net_neu[(gsr, lag)]); D = I - N
            df = pd.DataFrame(D, index=ids, columns=NAMES); df.to_csv(out / f"network_delta_sync_{gsr}_lag{lag}_S{ses}.csv")
            t, p = stats.ttest_1samp(D, 0); q = multipletests(p, method="fdr_bh")[1]
            for k, name in enumerate(NAMES):
                rows.append(dict(gsr=gsr, lag=lag, network=name, em_inf=I[:, k].mean(), em_neu=N[:, k].mean(), mean_delta=D[:, k].mean(), d=D[:, k].mean() / D[:, k].std(ddof=1), t=t[k], p=p[k], q=q[k], n_down=int((D[:, k] < 0).sum())))
        pd.DataFrame(rows).to_csv(out / f"network_stats_sync_S{ses}.csv", index=False)
        g = pd.DataFrame(glob_rows[ses]); g[g.gsr == "none"].drop(columns="gsr").to_csv(out / f"global_corrected_S{ses}.csv", index=False)
        g[g.gsr == "gsr"].drop(columns="gsr").to_csv(out / f"global_corrected_gsr_S{ses}.csv", index=False)
    pd.DataFrame(blocks).to_csv(out / "block_level_EM_corrected.csv", index=False)
    a = pd.DataFrame(amp); a["amp_diff"] = a.amp_inf - a.amp_neu; a.to_csv(out / "amplitude_check_subjects.csv", index=False)
    summ = []
    for (ses, gsr), g in a.groupby(["session", "gsr"]):
        x = g.amp_diff.values; t, p = stats.ttest_1samp(x, 0)
        r1, p1 = stats.pearsonr(g.amp_diff, g.dEM_edge_run); r2, p2 = stats.pearsonr(g.amp_diff, g.dEM_edge_cond)
        summ.append(dict(session=ses, gsr=gsr, n=len(g), amp_inf=g.amp_inf.mean(), amp_neu=g.amp_neu.mean(), d_amp=x.mean() / x.std(ddof=1), p_amp=p,
                         d_dEM_edge_run=g.dEM_edge_run.mean() / g.dEM_edge_run.std(ddof=1), d_dEM_edge_cond=g.dEM_edge_cond.mean() / g.dEM_edge_cond.std(ddof=1),
                         r_amp_dEMrun=r1, p_r_run=p1, r_amp_dEMcond=r2, p_r_cond=p2))
    pd.DataFrame(summ).to_csv(out / "amplitude_check.csv", index=False)
    for ses in (1, 2):
        f = out / f"network_stats_sync_S{ses}.csv"
        if f.exists(): print(f"\n=== network_stats_sync_S{ses}"); print(pd.read_csv(f).round(3).to_string())
    print("\n=== amplitude_check"); print(pd.DataFrame(summ).round(3).to_string())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("root"); ap.add_argument("--budget", type=float, default=160); a = ap.parse_args()
    root = Path(a.root); out = root / "bis_edge_metastability/results/tables/netneuro_corrected"; out.mkdir(parents=True, exist_ok=True)
    lab = pd.read_csv(root / "fMRIPrep_BIDS/parcellation/atlases_templateflow/schaefer400_7net_labels.tsv", sep="\t")
    yeo = np.array([KEY[s.split("_")[2]] for s in lab.name]); assert len(yeo) == 400
    pm = root / "fMRIPrep_BIDS/parcellation_multires"
    subs = sorted(int(m.group(1)) for p in pm.iterdir() if (m := re.match(r"sub-(\d+)$", p.name)))
    t0 = time.time(); pending = 0
    for ses in (1, 2):
        cache = out / f"_cache_maps_S{ses}"; cache.mkdir(exist_ok=True)
        for sid in subs:
            f = cache / f"sub-{sid:02d}.npz"
            if f.exists() or not (pm / f"sub-{sid:02d}/sub-{sid:02d}_S{ses}_schaefer400.npy").exists(): continue
            if time.time() - t0 > a.budget: pending += 1; continue
            res = per_subject(sid, ses, root, yeo); np.savez(f, **res); print(f"S{ses} sub-{sid:02d} {time.time() - t0:.0f}s", flush=True)
    if pending: print(f"{pending} subject-sessions pending; rerun"); return
    print("all cached; aggregating"); aggregate(root, out, yeo, subs)


if __name__ == "__main__":
    main()
