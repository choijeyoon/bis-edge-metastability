"""Parcellate the BIS fMRIPrep BOLD (MNI152NLin2009cAsym, 2 mm) with Schaefer 2018 atlases at six resolutions
(templateflow, same space and grid). Chunked over time so it runs in < 1 GB of RAM.
usage: python3 extract_multires.py <BIS_edge root> <atlas dir> [sub-ids...]
writes <root>/fMRIPrep_BIDS/parcellation_multires/sub-XX/sub-XX_S{ses}_schaefer{n}.npy  (n_parcels x n_volumes, float32, all 360 volumes)
"""
import sys, re, gzip, shutil, os, tempfile
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import sparse

ROOT, ATL = Path(sys.argv[1]), Path(sys.argv[2]); RES = [100, 200, 400, 600, 800, 1000]
BIDS = ROOT / "fMRIPrep_BIDS"; OUT = BIDS / "parcellation_multires"; CH = 60
atl = {}
for n in RES:
    lab = np.asarray(nib.load(str(ATL / f"schaefer{n}_7net_MNI152NLin2009cAsym_res2.nii.gz")).dataobj).astype(np.int32).ravel()
    vox = np.flatnonzero(lab); cnt = np.bincount(lab, minlength=n + 1)[1:].astype(np.float32)
    atl[n] = sparse.csr_matrix((1.0 / cnt[lab[vox] - 1], (lab[vox] - 1, vox)), shape=(n, lab.size), dtype=np.float32)   # parcel-mean operator
subs = sys.argv[3:] or sorted(p.name for p in (BIDS / "results").iterdir() if re.match(r"sub-\d+$", p.name))
for sub in subs:
    for ses in (1, 2):
        f = BIDS / f"results/{sub}/ses-S{ses}/func/{sub}_ses-S{ses}_task-picrate_space-MNI152NLin2009cAsym_res-2_desc-preproc_bold.nii.gz"
        dst = OUT / sub / f"{sub}_S{ses}_schaefer1000.npy"
        if not f.exists() or dst.exists(): continue
        tmp = Path(tempfile.gettempdir()) / f.name[:-3]
        with gzip.open(f, "rb") as fi, open(tmp, "wb") as fo: shutil.copyfileobj(fi, fo, 16 * 1024 * 1024)   # decompress once; gz slicing re-reads the file per chunk
        img = nib.load(str(tmp), mmap=True); T = img.shape[3]; assert img.shape[:3] == (97, 115, 97), img.shape
        out = {n: np.zeros((n, T), np.float32) for n in RES}
        for t0 in range(0, T, CH):
            block = np.asarray(img.dataobj[..., t0:t0 + CH], dtype=np.float32).reshape(-1, min(CH, T - t0))
            for n, S in atl.items():
                out[n][:, t0:t0 + block.shape[1]] = S @ block
        (OUT / sub).mkdir(parents=True, exist_ok=True)
        for n in RES: np.save(OUT / sub / f"{sub}_S{ses}_schaefer{n}.npy", out[n])
        del img; os.remove(tmp)
        print(sub, ses, "ok", flush=True)


def emulate_original_indexing(atlas_fsl_img, bold_img):
    """Diagnostic only: reproduce the faulty parcellation of the original version (see CORRECTION.md).
    The original MATLAB scripts loaded the Schaefer atlas in the FSL MNI152 2-mm grid (91 x 109 x 91, x-axis flipped
    relative to the fMRIPrep grid), took the linear (column-major) indices of each parcel's voxels, and applied them
    to the fMRIPrep image (97 x 115 x 97) as if the grids were identical. This function repeats that operation and
    returns parcels x volumes means over the resulting, anatomically arbitrary, voxel sets; it reproduces the original
    time series with r = 1.000 and is not used anywhere in the corrected pipeline."""
    lab = np.asarray(atlas_fsl_img.dataobj).astype(int)[::-1]          # x-flip into the fMRIPrep orientation
    lab_lin = lab.ravel(order="F")                                       # MATLAB column-major linear order
    bold = np.asarray(bold_img.dataobj, dtype=np.float32)
    X = bold.reshape(-1, bold.shape[3], order="F")                       # fMRIPrep voxels in the same linear order
    n = lab_lin.max(); out = np.zeros((n, bold.shape[3]), np.float32)
    for k in range(1, n + 1):
        idx = np.flatnonzero(lab_lin == k); idx = idx[idx < X.shape[0]]
        out[k - 1] = X[idx].mean(0)
    return out
