# Pathogen-related images reduce cortical edge metastability: a specification-curve analysis of a block-design fMRI study

Code and derived data for Choi, Berjaga-Buisan, Garcia-Guzman, Sugiura and Deco, *Network Neuroscience* (manuscript in preparation). Forty-four adults viewed alternating 18-s blocks of infection-related and neutral images in two fMRI runs; we ask whether the images reduce cortical edge metastability, and how much of the answer depends on analysis conventions.

This release supersedes the Zenodo record of the earlier version (doi:10.5281/zenodo.19807235). Two data-processing errors in that version were found and corrected; `CORRECTION.md` documents both, with every reported statistic before and after.

## What is here

```
data/                      de-identified per-participant data (numeric participant IDs only)
  participants.csv         SubjectID, Sex, Age
  edge_metastability/      global EM per condition (with and without GSR), block-level EM, all 480
                           specification-curve values per participant (sessions 1 and 2), within-network EM differences
  questionnaires/          item responses (pre, post) and scale scores: individualism-collectivism (HI, VI, HC, VC), PVD
  ratings/                 in-scanner disgust, fear and infection-threat ratings for every block
  motion/                  framewise displacement per block, run, condition and session (fMRIPrep)
  stimuli/                 low-level image statistics of the two image sets (the images themselves are not redistributed)
  example/                 one participant's Schaefer-400 parcel time series and event file (Figures 1 and 3d)
  atlases/                 Schaefer 2018 label tables (100-1000 parcels, 7 networks) as distributed by TemplateFlow
results/tables/            every table in the paper and its supplement (CSV)
code/
  extraction/extract_multires.py     parcellation of the fMRIPrep output (needs the imaging derivatives)
  analysis/recompute_schaefer400.py  global, network, parcel and block-level EM at 400 parcels (needs derivatives)
  analysis/spec_curve.py             480-specification curve with joint inference (needs derivatives)
  analysis/secondary_analyses.py     everything downstream of the per-participant tables (runs from data/)
  analysis/validity_checks.py        test-retest of the contrast; block-level rating models (runs from data/)
  figures/build_figures.py           Figures 1-4 and S1-S4 (runs from data/ and results/)
  figures/render_brains.py           surface renders for Figures 2 and 4 (brainspace + surfplot)
  figures/render_fig1_brains.py      surface renders for Figure 1b
CORRECTION.md              the two corrections, with before-and-after values
```

## Reproducing the tables and figures from the released data

```
pip install -r requirements.txt
python code/analysis/secondary_analyses.py      # results/tables/*.csv (sex, session, decomposition, motion, traits, grid)
python code/analysis/validity_checks.py         # retest_reliability.csv, rating_lmm.csv
python code/figures/render_brains.py            # surface renders -> figures/brains/ (run under xvfb-run on a headless machine)
python code/figures/render_fig1_brains.py
python code/figures/build_figures.py            # figures/figure*.pdf|png|svg
```

`build_figures.py` draws labelled placeholders where Figure 1 shows the two example stimuli, because the images are not ours to redistribute; it reads the surface renders that the two render scripts write to `figures/brains/`. The figures themselves are not stored in the repository; they appear in the article. Figure 1 also needs `brainspace` for the parcel centroids.

The three scripts marked "needs derivatives" start from the fMRIPrep output (MNI152NLin2009cAsym, 2 mm) and the TemplateFlow atlases in that space; they take the data root as their first argument and write the per-participant tables that `data/edge_metastability/` contains. The preprocessed imaging data are available from the corresponding author under an institutional data-sharing agreement.

## Analysis conventions in one paragraph

Parcel time series are z-scored (within run, or within the concatenated blocks of a condition), 9-TR windows are placed at each block onset (shifted by 0-3 TR), and the global cofluctuation R(t) = [(Σ z_i)² − Σ z_i²]/(N(N−1)) is computed. Edge metastability is SD_t[R(t)] over the concatenated windows of a condition (synchrony-based), its within-block component (mean over windows of the within-window SD), its between-block component (SD of the window means), or the mean over parcel pairs of SD_t[z_i z_j] (edge-wise). Global signal regression, when applied, removes the fMRIPrep whole-brain signal from each parcel within run before normalization. The specification curve crosses shift (4) × definition (4) × normalization (2) × parcellation (6) × GSR (2) × block set (2), minus block-set duplicates at non-zero shifts, for 480 specifications; joint inference uses 10,000 sign-flip draws shared across specifications.

## Citation

See `CITATION.cff`. Please cite the paper and this repository (a versioned DOI is issued through Zenodo for each GitHub release).

## Licence

Code: MIT (see `LICENSE`). Derived data: CC BY 4.0. The Schaefer atlas label tables are redistributed under the MIT licence of the original release; the stimulus images are not included.
