# Correction notes

Two data-processing errors and three errors of description in the original version of this work (Zenodo doi:10.5281/zenodo.19807235) have been found and corrected. This note records what each error was, how it was found, what it changed, and where the corrected files are. Every statistic reported in the Network Neuroscience manuscript is computed from the corrected data.

## 1. Questionnaire difference scores (found 2026-08-31)

**What happened.** The pre- and post-questionnaire Google Forms exports were joined on row position instead of participant identifier when the difference columns (post minus pre) of the individualism, collectivism and perceived-vulnerability scales were computed. The pre export had 51 rows (a pilot participant first, and three participants who have no post record); the post export had 48. Row positions matched for the first 17 analyzed participants and diverged afterwards, so each later difference combined one participant's post total with the preceding participant's pre total. The pre and post totals themselves were correct.

**How it was found.** A check of the public data file showed that the difference column did not equal post minus pre.

**What it changed.** The reported correlation between the change in individualism and the change in edge metastability was an artifact of the misaligned join. Rebuilt by identifier, the change score correlates r = -0.07 with delta EM; the change score also has no reliable variance (Supplementary Table S9), so the question cannot be tested with this scale. Two further problems in the same exports were fixed at the same time: post-scan perceived-vulnerability items 2, 4, 5, 10, 12 and 15 had not been reverse-keyed, and one pre-questionnaire participant number was mistyped (043 entered as 044).

**Files.** `data/questionnaires/items_{pre,post}.csv` (item responses keyed by participant identifier) and `data/questionnaires/scores.csv` (scale sums rebuilt from them); `code/analysis/secondary_analyses.py` recomputes every questionnaire statistic from these files.

## 2. Parcellation (found 2026-09-14)

**What happened.** The original MATLAB extraction scripts loaded the Schaefer 400 atlas distributed in the FSL MNI152 2-mm grid (91 x 109 x 91 voxels, left-right flipped relative to the fMRIPrep output), took the linear indices of the voxels of each parcel, and applied those indices to the fMRIPrep preprocessed images, which are in MNI152NLin2009cAsym at 2 mm (97 x 115 x 97 voxels). Because the two grids differ, each linear index lands on a different voxel, so every "parcel" in the original time series was a fixed but anatomically arbitrary set of voxels.

**How it was found.** While extending the specification curve to atlas resolutions of 100 to 1000 parcels, we parcellated the fMRIPrep images with TemplateFlow atlases in the correct space and compared the 400-parcel result with the original `.mat` files: the per-parcel correlation had a median of 0.38. A Python re-implementation of the original indexing (x-flip plus Fortran-order linear index) reproduces the original time series exactly (r = 1.000), which confirms the mechanism.

**What it changed.** Because the scrambled voxel sets were spatially dispersed, the original parcel signals were unusually dominated by the global signal. Global measures were therefore only mildly affected; parcel- and network-level results were invalid.

| Quantity | Original | Corrected |
|---|---|---|
| Pre-specified test, session 1: d, permutation p, n with reduction | -0.30, .047, 27/44 | -0.31, .042, 27/44 |
| Correlation of per-participant delta EM, original vs corrected | | 0.92 (S1), 0.86 (S2) |
| Specification curve, same 160 specifications: median d, n significant | -0.27, 71 | -0.33, 89 |
| Specification curve, 480 specifications (six resolutions) | | median -0.34, 263 significant, joint p <= .0001 |
| Condition x sex (S1): b, p, Holm p | -0.099, .017, .035 | -0.124, .019, .039 |
| Condition x session: p, BF01 | .11, 3.8 | .071, 2.7 |
| Session 2 pre-specified: d, p | +0.18, .25 | +0.23, .14 |
| Within-block component (S1): d, p | -0.21, .17 | -0.41, .007 |
| Trait HI (pre / post): r, family-wise p | 0.41 / .040; 0.45 / .017 | 0.35 / .13; 0.35 / .13 |
| Parcels with q < .05, no GSR / GSR | 218 / 63 | 0 / 143 |
| Spatial correlation of parcel-wise d, original vs corrected | | 0.15 to 0.19 |
| Networks with q < .05, no GSR | somatomotor, dorsal attention (both q = .050) | somatomotor (q = .011) |
| Networks with q < .05, GSR | six of seven (limbic excepted) | six of seven (limbic excepted); visual largest |
| Block-level order effect: b per 10 blocks, p | -0.003, .47 | +0.013, < .001 |
| Within-block amplitude difference (S1, no GSR): d, p | -0.35, .025 | -0.04, .77 |

The conclusions that changed: the trait horizontal individualism correlation no longer survives family-wise correction; the anatomical maps and their description are replaced; the within-block component is now reduced; block-level metastability rises over the run; the condition contrast has no test-retest stability across the two runs (r = -0.31, a new analysis prompted by the correction).

**Files.** `code/extraction/extract_multires.py` (corrected parcellation with TemplateFlow atlases in the fMRIPrep space; the re-implementation of the faulty indexing used for the diagnosis is documented in its header), `code/analysis/recompute_schaefer400.py`, `code/analysis/spec_curve.py`, `data/atlases/` (label tables), `data/edge_metastability/` (corrected per-participant values), `results/tables/old_vs_corrected_summary.csv` (every reported statistic before and after). The original scrambled time series are not distributed.

## 3. Descriptions corrected (no data affected)

The earlier supplement described four-TR windows and a Hilbert-phase order parameter; the analysis used nine-TR windows and amplitude z-scores. It also described each block as three images shown for 3 s each; the task logs show one image per block, displayed for 9 s while three ratings were collected, followed by 9 s of fixation. The run duration was given as 760 s; it is 720 s (360 volumes at TR 2 s).
