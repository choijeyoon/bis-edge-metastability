# Data dictionary

All files are keyed by `SubjectID` (or `sid`), an integer from 1 to 55 assigned at recruitment; 44 participants have imaging data. No names, dates or free text are included.

| file | one row per | columns |
|---|---|---|
| `participants.csv` | participant | `Sex` (F/M), `Age` (years) |
| `edge_metastability/global_S{1,2}.csv` | participant | `EM_Inf`, `EM_Neu`, `Delta` under the pre-specified convention (synchrony-based, concatenated windows, window at onset, z within condition, Schaefer 400, no GSR) |
| `edge_metastability/global_gsr_S{1,2}.csv` | participant | the same after global signal regression |
| `edge_metastability/spec_curve_subject_deltas_S{1,2}.csv` | participant × specification | `lag` (0-3 TR), `em` (concat, within, between, edgewise = edge-wise), `ztime` (run, cond), `parc`, `gsr`, `blockset` (all, common), `n_inf`, `n_neu`, `em_inf`, `em_neu`, `delta` |
| `edge_metastability/network_delta_sync_{none,gsr}_lag0_S{1,2}.csv` | participant | infection minus neutral within-network synchrony-based EM for the seven Yeo networks |
| `edge_metastability/block_level_EM.csv` | block | `order` (0-39 within run), `trial_type`, `em` (SD of R(t) within the 9-TR window, z within block), `onset` (s), `sid`, `session` |
| `questionnaires/items_{pre,post}.csv` | participant | item responses: `PVD_1..15` (0-6), `IND_1..8`, `COL_1..8` (0-6); IND 1-4 = horizontal, 5-8 = vertical individualism; COL likewise |
| `questionnaires/scores.csv` | participant | scale sums `IND`, `HI`, `VI`, `COL`, `HC`, `VC` at pre and post; `PVD_pre`, `PVD_post` (reverse-keyed, rebuilt by identifier) |
| `ratings/block_ratings.csv` | block | in-scanner ratings (1-4) of `disgust`, `fear`, `infect` (infection threat) and their `mean3`, keyed by `sid`, `session`, `order` |
| `motion/fd_per_block.csv` | block | `block_mean_fd` (mm, fMRIPrep framewise displacement) with `subject`, `session`, `trial_type`, `block_onset` |
| `motion/fd_per_run.csv`, `fd_per_subject_*.csv` | run / participant | summary FD statistics |
| `stimuli/stimulus_features.csv` (images not included) | image feature | means, SDs, t, d and FDR p for the ten low-level features (Supplementary Table S1) |
| `example/sub-01_S1_schaefer400.npy` | — | 400 × 360 parcel time series of one participant (four dummy volumes included), used in Figures 1 and 3d |
| `atlases/schaefer*_7net_labels.tsv` | parcel | TemplateFlow label tables; network is the third underscore-separated field of `name` |

Instruments: `IND`/`COL` items are the 16-item horizontal–vertical individualism–collectivism scale (Triandis & Gelfand, 1998, JPSP 74, 118–128) in the Japanese translation of Choi & Sugiura (2026, Personality and Individual Differences 250, 113524).

Timing: TR = 2 s; the first four volumes are dummies; blocks are 18 s (one image for 9 s, rated three times, then 9 s fixation); block windows are 9 TR at onset. Session 1 and session 2 are two consecutive runs with identical parameters.
