# Machine Learning for Predicting Sediment Particle Size Distributions

UE24CS352A Machine Learning mini-project (PES University), individual submission.
Author: [Ujwal Sanikam L, PES1UG24CS506]

Predicts two properties of suspended sediment particle size distributions (PSDs) in
South San Francisco Bay from hydrodynamic and water-chemistry measurements:

- **d50**: median particle diameter (um)
- **sigma2**: variance of the PSD (um^2)

Two regression models are compared: **Random Forest (RF)** and **Support Vector
Regression (SVR)**. The problem follows Egan, *Machine learning for predicting sediment
particle size distributions* (Stanford), which is the problem statement for this project.

## Quick start (no raw data needed)

The cleaned dataset `dataset.csv` (897 hourly rows) is committed, so everything from the
modelling steps onwards runs without downloading the 2.5 GB raw data.

```bash
python -m venv venv
# Windows PowerShell:  .\venv\Scripts\Activate.ps1
# macOS / Linux:       source venv/bin/activate
pip install -r requirements.txt

python rf_model.py      # Random Forest: importance, tuning, evaluation   (~seconds)
python svr_model.py     # SVR + RF-vs-SVR comparison                       (~seconds to minutes)
python predict.py --sample      # demo: predictions vs measurements on held-out rows
python predict.py               # demo: interactive prediction
```

All random seeds are fixed, so results are reproducible. Figures are written to `figures/`
and numeric results to `rf_results.txt`, `svr_results.txt`, `feature_importance.csv` and
`model_comparison.csv`.

## Dataset

The paper's own data link (Stanford Box) is dead (404). The same field campaigns are
published in the Stanford Digital Repository (CC BY-NC-SA):

> Egan, G., Cowherd, M., Scheu, K., Spada, F., Manning, A., Jones, C., Chang, G.,
> Fringer, O., & Monismith, S. (2019). *South San Francisco Bay boundary layer and
> sediment dynamics field data.* Stanford Digital Repository.
> https://purl.stanford.edu/wv787xr0534

Three one-month deployments (Summer 2018, Winter 2019, Spring 2019). We use the P2
platform, where a LISST-100x measured PSDs every minute.

### Rebuilding the dataset from raw data (optional)

1. Download `NonVectrinoData.zip` (~2.5 GB) from the link above into the project folder.
   The two Vectrino zips and `README.txt` are not needed.
2. Run the pipeline:

```bash
python extract.py        # extracts only the files we use into ./data
python inspect_data.py   # optional: prints variables, shapes and time ranges of raw files
python build_dataset.py  # builds dataset.csv and exploratory figures
```

## Method

**Targets** (from the LISST at P2): d50 is the volume-weighted median diameter and sigma2 the
volume-weighted variance of diameter, both computed from the 32-bin size distribution.

**Features** (hourly means), 4 of the paper's 8 (the biology instruments, ac-9 and ECO-FL,
are not in the public data):

| Feature | Meaning | Source |
|---|---|---|
| S | salinity | CTD at P1 (P2's salinity sensor failed) |
| T | temperature | CTD at P2 |
| ub | near-bed wave orbital velocity | RBR pressure logger at P2, recomputed from the depth-corrected surface-elevation spectrum with linear wave theory (band 0.05-1 Hz, kh <= 3) |
| u | mean current speed | depth-averaged ADP speed at P1 (P2 has no ADP) |

**Quality control** (assumptions, all set at the top of `build_dataset.py`): LISST optical
transmission between 0.30 and 0.98; d50 >= 2 um; the 2 days before each instrument
cleaning (and the cleaning day) removed as biofouled; at least 20 valid LISST minutes and
2 RBR bursts per hour. The stored RBR `Hsig`/`omega` values were corrupted in Winter and
Spring, so wave statistics are recomputed from the spectra.

**Models** (scikit-learn):

- *RF*: feature importance averaged over 1000 runs (10 trees, 70/30 split); number of
  trees (96) chosen from the out-of-bag score.
- *SVR*: RBF kernel, standardised features and targets; C and epsilon tuned on a CV set
  (70/15/15 split).

**Evaluation**: the paper's random split, plus two stricter checks: a chronological split
within each season (first 70% of time to train, last 30% to test) and leave-one-season-out.

## Results (test r2)

| Evaluation | RF d50 | SVR d50 | RF sigma2 | SVR sigma2 |
|---|---|---|---|---|
| Random 70/15/15 split, mean of 20 repeats | 0.73 | 0.61 | 0.61 | 0.45 |
| Chronological split | 0.31 | 0.19 | 0.17 | -0.30 |
| Leave-one-season-out | negative | negative | negative | negative |

Random Forest feature importance (d50): T 0.49, S 0.35, ub 0.10, u 0.06.

## Limitations

- **Season structure.** S and T mostly separate the three seasons; season alone explains
  57% (d50) and 46% (sigma2) of the variance. The random-split scores overstate how well the
  models generalise.
- **Temporal autocorrelation.** Neighbouring hours are nearly identical (lag-1
  autocorrelation 0.84-0.90), which inflates random-split scores; the chronological and
  leave-one-season-out checks expose this. All leave-one-season-out scores are negative.
- **Out-of-range inputs.** Models cannot extrapolate. `predict.py` warns when an input lies
  outside the training range.
- **Spatial mismatch.** S and u come from P1, about 1.2 km from the P2 LISST.
- **Wave data.** `ub` is close to zero for most of Winter and Spring, and the cause (calm
  conditions or RBR processing problems) is not resolved.
- **Different from the paper.** Four features instead of eight, 897 rows instead of 1648, and
  our own definitions of d50 and sigma2 (the paper does not give formulas), so absolute values
  are not comparable with the paper's.

## Repository contents

| File | Purpose |
|---|---|
| `extract.py` | extract the needed files from `NonVectrinoData.zip` |
| `inspect_data.py` | inspect raw `.mat` files |
| `build_dataset.py` | QC, feature engineering, hourly merge -> `dataset.csv` |
| `rf_model.py` | Random Forest experiments |
| `svr_model.py` | SVR experiments and RF-vs-SVR comparison |
| `predict.py` | live demo |
| `dataset.csv` | cleaned dataset (897 rows) |
| `figures/` | all plots |
| `*_results.txt`, `feature_importance.csv`, `model_comparison.csv` | saved results |

`data/`, `venv/` and the raw zip are git-ignored.