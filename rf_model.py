"""Step 4: Random Forest regression for d50 and sigma2.

4a  feature importance (1000 runs, 10 trees, 70/30 split - as in the paper)
4b  tune number of trees with the out-of-bag (OOB) score
4c  test r2 with the paper's random 70/30 split
4d  honest checks: chronological split and leave-one-season-out
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=UserWarning)

FEATURES = ["S", "T", "ub", "u"]
TARGETS = ["d50", "sigma2"]
SEED = 42
N_IMP_RUNS = 1000     # paper: 1000 runs
IMP_TREES = 10        # paper: 10 estimators
TEST_SIZE = 0.30
N_REPEATS = 30        # repeated random splits for mean +/- std of test r2
OUT = open("rf_results.txt", "w", encoding="utf-8")
os.makedirs("figures", exist_ok=True)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")


def metrics(y, p):
    return dict(r2=r2_score(y, p),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=float(mean_absolute_error(y, p)))


def fit_predict(Xtr, ytr, Xte, n_trees, seed=SEED):
    rf = RandomForestRegressor(n_estimators=n_trees, random_state=seed, n_jobs=-1)
    rf.fit(Xtr, ytr)
    return rf.predict(Xte)


data = pd.read_csv("dataset.csv", parse_dates=["hour"])
X = data[FEATURES].values
log(f"Loaded dataset.csv: {len(data)} rows; features {FEATURES}; targets {TARGETS}")

# ------------------------------------------------------------------ 4a
log("\n" + "=" * 64 + f"\n4a. Feature importance ({N_IMP_RUNS} runs, {IMP_TREES} trees, 70/30)")
imp = {t: np.zeros(len(FEATURES)) for t in TARGETS}
for run in range(N_IMP_RUNS):
    for t in TARGETS:
        Xtr, Xte, ytr, yte = train_test_split(X, data[t].values, test_size=TEST_SIZE,
                                              random_state=run)
        rf = RandomForestRegressor(n_estimators=IMP_TREES, random_state=run, n_jobs=1)
        rf.fit(Xtr, ytr)
        imp[t] += rf.feature_importances_
imp_df = pd.DataFrame({t: imp[t] / N_IMP_RUNS for t in TARGETS}, index=FEATURES)
imp_df = imp_df.sort_values("d50", ascending=False)
imp_df.columns = ["FI_d50", "FI_sigma2"]
log(imp_df.round(3).to_string())
imp_df.to_csv("feature_importance.csv")

ax = imp_df.plot.bar(figsize=(6, 4))
ax.set_ylabel("mean feature importance")
ax.set_title("Random Forest feature importance")
plt.tight_layout()
plt.savefig("figures/rf_importance.png", dpi=120)
plt.close()

# ------------------------------------------------------------------ 4b
log("\n" + "=" * 64 + "\n4b. Choosing the number of trees (OOB score, 70% training set)")
Xtr_all, Xte_all, idx_tr, idx_te = train_test_split(
    X, np.arange(len(data)), test_size=TEST_SIZE, random_state=SEED)
N_MIN, N_MAX = 15, 150
n_grid = np.arange(N_MIN, N_MAX + 1)
oob = {}
for t in TARGETS:
    y_tr = data[t].values[idx_tr]
    rf = RandomForestRegressor(n_estimators=N_MIN, warm_start=True, oob_score=True,
                               random_state=SEED, n_jobs=-1)
    sc = []
    for n in n_grid:
        rf.set_params(n_estimators=int(n))
        rf.fit(Xtr_all, y_tr)
        sc.append(rf.oob_score_)
    oob[t] = np.array(sc)

best_n = []
for t in TARGETS:
    s = oob[t]
    plateau = s[-30:].mean()
    smooth = pd.Series(s).rolling(5, min_periods=1).mean().values
    n_opt = int(n_grid[np.argmax(smooth >= plateau - 0.005)])
    best_n.append(n_opt)
    log(f"  {t}: plateau OOB r2 = {plateau:.3f}; first n within 0.005 of plateau = {n_opt}")
N_TREES = int(max(best_n))
log(f"  -> using n_estimators = {N_TREES} for both targets")

plt.figure(figsize=(6, 4))
for t in TARGETS:
    plt.plot(n_grid, oob[t], label=t)
plt.axvline(N_TREES, color="grey", ls=":")
plt.xlabel("number of trees")
plt.ylabel("OOB score (r2)")
plt.legend()
plt.tight_layout()
plt.savefig("figures/rf_oob.png", dpi=120)
plt.close()

# ------------------------------------------------------------------ 4c
log("\n" + "=" * 64 + f"\n4c. Random 70/30 split (as in the paper), {N_TREES} trees")
preds_primary = {}
for t in TARGETS:
    y = data[t].values
    p = fit_predict(Xtr_all, y[idx_tr], Xte_all, N_TREES)
    m = metrics(y[idx_te], p)
    preds_primary[t] = (y[idx_te], p)
    log(f"  {t}: primary split (seed {SEED}) r2={m['r2']:.3f}  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}")
    r2s = []
    for r in range(N_REPEATS):
        a, b, c, d = train_test_split(X, y, test_size=TEST_SIZE, random_state=100 + r)
        r2s.append(r2_score(d, fit_predict(a, c, b, N_TREES, seed=r)))
    log(f"       over {N_REPEATS} random splits: r2 = {np.mean(r2s):.3f} +/- {np.std(r2s):.3f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
for a, t in zip(ax, TARGETS):
    yt, yp = preds_primary[t]
    a.scatter(yt, yp, s=8, alpha=0.6)
    lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    a.plot(lim, lim, "r-")
    a.set_xlabel(f"True {t}")
    a.set_ylabel(f"Predicted {t}")
    a.set_title(f"RF random split: r2 = {r2_score(yt, yp):.2f}")
plt.tight_layout()
plt.savefig("figures/rf_random_split.png", dpi=120)
plt.close()

# ------------------------------------------------------------------ 4d
log("\n" + "=" * 64 + "\n4d. Honest checks")

log("\nHow much variance does season alone explain? (season-mean predictor, in-sample)")
for t in TARGETS:
    pm = data.groupby("season")[t].transform("mean")
    log(f"  {t}: r2 = {r2_score(data[t], pm):.3f}")

log("\nChronological split within each season (first 70% of time = train, last 30% = test)")
tr_idx, te_idx = [], []
for s, g in data.groupby("season"):
    g = g.sort_values("hour")
    cut = int(len(g) * (1 - TEST_SIZE))
    tr_idx += list(g.index[:cut])
    te_idx += list(g.index[cut:])
for t in TARGETS:
    p = fit_predict(data.loc[tr_idx, FEATURES].values, data.loc[tr_idx, t].values,
                    data.loc[te_idx, FEATURES].values, N_TREES)
    m = metrics(data.loc[te_idx, t].values, p)
    log(f"  {t}: r2={m['r2']:.3f}  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}")

log("\nLeave-one-season-out (train on two seasons, test on the third)")
seasons = ["Summer", "Winter", "Spring"]
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
for j, s in enumerate(seasons):
    te = data["season"] == s
    for i, t in enumerate(TARGETS):
        p = fit_predict(X[~te], data.loc[~te, t].values, X[te], N_TREES)
        yt = data.loc[te, t].values
        m = metrics(yt, p)
        log(f"  held out {s:6s} {t:6s}: r2={m['r2']:7.3f}  RMSE={m['rmse']:.2f}  MAE={m['mae']:.2f}")
        a = ax[i, j]
        a.scatter(yt, p, s=8, alpha=0.6)
        lim = [min(yt.min(), p.min()), max(yt.max(), p.max())]
        a.plot(lim, lim, "r-")
        a.set_title(f"held out {s}: {t}, r2={m['r2']:.2f}")
        a.set_xlabel("true")
        a.set_ylabel("predicted")
plt.tight_layout()
plt.savefig("figures/rf_leave_one_season_out.png", dpi=120)
plt.close()

OUT.close()
print("\nDone. Saved rf_results.txt, feature_importance.csv and figures/rf_*.png")