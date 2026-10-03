"""Step 5: Support Vector Regression + head-to-head comparison with Random Forest.

5a  70/15/15 split (train / CV / test) as in the paper
5b  tune C and epsilon on the CV set (RBF kernel, standardised features and targets)
5c  test r2 on the held-out 15%, plus repeated random splits (re-tuned each time)
5d  same honest checks as the RF: chronological split and leave-one-season-out
5e  RF vs SVR comparison table + figure
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

FEATURES = ["S", "T", "ub", "u"]
TARGETS = ["d50", "sigma2"]
SEED = 42
RF_TREES = 96                 # chosen in Step 4
N_REPEATS = 20                # repeated random 70/15/15 splits
C_GRID = 2.0 ** np.arange(-4, 12)                  # 0.0625 ... 2048 (paper used 2048)
EPS_GRID = np.logspace(-3, 0, 13)                  # in standardised-target units
C_COARSE = 2.0 ** np.arange(-2, 12, 2)
EPS_COARSE = np.logspace(-3, 0, 7)

OUT = open("svr_results.txt", "w", encoding="utf-8")
os.makedirs("figures", exist_ok=True)


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")


def metrics(y, p):
    return dict(r2=r2_score(y, p),
                rmse=float(np.sqrt(mean_squared_error(y, p))),
                mae=float(mean_absolute_error(y, p)))


def svr_fit_predict(Xtr, ytr, Xte, C, eps):
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(ytr.reshape(-1, 1))
    m = SVR(kernel="rbf", C=C, epsilon=eps, gamma="scale")
    m.fit(xs.transform(Xtr), ys.transform(ytr.reshape(-1, 1)).ravel())
    p = m.predict(xs.transform(Xte)).reshape(-1, 1)
    return ys.inverse_transform(p).ravel()


def tune(Xtr, ytr, Xcv, ycv, Cs, epss):
    """Grid search on the CV set. Returns r2 grid (len(Cs), len(epss)), best C, best eps."""
    xs = StandardScaler().fit(Xtr)
    ys = StandardScaler().fit(ytr.reshape(-1, 1))
    A, b = xs.transform(Xtr), ys.transform(ytr.reshape(-1, 1)).ravel()
    Acv, bcv = xs.transform(Xcv), ys.transform(ycv.reshape(-1, 1)).ravel()
    grid = np.zeros((len(Cs), len(epss)))
    for i, C in enumerate(Cs):
        for j, e in enumerate(epss):
            m = SVR(kernel="rbf", C=C, epsilon=e, gamma="scale").fit(A, b)
            grid[i, j] = r2_score(bcv, m.predict(Acv))   # r2 is unchanged by the scaling
    i, j = np.unravel_index(np.argmax(grid), grid.shape)
    return grid, float(Cs[i]), float(epss[j])


def rf_fit_predict(Xtr, ytr, Xte, seed=SEED):
    rf = RandomForestRegressor(n_estimators=RF_TREES, random_state=seed, n_jobs=-1)
    return rf.fit(Xtr, ytr).predict(Xte)


data = pd.read_csv("dataset.csv", parse_dates=["hour"])
X = data[FEATURES].values
idx = np.arange(len(data))
log(f"Loaded dataset.csv: {len(data)} rows; features {FEATURES}; targets {TARGETS}")

# ------------------------------------------------------------------ 5a / 5b
idx_trcv, idx_te = train_test_split(idx, test_size=0.15, random_state=SEED)
idx_tr, idx_cv = train_test_split(idx_trcv, test_size=0.15 / 0.85, random_state=SEED)
log("\n" + "=" * 64)
log(f"5a. Split: train {len(idx_tr)} / CV {len(idx_cv)} / test {len(idx_te)} rows (70/15/15)")
log("\n5b. Tuning C and epsilon on the CV set (RBF kernel; epsilon in standardised units)")

best = {}
grids = {}
for t in TARGETS:
    y = data[t].values
    grid, C, e = tune(X[idx_tr], y[idx_tr], X[idx_cv], y[idx_cv], C_GRID, EPS_GRID)
    best[t] = (C, e)
    grids[t] = grid
    log(f"  {t}: best C = {C:g}, epsilon = {e:.4f}   (CV r2 = {grid.max():.3f})")

fig, ax = plt.subplots(1, 2, figsize=(10, 4))
for t in TARGETS:
    C, e = best[t]
    ax[0].semilogx(C_GRID, grids[t][:, list(EPS_GRID).index(e)], label=t)
    ax[1].semilogx(EPS_GRID, grids[t][list(C_GRID).index(C), :], label=t)
ax[0].set_xlabel("C")
ax[0].set_ylabel("CV r2")
ax[0].set_title("(a) r2 vs C (best epsilon)")
ax[1].set_xlabel("epsilon (standardised)")
ax[1].set_ylabel("CV r2")
ax[1].set_title("(b) r2 vs epsilon (best C)")
ax[0].legend()
plt.tight_layout()
plt.savefig("figures/svr_tuning.png", dpi=120)
plt.close()

# ------------------------------------------------------------------ 5c
log("\n" + "=" * 64 + "\n5c. Test-set results (15% test set, model trained on the 70% training set)")
primary = {}
res = {}   # res[(scheme, model, target)] = r2
for t in TARGETS:
    y = data[t].values
    C, e = best[t]
    p_svr = svr_fit_predict(X[idx_tr], y[idx_tr], X[idx_te], C, e)
    p_rf = rf_fit_predict(X[idx_tr], y[idx_tr], X[idx_te])
    primary[t] = (y[idx_te], p_svr)
    ms, mr = metrics(y[idx_te], p_svr), metrics(y[idx_te], p_rf)
    res[("Random split", "SVR", t)] = ms["r2"]
    res[("Random split", "RF", t)] = mr["r2"]
    log(f"  {t}: SVR r2={ms['r2']:.3f} RMSE={ms['rmse']:.2f} MAE={ms['mae']:.2f}   |   "
        f"RF r2={mr['r2']:.3f} RMSE={mr['rmse']:.2f} MAE={mr['mae']:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(10, 4.5))
for a, t in zip(ax, TARGETS):
    yt, yp = primary[t]
    a.scatter(yt, yp, s=10, alpha=0.6)
    lim = [min(yt.min(), yp.min()), max(yt.max(), yp.max())]
    a.plot(lim, lim, "r-")
    a.set_xlabel(f"True {t}")
    a.set_ylabel(f"Predicted {t}")
    a.set_title(f"SVR test set: r2 = {r2_score(yt, yp):.2f}")
plt.tight_layout()
plt.savefig("figures/svr_random_split.png", dpi=120)
plt.close()

log(f"\n  Repeated random 70/15/15 splits (SVR re-tuned on a coarse grid each time), n={N_REPEATS}")
rep = {(m, t): [] for m in ["SVR", "RF"] for t in TARGETS}
for r in range(N_REPEATS):
    a, te = train_test_split(idx, test_size=0.15, random_state=1000 + r)
    tr, cv = train_test_split(a, test_size=0.15 / 0.85, random_state=1000 + r)
    for t in TARGETS:
        y = data[t].values
        _, C, e = tune(X[tr], y[tr], X[cv], y[cv], C_COARSE, EPS_COARSE)
        rep[("SVR", t)].append(r2_score(y[te], svr_fit_predict(X[tr], y[tr], X[te], C, e)))
        rep[("RF", t)].append(r2_score(y[te], rf_fit_predict(X[tr], y[tr], X[te], seed=r)))
for t in TARGETS:
    for m in ["SVR", "RF"]:
        v = rep[(m, t)]
        res[("Random split (mean of repeats)", m, t)] = float(np.mean(v))
        log(f"  {t:6s} {m:3s}: r2 = {np.mean(v):.3f} +/- {np.std(v):.3f}")

# ------------------------------------------------------------------ 5d
log("\n" + "=" * 64 + "\n5d. Honest checks (hyperparameters from 5b; same splits as the RF in Step 4)")
log("\nChronological split within each season (first 70% of time = train, last 30% = test)")
tr_i, te_i = [], []
for s, g in data.groupby("season"):
    g = g.sort_values("hour")
    cut = int(len(g) * 0.7)
    tr_i += list(g.index[:cut])
    te_i += list(g.index[cut:])
for t in TARGETS:
    y = data[t].values
    C, e = best[t]
    ms = metrics(y[te_i], svr_fit_predict(X[tr_i], y[tr_i], X[te_i], C, e))
    mr = metrics(y[te_i], rf_fit_predict(X[tr_i], y[tr_i], X[te_i]))
    res[("Chronological", "SVR", t)] = ms["r2"]
    res[("Chronological", "RF", t)] = mr["r2"]
    log(f"  {t:6s}: SVR r2={ms['r2']:.3f} RMSE={ms['rmse']:.2f}   |   RF r2={mr['r2']:.3f} RMSE={mr['rmse']:.2f}")

log("\nLeave-one-season-out (train on two seasons, test on the third)")
fig, ax = plt.subplots(2, 3, figsize=(13, 7))
seasons = ["Summer", "Winter", "Spring"]
loso = {(m, t): [] for m in ["SVR", "RF"] for t in TARGETS}
for j, s in enumerate(seasons):
    te = (data["season"] == s).values
    for i, t in enumerate(TARGETS):
        y = data[t].values
        C, e = best[t]
        p_svr = svr_fit_predict(X[~te], y[~te], X[te], C, e)
        p_rf = rf_fit_predict(X[~te], y[~te], X[te])
        r_s, r_r = r2_score(y[te], p_svr), r2_score(y[te], p_rf)
        loso[("SVR", t)].append(r_s)
        loso[("RF", t)].append(r_r)
        log(f"  held out {s:6s} {t:6s}: SVR r2={r_s:7.3f}   |   RF r2={r_r:7.3f}")
        a = ax[i, j]
        a.scatter(y[te], p_svr, s=8, alpha=0.6)
        lim = [min(y[te].min(), p_svr.min()), max(y[te].max(), p_svr.max())]
        a.plot(lim, lim, "r-")
        a.set_title(f"SVR held out {s}: {t}, r2={r_s:.2f}")
        a.set_xlabel("true")
        a.set_ylabel("predicted")
plt.tight_layout()
plt.savefig("figures/svr_leave_one_season_out.png", dpi=120)
plt.close()
for t in TARGETS:
    for m in ["SVR", "RF"]:
        res[("Leave-one-season-out (mean)", m, t)] = float(np.mean(loso[(m, t)]))

# ------------------------------------------------------------------ 5e
log("\n" + "=" * 64 + "\n5e. Comparison table (test r2)")
schemes = ["Random split", "Random split (mean of repeats)", "Chronological",
           "Leave-one-season-out (mean)"]
rows = []
for s in schemes:
    rows.append({"evaluation": s,
                 "RF d50": res[(s, "RF", "d50")], "SVR d50": res[(s, "SVR", "d50")],
                 "RF sigma2": res[(s, "RF", "sigma2")], "SVR sigma2": res[(s, "SVR", "sigma2")]})
comp = pd.DataFrame(rows).set_index("evaluation")
log(comp.round(3).to_string())
comp.to_csv("model_comparison.csv")

fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
labels = ["Random\nsplit", "Random\n(mean of 20)", "Chrono-\nlogical", "Leave-one-\nseason-out"]
x = np.arange(len(schemes))
for a, t in zip(ax, TARGETS):
    a.bar(x - 0.2, [res[(s, "RF", t)] for s in schemes], 0.4, label="RF")
    a.bar(x + 0.2, [res[(s, "SVR", t)] for s in schemes], 0.4, label="SVR")
    a.axhline(0, color="k", lw=0.8)
    a.set_xticks(x)
    a.set_xticklabels(labels)
    a.set_title(f"Test r2 for {t}")
    a.set_ylim(-3, 1)
    a.legend()
plt.tight_layout()
plt.savefig("figures/svr_vs_rf.png", dpi=120)
plt.close()

OUT.close()
print("\nDone. Saved svr_results.txt, model_comparison.csv and figures/svr_*.png")