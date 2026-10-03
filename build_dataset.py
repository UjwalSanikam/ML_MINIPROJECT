"""Step 3: build a cleaned hourly dataset (dataset.csv) from the raw .mat files.

Targets  : d50, sigma2   (from LISST at P2)
Features : S (CTD P1), T (CTD P2), ub (RBR P2), u (ADP P1)
"""
import os
import glob
import warnings
import numpy as np
import pandas as pd
from scipy.io import loadmat

ROOT = os.path.join("data", "NonVectrinoData")
RBR_BURST_MIN = {"Summer": 12, "Winter": 14, "Spring": 14}  # README: burst length

# Instrument cleaning dates from the README; data just before them may be biofouled
CLEANINGS = ["2018-07-25", "2018-08-01", "2019-01-24", "2019-04-29", "2019-05-07"]
BIOFOUL_DAYS = 2           # drop LISST data this many days before each cleaning

# QC thresholds (assumptions: documented in the write-up)
TAU_MIN, TAU_MAX = 0.30, 0.98   # LISST optical transmission rule of thumb
D50_MIN = 2.0                   # um; d50 stuck in the lowest bins is not trusted
MIN_LISST_PER_HOUR = 20         # valid 1-min LISST samples needed per hour
MIN_RBR_PER_HOUR = 2            # valid RBR bursts needed per hour
FMIN, FMAX, KH_MAX = 0.05, 1.0, 3.0   # wave band used for ub
G = 9.81


# ---------------------------------------------------------------- helpers
def to_hour(datenum):
    """MATLAB datenum -> pandas hour timestamps (rounded to the second first)."""
    t = pd.to_datetime(np.asarray(datenum, dtype=float) - 719529, unit="D")
    return pd.Series(t).dt.round("s").dt.floor("h")


def psd_stats(psd, dias):
    """Volume-weighted median diameter d50 (um) and variance sigma2 (um^2).

    psd  : (n_samples, n_bins) volume concentration per size bin
    dias : (n_bins,) bin median diameters
    """
    psd = np.asarray(psd, dtype=float)
    dias = np.asarray(dias, dtype=float).ravel()
    total = psd.sum(axis=1)
    bad = np.isnan(psd).any(axis=1) | ~(total > 0)
    safe_total = np.where(bad, 1.0, total)
    w = np.where(bad[:, None], 0.0, np.nan_to_num(psd)) / safe_total[:, None]

    # bin edges = geometric midpoints between bin medians
    ld = np.log(dias)
    mid = 0.5 * (ld[1:] + ld[:-1])
    edges = np.concatenate([[2 * ld[0] - mid[0]], mid, [2 * ld[-1] - mid[-1]]])  # ln(d)

    # median: interpolate the cumulative curve (defined at upper bin edges) in ln(d)
    cum = np.cumsum(w, axis=1)
    idx = (cum < 0.5).sum(axis=1).clip(0, len(dias) - 1)
    rows = np.arange(len(w))
    c_hi = cum[rows, idx]
    c_lo = np.where(idx > 0, cum[rows, np.maximum(idx - 1, 0)], 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(c_hi > c_lo, (0.5 - c_lo) / (c_hi - c_lo), 1.0).clip(0, 1)
    d50 = np.exp(edges[idx] + frac * (edges[idx + 1] - edges[idx]))

    mean = (w * dias).sum(axis=1)
    sigma2 = (w * (dias - mean[:, None]) ** 2).sum(axis=1)
    d50 = np.where(bad, np.nan, d50)
    sigma2 = np.where(bad, np.nan, sigma2)
    return d50, sigma2


def trapz(y, x):
    return float(np.sum(0.5 * (y[1:] + y[:-1]) * np.diff(x)))


def wavenumber(omega, h):
    """Linear-wave dispersion relation  omega^2 = g k tanh(k h)  (Newton)."""
    k = omega ** 2 / G
    for _ in range(60):
        th = np.tanh(k * h)
        f = G * k * th - omega ** 2
        df = G * th + G * k * h / np.cosh(k * h) ** 2
        k = k - f / df
    return k


def orbital_velocity(f, sse, depth):
    """Near-bed orbital velocity amplitude ub = sqrt(2) * u_rms (m/s).

    u_rms^2 = integral SSE(f) * (omega / sinh(kh))^2 df  (linear wave theory),
    restricted to a band where the depth-corrected spectrum is trustworthy.
    """
    f = np.asarray(f, dtype=float).ravel()
    sse = np.asarray(sse, dtype=float).ravel()
    if not np.isfinite(depth) or depth <= 0.2:
        return np.nan
    omega = 2 * np.pi * f
    ok = np.isfinite(f) & np.isfinite(sse) & (f >= FMIN) & (f <= FMAX)
    if ok.sum() < 5:
        return np.nan
    k = wavenumber(omega[ok], depth)
    keep = k * depth <= KH_MAX
    if keep.sum() < 5:
        return np.nan
    fb, sb, ob, kb = f[ok][keep], sse[ok][keep], omega[ok][keep], k[keep]
    u2 = trapz(sb * (ob / np.sinh(kb * depth)) ** 2, fb)
    return float(np.sqrt(2.0 * u2)) if u2 >= 0 else np.nan


# ---------------------------------------------------------------- loaders
def lisst_hourly(season):
    d = loadmat(os.path.join(ROOT, "LISST", season, "P2", "lisst.mat"), squeeze_me=True)
    hour = to_hour(d["time"])
    psd = np.asarray(d["psd"], dtype=float)
    dias = np.asarray(d["dias"], dtype=float).ravel()
    if psd.shape[0] != len(hour):
        psd = psd.T
    tau = np.asarray(d["tau"], dtype=float)
    d50, s2 = psd_stats(psd, dias)

    n0 = len(hour)
    ok = np.isfinite(d50) & np.isfinite(s2)
    n_valid = ok.sum()
    ok &= (tau > TAU_MIN) & (tau < TAU_MAX)
    n_tau = ok.sum()
    ok &= d50 >= D50_MIN
    n_d50 = ok.sum()

    t_real = pd.to_datetime(np.asarray(d["time"], dtype=float) - 719529, unit="D")
    foul = np.zeros(n0, dtype=bool)
    for c in CLEANINGS:
        c = pd.Timestamp(c) + pd.Timedelta(days=1)  # through end of cleaning day
        foul |= (t_real >= c - pd.Timedelta(days=BIOFOUL_DAYS + 1)) & (t_real <= c)
    ok &= ~foul
    n_foul = ok.sum()

    print(f"  [{season}] LISST samples: {n0} -> valid psd {n_valid} -> tau OK {n_tau} "
          f"-> d50>={D50_MIN} {n_d50} -> not biofouled {n_foul}")

    df = pd.DataFrame({"hour": hour[ok].values, "d50": d50[ok], "sigma2": s2[ok]})
    g = df.groupby("hour").agg(d50=("d50", "mean"), sigma2=("sigma2", "mean"),
                               n_lisst=("d50", "size"))
    return g[g.n_lisst >= MIN_LISST_PER_HOUR]


def ctd_hourly(season, plat, var, lo, hi):
    d = loadmat(os.path.join(ROOT, "CTD", season, plat, "ctd.mat"), squeeze_me=True)
    hour = to_hour(d["time"])
    x = np.asarray(d[var], dtype=float)
    x = np.where((x > lo) & (x < hi), x, np.nan)
    s = pd.Series(x, index=hour.values)
    return s.groupby(level=0).mean().dropna()


def adp_hourly(season):
    d = loadmat(os.path.join(ROOT, "ADP", season, "P1", "adp.mat"), squeeze_me=True)
    hour = to_hour(d["time"])
    ve = np.asarray(d["veleast"], dtype=float)
    vn = np.asarray(d["velnorth"], dtype=float)
    if ve.shape[1] != len(hour):
        ve, vn = ve.T, vn.T
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        speed = np.nanmean(np.sqrt(ve ** 2 + vn ** 2), axis=0)   # depth-averaged
    speed = np.where(speed < 3.0, speed, np.nan)
    s = pd.Series(speed, index=hour.values)
    return s.groupby(level=0).mean().dropna()


def rbr_hourly(season):
    files = glob.glob(os.path.join(ROOT, "RBR", season, "P2", "rbr_*.mat"))
    half = RBR_BURST_MIN[season] / 2 / 1440.0
    tmid, ub = [], []
    for fp in files:
        try:
            d = loadmat(fp, squeeze_me=True,
                        variable_names=["tstart", "depth", "SSE", "f"])
            depth = float(np.nanmean(d["depth"]))
            ub.append(orbital_velocity(d["f"], d["SSE"], depth))
            tmid.append(float(d["tstart"]) + half)
        except Exception:
            continue
    hour = to_hour(tmid)
    df = pd.DataFrame({"hour": hour.values, "ub": ub}).dropna()
    g = df.groupby("hour").agg(ub=("ub", "mean"), n_rbr=("ub", "size"))
    print(f"  [{season}] RBR bursts: {len(files)} files -> {len(df)} valid ub "
          f"-> {int((g.n_rbr >= MIN_RBR_PER_HOUR).sum())} hours")
    return g[g.n_rbr >= MIN_RBR_PER_HOUR]


# ---------------------------------------------------------------- main
def main():
    parts = []
    for season in ["Summer", "Winter", "Spring"]:
        print(f"\n== {season} ==")
        L = lisst_hourly(season)
        S = ctd_hourly(season, "P1", "sal", 0.0, 40.0).rename("S")
        T = ctd_hourly(season, "P2", "temp", 0.0, 35.0).rename("T")
        U = adp_hourly(season).rename("u")
        W = rbr_hourly(season)
        df = L.join([S, T, U, W[["ub"]]], how="inner").dropna()
        print(f"  hours: LISST {len(L)}, S {len(S)}, T {len(T)}, u {len(U)}, "
              f"ub {len(W)}  ->  merged {len(df)}")
        df["season"] = season
        parts.append(df)

    data = pd.concat(parts).reset_index().rename(columns={"index": "hour"})
    cols = ["hour", "season", "S", "T", "ub", "u", "d50", "sigma2"]
    data = data[cols]
    data.to_csv("dataset.csv", index=False)

    print("\n== FINAL DATASET ==")
    print("rows:", len(data))
    print(data.groupby("season").size().to_string())
    print(data[["S", "T", "ub", "u", "d50", "sigma2"]].describe().round(3).to_string())
    print("\ncorrelation with targets:")
    print(data[["S", "T", "ub", "u", "d50", "sigma2"]].corr().loc[
        ["S", "T", "ub", "u"], ["d50", "sigma2"]].round(3).to_string())

    # ---- plots
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs("figures", exist_ok=True)
    num = ["S", "T", "ub", "u", "d50", "sigma2"]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    for a, c in zip(ax.ravel(), num):
        a.hist(data[c], bins=40)
        a.set_title(c)
    fig.tight_layout()
    fig.savefig("figures/histograms.png", dpi=120)

    fig, ax = plt.subplots(2, 4, figsize=(14, 6))
    for j, feat in enumerate(["S", "T", "ub", "u"]):
        for i, tgt in enumerate(["d50", "sigma2"]):
            ax[i, j].scatter(data[feat], data[tgt], s=4, alpha=0.5)
            ax[i, j].set_xlabel(feat)
            ax[i, j].set_ylabel(tgt)
    fig.tight_layout()
    fig.savefig("figures/scatter_features_targets.png", dpi=120)

    fig, ax = plt.subplots(3, 1, figsize=(12, 7))
    for a, (s, g) in zip(ax, data.groupby("season")):
        a.plot(pd.to_datetime(g["hour"]), g["d50"], ".", ms=3, label="d50")
        a.set_title(s)
        a.set_ylabel("d50 (um)")
    fig.tight_layout()
    fig.savefig("figures/d50_timeseries.png", dpi=120)
    print("\nSaved dataset.csv and figures/*.png")


if __name__ == "__main__":
    main()