import os, glob
import numpy as np
from datetime import datetime, timedelta
from scipy.io import loadmat

ROOT = os.path.join("data", "NonVectrinoData")
SEASONS = ["Summer", "Winter", "Spring"]
OUT = open("inspect_out.txt", "w", encoding="utf-8")


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    OUT.write(s + "\n")


def load(path):
    """Load a .mat file; fall back to h5py for v7.3 files."""
    try:
        return loadmat(path, squeeze_me=True)
    except NotImplementedError:
        import h5py
        d = {}
        with h5py.File(path, "r") as f:
            for k in f.keys():
                d[k] = np.array(f[k]).squeeze()
        log("   (v7.3 file, read with h5py - arrays may be transposed)")
        return d


def dn(x):
    """MATLAB datenum -> python datetime."""
    x = float(x)
    return datetime.fromordinal(int(x) - 366) + timedelta(days=x % 1)


def describe(d):
    for k, v in d.items():
        if k.startswith("__"):
            continue
        v = np.asarray(v)
        extra = ""
        if v.dtype.kind == "f" and v.size > 0:
            extra = f" nan={np.isnan(v).mean():.1%}"
        log(f"   {k:10s} shape={v.shape} dtype={v.dtype}{extra}")


def time_summary(t, label="time"):
    t = np.asarray(t, dtype=float).ravel()
    t = t[~np.isnan(t)]
    if t.size == 0:
        log(f"   {label}: empty")
        return
    dt_min = np.median(np.diff(t)) * 24 * 60 if t.size > 1 else float("nan")
    log(f"   {label}: n={t.size}  {dn(t.min())} -> {dn(t.max())}  median step={dt_min:.2f} min")


# ---------------- LISST (targets) ----------------
log("=" * 60, "\nLISST (P2)")
for s in SEASONS:
    p = os.path.join(ROOT, "LISST", s, "P2", "lisst.mat")
    log(f"\n[{s}] {p}")
    d = load(p)
    describe(d)
    time_summary(d["time"])
    dias = np.asarray(d["dias"]).ravel()
    psd = np.asarray(d["psd"])
    log(f"   dias: {len(dias)} bins, {dias.min():.2f} -> {dias.max():.2f} um")
    # orient psd as (n_time, n_bins)
    if psd.shape[0] == len(dias) and psd.shape[1] != len(dias):
        psd = psd.T
    # quick sanity d50 from the first non-NaN row
    for row in psd:
        if not np.isnan(row).any() and row.sum() > 0:
            c = np.cumsum(row) / row.sum()
            log(f"   sample d50 (first valid row) ~ {np.interp(0.5, c, dias):.1f} um")
            break
    log(f"   rows with any NaN: {np.isnan(psd).any(axis=1).mean():.1%}")

# ---------------- CTD ----------------
log("\n" + "=" * 60, "\nCTD (P1 for salinity, P2 for temperature)")
for s in SEASONS:
    for plat in ["P1", "P2"]:
        p = os.path.join(ROOT, "CTD", s, plat, "ctd.mat")
        log(f"\n[{s} {plat}] {p}")
        d = load(p)
        describe(d)
        if "time" in d:
            time_summary(d["time"])

# ---------------- RBR (P2) ----------------
log("\n" + "=" * 60, "\nRBR (P2)")
for s in SEASONS:
    files = glob.glob(os.path.join(ROOT, "RBR", s, "P2", "rbr_*.mat"))
    log(f"\n[{s}] {len(files)} burst files")
    if not files:
        continue
    log("   --- variables in first file ---")
    describe(load(files[0]))
    ts, hs, om, dp = [], [], [], []
    bad = 0
    for f in files:
        try:
            d = load(f)
            ts.append(float(d["tstart"]))
            hs.append(float(d["Hsig"]))
            om.append(float(d["omega"]))
            dp.append(float(np.nanmean(d["depth"])))
        except Exception:
            bad += 1
    log(f"   unreadable/missing-field files: {bad}")
    time_summary(np.array(ts), "tstart")
    for name, arr in [("Hsig", hs), ("omega", om), ("depth", dp)]:
        a = np.array(arr)
        log(f"   {name}: min={np.nanmin(a):.3f} median={np.nanmedian(a):.3f} "
            f"max={np.nanmax(a):.3f} nan={np.isnan(a).mean():.1%}")

# ---------------- ADP (P1) ----------------
log("\n" + "=" * 60, "\nADP (P1)")
for s in SEASONS:
    p = os.path.join(ROOT, "ADP", s, "P1", "adp.mat")
    log(f"\n[{s}] {p}")
    d = load(p)
    describe(d)
    time_summary(d["time"])

OUT.close()
print("\nDone. Full output saved to inspect_out.txt")