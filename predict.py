"""Live demo: predict median particle diameter (d50) and PSD variance (sigma2).

Usage
-----
  python predict.py --sample            # compare predictions with measurements on unseen rows
  python predict.py --sample 12         # ... showing 12 rows
  python predict.py --S 28 --T 20 --ub 0.05 --u 0.2     # predict for given conditions
  python predict.py                     # interactive prompts

The Random Forest (96 trees, chosen via the OOB score in rf_model.py) is trained on
85% of dataset.csv. The remaining 15% (same split as the SVR test set in svr_model.py)
is never used for training, so --sample shows genuine held-out predictions.
"""
import argparse
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

FEATURES = ["S", "T", "ub", "u"]
TARGETS = ["d50", "sigma2"]
LABELS = {"S": "salinity", "T": "temperature", "ub": "near-bed wave orbital velocity",
          "u": "mean current speed"}
UNITS = {"S": "PSU", "T": "deg C", "ub": "m/s", "u": "m/s", "d50": "um", "sigma2": "um^2"}
N_TREES = 96
SEED = 42


def load_and_train(path="dataset.csv"):
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"Could not find {path}. Run build_dataset.py first (see README.md).")
    idx = np.arange(len(data))
    idx_train, idx_test = train_test_split(idx, test_size=0.15, random_state=SEED)
    train, test = data.iloc[idx_train], data.iloc[idx_test]
    models = {}
    for t in TARGETS:
        rf = RandomForestRegressor(n_estimators=N_TREES, random_state=SEED, n_jobs=-1)
        models[t] = rf.fit(train[FEATURES], train[t])
    return models, train, test


def predict_one(models, train, values):
    row = pd.DataFrame([values], columns=FEATURES)
    out = {t: float(models[t].predict(row)[0]) for t in TARGETS}
    warnings = []
    for f in FEATURES:
        lo, hi = train[f].min(), train[f].max()
        if not lo <= values[f] <= hi:
            warnings.append(f"{f}={values[f]} is outside the training range "
                            f"[{lo:.3g}, {hi:.3g}] {UNITS[f]} - treat this prediction with caution")
    return out, warnings


def show_prediction(models, train, values):
    out, warns = predict_one(models, train, values)
    print("\nInputs : " + ", ".join(f"{f}={values[f]:g} {UNITS[f]}" for f in FEATURES))
    print(f"d50    : {out['d50']:.1f} {UNITS['d50']}   (median particle diameter)")
    print(f"sigma2 : {out['sigma2']:.0f} {UNITS['sigma2']}   (PSD variance)")
    for w in warns:
        print("WARNING:", w)


def show_sample(models, test, n):
    sample = test.sample(n=min(n, len(test)), random_state=SEED).sort_values("hour")
    print(f"\nPredictions on {len(sample)} held-out rows (not used for training):\n")
    print(f"{'hour':<20}{'season':<8}{'d50 true':>10}{'d50 pred':>10}"
          f"{'s2 true':>10}{'s2 pred':>10}")
    for _, r in sample.iterrows():
        p, _ = predict_one(models, test, {f: r[f] for f in FEATURES})
        print(f"{str(r['hour']):<20}{r['season']:<8}{r['d50']:>10.1f}{p['d50']:>10.1f}"
              f"{r['sigma2']:>10.0f}{p['sigma2']:>10.0f}")
    print("\nOverall r2 on all held-out rows:")
    for t in TARGETS:
        pred = models[t].predict(test[FEATURES])
        print(f"  {t}: {r2_score(test[t], pred):.3f}   (n = {len(test)})")


def interactive(models, train):
    print("Enter conditions (blank line to quit).")
    ranges = {f: (train[f].min(), train[f].max()) for f in FEATURES}
    while True:
        values = {}
        for f in FEATURES:
            lo, hi = ranges[f]
            while True:
                s = input(f"  {f} - {LABELS[f]} [{UNITS[f]}, training range {lo:.3g} to {hi:.3g}]: ").strip()
                if s == "":
                    return
                try:
                    values[f] = float(s)
                    break
                except ValueError:
                    print("    please enter a number")
        show_prediction(models, train, values)
        print()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", nargs="?", const=8, type=int, metavar="N",
                    help="show N held-out rows with true vs predicted values (default 8)")
    for f in FEATURES:
        ap.add_argument(f"--{f}", type=float, help=f"{LABELS[f]} ({UNITS[f]})")
    ap.add_argument("--data", default="dataset.csv", help="path to dataset.csv")
    args = ap.parse_args()

    models, train, test = load_and_train(args.data)
    print(f"Random Forest trained on {len(train)} rows "
          f"({N_TREES} trees); {len(test)} rows held out.")

    given = {f: getattr(args, f) for f in FEATURES}
    if args.sample is not None:
        show_sample(models, test, args.sample)
    elif all(v is not None for v in given.values()):
        show_prediction(models, train, given)
    elif any(v is not None for v in given.values()):
        sys.exit("Please give all of --S --T --ub --u, or none for interactive mode.")
    else:
        interactive(models, train)


if __name__ == "__main__":
    main()