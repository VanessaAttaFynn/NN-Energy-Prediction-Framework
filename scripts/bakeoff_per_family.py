"""Model bake-off run separately within each benchmark family.

Replaces the pooled shared-core comparison: each family uses its own final
feature set (the same sets as the within-family experiments) and a grouped
train/test split, repeated over several split seeds. Fit time is recorded so
the speed argument for Random Forest is backed by numbers.

Usage:
    python scripts/bakeoff_per_family.py [--seeds 42 0 1 2 3]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_experiments import COMBINED, feature_set, metrics  # noqa: E402

TEST_SIZE = 0.2


def make_model(name: str, seed: int):
    if name == "rf":
        return RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=seed)
    if name == "xgb":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                            n_jobs=-1, random_state=seed, verbosity=0)
    if name == "mlp":
        from src.models.neural_net import TorchMLPRegressor
        return TorchMLPRegressor(hidden_sizes=(128, 64, 32), epochs=200,
                                 random_state=seed)
    if name == "linear":
        return LinearRegression()
    raise ValueError(name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    ap.add_argument("--models", nargs="+", default=["linear", "rf", "xgb", "mlp"])
    ap.add_argument("--out", default="results/tables/bakeoff_per_family.csv")
    args = ap.parse_args()

    df = pd.read_csv(COMBINED)
    fams = {"butter_e": (df[df.family == "MLP"].reset_index(drop=True), "butter_full"),
            "ec_nas": (df[df.family == "CNN"].reset_index(drop=True), "ecnas_full")}

    rows = []
    for fam, (sub, kind) in fams.items():
        feats = feature_set(sub, kind)
        X = sub[feats].to_numpy(float)
        y = sub["target"].to_numpy(float)
        for seed in args.seeds:
            tr, te = next(GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                            random_state=seed).split(sub, groups=sub["group_key"]))
            for name in args.models:
                est = make_model(name, seed)
                t0 = time.perf_counter()
                est.fit(X[tr], np.log1p(y[tr]))
                fit_s = time.perf_counter() - t0
                pred = np.expm1(est.predict(X[te]))
                res = metrics(y[te], pred)
                rows.append({"family": fam, "seed": seed, "model": name,
                             "n_features": len(feats), "n_train": len(tr),
                             "n_test": len(te), "fit_seconds": fit_s, **res})
                print(fam, seed, name, {k: round(v, 4) for k, v in res.items()},
                      f"{fit_s:.1f}s", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    summ = (out.groupby(["family", "model"])
               [["mape", "r2_raw", "tau", "fit_seconds"]]
               .agg(["mean", "std"]).round(4))
    pd.set_option("display.width", 200)
    print(summ.to_string())
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
