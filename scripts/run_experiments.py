"""Run the experiment matrix on the rebuilt feature tables.

Splits are group-wise by default: every row carries a `group_key` (a repeated
BUTTER-E configuration, or an EC-NAS architecture hash), and a split never
places two rows of the same group on opposite sides. Row-level splits are also
reported so the effect of the change is visible side by side.

Models: Random Forest on a log1p-transformed target, matching the model
selection recorded in the thesis (Section 4.2). Metrics are reported on the raw
scale after expm1 (MAPE, R2) and on the log scale (R2), with Kendall-Tau on the
raw predictions.

Usage:
    python scripts/run_experiments.py [--out results/tables/experiments.csv]
"""
from __future__ import annotations

import argparse
import json

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

# Allow `python scripts/run_experiments.py` to import the project's src package.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COMBINED = "data/processed/combined/combined_features.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.2

CORE = ["params", "depth", "epochs", "batch_size"]
BUTTER_AUX = ["is_gpu", "n_observations", "n_features", "n_classes",
              "task_encoded", "imbalance"]
ECNAS_AUX = ["n_edges", "n_conv3x3", "n_conv1x1", "n_maxpool"]


def feature_set(df: pd.DataFrame, kind: str) -> list[str]:
    shape_cols = [c for c in df.columns if c.startswith("shape_")]
    if kind == "core":
        return CORE
    if kind == "butter_full":
        return CORE + BUTTER_AUX + shape_cols
    if kind == "ecnas_full":
        return CORE + ECNAS_AUX
    if kind == "pooled_full":
        return CORE + BUTTER_AUX + ECNAS_AUX + shape_cols + ["is_mlp_family"]
    raise ValueError(kind)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 1e-9, None)

    mape = float(np.mean(np.abs(y_pred - y_true) / y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2_raw = 1.0 - ss_res / ss_tot if ss_tot else float("nan")

    lt, lp = np.log1p(y_true), np.log1p(y_pred)
    ss_res_l = float(np.sum((lt - lp) ** 2))
    ss_tot_l = float(np.sum((lt - lt.mean()) ** 2))
    r2_log = 1.0 - ss_res_l / ss_tot_l if ss_tot_l else float("nan")

    tau = float(stats.kendalltau(y_true, y_pred).statistic)
    return {"mape": mape, "r2_raw": r2_raw, "r2_log": r2_log, "tau": tau}


def fit_eval(train: pd.DataFrame, test: pd.DataFrame, feats: list[str],
             target: str = "target", model: str = "rf",
             log_target: bool = True) -> dict:
    x_tr = train[feats].to_numpy(dtype=float)
    x_te = test[feats].to_numpy(dtype=float)
    y_tr = train[target].to_numpy(dtype=float)
    y_te = test[target].to_numpy(dtype=float)

    if model == "rf":
        est = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                    random_state=RANDOM_STATE)
    elif model == "xgb":
        from xgboost import XGBRegressor
        est = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.1,
                           n_jobs=-1, random_state=RANDOM_STATE, verbosity=0)
    elif model == "mlp":
        from importlib import import_module
        TorchMLPRegressor = import_module(
            "src.models.neural_net").TorchMLPRegressor
        est = TorchMLPRegressor(hidden_sizes=(128, 64, 32), epochs=200,
                                random_state=RANDOM_STATE)
    else:
        est = LinearRegression()

    if log_target:
        est.fit(x_tr, np.log1p(y_tr))
        pred = np.expm1(est.predict(x_te))
    else:
        est.fit(x_tr, y_tr)
        pred = est.predict(x_te)
    return metrics(y_te, pred)


def split_rows(df: pd.DataFrame, group_wise: bool):
    if group_wise:
        splitter = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                                     random_state=RANDOM_STATE)
        return next(splitter.split(df, groups=df["group_key"]))
    splitter = ShuffleSplit(n_splits=1, test_size=TEST_SIZE,
                            random_state=RANDOM_STATE)
    return next(splitter.split(df))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/tables/experiments.csv")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore any existing results file and recompute all")
    args = ap.parse_args()

    df = pd.read_csv(COMBINED)
    butter = df[df.family == "MLP"].copy()
    ecnas = df[df.family == "CNN"].copy()

    records = []

    # Reuse results from a previous run when the out file already exists, so
    # adding an experiment does not mean recomputing the slow ones (the MLP
    # baseline takes minutes per fit). Pass --fresh to ignore the cache.
    cache: dict[tuple, dict] = {}
    if os.path.exists(args.out) and not args.fresh:
        prev = pd.read_csv(args.out)
        for _, row in prev.iterrows():
            cache[(row["experiment"], row["split"], row["model"])] = row.to_dict()

    def add(name, res, split, feats, model, n_train, n_test):
        key = (name, split, model)
        if key in cache:
            hit = cache.pop(key)
            records.append({**hit})
            return
        records.append({"experiment": name, "split": split,
                        "n_features": len(feats), "model": model,
                        "n_train": n_train, "n_test": n_test, **res})

    # ---- within-family, both split protocols -----------------------------
    for label, subset, kind in (("butter_e", butter, "butter_full"),
                                ("ec_nas", ecnas, "ecnas_full")):
        feats = feature_set(subset, kind)
        for group_wise, split_name in ((True, "group"), (False, "row")):
            tr, te = split_rows(subset, group_wise)
            res = fit_eval(subset.iloc[tr], subset.iloc[te], feats)
            add(f"within:{label}", res, split_name, feats, "rf", len(tr), len(te))

    # ---- within-family, core features only (isolates the auxiliary gain) --
    for label, subset in (("butter_e", butter), ("ec_nas", ecnas)):
        feats = CORE
        tr, te = split_rows(subset, True)
        res = fit_eval(subset.iloc[tr], subset.iloc[te], feats)
        add(f"within_core:{label}", res, "group", feats, "rf", len(tr), len(te))

    # ---- BUTTER-E feature progression ------------------------------------
    # The successive feature additions reported for BUTTER-E: the shared core
    # set, then the hardware/shape auxiliary branch, then the dataset
    # properties that were the actual source of the residual variance.
    shape_cols_b = [c for c in butter.columns if c.startswith("shape_")]
    levels = [
        ("core", CORE),
        ("core_plus_hardware_shape", CORE + ["is_gpu"] + shape_cols_b),
        ("core_plus_hardware_shape_dataset",
         feature_set(butter, "butter_full")),
    ]
    tr, te = split_rows(butter, True)
    for label, f in levels:
        res = fit_eval(butter.iloc[tr], butter.iloc[te], f)
        add("butter_e_feature_progression", res, f"group:{label}", f, "rf",
            len(tr), len(te))

    # ---- pooled ----------------------------------------------------------
    feats = feature_set(df, "pooled_full")
    for group_wise, split_name in ((True, "group"), (False, "row")):
        tr, te = split_rows(df, group_wise)
        res = fit_eval(df.iloc[tr], df.iloc[te], feats)
        add("pooled", res, split_name, feats, "rf", len(tr), len(te))

    # ---- weak baseline: parameter count only, per family ------------------
    for label, subset in (("butter_e", butter), ("ec_nas", ecnas)):
        tr, te = split_rows(subset, True)
        res = fit_eval(subset.iloc[tr], subset.iloc[te], ["params"], model="linear")
        add(f"baseline_params_only:{label}", res, "group", ["params"],
            "linear", len(tr), len(te))

    # ---- RQ4: cross-family transfer, core features only -------------------
    for train_fam, test_fam, a, b in (("cnn", "mlp", ecnas, butter),
                                      ("mlp", "cnn", butter, ecnas)):
        res = fit_eval(a, b, CORE)
        add(f"transfer:{train_fam}_to_{test_fam}", res, "full-holdout", CORE,
            "rf", len(a), len(b))

    # ---- RQ2: measurement-bias correction, pooled, group split ------------
    tr, te = split_rows(df, True)
    for corrected, label in ((False, "uncorrected"), (True, "corrected")):
        frame = df.copy()
        if corrected:
            # BUTTER-E was watt-meter measured, so only EC-NAS is corrected,
            # by the 1.25x midpoint of Fischer's 20-30% underestimation range.
            cnn = frame.family == "CNN"
            frame.loc[cnn, "target"] = frame.loc[cnn, "target"] * 1.25
        res = fit_eval(frame.iloc[tr], frame.iloc[te], feats)
        add("pooled_bias_correction", res, f"group:{label}", feats, "rf",
            len(tr), len(te))

    # ---- leave-one-dataset-out, BUTTER-E ---------------------------------
    lodo = []
    bfeats = feature_set(butter, "butter_full")
    for dataset, held in butter.groupby("n_observations"):
        train = butter[butter.n_observations != dataset]
        res = fit_eval(train, held, bfeats)
        lodo.append({"n_observations": dataset, "n_test": len(held), **res})
    lodo_df = pd.DataFrame(lodo)
    weights = lodo_df["n_test"] / lodo_df["n_test"].sum()
    add("lodo:butter_e",
        {k: float((lodo_df[k] * weights).sum())
         for k in ("mape", "r2_raw", "r2_log", "tau")},
        "leave-one-dataset-out", bfeats, "rf",
        len(butter) - int(lodo_df["n_test"].max()), int(lodo_df["n_test"].max()))

    # ---- model comparison on the pooled core feature set ------------------
    # Reproduces the model-selection table: a parameter-count-only linear
    # baseline, Random Forest, XGBoost, and a from-scratch MLP, each fitted
    # both on the raw target and on log1p(target).
    core = CORE
    tr, te = split_rows(df, True)
    for name in ("linear", "rf", "xgb", "mlp"):
        for log_target, scaling in ((False, "raw"), (True, "log")):
            res = fit_eval(df.iloc[tr], df.iloc[te], core, model=name,
                           log_target=log_target)
            add("model_comparison", res, f"group:core:{scaling}", core, name,
                len(tr), len(te))

    # ---- within-family MLP surrogate (the stronger baseline) --------------
    # EC-NAS's own surrogate design, re-implemented per family on that
    # family's full feature set, matching the baseline table.
    for label, subset, kind in (("butter_e", butter, "butter_full"),
                                ("ec_nas", ecnas, "ecnas_full")):
        f = feature_set(subset, kind)
        tr, te = split_rows(subset, True)
        res = fit_eval(subset.iloc[tr], subset.iloc[te], f, model="mlp")
        add(f"baseline_mlp_surrogate:{label}", res, "group", f, "mlp",
            len(tr), len(te))

    # ---- RQ2: hardware holdout, EC-NAS 4V benchmark ------------------------
    # Train on the Quadro RTX 6000 measurements only, test on each remaining
    # GPU class. Core features only; the benchmark records no op-mix detail.
    hw_path = "data/processed/ec_nas/ec_nas_4v_hardware_features.csv"
    if os.path.exists(hw_path):
        hw = pd.read_csv(hw_path)
        train_gpu = "Quadro RTX 6000"
        train = hw[hw.gpu_type == train_gpu]
        for gpu in sorted(hw.gpu_type.unique()):
            if gpu == train_gpu:
                continue
            test = hw[hw.gpu_type == gpu]
            res = fit_eval(train, test, CORE)
            add(f"hardware_holdout:{gpu}", res, "full-holdout", CORE, "rf",
                len(train), len(test))

    # ---- RQ3: output structure, pooled, group split -----------------------
    # Variant A predicts log power and log duration separately and recomposes
    # energy as their product; Variant B predicts energy directly (the pooled
    # run above, recomputed here so both share one split).
    tr, te = split_rows(df, True)
    train, test = df.iloc[tr], df.iloc[te]
    res_b = fit_eval(train, test, feats)
    add("output_structure", res_b, "group:B_direct_energy", feats, "rf",
        len(tr), len(te))

    power_pred = duration_pred = None
    for col in ("power", "duration"):
        est = RandomForestRegressor(n_estimators=300, n_jobs=-1,
                                    random_state=RANDOM_STATE)
        est.fit(train[feats].to_numpy(float), np.log1p(train[col].to_numpy(float)))
        pred = np.expm1(est.predict(test[feats].to_numpy(float)))
        sub = metrics(test[col].to_numpy(float), pred)
        add(f"output_structure_submodel:{col}", sub, "group", feats, "rf",
            len(tr), len(te))
        if col == "power":
            power_pred = pred
        else:
            duration_pred = pred

    res_a = metrics(test["target"].to_numpy(float),
                    power_pred * duration_pred)
    add("output_structure", res_a, "group:A_power_x_duration", feats, "rf",
        len(tr), len(te))

    out = pd.DataFrame(records)
    out.to_csv(args.out, index=False)
    lodo_df.to_csv("results/tables/lodo_butter_e.csv", index=False)

    pd.set_option("display.width", 200)
    print(out[["experiment", "split", "n_features", "n_train", "n_test",
               "mape", "r2_raw", "r2_log", "tau"]].to_string(index=False))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
