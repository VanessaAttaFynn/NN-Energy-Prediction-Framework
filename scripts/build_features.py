"""Build the feature tables for the BUTTER-E / EC-NAS energy-prediction study.

Produces three files under data/processed/:

    butter_e/butter_e_features.csv    37,055 MLP training runs
    ec_nas/ec_nas_features.csv         2,532 CNN architectures
    combined/combined_features.csv     the two stacked, shared-schema

Decisions encoded here, and why
-------------------------------
1. EC-NAS uses the 5V9E search space only (2,532 rows), not 4V9E+5V9E
   concatenated (2,805 rows). All 91 of 4V9E's architectures are also present
   in 5V9E, so concatenating gave those 91 architectures four measurements each
   against one for every other architecture -- 4x over-weighting, and a random
   row-level split could place repeats of the same architecture on both sides
   of the train/test boundary. This is the decision recorded in
   notebooks/01_exploration/01b_ec_nas_eda.ipynb section 8, now implemented.

2. No `flops` column. The earlier column was a lookup on parameter count, not a
   computation over the architecture: every one of the 302 distinct parameter
   values mapped to exactly one FLOPs value, so 28 structurally different cell
   graphs shared a single FLOPs figure, and cell graphs with different vertex
   counts were assigned identical FLOPs. A reconstruction from the
   NAS-Bench-101 cell specification was attempted (scripts/ecnas_flops.py) and
   reproduced only 24 of 2,532 recorded parameter counts, so it could not be
   validated. See docs/write-ups/corrections.txt. Instead, real structural
   features are computed directly from each cell graph: the count of each
   operation type and the edge count.

3. Every row carries a `group_key`. Splitting must be done on this key, not on
   rows: (params, depth) has only 279 distinct values across BUTTER-E's 37,055
   rows and 349 across EC-NAS's 2,532, so a row-level split puts near-identical
   feature vectors in both train and test.

4. EC-NAS energy is converted to joules to match BUTTER-E's units, and a
   measurement-bias-corrected target (`target_corrected`, x1.25) is carried
   alongside so the RQ2 correction condition can be run from the same table.

Usage:
    python scripts/build_features.py
"""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import pandas as pd

RAW_BUTTER_E = "data/raw/butter_e"
RAW_EC_NAS = "data/raw/ec_nas"
OUT_BUTTER_E = "data/processed/butter_e/butter_e_features.csv"
OUT_EC_NAS = "data/processed/ec_nas/ec_nas_features.csv"
OUT_COMBINED = "data/processed/combined/combined_features.csv"

# BUTTER-E's primary sweep measures a fixed 3000-epoch budget on every run.
BUTTER_E_EPOCHS = 3000
BUTTER_E_BATCH = 256
# EC-NAS's directly measured subset is a fixed 4-epoch budget.
EC_NAS_EPOCHS = 4
EC_NAS_BATCH = 256
EC_NAS_KWH_TO_J = 3.6e6
# Midpoint of the 20-30% software-vs-watt-meter underestimation Fischer (2025).
EC_NAS_BIAS_CORRECTION = 1.25

OPS = {-1: "input", -2: "output", 0: "conv3x3", 1: "conv1x1", 2: "maxpool"}


def build_butter_e() -> pd.DataFrame:
    df = pd.read_csv(f"{RAW_BUTTER_E}/runs_with_standardized_energy.csv")
    pmlb = pd.read_csv(f"{RAW_BUTTER_E}/pmlb.csv")

    pmlb = pmlb.rename(columns={"n_instances": "n_observations"})
    pmlb["task_encoded"] = (pmlb["task"] == "classification").astype(int)
    props = pmlb[["dataset", "n_observations", "n_features", "n_classes",
                  "task_encoded", "imbalance"]]

    df = df.merge(props, on="dataset", how="left")
    missing = df["n_observations"].isna().sum()
    if missing:
        raise SystemExit(f"{missing} BUTTER-E rows did not match a PMLB dataset")

    out = pd.DataFrame({
        "run_id": df["run_id"],
        "params": df["size"].astype(int),
        "depth": df["depth"].astype(int),
        "epochs": BUTTER_E_EPOCHS,
        "batch_size": BUTTER_E_BATCH,
        "target": df["std_energy"].astype(float),
        "family": "MLP",
        "source_dataset": "BUTTER-E",
        "is_gpu": df["is_gpu"].astype(int),
        "n_observations": df["n_observations"].astype(float),
        "n_features": df["n_features"].astype(float),
        "n_classes": df["n_classes"].astype(float),
        "task_encoded": df["task_encoded"].astype(float),
        "imbalance": df["imbalance"].astype(float),
        # Power and duration are recorded separately by the watt-meter, and
        # their product reproduces the target energy exactly. They are carried
        # so the power x duration output structure can be evaluated.
        "power": df["std_power"].astype(float),
        "duration": df["run_time"].astype(float),
    })

    for shape in sorted(df["shape"].dropna().unique()):
        out[f"shape_{shape}"] = (df["shape"] == shape).astype(int)

    # A repeated configuration: same architecture, device class and training
    # dataset. Repeats of one configuration must not straddle the split.
    out["group_key"] = (
        out["params"].astype(str) + "|" + out["depth"].astype(str) + "|"
        + out["is_gpu"].astype(str) + "|" + out["n_observations"].astype(str)
        + "|" + df["shape"].astype(str).values
    )
    return out


def build_ec_nas() -> pd.DataFrame:
    graphs = json.load(open(f"{RAW_EC_NAS}/graphs/generated_graphs_5V9E.json",
                            encoding="utf-8"))
    pattern = f"{RAW_EC_NAS}/train_model_results/energy/5V9E/4_epochs/*/*/repeat_*/results.json"
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise SystemExit(f"no EC-NAS result files matched {pattern}")

    rows = []
    for path in paths:
        rec = json.load(open(path, encoding="utf-8"))
        arch = rec["model_hash"]
        graph = graphs.get(arch)
        if graph is None:
            raise SystemExit(f"architecture {arch} has no graph definition")
        matrix, ops = graph
        ops = list(ops)

        rows.append({
            "run_id": f"5V9E_{arch}_repeat_1",
            "architecture_hash": arch,
            "params": int(rec["trainable_params"]),
            "depth": len(matrix),
            "epochs": EC_NAS_EPOCHS,
            "batch_size": EC_NAS_BATCH,
            "target": float(rec["total_energy (kWh)"]) * EC_NAS_KWH_TO_J,
            "family": "CNN",
            "source_dataset": "EC-NAS",
            "n_edges": int(np.sum(matrix)),
            "n_conv3x3": sum(1 for o in ops if o == 0),
            "n_conv1x1": sum(1 for o in ops if o == 1),
            "n_maxpool": sum(1 for o in ops if o == 2),
            "duration": float(rec["total_time"]),
        })

    out = pd.DataFrame(rows)
    out["target_corrected"] = out["target"] * EC_NAS_BIAS_CORRECTION
    # EC-NAS records no measured power field, so power is derived from energy
    # and duration. The thesis notes this provenance difference explicitly.
    out["power"] = out["target"] / out["duration"]
    # Keyed on the architecture, so a split cannot separate measurements of the
    # same architecture. Each architecture appears once, so this equals run_id.
    out["group_key"] = out["architecture_hash"]
    return out


def build_combined(butter: pd.DataFrame, ecnas: pd.DataFrame) -> pd.DataFrame:
    butter = butter.copy()
    ecnas = ecnas.copy()

    # Auxiliary columns are family-specific; the shared schema is the
    # intersection of the two tables, plus an explicit family indicator.
    # group_key is excluded from `shared` because it is rebuilt below.
    shared = [c for c in butter.columns
              if c in ecnas.columns and c not in ("run_id", "group_key")]
    aux = [c for c in list(butter.columns) + list(ecnas.columns)
           if c not in shared and c not in ("run_id", "architecture_hash",
                                            "target_corrected", "duration_s",
                                            "group_key")]

    frame = pd.concat([butter, ecnas], ignore_index=True)
    for col in aux:
        if col not in frame.columns:
            frame[col] = 0.0
    frame[aux] = frame[aux].fillna(0.0)

    frame["is_mlp_family"] = (frame["family"] == "MLP").astype(int)
    frame["group_key"] = frame["source_dataset"] + "|" + frame["group_key"]
    keep = ["run_id"] + shared + sorted(aux) + ["is_mlp_family", "group_key"]
    return frame[keep]


def main() -> None:
    butter = build_butter_e()
    ecnas = build_ec_nas()
    combined = build_combined(butter, ecnas)

    for path in (OUT_BUTTER_E, OUT_EC_NAS, OUT_COMBINED):
        os.makedirs(os.path.dirname(path), exist_ok=True)

    butter.to_csv(OUT_BUTTER_E, index=False)
    ecnas.to_csv(OUT_EC_NAS, index=False)
    combined.to_csv(OUT_COMBINED, index=False)

    print(f"BUTTER-E  {butter.shape[0]:>6,} rows x {butter.shape[1]:>2} cols  "
          f"groups={butter['group_key'].nunique():,}")
    print(f"EC-NAS    {ecnas.shape[0]:>6,} rows x {ecnas.shape[1]:>2} cols  "
          f"groups={ecnas['group_key'].nunique():,}")
    print(f"combined  {combined.shape[0]:>6,} rows x {combined.shape[1]:>2} cols")
    print()
    print("EC-NAS op-count distribution:")
    print(ecnas[["n_conv3x3", "n_conv1x1", "n_maxpool", "n_edges"]]
          .describe().loc[["min", "50%", "max"]].to_string())
    print()
    print(f"EC-NAS distinct (params, depth): "
          f"{ecnas.groupby(['params', 'depth']).ngroups} for {len(ecnas):,} rows")
    print(f"EC-NAS params==0 rows: {(ecnas['params'] == 0).sum()}")


if __name__ == "__main__":
    main()
