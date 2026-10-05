"""BUTTER-E feature progression including the memory-fit proxy, grouped split.

memory_fit_ratio = params * 4 bytes / RAM of the node the run used
(node RAM from node_sinfo.csv, MEMORY column in MB), as defined in
notebooks/03_modeling/03a_within_butter_e.ipynb.

Usage: python scripts/memory_proxy_check.py --model xgb
"""
from __future__ import annotations
import argparse, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import run_experiments as rx  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="xgb")
args = ap.parse_args(); rx.MAIN_MODEL = args.model
df = pd.read_csv(rx.COMBINED)
b = df[df.family == "MLP"].reset_index(drop=True)
raw = pd.read_csv("data/raw/butter_e/runs_with_standardized_energy.csv", usecols=["run_id", "node"])
sinfo = pd.read_csv("data/raw/butter_e/node_sinfo.csv", skipinitialspace=True)
sinfo.columns = [c.strip() for c in sinfo.columns]
mem = sinfo.groupby("HOSTNAMES")["MEMORY"].max()
raw["node_ram_bytes"] = raw["node"].map(mem) * 1024 ** 2
b = b.merge(raw[["run_id", "node_ram_bytes"]], on="run_id", how="left")
print("runs without node RAM:", int(b.node_ram_bytes.isna().sum()))
b["memory_fit_ratio"] = b["params"] * 4 / b["node_ram_bytes"]
b = b.dropna(subset=["memory_fit_ratio"]).reset_index(drop=True)
shape = [c for c in b.columns if c.startswith("shape_")]
levels = [("core", rx.CORE), ("+hardware+shape", rx.CORE + ["is_gpu"] + shape),
          ("+memory proxy", rx.CORE + ["is_gpu"] + shape + ["memory_fit_ratio"]),
          ("+dataset (final)", rx.feature_set(b, "butter_full"))]
tr, te = rx.split_rows(b, True)
rows = [{"level": n, **rx.fit_eval(b.iloc[tr], b.iloc[te], f)} for n, f in levels]
out = pd.DataFrame(rows); print(out.round(4).to_string(index=False))
out.to_csv(f"results/tables/butter_e_feature_progression_{args.model}.csv", index=False)
