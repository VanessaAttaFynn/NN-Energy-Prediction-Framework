"""Seed sensitivity for the EC-NAS 4V hardware benchmark (as in notebook 03i).

Ceiling: random 80/20 row splits of the Quadro RTX 6000 data, 7 seeds.
Held-out GPUs: model fitted on all Quadro RTX 6000 runs with 7 model seeds.
Usage: python scripts/hardware_seed_check.py --model xgb
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import run_experiments as rx  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="xgb"); a = ap.parse_args()
SEEDS = [0, 1, 2, 3, 4, 42, 100]
hw = pd.read_csv("data/processed/ec_nas/ec_nas_4v_hardware_features.csv")
q = hw[hw.gpu_type == "Quadro RTX 6000"]
rows = []
for s in SEEDS:
    tr, te = train_test_split(q, test_size=0.2, random_state=s)
    m = rx.make_estimator(a.model, s).fit(tr[rx.CORE].to_numpy(float), np.log1p(tr.target))
    rows.append({"test": "ceiling (Quadro RTX 6000)", "seed": s,
                 **rx.metrics(te.target.to_numpy(float), np.expm1(m.predict(te[rx.CORE].to_numpy(float))))})
    m = rx.make_estimator(a.model, s).fit(q[rx.CORE].to_numpy(float), np.log1p(q.target))
    for g in ["RTX 3060", "RTX 3090", "Titan Xp"]:
        t = hw[hw.gpu_type == g]
        rows.append({"test": g, "seed": s,
                     **rx.metrics(t.target.to_numpy(float), np.expm1(m.predict(t[rx.CORE].to_numpy(float))))})
d = pd.DataFrame(rows)
d.to_csv(f"results/tables/hardware_seed_check_{a.model}.csv", index=False)
print(d.groupby("test")[["mape", "r2_raw", "tau"]].agg(["min", "max", "mean", "std"]).round(4).to_string())
