"""Held-out-GPU baselines on the EC-NAS 4V benchmark.

(a) main predictor on core features (as in Table 12);
(b) main predictor on core + static GPU specifications (peak FP32 throughput,
    memory bandwidth, arithmetic-intensity proxy);
(c) peak-throughput scaling rule, no training: measured Quadro RTX 6000
    energy of the same architecture x (Quadro peak FLOPs / target peak FLOPs).
Usage: python scripts/hardware_baselines.py --model xgb
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import run_experiments as rx  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--model", default="xgb"); a = ap.parse_args()
hw = pd.read_csv("data/processed/ec_nas/ec_nas_4v_hardware_features.csv")
SRC = "Quadro RTX 6000"
HWF = rx.CORE + ["peak_flops_gflops", "memory_bandwidth_gbs", "arithmetic_intensity_ratio"]
q = hw[hw.gpu_type == SRC]
rows = []
for g in ["RTX 3060", "RTX 3090", "Titan Xp"]:
    t = hw[hw.gpu_type == g]
    y = t.target.to_numpy(float)
    for name, f in (("core", rx.CORE), ("core+gpu_specs", HWF)):
        m = rx.make_estimator(a.model).fit(q[f].to_numpy(float), np.log1p(q.target))
        rows.append({"gpu": g, "method": f"{a.model}:{name}",
                     **rx.metrics(y, np.expm1(m.predict(t[f].to_numpy(float))))})
    mg = t.merge(q[["architecture_hash", "target", "peak_flops_gflops"]]
                 .rename(columns={"target": "src_e", "peak_flops_gflops": "src_peak"}),
                 on="architecture_hash")
    pred = mg.src_e * mg.src_peak / mg.peak_flops_gflops
    rows.append({"gpu": g, "method": "peak_flops_scaling",
                 **rx.metrics(mg.target.to_numpy(float), pred.to_numpy(float))})
d = pd.DataFrame(rows); d.to_csv(f"results/tables/hardware_baselines_{a.model}.csv", index=False)
print(d.round(4).to_string(index=False))
