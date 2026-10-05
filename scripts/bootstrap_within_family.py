"""Per-example predictions and cluster-bootstrap uncertainty, within family.

For each family, one grouped 80/20 split (seed 42, as in Table 7) is fitted
with XGBoost, Random Forest, and the four-layer MLP surrogate; per-example
test predictions are saved to results/predictions/. Confidence intervals and
paired differences are computed by resampling test *groups* (configurations
or architectures) with replacement, so repeated runs stay together.
Usage: python scripts/bootstrap_within_family.py [--B 1000]
"""
from __future__ import annotations
import argparse, os, sys
import numpy as np, pandas as pd
from sklearn.model_selection import GroupShuffleSplit
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts import run_experiments as rx  # noqa: E402
from src.models.neural_net import TorchMLPRegressor  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--B", type=int, default=1000); a = ap.parse_args()
df = pd.read_csv(rx.COMBINED)
rng = np.random.default_rng(42)
summary = []
for fam, famcode, kind in (("butter_e", "MLP", "butter_full"), ("ec_nas", "CNN", "ecnas_full")):
    sub = df[df.family == famcode].reset_index(drop=True)
    f = rx.feature_set(sub, kind)
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(sub, groups=sub.group_key))
    Xtr, Xte = sub.loc[tr, f].to_numpy(float), sub.loc[te, f].to_numpy(float)
    ytr = np.log1p(sub.loc[tr, "target"].to_numpy(float))
    out = sub.loc[te, ["run_id", "group_key", "target"]].reset_index(drop=True)
    for name, est in (("xgb", rx.make_estimator("xgb")), ("rf", rx.make_estimator("rf")),
                      ("mlp_surrogate", TorchMLPRegressor(hidden_sizes=(128, 64, 32), epochs=200, random_state=42))):
        est.fit(Xtr, ytr); out[f"pred_{name}"] = np.expm1(est.predict(Xte))
    out.to_csv(f"results/predictions/within_{fam}_seed42.csv", index=False)
    y = out.target.to_numpy(float)
    groups = out.group_key.to_numpy()
    ug, inv = np.unique(groups, return_inverse=True)
    idx_by_g = [np.where(inv == k)[0] for k in range(len(ug))]
    models = ["xgb", "rf", "mlp_surrogate"]
    point = {m: rx.metrics(y, out[f"pred_{m}"].to_numpy(float)) for m in models}
    boot = {m: [] for m in models}
    for _ in range(a.B):
        pick = rng.integers(0, len(ug), len(ug))
        ii = np.concatenate([idx_by_g[k] for k in pick])
        for m in models:
            boot[m].append(rx.metrics(y[ii], out[f"pred_{m}"].to_numpy(float)[ii]))
    for m in models:
        bm = pd.DataFrame(boot[m])
        row = {"family": fam, "model": m}
        for k in ("mape", "r2_raw", "tau"):
            row[k] = point[m][k]; row[f"{k}_lo"] = bm[k].quantile(.025); row[f"{k}_hi"] = bm[k].quantile(.975)
            if m != "xgb":
                diff = pd.DataFrame(boot["xgb"])[k] - bm[k]
                row[f"xgb_minus_{k}_lo"] = diff.quantile(.025); row[f"xgb_minus_{k}_hi"] = diff.quantile(.975)
        summary.append(row)
s = pd.DataFrame(summary); s.to_csv("results/tables/bootstrap_within_family.csv", index=False)
pd.set_option("display.width", 250); print(s.round(4).T.to_string())
