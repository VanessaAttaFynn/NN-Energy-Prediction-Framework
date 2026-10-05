"""Evaluate CAL-GATE on the CAL-ENERGY experiment regimes.

Part A - support check (Eqs. 10-11): how often requests are flagged, and
whether flagged requests are the ones with large errors, in
  (1) within-family grouped test sets,
  (2) BUTTER-E leave-one-dataset-out,
  (3) leave-one-family-out (shared core features),
  (4) held-out GPUs (EC-NAS 4V, shared core features).

Part B - pilot recalibration (Eq. 12): for the held-out GPUs and both
cross-family directions, draw k labelled pilot runs from the target domain,
estimate alpha, apply it to the remaining target runs, and compare with the
uncorrected predictor and with a Random Forest refitted on the same k pilots.

Usage:
    python scripts/cal_gate_experiments.py [--draws 500]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupShuffleSplit

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.run_experiments import COMBINED, CORE, feature_set, metrics, make_estimator  # noqa: E402
from scripts.cal_gate import CalGate, pilot_alpha  # noqa: E402

RS = 42
HW = "data/processed/ec_nas/ec_nas_4v_hardware_features.csv"
SOURCE_GPU = "Quadro RTX 6000"
KS = (1, 3, 5, 10)
MODEL = "rf"  # set with --model


def rf():
    """Main predictor (name kept for brevity; follows --model)."""
    return make_estimator(MODEL, RS)


def fit_predict(tr: pd.DataFrame, te: pd.DataFrame, feats):
    m = rf().fit(tr[feats].to_numpy(float), np.log1p(tr["target"].to_numpy(float)))
    return m, np.expm1(m.predict(te[feats].to_numpy(float)))


def ape(y, p):
    return np.abs(np.clip(p, 1e-9, None) - y) / y


# ---------------------------------------------------------------- Part A
def part_a(df, butter, ecnas, hw):
    rows, lodo_rows = [], []

    def record(regime, gate, X, y, pred, fam=None, hwl=None):
        c = gate.check(X, families=fam, hardware=hwl)
        e = ape(y, pred)
        f = c["flag"]
        rows.append({
            "regime": regime, "n": len(y), "delta": gate.delta_,
            "flag_rate_distance": c["flag_distance"].mean(),
            "flag_rate_label": c["flag_label"].mean(),
            "flag_rate_total": f.mean(),
            "mape_all": e.mean(),
            "mape_flagged": e[f].mean() if f.any() else np.nan,
            "mape_unflagged": e[~f].mean() if (~f).any() else np.nan,
        })

    # (1) within-family grouped test sets, family feature sets
    for name, sub, kind in (("within:BUTTER-E", butter, "butter_full"),
                            ("within:EC-NAS", ecnas, "ecnas_full")):
        feats = feature_set(sub, kind)
        tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=RS)
                      .split(sub, groups=sub["group_key"]))
        tr, te = sub.iloc[tr], sub.iloc[te]
        _, pred = fit_predict(tr, te, feats)
        gate = CalGate().fit(tr[feats].to_numpy(float))
        record(name, gate, te[feats].to_numpy(float), te["target"].to_numpy(float), pred)

    # (2) BUTTER-E leave-one-dataset-out
    feats = feature_set(butter, "butter_full")
    for nobs, held in butter.groupby("n_observations"):
        tr = butter[butter.n_observations != nobs]
        _, pred = fit_predict(tr, held, feats)
        gate = CalGate().fit(tr[feats].to_numpy(float))
        c = gate.check(held[feats].to_numpy(float))
        m = metrics(held["target"].to_numpy(float), pred)
        lodo_rows.append({"n_observations": nobs, "n_test": len(held),
                          "flag_rate_distance": c["flag_distance"].mean(),
                          "median_distance_over_delta": float(np.median(c["distance"]) / gate.delta_),
                          **m})

    # (3) leave-one-family-out, core features
    for name, src, tgt, sfam, tfam in (("family:EC-NAS->BUTTER-E", ecnas, butter, "CNN", "MLP"),
                                       ("family:BUTTER-E->EC-NAS", butter, ecnas, "MLP", "CNN")):
        _, pred = fit_predict(src, tgt, CORE)
        gate = CalGate().fit(src[CORE].to_numpy(float), families=[sfam])
        record(name, gate, tgt[CORE].to_numpy(float), tgt["target"].to_numpy(float),
               pred, fam=[tfam] * len(tgt))

    # (4) held-out GPUs, core features
    src = hw[hw.gpu_type == SOURCE_GPU]
    gate = CalGate().fit(src[CORE].to_numpy(float), hardware=[SOURCE_GPU])
    for gpu in sorted(hw.gpu_type.unique()):
        if gpu == SOURCE_GPU:
            continue
        tgt = hw[hw.gpu_type == gpu]
        _, pred = fit_predict(src, tgt, CORE)
        record(f"gpu:{gpu}", gate, tgt[CORE].to_numpy(float),
               tgt["target"].to_numpy(float), pred, hwl=tgt["gpu_type"].to_numpy())

    return pd.DataFrame(rows), pd.DataFrame(lodo_rows)


# ---------------------------------------------------------------- Part B
def recal_curve(name, src, tgt, feats, group_col, draws, rng):
    """Pilot recalibration on a target domain, k pilot groups per draw."""
    model, pred_all = fit_predict(src, tgt, feats)
    tgt = tgt.assign(_pred=pred_all)
    groups = tgt[group_col].unique()
    base = metrics(tgt["target"].to_numpy(float), tgt["_pred"].to_numpy(float))
    rows = [{"regime": name, "k": 0, "method": "uncorrected", "draws": 1,
             "mape_median": base["mape"], "mape_q25": base["mape"], "mape_q75": base["mape"],
             "r2_median": base["r2_raw"], "tau_median": base["tau"]}]
    for k in KS:
        if k >= len(groups):
            continue
        res = {"alpha": [], "uncorrected": [], "refit_on_pilots": []}
        r2 = {m: [] for m in res}
        tau = {m: [] for m in res}
        for _ in range(draws):
            pg = rng.choice(groups, size=k, replace=False)
            is_p = tgt[group_col].isin(pg).to_numpy()
            P, R = tgt[is_p], tgt[~is_p]
            y = R["target"].to_numpy(float)
            a = pilot_alpha(P["target"], P["_pred"])
            preds = {"uncorrected": R["_pred"].to_numpy(float),
                     "alpha": a * R["_pred"].to_numpy(float)}
            if k >= 3:
                m = rf().fit(P[feats].to_numpy(float), np.log1p(P["target"].to_numpy(float)))
                preds["refit_on_pilots"] = np.expm1(m.predict(R[feats].to_numpy(float)))
            for meth, p in preds.items():
                mm = metrics(y, p)
                res[meth].append(mm["mape"]); r2[meth].append(mm["r2_raw"]); tau[meth].append(mm["tau"])
        for meth in res:
            if not res[meth]:
                continue
            v = np.array(res[meth])
            rows.append({"regime": name, "k": k, "method": meth, "draws": len(v),
                         "mape_median": np.median(v), "mape_q25": np.quantile(v, .25),
                         "mape_q75": np.quantile(v, .75),
                         "r2_median": float(np.median(r2[meth])),
                         "tau_median": float(np.nanmedian(tau[meth]))})
    return rows


def within_target_reference(tgt, feats, group_col, seeds=range(7)):
    """5-fold grouped CV on the target domain itself (upper reference)."""
    from sklearn.model_selection import GroupKFold
    out = []
    for s in seeds:
        sh = tgt.sample(frac=1, random_state=s).reset_index(drop=True)
        pred = np.zeros(len(sh))
        for tr, te in GroupKFold(5).split(sh, groups=sh[group_col]):
            _, pred[te] = fit_predict(sh.iloc[tr], sh.iloc[te], feats)
        out.append(metrics(sh["target"].to_numpy(float), pred))
    o = pd.DataFrame(out)
    return {"mape_mean": o.mape.mean(), "mape_sd": o.mape.std(),
            "r2_mean": o.r2_raw.mean(), "tau_mean": o.tau.mean(), "tau_sd": o.tau.std()}


def part_b(butter, ecnas, hw, draws):
    rng = np.random.default_rng(RS)
    rows, refs = [], []
    src = hw[hw.gpu_type == SOURCE_GPU]
    for gpu in sorted(hw.gpu_type.unique()):
        if gpu == SOURCE_GPU:
            continue
        tgt = hw[hw.gpu_type == gpu].reset_index(drop=True)
        rows += recal_curve(f"gpu:{gpu}", src, tgt, CORE, "architecture_hash", draws, rng)
        refs.append({"regime": f"gpu:{gpu}", **within_target_reference(tgt, CORE, "architecture_hash")})
    rows += recal_curve("family:BUTTER-E->EC-NAS", butter, ecnas.reset_index(drop=True),
                        CORE, "group_key", draws, rng)
    rows += recal_curve("family:EC-NAS->BUTTER-E", ecnas, butter.reset_index(drop=True),
                        CORE, "group_key", max(draws // 5, 50), rng)
    return pd.DataFrame(rows), pd.DataFrame(refs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=500)
    ap.add_argument("--outdir", default="results/tables")
    ap.add_argument("--model", default="rf", choices=["rf", "xgb"])
    args = ap.parse_args()
    global MODEL
    MODEL = args.model
    sfx = "" if MODEL == "rf" else f"_{MODEL}"

    df = pd.read_csv(COMBINED)
    butter = df[df.family == "MLP"].reset_index(drop=True)
    ecnas = df[df.family == "CNN"].reset_index(drop=True)
    hw = pd.read_csv(HW)

    pd.set_option("display.width", 220)
    a, lodo = part_a(df, butter, ecnas, hw)
    a.to_csv(f"{args.outdir}/cal_gate_support{sfx}.csv", index=False)
    lodo.to_csv(f"{args.outdir}/cal_gate_lodo{sfx}.csv", index=False)
    print(a.round(4).to_string(index=False)); print()
    print(lodo.round(4).to_string(index=False)); print()

    b, refs = part_b(butter, ecnas, hw, args.draws)
    b.to_csv(f"{args.outdir}/cal_gate_pilot_recalibration{sfx}.csv", index=False)
    refs.to_csv(f"{args.outdir}/cal_gate_within_target_reference{sfx}.csv", index=False)
    print(b.round(4).to_string(index=False)); print()
    print(refs.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
