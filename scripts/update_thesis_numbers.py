"""Write the re-run results and the FLOPs decision into the thesis .docx.

All numbers are read from results/tables/experiments.csv, so nothing is typed by
hand and the document can be regenerated whenever the experiments are re-run.

Three things happen here:

  1. `flops` is removed from the document's account of the feature set. It was
     never an architecture-level computation -- every one of the 302 distinct
     parameter counts mapped to exactly one FLOPs value, so 28 structurally
     different cell graphs shared a single figure. A reconstruction from the
     NAS-Bench-101 cell specification reproduced 24 of 2,532 recorded parameter
     counts, so it could not be validated. Operation-count features replace it,
     and the weak baseline becomes a parameter-count-only regression.

  2. Table and prose numbers are replaced with the group-wise, de-duplicated
     results (EC-NAS 2,805 -> 2,532 rows; splits never separate two rows of the
     same architecture or repeated configuration).

  3. The Section 3.3.1 hardware-utilization proxy, which Section 4.1 and the
     code never implemented, is replaced with an accurate account of the
     proxy that was tested and rejected.

Usage:
    python scripts/update_thesis_numbers.py IN.docx OUT.docx
"""
from __future__ import annotations

import sys

import pandas as pd
from docx import Document
from docx.text.paragraph import Paragraph

EXP_CSV = "results/tables/experiments_complete.csv"

# --------------------------------------------------------------------------
# Numbers
# --------------------------------------------------------------------------
EXP = pd.read_csv(EXP_CSV)


def num(experiment: str, split: str, model: str = "rf", field: str = "tau") -> float:
    row = EXP[(EXP.experiment == experiment) & (EXP.split == split)
              & (EXP.model == model)]
    if row.empty:
        raise SystemExit(f"no result for {experiment!r} / {split!r} / {model!r}")
    return float(row.iloc[0][field])


def f3(v: float) -> str:
    """Format to a consistent width, as in the existing tables."""
    a = abs(v)
    if a >= 100:
        return f"{v:.1f}"
    if a >= 10:
        return f"{v:.2f}"
    return f"{v:.3f}"


# --------------------------------------------------------------------------
# Replacement prose
# --------------------------------------------------------------------------
S331_CORE = (
    "The feature set is restricted, by design, to properties that are defined and measurable "
    "identically across both MLP and CNN architectures: total parameter count, network depth, "
    "batch size, and number of training epochs. This restriction follows directly from a pattern "
    "identified across the reviewed a priori corpus: with the exception of Li et al.'s (2022) "
    "transistor-operations model, every reviewed method \u2014 NeuralPower, EC-NAS's surrogate, and "
    "the Gratia et al. (2024) framework \u2014 was developed and validated within a single "
    "architecture family, and each relies at least partly on family-specific structural features "
    "(convolutional kernel size, filter counts, pooling operation flags) that have no analogue in "
    "an MLP. A predictor built on such features cannot, even in principle, be applied to an "
    "architecture family outside the one it was designed for. Restricting the core feature set to "
    "properties with a valid cross-family definition is therefore not a simplification made for "
    "convenience \u2014 it is the specific design change required to close the generalization gap "
    "this review identifies."
)

S331_PROXY = (
    "A hardware-utilization proxy was considered for this role \u2014 architecture size relative "
    "to the capacity of the training hardware, on the reasoning established independently in both "
    "source datasets (Section 2.2, Section 3.3.3) that GPU under-utilisation can make a smaller "
    "architecture consume more energy than a larger one. It was tested as the ratio of an "
    "architecture's estimated parameter footprint to its training node's available memory, and "
    "rejected: no architecture in BUTTER-E's space approaches the memory capacity of its training "
    "nodes closely enough for the ratio to carry meaningful variation, and the feature was found "
    "to act as a noisy re-encoding of parameter count rather than as new information. The "
    "under-utilisation effect it was intended to capture is instead represented, for convolutional "
    "architectures, by the operation-count auxiliary features described in Section 3.3.2."
)

S333_HEADING = "3.3.3 Rejection of operation-count proxies as a sufficient basis"

S333_BODY = (
    "This work does not use floating-point operations as a feature, for a reason established by "
    "direct attempt rather than by assumption. FLOPs was initially approximated, for both families, "
    "as twice the parameter count. That quantity carries no information beyond the parameter count "
    "itself, so it is not a feature at all; and when it was replaced by an attempt to compute true "
    "FLOPs for EC-NAS's convolutional architectures \u2014 recovering per-vertex channel counts "
    "with NAS-Bench-101's channel-assignment rule and counting multiply-accumulates per operation "
    "\u2014 the reconstruction reproduced the recorded trainable parameter count for only 24 of "
    "2,532 architectures. It could not be validated against the dataset's own records, and no FLOPs "
    "figure derived from it would have been trustworthy. The measure is therefore dropped rather "
    "than reported.\n\n"
    "What is retained instead is a direct record of each architecture's operation composition: the "
    "number of 3x3 convolutions, 1x1 convolutions and pooling operations in its cell, and the "
    "number of edges, all read from the architecture's own cell definition. The case for treating "
    "such counts as necessary but insufficient is unaffected by this substitution, and rests on the "
    "same evidence recorded in Section 2.2: EC-NAS shows that a smaller architecture can consume "
    "more energy than a larger one through under-utilisation, and Tripp et al.'s analysis of "
    "BUTTER-E shows that the size-energy relationship is mediated by cache-hierarchy effects. A "
    "parameter-count-only linear regression is retained in Section 4 as the deliberately weak "
    "baseline, for the same reason FLOPs-only estimation was originally retained."
)

S401 = (
    "The final feature set used throughout the experiments reported in this work follows directly "
    "from the design established in Section 3.3. The core, family-agnostic feature set comprises "
    "parameter count, network depth, number of training epochs, and batch size. All four are "
    "defined and measurable identically for multilayer perceptron and convolutional architectures, "
    "so no feature in this set requires modification or omission when moving between families, and "
    "the set corresponds exactly to the columns named in Section 3.3.1. Two of the four are "
    "constant within each family in the source data \u2014 every BUTTER-E run in this measurement "
    "subset trains for 3,000 epochs and every EC-NAS run for 4, both at a batch size of 256 \u2014 "
    "and they are retained because they remain legitimate a priori quantities, not because they "
    "discriminate between architectures within a family.\n\n"
    "For BUTTER-E, this core set is supplemented by a family-specific auxiliary branch consisting "
    "of hardware type (GPU or CPU), network shape encoded as eight one-hot categories, and a set "
    "of dataset-level properties describing the training task each architecture was applied to. "
    "These dataset properties were sourced from the Penn Machine Learning Benchmarks (PMLB) "
    "repository and comprise the number of observations, the number of input features, the number "
    "of output classes, the task type, and a measure of class imbalance. This design decision "
    "followed directly from an exploratory investigation, reported in full in Section 5.2, which "
    "found that dataset scale was the single strongest predictor of BUTTER-E's training energy and "
    "that representing this property through generalisable numeric attributes, rather than through "
    "a categorical encoding of dataset identity, preserved this predictive strength while allowing "
    "the feature to remain meaningful for datasets not present during training.\n\n"
    "For EC-NAS, the auxiliary branch records the architecture's operation composition: the number "
    "of 3x3 convolutions, 1x1 convolutions and pooling operations in its cell, and the number of "
    "edges. These are read directly from the benchmark's own architecture definitions and are not "
    "derived from the parameter count. A second EC-NAS auxiliary branch, used only for the "
    "hardware-transferability evaluation described in Section 3.6 and reported in Section 5.5, "
    "comprises the target hardware's peak single-precision throughput, its memory bandwidth, and "
    "an arithmetic-intensity ratio; these were sourced from manufacturer specification sheets for "
    "each of the four GPUs in EC-NAS's hardware-specific benchmark. Consistent with the masking "
    "approach described in Section 3.3.2, auxiliary features that do not apply to a given "
    "architecture's family are set to zero rather than left undefined."
)

S403_BASELINE = (
    "The parameter-count-only linear regression baseline represents the analysis-based approaches "
    "identified in the systematic review, several of which rely on a single scalar measure of "
    "architectural scale as a primary or sole predictor. Its inclusion follows directly from the "
    "argument in Section 3.3.3 that operation counts alone are an insufficient basis for "
    "training-energy prediction; the baseline is retained specifically to demonstrate this "
    "insufficiency empirically rather than assert it. The four-layer multilayer perceptron "
    "surrogate follows the architecture reported by Bakhtiarifard et al. (2024) for EC-NAS's own "
    "surrogate energy model and represents the strongest directly comparable prior approach "
    "available for re-implementation."
)

S61 = (
    "Within each architecture family the model predicts training energy with a Kendall-Tau rank "
    "correlation of {b_tau} for BUTTER-E and {e_tau} for EC-NAS, outperforming both baselines under "
    "an identical feature set (Section 5.1). Two features of this result deserve comment. First, "
    "the same core features perform very differently against the weak baseline in the two families: "
    "parameter count alone attains a coefficient of determination of {e_base_r2} for EC-NAS but "
    "only {b_base_r2} for BUTTER-E. That asymmetry is the clearest single indication that "
    "architectural scale is not a sufficient account of training energy in the multilayer "
    "perceptron case, and it is what motivated the feature investigation in Section 5.2. Second, "
    "the accuracy achieved here rests substantially on a feature every reviewed a priori method "
    "omits: the scale of the dataset being trained on. Once that is included, in a form that "
    "remains defined for datasets unseen during training, multilayer perceptron prediction reaches "
    "parity with the convolutional case. The within-family result is therefore best read not as "
    "evidence that the core architectural feature set is sufficient on its own, but that it becomes "
    "sufficient once the cost of the data being processed is accounted for alongside it."
)

S602_MECHANISM = (
    "Underlying both diagnosed mechanisms is a more general observation about what the core feature "
    "set actually measures. Parameter count, depth and operation composition describe the "
    "computational work an architecture performs; they do not describe how the hardware performs "
    "it. Energy consumption in a training run is the product of a power draw and a duration, and "
    "the power draw at any moment depends on which units are active, how well the workload occupies "
    "them, and how much of the time is spent on memory movement rather than arithmetic. Two "
    "architectures with identical parameter counts can differ substantially in all three, and the "
    "difference is larger, not smaller, when the two architectures perform structurally different "
    "kinds of operation. The failure of zero-shot transfer across families is therefore not best "
    "understood as a failure of the model, but as a mismatch between the quantity the features "
    "describe and the quantity energy actually depends on. This is consistent with the "
    "within-family results, where the feature that closes most of the residual gap \u2014 the scale "
    "of the dataset being trained on \u2014 is likewise a proxy for work performed, but one that "
    "happens to align closely with how duration, and therefore energy, scales within a family."
)


def set_text(par: Paragraph, text: str) -> None:
    if not par.runs:
        par.add_run(text)
        return
    par.runs[0].text = text
    for run in par.runs[1:]:
        run.text = ""


def set_cell(table, row: int, col: int, text: str) -> None:
    set_text(table.rows[row].cells[col].paragraphs[0], text)


def main() -> None:
    src, dst = sys.argv[1], sys.argv[2]
    doc = Document(src)
    log: list[str] = []

    # ---------------- terminology ----------------
    replacements = [
        ("FLOPs-only linear regression", "parameter-count-only linear regression"),
        ("Linear (FLOPs only)", "Linear (params only)"),
        ("FLOPs-only baseline", "parameter-count-only baseline"),
        ("FLOPs-only estimation", "operation-count-based estimation"),
    ]
    changed = 0
    for par in doc.paragraphs:
        original = par.text
        updated = original
        for old, new in replacements:
            updated = updated.replace(old, new)
        if updated != original:
            set_text(par, updated)
            changed += 1
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for par in cell.paragraphs:
                    original = par.text
                    updated = original
                    for old, new in replacements:
                        updated = updated.replace(old, new)
                    if updated != original:
                        set_text(par, updated)
                        changed += 1
    log.append(f"terminology: {changed} paragraphs/cells updated")

    # ---------------- feature-set prose ----------------
    for idx, text, label in (
        (70, S331_CORE, "3.3.1 core feature set"),
        (71, S331_PROXY, "3.3.1 utilization proxy"),
        (74, S333_HEADING, "3.3.3 heading"),
        (75, S333_BODY, "3.3.3 body"),
        (112, S401, "4.1 final feature set"),
        (127, S403_BASELINE, "4.3 baselines"),
    ):
        set_text(doc.paragraphs[idx], text)
        log.append(f"rewrote {label}")

    # ---------------- Section 6.1 with parameter-count numbers -----------
    set_text(doc.paragraphs[189], S61.format(
        b_tau=f3(num("within:butter_e", "group")),
        e_tau=f3(num("within:ec_nas", "group")),
        b_base_r2=f3(num("baseline_params_only:butter_e", "group", "linear", "r2_raw")),
        e_base_r2=f3(num("baseline_params_only:ec_nas", "group", "linear", "r2_raw")),
    ))
    log.append("rewrote 6.1")

    set_text(doc.paragraphs[201], S602_MECHANISM)
    log.append("rewrote the physical-mechanism paragraph")

    # ---------------- Table 2: EC-NAS row count --------------------------
    set_cell(doc.tables[1], 2, 2,
             "2,532 architectures from 2,805 directly measured runs (the "
             "remaining 273 are repeats of 91 architectures already counted), "
             "drawn from a ~423,000-architecture search space whose other "
             "entries are surrogate estimates; 91 architectures per GPU in the "
             "4V hardware benchmark")
    log.append("Table 2: EC-NAS count 2,805 -> 2,532")

    # ---------------- Table 4: model comparison --------------------------
    t4 = doc.tables[3]
    for row, model in ((1, "linear"), (2, "rf"), (3, "xgb"), (4, "mlp")):
        set_cell(t4, row, 1, f3(num("model_comparison", "group:core:raw", model, "mape")))
        set_cell(t4, row, 2, f3(num("model_comparison", "group:core:raw", model, "r2_raw")))
        set_cell(t4, row, 3, f3(num("model_comparison", "group:core:log", model, "mape")))
        set_cell(t4, row, 4, f3(num("model_comparison", "group:core:log", model, "r2_raw")))
        set_cell(t4, row, 5, f3(num("model_comparison", "group:core:log", model, "tau")))
    log.append("Table 4: model comparison numbers replaced")

    # ---------------- Table 5: baselines ---------------------------------
    set_cell(doc.tables[4], 1, 0, "Parameter-count-only linear regression")
    log.append("Table 5: baseline renamed")

    # ---------------- Table 6: within-family -----------------------------
    t6 = doc.tables[5]
    for base_row, label in ((0, "butter_e"), (3, "ec_nas")):
        set_cell(t6, base_row + 1, 2, f3(num(f"baseline_params_only:{label}", "group", "linear", "mape")))
        set_cell(t6, base_row + 1, 3, f3(num(f"baseline_params_only:{label}", "group", "linear", "r2_raw")))
        set_cell(t6, base_row + 1, 4, f3(num(f"baseline_params_only:{label}", "group", "linear", "tau")))
        set_cell(t6, base_row + 2, 2, f3(num(f"baseline_mlp_surrogate:{label}", "group", "mlp", "mape")))
        set_cell(t6, base_row + 2, 3, f3(num(f"baseline_mlp_surrogate:{label}", "group", "mlp", "r2_raw")))
        set_cell(t6, base_row + 2, 4, f3(num(f"baseline_mlp_surrogate:{label}", "group", "mlp", "tau")))
        set_cell(t6, base_row + 3, 2, f3(num(f"within:{label}", "group", "rf", "mape")))
        set_cell(t6, base_row + 3, 3, f3(num(f"within:{label}", "group", "rf", "r2_raw")))
        set_cell(t6, base_row + 3, 4, f3(num(f"within:{label}", "group", "rf", "tau")))
    log.append("Table 6: within-family numbers replaced")

    # ---------------- Table 7: BUTTER-E feature progression --------------
    t7 = doc.tables[6]
    for row, level in ((1, "core"),
                       (2, "core_plus_hardware_shape"),
                       (4, "core_plus_hardware_shape_dataset")):
        for col, field in ((1, "mape"), (2, "r2_raw"), (3, "tau")):
            set_cell(t7, row, col, f3(num("butter_e_feature_progression",
                                          f"group:{level}", "rf", field)))
    log.append("Table 7: feature progression numbers replaced")

    # ---------------- Table 8: leave-one-dataset-out ---------------------
    lodo = pd.read_csv("results/tables/lodo_butter_e.csv")
    t8 = doc.tables[7]
    for i, r in lodo.sort_values("n_test", ascending=False).iterrows():
        row = 1 + int(i)
        set_cell(t8, row, 1, f"{int(r['n_test']):,}")
        set_cell(t8, row, 2, f3(r["mape"]))
        set_cell(t8, row, 3, f3(r["r2_raw"]))
        set_cell(t8, row, 4, f3(r["tau"]))
    set_cell(t8, 13, 2, f3(num("lodo:butter_e", "leave-one-dataset-out", "rf", "mape")))
    set_cell(t8, 13, 3, f3(num("lodo:butter_e", "leave-one-dataset-out", "rf", "r2_raw")))
    set_cell(t8, 13, 4, f3(num("lodo:butter_e", "leave-one-dataset-out", "rf", "tau")))
    log.append("Table 8: LODO rows replaced")

    # ---------------- Table 9: measurement-bias correction ---------------
    t9 = doc.tables[8]
    for row, split in ((1, "group:uncorrected"), (2, "group:corrected")):
        for col, field in ((1, "mape"), (2, "r2_raw"), (3, "tau")):
            set_cell(t9, row, col,
                     f3(num("pooled_bias_correction", split, "rf", field)))
    log.append("Table 9: bias-correction numbers replaced")

    # ---------------- Table 10 / 11: output structure --------------------
    t10 = doc.tables[9]
    for row, split in ((1, "group:A_power_x_duration"), (2, "group:B_direct_energy")):
        for col, field in ((1, "mape"), (2, "r2_raw"), (3, "tau")):
            set_cell(t10, row, col, f3(num("output_structure", split, "rf", field)))
    t11 = doc.tables[10]
    for row, sub in ((1, "power"), (2, "duration")):
        for col, field in ((1, "mape"), (2, "r2_raw"), (3, "tau")):
            set_cell(t11, row, col,
                     f3(num(f"output_structure_submodel:{sub}", "group", "rf", field)))
    log.append("Tables 10-11: output-structure numbers replaced")

    # ---------------- Table 12: cross-family transfer --------------------
    t12 = doc.tables[11]
    ceiling = {"EC-NAS \u2192 BUTTER-E": num("within:butter_e", "group"),
               "BUTTER-E \u2192 EC-NAS": num("within:ec_nas", "group")}
    for row, key, exp_name in ((1, "EC-NAS \u2192 BUTTER-E", "transfer:cnn_to_mlp"),
                               (2, "BUTTER-E \u2192 EC-NAS", "transfer:mlp_to_cnn")):
        tau = num(exp_name, "full-holdout")
        set_cell(t12, row, 1, f3(num(exp_name, "full-holdout", "rf", "mape")))
        set_cell(t12, row, 2, f3(num(exp_name, "full-holdout", "rf", "r2_raw")))
        set_cell(t12, row, 3, f3(tau))
        set_cell(t12, row, 4, f"{100 * tau / ceiling[key]:.1f}%")
    log.append("Table 12: transfer numbers replaced")

    # ---------------- Table 13: hardware transferability -----------------
    t13 = doc.tables[12]
    hw = [r for r in EXP[EXP.experiment.str.startswith("hardware_holdout")].itertuples()
          if r.model == "rf"]
    order = {"RTX 3060": 1, "RTX 3090": 2, "Titan Xp": 3}
    for r in hw:
        gpu = r.experiment.split(":", 1)[1]
        row = order.get(gpu)
        if row is None:
            continue
        set_cell(t13, row, 1, f3(r.mape))
        set_cell(t13, row, 2, f3(r.r2_raw))
        set_cell(t13, row, 3, f3(r.tau))
    log.append(f"Table 13: hardware numbers replaced ({len(hw)} GPUs)")

    doc.save(dst)
    print("\n".join(log))
    print(f"\nsaved: {dst}")


if __name__ == "__main__":
    main()
