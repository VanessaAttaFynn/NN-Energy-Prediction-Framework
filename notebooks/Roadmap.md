# Project Plan: Cross-Architecture A Priori Training Energy Prediction

## 1. Goal

Predict neural network training energy **before training starts**, using a model that works across architecture families (MLP + CNN) and tests whether it also holds across hardware.

## 2. Research Questions

- **RQ1 (architecture generalization):** Trained on MLP+CNN jointly, does the model stay accurate on an architecture family it never saw in training, vs. a FLOPs-only baseline?
- **RQ2 (hardware generalization):** Trained on one GPU (RTX 6000), does it stay accurate on other GPUs (RTX 3060, RTX 3090, Titan Xp)?
- **RQ3 (bias correction):** Does correcting EC-NAS's power values for known software-measurement underestimation (Fischer et al., 20–30%) improve joint-model accuracy vs. no correction?
- **RQ4 (output structure):** Does predicting Power and Duration separately (E = P × D) beat predicting Energy directly as one number?

## 3. Datasets — raw form (nothing merged yet)

**BUTTER-E** (MLP, watt-meter ground truth)
- Source file to use: `runs_with_standardized_energy.csv`
- Relevant raw columns: `size` (params), `depth`, `shape`, `dataset`, `batch_size`, `learning_rate`, `optimizer`, `is_gpu`, `energy` (J), `run_time` (s)
- Power is NOT a native column → derive as `power = energy / run_time`

**EC-NAS** (CNN, CarbonTracker software estimate)
- Source: `energy_7V9E_surrogate.tfrecord` (main, ~423K archs) + `energy_4V9E_*` hardware-specific files (91 archs each, across 4 GPUs — this is the RQ2 test set)
- Relevant raw fields: adjacency matrix + operation labels (DAG), `θ` (params), `T(s)` (duration), `E(kWh)`, `E(W)` (power — native column, no derivation needed), batch size (check metadata/repo for exact field)

**Known issue to carry into Limitations:** BUTTER-E power is watt-meter-integrated (1-min sampling); EC-NAS power is CarbonTracker software estimate (10-sec sampling, PUE-adjusted at 1.59). These are not the same measurement instrument — this is exactly what RQ3 tests a fix for.

## 4. Feature Engineering — build a shared table

Target schema, one row per architecture, populated separately from each dataset then stacked:

| column | from BUTTER-E | from EC-NAS |
|---|---|---|
| depth | direct (`depth`) | count longest path in DAG (needs a graph traversal function) |
| params | direct (`size`) | direct (`θ`) |
| flops | **must compute** from layer widths | **must compute** from DAG node ops + tensor sizes (use `fvcore`/`ptflops` or hand-derive per op type) |
| batch_size | direct | direct (check field name in repo) |
| epochs | check if present in metadata; else derive | check field |
| power (target) | derived: `energy/run_time` | direct (`E(W)`) |
| duration (target) | direct (`run_time`) | direct (`T(s)`) |
| family | "MLP" | "CNN" |
| aux: kernel_size / pooling flags | 0 (n/a) | direct from op labels |
| aux: shape (e.g. "rectangle") | direct (`shape`) | 0 (n/a) |

**Deliverable of this stage:** two scripts (`prep_butter_e.py`, `prep_ecnas.py`) that each output a CSV in this exact schema. Nothing is combined inside these scripts — combination happens later, per training condition.

## 5. Model variants to build

Two output-structure options × applied identically across all training conditions:

- **Variant A — Decomposed:** two regressors (or one model, two heads) → predict `power`, predict `duration` → `energy = power × duration`
- **Variant B — Direct:** one regressor → predict `energy` directly (recomposed table needs an `energy` column too — compute as `power × duration` after building the shared table, for both datasets, so Variant B has a consistent target)

Both variants use the same feature set (Section 4) and same model class (whole-architecture regression — no layer-wise approach, per Section 3.4's argument and the "data doesn't support layer-wise" conclusion from this conversation).

## 6. Training conditions (apply to both variants A and B)

- **Condition 1 — Separate:** one model (or pair) trained on BUTTER-E rows only, a second trained on EC-NAS rows only. No pooling.
- **Condition 2 — Naive pooled:** stack both prepped CSVs as-is, train one model (or pair) on the combined table.
- **Condition 3 — Corrected pooled:** same as Condition 2, but multiply every EC-NAS row's `power` value by 1.2–1.3 (Fischer et al. correction) before stacking. `duration` left untouched.

**Total model configs: 2 variants × 3 conditions = 6 training runs** (Condition 1 counts as 2 sub-models — MLP-only and CNN-only — so really 2×(2+1+1) = 8 individual trained models, but 6 reportable configurations).

## 7. Evaluation protocol

Run every trained config through:
- **Random split** (standard practice, for comparability to prior literature — not evidence of generalization)
- **Held-out family split** (train on both families pooled, test only on whichever family was fully excluded from that training run) → answers RQ1
- **Held-out hardware split** (train on RTX 6000 EC-NAS data, test on RTX 3060/3090/Titan Xp 4V-space data) → answers RQ2

**Baseline for comparison throughout:** FLOPs-only linear regression, same train/test splits, same conditions. This is the floor everything must beat.

## 8. What answers each RQ

- RQ1 → Condition 2/3 vs Condition 1, under held-out-family split
- RQ2 → best config from RQ1, under held-out-hardware split
- RQ3 → Condition 2 vs Condition 3, held-out-family split (isolate correction's effect)
- RQ4 → Variant A vs Variant B, same condition, same split (isolate decomposition's effect)

## 9. Implementation checklist (cross off as we go)

- [ ] Download BUTTER-E `runs_with_standardized_energy.csv`
- [ ] Download EC-NAS `energy_7V9E_surrogate.tfrecord` + `energy_4V9E_*` (4 hardware variants)
- [ ] Write FLOPs computation for BUTTER-E (MLP layer widths → FLOPs)
- [ ] Write FLOPs computation for EC-NAS (DAG + op labels → FLOPs)
- [ ] Write DAG depth-from-adjacency-matrix function for EC-NAS
- [ ] Build `prep_butter_e.py` → shared-schema CSV
- [ ] Build `prep_ecnas.py` → shared-schema CSV (main 7V space)
- [ ] Build `prep_ecnas_4v_hardware.py` → shared-schema CSV per GPU (for RQ2)
- [ ] Confirm units match (EC-NAS power in Watts vs BUTTER-E derived power — sanity check ranges before pooling)
- [ ] Build pooling function with correction toggle (Condition 2 vs 3)
- [ ] Build FLOPs-only baseline regressor
- [ ] Build Variant A (power+duration two-head/two-model) trainer
- [ ] Build Variant B (direct energy) trainer
- [ ] Run all 6 configs × 3 evaluation splits
- [ ] Compile results tables per RQ

## 10. Open assumptions to verify once data is in hand

- EC-NAS batch size/epoch fields — exact column names TBD once `.tfrecord` is opened
- BUTTER-E epochs — check `summary_by_epoch` file for per-run epoch count if not in metadata
- Whether `fvcore`/`ptflops`-style FLOPs computation is feasible directly from EC-NAS's DAG format, or needs custom per-op-type formulas
