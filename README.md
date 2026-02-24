# Energy Consumption Prediction Framework

**A Machine Learning-Based Framework for Predicting Energy Consumption of Neural Network Training from Architectural and Configuration Features**

MSc Computer Science Thesis | University of Ghana, Legon | 2026

---

## Overview

This project develops ML regression models that predict the energy consumption of neural network training **before training begins**, using only architectural features, hardware specifications, and training configuration parameters.

## Project Structure

```
energy-prediction-framework/
│
├── data/
│   ├── raw/                        # Original datasets (gitignored)
│   │   ├── butter_e/               # BUTTER-E dataset files
│   │   ├── ec_nas/                 # EC-NAS benchmark files
│   │   └── supplementary/          # Node info, GPU specs, PMLB metadata
│   ├── processed/                  # Cleaned & feature-engineered data
│   │   ├── butter_e/
│   │   ├── ec_nas/
│   │   └── combined/              # Merged cross-architecture dataset
│
├── notebooks/
│   ├── 01_exploration/             # EDA for each dataset
│   │   ├── 01a_butter_e_eda.ipynb
│   │   ├── 01b_ec_nas_eda.ipynb
│   │   └── 01c_data_quality.ipynb
│   ├── 02_feature_engineering/     # Feature extraction & preprocessing
│   │   ├── 02a_butter_e_features.ipynb
│   │   ├── 02b_ec_nas_features.ipynb
│   │   └── 02c_combined_features.ipynb
│   ├── 03_modeling/                # Model training
│   │   ├── 03a_within_butter_e.ipynb       # MLP → MLP
│   │   ├── 03b_within_ec_nas.ipynb         # CNN → CNN
│   │   ├── 03c_cross_architecture.ipynb    # MLP ↔ CNN
│   │   ├── 03d_combined_model.ipynb        # MLP+CNN → Any
│   │   └── 03e_baselines.ipynb             # FLOP & TDP baselines
│   ├── 04_evaluation/              # Results comparison
│   │   ├── 04a_model_comparison.ipynb
│   │   ├── 04b_feature_importance.ipynb    # SHAP analysis
│   │   └── 04c_transferability.ipynb
│   └── 05_analysis/                # Final analysis & figures
│       └── 05a_thesis_figures.ipynb
│
├── src/                            # Reusable Python modules
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_butter_e.py
│   │   ├── load_ec_nas.py
│   │   └── merge_datasets.py
│   ├── features/
│   │   ├── __init__.py
│   │   └── engineer.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── random_forest.py
│   │   ├── xgboost_model.py
│   │   └── neural_net.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py
│   │   └── visualize.py
│   └── baselines/
│       ├── __init__.py
│       ├── flop_estimator.py
│       └── tdp_estimator.py
│
├── results/
│   ├── figures/                    # Generated plots
│   ├── tables/                     # Results tables (CSV)
│   └── models/                     # Saved trained models (.pkl, .pt)
│
├── config/
│   └── experiment_config.yaml      # Hyperparameters & settings
│
├── docs/
│   ├── papers/                     # Key reference PDFs
│   └── notes/                      # Research notes
│
├── .gitignore
├── README.md
├── requirements.txt
└── setup.py
```

## Datasets

| Dataset | Architecture | Runs | Energy Source | Link |
|---------|-------------|------|--------------|------|
| BUTTER-E | Fully Connected (MLPs) | 63,527 | Hardware watt-meters | [OEDI](https://data.openei.org/submissions/5991) |
| EC-NAS | CNNs (NAS-Bench-101) | ~423,000 | Measured + surrogate | [GitHub](https://github.com/saintslab/EC-NAS-Bench) |

## Research Questions

1. Which features most strongly predict training energy consumption?
2. How accurately can ML models predict energy from pre-training features?
3. Do predictions generalize across different architecture types?
4. Does ML outperform FLOP-based and TDP-based analytical baselines?

## Evaluation Protocols

- **Within-architecture**: Train/test on same architecture type (MLP→MLP, CNN→CNN)
- **Cross-architecture**: Train on one type, test on other (MLP→CNN, CNN→MLP)
- **Combined**: Unified model trained on merged dataset (MLP+CNN→Any)

## Requirements

See `requirements.txt` for full dependencies. Key packages:
- Python 3.10+
- scikit-learn, xgboost, pytorch
- pandas, numpy, matplotlib, seaborn
- shap, optuna

## License

MIT License
