# 🏃‍♂️ How to Run the Game Sales Forecasting Engine  
**Execution Guide**

This document explains how to run each component of the forecasting pipeline:
1. Synthetic dataset generator  
2. Feature validator
3. Feature pruner  
4. Weekly forecasting engine  
5. Power BI dashboard refresh  

Designed so anyone can execute the full workflow without guessing.

---

# 1. 📦 Prerequisites

### ✔ Install Python 3.9–3.11  
Any version in this range will work.

### ✔ Install required Python packages  
From the repository root:

```bash
pip install -r requirements.txt
```

### ✔ Core dependencies used in the model
- numpy  
- pandas  
- xgboost  
- scikit-learn  
- python-dateutil  
- (optional) scikit-optimize — only needed for Bayesian optimisation  

---

# 2. 📁 Folder Structure Overview

Your repository is organised as:

```
game-sales-forecasting-engine/
│
├── src/
│   ├── 01_synthetic_game_generator.py
│   ├── 02_feature_validator.py
│   ├── 03_game_feature_pruner.py
│   ├── 04_model_runner.py
│   └── utils/...
│
├── data/
│   ├── synthetic_examples/
│   │   ├── synthetic_game_sales_timeseries_sample.csv
│   │   ├── new_game_NeonRift_AAA.csv
│   │   ├── new_game_Ashbound_AA.csv
│   │   └── new_game_Pulsebreak_Indie.csv
│   └── (full datasets if available)
│
├── dashboard/
│   ├── GameSalesForecast.pbix
│   └── screenshots/
│
└── docs/
    └── how_to_run.md   ← You are here
```

---

# 3. ▶️ Running the Synthetic Generator

Use this to recreate the entire synthetic dataset from scratch.

```bash
python src/01_synthetic_game_generator.py
```

### Outputs:
```
synthetic_data/
    synthetic_game_sales_timeseries.csv
    new_game_<title>_<dev>.csv
```

These become inputs for *validation*, *pruning*, and *forecasting*.

---

# 4. 🔍 Running the Feature Validator (RUN THIS BEFORE PRUNER)

The validator acts as the quality gate before any modelling or pruning happens.

It checks for:

- missing required columns  
- invalid or out-of-range values  
- broken week_index sequences  
- price / discount / DLC inconsistencies  
- malformed marketing_index  
- non-deterministic or corrupted synthetic rows  

Run:

```bash
python src/02_feature_validator.py
```

### Outputs:
```
synthetic_data/feature_validator/
    validation_report.txt
    validation_flags.csv
```

---

# 5. ✂️ Running the Feature Pruner (AFTER Validator)

This script performs institutional-grade feature pruning:

- missingness filtering  
- zero-variance filtering  
- high-correlation cluster pruning  
- XGBoost gain scores  
- linear coefficients + correlations  
- economic sign checks  
- final Feature Quality Score (0–100)  

Run:

```bash
python src/03_game_feature_pruner.py
```

### Outputs:
```
synthetic_data/feature_pruner/
    pruner_missingness_report.csv
    pruner_corr_matrix.csv
    feature_scores.csv
    selected_features.csv
    selected_feature_list.txt
```

`selected_feature_list.txt` is REQUIRED by the forecasting engine.

---

# 6. 🚀 Running the Weekly Forecasting Engine

This is the main model script.  

It includes:

- Real-space prior decay curve  
- Log-residual XGBoost  
- XSTL similarity learning  
- Reliability scoring (Regime × Drift × XSTL)  
- Promo-safe smoothing  
- P10/P90 uncertainty bands  
- Baseline vs marketing uplift decomposition  

Run:

```bash
python src/04_model_runner.py
```

### Outputs:
```
sales_model_runs/
    new_games_sales_forecast_detailed.csv
    new_games_sales_summary.csv
    new_games_sales_metadata.csv
```

These feed directly into the Power BI dashboard.

---

# 7. 📊 Updating the Power BI Dashboard

After generating forecasts, open:

```
dashboard/GameSalesForecast.pbix
```

Then:

1. Go to **Transform Data → Data Source Settings**  
2. Update paths to the new CSVs:

```
sales_model_runs/new_games_sales_forecast_detailed.csv
sales_model_runs/new_games_sales_summary.csv
sales_model_runs/new_games_sales_metadata.csv
```

3. Click **Refresh**

Your dashboard will now update with:

- Cumulative forecast  
- Lifecycle segmentation (Launch / Mid-tail / Long-tail)  
- P10/P90 uncertainty bands  
- Promo impact timeline  
- Reliability over time  
- Marketing uplift analysis  

---

# 🎉 End-to-End Pipeline Complete


