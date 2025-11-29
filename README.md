🎮 Game Sales Forecasting Engine

A fully reproducible, end-to-end forecasting pipeline for weekly video-game sales prediction.

This repository contains the complete implementation, including:


Synthetic data generator

Feature validation and pruning pipeline

Multi-layer forecasting engine
(Prior Curve → Log-Residual XGB → XSTL → Reliability Blending)

Power BI analytics dashboard

Full execution instructions for replicability


Designed for AAA, AA, Indie, and New IP titles, the engine provides:


52-week weekly forecasts

P10/P50/P90 scenario ranges

Reliability-aware predictions

Marketing-uplift attribution

Lifecycle segmentation (Launch → Mid-Tail → Long-Tail)

Promo-safe smoothing

Explainability via an integrated dashboard


🏃‍♂️ How to Run the Game Sales Forecasting Engine

Execution Guide

This document explains how to run each component of the forecasting pipeline:

Synthetic dataset generator

Feature validator

Feature pruner

Weekly forecasting engine

Power BI dashboard refresh

Designed so anyone can execute the full workflow without guessing.


1. 📦 Prerequisites
✔ Install Python 3.9–3.11

Any version in this range will work.

✔ Install required Python packages

From the repository root:

pip install -r requirements.txt

Core dependencies

numpy

pandas

scikit-learn

xgboost

python-dateutil

(optional) scikit-optimize


2. 📁 Repository Structure
game-sales-forecasting-engine/
│
├── src/
│   ├── 01_synthetic_game_generator.py
│   ├── 02_feature_validator.py
│   ├── 03_game_feature_pruner.py
│   ├── 04_model_runner.py
│   └── utils/
│
├── data/
│   ├── synthetic_examples/
│   └── (full datasets if available)
│
├── sales_model_runs/
│   ├── new_games_sales_forecast_detailed.csv
│   ├── new_games_sales_summary.csv
│   └── new_games_sales_metadata.csv
│
├── dashboard/
│   ├── GameSalesForecast.pbix
│   └── screenshots/
│
└── docs/
    └── how_to_run.md


Each script has a single responsibility to ensure clarity, reproducibility, and correct sequencing.


3. 📊 Model Architecture (Summary)
3.1 Behavioural Prior Curve

A real-space structural baseline incorporating:

Dev-type decay behaviour

Seasonal launch effects

DLC and discount uplift + echo windows

Indie late-tail flattening

3.2 Log-Residual XGBoost

The model predicts log residuals, not raw sales:

residual = log(1 + actual) − log(1 + prior)


Residuals are added back to produce stable, variance-controlled predictions.

3.3 XSTL (Cross-Sectional Transfer Learning)

Compares the new title with a historical library of synthetic games using:

Mahalanobis distance

Weekly similarity

Dev-type and franchise patterns

Supports cold-start forecasting.

3.4 Reliability Framework

Three diagnostics:

Regime Confidence

Drift Score

XSTL Similarity

Combined to weight Prior vs XGB (0.30 → 0.90).

3.5 Final Outputs

Each weekly forecast includes:

Final blended prediction

P10 / P50 / P90 uncertainty ranges

Forecast Uncertainty Index

Reliability score

Baseline vs marketing uplift

Metadata for Power BI


4. ▶️ Running the Synthetic Generator

Recreates the entire synthetic dataset from scratch.

python src/01_synthetic_game_generator.py


Outputs:

synthetic_data/
    synthetic_game_sales_timeseries.csv
    new_game_<title>_<dev>.csv
    

5. 🔍 Running the Feature Validator

Ensures data quality before pruning or modelling.

python src/02_feature_validator.py


Checks:

Missing or invalid values

Week-index consistency

Price / discount / DLC issues

Marketing index integrity

Synthetic-data stability

Outputs:

synthetic_data/feature_validator/
    validation_report.txt
    validation_flags.csv
    

6. ✂️ Running the Feature Pruner

Institutional-grade pruning stage:

Missingness filtering

Zero-variance removal

High-correlation cluster pruning

XGBoost gain-ranking

Economic-sign consistency

Feature Quality Score (0–100)

Run:

python src/03_game_feature_pruner.py


Outputs:

synthetic_data/feature_pruner/
    pruner_missingness_report.csv
    pruner_corr_matrix.csv
    feature_scores.csv
    selected_features.csv
    selected_feature_list.txt


selected_feature_list.txt is required by the model runner.


7. 🚀 Running the Weekly Forecasting Engine

Main model execution:

python src/04_model_runner.py


Includes:

Prior curve

Log-residual XGB

XSTL

Reliability blending

Promo-safe smoothing

P10/P90 uncertainty

Marketing uplift decomposition

Outputs:

sales_model_runs/
    new_games_sales_forecast_detailed.csv
    new_games_sales_summary.csv
    new_games_sales_metadata.csv


8. 📊 Updating the Power BI Dashboard

Open:

dashboard/GameSalesForecast.pbix


Then update your CSV paths:

sales_model_runs/new_games_sales_forecast_detailed.csv
sales_model_runs/new_games_sales_summary.csv
sales_model_runs/new_games_sales_metadata.csv


Press Refresh to update:

Lifecycle segmentation

Cumulative forecast

Reliability timeline

Promo impact

Marketing uplift

Uncertainty bands
