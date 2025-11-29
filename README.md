🎮 Game Sales Forecasting Engine

A fully reproducible, end-to-end forecasting pipeline for weekly video-game sales prediction.

This repository contains the complete implementation described in the Game Sales Forecasting Engine – Brief Proposal Document 

Brief Proposal - Game Sales For…

, including:

Synthetic data generator

Feature validation and pruning pipeline

Multi-layer forecasting engine (Prior Curve → Log-Residual XGB → XSTL → Reliability Blending)

Power BI analytics dashboard

Full execution instructions for replicability

Designed for AAA, AA, Indie, and New IP titles, the engine provides:

52-week weekly forecasts

P10/P50/P90 scenario ranges

Reliability-aware predictions

Marketing uplift attribution

Lifecycle segmentation

Promo impact smoothing

Cold-start capability

📘 1. Overview

Publishers operate in a hit-driven, volatile market where traditional forecasting (analogues, franchise ratios, static curves) fails to capture real-world behaviour.
As highlighted in the proposal (pp. 1–2) 

Brief Proposal - Game Sales For…

, sales are shaped by:

DLC drops

Discount cadence

Seasonal effects

Marketing intensity

Competitive pressure

Platform momentum

This repository provides a modern, risk-aware, multi-layer forecasting system inspired by quantitative finance and production-grade ML.

Architecture Layers (p. 3–4) 

Brief Proposal - Game Sales For…

:

Behavioural Prior Curve (real-space, dev-type lifecycle model)

Log-Residual XGBoost (stable residual learning)

XSTL Cross-Sectional Transfer Layer (borrows behaviour from similar titles)

Reliability Framework (Regime × Drift × XSTL)

Promo-Safe Smoothing

Uncertainty Bands & Baseline Decomposition

The repository implements the complete system—including all data, model code, reproducibility assets, and Power BI dashboards.

📁 2. Repository Structure
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
│   └── (full datasets when available)
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


Each script has a single responsibility to ensure auditability, reproducibility, and clean debugging.

🧠 3. Model Architecture (From Proposal Summary)
3.1 Behavioural Prior Curve

A structural lifecycle model with dev-type decay profiles, seasonality, DLC/discount echoes, and late-tail flattening (Indie).
See Proposal p. 3 

Brief Proposal - Game Sales For…

.

3.2 Residual XGBoost (Log Space)

Predicts log-residuals rather than raw sales → reduces variance, stabilises training, avoids overfitting, and ensures commercially realistic behaviour.

3.3 XSTL Cross-Sectional Transfer Learning

Compares the new title with 200 synthetic historical titles using distance + weekly similarity.
Supports cold-start forecasting.
Described in Proposal p. 3–4 

Brief Proposal - Game Sales For…

.

3.4 Reliability Framework

Three diagnostics:

Regime Confidence

Drift Score

XSTL Similarity

Combined into a final reliability score that determines how much weight XGB receives versus the structural prior.
See Example on p. 4 (e.g., Week-0 weight = 0.30) 

Brief Proposal - Game Sales For…

.

3.5 Outputs (all weekly)

Final blended forecast

Baseline vs marketing uplift

P10 / P50 / P90 ranges

Reliability score

Metadata for Power BI

Promo-safe adjustments

🚀 4. Running the Pipeline (End-to-End)
4.1 Install requirements
pip install -r requirements.txt


Key dependencies:

numpy

pandas

scikit-learn

xgboost

python-dateutil

(optional) scikit-optimize

▶️ Step 1 — Generate Synthetic Dataset
python src/01_synthetic_game_generator.py


Outputs include:

synthetic_data/
    synthetic_game_sales_timeseries.csv
    new_game_<title>_<dev>.csv

🔍 Step 2 — Run Feature Validator
python src/02_feature_validator.py


Outputs:

synthetic_data/feature_validator/
    validation_report.txt
    validation_flags.csv

✂️ Step 3 — Run Feature Pruner
python src/03_game_feature_pruner.py


Outputs:

synthetic_data/feature_pruner/
    selected_feature_list.txt   ← REQUIRED

📈 Step 4 — Run Weekly Forecasting Engine
python src/04_model_runner.py


Outputs:

sales_model_runs/
    new_games_sales_forecast_detailed.csv
    new_games_sales_summary.csv
    new_games_sales_metadata.csv


These are the inputs for Power BI.

📊 5. Power BI Dashboard

Open:

dashboard/GameSalesForecast.pbix


Update the data source paths to the newest CSVs, then hit Refresh.

Dashboard visuals include (Proposal p. 6) 

Brief Proposal - Game Sales For…

:

Executive KPI Header (Dev-type, strength, totals, SMAPE, half-life, plateau, uncertainty)

Baseline vs Blended Curve

Cumulative Forecast + Lifecycle Segmentation (Launch/Mid-tail/Long-tail)

Uncertainty Bands (P10–P90)

Promo Impact Timeline (DLC, discount)

Marketing ROI vs Baseline

Model Reliability Over Time

These visuals make the model explainable for commercial, finance, and marketing teams.

📘 6. Example: NeonRift AAA (Proposal Reference)

The README can optionally include a short example (derived from pp. 4–5) 

Brief Proposal - Game Sales For…

:

Peak: 1.45M

k-decay: 0.087

Week-0 blended: 1.48M

Week-0 marketing uplift: ~520k

Year-1 total: ~17.6M units

Reliability scaling XGB weight: 0.30 → ~0.59

If you'd like this inside the README, I can format it into a collapsible <details> block.

🔁 7. Reproducibility

The entire system is:

Deterministic

Seed-ledger controlled

Versioned (v25_1 → v31)

Modular (each stage independently executable)

Every forecast is reproducible on any machine running Python 3.9–3.11.
