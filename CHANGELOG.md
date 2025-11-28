---------------------------------------
FEATURE GENERATOR — Version History
---------------------------------------

v1 — Created full synthetic multi-feature game-sales dataset with metadata, rolling windows, momentum, log features, normalised indicators, seasonal events, hype, competition, marketing dynamics, pricing interactions, and one-hot dummies.

v2 — Added franchise history, pre-launch hype, event flags, days-since-event features, new rolling windows, volatility stats, z-scores, EMAs, skew/kurtosis, min/max windows, and advanced competitor features.

v3 — Added generation of demo new-game CSVs for cold-start forecasting.

v4 — Added proxy features for new games using synthetic priors from similar titles.

v5 — Added realistic sales scales, deterministic DLC/discount structures, realistic uplift magnitudes, lognormal noise, proper pricing behaviour, and fully preserved pipeline compatibility.

v6 — Added multi-week discounts and dynamic END_DATE = today.

v7 — Added output for three demo new games (AAA, AA, Indie).

v8 — Set fixed DLC/discount counts per dev-type for full determinism.

v9 — Added realistic peak_sales_range per dev-type, creating true AAA/AA/Indie lifecycle curves.

v10 — Introduced seed ledger for deterministic reproducibility.

v11 — Updated franchise, decay class, and decay-rate logic to more realistic distributions.

v12 — Set discount/DLC timing to late-year, improved franchise strength realism, and refined peak sales ranges.

v13 — Updated decay rates and peak sales realism for new-game CSV generation.

v14 — Made discount and decay rules deterministic for new-game generation.

v15 — Added dev-type-specific discount depth and uplift rules.

v16 — Ensured DLC and discount uplift correctly applies to sales values.

v17 — Global realism pass across all features and lifecycle behaviour.

v18 — Set new-game peak sales into median training-data ranges.

v19 — Added noise to new-game rolling windows, seasonality, marketing, and competition columns.

v20 — Sorted marketing index hierarchy to AAA > AA > Indie.

---------------------------------------
DATA VALIDATOR — Version History
---------------------------------------

v1 — Added full institutional-grade validator with per-title checks (dates, sales, price, DLC, marketing, competition, events, rolling stats) and global checks (duplicates, alignment, NaNs, divergences, leakage).

---------------------------------------
FEATURE PRUNER — Version History
---------------------------------------

v1 — Implemented full feature-pruning pipeline with numeric-only filtering, missingness, zero-variance, correlation clustering, XGB importance, linear explainability, economic sign tests, and final quality scoring.

v2 — Added ability to prune features specifically for a week-1 sales forecasting horizon.

---------------------------------------
MODEL RUNNER — Version History
---------------------------------------

v1 — Built global hybrid ensemble (XGB + LGBM + LSTM + decay prior) with full walk-forward analysis.

v2 — Updated prediction target to Week-1 instead of Week-4.

v3 — Removed LSTM from ensemble due to structural mismatch with dataset.

v4 — Added residual-learning XGB, monotonic constraints, and parametric decay prior for stable lifecycle modelling.

v5 — Added forecast distribution outputs (ensemble_p10 / p90 / spread).

v6 — Added SMAPE and weighting metrics to output CSV.

v7 — Established final hybrid model: Prior Curve + Residual XGB with stability scoring.

v8 — Added full XSTL cross-sectional transfer learning layer (pooled stats + similarity + reliability).

v9 — Added Bayesian SMAPE-driven hyperparameter optimisation.

v10 — Updated structure and formatting of output CSV.

v11 — Updated prior curve to incorporate DLC and discount spikes.

v12 — Improved residual clipping around promo weeks.

v13 — Implemented promo + echo override logic for spike handling.

v14 — Added fixed 70/30 blending rule between XGB and Prior.

v15 — Added 4 transparency diagnostic columns for prediction inspection.

v16 — Added per-row SMAPE output.

v17 — Switched model to log-sales training pipeline.

v18 — Added real-space SMAPE inside cross-validation logic.

v19 — Rebuilt reliability-score formulation to spread values evenly from 0.0–1.0.

v20 — Added dynamic XGB–Prior blending driven by reliability (min 30% XGB).

v21 — Standardised and sorted all output CSV columns.

v22 — Added no-marketing alternative prediction variant.

v23 — Added metadata CSV output for transparency/documentation.

v24 — Final GitHub release version of the model.
