## 🎮 Game Sales Forecasting Engine

**A fully reproducible, end-to-end forecasting pipeline for weekly video-game sales prediction.**

This repository contains the complete implementation, including:
1. Synthetic dataset generator  
2. Feature validation and pruning pipeline
3. Multi-layer forecasting engine (Prior Curve → Log-Residual XGB → XSTL → Reliability Blending)
4. Power BI analytics dashboard 
5. Full execution instructions for replicability 

Designed for AAA, AA, Indie and New IP titles, the engine provides:


1. 52-week weekly forecasts
2. P10/P50/P90 scenario ranges
3. Reliability-aware predictions
4. Marketing-uplift attribution
5. Lifecycle segmentation (Launch → Mid-Tail → Long-Tail)
6. Promo-safe smoothing
7. Explainability via an integrated dashboard

---

## 🚀 Key Features

- Fully deterministic, seed-ledger controlled forecasting
- Cold-start support for AAA / AA / Indie / New IP
- Structural prior curves with realistic lifecycle behaviour
- Residual XGBoost with promo-safe smoothing
- XSTL cross-sectional similarity learning
- Automated feature pruning pipeline
- Power BI dashboard for full explainability

---

## 🧱 High-Level Architecture

1. Data Generation (Synthetic)
2. Feature Validation
3. Feature Pruning
4. Prior Curve Generation
5. Residual XGB Modelling
6. XSTL Similarity Layer
7. Reliability Framework
8. Blending + Uncertainty + Uplift
9. Dashboard Integration

---

## 📁 Folder Structure Overview

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
    ├── how_to_run.md
    ├── Brief Proposal - Game Sales Forecasting Engine.pdf   
    ├── Detailed Proposal - Game Sales Forecasting Engine.pdf
    ├── Power BI Dashboard - NeonRift.pdf
    ├── Power BI Dashboard - Ashbound.pdf
    └── Power BI Dashboard - Pulsebreak.pdf 
```

---   

## 📊 Example Forecast Output

Below is a sample from the AAA title *NeonRift*:

| Week | Prior | Blended | P10 | P90 | Reliability |
|------|--------|----------|----------|----------|--------------|
| 0 | 1,450,000 | 1,484,642 | 1,261,946 | 1,707,339 | 0.0677 |
| 1 | 1,328,617 | 1,362,893 | 1,158,459 | 1,567,327 | 0.0732 |

---   

## 🛠 Technologies Used

- Python 3.10  
- NumPy, Pandas, SciKit-Learn, XGBoost  
- Power BI  

---   

## 👤 Author
George Pearson  

