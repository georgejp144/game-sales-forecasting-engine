# ============================================================
# Game Feature Pruner — v2
#
# PURPOSE:
#   Perform institutional-grade feature pruning on the unified
#   game-sales dataset to prepare high-quality inputs for the
#   forecasting engine. Includes:
#       • Numeric-only cleaning
#       • Missingness / zero-variance filtering
#       • High-correlation cluster pruning
#       • XGBoost gain importance ranking
#       • Linear explainability (coefficients + correlations)
#       • Economic sign alignment (domain checks)
#       • Combined 0–100 Feature Quality Score
#       • Final selection (Top-N + Score threshold)
#
# INPUT:
#   synthetic_data/synthetic_game_sales_timeseries.csv
#
# OUTPUT:
#   synthetic_data/feature_pruner/
#       pruner_missingness_report.csv
#       pruner_corr_matrix.csv
#       feature_scores.csv
#       selected_features.csv
#       selected_feature_list.txt
#
# RUN:
#   03_feature_pruner.py
#
# ============================================================


# =============================================================
# IMPORTS
# =============================================================

import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


# =============================================================
# CONFIG
# =============================================================

RAW_PATH = "synthetic_data/synthetic_game_sales_timeseries.csv"
OUT_DIR = "synthetic_data/feature_pruner"

TARGET = "sales_1w_forward"

MAX_MISSING_PCT = 40.0
HIGH_CORR_THRESHOLD = 0.92
TOP_N_FEATURES = 30
MIN_SCORE_THRESHOLD = 45.0

os.makedirs(OUT_DIR, exist_ok=True)

PRUNER_XGB_PARAMS = dict(
    n_estimators=400,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.6,
    min_child_weight=4,
    reg_lambda=1.0,
    reg_alpha=0.1,
    random_state=42,
    tree_method="hist",
    n_jobs=1,
)

# ID-like fields that should not be used as predictors
ID_LIKE_NUMERIC = ["title_id", "week_index"]


# =============================================================
# LOAD DATA AND BUILD TARGET
# =============================================================

if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Could not find game feature file at: {RAW_PATH}")

df_raw = pd.read_csv(RAW_PATH)
print(f"[GAME_PRUNER] Loaded raw dataset: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")

if "title_id" not in df_raw.columns or "sales" not in df_raw.columns:
    raise ValueError("[GAME_PRUNER] Expected columns 'title_id' and 'sales' not found in dataset")

# -------------------------------------------------------------
# Build 1-week-ahead target
# -------------------------------------------------------------
df_raw[TARGET] = (
    df_raw
    .groupby("title_id")["sales"]
    .shift(-1)
)

df_raw = df_raw.dropna(subset=[TARGET]).reset_index(drop=True)
print(f"[GAME_PRUNER] After dropping NaN target rows: {df_raw.shape[0]} rows")


# =============================================================
# NUMERIC-ONLY CLEANING
# =============================================================

target_series = df_raw[TARGET].copy()

df_num = df_raw.select_dtypes(include=[np.number]).copy()

if TARGET not in df_num.columns:
    df_num[TARGET] = target_series.values

df = df_num.copy()
del df_num

print(f"[GAME_PRUNER] After numeric-only filter: {df.shape[0]} rows, {df.shape[1]} numeric columns")

if df[TARGET].isna().any():
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"[GAME_PRUNER] After cleaning NaN target rows: {df.shape[0]} rows")


# =============================================================
# STEP 1 — MISSINGNESS + ZERO-VARIANCE FILTERS
# =============================================================

def _describe_filters():
    """
    Internal helper (unused at runtime).
    Describes the logic for:
        - Removing zero-variance features
        - Removing features with excessive missingness
    Provided only for documentation consistency.
    """
    return None

feature_cols = [c for c in df.columns if c != TARGET]
feature_cols = [c for c in feature_cols if c not in ID_LIKE_NUMERIC]

missing_pct = df[feature_cols].isna().mean() * 100.0
nunique = df[feature_cols].nunique()

zero_var_features = nunique[nunique <= 1].index.tolist()
too_missing_features = missing_pct[missing_pct > MAX_MISSING_PCT].index.tolist()

keep_features = [
    c for c in feature_cols
    if (c not in zero_var_features) and (c not in too_missing_features)
]

print(f"[GAME_PRUNER] Features before filters       : {len(feature_cols)}")
print(f"[GAME_PRUNER] Zero-variance features dropped: {len(zero_var_features)}")
print(f"[GAME_PRUNER] High-missing features dropped : {len(too_missing_features)}")
print(f"[GAME_PRUNER] Features after basic filters  : {len(keep_features)}")

missing_report = pd.DataFrame(
    {
        "Feature": feature_cols,
        "Missing_Pct": missing_pct.reindex(feature_cols).values,
        "N_Unique": nunique.reindex(feature_cols).values,
        "Zero_Var_Flag": [1 if f in zero_var_features else 0 for f in feature_cols],
        "Too_Missing_Flag": [1 if f in too_missing_features else 0 for f in feature_cols],
    }
)
missing_report.to_csv(os.path.join(OUT_DIR, "pruner_missingness_report.csv"), index=False)
print("[GAME_PRUNER] Saved pruner_missingness_report.csv")

df = df[keep_features + [TARGET]]

if len(keep_features) == 0:
    raise RuntimeError("[GAME_PRUNER] No features left after missingness/variance filtering")


# =============================================================
# STEP 2 — HIGH-CORRELATION CLUSTER PRUNING
# =============================================================

def _describe_corr_pruning():
    """
    Document the correlation-cluster pruning step.

    Removes features highly correlated (> threshold) with others
    to reduce redundancy and multicollinearity.
    """
    return None

corr = df[keep_features].corr().abs()

corr.to_csv(os.path.join(OUT_DIR, "pruner_corr_matrix.csv"))
print("[GAME_PRUNER] Saved pruner_corr_matrix.csv")

upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_corr_to_drop = [
    column for column in upper.columns if any(upper[column] > HIGH_CORR_THRESHOLD)
]

cluster_pruned_features = [f for f in keep_features if f not in high_corr_to_drop]

print(f"[GAME_PRUNER] High-corr features dropped    : {len(high_corr_to_drop)}")
print(f"[GAME_PRUNER] Features after corr pruning   : {len(cluster_pruned_features)}")

if len(cluster_pruned_features) < 3:
    print("[GAME_PRUNER] Warning: too few features after corr pruning, reverting to pre-corr set")
    cluster_pruned_features = keep_features.copy()

df = df[cluster_pruned_features + [TARGET]]
features = cluster_pruned_features


# =============================================================
# STEP 3 — XGBOOST FEATURE IMPORTANCE
# =============================================================

def _describe_xgb_importance():
    """
    Document the XGBoost importance step.

    Fits a small XGBoost model and extracts feature gain scores,
    measuring each feature's contribution to model splits.
    """
    return None

X = df[features].copy()
y = df[TARGET].values

X = X.fillna(X.median())

xgb = XGBRegressor(**PRUNER_XGB_PARAMS)
xgb.fit(X.values, y)

booster = xgb.get_booster()
gain_raw = booster.get_score(importance_type="gain")

feature_map = {f"f{i}": feat for i, feat in enumerate(features)}

xgb_gain = {}
for f_idx, gain_val in gain_raw.items():
    feat_name = feature_map.get(f_idx)
    if feat_name is not None:
        xgb_gain[feat_name] = gain_val

xgb_gain_full = {feat: xgb_gain.get(feat, 0.0) for feat in features}


# =============================================================
# STEP 4 — LINEAR EXPLAINABILITY
# =============================================================

def _describe_linear_explainability():
    """
    Document linear-explainability step:
        - Standardised coefficients
        - Correlation with target
        - Measures linear influence strength
    """
    return None

scaler = StandardScaler()
X_std = scaler.fit_transform(X.values)

lin = LinearRegression()
lin.fit(X_std, y)
coefs = lin.coef_

corrs = []
for col in features:
    series = df[col].values
    if np.std(series) == 0:
        corrs.append(0.0)
    else:
        corrs.append(np.corrcoef(series, y)[0, 1])
corrs = np.array(corrs, dtype=float)


# =============================================================
# STEP 5 — ECONOMIC SIGN MAP
# =============================================================

def _describe_econ_signs():
    """
    Document the economic sign check:

        +1 → feature expected to increase sales
        -1 → feature expected to decrease sales
         0 → neutral / unknown

    Used to penalise features whose estimated sign contradicts domain logic.
    """
    return None

economic_sign_map = {
    "sales": 1,
    "log_sales": 1,
    "sales_rolling_2w": 1,
    "sales_rolling_4w": 1,
    "sales_rolling_6w": 1,
    "sales_rolling_8w": 1,
    "sales_rolling_13w": 1,

    "franchise_strength": 1,
    "marketing_index": 1,
    "marketing_norm": 1,
    "hype_score": 1,

    "price": -1,
    "competitor_intensity": -1,
    "competitor_norm": -1,
}

econ_scores = []
for feat, coef_val in zip(features, coefs):
    coef_sign = np.sign(coef_val)
    expected = economic_sign_map.get(feat, None)

    if coef_sign == 0 or expected is None:
        econ_scores.append(0.5)
    elif expected == 0:
        econ_scores.append(0.5)
    elif coef_sign == expected:
        econ_scores.append(1.0)
    else:
        econ_scores.append(0.0)

econ_scores = np.array(econ_scores, dtype=float)


# =============================================================
# STEP 6 — FEATURE QUALITY SCORE
# =============================================================

def _describe_quality_score():
    """
    Document the combined Feature Quality Score:

    Weighted blend:
        - 35% standardised linear coefficient
        - 35% target correlation strength
        - 25% XGBoost gain importance
        -  5% economic sign alignment

    Produces a 0–100 composite score.
    """
    return None

coef_abs = np.abs(coefs)
max_coef = np.max(coef_abs) if np.any(np.isfinite(coef_abs)) else 0.0
coef_norm = coef_abs / max_coef if max_coef > 0 else np.zeros_like(coef_abs)

max_corr = np.nanmax(np.abs(corrs)) if np.any(np.isfinite(corrs)) else 0.0
corr_norm = np.nan_to_num(np.abs(corrs) / max_corr, nan=0.0) if max_corr > 0 else np.zeros_like(corrs)

xgb_vals = np.array([xgb_gain_full[f] for f in features], dtype=float)
max_gain = np.nanmax(xgb_vals) if np.any(np.isfinite(xgb_vals)) else 0.0
xgb_norm = np.nan_to_num(xgb_vals / max_gain, nan=0.0) if max_gain > 0 else np.zeros_like(xgb_vals)

feature_quality = 100.0 * (
    0.35 * coef_norm
    + 0.35 * corr_norm
    + 0.25 * xgb_norm
    + 0.05 * econ_scores
)

feature_scores = pd.DataFrame(
    {
        "Feature": features,
        "Coef_Standardised": coefs,
        "Corr_with_Target": corrs,
        "XGB_Gain": [xgb_gain_full[f] for f in features],
        "Economic_Score": econ_scores,
        "Feature_Quality_Score": feature_quality,
    }
).sort_values("Feature_Quality_Score", ascending=False).reset_index(drop=True)

feature_scores.to_csv(os.path.join(OUT_DIR, "feature_scores.csv"), index=False)
print("[GAME_PRUNER] Saved feature_scores.csv")


# =============================================================
# STEP 7 — FINAL SELECTION
# =============================================================

def _describe_final_selection():
    """
    Document the final feature selection logic:

        - Keep Top-N highest scoring features
        - Add all features above quality threshold
        - Combine, de-duplicate, and sort list
        - Ensure minimum feature count for modelling
    """
    return None

topN = feature_scores.head(TOP_N_FEATURES)["Feature"].tolist()
strong = feature_scores[
    feature_scores["Feature_Quality_Score"] >= MIN_SCORE_THRESHOLD
]["Feature"].tolist()

final_features = sorted(set(topN + strong))

if len(final_features) < 3:
    print("[GAME_PRUNER] Warning: very few features selected; relaxing to Top-N only")
    final_features = sorted(set(topN))

print(f"[GAME_PRUNER] Final selected feature count : {len(final_features)}")

sel_df = feature_scores.copy()
sel_df["Included"] = sel_df["Feature"].isin(final_features).astype(int)
sel_df.to_csv(os.path.join(OUT_DIR, "selected_features.csv"), index=False)
print("[GAME_PRUNER] Saved selected_features.csv")

with open(os.path.join(OUT_DIR, "selected_feature_list.txt"), "w", encoding="utf-8") as f:
    for feat in final_features:
        f.write(str(feat) + "\n")

print("[GAME_PRUNER] Saved selected_feature_list.txt")
print("[GAME_PRUNER] Pruning complete.")
