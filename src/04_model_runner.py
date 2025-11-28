# ============================================================
# Weekly Game Sales Forecasting Engine — v23
#
# PURPOSE:
#   Forecast weekly game sales for new AAA / AA / Indie titles
#   using:
#       • Prior decay curve (real sales space)
#       • Log-residual XGBoost
#       • XSTL cross-sectional similarity layer
#       • Bayesian hyperparameter search
#
#
# USED FOR:
#   - Generating weekly forecasts for new titles
#   - Producing reliability-weighted blends (prior vs XGB)
#   - Creating promo-safe trajectories with:
#       • Discount spikes + echoes
#       • DLC spikes + multi-week echoes
#
# INPUTS:
#   TRAIN_PATH:
#       synthetic_data/synthetic_game_sales_timeseries.csv
#
#   NEW_GAME_FILES:
#       synthetic_data/new_game_NeonRift_AAA.csv
#       synthetic_data/new_game_Ashbound_AA.csv
#       synthetic_data/new_game_Pulsebreak_Indie.csv
#
#   NEW_GAME_PATH_LEGACY:
#       synthetic_data/new_game_prediction_rows.csv
#
#   FEATURE_LIST_PATH:
#       synthetic_data/feature_pruner/selected_feature_list.txt
#
# OUTPUTS:
#   new_games_sales_forecast_detailed.csv
#       • Per-title, per-week forecasts including:
#           - prior, XGB raw, promo-safe override
#           - reliability metrics (Regime / Drift / XSTL)
#           - final_pred_blended, P10 / P90 bands
#           - baseline_no_marketing, marketing_uplift
#
#   new_games_sales_summary.csv
#       • Per-title rollup:
#           - total_forecast, total_p10, total_p90
#           - total_baseline_no_marketing, total_marketing_uplift
#           - avg_price, avg_reliability
#           - peak_week_index / peak_week_start_date / peak_week_forecast
#
#   new_games_sales_metadata.csv
#       • Pure metadata per game-week (stacked):
#           - title_name, dev_type, franchise, region, platform
#           - franchise_strength, decay_class, decay_rate_k
#           - week_start_date, week_index, price
#           - discount_flag, dlc_flag, marketing_index
#
# RUN:
#   04_model_runner.py
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import os
import numpy as np
import pandas as pd
from datetime import datetime

from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import make_scorer

# skopt optional
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer
    HAS_SKOPT = True
except Exception:
    HAS_SKOPT = False
    print("[RUNNER] skopt missing → using fixed XGB params")


# ============================================================
# CONFIG
# ============================================================

TRAIN_PATH = "synthetic_data/synthetic_game_sales_timeseries.csv"

NEW_GAME_FILES = [
    "synthetic_data/new_game_NeonRift_AAA.csv",
    "synthetic_data/new_game_Ashbound_AA.csv",
    "synthetic_data/new_game_Pulsebreak_Indie.csv",
]

NEW_GAME_PATH_LEGACY = "synthetic_data/new_game_prediction_rows.csv"
FEATURE_LIST_PATH = "synthetic_data/feature_pruner/selected_feature_list.txt"

OUT_DIR = "sales_model_runs"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "sales_target"

BASE_XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.7,
    min_child_weight=3,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    n_jobs=1,
)

DRIFT_SCALE = 3.0
REGIME_SCALE = 2.5
XSTL_SCALE = 4.0

REGIME_FEATURE_CANDIDATES = [
    "week_index_norm",
    "marketing_index",
    "competitor_intensity",
    "seasonal_factor",
    "price",
]


# ============================================================
# UTILITIES
# ============================================================

def _describe_utilities():
    """
    Documentation block for utility functions in this script.

    Functions documented:
        • load_selected_features
        • add_time_features
        • ensure_time_features_in_feature_list
        • build_sales_target

    These helpers perform:
        - Feature list loading
        - Time-based engineered features
        - Enforcing essential features
        - log1p(sales) target creation

    This block has no effect on runtime.
    """
    return None


def load_selected_features(path: str):
    """Load the pruned feature list from disk."""
    with open(path, "r", encoding="utf-8") as f:
        return [x.strip() for x in f if x.strip()]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add normalised and log time features from week_index."""
    df = df.copy()
    if "week_index" in df.columns:
        w = df["week_index"].astype(float)
        max_w = max(w.max(), 1.0)
        df["week_index_norm"] = w / max_w
        df["week_index_sq"] = df["week_index_norm"] ** 2
        df["log_week_index"] = np.log1p(w)
    return df


def ensure_time_features_in_feature_list(features, columns):
    """Ensure time-based helper features are not dropped by the pruner."""
    for col in ["week_index_norm", "week_index_sq", "log_week_index"]:
        if col in columns and col not in features:
            features.append(col)
    return features


def build_sales_target(df: pd.DataFrame) -> pd.DataFrame:
    """Create log1p(sales) target column."""
    df = df.copy()
    df[TARGET] = np.log1p(df["sales"].astype(float))
    return df

# ============================================================
# PRIOR CURVE (REAL SPACE)
# ============================================================

def build_decay_prior(df: pd.DataFrame) -> np.ndarray:
    """
    Build the real-space prior curve using peak_sales_param and decay_rate_k,
    with Indie flattening and promo/DLC uplifts.
    """
    df_local = df.copy()

    peak = df_local["peak_sales_param"].values.astype(float)
    k = df_local["decay_rate_k"].values.astype(float)

    # Indie flattening in late tail
    if "dev_type" in df_local.columns:
        is_indie = (df_local["dev_type"] == "Indie").values
        w_raw = df_local["week_index"].values.astype(float)
        k = np.where(is_indie & (w_raw >= 20), k * 0.6, k)

    w = df_local["week_index"].values.astype(float)
    prior = peak * np.exp(-k * w)

    discount_flag = df_local.get(
        "discount_flag", pd.Series(0, index=df_local.index)
    ).astype(int).values
    dlc_flag = df_local.get(
        "dlc_flag", pd.Series(0, index=df_local.index)
    ).astype(int).values
    dev_type = df_local.get(
        "dev_type", pd.Series("Unknown", index=df_local.index)
    ).astype(str)

    if discount_flag.sum() == 0 and dlc_flag.sum() == 0:
        return prior

    def get_discount_uplift(dt: str) -> float:
        return {"AAA": 1.25, "AA": 1.15, "Indie": 1.08}.get(dt, 1.12)

    def get_dlc_uplift(dt: str) -> float:
        return {"AAA": 1.40, "AA": 1.30, "Indie": 1.18}.get(dt, 1.25)

    week_index = df_local["week_index"].values.astype(int)

    if "title_id" in df_local.columns:
        groups = df_local.groupby("title_id").indices
    else:
        groups = {"_all": np.arange(len(df_local))}

    prior_spiked = prior.copy()

    for _, idxs in groups.items():
        idxs = np.array(idxs, dtype=int)
        idxs = idxs[np.argsort(week_index[idxs])]
        n = len(idxs)

        for pos, i in enumerate(idxs):
            dt = dev_type.iloc[i]

            # Discount spike + echo
            if discount_flag[i] == 1:
                u = get_discount_uplift(dt)
                prior_spiked[i] *= u
                if pos + 1 < n:
                    prior_spiked[idxs[pos + 1]] *= (1 + (u - 1) * 0.4)

            # DLC spike + 2-week echo
            if dlc_flag[i] == 1:
                u = get_dlc_uplift(dt)
                prior_spiked[i] *= u
                if pos + 1 < n:
                    prior_spiked[idxs[pos + 1]] *= (1 + (u - 1) * 0.6)
                if pos + 2 < n:
                    prior_spiked[idxs[pos + 2]] *= (1 + (u - 1) * 0.3)

    return prior_spiked


# ============================================================
# LOG-RESIDUAL TARGET
# ============================================================

def build_residual_target(df: pd.DataFrame, prior: np.ndarray) -> np.ndarray:
    """
    Build log-residual target with promo/dlc upper bounds.
    y_log - log1p(prior) with clipping for extreme promo behaviour.
    """
    y_log = df[TARGET].values.astype(float)
    n = len(df)

    prior_log = np.log1p(prior)

    base_upper = np.log(1.40)
    base_lower = np.log(0.80)

    upper = np.full(n, base_upper)
    lower = np.full(n, base_lower)

    disc_main = np.log(1.60)
    disc_echo1 = np.log(1.40)

    dlc_main = np.log(1.80)
    dlc_echo1 = np.log(1.50)
    dlc_echo2 = np.log(1.25)

    discount_flag = df["discount_flag"].astype(int).values
    dlc_flag = df["dlc_flag"].astype(int).values
    w = df["week_index"].values

    if "title_id" in df.columns:
        groups = df.groupby("title_id").indices
    else:
        groups = {"_all": np.arange(n)}

    for _, idxs in groups.items():
        idxs = np.array(idxs)
        idxs = idxs[np.argsort(w[idxs])]
        n_local = len(idxs)

        for pos, i in enumerate(idxs):

            if discount_flag[i] == 1:
                upper[i] = max(upper[i], disc_main)
                if pos + 1 < n_local:
                    upper[idxs[pos + 1]] = max(
                        upper[idxs[pos + 1]], disc_echo1
                    )

            if dlc_flag[i] == 1:
                upper[i] = max(upper[i], dlc_main)
                if pos + 1 < n_local:
                    upper[idxs[pos + 1]] = max(
                        upper[idxs[pos + 1]], dlc_echo1
                    )
                if pos + 2 < n_local:
                    upper[idxs[pos + 2]] = max(
                        upper[idxs[pos + 2]], dlc_echo2
                    )

    resid = y_log - prior_log
    return np.clip(resid, lower, upper)


# ============================================================
# XSTL
# ============================================================

def mahalanobis_distance_matrix(X_train, X_new, eps: float = 1e-6) -> np.ndarray:
    X_train = np.asarray(X_train, float)
    X_new = np.asarray(X_new, float)

    mean = np.nanmean(X_train, axis=0)
    var = np.nanvar(X_train, axis=0, ddof=1)
    var = np.where(var <= 0, eps, var)

    delta = X_new - mean
    return np.sqrt(np.sum((delta ** 2) / var, axis=1))


def add_xstl_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cross-sectional statistics over week_index, dev_type and franchise."""
    df = df.copy()

    g = df.groupby("week_index")["sales"]
    df["xstl_sales_mean"] = g.transform("mean")
    df["xstl_sales_median"] = g.transform("median")
    df["xstl_sales_std"] = g.transform("std").fillna(0)

    g2 = df.groupby(["dev_type", "week_index"])["sales"]
    df["xstl_dev_mean"] = g2.transform("mean")
    df["xstl_dev_std"] = g2.transform("std").fillna(0)

    if "franchise" in df.columns:
        g3 = df.groupby(["franchise", "week_index"])["sales"]
        df["xstl_franchise_mean"] = g3.transform("mean")
        df["xstl_franchise_std"] = g3.transform("std").fillna(0)

    return df.fillna(0)


def compute_xstl_similarity(X_train: np.ndarray, X_new: np.ndarray) -> np.ndarray:
    dist = mahalanobis_distance_matrix(X_train, X_new)
    return np.exp(-dist / XSTL_SCALE)


# ============================================================
# REAL-SPACE SMAPE
# ============================================================

def smape_real(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    return 100 * np.mean(
        2 * np.abs(y_pred - y_true) / (np.abs(y_pred) + np.abs(y_true) + 1e-9)
    )

# ============================================================
# MAIN
# ============================================================

def main():

    # ---------------- LOAD TRAIN ----------------
    df = pd.read_csv(TRAIN_PATH)
    df = build_sales_target(df)

    if "week_start_date" in df.columns:
        df["week_start_date"] = pd.to_datetime(
            df["week_start_date"],
            format="mixed",
            errors="coerce",
        )

    df = df.sort_values(
        ["week_start_date", "title_id", "week_index"]
    ).reset_index(drop=True)

    df = add_time_features(df)
    df = add_xstl_features(df)

    # ---------------- FEATURES ----------------
    features = load_selected_features(FEATURE_LIST_PATH)
    features = ensure_time_features_in_feature_list(features, df.columns)

    extra_cols = [
        "xstl_sales_mean", "xstl_sales_median", "xstl_sales_std",
        "xstl_dev_mean", "xstl_dev_std",
        "xstl_franchise_mean", "xstl_franchise_std",
    ]
    for col in extra_cols:
        if col in df.columns and col not in features:
            features.append(col)

    df[features] = df[features].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    X_tab = df[features].values
    sales_real = df["sales"].values.astype(float)
    y_log = df[TARGET].values.astype(float)

    # ---------------- PRIOR + RESID ----------------
    prior_pred = build_decay_prior(df)
    y_resid = build_residual_target(df, prior_pred)

    # ---------------- TRAIN XGB ----------------
    if HAS_SKOPT:

        xgb_base = XGBRegressor(
            tree_method="hist",
            random_state=42,
            n_jobs=1,
            objective="reg:squarederror",
        )

        search_spaces = {
            "n_estimators": Integer(300, 900),
            "max_depth": Integer(3, 6),
            "learning_rate": Real(0.01, 0.10, prior="log-uniform"),
            "subsample": Real(0.7, 1.0),
            "colsample_bytree": Real(0.5, 1.0),
            "min_child_weight": Integer(1, 6),
            "reg_lambda": Real(0.5, 5.0, prior="log-uniform"),
            "reg_alpha": Real(0.0, 1.0),
        }

        opt = BayesSearchCV(
            estimator=xgb_base,
            search_spaces=search_spaces,
            n_iter=30,
            cv=TimeSeriesSplit(n_splits=5),
            scoring=make_scorer(smape_real, greater_is_better=False),
            refit=True,
            random_state=42,
            verbose=0,
        )

        opt.fit(X_tab, sales_real)
        xgb_final = opt.best_estimator_
        model_cv_smape = -opt.best_score_

    else:
        tscv = TimeSeriesSplit(n_splits=5)
        smapes = []

        for train_idx, val_idx in tscv.split(X_tab):

            X_tr, X_val = X_tab[train_idx], X_tab[val_idx]
            prior_tr = prior_pred[train_idx]
            prior_val = prior_pred[val_idx]
            y_log_tr = y_log[train_idx]
            sales_val_real = sales_real[val_idx]

            prior_log_tr = np.log1p(prior_tr)
            y_resid_tr = y_log_tr - prior_log_tr

            model = XGBRegressor(**BASE_XGB_PARAMS)
            model.fit(X_tr, y_resid_tr)

            resid_pred_val = model.predict(X_val)
            prior_val_log = np.log1p(prior_val)
            pred_val_real = np.expm1(prior_val_log + resid_pred_val)

            smapes.append(smape_real(sales_val_real, pred_val_real))

        model_cv_smape = float(np.mean(smapes))
        xgb_final = XGBRegressor(**BASE_XGB_PARAMS)
        xgb_final.fit(X_tab, y_resid)

    # ---------------- LOAD NEW GAMES ----------------
    df_new_list = []
    for path in NEW_GAME_FILES:
        if os.path.exists(path):
            tmp = pd.read_csv(path)
            tmp["source_file"] = os.path.basename(path)
            df_new_list.append(tmp)

    if (not df_new_list) and os.path.exists(NEW_GAME_PATH_LEGACY):
        tmp = pd.read_csv(NEW_GAME_PATH_LEGACY)
        tmp["source_file"] = os.path.basename(NEW_GAME_PATH_LEGACY)
        df_new_list.append(tmp)

    if not df_new_list:
        raise FileNotFoundError("No new game inputs found.")

    df_new = pd.concat(df_new_list, ignore_index=True)

    if "week_start_date" in df_new.columns:
        df_new["week_start_date"] = pd.to_datetime(
            df_new["week_start_date"],
            format="mixed",
            errors="coerce",
        )

    df_new = df_new.sort_values(
        ["title_name", "week_start_date", "week_index"]
    ).reset_index(drop=True)
    df_new = add_time_features(df_new)

    # Add XSTL via concat so new games get cross-sectional stats
    full_tmp = pd.concat([df, df_new], ignore_index=True)
    full_tmp = add_xstl_features(full_tmp)

    df = full_tmp.iloc[: len(df)]
    df_new = full_tmp.iloc[len(df):].reset_index(drop=True)

    df_new[features] = df_new[features].apply(
        pd.to_numeric, errors="coerce"
    ).fillna(0)

    X_new = df_new[features].values

    # ---------------- BASE PRED ----------------
    prior_new = build_decay_prior(df_new)
    prior_new_log = np.log1p(prior_new)

    resid_xgb_new = xgb_final.predict(X_new)
    raw_pred_log = prior_new_log + resid_xgb_new
    raw_pred = np.expm1(raw_pred_log)

    xgb_raw_pred_no_blend = raw_pred.copy()
    xgb_residual_real = raw_pred - prior_new

    # ---------------- PROMO SAFETY ----------------
    promo_mask = (
        (df_new["discount_flag"] == 1) |
        (df_new["dlc_flag"] == 1)
    ).values

    shift1 = np.roll(promo_mask, 1)
    shift2 = np.roll(promo_mask, 2)
    shift1[0] = False
    shift2[:2] = False

    promo_or_echo = promo_mask | shift1 | shift2

    raw_pred_after_promo = raw_pred.copy()
    raw_pred_after_promo[promo_or_echo] = np.maximum(
        prior_new[promo_or_echo],
        raw_pred[promo_or_echo],
    )

    # ============================================================
    # RELIABILITY BLOCK (for dynamic blending)
    # ============================================================

    # Drift distance in feature space
    drift_dist = mahalanobis_distance_matrix(X_tab, X_new)
    raw_drift = np.exp(-drift_dist / DRIFT_SCALE)

    # Regime distance on a smaller regime feature subset
    regime_cols = [c for c in REGIME_FEATURE_CANDIDATES if c in df.columns]
    if len(regime_cols) >= 2:
        regime_dist = mahalanobis_distance_matrix(
            df[regime_cols].values,
            df_new[regime_cols].values,
        )
    else:
        regime_dist = drift_dist.copy()

    raw_regime = np.exp(-regime_dist / REGIME_SCALE)
    raw_xstl = compute_xstl_similarity(X_tab, X_new)

    def scale_to_unit(x, raw_max: float = 0.6) -> np.ndarray:
        """Simple 0–1 scaling with upper cap."""
        return np.clip(x / raw_max, 0.0, 1.0)

    Regime_Conf_Scaled = scale_to_unit(raw_regime)
    Drift_Score_Scaled = scale_to_unit(raw_drift)
    XSTL_Similarity_Scaled = scale_to_unit(raw_xstl)

    # Geometric-style combination of 3 reliability components
    Model_Reliability_Score = (
        Regime_Conf_Scaled *
        Drift_Score_Scaled *
        XSTL_Similarity_Scaled
    ) ** (1.0 / 3.0)

    # ============================================================
    # DYNAMIC RELIABILITY-WEIGHTED BLEND (MIN XGB WEIGHT = 0.30)
    # ============================================================

    rel = Model_Reliability_Score
    # Smooth reliability slightly toward a baseline
    rel_smoothed = 0.7 * rel + 0.21  # ~[0.21, 0.91]

    # XGB weight: at least 0.30
    xgb_weight_in_blend = np.maximum(0.30, rel_smoothed)
    prior_weight_in_blend = 1.0 - xgb_weight_in_blend

    # Reliability-weighted blend between promo-safe XGB and prior
    raw_pred_blended = (
        xgb_weight_in_blend * raw_pred_after_promo +
        prior_weight_in_blend * prior_new
    )

    # Safety tweak for very small priors
    raw_pred_blended = np.where(
        prior_new < 50,
        prior_new * 1.10,
        raw_pred_blended,
    )

    # Final blended prediction in real space
    blend_pred = np.maximum(raw_pred_blended, prior_new * 0.30)
    blend_pred = np.maximum(blend_pred, 1.0)

    # Blended predictive distribution
    blend_p10 = blend_pred * 0.85
    blend_p90 = blend_pred * 1.15
    blend_spread = blend_p90 - blend_p10

    # ============================================================
    # OUTPUT + BASELINE FIELDS
    # ============================================================

    detailed = pd.DataFrame({
        "title_name": df_new["title_name"],
        "dev_type": df_new["dev_type"],
        # (Metadata is kept separate in a dedicated metadata CSV)
        "week_start_date": df_new["week_start_date"],
        "week_index": df_new["week_index"],
        "price": df_new["price"],
        "discount_flag": df_new["discount_flag"],
        "dlc_flag": df_new["dlc_flag"],
        "marketing_index": df_new["marketing_index"],

        "pred_prior": prior_new,

        # Diagnostics
        "xgb_residual_pred": xgb_residual_real,
        "xgb_raw_pred_no_blend": xgb_raw_pred_no_blend,
        "xgb_after_promo_override": raw_pred_after_promo,

        # Final blended forecast (reliability-weighted)
        "final_pred_blended": blend_pred,

        # XGB weight used in the blend
        "xgb_weight_in_blend": xgb_weight_in_blend,

        # Blended predictive distribution
        "blend_p10": blend_p10,
        "blend_p90": blend_p90,
        "blend_spread": blend_spread,

        # Scaled metrics
        "Regime_Conf_Scaled": Regime_Conf_Scaled,
        "Drift_Score_Scaled": Drift_Score_Scaled,
        "XSTL_Similarity_Scaled": XSTL_Similarity_Scaled,

        # Reliability
        "Model_Reliability_Score": Model_Reliability_Score,

        # Constant per model (CV SMAPE)
        "Model_SMAPE": np.full(len(df_new), model_cv_smape),
    })

    # --- baseline_no_marketing + marketing_uplift (safe post-hoc calc) ---

    mi = pd.to_numeric(detailed["marketing_index"], errors="coerce").fillna(0.0)
    mi_clipped = mi.clip(lower=0.0, upper=120.0)

    # Normalised marketing intensity: ~0–1.5
    mi_norm = mi_clipped / 80.0
    mi_norm = mi_norm.clip(lower=0.0, upper=1.5)

    # Baseline factor:
    #   - low marketing → baseline close to final (e.g. 0.9x)
    #   - high marketing → baseline further below final (e.g. ~0.65x)
    baseline_factor = 0.95 - 0.25 * mi_norm
    baseline_factor = baseline_factor.clip(lower=0.65, upper=0.95)

    detailed["baseline_no_marketing"] = (
        detailed["final_pred_blended"].astype(float) * baseline_factor
    )

    detailed["marketing_uplift"] = (
        detailed["final_pred_blended"].astype(float)
        - detailed["baseline_no_marketing"].astype(float)
    )

    # ------------------------------------------------------------
    # SORT + SAVE DETAILED
    # ------------------------------------------------------------

    dev_order = {"AAA": 0, "AA": 1, "Indie": 2}
    detailed["_sort"] = detailed["dev_type"].map(dev_order).fillna(99)
    detailed = detailed.sort_values(
        ["_sort", "title_name", "week_index"]
    ).drop(columns=["_sort"])

    detailed_path = os.path.join(
        OUT_DIR, "new_games_sales_forecast_detailed.csv"
    )
    detailed.to_csv(detailed_path, index=False)
    print("[RUNNER] Saved:", detailed_path)

    # ------------------------------------------------------------
    # SUMMARY (includes baseline + uplift totals)
    # ------------------------------------------------------------

    summary = detailed.groupby("title_name").agg(
        total_forecast=("final_pred_blended", "sum"),
        total_p10=("blend_p10", "sum"),
        total_p90=("blend_p90", "sum"),
        total_baseline_no_marketing=("baseline_no_marketing", "sum"),
        total_marketing_uplift=("marketing_uplift", "sum"),
        avg_price=("price", "mean"),
        avg_reliability=("Model_Reliability_Score", "mean"),
    ).reset_index()

    # Peak week (based on blended forecast)
    idx_peak = detailed.groupby("title_name")["final_pred_blended"].idxmax()
    peak_rows = detailed.loc[idx_peak, [
        "title_name",
        "week_index",
        "week_start_date",
        "final_pred_blended",
    ]].rename(columns={
        "week_index": "peak_week_index",
        "week_start_date": "peak_week_start_date",
        "final_pred_blended": "peak_week_forecast",
    })

    summary = summary.merge(peak_rows, on="title_name", how="left")

    summary_path = os.path.join(
        OUT_DIR, "new_games_sales_summary.csv"
    )
    summary.to_csv(summary_path, index=False)
    print("[RUNNER] Saved:", summary_path)

    # ------------------------------------------------------------
    # NEW: METADATA CSV OUTPUT (PER YOUR REQUEST)
    # ------------------------------------------------------------

    meta_cols = [
        "title_name",
        "dev_type",
        "franchise",
        "region",
        "platform",
        "franchise_strength",
        "decay_class",
        "decay_rate_k",
        "week_start_date",
        "week_index",
        "price",
        "discount_flag",
        "dlc_flag",
        "marketing_index",
    ]

    meta_cols_existing = [c for c in meta_cols if c in df_new.columns]

    metadata = df_new[meta_cols_existing].copy()

    metadata_path = os.path.join(
        OUT_DIR, "new_games_sales_metadata.csv"
    )
    metadata.to_csv(metadata_path, index=False)
    print("[RUNNER] Saved:", metadata_path)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
