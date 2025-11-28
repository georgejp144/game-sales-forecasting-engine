# ============================================================
# Synthetic Game Sales Dataset Generator — v20
#
# PURPOSE:
#   Generate a fully deterministic synthetic game-sales universe
#   for AAA / AA / Indie titles including:
#       • Sales time series (52 weeks)
#       • Marketing decay curves
#       • DLC / Discount promo logic
#       • Seasonal factors & competitor intensity
#       • Proxy feature generation for new titles
#
# USED FOR:
#   - Training the weekly forecasting engine (prior + XGB + XSTL)
#   - Creating cold-start new game CSVs for:
#       • NeonRift (AAA)
#       • Ashbound (AA)
#       • Pulsebreak (Indie)
#
# INPUTS:
#   None (self-contained)
#
# OUTPUTS:
#   synthetic_data/synthetic_game_sales_timeseries.csv
#   synthetic_data/new_game_<Title>_<DevType>.csv
#
# FEATURES:
#   • Deterministic seed ledger for reproducibility
#   • AAA > AA > Indie calibrated marketing peaks
#   • Behaviour templates per dev type
#   • DLC + discount templates per dev type
#
# RUN:
#   01_synthetic_game_generator.py
#
# ============================================================


# ============================================================
# IMPORTS
# ============================================================

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import json


# ============================================================
# GLOBAL CONFIG
# ============================================================

OUT_DIR = "synthetic_data"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_TITLES = 200
WEEKS_AFTER_RELEASE = 52

SEED_LEDGER_PATH = os.path.join(OUT_DIR, "seed_ledger.json")

GENRES = ["RPG", "Platformer", "Racing", "Party", "Action", "Indie"]

FRANCHISES = [
    "Zelda", "Mario", "Pokemon", "FireEmblem",
    "Splatoon", "SmashBros", "IndieIP"
]

DEV_TYPES = ["AAA", "AA", "Indie"]

REGIONS = ["NA", "EU", "JP"]

PLATFORM_FIXED = "Switch"

# Deterministic, calibrated franchise strength index

FRANCHISE_STRENGTHS = {
    "Mario": 1.00,
    "Pokemon": 0.92,
    "Zelda": 0.88,
    "SmashBros": 0.70,
    "Splatoon": 0.55,
    "FireEmblem": 0.50,
    "IndieIP": 0.15,
}


# ============================================================
# PROMO TEMPLATES (DLC + DISCOUNT WEEKS PER DEV TYPE)
# ============================================================

AAA_PROMO_TEMPLATES = [
    {"dlc_weeks": [18, 30, 42], "discount_weeks": [22, 34]},
    {"dlc_weeks": [24, 36, 48], "discount_weeks": [28, 40]},
    {"dlc_weeks": [30, 42, 50], "discount_weeks": [32, 46]},
    {"dlc_weeks": [20, 35, 45], "discount_weeks": [26, 38]},
]

AA_PROMO_TEMPLATES = [
    {"dlc_weeks": [18, 34], "discount_weeks": [20, 28, 40]},
    {"dlc_weeks": [22, 36], "discount_weeks": [24, 32, 44]},
    {"dlc_weeks": [26, 38], "discount_weeks": [28, 36, 48]},
    {"dlc_weeks": [20, 40], "discount_weeks": [22, 35, 47]},
]

INDIE_PROMO_TEMPLATES = [
    {"dlc_weeks": [24], "discount_weeks": [26, 32, 40, 48]},
    {"dlc_weeks": [28], "discount_weeks": [22, 30, 36, 44]},
    {"dlc_weeks": [20], "discount_weeks": [24, 34, 42, 50]},
    {"dlc_weeks": [30], "discount_weeks": [28, 35, 45, 52]},
]


# ============================================================
# SEED LEDGER
# ============================================================

def load_or_create_seed_ledger(num_titles: int) -> dict:
    """
    Load deterministic seed ledger if it exists, otherwise create it.

    Each title_id gets a fixed seed
    for the three demonstration new games.
    """
    if os.path.exists(SEED_LEDGER_PATH):
        with open(SEED_LEDGER_PATH, "r") as f:
            return json.load(f)

    ledger = {}
    for tid in range(1, num_titles + 1):
        ledger[str(tid)] = tid

    ledger["901"] = 901
    ledger["902"] = 902
    ledger["903"] = 903

    with open(SEED_LEDGER_PATH, "w") as f:
        json.dump(ledger, f, indent=2)

    return ledger


SEED_LEDGER = load_or_create_seed_ledger(NUM_TITLES)


def seed_for_title(title_id: int):
    """
    Set NumPy seed based on the deterministic seed ledger entry
    for a given title_id.
    """
    sid = SEED_LEDGER[str(title_id)]
    np.random.seed(sid)


# ============================================================
# AAA / AA / INDIE BEHAVIOUR TEMPLATES
# ============================================================

def get_behaviour_template(dev_type: str) -> dict:
    """
    Return calibrated behavioural template for a given dev type.

    Controls:
        - Base price mean/std
        - Peak sales range
        - Decay rate range
        - DLC and discount patterns (counts, depth)
        - Marketing peak and decay ranges
    """
    if dev_type == "AAA":
        return {
            "base_price_mean": 64.99,
            "base_price_std": 5.0,
            "peak_sales_range": (300_000, 3_000_000),
            "decay_rate_range": (0.06, 0.09),
            "dlc_count_range": (3, 3),
            "discount_weeks_range": (2, 2),
            "discount_depth_range": (0.20, 0.40),
            "marketing_peak_range": (60, 120),
            "marketing_decay_range": (0.03, 0.06),
        }
    elif dev_type == "AA":
        return {
            "base_price_mean": 44.99,
            "base_price_std": 4.0,
            "peak_sales_range": (25_000, 200_000),
            "decay_rate_range": (0.09, 0.13),
            "dlc_count_range": (2, 2),
            "discount_weeks_range": (3, 3),
            "discount_depth_range": (0.15, 0.30),
            "marketing_peak_range": (45, 95),
            "marketing_decay_range": (0.05, 0.09),
        }
    else:
        return {
            "base_price_mean": 19.99,
            "base_price_std": 3.0,
            "peak_sales_range": (3_000, 40_000),
            "decay_rate_range": (0.13, 0.18),
            "dlc_count_range": (1, 1),
            "discount_weeks_range": (4, 4),
            "discount_depth_range": (0.10, 0.25),
            "marketing_peak_range": (25, 75),
            "marketing_decay_range": (0.06, 0.12),
        }


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def random_date(start: datetime, end: datetime) -> datetime:
    """Return a random datetime between start and end (inclusive)."""
    delta = end - start
    offset = np.random.randint(0, delta.days + 1)
    return start + timedelta(days=int(offset))


def compute_seasonal_factor(week_date: datetime) -> float:
    """
    Compute a simple seasonal factor:
        - Boost around Christmas, late November
        - Smaller boosts in spring, summer, January
    """
    m = week_date.month
    d = week_date.day
    if m == 12:
        return 1.4
    if m == 11 and d >= 20:
        return 1.25
    if (m == 4 and d >= 25) or (m == 5 and d <= 7):
        return 1.15
    if m in (7, 8):
        return 1.10
    if m == 1:
        return 1.05
    return 1.0


def days_to_christmas(date: datetime) -> int:
    """Return number of days until Christmas for the given date."""
    christmas = datetime(year=date.year, month=12, day=25)
    return (christmas - date).days


# ============================================================
# METADATA GENERATOR
# ============================================================

def generate_title_metadata(title_id: int,
                            forced_name: str | None = None,
                            forced_dev_type: str | None = None) -> dict:
    """
    Generate static metadata for a single title.

    Randomises:
        - Genre
        - Dev type (unless forced)
        - Franchise (within dev type)
        - Region
        - Decay class
        - Release date

    Also maps franchise to franchise_strength and fixes platform.
    """
    genre = np.random.choice(GENRES)
    dev_type = forced_dev_type if forced_dev_type else np.random.choice(
        DEV_TYPES, p=[0.3, 0.4, 0.3]
    )

    if dev_type == "AAA":
        franchise = np.random.choice(["Zelda", "Mario", "Pokemon"])
    elif dev_type == "AA":
        franchise = np.random.choice(["SmashBros", "Splatoon", "FireEmblem"])
    else:
        franchise = "IndieIP"

    region = np.random.choice(REGIONS)
    franchise_strength = FRANCHISE_STRENGTHS[franchise]

    if dev_type == "AAA":
        decay_class = "LongTail"
    elif dev_type == "AA":
        decay_class = "Standard"
    else:
        decay_class = "Fast"

    release_date = random_date(
        datetime(2020, 1, 1),
        datetime(2024, 12, 31)
    )

    title_name = forced_name if forced_name else f"Game_{title_id:03d}"

    return {
        "title_id": title_id,
        "title_name": title_name,
        "genre": genre,
        "franchise": franchise,
        "dev_type": dev_type,
        "region": region,
        "platform": PLATFORM_FIXED,
        "franchise_strength": franchise_strength,
        "decay_class": decay_class,
        "release_date": release_date,
    }


# ============================================================
# CORE TITLE TIMESERIES GENERATOR (WITH AAA>AA>INDIE)
# ============================================================

def generate_title_timeseries(meta: dict,
                              weeks_after_release: int = WEEKS_AFTER_RELEASE) -> pd.DataFrame:
    """
    Generate a 52-week synthetic sales time series for a single title.

    Includes:
        - Exponential decay around a peak sales parameter
        - Marketing decay curve (with AAA > AA > Indie adjustment)
        - DLC + discount boosts
        - Seasonal and competitive effects
        - Derived rolling and log-space features
    """
    template = get_behaviour_template(meta["dev_type"])
    is_new_game = bool(meta.get("is_new_game", False))

    # Base price with optional override
    if "base_price_override" in meta:
        base_price = float(meta["base_price_override"])
    else:
        base_price = max(
            4.99,
            np.random.normal(template["base_price_mean"], template["base_price_std"])
        )

    # Peak sales and decay rate with overrides / jitter
    peak_sales = np.random.uniform(*template["peak_sales_range"])
    if "peak_sales_override" in meta:
        peak_sales = float(meta["peak_sales_override"])

    decay_rate = np.random.uniform(*template["decay_rate_range"])
    if "decay_rate_override" in meta:
        decay_rate = float(meta["decay_rate_override"])
    else:
        decay_rate *= np.random.normal(1.0, 0.05)
        decay_rate = float(np.clip(
            decay_rate,
            template["decay_rate_range"][0],
            template["decay_rate_range"][1],
        ))

    # Marketing curve parameters
    marketing_peak = np.random.uniform(*template["marketing_peak_range"])
    marketing_decay_rate = np.random.uniform(*template["marketing_decay_range"])

    # ------------------------------------------------------------------
    # Force AAA > AA > Indie initial marketing levels
    # ------------------------------------------------------------------

    if meta["dev_type"] == "AAA":
        marketing_peak *= 1.30
    elif meta["dev_type"] == "AA":
        marketing_peak *= 1.00
    else:
        marketing_peak *= 0.70
    # ------------------------------------------------------------------

    # Deterministic selection of promo template
    template_index = SEED_LEDGER[str(meta["title_id"])] % 4

    if meta["dev_type"] == "AAA":
        promo = AAA_PROMO_TEMPLATES[template_index]
    elif meta["dev_type"] == "AA":
        promo = AA_PROMO_TEMPLATES[template_index]
    else:
        promo = INDIE_PROMO_TEMPLATES[template_index]

    dlc_weeks = promo["dlc_weeks"]
    discount_weeks = promo["discount_weeks"]

    # Optional overrides for DLC / discount weeks (for hand-crafted titles)
    if "fixed_dlc_weeks" in meta:
        dlc_weeks = meta["fixed_dlc_weeks"]

    if "fixed_discount_weeks" in meta:
        discount_weeks = meta["fixed_discount_weeks"]

    # DLC uplift by dev type
    if meta["dev_type"] == "AAA":
        dlc_uplift_value = 0.40
    elif meta["dev_type"] == "AA":
        dlc_uplift_value = 0.30
    else:
        dlc_uplift_value = 0.15

    rows = []
    last_dlc_week = None
    last_discount_week = None

    competitor_prev = float(np.clip(np.random.normal(50, 25), 0, 100))
    marketing_prev = None

    # ------------------------------------------------------------
    # Weekly time series generation
    # ------------------------------------------------------------

    for w in range(weeks_after_release):

        week_index = w
        week_start_date = meta["release_date"] + timedelta(days=7 * week_index)

        # Baseline exponential decay + noise
        baseline = peak_sales * np.exp(-decay_rate * week_index)
        noise = np.random.normal(0, baseline * 0.05)

        seasonal_factor = compute_seasonal_factor(week_start_date)

        # New-game tweak: extra seasonal jitter around launch
        if is_new_game:
            seasonal_factor *= (1.0 + np.random.normal(0.0, 0.03))
            if week_start_date.day >= 26 or week_start_date.day <= 3:
                seasonal_factor *= 1.04

        sales = max(0.0, baseline * seasonal_factor + noise)

        # Marketing index with exponential decay and noise
        marketing_index = marketing_peak * np.exp(-marketing_decay_rate * week_index)
        marketing_index += np.random.normal(0, marketing_peak * 0.05)
        marketing_index = max(0.0, marketing_index)

        # AR(1)-style smoothing for new-game marketing noise
        if is_new_game:
            base_level = marketing_index
            if base_level > 0:
                shock = np.random.normal(0.0, base_level * 0.08)
            else:
                shock = 0.0

            if marketing_prev is None:
                marketing_index = base_level + shock
            else:
                marketing_index = 0.75 * marketing_prev + 0.25 * (base_level + shock)

            marketing_index = max(0.0, marketing_index)
            marketing_prev = marketing_index

        # Competitor intensity (smoothed noise)
        comp_noise = float(np.clip(np.random.normal(50, 25), 0, 100))
        competitor_intensity = 0.7 * competitor_prev + 0.3 * comp_noise
        competitor_intensity = float(np.clip(competitor_intensity, 0, 100))
        competitor_prev = competitor_intensity

        # Extra competitor events and seasonal tweaks for new games
        if is_new_game:
            if np.random.rand() < 0.08:
                competitor_intensity += np.random.uniform(10, 30)

            if week_start_date.month in (10, 11, 12):
                competitor_intensity *= 1.10
            elif week_start_date.month in (6, 7):
                competitor_intensity *= 1.05

            competitor_intensity = float(np.clip(competitor_intensity, 0, 100))

        promo_multiplier = 1.0

        # DLC weeks
        dlc_flag = 1 if week_index in dlc_weeks else 0
        if dlc_flag:
            last_dlc_week = week_index
            promo_multiplier *= (1 + dlc_uplift_value)

        # Discount weeks
        discount_flag = 1 if week_index in discount_weeks else 0
        if discount_flag:
            last_discount_week = week_index

            if meta["dev_type"] == "AAA":
                discount_depth = 0.25
                discount_uplift = 0.12
            elif meta["dev_type"] == "AA":
                discount_depth = 0.30
                discount_uplift = 0.18
            else:
                discount_depth = 0.40
                discount_uplift = 0.25

            price = base_price * (1 - discount_depth)
            promo_multiplier *= (1 + discount_uplift)

        else:
            price = base_price
            discount_depth = 0.0

        # New-game marketing interplay with DLC / discounts / competitors
        if is_new_game:
            if dlc_flag:
                marketing_index *= 1.25

            if discount_flag:
                marketing_index *= 1.15

            if competitor_intensity > 0:
                marketing_index *= (1.0 - 0.10 * (competitor_intensity / 100.0))

            marketing_index *= (1.0 + 0.05 * (seasonal_factor - 1.0))

            marketing_index = max(0.0, marketing_index)

        # Apply promo multiplier to sales
        sales *= promo_multiplier

        # Days since last DLC
        if last_dlc_week is None:
            days_since_dlc = -1
        else:
            days_since_dlc = (week_index - last_dlc_week) * 7

        # Days since last discount
        if last_discount_week is None:
            days_since_last_discount = -1
        else:
            days_since_last_discount = (week_index - last_discount_week) * 7

        # Christmas / BF proximity and window flags
        dtc = days_to_christmas(week_start_date)
        in_christmas_window = int(-21 <= dtc <= 7)

        if week_start_date.month == 11:
            bf_date = datetime(week_start_date.year, 11, 27)
            bf_proximity = abs((week_start_date - bf_date).days)
        else:
            bf_proximity = 99

        # Hype score blending marketing and franchise strength
        hype_score = (
            0.6 * marketing_index +
            0.4 * (meta["franchise_strength"] * 100)
        ) / 2.0

        row = {
            "title_id": meta["title_id"],
            "title_name": meta["title_name"],
            "genre": meta["genre"],
            "franchise": meta["franchise"],
            "dev_type": meta["dev_type"],
            "region": meta["region"],
            "platform": meta["platform"],
            "franchise_strength": meta["franchise_strength"],
            "decay_class": meta["decay_class"],
            "release_date": meta["release_date"].date(),
            "week_index": week_index,
            "week_start_date": week_start_date.date(),
            "sales": float(sales),
            "price": float(round(price, 2)),
            "discount_flag": discount_flag,
            "dlc_flag": dlc_flag,
            "marketing_index": float(marketing_index),
            "competitor_intensity": float(competitor_intensity),
            "seasonal_factor": float(seasonal_factor),
            "decay_rate_k": float(decay_rate),
            "peak_sales_param": float(peak_sales),
            "days_since_last_discount": days_since_last_discount,
            "days_since_dlc": days_since_dlc,
            "days_to_christmas": dtc,
            "in_christmas_window": in_christmas_window,
            "black_friday_proximity": bf_proximity,
            "hype_score": float(hype_score),
            "marketing_decay_rate": float(marketing_decay_rate),
            "discount_depth": float(discount_depth),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by title and week index
    df = df.sort_values(["title_id", "week_index"]).reset_index(drop=True)

    # Rolling sales features (multi-horizon moving averages)
    for win, col_name in [(4, "sales_rolling_4w"),
                          (8, "sales_rolling_8w"),
                          (13, "sales_rolling_13w")]:

        df[col_name] = (
            df.groupby("title_id")["sales"]
            .transform(lambda s: s.rolling(win, min_periods=1).mean())
        )

    # Simple change features (level and log-level)
    df["sales_change_1w"] = df.groupby("title_id")["sales"].diff(1).fillna(0.0)
    df["sales_change_4w"] = df.groupby("title_id")["sales"].diff(4).fillna(0.0)

    df["log_sales"] = np.log1p(df["sales"])
    df["log_sales_change_1w"] = df.groupby("title_id")["log_sales"].diff(1).fillna(0.0)

    # Normalisation helpers
    def minmax(series):
        if series.max() == series.min():
            return series * 0.0
        return (series - series.min()) / (series.max() - series.min())

    df["marketing_norm"] = df.groupby("title_id")["marketing_index"].transform(minmax)
    df["competitor_norm"] = df.groupby("title_id")["competitor_intensity"].transform(minmax)

    # Interaction feature: price during discount weeks
    df["price_x_discount"] = df["price"] * df["discount_flag"]

    return df


# ============================================================
# PROXY FEATURES FOR NEW GAMES
# ============================================================

def apply_proxy_sales_features_for_new_game(df: pd.DataFrame) -> pd.DataFrame:
    """
    For new titles, construct proxy sales-based features from the
    modelled peak_sales_param / decay / seasonality, without
    revealing true sales (sets sales to NaN).
    """
    df = df.copy()

    proxy_baseline = df["peak_sales_param"] * np.exp(-df["decay_rate_k"] * df["week_index"])
    proxy_baseline = proxy_baseline * df["seasonal_factor"]

    n = len(df)
    ar_noise = np.zeros(n)
    if n > 0:
        ar_noise[0] = np.random.normal(0.0, 0.06)
        for t in range(1, n):
            ar_noise[t] = 0.75 * ar_noise[t - 1] + np.random.normal(0.0, 0.03)

    noise_factor = 1.0 + ar_noise
    proxy_sales = proxy_baseline * noise_factor
    proxy_sales = np.maximum(proxy_sales, 0.0)

    # Rolling features based on proxy sales
    df["sales_rolling_4w"] = pd.Series(proxy_sales).rolling(4, min_periods=1).mean().values
    df["sales_rolling_8w"] = pd.Series(proxy_sales).rolling(8, min_periods=1).mean().values
    df["sales_rolling_13w"] = pd.Series(proxy_sales).rolling(13, min_periods=1).mean().values

    df["sales_change_1w"] = pd.Series(proxy_sales).diff(1).fillna(0.0).values
    df["sales_change_4w"] = pd.Series(proxy_sales).diff(4).fillna(0.0).values

    proxy_log = np.log1p(proxy_sales)
    df["log_sales"] = proxy_log
    df["log_sales_change_1w"] = pd.Series(proxy_log).diff(1).fillna(0.0).values

    # Hide actual sales for new titles
    df["sales"] = np.nan
    return df


# ============================================================
# TRAINING DATA GENERATOR
# ============================================================

def generate_training_dataset(num_titles: int = NUM_TITLES) -> pd.DataFrame:
    """
    Generate the full training dataset across NUM_TITLES titles.

    Uses deterministic seeding per title so the full dataset is
    reproducible across runs.
    """
    all_frames = []
    for tid in range(1, num_titles + 1):
        seed_for_title(tid)
        meta = generate_title_metadata(title_id=tid)
        df_title = generate_title_timeseries(meta)
        all_frames.append(df_title)
    return pd.concat(all_frames, ignore_index=True)


# ============================================================
# NEW GAME GENERATOR
# ============================================================

def generate_new_game(title_name: str,
                      dev_type: str,
                      title_id: int) -> pd.DataFrame:
    """
    Generate a single new-game dataset for a named title and dev type.

    This uses:
        - Deterministic seed per title_id
        - Forced dev_type and title_name
        - Hand-crafted franchise, peak_sales, decay, DLC/discount weeks
        - Proxy feature generation to simulate history without true sales
    """
    seed_for_title(title_id)

    meta = generate_title_metadata(
        title_id=title_id,
        forced_name=title_name,
        forced_dev_type=dev_type,
    )

    meta["is_new_game"] = True

    # Map each named title to a franchise
    if title_name == "NeonRift":
        meta["franchise"] = "Zelda"
    elif title_name == "Ashbound":
        meta["franchise"] = "SmashBros"
    elif title_name == "Pulsebreak":
        meta["franchise"] = "IndieIP"

    meta["franchise_strength"] = FRANCHISE_STRENGTHS[meta["franchise"]]
    meta["region"] = "EU"
    meta["release_date"] = datetime(2025, 12, 1)

    # Hand-picked peak sales and decay parameters per title
    if title_name == "NeonRift":
        meta["peak_sales_override"] = 1_450_000
        meta["decay_rate_override"] = 0.087425

    elif title_name == "Ashbound":
        meta["peak_sales_override"] = 95_000
        meta["decay_rate_override"] = 0.128172

    elif title_name == "Pulsebreak":
        meta["peak_sales_override"] = 14_300
        meta["decay_rate_override"] = 0.163956

    # Hand-picked DLC and discount schedules per title
    if title_name == "NeonRift":
        meta["fixed_dlc_weeks"] = [20, 32, 44]
        meta["fixed_discount_weeks"] = [26, 40]

    elif title_name == "Ashbound":
        meta["fixed_dlc_weeks"] = [22, 35]
        meta["fixed_discount_weeks"] = [28, 39, 48]

    elif title_name == "Pulsebreak":
        meta["fixed_dlc_weeks"] = [24]
        meta["fixed_discount_weeks"] = [30, 36, 42, 48]

    df_new = generate_title_timeseries(meta)
    df_new = apply_proxy_sales_features_for_new_game(df_new)
    return df_new


# ============================================================
# DUMMY ENCODING
# ============================================================

def apply_consistent_dummies(df: pd.DataFrame, categories: dict) -> pd.DataFrame:
    """
    Apply consistent one-hot encoding for all categorical columns,
    ensuring new-game files share the same dummy columns as the
    training dataset.
    """
    for col, all_values in categories.items():
        for val in all_values:
            colname = f"{col}_{val}"
            df[colname] = (df[col] == val).astype(int)
    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print("[GEN] Seed Ledger loaded → deterministic generation enabled.")
    print(f"[GEN] Ledger path: {SEED_LEDGER_PATH}")

    # --------------------------------------------------------
    # 1) Generate full training dataset
    # --------------------------------------------------------

    print("[GEN] Generating training dataset...")
    full_df = generate_training_dataset(NUM_TITLES)

    category_sets = {
        "genre": sorted(full_df["genre"].unique()),
        "franchise": sorted(full_df["franchise"].unique()),
        "dev_type": sorted(full_df["dev_type"].unique()),
        "region": sorted(full_df["region"].unique()),
    }

    full_df = apply_consistent_dummies(full_df, category_sets)
    cols_order = list(full_df.columns)

    train_path = os.path.join(OUT_DIR, "synthetic_game_sales_timeseries.csv")
    full_df.to_csv(train_path, index=False)
    print(f"[GEN] Saved training dataset: {train_path} | rows={len(full_df)}")

    # --------------------------------------------------------
    # 2) Generate new AAA game: NeonRift
    # --------------------------------------------------------

    print("[GEN] Generating new AAA game: NeonRift...")
    neon_df = generate_new_game("NeonRift", "AAA", 901)
    neon_df = apply_consistent_dummies(neon_df, category_sets)
    neon_df = neon_df.reindex(columns=cols_order, fill_value=0)
    neon_path = os.path.join(OUT_DIR, "new_game_NeonRift_AAA.csv")
    neon_df.to_csv(neon_path, index=False)

    # --------------------------------------------------------
    # 3) Generate new AA game: Ashbound
    # --------------------------------------------------------

    print("[GEN] Generating new AA game: Ashbound...")
    ash_df = generate_new_game("Ashbound", "AA", 902)
    ash_df = apply_consistent_dummies(ash_df, category_sets)
    ash_df = ash_df.reindex(columns=cols_order, fill_value=0)
    ash_path = os.path.join(OUT_DIR, "new_game_Ashbound_AA.csv")
    ash_df.to_csv(ash_path, index=False)

    # --------------------------------------------------------
    # 4) Generate new Indie game: Pulsebreak
    # --------------------------------------------------------

    print("[GEN] Generating new Indie game: Pulsebreak...")
    pulse_df = generate_new_game("Pulsebreak", "Indie", 903)
    pulse_df = apply_consistent_dummies(pulse_df, category_sets)
    pulse_df = pulse_df.reindex(columns=cols_order, fill_value=0)
    pulse_path = os.path.join(OUT_DIR, "new_game_Pulsebreak_Indie.csv")
    pulse_df.to_csv(pulse_path, index=False)

    print("[GEN] Done — fully deterministic synthetic game universe generated "
          "(v17, high-reliability peaks + new-game noise).")
