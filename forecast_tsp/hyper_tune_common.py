"""Shared utilities for hyperparameter tuning scripts.

Eliminates ~80 lines of duplicated code across 4 tuning scripts.
All scripts run from project root so paths are relative.
VMD cache is unified under outputs/vmd_cache/ to avoid redundant computation.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

# ══════════════════════════════════════════════
# Common constants
# ══════════════════════════════════════════════
SEQ_LEN = 120
BATCH_SIZE = 64
OUTPUT_DIM = 24
CAPACITY = 2000.0
WEATHER_DIM = 8
SEED = 42

DATA_CSV = "data/wind_nc/output/wind_data.csv"
VMD_CACHE_DIR = os.path.join("outputs", "vmd_cache")

# Weather feature columns (shared by all models)
WEATHER_COLS = [
    "power_current", "wind_speed_100m", "air_density",
    "u100", "v100", "t2m", "hour_sin", "hour_cos",
]


# ══════════════════════════════════════════════
# Power curve (Vestas V90-2.0MW)
# ══════════════════════════════════════════════
def power_curve_v90(v_hub):
    curve = np.array([
        [0, 0], [1, 0], [2, 0], [3, 0], [4, 35], [5, 80],
        [6, 150], [7, 260], [8, 410], [9, 610], [10, 870],
        [11, 1180], [12, 1540], [13, 1850], [14, 1970],
        [15, 2000], [16, 2000], [17, 2000], [18, 2000],
        [19, 2000], [20, 2000], [21, 2000], [22, 2000],
        [23, 2000], [24, 2000], [25, 2000], [26, 0],
    ], dtype=float)
    return np.interp(v_hub, curve[:, 0], curve[:, 1])


def wind_at_hub(v100, z_ref=100, z_hub=90, z0=0.03):
    return v100 * (np.log(z_hub / z0) / np.log(z_ref / z0))


def compute_power(v100, rho, rho_ref=1.225):
    return power_curve_v90(wind_at_hub(v100)) * (rho / rho_ref)


# ══════════════════════════════════════════════
# Data loading & preparation
# ══════════════════════════════════════════════
def load_wind_data(csv_path=None):
    """Load, clean, and prepare wind data.

    Returns a DataFrame with columns:
      - valid_time, point_id, weather features
      - power_t1..power_t24 (24h target labels)
      - hour_sin, hour_cos, power_current
    """
    if csv_path is None:
        csv_path = DATA_CSV
    df = pd.read_csv(csv_path)
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df = df[df["point_id"] == 1].sort_values("valid_time").reset_index(drop=True)

    # 24h target power labels
    target_cols = []
    for h in range(1, OUTPUT_DIM + 1):
        col = f"power_t{h}"
        ws_shifted = df["wind_speed_100m"].shift(-h).values
        rho_shifted = df["air_density"].shift(-h).values
        df[col] = compute_power(ws_shifted, rho_shifted)
        target_cols.append(col)
    df = df.dropna().reset_index(drop=True)

    # Time encoding
    hour = pd.DatetimeIndex(df["valid_time"]).hour
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    # Current power (same-timestamp, not a future label)
    df["power_current"] = compute_power(
        df["wind_speed_100m"].values, df["air_density"].values
    )

    return df


def get_sample_domain_masks(times):
    """Return per-sample boolean masks for train/val/test domains.

    These operate on the raw sample index space (length = len(times)).
    """
    years = pd.DatetimeIndex(times).year
    months = pd.DatetimeIndex(times).month
    train_mask = years.isin([2024, 2025]) & ~((years == 2025) & (months >= 10))
    val_mask   = (years == 2025) & (months >= 10)
    test_mask  = years == 2026
    return train_mask, val_mask, test_mask


def get_sequence_domain_masks(seq_times):
    """Return sequence-level boolean masks for train/val/test.

    These operate on the sequence index space (after sliding window).
    """
    years = pd.DatetimeIndex(seq_times).year
    months = pd.DatetimeIndex(seq_times).month
    train_mask = years.isin([2024, 2025]) & ~((years == 2025) & (months >= 10))
    val_mask   = (years == 2025) & (months >= 10)
    test_mask  = years == 2026
    return train_mask, val_mask, test_mask


def clean_path_b_param_defaults():
    """Return the best-known Path B hyperparameters after tuning."""
    return {
        "trend_hidden": 128,
        "fluct_hidden": 64,
        "n_layers": 1,
        "path_b_dropout": 0.444,
        "fc_hidden": 64,
    }


def clean_path_a_param_defaults():
    """Return the best-known Path A hyperparameters after tuning."""
    return {
        "conv1_filters": 128,
        "conv2_filters": 192,
        "conv_kernel": 3,
        "pool_size": 2,
        "cnn_lstm_hidden": 32,
        "cnn_lstm_layers": 1,
        "path_a_dropout": 0.14,
    }
