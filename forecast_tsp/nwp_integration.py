"""NWP (Numerical Weather Prediction) feature integration module.

Provides:
  - Horizon-aware cyclic features (deterministic, always available)
  - NWP forecast feature builder (when real forecast data is available)
  - ECMWF forecast download script (via CDS API)

Architecture:
  Without NWP data → fallback to cyclic features (hour_sin/cos per horizon)
  With NWP data   → real forecast wind/temp/pressure for each future hour
"""

import numpy as np
import pandas as pd
from datetime import datetime

import torch


def build_horizon_cyclic_features(times, vmd_out, cnn_out):
    """Build per-horizon cyclic features for Path B (deterministic).

    For each future hour, computes hour_sin/cos so the model knows
    what time of day it is predicting for.

    Parameters
    ----------
    times   : (N,) array of datetime-like
        Base timestamps for each sample.
    vmd_out : int
        Number of hours predicted by Path B.
    cnn_out : int
        Hours predicted by Path A (offset).

    Returns
    -------
    cyclic : (N, vmd_out, 2) ndarray
        [hour_sin, hour_cos] for each of the vmd_out future hours.
    """
    N = len(times)
    hours = pd.DatetimeIndex(times).hour.values
    cyclic = np.zeros((N, vmd_out, 2), dtype=np.float32)

    for h_idx in range(vmd_out):
        future_hr = (hours + (cnn_out + h_idx + 1)) % 24
        cyclic[:, h_idx, 0] = np.sin(2 * np.pi * future_hr / 24)
        cyclic[:, h_idx, 1] = np.cos(2 * np.pi * future_hr / 24)

    return cyclic


def build_horizon_cyclic_features_torch(times_batch, vmd_out, cnn_out, device):
    """PyTorch version — builds cyclic features on-device for inference.

    Parameters
    ----------
    times_batch : (B,) list/array of pandas Timestamps or datetime
    vmd_out, cnn_out : int
    device : torch.device

    Returns
    -------
    cyclic : (B, vmd_out, 2) tensor on device
    """
    B = len(times_batch)

    if isinstance(times_batch[0], (pd.Timestamp, datetime)):
        hr = np.array([t.hour for t in times_batch], dtype=np.float32)
    else:
        hr = np.array(
            [pd.Timestamp(t).hour for t in times_batch], dtype=np.float32
        )

    cyclic = np.zeros((B, vmd_out, 2), dtype=np.float32)
    for h_idx in range(vmd_out):
        future_hr = (hr + (cnn_out + h_idx + 1)) % 24
        cyclic[:, h_idx, 0] = np.sin(2 * np.pi * future_hr / 24)
        cyclic[:, h_idx, 1] = np.cos(2 * np.pi * future_hr / 24)

    return torch.from_numpy(cyclic).to(device)


def build_day_of_year_features(times):
    """Build day_of_year sin/cos features.

    Captures annual seasonal patterns (winter vs summer wind regime).

    Parameters
    ----------
    times : (N,) array of datetime-like

    Returns
    -------
    doy : (N, 2) ndarray
        [day_sin, day_cos]
    """
    doy = pd.DatetimeIndex(times).dayofyear.values.astype(np.float32)
    n_day = 366.0
    doy_feat = np.zeros((len(times), 2), dtype=np.float32)
    doy_feat[:, 0] = np.sin(2 * np.pi * doy / n_day)
    doy_feat[:, 1] = np.cos(2 * np.pi * doy / n_day)
    return doy_feat


def build_nwp_features_from_era5(df, vmd_out, cnn_out, nwp_cols=None):
    """Build NWP-like forecast features from ERA5 reanalysis.

    This is a PROXY — for actual deployment, replace with real ECMWF
    HRES forecast data. The difference: ERA5 is the "true" historical
    state, so using it as "forecast" creates an upper-bound estimate.

    Returns full-length (N, vmd_out, n_feat) with NaN at the end where
    shift goes beyond data. Caller should slice and drop NaN.

    Parameters
    ----------
    df      : DataFrame with ERA5 data
    vmd_out, cnn_out : int
    nwp_cols : list of column names to extract per horizon

    Returns
    -------
    nwp_feat : (N, vmd_out, n_feat) ndarray
        NaN where shift exceeds data bounds.
    """
    if nwp_cols is None:
        nwp_cols = ["wind_speed_100m", "air_density", "u100", "v100", "t2m"]

    N = len(df)

    # For each horizon, shift the column forward by that many hours
    feat = np.full((N, vmd_out, len(nwp_cols)), np.nan, dtype=np.float32)
    for h_idx in range(vmd_out):
        shift = cnn_out + h_idx + 1
        for c_idx, col in enumerate(nwp_cols):
            feat[:, h_idx, c_idx] = df[col].shift(-shift).values

    return feat
