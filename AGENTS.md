# AGENTS.md — 6.14_wind_forecast_glide

## Repo overview

ERA5 NetCDF → CSV ETL + VMD-LSTM Hybrid 24h multi-step wind power forecasting.

```
data/wind_nc/
  nc2wind_csv.py            # ETL: .nc → wind_data.csv (5 points, noise injection)
  test.py                   # smoke test (open one .nc)
  output/wind_data.csv      # generated output
  *.nc                      # ERA5 instant files (7 files)

forecast_tsp/
  vmd_utils.py              # VMD decomposition (NumPy CPU + PyTorch GPU, disk cache)
  vmd_hybrid_model.py       # VMDLSTMHybrid PyTorch model
  forecast_vmd_hybrid.py    # training + evaluation (self-contained)
  evaluate_vmd_hybrid.py    # standalone evaluation dashboard

Hyperparameter_Tuning/
  Joint_Bayesian/                     # Optuna TPE full 12-param joint search (150 trials)
    joint_bayesian_search.py
  CNN_Hyperparameter_Tuning_random/   # Path A (CNN-LSTM) random search (30 trials)
    cnn_param_search.py
  VMD_Hyperparameter_Tuning_random/   # Path B (VMD-LSTM) random search (60 trials)
    vmd_param_search.py
  Dropout_Warmup_Tuning_bys/          # Bayesian GP+EI dropout+warmup optimisation
    bayesian_dropout_warmup.py
  A_B_hours_bys/                      # A/B split point grid search
    find_optimal_split.py
```

## Pipeline

```
.nc files → nc2wind_csv.py → wind_data.csv (point_id=1~5, 5 lat/lon)
                                ↓
   forecast_vmd_hybrid.py  ─→ VMD per-domain (train/val/test independent)
                           ─→ build sequences (120h weather+IMFs → 24h power)
                           ─→ train VMD-LSTM Hybrid (Joint Bayesian best params)
                           ─→ predict + metrics + plot
```

## Architecture

- **VMD decomposition**: K=4 modes, alpha=500 bandwidth penalty, per-domain (no future data leakage), disk cache
- **Dual-path model** (VMDLSTMHybrid):
  - Weather features (8 dim): power_current, ws100, ρ, u100, v100, t2m, hour_sin, hour_cos
  - IMF features (4 dim): IMF1~IMF4 from VMD
  - Path A (CNN-LSTM, h=1~16): weather → Conv1D×2(160,96) → MaxPool → LSTM(64,1层,DO=0.3879) → FC(16)
  - Path B Trend (VMD-LSTM, h=17~24): IMF1+IMF2 → LSTM(128,1层) → FC(128→32) → DO(0.3936) → FC(32→8)
  - Path B Fluct (VMD-LSTM, h=17~24): IMF3+IMF4 → LSTM(96,1层) → FC(96→32) → DO(0.3936) → FC(32→8)
  - Sum Path B → Concat Path A → 24h prediction → clamp[0, 2000] (inference only)
- **Total params**: ~200K (varies with cnn_out)
- **Sequence**: 120-h sliding windows; 12 features (8 weather + 4 IMFs)
- **Labels**: Power computed via Vestas V90 physics formula (training only)
- **Train/Val/Test**: 2024–2025-Sep train, 2025 Oct–Dec val, 2026 test
- **Point**: point_id=1 only (lat=41, lon=96)
- **Multi-point ETL**: nc2wind_csv.py extracts 5 points (lat=41→39, lon=96→98) with Gaussian noise

## Current best results (no data leakage)

```
MAE = 259.05 kW   R² = 0.4315
NMAE = 12.95%     NRMSE = 18.44%
h=1:  MAE=98.6   R²=0.89   (CNN-LSTM)
h=12: MAE=266.5  R²=0.44   (VMD-LSTM)
h=24: MAE=317.9  R²=0.25   (VMD-LSTM)
```

## Training config

| Parameter | Value |
|-----------|-------|
| seq_len | 120 h |
| cnn_out (A/B split) | **16** (CNN-LSTM 1~16h, VMD-LSTM 17~24h) |
| batch_size | 64 |
| lr | **1.7035e-5** |
| weight_decay | **9.7659e-5** |
| Path A Conv filters | **160 → 96** |
| Path A LSTM hidden | **64** |
| Path A dropout | **0.3879** |
| Path B Trend hidden | **128** |
| Path B Fluct hidden | **96** |
| Path B n_layers | **1** |
| Path B fc_hidden | **32** |
| Path B dropout | **0.3936** |
| optimizer | Adam |
| scheduler | ReduceLROnPlateau(factor=0.5, patience=10) |
| early stop | patience=30 |
| grad clip | max_norm=1.0 |
| scaler | StandardScaler (fit train only) |
| VMD cache | `forecast_tsp/vmd_cache/` (per-domain .npz) |

## Key design decisions

- **VMD per-domain**: train/val/test each run independent VMD to prevent frequency-domain leakage. IMF scaler fitted on train domain only.
- **Joint Bayesian optimisation**: All 12 hyperparameters tuned simultaneously via Optuna TPE (150 trials). Best params used in `forecast_vmd_hybrid.py`.
- **Fixed A/B split at h=16**: CNN-LSTM handles 1~16h, VMD-LSTM handles 17~24h (found by grid search).
- **Low LR enables multi-epoch training**: lr=1.7035e-5 + wd=9.7659e-5 allows training beyond epoch 1 without severe overfitting.
- **No data leakage**: weather & IMF scalers fit on train only, VMD per-domain.

## Dependencies

```
pip install xarray pandas numpy netCDF4 scikit-learn matplotlib torch optuna
```

## Run

```powershell
# Step 0: generate wind_data.csv (with noise injection)
cd data\wind_nc
python nc2wind_csv.py

# Step 1: train + predict (from project root)
cd ..\..
python forecast_tsp\forecast_vmd_hybrid.py

# Step 2: generate evaluation dashboard
python forecast_tsp\evaluate_vmd_hybrid.py
```

## Hyperparameter tuning sub-projects

```powershell
# Joint Bayesian — 12 params, 150 trials (Optuna TPE)
python Hyperparameter_Tuning\Joint_Bayesian\joint_bayesian_search.py

# Path A (CNN-LSTM) random search — 5 params, 30 trials
python Hyperparameter_Tuning\CNN_Hyperparameter_Tuning_random\cnn_param_search.py

# Path B (VMD-LSTM) random search — 6 params, 60 trials, alpha grid
python Hyperparameter_Tuning\VMD_Hyperparameter_Tuning_random\vmd_param_search.py

# A/B split point grid search — find optimal cnn_out
python Hyperparameter_Tuning\A_B_hours_bys\find_optimal_split.py

# Bayesian GP+EI dropout + warmup optimisation
python Hyperparameter_Tuning\Dropout_Warmup_Tuning_bys\bayesian_dropout_warmup.py
```

## Constraints

- No CI, tests, linting, formatter, typechecker, or version management
- No `.gitignore` — watch out for generated `*.csv`, `*.pth`, `*.png`, `*.npz` files, `vmd_cache/`
- Hardcoded paths use backslash separators (`data\wind_nc\`)
- GPU used if available (falls back to CPU)
- VMD decomposition: NumPy CPU by default (`vmd_torch()` GPU path available for large N > 100k)
- **Data leakage prevention**: VMD must run per-domain (train/val/test independent), IMF scaler fit on train only
- **Multi-point ETL**: nc2wind_csv.py generates 5 points; training uses only point_id=1
- **Gaussian noise injection**: `u100` (0.1), `v100` (0.1), `t2m` (0.5K), `sp` (10Pa), `blh` (10m)
