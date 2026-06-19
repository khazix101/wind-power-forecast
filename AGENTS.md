# AGENTS.md — 6.14_wind_forecast_glide

## Repo overview

ERA5 NetCDF → CSV ETL + VMD-LSTM Hybrid 24h multi-step wind power forecasting.

```
data/wind_nc/
  nc2wind_csv.py            # ETL: .nc → wind_data.csv
  test.py                   # smoke test (open one .nc)
  output/wind_data.csv      # generated output
  *.nc                      # ERA5 instant files (7 files)

forecast_tsp/
  vmd_utils.py              # VMD decomposition (NumPy CPU + PyTorch GPU)
  vmd_hybrid_model.py       # VMDLSTMHybrid PyTorch model
  forecast_vmd_hybrid.py    # training + evaluation (self-contained)
  evaluate_vmd_hybrid.py    # standalone evaluation dashboard

Hyperparameter_Tuning/
  CNN_Hyperparameter_Tuning/     # Path A (CNN-LSTM) random search
  VMD_Hyperparameter_Tuning/     # Path B (VMD-LSTM) random search
  Dropout_Warmup_Tuning/         # Bayesian dropout+warmup optimisation

A_B_hours/
  find_optimal_split.py          # A/B split point grid search
```

## Pipeline

```
.nc files → nc2wind_csv.py → wind_data.csv
                                ↓
   forecast_vmd_hybrid.py  ─→ VMD per-domain (train/val/test independent)
                           ─→ build sequences (120h IMFs → 24h power)
                           ─→ train VMD-LSTM Hybrid
                           ─→ predict + metrics + plot
```

## Architecture

- **VMD decomposition**: K=4 modes, alpha=500 bandwidth penalty, per-domain (no future data leakage)
- **Dual-path model** (VMDLSTMHybrid):
  - Path A (CNN-LSTM, h=1~8): weather features → Conv1D×2(128,192) → MaxPool → LSTM(32,1层,DO=0.31) → FC(8)
  - Path B Trend (VMD-LSTM, h=9~24): IMF1+IMF2 → LSTM(128,1层) → FC(128→64) → DO(0.444) → FC(64→16)
  - Path B Fluct (VMD-LSTM, h=9~24): IMF3+IMF4 → LSTM(64,1层) → FC(64→64) → DO(0.444) → FC(64→16)
  - Sum Path B → Concat Path A → clamp[0, 2000] → 24h prediction
- **Total params**: ~206K
- **Sequence**: 120-h sliding windows; 4 IMF channels per timestep
- **Labels**: Power computed via Vestas V90 physics formula (training only)
- **Train/Val/Test**: 2024–2025-Sep train, 2025 Oct–Dec val, 2026 test
- **Point**: point_id=1 only (lat=41, lon=96)

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
| batch_size | 64 |
| lr | 5e-4 |
| weight_decay | 5e-4 |
| dropout | 0.3 (Path A 0.31, Path B 0.444) |
| optimizer | Adam |
| scheduler | ReduceLROnPlateau(factor=0.5, patience=10) |
| early stop | patience=30 |
| grad clip | max_norm=1.0 |
| scaler | StandardScaler (fit train only) |

## Key design decisions

- **VMD per-domain**: train/val/test each run independent VMD to prevent frequency-domain leakage. IMF scaler fitted on train domain only.
- **epoch-1 convergence**: model reaches best validation performance at epoch 1 with current LR=5e-4. Lower LR allows more epochs but degrades test performance due to validation/test distribution mismatch.
- **Short CNN vs long VMD split**: optimal boundary is cnn_out=8 (determined by grid search in A_B_hours/).

## Dependencies

```
pip install xarray pandas numpy netCDF4 scikit-learn matplotlib torch
```

## Run

```powershell
# Step 0: generate wind_data.csv
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
# Path A (CNN-LSTM) random search
python Hyperparameter_Tuning\CNN_Hyperparameter_Tuning\cnn_param_search.py

# Path B (VMD-LSTM) random search — 60 trials, 6 params
python Hyperparameter_Tuning\VMD_Hyperparameter_Tuning\vmd_param_search.py

# A/B split point grid search — find optimal cnn_out
python A_B_hours\find_optimal_split.py

# Bayesian dropout + warmup optimisation
python Hyperparameter_Tuning\Dropout_Warmup_Tuning\bayesian_dropout_warmup.py
```

## Constraints

- No CI, tests, linting, formatter, typechecker, or version management
- No `.gitignore` — watch out for generated `*.csv`, `*.pth`, `*.png`, `*.npz` files
- Hardcoded paths use backslash separators (`data\wind_nc\`)
- GPU used if available (falls back to CPU)
- VMD decomposition: NumPy CPU by default (`vmd_torch()` GPU path available for large N > 100k)
- **Data leakage prevention**: VMD must run per-domain (train/val/test independent), IMF scaler fit on train only
- **Known limitation**: best model at epoch 1; extended training overfits due to small val set (2,208 seqs) not representing test set
