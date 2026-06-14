# AGENTS.md — 5.30_wind_forecast_glide

## Repo overview

ERA5 NetCDF → CSV ETL + VMD-LSTM Hybrid 24h multi-step wind power forecasting.

```
data/wind_nc/
  nc2wind_csv.py            # ETL: .nc → wind_data.csv
  test.py                   # smoke test (open one .nc)
  output/wind_data.csv      # generated output
  *.nc                      # ERA5 instant files (7 files)

forecast_tsp/
  vmd_utils.py              # VMD decomposition (pure NumPy)
  vmd_hybrid_model.py       # VMDLSTMHybrid PyTorch model
  forecast_vmd_hybrid.py    # training + evaluation (self-contained)
  evaluate_vmd_hybrid.py    # standalone evaluation dashboard
```

## Pipeline

```
.nc files → nc2wind_csv.py → wind_data.csv
                                ↓
   forecast_vmd_hybrid.py  ─→ VMD decompose power → IMFs
                           ─→ build sequences (120h IMFs → 24h power)
                           ─→ train VMD-LSTM Hybrid
                           ─→ predict + metrics + plot
```

## Architecture

- **VMD decomposition**: K=4 modes, alpha=2000 bandwidth penalty, pure NumPy/FFT
- **Dual-path model** (VMDLSTMHybrid):
  - Path A (Trend): IMF1+IMF2 → LSTM(hidden=100, layers=2, dropout=0.3) → FC(50) → 24 outputs
  - Path B (Fluctuation): IMF3+IMF4 → LSTM(hidden=128, layers=2, dropout=0.3) → FC(50) → 24 outputs
  - Sum + clamp[0, 2000] → final 24h power prediction
- **Sequence**: 120-h sliding windows; 4 IMF channels per timestep
- **Labels**: Power computed via Vestas V90 physics formula (training only)
- **Train/Val/Test**: 2024–2025-Sep train, 2025 Oct–Dec val, 2026 test
- **Point**: point_id=1 only (lat=41, lon=96)

## Dependencies

```
pip install xarray pandas numpy netCDF4 scikit-learn matplotlib torch
```

## Run

```powershell
# Step 0: generate wind_data.csv
cd data\wind_nc
python nc2wind_csv.py

# Step 1: train + predict
cd ..\forecast_tsp
python forecast_vmd_hybrid.py

# Step 2: generate evaluation dashboard
python evaluate_vmd_hybrid.py
```

## Constraints

- No CI, tests, linting, formatter, typechecker, or version management
- No `.gitignore` — watch out for generated `*.csv`, `*.pth`, `*.png`, `*.npz` files
- Hardcoded paths use backslash separators (`data\wind_nc\`)
- GPU used if available (falls back to CPU)
- No external VMD library — self-contained NumPy/SciPy FFT implementation
