"""Optuna TPE joint Bayesian optimization for all 12 hyperparameters.

Parameters:
  lr, weight_decay, dropout_a, dropout_b,
  n_filters1, n_filters2, lstm_hidden_a,
  trend_hidden, fluct_hidden, n_layers, fc_hidden, cnn_out

No data leakage: VMD per-domain, IMF scaler fit on train only.
Metric: validation RMSE (kW, unscaled).
Uses Optuna TPE sampler for mixed continuous/discrete spaces.
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna
from optuna.samplers import TPESampler
import os, sys, time, warnings, json

warnings.filterwarnings("ignore")

sys.path.insert(0, r"D:\net.zero\6.14_wind_forecast_glide\forecast_tsp")
from vmd_utils import decompose_by_domain
from vmd_hybrid_model import VMDLSTMHybrid

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════
SEQ_LEN = 120
BATCH_SIZE = 64
OUTPUT_DIM = 24
CAPACITY = 2000.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
WEATHER_DIM = 8
N_IMFS = 4
N_TRIALS = 150
TRIAL_EPOCHS = 80
PATIENCE = 20
VMD_ALPHA = 500

BASE_DIR = r"D:\net.zero\6.14_wind_forecast_glide"
OUT_DIR = os.path.join(BASE_DIR, "Hyperparameter_Tuning", "Joint_Bayesian")
os.makedirs(OUT_DIR, exist_ok=True)

VMD_CACHE_DIR = os.path.join(OUT_DIR, "vmd_cache")
os.makedirs(VMD_CACHE_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════
# Power curve functions
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
# Data loading (once, shared across all trials)
# ══════════════════════════════════════════════
print("=" * 60)
print("  Optuna TPE — Joint Bayesian Search (12 params)")
print(f"  Device: {DEVICE}")
print("=" * 60)

df = pd.read_csv(os.path.join(BASE_DIR, "data", "wind_nc", "output", "wind_data.csv"))
df["valid_time"] = pd.to_datetime(df["valid_time"])
df = df[df["point_id"] == 1].sort_values("valid_time").reset_index(drop=True)

target_cols = []
for h in range(1, OUTPUT_DIM + 1):
    col = f"power_t{h}"
    ws_shifted = df["wind_speed_100m"].shift(-h).values
    rho_shifted = df["air_density"].shift(-h).values
    df[col] = compute_power(ws_shifted, rho_shifted)
    target_cols.append(col)
df = df.dropna().reset_index(drop=True)

hour = pd.DatetimeIndex(df["valid_time"]).hour
df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
df["power_current"] = compute_power(df["wind_speed_100m"].values, df["air_density"].values)

weather_cols = ["power_current", "wind_speed_100m", "air_density",
                "u100", "v100", "t2m", "hour_sin", "hour_cos"]

times = df["valid_time"].values

# ── Labels ──
y_raw = df[target_cols].values.astype(np.float32)
train_years_mask = pd.DatetimeIndex(times).year.isin([2024, 2025])
scaler_y = StandardScaler()
scaler_y.fit(y_raw[train_years_mask])
y_scaled = scaler_y.transform(y_raw)

# ── Weather ──
weather_raw = df[weather_cols].values.astype(np.float32)
weather_scaler = StandardScaler()
weather_scaler.fit(weather_raw[train_years_mask])
weather_scaled = weather_scaler.transform(weather_raw)

# ── VMD per-domain ──
sample_years = pd.DatetimeIndex(times).year
sample_months = pd.DatetimeIndex(times).month
dom_tr = sample_years.isin([2024, 2025]) & ~((sample_years == 2025) & (sample_months >= 10))
dom_va = (sample_years == 2025) & (sample_months >= 10)
dom_te = sample_years == 2026

print(f"\n[VMD] alpha={VMD_ALPHA} per-domain ...")
t0 = time.time()
power_raw = df[target_cols].values[:, 0].astype(float)
imfs, omegas = decompose_by_domain(
    power_raw,
    [("train", dom_tr), ("val", dom_va), ("test", dom_te)],
    K=N_IMFS, alpha=VMD_ALPHA, tol=1e-7, max_iter=500, seed=SEED,
    cache_dir=VMD_CACHE_DIR,
)
print(f"  VMD total: {time.time() - t0:.1f}s")

# ── Scale IMFs (fit on train) ──
imfs_scaler = StandardScaler()
imfs_scaler.fit(imfs[dom_tr])
imfs_scaled = imfs_scaler.transform(imfs)

# ── Build sequences ──
features = np.concatenate([weather_scaled, imfs_scaled], axis=1).astype(np.float32)
X_list, y_list = [], []
for i in range(SEQ_LEN - 1, len(features)):
    X_list.append(features[i - SEQ_LEN + 1: i + 1])
    y_list.append(y_scaled[i])
X_all = np.array(X_list, dtype=np.float32)
y_all = np.array(y_list, dtype=np.float32)
seq_times = times[SEQ_LEN - 1:]
seq_years = pd.DatetimeIndex(seq_times).year
seq_months = pd.DatetimeIndex(seq_times).month

train_mask = seq_years.isin([2024, 2025]) & ~((seq_years == 2025) & (seq_months >= 10))
val_mask   = (seq_years == 2025) & (seq_months >= 10)
test_mask  = seq_years == 2026

X_train, y_train = X_all[train_mask], y_all[train_mask]
X_val,   y_val   = X_all[val_mask],   y_all[val_mask]
X_test,  y_test  = X_all[test_mask],  y_all[test_mask]

print(f"  Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}\n")


# ══════════════════════════════════════════════
# Dataset + objective function
# ══════════════════════════════════════════════
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def objective(trial):
    """Optuna objective: return val RMSE (kW)."""

    # ── Sample hyperparameters ──
    lr          = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
    wd          = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    dropout_a   = trial.suggest_float("dropout_a", 0.1, 0.5)
    dropout_b   = trial.suggest_float("dropout_b", 0.2, 0.6)
    n_filters1  = trial.suggest_int("n_filters1", 64, 256, step=32)
    n_filters2  = trial.suggest_int("n_filters2", 96, 384, step=32)
    lstm_h_a    = trial.suggest_int("lstm_hidden_a", 16, 64, step=16)
    trend_h     = trial.suggest_int("trend_hidden", 64, 256, step=32)
    fluct_h     = trial.suggest_int("fluct_hidden", 32, 128, step=32)
    n_layers    = trial.suggest_int("n_layers", 1, 2)
    fc_h        = trial.suggest_int("fc_hidden", 32, 128, step=32)
    cnn_out     = trial.suggest_int("cnn_out", 4, 16, step=2)

    # ── Build model ──
    model = VMDLSTMHybrid(
        weather_dim=WEATHER_DIM, n_imfs=N_IMFS,
        cnn_out=cnn_out, output_dim=OUTPUT_DIM,
        trend_hidden=trend_h, fluct_hidden=fluct_h,
        n_layers=n_layers, dropout=dropout_b,
        fc_hidden=fc_h, path_b_dropout=dropout_b,
        capacity=CAPACITY,
        conv1_filters=n_filters1, conv2_filters=n_filters2,
        conv_kernel=3, pool_size=2,
        cnn_lstm_hidden=lstm_h_a, cnn_lstm_layers=n_layers,
        path_a_dropout=dropout_a,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())

    train_loader = DataLoader(SeqDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SeqDataset(X_val, y_val), BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_ctr = 0
    best_epoch = 0
    best_state = None
    train_t0 = time.time()

    for epoch in range(1, TRIAL_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        val_loss = 0.0
        val_n = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(x), y).item() * x.size(0)
                val_n += x.size(0)
        val_loss = val_loss / val_n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_ctr = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    # Evaluate on val set (unscaled RMSE, kW) using best checkpoint
    model.load_state_dict(best_state)
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in val_loader:
            all_pred.append(model(x.to(DEVICE)).cpu().numpy())
            all_true.append(y.numpy())
    yp_val = scaler_y.inverse_transform(np.concatenate(all_pred, axis=0))
    yt_val = scaler_y.inverse_transform(np.concatenate(all_true, axis=0))
    yp_val = np.clip(yp_val, 0, CAPACITY)
    rmse_val = float(np.sqrt(mean_squared_error(yt_val.flatten(), yp_val.flatten())))

    elapsed = time.time() - train_t0

    # Report intermediate value to Optuna
    trial.report(rmse_val, epoch)

    # Store metadata
    trial.set_user_attr("n_params", n_params)
    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("time_s", round(elapsed, 1))

    torch.cuda.empty_cache()
    return rmse_val


# ══════════════════════════════════════════════
# Run Optuna TPE
# ══════════════════════════════════════════════
print(f"\n--- Starting TPE optimisation ({N_TRIALS} trials) ---")
print(f"  12 parameters, objective = val RMSE (kW)")
print(f"  Sampler: TPESampler(n_startup_trials=10, multivariate=True)")

storage_url = f"sqlite:///{os.path.join(OUT_DIR, 'optuna_study.db')}"
study = optuna.create_study(
    direction="minimize",
    sampler=TPESampler(n_startup_trials=10, multivariate=True, seed=SEED),
    study_name="joint_bayesian_12params",
    storage=storage_url,
    load_if_exists=True,
)

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# ══════════════════════════════════════════════
# Results
# ══════════════════════════════════════════════
print(f"\n{'=' * 60}")
print(f"  BEST TRIAL (val RMSE={study.best_value:.1f} kW)")
print(f"  Trial #{study.best_trial.number}")
print(f"{'=' * 60}")
for k, v in study.best_trial.params.items():
    print(f"    {k:20s} = {v}")
print(f"    {'n_params':20s} = {study.best_trial.user_attrs['n_params']:,}")
print(f"    {'best_epoch':20s} = {study.best_trial.user_attrs['best_epoch']}")
print(f"{'=' * 60}")

# ── Save all trials ──
trials_df = study.trials_dataframe()
csv_path = os.path.join(OUT_DIR, "results.csv")
trials_df.to_csv(csv_path, index=False)
print(f"\n  All trials saved -> {csv_path}")

# ── Save best params as JSON ──
best_params = dict(study.best_trial.params)
best_params["n_params"] = study.best_trial.user_attrs["n_params"]
best_params["best_epoch"] = study.best_trial.user_attrs["best_epoch"]
best_params["val_rmse_kw"] = round(study.best_value, 2)
json_path = os.path.join(OUT_DIR, "best_params.json")
with open(json_path, "w") as f:
    json.dump(best_params, f, indent=2)
print(f"  Best params saved -> {json_path}")

# ══════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════
trial_nums = trials_df["number"].values + 1
objective_vals = trials_df["value"].values
cumulative_best = np.minimum.accumulate(objective_vals)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle("Optuna TPE — Joint Bayesian Search (12 params)", fontsize=14, fontweight="bold")

# (0,0) Convergence
ax = axes[0, 0]
ax.plot(trial_nums, objective_vals, "o", color="#3498DB", markersize=4, alpha=0.4, label="Observed RMSE")
ax.plot(trial_nums, cumulative_best, "-", color="#E74C3C", linewidth=2, label="Best so far")
ax.set_xlabel("Trial"); ax.set_ylabel("Val RMSE (kW)")
ax.set_title(f"Convergence (best={cumulative_best[-1]:.0f} kW)"); ax.legend(); ax.grid(True, alpha=0.2)

# (0,1) Parameter importance
ax = axes[0, 1]
importances = optuna.importance.get_param_importances(study)
params_imp = list(importances.keys())
vals_imp = list(importances.values())
colors_imp = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(params_imp)))
bars = ax.barh(range(len(params_imp)), vals_imp, color=colors_imp, edgecolor="white")
ax.set_yticks(range(len(params_imp))); ax.set_yticklabels(params_imp)
ax.set_xlabel("Importance"); ax.set_title("Parameter Importance (FANOVA)")
ax.invert_yaxis(); ax.grid(True, alpha=0.2, axis="x")

# (0,2) cnn_out vs RMSE
ax = axes[0, 2]
cnn_vals = trials_df["params_cnn_out"].values
sc = ax.scatter(cnn_vals, objective_vals, c=objective_vals, cmap="RdYlGn_r",
                s=40, alpha=0.6, edgecolors="white")
ax.set_xlabel("cnn_out (A/B split point)"); ax.set_ylabel("Val RMSE (kW)")
ax.set_title("cnn_out vs RMSE"); ax.grid(True, alpha=0.2)
cbar = plt.colorbar(sc, ax=ax); cbar.set_label("RMSE (kW)")

# (1,0) lr vs weight_decay
ax = axes[1, 0]
lr_vals = trials_df["params_lr"].values
wd_vals = trials_df["params_weight_decay"].values
sc2 = ax.scatter(np.log10(lr_vals), np.log10(wd_vals), c=objective_vals,
                 cmap="RdYlGn_r", s=40, alpha=0.6, edgecolors="white")
ax.set_xlabel("log10(lr)"); ax.set_ylabel("log10(weight_decay)")
ax.set_title("LR × Weight Decay → RMSE"); ax.grid(True, alpha=0.2)
cbar2 = plt.colorbar(sc2, ax=ax); cbar2.set_label("RMSE (kW)")

# (1,1) n_params vs RMSE
ax = axes[1, 1]
n_params_vals = [study.trials[i].user_attrs.get("n_params", 0) for i in range(len(study.trials))]
sc3 = ax.scatter(n_params_vals, objective_vals, c=objective_vals, cmap="RdYlGn_r",
                 s=40, alpha=0.6, edgecolors="white")
ax.set_xlabel("Number of Parameters"); ax.set_ylabel("Val RMSE (kW)")
ax.set_title("Model Size vs RMSE"); ax.grid(True, alpha=0.2)
cbar3 = plt.colorbar(sc3, ax=ax); cbar3.set_label("RMSE (kW)")

# (1,2) Dropout comparison
ax = axes[1, 2]
da_vals = trials_df["params_dropout_a"].values
db_vals = trials_df["params_dropout_b"].values
sc4 = ax.scatter(da_vals, db_vals, c=objective_vals, cmap="RdYlGn_r",
                 s=40, alpha=0.6, edgecolors="white")
ax.set_xlabel("Path A Dropout"); ax.set_ylabel("Path B Dropout")
ax.set_title("Dropout Space → RMSE"); ax.grid(True, alpha=0.2)
cbar4 = plt.colorbar(sc4, ax=ax); cbar4.set_label("RMSE (kW)")

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bayesian_results.png"), dpi=150)
plt.close(fig)
print(f"  Plot saved -> {os.path.join(OUT_DIR, 'bayesian_results.png')}")

print("\nDone.")
