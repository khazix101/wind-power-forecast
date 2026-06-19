"""Random search for Path A (CNN-LSTM) hyperparameters."""
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
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, r"D:\net.zero\6.14_wind_forecast_glide\forecast_tsp")
from vmd_utils import VMDDecomposer, decompose_by_domain
from vmd_hybrid_model import VMDLSTMHybrid

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ══════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════
SEQ_LEN = 120
BATCH_SIZE = 64
OUTPUT_DIM = 24
CNN_OUT = 12
CAPACITY = 2000.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_TRIALS = 30
MAX_EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 5e-4
WEATHER_DIM = 8
N_IMFS = 4
EARLY_STOP_PATIENCE = 10

# Fixed Path B params
TREND_HIDDEN = 100
FLUCT_HIDDEN = 128
N_LAYERS = 2
PATH_B_DROPOUT = 0.3

BASE_DIR = r"D:\net.zero\6.14_wind_forecast_glide"
OUT_DIR = os.path.join(BASE_DIR, "Hyperparameter_Tuning", "CNN_Hyperparameter_Tuning")
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
# 1. Load data (once)
# ══════════════════════════════════════════════
print("=" * 60)
print("  CNN-LSTM Hyperparameter Tuning — Data Preparation")
print("=" * 60)

df = pd.read_csv(os.path.join(BASE_DIR, "data", "wind_nc", "output", "wind_data.csv"))
df["valid_time"] = pd.to_datetime(df["valid_time"])
df = df[df["point_id"] == 1].sort_values("valid_time").reset_index(drop=True)

# ── 24h power labels ──
target_cols = []
for h in range(1, OUTPUT_DIM + 1):
    col = f"power_t{h}"
    ws_shifted = df["wind_speed_100m"].shift(-h).values
    rho_shifted = df["air_density"].shift(-h).values
    df[col] = compute_power(ws_shifted, rho_shifted)
    target_cols.append(col)
df = df.dropna().reset_index(drop=True)

# ── Time encoding ──
hour = pd.DatetimeIndex(df["valid_time"]).hour
df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

# ── Current power ──
df["power_current"] = compute_power(df["wind_speed_100m"].values, df["air_density"].values)

weather_cols = ["power_current", "wind_speed_100m", "air_density",
                "u100", "v100", "t2m", "hour_sin", "hour_cos"]

times = df["valid_time"].values
train_years_mask = pd.DatetimeIndex(times).year.isin([2024, 2025])

# ══════════════════════════════════════════════
# 2. VMD decomposition per-domain (no leakage)
# ══════════════════════════════════════════════
power_raw = df[target_cols].values[:, 0].astype(float)
sample_years = pd.DatetimeIndex(times).year
sample_months = pd.DatetimeIndex(times).month
dom_tr = sample_years.isin([2024, 2025]) & ~((sample_years == 2025) & (sample_months >= 10))
dom_va = (sample_years == 2025) & (sample_months >= 10)
dom_te = sample_years == 2026

print(f"\n[VMD] Domain-separated (alpha=2000) ...")
t0 = time.time()
imfs, omegas = decompose_by_domain(
    power_raw,
    [("train", dom_tr), ("val", dom_va), ("test", dom_te)],
    K=4, alpha=2000, tol=1e-7, max_iter=500, seed=SEED,
    cache_dir=VMD_CACHE_DIR,
)
print(f"  Done in {time.time()-t0:.1f}s  |  IMFs: {imfs.shape}")

# ══════════════════════════════════════════════
# 3. Scale & build sequences (once)
# ══════════════════════════════════════════════
y_raw = df[target_cols].values.astype(np.float32)

scaler_y = StandardScaler()
scaler_y.fit(y_raw[train_years_mask])
y_scaled = scaler_y.transform(y_raw)

weather_raw = df[weather_cols].values.astype(np.float32)
weather_scaler = StandardScaler()
weather_scaler.fit(weather_raw[train_years_mask])
weather_scaled = weather_scaler.transform(weather_raw)

imfs_scaler = StandardScaler()
imfs_scaler.fit(imfs[dom_tr])
imfs_scaled = imfs_scaler.transform(imfs)

features = np.concatenate([weather_scaled, imfs_scaled], axis=1).astype(np.float32)

# Sequences
X_list, y_list, idx_list = [], [], []
n = len(features)
for i in range(SEQ_LEN - 1, n):
    X_list.append(features[i - SEQ_LEN + 1: i + 1])
    y_list.append(y_scaled[i])
    idx_list.append(i)
X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)
seq_times = times[SEQ_LEN - 1:]
seq_years = pd.DatetimeIndex(seq_times).year
seq_months = pd.DatetimeIndex(seq_times).month

train_mask = seq_years.isin([2024, 2025]) & ~((seq_years == 2025) & (seq_months >= 10))
val_mask   = (seq_years == 2025) & (seq_months >= 10)
test_mask  = seq_years == 2026

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\n  Train: {len(X_train)}  Val: {len(X_val)}  Test: {X_test.shape[0]}")
print(f"  Feature dim: {features.shape[1]}")

# ══════════════════════════════════════════════
# 4. Training utilities
# ══════════════════════════════════════════════
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

@torch.no_grad()
def predict_loader(model, loader, device):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x = x.to(device)
        preds.append(model(x).cpu().numpy())
        trues.append(y.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)

# ══════════════════════════════════════════════
# 5. Random search
# ══════════════════════════════════════════════
PARAM_KEYS = ["conv1_filters", "conv2_filters", "cnn_lstm_hidden",
              "cnn_lstm_layers", "path_a_dropout"]

# Pre-computed candidate lists
conv1_candidates = [32, 48, 64, 96, 128]
conv2_candidates = [64, 96, 128, 192, 256]
hidden_candidates = [32, 50, 64, 100, 128]

print(f"\n{'=' * 60}")
print(f"  Random Search — {N_TRIALS} trials")
print(f"  Device: {DEVICE}")
print(f"{'=' * 60}")

results = []
next_trial_path = os.path.join(OUT_DIR, "results_partial.csv")
start_trial = 0

# Resume from partial results
if os.path.exists(next_trial_path):
    existing = pd.read_csv(next_trial_path)
    results = existing.to_dict("records")
    start_trial = len(results)
    print(f"  Resuming from trial {start_trial + 1}/{N_TRIALS}")

for trial in range(start_trial, N_TRIALS):
    # Sample hyperparameters
    cf1 = np.random.choice(conv1_candidates)
    cf2 = np.random.choice(conv2_candidates)
    if cf2 <= cf1:
        cf2 = min(cf1 * 2, max(conv2_candidates))
    lstm_h = np.random.choice(hidden_candidates)
    lstm_l = np.random.choice([1, 2])
    p_drop = round(np.random.uniform(0.1, 0.5), 3)

    params = {
        "trial": trial + 1,
        "conv1_filters": cf1,
        "conv2_filters": cf2,
        "cnn_lstm_hidden": lstm_h,
        "cnn_lstm_layers": lstm_l,
        "path_a_dropout": p_drop,
        "n_params": 0,
        "best_epoch": 0,
        "train_time_s": 0,
        "val_rmse_1to12": 0,
        "val_rmse_all": 0,
        "test_rmse_1to12": 0,
        "test_rmse_all": 0,
        "test_mae_all": 0,
    }

    param_desc = f"cf1={cf1} cf2={cf2} h={lstm_h} l={lstm_l} d={p_drop}"
    print(f"\n  Trial {trial+1}/{N_TRIALS} | {param_desc}")

    # Build model (cast int64→int for PyTorch compat)
    model = VMDLSTMHybrid(
        weather_dim=WEATHER_DIM, n_imfs=N_IMFS, cnn_out=CNN_OUT,
        output_dim=OUTPUT_DIM, trend_hidden=TREND_HIDDEN,
        fluct_hidden=FLUCT_HIDDEN, n_layers=N_LAYERS, dropout=PATH_B_DROPOUT,
        capacity=CAPACITY,
        conv1_filters=int(cf1), conv2_filters=int(cf2), conv_kernel=3,
        pool_size=2, cnn_lstm_hidden=int(lstm_h), cnn_lstm_layers=int(lstm_l),
        path_a_dropout=float(p_drop),
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    params["n_params"] = n_params
    print(f"         Params: {n_params:,}")

    # Data
    train_loader = DataLoader(SeqDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_val, y_val), BATCH_SIZE)

    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    # Training loop
    best_val_rmse_12 = float("inf")
    best_model_state = None
    patience_counter = 0
    train_t0 = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Evaluate
        model.eval()
        val_loss = 0.0
        val_n = 0
        all_pred, all_true = [], []
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                out = model(x)
                val_loss += criterion(out, y).item() * x.size(0)
                val_n += x.size(0)
                all_pred.append(out.cpu().numpy())
                all_true.append(y.cpu().numpy())
        val_loss /= val_n
        scheduler.step(val_loss)

        y_pred_v = scaler_y.inverse_transform(np.concatenate(all_pred, axis=0))
        y_true_v = scaler_y.inverse_transform(np.concatenate(all_true, axis=0))
        y_pred_v = np.clip(y_pred_v, 0, CAPACITY)

        rmse_12 = compute_rmse(y_true_v[:, :CNN_OUT], y_pred_v[:, :CNN_OUT])
        rmse_all = compute_rmse(y_true_v, y_pred_v)

        if rmse_12 < best_val_rmse_12:
            best_val_rmse_12 = rmse_12
            patience_counter = 0
            best_model_state = model.state_dict()
            params["best_epoch"] = epoch
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

    params["train_time_s"] = round(time.time() - train_t0, 1)
    params["val_rmse_1to12"] = round(float(best_val_rmse_12), 2)
    params["val_rmse_all"] = round(float(rmse_all), 2)

    # Evaluate on test set
    model.load_state_dict(best_model_state)
    test_loader = DataLoader(SeqDataset(X_test, y_test), BATCH_SIZE)
    y_pred_t, y_true_t = predict_loader(model, test_loader, DEVICE)
    y_pred_t = scaler_y.inverse_transform(y_pred_t)
    y_pred_t = np.clip(y_pred_t, 0, CAPACITY)
    y_true_t = scaler_y.inverse_transform(y_true_t)

    params["test_rmse_1to12"] = round(float(compute_rmse(y_true_t[:, :CNN_OUT], y_pred_t[:, :CNN_OUT])), 2)
    params["test_rmse_all"] = round(float(compute_rmse(y_true_t, y_pred_t)), 2)
    params["test_mae_all"] = round(float(np.abs(y_true_t - y_pred_t).mean()), 2)

    print(f"         Val RMSE[1-12]={params['val_rmse_1to12']:.1f}  Test MAE={params['test_mae_all']:.1f}  "
          f"Best epoch={params['best_epoch']}  Time={params['train_time_s']:.0f}s")

    results.append(params)

    # Save partial results
    pd.DataFrame(results).to_csv(next_trial_path, index=False)

# ══════════════════════════════════════════════
# 6. Save & generate plots
# ══════════════════════════════════════════════
df_results = pd.DataFrame(results)
df_results = df_results.sort_values("val_rmse_1to12").reset_index(drop=True)
csv_path = os.path.join(OUT_DIR, "results.csv")
df_results.to_csv(csv_path, index=False)
print(f"\n  Results saved -> {csv_path}")

best_idx = df_results["val_rmse_1to12"].idxmin()
best = df_results.iloc[best_idx]

print(f"\n{'=' * 60}")
print(f"  BEST TRIAL #{int(best['trial'])}:")
print(f"    conv1_filters={int(best['conv1_filters'])}, conv2_filters={int(best['conv2_filters'])}")
print(f"    cnn_lstm_hidden={int(best['cnn_lstm_hidden'])}, cnn_lstm_layers={int(best['cnn_lstm_layers'])}")
print(f"    path_a_dropout={best['path_a_dropout']:.3f}")
print(f"    Val RMSE[1-12]={best['val_rmse_1to12']:.1f}  Test MAE={best['test_mae_all']:.1f}")
print(f"    Params: {int(best['n_params']):,}")
print(f"{'=' * 60}")

# ── Plot 1: RMSE vs trial ──
fig, ax = plt.subplots(figsize=(10, 6))
trials = df_results["trial"].values
rmse_vals = df_results["val_rmse_1to12"].values
colors = ["#E74C3C" if i == best_idx else "#3498DB" for i in range(len(df_results))]
ax.scatter(trials, rmse_vals, c=colors, s=50, alpha=0.7, edgecolors="white", linewidth=0.5)
ax.axhline(best["val_rmse_1to12"], color="#E74C3C", linestyle="--", alpha=0.5,
           label=f"Best = {best['val_rmse_1to12']:.1f} kW")
ax.set_xlabel("Trial"); ax.set_ylabel("Val RMSE h=1~12 (kW)")
ax.set_title("Random Search — RMSE by Trial"); ax.legend(); ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rmse_vs_trial.png"), dpi=150); plt.close(fig)

# ── Plot 2: Hyperparameter importance (each param vs RMSE) ──
param_names = ["conv1_filters", "conv2_filters", "cnn_lstm_hidden",
               "cnn_lstm_layers", "path_a_dropout"]
param_labels = ["conv1_filters", "conv2_filters", "cnn_lstm_hidden",
                "cnn_lstm_layers", "path_a_dropout"]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, (pk, pl) in enumerate(zip(param_names, param_labels)):
    ax = axes[i]
    vals = df_results[pk].values
    r2 = np.corrcoef(vals, df_results["val_rmse_1to12"].values)[0, 1]
    ax.scatter(vals, rmse_vals, c="#3498DB", s=30, alpha=0.6, edgecolors="white")
    ax.set_xlabel(pl); ax.set_ylabel("Val RMSE (kW)")
    ax.set_title(f"{pl}  (r={r2:.3f})"); ax.grid(True, alpha=0.2)
    if pk == "cnn_lstm_layers":
        ax.set_xticks([1, 2])
    elif pk == "path_a_dropout":
        pass
fig.delaxes(axes[5])
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "param_importance.png"), dpi=150); plt.close(fig)

# ── Plot 3: Top 5 comparison ──
top5 = df_results.head(5)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Train best model for deep eval and get per-hour metrics
best_model = VMDLSTMHybrid(
    weather_dim=WEATHER_DIM, n_imfs=N_IMFS, cnn_out=CNN_OUT,
    output_dim=OUTPUT_DIM, trend_hidden=TREND_HIDDEN,
    fluct_hidden=FLUCT_HIDDEN, n_layers=N_LAYERS, dropout=PATH_B_DROPOUT,
    capacity=CAPACITY,
    conv1_filters=int(best["conv1_filters"]), conv2_filters=int(best["conv2_filters"]),
    conv_kernel=3, pool_size=2,
    cnn_lstm_hidden=int(best["cnn_lstm_hidden"]),
    cnn_lstm_layers=int(best["cnn_lstm_layers"]),
    path_a_dropout=float(best["path_a_dropout"]),
).to(DEVICE)

# Retrain best model for deeper test eval
train_loader = DataLoader(SeqDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
val_loader = DataLoader(SeqDataset(X_val, y_val), BATCH_SIZE)
optimizer = torch.optim.Adam(best_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
criterion = nn.MSELoss()

best_val_loss = float("inf")
best_ep = 0; patience_c = 0
train_losses, val_losses = [], []
for epoch in range(1, MAX_EPOCHS + 1):
    best_model.train()
    for x, y in train_loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad(); criterion(best_model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(best_model.parameters(), max_norm=1.0); optimizer.step()
    best_model.eval(); tl, vl = 0.0, 0.0; tn, vn = 0, 0
    with torch.no_grad():
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            tl += criterion(best_model(x), y).item() * x.size(0); tn += x.size(0)
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            vl += criterion(best_model(x), y).item() * x.size(0); vn += x.size(0)
    train_losses.append(tl / tn); val_losses.append(vl / vn)
    scheduler.step(val_losses[-1])
    if val_losses[-1] < best_val_loss:
        best_val_loss = val_losses[-1]; best_ep = epoch; patience_c = 0
    else:
        patience_c += 1
        if patience_c >= EARLY_STOP_PATIENCE:
            break

# Evaluate best model per hour on test set
test_loader = DataLoader(SeqDataset(X_test, y_test), BATCH_SIZE)
y_pred_t, y_true_t = predict_loader(best_model, test_loader, DEVICE)
y_pred_t = scaler_y.inverse_transform(y_pred_t)
y_pred_t = np.clip(y_pred_t, 0, CAPACITY)
y_true_t = scaler_y.inverse_transform(y_true_t)

mae_h = np.array([np.abs(y_true_t[:, h] - y_pred_t[:, h]).mean() for h in range(OUTPUT_DIM)])
rmse_h = np.array([np.sqrt(((y_true_t[:, h] - y_pred_t[:, h])**2).mean()) for h in range(OUTPUT_DIM)])

horizons = np.arange(1, OUTPUT_DIM + 1)

ax = axes[0]
ax.plot(horizons, mae_h, "o-", color="#E74C3C", markersize=4, label="MAE")
ax.plot(horizons, rmse_h, "s-", color="#2C3E50", markersize=4, label="RMSE")
ax.axvline(x=CNN_OUT + 0.5, color="gray", linestyle="--", alpha=0.5, label=f"CNN|VMD split")
ax.set_xlabel("Horizon (hours ahead)"); ax.set_ylabel("Error (kW)")
ax.set_title(f"Best Trial — Per-hour MAE/RMSE"); ax.legend(); ax.grid(True, alpha=0.2)
ax.set_xticks(range(1, 25, 3))

ax = axes[1]; ax.axis("off")
tbl_data = [
    ["Hyperparameter", "Best Value"],
    ["conv1_filters", str(int(best["conv1_filters"]))],
    ["conv2_filters", str(int(best["conv2_filters"]))],
    ["cnn_lstm_hidden", str(int(best["cnn_lstm_hidden"]))],
    ["cnn_lstm_layers", str(int(best["cnn_lstm_layers"]))],
    ["path_a_dropout", f"{best['path_a_dropout']:.3f}"],
    ["Total Params", f"{int(best['n_params']):,}"],
    ["Val RMSE (h=1~12)", f"{best['val_rmse_1to12']:.1f} kW"],
    ["Test RMSE (h=1~12)", f"{best['test_rmse_1to12']:.1f} kW"],
    ["Test MAE (all 24h)", f"{best['test_mae_all']:.1f} kW"],
    ["Best epoch", str(int(best["best_epoch"]))],
]
tbl = ax.table(cellText=tbl_data, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.4)
for j in range(2):
    tbl[0, j].set_facecolor("#2C3E50"); tbl[0, j].set_text_props(color="white", fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "top5_comparison.png"), dpi=150); plt.close(fig)

# ── Plot 4: Training curve for best model ──
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(train_losses) + 1), train_losses, label="Train Loss", color="#3498DB")
ax.plot(range(1, len(val_losses) + 1), val_losses, label="Val Loss", color="#E74C3C")
ax.axvline(x=best_ep, color="green", linestyle="--", alpha=0.5, label=f"Best epoch={best_ep}")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("Best Trial — Loss Curve"); ax.legend(); ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "best_trial_loss.png"), dpi=150); plt.close(fig)

# ── Plot 5: Parallel coordinates (top 10 highlights) ──
try:
    from pandas.plotting import parallel_coordinates
    df_plot = df_results.copy()
    # Normalize params for parallel plot
    for pk in param_names:
        if pk != "cnn_lstm_layers":
            df_plot[pk] = (df_plot[pk] - df_plot[pk].min()) / (df_plot[pk].max() - df_plot[pk].min() + 1e-8)
    df_plot["rmse_norm"] = (df_plot["val_rmse_1to12"] - df_plot["val_rmse_1to12"].min()) / \
                           (df_plot["val_rmse_1to12"].max() - df_plot["val_rmse_1to12"].min() + 1e-8)
    df_plot["rank"] = pd.qcut(df_plot["val_rmse_1to12"], q=3, labels=["Good", "Mid", "Poor"])

    fig, ax = plt.subplots(figsize=(12, 6))
    parallel_coordinates(
        df_plot, class_column="rank",
        cols=param_names + ["rmse_norm"],
        color=["#2ECC71", "#F39C12", "#E74C3C"],
        alpha=0.4, linewidth=0.8, ax=ax
    )
    ax.set_xticklabels(param_names + ["RMSE"]); ax.legend(title="Rank")
    ax.set_title("Parallel Coordinates — Parameter Combinations"); ax.grid(True, alpha=0.15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "parallel_coords.png"), dpi=150)
    plt.close(fig)
except ImportError:
    pass

print(f"\n  All plots saved to: {OUT_DIR}")
print(f"\nDone.")
