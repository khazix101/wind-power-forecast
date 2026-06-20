"""VMD-LSTM Hybrid 24h Wind Power Forecasting.

Pipeline:
  1. Load wind_data.csv (point_id=1)
  2. Compute 24h-ahead power labels via Vestas V90 physics
  3. VMD-decompose power → 4 IMF channels
  4. Build sequences: 120h IMFs → 24h power
  5. Train VMD-LSTM Hybrid (dual-path: Trend + Fluctuation)
  6. Evaluate & plot
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import os
import time

from vmd_utils import VMDDecomposer, decompose_by_domain
from vmd_hybrid_model import VMDLSTMHybrid

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ═══════════════════════════════════════════════════════════
# 0. Config
# ═══════════════════════════════════════════════════════════
SEQ_LEN = 120
BATCH_SIZE = 64
EPOCHS = 200
LR = 5e-4
DROPOUT = 0.3
PATIENCE = 30
WEIGHT_DECAY = 5e-4
OUTPUT_DIM = 24
CAPACITY = 2000.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
CNN_OUT = 8
WEATHER_DIM = 8

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)


# ═══════════════════════════════════════════════════════════
# 1. Utilities (inlined from lstm_utils.py)
# ═══════════════════════════════════════════════════════════
class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def create_sequences(features, targets, seq_len):
    X_list, y_list, idx_list = [], [], []
    n = len(features)
    for i in range(seq_len - 1, n):
        X_list.append(features[i - seq_len + 1 : i + 1])
        y_list.append(targets[i])
        idx_list.append(i)
    return (np.array(X_list, dtype=np.float32),
            np.array(y_list, dtype=np.float32),
            np.array(idx_list))


def per_horizon_metrics(y_true, y_pred):
    out_dim = y_true.shape[1]
    mae_h = np.zeros(out_dim)
    rmse_h = np.zeros(out_dim)
    r2_h = np.zeros(out_dim)
    for h in range(out_dim):
        mae_h[h] = mean_absolute_error(y_true[:, h], y_pred[:, h])
        rmse_h[h] = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
        r2_h[h] = r2_score(y_true[:, h], y_pred[:, h])
    return mae_h, rmse_h, r2_h


def overall_metrics(y_true, y_pred):
    yt = y_true.flatten()
    yp = y_pred.flatten()
    return (mean_absolute_error(yt, yp),
            np.sqrt(mean_squared_error(yt, yp)),
            r2_score(yt, yp))


@torch.no_grad()
def predict_sequences(model, X, batch_size, device):
    loader = DataLoader(TensorDataset(torch.from_numpy(X)), batch_size=batch_size)
    preds = []
    for (x,) in loader:
        x = x.to(device)
        preds.append(model(x).cpu().numpy())
    return np.concatenate(preds, axis=0)


def _run_epoch(model, loader, optimizer, criterion, device, training):
    if training:
        model.train()
    else:
        model.eval()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if training:
            optimizer.zero_grad()
        with torch.set_grad_enabled(training):
            loss = criterion(model(x), y)
        if training:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
    return total_loss / n


def train_model(model, train_loader, val_loader, device,
                epochs=150, lr=1e-4, patience=20, weight_decay=5e-4,
                model_path="model.pth"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss = _run_epoch(model, train_loader, optimizer, criterion, device, True)
        val_loss = _run_epoch(model, val_loader, optimizer, criterion, device, False)
        scheduler.step(val_loss)

        if epoch % 20 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch} (best: {best_epoch})")
                break

    model.load_state_dict(torch.load(model_path, map_location=device))
    print(f"  Loaded best model (epoch {best_epoch}) | Val Loss: {best_val_loss:.6f}")
    return model


# ═══════════════════════════════════════════════════════════
# 2. Power curve (Vestas V90-2.0MW)
# ═══════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════
# 3. Load & prepare data
# ═══════════════════════════════════════════════════════════
df = pd.read_csv("data/wind_nc/output/wind_data.csv")
df["valid_time"] = pd.to_datetime(df["valid_time"])
df = df[df["point_id"] == 1].sort_values("valid_time").reset_index(drop=True)

# ── 24h target power labels ──
target_cols = []
for h in range(1, OUTPUT_DIM + 1):
    col = f"power_t{h}"
    ws_shifted = df["wind_speed_100m"].shift(-h).values
    rho_shifted = df["air_density"].shift(-h).values
    df[col] = compute_power(ws_shifted, rho_shifted)
    target_cols.append(col)
df = df.dropna().reset_index(drop=True)

# ── Time encoding features ──
hour = pd.DatetimeIndex(df["valid_time"]).hour
df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
df["hour_cos"] = np.cos(2 * np.pi * hour / 24)

# ── Current power (same-timestamp, not a future label) ──
df["power_current"] = compute_power(df["wind_speed_100m"].values, df["air_density"].values)

# ── Weather features for Path A (CNN-LSTM short term) ──
weather_cols = ["power_current", "wind_speed_100m", "air_density",
                "u100", "v100", "t2m", "hour_sin", "hour_cos"]

times = df["valid_time"].values
print(f"  Total samples (after dropna): {len(df)}")
print(f"  Time range: {times[0]}  →  {times[-1]}")
print(f"  Device: {DEVICE}")

# ═══════════════════════════════════════════════════════════
# 4. VMD decomposition — per-domain to prevent data leakage
# ═══════════════════════════════════════════════════════════
power_raw = df[target_cols].values[:, 0].astype(float)
train_years_mask = pd.DatetimeIndex(times).year.isin([2024, 2025])

# Build sample-level domain masks (same index space as power_raw)
sample_years = pd.DatetimeIndex(times).year
sample_months = pd.DatetimeIndex(times).month
domain_train_mask = sample_years.isin([2024, 2025]) & ~((sample_years == 2025) & (sample_months >= 10))
domain_val_mask   = (sample_years == 2025) & (sample_months >= 10)
domain_test_mask  = sample_years == 2026

OUTPUT_DIR = "outputs"
VMD_CACHE = os.path.join(OUTPUT_DIR, "vmd_cache")
os.makedirs(VMD_CACHE, exist_ok=True)

print(f"\n[VMD] Domain-separated decomposition (alpha=500, K=4) ...")
t0 = time.time()
imfs, omegas = decompose_by_domain(
    power_raw,
    [("train", domain_train_mask),
     ("val",   domain_val_mask),
     ("test",  domain_test_mask)],
    K=4, alpha=500, tol=1e-7, max_iter=500, seed=SEED,
    cache_dir=VMD_CACHE,
)
for name, omega in omegas.items():
    print(f"    {name} omega: {np.round(omega, 4)}")
print(f"  VMD total: {time.time() - t0:.1f}s  |  IMFs shape: {imfs.shape}")

# ═══════════════════════════════════════════════════════════
# 5. Scale & build sequences
# ═══════════════════════════════════════════════════════════
y_raw = df[target_cols].values.astype(np.float32)

scaler_y = StandardScaler()
scaler_y.fit(y_raw[train_years_mask])
y_scaled = scaler_y.transform(y_raw)

# ── Scale weather features (Path A) ──
weather_raw = df[weather_cols].values.astype(np.float32)
weather_scaler = StandardScaler()
weather_scaler.fit(weather_raw[train_years_mask])
weather_scaled = weather_scaler.transform(weather_raw)

# ── Scale IMFs fit on train domain only ──
imfs_scaler = StandardScaler()
imfs_scaler.fit(imfs[domain_train_mask])
imfs_scaled = imfs_scaler.transform(imfs)

# ── Combine weather + IMFs into single feature matrix ──
features = np.concatenate([weather_scaled, imfs_scaled], axis=1).astype(np.float32)

X_seq, y_seq, idx = create_sequences(features, y_scaled, SEQ_LEN)
seq_times = times[idx]
seq_years = pd.DatetimeIndex(seq_times).year
seq_months = pd.DatetimeIndex(seq_times).month

train_mask = seq_years.isin([2024, 2025]) & ~((seq_years == 2025) & (seq_months >= 10))
val_mask   = (seq_years == 2025) & (seq_months >= 10)
test_mask  = seq_years == 2026

X_train, y_train = X_seq[train_mask], y_seq[train_mask]
X_val,   y_val   = X_seq[val_mask],   y_seq[val_mask]
X_test,  y_test  = X_seq[test_mask],  y_seq[test_mask]
test_times = seq_times[test_mask]

print(f"\n[VMD-Hybrid] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
print(f"  Feature shape: {features.shape[1]} (weather={WEATHER_DIM} + imfs=4)")

# ═══════════════════════════════════════════════════════════
# 6. Train VMD-LSTM Hybrid
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print(f"  CNN-LSTM({CNN_OUT}h) + VMD-LSTM({OUTPUT_DIM - CNN_OUT}h) Hybrid Training")
print("=" * 60)

model = VMDLSTMHybrid(
    weather_dim=WEATHER_DIM,
    n_imfs=4,
    cnn_out=CNN_OUT,
    output_dim=OUTPUT_DIM,
    trend_hidden=128,
    fluct_hidden=64,
    n_layers=1,
    dropout=DROPOUT,
    fc_hidden=64,
    path_b_dropout=0.444,
    capacity=CAPACITY,
    # Best Path A hyperparams from random search
    conv1_filters=128, conv2_filters=192,
    conv_kernel=3, pool_size=2,
    cnn_lstm_hidden=32, cnn_lstm_layers=1,
    path_a_dropout=0.143,
).to(DEVICE)

train_loader = DataLoader(SequenceDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(SequenceDataset(X_val, y_val), BATCH_SIZE)

model = train_model(
    model, train_loader, val_loader, DEVICE,
    epochs=EPOCHS, lr=LR, patience=PATIENCE,
    weight_decay=WEIGHT_DECAY,
    model_path=os.path.join(OUTPUT_DIR, "vmd_hybrid.pth"),
)

# ═══════════════════════════════════════════════════════════
# 7. Evaluate on test set
# ═══════════════════════════════════════════════════════════
y_pred_scaled = predict_sequences(model, X_test, BATCH_SIZE, DEVICE)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_pred = np.clip(y_pred, 0, CAPACITY)
y_true = scaler_y.inverse_transform(y_test)

mae_h, rmse_h, r2_h = per_horizon_metrics(y_true, y_pred)
mae_all, rmse_all, r2_all = overall_metrics(y_true, y_pred)

print(f"\n[VMD-Hybrid Results]")
print(f"  Overall  MAE={mae_all:.2f} kW  RMSE={rmse_all:.2f} kW  R2={r2_all:.4f}")
print(f"  NMAE={mae_all / CAPACITY * 100:.2f}%  NRMSE={rmse_all / CAPACITY * 100:.2f}%")
print(f"  Per-horizon min/mean/max MAE:  {mae_h.min():.1f} / {mae_h.mean():.1f} / {mae_h.max():.1f} kW")
print(f"  Per-horizon min/mean/max R2:   {r2_h.min():.4f} / {r2_h.mean():.4f} / {r2_h.max():.4f}")
print(f"  h=1  MAE={mae_h[0]:.1f} kW  R2={r2_h[0]:.4f}  (CNN-LSTM)")
print(f"  h=4  MAE={mae_h[3]:.1f} kW  R2={r2_h[3]:.4f}  (CNN-LSTM)")
print(f"  h=6  MAE={mae_h[5]:.1f} kW  R2={r2_h[5]:.4f}  (VMD-LSTM)")
print(f"  h=12 MAE={mae_h[11]:.1f} kW R2={r2_h[11]:.4f}  (VMD-LSTM)")
print(f"  h=24 MAE={mae_h[23]:.1f} kW R2={r2_h[23]:.4f}  (VMD-LSTM)")

# ═══════════════════════════════════════════════════════════
# 8. Save predictions & IMFs
# ═══════════════════════════════════════════════════════════
col_names = ["valid_time"] + [f"power_h{h}" for h in range(1, OUTPUT_DIM + 1)]
pred_out = pd.DataFrame(columns=col_names)
pred_out["valid_time"] = test_times
for h in range(OUTPUT_DIM):
    pred_out[f"power_h{h+1}"] = y_pred[:, h]
pred_out.to_csv(os.path.join(OUTPUT_DIR, "vmd_hybrid_predictions.csv"), index=False)
print(f"  Saved -> outputs/vmd_hybrid_predictions.csv")

np.savez(os.path.join(OUTPUT_DIR, "vmd_imfs.npz"), imfs=imfs, omega_train=omegas["train"],
         omega_val=omegas["val"], omega_test=omegas["test"])
print(f"  Saved -> outputs/vmd_imfs.npz")

# ═══════════════════════════════════════════════════════════
# 9. Plots
# ═══════════════════════════════════════════════════════════
horizons = np.arange(1, OUTPUT_DIM + 1)

fig, axes = plt.subplots(2, 3, figsize=(22, 13))
fig.suptitle("CNN-LSTM(1-8h) + VMD-LSTM(9-24h) — 24h Wind Power Forecast", fontsize=15, fontweight="bold")

# (0,0) Per-horizon error curves
ax = axes[0, 0]
ax.plot(horizons, mae_h, "o-", label="MAE", color="#E74C3C", markersize=4)
ax.plot(horizons, rmse_h, "s-", label="RMSE", color="#2C3E50", markersize=4)
ax.set_xlabel("Horizon (hours ahead)")
ax.set_ylabel("Error (kW)")
ax.set_title("Error by Forecast Horizon")
ax.legend(fontsize=8); ax.set_xticks(range(1, OUTPUT_DIM + 1, 3))
ax.grid(True, alpha=0.3)

# (0,1) R2 by horizon
ax = axes[0, 1]
colors_r2 = ["#2ECC71" if v > 0 else "#E74C3C" for v in r2_h]
ax.bar(horizons, r2_h, color=colors_r2, edgecolor="white")
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_xlabel("Horizon (hours ahead)")
ax.set_ylabel("R2")
ax.set_title(f"R2 by Horizon (avg={r2_h.mean():.3f})")
ax.set_xticks(range(1, OUTPUT_DIM + 1, 3))

# (0,2) Sample 24h profiles
ax = axes[0, 2]
for s in [0, len(y_true) // 3, len(y_true) * 2 // 3]:
    ax.plot(horizons, y_true[s], "o-", markersize=4, alpha=0.7,
            label=f"True #{s}" if s == 0 else f"True #{s}")
    ax.plot(horizons, y_pred[s], "s--", markersize=4, alpha=0.7,
            label=f"Pred #{s}")
ax.set_xlabel("Hours ahead"); ax.set_ylabel("Power (kW)")
ax.set_title("Sample 24h Power Profiles"); ax.legend(fontsize=7, ncol=2)

# (1,0) Scatter
ax = axes[1, 0]
ax.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.06, s=3, color="#E74C3C")
pmax = max(y_true.max(), y_pred.max()) * 1.05
ax.plot([0, pmax], [0, pmax], "r--", linewidth=1)
ax.set_xlim(0, pmax); ax.set_ylim(0, pmax)
ax.set_xlabel("Actual (kW)"); ax.set_ylabel("Predicted (kW)")
ax.set_title(f"Scatter | MAE={mae_all:.0f}kW  RMSE={rmse_all:.0f}kW  R2={r2_all:.3f}")

# (1,1) Error distribution
ax = axes[1, 1]
res = (y_true - y_pred).flatten()
ax.hist(res, bins=80, color="#95A5A6", edgecolor="white", alpha=0.85)
ax.axvline(0, color="#E74C3C", linewidth=2, linestyle="--")
ax.axvline(res.mean(), color="black", linewidth=1.5, linestyle="-",
           label=f"bias={res.mean():.1f}kW")
ax.set_xlabel("Residual (kW)")
ax.set_title(f"Error Dist | std={res.std():.0f}kW  NMAE={mae_all/CAPACITY*100:.1f}%")
ax.legend(fontsize=8)

# (1,2) Power curve overlay
ax = axes[1, 2]
ws_range = np.linspace(0, 28, 200)
p_std = power_curve_v90(wind_at_hub(ws_range))
ax.plot(ws_range, p_std, "b-", linewidth=2.5, label="Ideal curve")
ax.fill_between(ws_range, p_std * 0.8, p_std * 1.2, alpha=0.08, color="blue")
ws_test_flat = df["wind_speed_100m"].values[:min(3000, len(y_true) * OUTPUT_DIM)]
pwr_test_flat = y_true.flatten()[:min(3000, len(y_true) * OUTPUT_DIM)]
ax.scatter(ws_test_flat[::5], pwr_test_flat[::5], s=3, alpha=0.3, color="red")
ax.set_xlabel("Wind Speed 100m (m/s)"); ax.set_ylabel("Power (kW)")
ax.set_title("Power Curve + Test Labels"); ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vmd_hybrid_results.png"), dpi=150, bbox_inches="tight")
plt.close()
print("  Plot saved -> outputs/vmd_hybrid_results.png")

# ── IMF decomposition visualisation ──
fig2, axes2 = plt.subplots(5, 1, figsize=(18, 12), sharex=True)
fig2.suptitle("VMD Decomposition — Train Domain Power (first 2000 hours)",
              fontsize=14, fontweight="bold")
n_plot = min(2000, len(power_raw))
t_plot = np.arange(n_plot)
axes2[0].plot(t_plot, power_raw[:n_plot], color="black", linewidth=0.8)
axes2[0].set_ylabel("Power (kW)")
axes2[0].set_title("Original Power Series")
for k in range(4):
    axes2[k + 1].plot(t_plot, imfs[:n_plot, k], linewidth=0.6)
    axes2[k + 1].set_ylabel(f"IMF {k + 1}")
    axes2[k + 1].set_title(f"IMF {k + 1}  (omega_train={omegas['train'][k]:.4f})")
axes2[-1].set_xlabel("Time index (hours)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "vmd_decomposition.png"), dpi=120, bbox_inches="tight")
plt.close()
print("  Plot saved -> outputs/vmd_decomposition.png")

print("\nDone.")
