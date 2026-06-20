"""Pure LSTM baseline for 24h wind power forecasting.

对比 VMD-LSTM 混合模型，量化 VMD 分解 + CNN 双路径架构的实际增益。
使用相同的训练/验证/测试划分和超参数，确保公平对比。
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

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SEQ_LEN = 120
BATCH_SIZE = 64
EPOCHS = 200
LR = 5e-4
PATIENCE = 30
WEIGHT_DECAY = 5e-4
OUTPUT_DIM = 24
CAPACITY = 2000.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
OUTPUT_DIR = "outputs"

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)


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


class PureLSTM(nn.Module):
    """纯 LSTM 基线：气象特征直接预测 24h 功率，无 VMD 分解，无 CNN。"""
    def __init__(self, input_dim=8, hidden_size=128, num_layers=2,
                 dropout=0.3, output_dim=24):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)


def main():
    # ── 加载数据（与 hybrid 完全一致） ──
    df = pd.read_csv("data/wind_nc/output/wind_data.csv")
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
    print(f"  Total samples: {len(df)}")
    print(f"  Time range: {times[0]}  ->  {times[-1]}")
    print(f"  Device: {DEVICE}")

    # ── 缩放与构建序列（仅气象特征，无 IMFs） ──
    y_raw = df[target_cols].values.astype(np.float32)
    train_years_mask = pd.DatetimeIndex(times).year.isin([2024, 2025])

    scaler_y = StandardScaler()
    scaler_y.fit(y_raw[train_years_mask])
    y_scaled = scaler_y.transform(y_raw)

    weather_raw = df[weather_cols].values.astype(np.float32)
    weather_scaler = StandardScaler()
    weather_scaler.fit(weather_raw[train_years_mask])
    weather_scaled = weather_scaler.transform(weather_raw)

    X_seq, y_seq, idx = create_sequences(weather_scaled, y_scaled, SEQ_LEN)
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

    print(f"\n[Pure LSTM] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")
    print(f"  Feature dim: {weather_scaled.shape[1]} (weather only)")

    # ── 训练 ──
    print("\n" + "=" * 60)
    print("  Pure LSTM Baseline — weather features only, no VMD/CNN")
    print("=" * 60)

    model = PureLSTM(
        input_dim=len(weather_cols),
        hidden_size=128,
        num_layers=2,
        dropout=0.3,
        output_dim=OUTPUT_DIM,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model params: {total_params:,}")

    train_loader = DataLoader(SequenceDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SequenceDataset(X_val, y_val), BATCH_SIZE)

    model = train_model(
        model, train_loader, val_loader, DEVICE,
        epochs=EPOCHS, lr=LR, patience=PATIENCE,
        weight_decay=WEIGHT_DECAY,
        model_path=os.path.join(OUTPUT_DIR, "baseline_lstm.pth"),
    )

    # ── 测试集评估 ──
    y_pred_scaled = predict_sequences(model, X_test, BATCH_SIZE, DEVICE)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_pred = np.clip(y_pred, 0, CAPACITY)
    y_true = scaler_y.inverse_transform(y_test)

    mae_h, rmse_h, r2_h = per_horizon_metrics(y_true, y_pred)
    mae_all, rmse_all, r2_all = overall_metrics(y_true, y_pred)

    print(f"\n[Pure LSTM Results]")
    print(f"  Overall  MAE={mae_all:.2f} kW  RMSE={rmse_all:.2f} kW  R2={r2_all:.4f}")
    print(f"  NMAE={mae_all / CAPACITY * 100:.2f}%  NRMSE={rmse_all / CAPACITY * 100:.2f}%")
    print(f"  h=1  MAE={mae_h[0]:.1f} kW  R2={r2_h[0]:.4f}")
    print(f"  h=4  MAE={mae_h[3]:.1f} kW  R2={r2_h[3]:.4f}")
    print(f"  h=8  MAE={mae_h[7]:.1f} kW  R2={r2_h[7]:.4f}")
    print(f"  h=12 MAE={mae_h[11]:.1f} kW R2={r2_h[11]:.4f}")
    print(f"  h=24 MAE={mae_h[23]:.1f} kW R2={r2_h[23]:.4f}")

    # ── 保存预测结果 ──
    col_names = ["valid_time"] + [f"power_h{h}" for h in range(1, OUTPUT_DIM + 1)]
    pred_out = pd.DataFrame(columns=col_names)
    pred_out["valid_time"] = test_times
    for h in range(OUTPUT_DIM):
        pred_out[f"power_h{h+1}"] = y_pred[:, h]
    pred_out.to_csv(os.path.join(OUTPUT_DIR, "baseline_lstm_predictions.csv"), index=False)
    print("  Saved -> outputs/baseline_lstm_predictions.csv")

    # ── 加载 hybrid 结果并对比 ──
    hybrid_csv = os.path.join(OUTPUT_DIR, "vmd_hybrid_predictions.csv")
    if not os.path.exists(hybrid_csv):
        print("\n  [WARN] hybrid predictions not found, skip comparison. Run forecast_vmd_hybrid.py first.")
        return

    hybrid_df = pd.read_csv(hybrid_csv)
    hybrid_df["valid_time"] = pd.to_datetime(hybrid_df["valid_time"])
    merged = pred_out.merge(hybrid_df, on="valid_time", how="inner",
                            suffixes=("_base", "_hybrid"))

    n = len(merged)
    y_base = np.zeros((n, OUTPUT_DIM))
    y_hyb  = np.zeros((n, OUTPUT_DIM))
    for h in range(OUTPUT_DIM):
        y_base[:, h] = merged[f"power_h{h+1}_base"].values
        y_hyb[:, h]  = merged[f"power_h{h+1}_hybrid"].values

    merged_times = merged["valid_time"]

    # 用数据中的真实功率标签对齐
    y_true_aligned = np.zeros((n, OUTPUT_DIM))
    for i, t in enumerate(merged_times):
        row = df[df["valid_time"] == t]
        if len(row) > 0:
            for h_idx in range(OUTPUT_DIM):
                y_true_aligned[i, h_idx] = row[f"power_t{h_idx+1}"].values[0]

    mae_h_base, _, r2_h_base = per_horizon_metrics(y_true_aligned, y_base)
    mae_h_hyb,  _, r2_h_hyb  = per_horizon_metrics(y_true_aligned, y_hyb)
    mae_base_all, rmse_base_all, r2_base_all = overall_metrics(y_true_aligned, y_base)
    mae_hyb_all,  rmse_hyb_all,  r2_hyb_all  = overall_metrics(y_true_aligned, y_hyb)

    # ── 对比图表 ──
    horizons = np.arange(1, OUTPUT_DIM + 1)
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle("Pure LSTM vs VMD-LSTM Hybrid — 24h Forecast Comparison",
                 fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(horizons, mae_h_base, "o-", label="Pure LSTM", color="#E74C3C", markersize=5)
    ax.plot(horizons, mae_h_hyb, "s-", label="VMD-LSTM Hybrid", color="#2C3E50", markersize=5)
    ax.set_xlabel("Horizon (hours ahead)"); ax.set_ylabel("MAE (kW)")
    ax.set_title(f"MAE by Horizon | Pure={mae_base_all:.0f}kW  Hybrid={mae_hyb_all:.0f}kW")
    ax.legend(); ax.grid(True, alpha=0.25); ax.set_xticks(range(1, 25, 3))

    ax = axes[0, 1]
    x = np.arange(len(horizons))
    w = 0.35
    ax.bar(x - w/2, r2_h_base, w, label="Pure LSTM", color="#E74C3C", alpha=0.85)
    ax.bar(x + w/2, r2_h_hyb, w, label="VMD-LSTM Hybrid", color="#2C3E50", alpha=0.85)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("Horizon (hours ahead)"); ax.set_ylabel("R2")
    ax.set_title(f"R2 by Horizon | Pure={r2_base_all:.3f}  Hybrid={r2_hyb_all:.3f}")
    ax.legend(); ax.set_xticks(range(0, 24, 3)); ax.set_xticklabels(range(1, 25, 3))

    ax = axes[1, 0]
    ax.scatter(y_base.flatten(), y_hyb.flatten(), alpha=0.08, s=2, color="#7B1FA2")
    pmax = max(y_base.max(), y_hyb.max()) * 1.05
    ax.plot([0, pmax], [0, pmax], "r--", linewidth=1)
    ax.set_xlim(0, pmax); ax.set_ylim(0, pmax)
    ax.set_xlabel("Pure LSTM (kW)"); ax.set_ylabel("VMD-LSTM Hybrid (kW)")
    ax.set_title("Prediction Comparison (all horizons)")

    ax = axes[1, 1]
    tbl_data = [
        ["Metric", "Pure LSTM", "VMD-Hybrid", "Delta"],
        ["MAE (kW)",  f"{mae_base_all:.1f}", f"{mae_hyb_all:.1f}",
         f"{mae_base_all - mae_hyb_all:+.1f}"],
        ["RMSE (kW)", f"{rmse_base_all:.1f}", f"{rmse_hyb_all:.1f}",
         f"{rmse_base_all - rmse_hyb_all:+.1f}"],
        ["R2",        f"{r2_base_all:.4f}", f"{r2_hyb_all:.4f}",
         f"{r2_hyb_all - r2_base_all:+.4f}"],
        ["NMAE",      f"{mae_base_all/CAPACITY*100:.2f}%", f"{mae_hyb_all/CAPACITY*100:.2f}%", ""],
        ["MAE h=1",   f"{mae_h_base[0]:.1f}", f"{mae_h_hyb[0]:.1f}",
         f"{mae_h_base[0] - mae_h_hyb[0]:+.1f}"],
        ["MAE h=12",  f"{mae_h_base[11]:.1f}", f"{mae_h_hyb[11]:.1f}",
         f"{mae_h_base[11] - mae_h_hyb[11]:+.1f}"],
        ["MAE h=24",  f"{mae_h_base[23]:.1f}", f"{mae_h_hyb[23]:.1f}",
         f"{mae_h_base[23] - mae_h_hyb[23]:+.1f}"],
    ]
    ax.axis("off")
    tbl = ax.table(cellText=tbl_data, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1.15, 1.5)
    for j in range(4):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Metrics Summary", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "baseline_vs_hybrid.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  Plot saved -> outputs/baseline_vs_hybrid.png")

    print(f"\n[Comparison Summary]")
    print(f"  {'Metric':<12} {'Pure LSTM':>12} {'VMD-Hybrid':>12} {'Improvement':>12}")
    print(f"  {'MAE (kW)':<12} {mae_base_all:>12.2f} {mae_hyb_all:>12.2f} {mae_base_all - mae_hyb_all:>+12.2f}")
    print(f"  {'RMSE (kW)':<12} {rmse_base_all:>12.2f} {rmse_hyb_all:>12.2f} {rmse_base_all - rmse_hyb_all:>+12.2f}")
    print(f"  {'R2':<12} {r2_base_all:>12.4f} {r2_hyb_all:>12.4f} {r2_hyb_all - r2_base_all:>+12.4f}")
    print(f"  {'NMAE':<12} {mae_base_all/CAPACITY*100:>11.2f}% {mae_hyb_all/CAPACITY*100:>11.2f}%")

    print("\nDone.")


if __name__ == "__main__":
    main()
