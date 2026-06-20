"""Find optimal split point between Path A (CNN-LSTM) and Path B (VMD-LSTM).

Tests different cnn_out values (hours predicted by CNN-LSTM) to determine
which gives the best overall 24h wind power forecast.

Outputs:
  - results.csv: full metric table
  - split_comparison.png: performance vs split point chart
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os, sys, time, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "forecast_tsp"))
from vmd_utils import decompose_by_domain
from vmd_hybrid_model import VMDLSTMHybrid
from hyper_tune_common import (
    SEQ_LEN, BATCH_SIZE, OUTPUT_DIM, CAPACITY, WEATHER_DIM, SEED,
    VMD_CACHE_DIR, WEATHER_COLS,
    load_wind_data, get_sample_domain_masks, compute_power,
    clean_path_a_param_defaults, clean_path_b_param_defaults,
)

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
EPOCHS = 100
LR = 5e-4
PATIENCE = 15
WEIGHT_DECAY = 5e-4

# Best parameters from hyperparameter search
pa = clean_path_a_param_defaults()
pb = clean_path_b_param_defaults()

# Split point candidates
CNN_OUT_CANDIDATES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]

OUT_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(VMD_CACHE_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════
# Data loading (once)
# ══════════════════════════════════════════════
print("=" * 60)
print("  Optimal A/B Split Point Search")
print("=" * 60)

df = load_wind_data()
times = df["valid_time"].values
train_years_mask = pd.DatetimeIndex(times).year.isin([2024, 2025])

target_cols = [f"power_t{h}" for h in range(1, OUTPUT_DIM + 1)]

print(f"  Data: {len(df)} rows  |  Device: {DEVICE}")

# ── Scales ──
y_raw = df[target_cols].values.astype(np.float32)
scaler_y = StandardScaler()
scaler_y.fit(y_raw[train_years_mask])
y_scaled = scaler_y.transform(y_raw)

weather_raw = df[WEATHER_COLS].values.astype(np.float32)
weather_scaler = StandardScaler()
weather_scaler.fit(weather_raw[train_years_mask])
weather_scaled = weather_scaler.transform(weather_raw)

# ── VMD per-domain (shared cache) ──
dom_tr, dom_va, dom_te = get_sample_domain_masks(times)

print(f"\n[VMD] Domain-separated (alpha=500, shared cache={VMD_CACHE_DIR}) ...")
t0 = time.time()
imfs, omegas = decompose_by_domain(
    df[target_cols].values[:, 0].astype(float),
    [("train", dom_tr), ("val", dom_va), ("test", dom_te)],
    K=4, alpha=500, tol=1e-7, max_iter=500, seed=SEED,
    cache_dir=VMD_CACHE_DIR,
)
print(f"  VMD total: {time.time() - t0:.1f}s")

imfs_scaler = StandardScaler()
imfs_scaler.fit(imfs[dom_tr])
imfs_scaled = imfs_scaler.transform(imfs)

# ── Build sequences (same for all split points) ──
features = np.concatenate([weather_scaled, imfs_scaled], axis=1).astype(np.float32)
X_list, y_list, idx_list = [], [], []
n = len(features)
for i in range(SEQ_LEN - 1, n):
    X_list.append(features[i - SEQ_LEN + 1: i + 1])
    y_list.append(y_scaled[i])
    idx_list.append(i)
X_seq = np.array(X_list, dtype=np.float32)
y_seq = np.array(y_list, dtype=np.float32)
idx_seq = np.array(idx_list)
seq_times = times[idx_seq]
seq_years = pd.DatetimeIndex(seq_times).year
seq_months = pd.DatetimeIndex(seq_times).month

train_mask = seq_years.isin([2024, 2025]) & ~((seq_years == 2025) & (seq_months >= 10))
val_mask   = (seq_years == 2025) & (seq_months >= 10)
test_mask  = seq_years == 2026

X_train, y_train = X_seq[train_mask], y_seq[train_mask]
X_val,   y_val   = X_seq[val_mask],   y_seq[val_mask]
X_test,  y_test  = X_seq[test_mask],  y_seq[test_mask]

print(f"  Sequences: Train={len(X_train)}  Val={len(X_val)}  Test={len(X_test)}\n")


# ══════════════════════════════════════════════
# Dataset & training helpers
# ══════════════════════════════════════════════
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def _run_epoch(model, loader, optimizer, criterion, training):
    if training:
        model.train()
    else:
        model.eval()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
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


@torch.no_grad()
def predict_all(model, X, batch_size):
    loader = DataLoader(torch.utils.data.TensorDataset(
        torch.from_numpy(X.astype(np.float32))), batch_size=batch_size)
    preds = []
    for (x,) in loader:
        preds.append(model(x.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds, axis=0)


def train_eval_split(cnn_out):
    """Train model with given cnn_out, return metrics dict."""
    model = VMDLSTMHybrid(
        weather_dim=WEATHER_DIM, n_imfs=4, cnn_out=cnn_out,
        output_dim=OUTPUT_DIM,
        trend_hidden=pb["trend_hidden"], fluct_hidden=pb["fluct_hidden"],
        n_layers=pb["n_layers"], dropout=0.3,
        fc_hidden=pb["fc_hidden"], path_b_dropout=pb["path_b_dropout"],
        capacity=CAPACITY,
        conv1_filters=pa["conv1_filters"], conv2_filters=pa["conv2_filters"],
        conv_kernel=pa["conv_kernel"], pool_size=pa["pool_size"],
        cnn_lstm_hidden=pa["cnn_lstm_hidden"], cnn_lstm_layers=pa["cnn_lstm_layers"],
        path_a_dropout=pa["path_a_dropout"],
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())

    train_loader = DataLoader(SeqDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SeqDataset(X_val, y_val), BATCH_SIZE)
    test_loader  = DataLoader(SeqDataset(X_test, y_test), BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, EPOCHS + 1):
        _run_epoch(model, train_loader, optimizer, criterion, True)
        val_loss = _run_epoch(model, val_loader, optimizer, criterion, False)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    if best_epoch == 0:
        best_epoch = epoch

    model.eval()

    # Test predictions
    y_pred_scaled = predict_all(model, X_test, BATCH_SIZE)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_pred = np.clip(y_pred, 0, CAPACITY)
    y_true = scaler_y.inverse_transform(y_test)

    # Overall metrics
    mae_all = float(mean_absolute_error(y_true.flatten(), y_pred.flatten()))
    rmse_all = float(np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten())))
    r2_all = float(r2_score(y_true.flatten(), y_pred.flatten()))

    # Per-path metrics
    a_mask = np.arange(0, cnn_out)
    b_mask = np.arange(cnn_out, OUTPUT_DIM)

    yt_a = y_true[:, a_mask]; yp_a = y_pred[:, a_mask]
    yt_b = y_true[:, b_mask]; yp_b = y_pred[:, b_mask]

    mae_a = float(mean_absolute_error(yt_a.flatten(), yp_a.flatten()))
    rmse_a = float(np.sqrt(mean_squared_error(yt_a.flatten(), yp_a.flatten())))
    r2_a = float(r2_score(yt_a.flatten(), yp_a.flatten()))

    mae_b = float(mean_absolute_error(yt_b.flatten(), yp_b.flatten()))
    rmse_b = float(np.sqrt(mean_squared_error(yt_b.flatten(), yp_b.flatten())))
    r2_b = float(r2_score(yt_b.flatten(), yp_b.flatten()))

    # Per-horizon detail
    mae_h = np.array([mean_absolute_error(y_true[:, h], y_pred[:, h])
                      for h in range(OUTPUT_DIM)])
    r2_h = np.array([r2_score(y_true[:, h], y_pred[:, h])
                     for h in range(OUTPUT_DIM)])

    del model
    torch.cuda.empty_cache()

    return {
        "cnn_out": cnn_out,
        "vmd_out": OUTPUT_DIM - cnn_out,
        "n_params": n_params,
        "best_epoch": best_epoch,
        "mae_all": round(mae_all, 2),
        "rmse_all": round(rmse_all, 2),
        "r2_all": round(r2_all, 4),
        "mae_a": round(mae_a, 2),
        "rmse_a": round(rmse_a, 2),
        "r2_a": round(r2_a, 4),
        "mae_b": round(mae_b, 2),
        "rmse_b": round(rmse_b, 2),
        "r2_b": round(r2_b, 4),
        "mae_h": mae_h,
        "r2_h": r2_h,
    }


# ══════════════════════════════════════════════
# Run all split points
# ══════════════════════════════════════════════
print(f"{'cnn_out':>7s}  {'VMD_out':>7s}  {'#Params':>10s}  "
      f"{'MAE_all':>8s}  {'R2_all':>8s}  {'MAE_A':>8s}  {'MAE_B':>8s}  {'Ep':>4s}  {'Time':>6s}")
print("-" * 90)

results = []
for cnn_out in CNN_OUT_CANDIDATES:
    t0 = time.time()
    r = train_eval_split(cnn_out)
    elapsed = time.time() - t0
    results.append(r)

    print(f"{r['cnn_out']:7d}  {r['vmd_out']:7d}  {r['n_params']:>10,}  "
          f"{r['mae_all']:>8.1f}  {r['r2_all']:>8.4f}  "
          f"{r['mae_a']:>8.1f}  {r['mae_b']:>8.1f}  "
          f"{r['best_epoch']:>4d}  {elapsed:>5.0f}s")

# ── Save CSV ──
df_res = pd.DataFrame([{k: v for k, v in r.items()
                        if k not in ("mae_h", "r2_h")}
                       for r in results])
csv_path = os.path.join(OUT_DIR, "results.csv")
df_res.to_csv(csv_path, index=False)
print(f"\n  Results -> {csv_path}")

# ── Find best ──
best = min(results, key=lambda r: r["mae_all"])
print(f"\n{'=' * 60}")
print(f"  BEST Split Point: cnn_out={best['cnn_out']} (VMD_out={best['vmd_out']})")
print(f"    Overall:  MAE={best['mae_all']:.1f} kW  RMSE={best['rmse_all']:.1f} kW  R2={best['r2_all']:.4f}")
print(f"    Path A (h=1~{best['cnn_out']}):   MAE={best['mae_a']:.1f} kW  R2={best['r2_a']:.4f}")
print(f"    Path B (h={best['cnn_out']+1}~24): MAE={best['mae_b']:.1f} kW  R2={best['r2_b']:.4f}")
print(f"    Params: {best['n_params']:,}  |  Best epoch: {best['best_epoch']}")
print(f"{'=' * 60}")


# ══════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════
cnn_outs = [r["cnn_out"] for r in results]
mae_all = [r["mae_all"] for r in results]
r2_all  = [r["r2_all"] for r in results]
mae_a   = [r["mae_a"] for r in results]
mae_b   = [r["mae_b"] for r in results]

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("A/B Split Point — Performance Comparison", fontsize=14, fontweight="bold")

ax = axes[0, 0]
ax.plot(cnn_outs, mae_all, "o-", color="#E74C3C", markersize=8, linewidth=2, label="Overall MAE")
best_idx = cnn_outs.index(best["cnn_out"])
ax.scatter([best["cnn_out"]], [best["mae_all"]], s=200, c="red", zorder=5,
           edgecolors="white", linewidth=2)
ax.annotate(f"BEST\ncnn_out={best['cnn_out']}\nMAE={best['mae_all']:.0f}kW",
            (best["cnn_out"], best["mae_all"]),
            textcoords="offset points", xytext=(15, -25),
            fontsize=9, color="red", fontweight="bold")
ax.set_xlabel("cnn_out (Path A hours)"); ax.set_ylabel("Overall MAE (kW)")
ax.set_title("Overall MAE vs Split Point"); ax.grid(True, alpha=0.25)
ax.set_xticks(cnn_outs)

ax = axes[0, 1]
ax.plot(cnn_outs, r2_all, "s-", color="#2C3E50", markersize=8, linewidth=2)
ax.scatter([best["cnn_out"]], [best["r2_all"]], s=200, c="green", zorder=5,
           edgecolors="white", linewidth=2)
ax.annotate(f"R2={best['r2_all']:.4f}",
            (best["cnn_out"], best["r2_all"]),
            textcoords="offset points", xytext=(15, 10),
            fontsize=9, color="green", fontweight="bold")
ax.set_xlabel("cnn_out (Path A hours)"); ax.set_ylabel("Overall R2")
ax.set_title("Overall R2 vs Split Point"); ax.grid(True, alpha=0.25)
ax.set_xticks(cnn_outs)

ax = axes[1, 0]
x = np.arange(len(cnn_outs))
w = 0.35
ax.bar(x - w/2, mae_a, w, label="Path A (CNN-LSTM)", color="#3498DB", edgecolor="white")
ax.bar(x + w/2, mae_b, w, label=f"Path B (VMD-LSTM)", color="#E67E22", edgecolor="white")
ax.set_xlabel("cnn_out (Path A hours)"); ax.set_ylabel("MAE (kW)")
ax.set_title("Per-Path MAE by Split Point"); ax.legend(fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(cnn_outs); ax.grid(True, alpha=0.15, axis="y")

ax = axes[1, 1]
top3 = sorted(results, key=lambda r: r["mae_all"])[:3]
colors = ["#E74C3C", "#2ECC71", "#3498DB"]
styles = ["-", "--", "-."]
for i, r in enumerate(top3):
    h = np.arange(1, OUTPUT_DIM + 1)
    ax.plot(h, r["r2_h"], styles[i], color=colors[i], linewidth=1.5,
            label=f"cnn_out={r['cnn_out']} (MAE={r['mae_all']:.0f})")
    ax.axvline(x=r["cnn_out"] + 0.5, color=colors[i], linestyle=":", alpha=0.4)
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_xlabel("Horizon (hours ahead)"); ax.set_ylabel("R2")
ax.set_title("Per-Horizon R2 — Top 3 Split Points"); ax.legend(fontsize=8)
ax.set_xticks(range(1, 25, 3)); ax.grid(True, alpha=0.15)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "split_comparison.png"), dpi=150)
plt.close(fig)
print(f"  Plot -> {os.path.join(OUT_DIR, 'split_comparison.png')}")

print("\nDone.")
