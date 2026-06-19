"""Random search for Path B (VMD-LSTM) hyperparameters."""
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
import os, sys, time, warnings
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
CNN_OUT = 8
CAPACITY = 2000.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
N_TRIALS = 60
MAX_EPOCHS = 50
LR = 5e-4
WEIGHT_DECAY = 5e-4
WEATHER_DIM = 8
EARLY_STOP_PATIENCE = 10

# Fixed Path A params (best from CNN tuning)
CONV1_FILTERS = 128
CONV2_FILTERS = 192
CONV_KERNEL = 3
POOL_SIZE = 2
CNN_LSTM_HIDDEN = 32
CNN_LSTM_LAYERS = 1
PATH_A_DROPOUT = 0.31

BASE_DIR = r"D:\net.zero\6.14_wind_forecast_glide"
OUT_DIR = os.path.join(BASE_DIR, "Hyperparameter_Tuning", "VMD_Hyperparameter_Tuning")
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
print("  VMD-LSTM Hyperparameter Tuning — Data Preparation")
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
train_years_mask = pd.DatetimeIndex(times).year.isin([2024, 2025])

# ── Weather & labels (do once) ──
y_raw = df[target_cols].values.astype(np.float32)
scaler_y = StandardScaler()
scaler_y.fit(y_raw[train_years_mask])
y_scaled = scaler_y.transform(y_raw)

weather_raw = df[weather_cols].values.astype(np.float32)
weather_scaler = StandardScaler()
weather_scaler.fit(weather_raw[train_years_mask])
weather_scaled = weather_scaler.transform(weather_raw)

# Sequence indices (shared by all trials)
idx_start = SEQ_LEN - 1
seq_years = pd.DatetimeIndex(times[idx_start:]).year
seq_months = pd.DatetimeIndex(times[idx_start:]).month

train_mask = seq_years.isin([2024, 2025]) & ~((seq_years == 2025) & (seq_months >= 10))
val_mask   = (seq_years == 2025) & (seq_months >= 10)
test_mask  = seq_years == 2026

# ══════════════════════════════════════════════
# 2. Sequence builder (called per trial with different IMFs)
# ══════════════════════════════════════════════
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])

def build_sequences(imfs_scaled):
    features = np.concatenate([weather_scaled, imfs_scaled], axis=1).astype(np.float32)
    X_list, y_list = [], []
    for i in range(SEQ_LEN - 1, len(features)):
        X_list.append(features[i - SEQ_LEN + 1: i + 1])
        y_list.append(y_scaled[i])
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y

def compute_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))

def train_eval_model(model, X_tr, y_tr, X_va, y_va, X_te, y_te):
    train_loader = DataLoader(SeqDataset(X_tr, y_tr), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(SeqDataset(X_va, y_va), BATCH_SIZE)
    test_loader = DataLoader(SeqDataset(X_te, y_te), BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best_val_rmse_1724 = float("inf")
    best_epoch = 0
    patience_counter = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(x), y).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

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

        rmse_1724 = compute_rmse(y_true_v[:, CNN_OUT:], y_pred_v[:, CNN_OUT:])

        if rmse_1724 < best_val_rmse_1724:
            best_val_rmse_1724 = rmse_1724
            patience_counter = 0
            best_epoch = epoch
            best_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOP_PATIENCE:
                break

    # Test evaluation
    model.load_state_dict(best_state)
    model.eval()
    all_pred_t, all_true_t = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(DEVICE)
            all_pred_t.append(model(x).cpu().numpy())
            all_true_t.append(y.numpy())
    y_pred_t = scaler_y.inverse_transform(np.concatenate(all_pred_t, axis=0))
    y_true_t = scaler_y.inverse_transform(np.concatenate(all_true_t, axis=0))
    y_pred_t = np.clip(y_pred_t, 0, CAPACITY)

    return {
        "val_rmse_1724": round(float(best_val_rmse_1724), 2),
        "test_rmse_1724": round(float(compute_rmse(y_true_t[:, CNN_OUT:], y_pred_t[:, CNN_OUT:])), 2),
        "test_mae_1724": round(float(np.abs(y_true_t[:, CNN_OUT:] - y_pred_t[:, CNN_OUT:]).mean()), 2),
        "test_mae_all": round(float(np.abs(y_true_t - y_pred_t).mean()), 2),
        "test_r2_all": round(float(1 - np.sum((y_true_t - y_pred_t)**2) / np.sum((y_true_t - y_true_t.mean())**2)), 4),
        "best_epoch": best_epoch,
    }

# ══════════════════════════════════════════════
# 3. Pre-compute VMD for each unique alpha (per-domain, no leakage)
# ══════════════════════════════════════════════
ALPHA_CANDIDATES = [500, 1000, 2000, 4000, 8000]
vmd_cache = {}

power_raw = df[target_cols].values[:, 0].astype(float)
sample_years = pd.DatetimeIndex(times).year
sample_months = pd.DatetimeIndex(times).month
dom_tr = sample_years.isin([2024, 2025]) & ~((sample_years == 2025) & (sample_months >= 10))
dom_va = (sample_years == 2025) & (sample_months >= 10)
dom_te = sample_years == 2026

print(f"\n[VMD] Pre-computing {len(ALPHA_CANDIDATES)} alpha values (per-domain) ...")
for a in ALPHA_CANDIDATES:
    t0 = time.time()
    imfs, omegas = decompose_by_domain(
        power_raw,
        [("train", dom_tr), ("val", dom_va), ("test", dom_te)],
        K=4, alpha=a, tol=1e-7, max_iter=500, seed=SEED,
        cache_dir=VMD_CACHE_DIR,
    )
    imfs_scaler = StandardScaler()
    imfs_scaler.fit(imfs[dom_tr])
    full_imfs_scaled = imfs_scaler.transform(imfs)
    vmd_cache[a] = {
        "imfs_scaled": full_imfs_scaled.astype(np.float32),
        "omegas": omegas,
    }
    print(f"  alpha={a:5d}  |  {time.time()-t0:.1f}s  |  omegas={ {k: np.round(v,3).tolist() for k,v in omegas.items()} }")

# ══════════════════════════════════════════════
# 4. Random search
# ══════════════════════════════════════════════
PARAM_KEYS = ["alpha", "trend_hidden", "fluct_hidden", "n_layers",
              "path_b_dropout", "fc_hidden"]
trend_candidates = [64, 100, 128, 192, 256]
fluct_candidates = [64, 128, 192, 256, 320]
fc_candidates = [32, 50, 64, 100]

print(f"\n{'=' * 60}")
print(f"  Random Search — {N_TRIALS} trials on 6 params")
print(f"  Device: {DEVICE}")
print(f"{'=' * 60}")

results = []
partial_path = os.path.join(OUT_DIR, "results_partial.csv")
start_trial = 0
if os.path.exists(partial_path):
    existing = pd.read_csv(partial_path)
    results = existing.to_dict("records")
    start_trial = len(results)
    print(f"  Resuming from trial {start_trial + 1}/{N_TRIALS}")

for trial in range(start_trial, N_TRIALS):
    # Sample params
    a = int(np.random.choice(ALPHA_CANDIDATES))
    th = int(np.random.choice(trend_candidates))
    fh = int(np.random.choice(fluct_candidates))
    nl = int(np.random.choice([1, 2, 3]))
    pb_drop = round(np.random.uniform(0.1, 0.5), 3)
    fc = int(np.random.choice(fc_candidates))

    params = {
        "trial": trial + 1,
        "alpha": a,
        "trend_hidden": th,
        "fluct_hidden": fh,
        "n_layers": nl,
        "path_b_dropout": pb_drop,
        "fc_hidden": fc,
        "n_params": 0,
        "train_time_s": 0,
        "val_rmse_1724": 0,
        "test_rmse_1724": 0,
        "test_mae_1724": 0,
        "test_mae_all": 0,
        "test_r2_all": 0,
        "best_epoch": 0,
    }

    desc = f"alpha={a} th={th} fh={fh} nl={nl} d={pb_drop} fc={fc}"
    print(f"\n  Trial {trial+1}/{N_TRIALS} | {desc}")

    # Build sequences for this alpha
    imfs_scaled = vmd_cache[a]["imfs_scaled"]
    X_all, y_seq = build_sequences(imfs_scaled)
    X_tr, y_tr = X_all[train_mask], y_seq[train_mask]
    X_va, y_va = X_all[val_mask], y_seq[val_mask]
    X_te, y_te = X_all[test_mask], y_seq[test_mask]

    # Build model (Path A fixed, Path B = sampled)
    model = VMDLSTMHybrid(
        weather_dim=WEATHER_DIM, n_imfs=4, cnn_out=CNN_OUT,
        output_dim=OUTPUT_DIM,
        trend_hidden=int(th), fluct_hidden=int(fh),
        n_layers=int(nl), dropout=0.3,
        fc_hidden=int(fc), path_b_dropout=float(pb_drop),
        capacity=CAPACITY,
        conv1_filters=CONV1_FILTERS, conv2_filters=CONV2_FILTERS,
        conv_kernel=CONV_KERNEL, pool_size=POOL_SIZE,
        cnn_lstm_hidden=CNN_LSTM_HIDDEN, cnn_lstm_layers=CNN_LSTM_LAYERS,
        path_a_dropout=PATH_A_DROPOUT,
    ).to(DEVICE)

    n_params = sum(p.numel() for p in model.parameters())
    params["n_params"] = n_params
    print(f"         Params: {n_params:,}")

    t0 = time.time()
    metrics = train_eval_model(model, X_tr, y_tr, X_va, y_va, X_te, y_te)
    train_time = time.time() - t0
    params.update(metrics)
    params["train_time_s"] = round(train_time, 1)

    print(f"         Val RMSE[17-24]={params['val_rmse_1724']:.1f}  "
          f"Test MAE={params['test_mae_all']:.1f}  "
          f"R2={params['test_r2_all']:.3f}  ep={params['best_epoch']}  {train_time:.0f}s")

    results.append(params)
    pd.DataFrame(results).to_csv(partial_path, index=False)

# ══════════════════════════════════════════════
# 5. Save & plots
# ══════════════════════════════════════════════
df_res = pd.DataFrame(results).sort_values("val_rmse_1724").reset_index(drop=True)
csv_path = os.path.join(OUT_DIR, "results.csv")
df_res.to_csv(csv_path, index=False)
print(f"\n  Results saved -> {csv_path}")

b = df_res.iloc[0]
print(f"\n{'=' * 60}")
print(f"  BEST Trial #{int(b['trial'])}:")
print(f"    alpha={int(b['alpha'])}, trend_hidden={int(b['trend_hidden'])}, fluct_hidden={int(b['fluct_hidden'])}, n_layers={int(b['n_layers'])}, path_b_dropout={b['path_b_dropout']:.3f}, fc_hidden={int(b['fc_hidden'])}")
print(f"    Val RMSE[17-24]={b['val_rmse_1724']:.1f}  Test MAE={b['test_mae_all']:.1f}  R2={b['test_r2_all']:.3f}")
print(f"    Params: {int(b['n_params']):,}")
print(f"{'=' * 60}")

# ── Plot 1: RMSE vs trial ──
fig, ax = plt.subplots(figsize=(10, 6))
trials = df_res["trial"].values
rmse_v = df_res["val_rmse_1724"].values
colors = ["#E74C3C" if i == 0 else "#3498DB" for i in range(len(df_res))]
ax.scatter(trials, rmse_v, c=colors, s=50, alpha=0.7, edgecolors="white", linewidth=0.5)
ax.axhline(b["val_rmse_1724"], color="#E74C3C", linestyle="--", alpha=0.5,
           label=f"Best = {b['val_rmse_1724']:.1f} kW")
ax.set_xlabel("Trial"); ax.set_ylabel("Val RMSE h=17~24 (kW)")
ax.set_title("Random Search — RMSE by Trial"); ax.legend(); ax.grid(True, alpha=0.25)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "rmse_vs_trial.png"), dpi=150); plt.close(fig)

# ── Plot 2: Parameter importance ──
param_names = PARAM_KEYS
param_labels = ["alpha", "trend_hidden", "fluct_hidden", "n_layers",
                "path_b_dropout", "fc_hidden"]
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()
for i, (pk, pl) in enumerate(zip(param_names, param_labels)):
    ax = axes[i]
    vals = df_res[pk].values
    corr = np.corrcoef(vals, rmse_v)[0, 1]
    ax.scatter(vals, rmse_v, c="#9B59B6", s=30, alpha=0.6, edgecolors="white")
    ax.set_xlabel(pl); ax.set_ylabel("Val RMSE (kW)")
    ax.set_title(f"{pl}  (r={corr:.3f})"); ax.grid(True, alpha=0.2)
    if pk in ("n_layers",):
        ax.set_xticks(sorted(df_res[pk].unique()))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "param_importance.png"), dpi=150); plt.close(fig)

# ── Plot 3: Top 5 comparison + best trial detail ──
# Retrain best model for per-hour metrics
best_model = VMDLSTMHybrid(
    weather_dim=WEATHER_DIM, n_imfs=4, cnn_out=CNN_OUT,
    output_dim=OUTPUT_DIM,
    trend_hidden=int(b["trend_hidden"]), fluct_hidden=int(b["fluct_hidden"]),
    n_layers=int(b["n_layers"]), dropout=0.3,
    fc_hidden=int(b["fc_hidden"]), path_b_dropout=float(b["path_b_dropout"]),
    capacity=CAPACITY,
    conv1_filters=CONV1_FILTERS, conv2_filters=CONV2_FILTERS,
    conv_kernel=CONV_KERNEL, pool_size=POOL_SIZE,
    cnn_lstm_hidden=CNN_LSTM_HIDDEN, cnn_lstm_layers=CNN_LSTM_LAYERS,
    path_a_dropout=PATH_A_DROPOUT,
).to(DEVICE)

imfs_scaled_best = vmd_cache[int(b["alpha"])]["imfs_scaled"]
X_all_best, y_seq_best = build_sequences(imfs_scaled_best)
X_tr_b, y_tr_b = X_all_best[train_mask], y_seq_best[train_mask]
X_va_b, y_va_b = X_all_best[val_mask], y_seq_best[val_mask]
X_te_b, y_te_b = X_all_best[test_mask], y_seq_best[test_mask]

train_loader_b = DataLoader(SeqDataset(X_tr_b, y_tr_b), BATCH_SIZE, shuffle=True)
val_loader_b = DataLoader(SeqDataset(X_va_b, y_va_b), BATCH_SIZE)
optimizer_b = torch.optim.Adam(best_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler_b = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_b, mode="min", factor=0.5, patience=5)
criterion_b = nn.MSELoss()
train_losses, val_losses = [], []
best_vl = float("inf"); best_ep_b = 0; patience_c = 0
for epoch in range(1, MAX_EPOCHS + 1):
    best_model.train()
    for x, y in train_loader_b:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer_b.zero_grad(); criterion_b(best_model(x), y).backward()
        torch.nn.utils.clip_grad_norm_(best_model.parameters(), max_norm=1.0); optimizer_b.step()
    best_model.eval(); tl, vl = 0.0, 0.0; tn, vn = 0, 0
    with torch.no_grad():
        for x, y in train_loader_b:
            x, y = x.to(DEVICE), y.to(DEVICE)
            tl += criterion_b(best_model(x), y).item() * x.size(0); tn += x.size(0)
        for x, y in val_loader_b:
            x, y = x.to(DEVICE), y.to(DEVICE)
            vl += criterion_b(best_model(x), y).item() * x.size(0); vn += x.size(0)
    train_losses.append(tl / tn); val_losses.append(vl / vn)
    scheduler_b.step(val_losses[-1])
    if val_losses[-1] < best_vl:
        best_vl = val_losses[-1]; best_ep_b = epoch; patience_c = 0
    else:
        patience_c += 1
        if patience_c >= EARLY_STOP_PATIENCE: break

# Per-hour test eval
test_loader_b = DataLoader(SeqDataset(X_te_b, y_te_b), BATCH_SIZE)
all_pt, all_tt = [], []
best_model.eval()
with torch.no_grad():
    for x, y in test_loader_b:
        all_pt.append(best_model(x.to(DEVICE)).cpu().numpy())
        all_tt.append(y.numpy())
y_pt = scaler_y.inverse_transform(np.concatenate(all_pt, axis=0))
y_tt = scaler_y.inverse_transform(np.concatenate(all_tt, axis=0))
y_pt = np.clip(y_pt, 0, CAPACITY)

mae_h = np.array([np.abs(y_tt[:, h] - y_pt[:, h]).mean() for h in range(OUTPUT_DIM)])
rmse_h = np.array([np.sqrt(((y_tt[:, h] - y_pt[:, h])**2).mean()) for h in range(OUTPUT_DIM)])
horizons = np.arange(1, OUTPUT_DIM + 1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(horizons, mae_h, "o-", color="#E74C3C", markersize=4, label="MAE")
ax.plot(horizons, rmse_h, "s-", color="#2C3E50", markersize=4, label="RMSE")
ax.axvline(x=CNN_OUT + 0.5, color="gray", linestyle="--", alpha=0.5, label="CNN|VMD split")
ax.set_xlabel("Horizon (hours ahead)"); ax.set_ylabel("Error (kW)")
ax.set_title(f"Best Trial — Per-hour MAE/RMSE"); ax.legend(); ax.grid(True, alpha=0.2)
ax.set_xticks(range(1, 25, 3))

ax = axes[1]; ax.axis("off")
tbl_data = [
    ["Hyperparameter", "Best Value"],
    ["alpha", str(int(b["alpha"]))],
    ["trend_hidden", str(int(b["trend_hidden"]))],
    ["fluct_hidden", str(int(b["fluct_hidden"]))],
    ["n_layers", str(int(b["n_layers"]))],
    ["path_b_dropout", f"{b['path_b_dropout']:.3f}"],
    ["fc_hidden", str(int(b["fc_hidden"]))],
    ["Total Params", f"{int(b['n_params']):,}"],
    ["Val RMSE (h=17~24)", f"{b['val_rmse_1724']:.1f} kW"],
    ["Test MAE (all)", f"{b['test_mae_all']:.1f} kW"],
    ["Test R2 (all)", f"{b['test_r2_all']:.3f}"],
    ["Best epoch", str(int(b["best_epoch"]))],
]
tbl = ax.table(cellText=tbl_data, cellLoc="center", loc="center")
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.2, 1.4)
for j in range(2):
    tbl[0, j].set_facecolor("#2C3E50"); tbl[0, j].set_text_props(color="white", fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "best_trial_detail.png"), dpi=150); plt.close(fig)

# ── Plot 4: Loss curve ──
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, len(train_losses)+1), train_losses, label="Train Loss", color="#3498DB")
ax.plot(range(1, len(val_losses)+1), val_losses, label="Val Loss", color="#E74C3C")
ax.axvline(x=best_ep_b, color="green", linestyle="--", alpha=0.5, label=f"Best epoch={best_ep_b}")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("Best Trial — Loss Curve"); ax.legend(); ax.grid(True, alpha=0.2)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "best_trial_loss.png"), dpi=150); plt.close(fig)

# ── Plot 5: Parallel coordinates ──
try:
    from pandas.plotting import parallel_coordinates
    df_p = df_res.copy()
    for pk in param_names:
        if pk != "n_layers":
            mn = df_p[pk].min(); mx = df_p[pk].max()
            df_p[pk] = (df_p[pk] - mn) / (mx - mn + 1e-8)
    df_p["rmse_n"] = (df_p["val_rmse_1724"] - df_p["val_rmse_1724"].min()) / \
                     (df_p["val_rmse_1724"].max() - df_p["val_rmse_1724"].min() + 1e-8)
    df_p["rank"] = pd.qcut(df_p["val_rmse_1724"], q=3, labels=["Good", "Mid", "Poor"])

    fig, ax = plt.subplots(figsize=(13, 6))
    parallel_coordinates(
        df_p, class_column="rank",
        cols=param_names + ["rmse_n"],
        color=["#2ECC71", "#F39C12", "#E74C3C"],
        alpha=0.4, linewidth=0.8, ax=ax
    )
    ax.set_xticklabels(param_labels + ["RMSE"]); ax.legend(title="Rank")
    ax.set_title("Parallel Coordinates — Path B Parameter Combinations"); ax.grid(True, alpha=0.15)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "parallel_coords.png"), dpi=150); plt.close(fig)
except ImportError:
    pass

print(f"\n  All plots saved to: {OUT_DIR}")
print(f"\nDone.")
