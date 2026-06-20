"""Bayesian optimization for dropout + warmup hyperparameters.

Uses Gaussian Process (sklearn) + Expected Improvement to find optimal:
  - path_a_dropout, path_b_dropout, dropout
  - warmup_epochs, warmup_start_factor
  - lr, weight_decay

No data leakage: VMD per-domain, IMF scaler fit on train only.
Metric: validation RMSE (kW, unscaled).
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
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
CNN_OUT = 8
CAPACITY = 2000.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
WEATHER_DIM = 8
N_IMFS = 4
TRIAL_EPOCHS = 80
PATIENCE = 20

# Fixed architecture params (from previous optimization)
VMD_ALPHA = 500
pa = clean_path_a_param_defaults()
pb = clean_path_b_param_defaults()
TREND_HIDDEN = pb["trend_hidden"]
FLUCT_HIDDEN = pb["fluct_hidden"]
N_LAYERS = pb["n_layers"]
FC_HIDDEN = pb["fc_hidden"]
CONV1_FILTERS = pa["conv1_filters"]
CONV2_FILTERS = pa["conv2_filters"]
CONV_KERNEL = pa["conv_kernel"]
POOL_SIZE = pa["pool_size"]
CNN_LSTM_HIDDEN = pa["cnn_lstm_hidden"]
CNN_LSTM_LAYERS = pa["cnn_lstm_layers"]

# Bayesian optimization settings
N_INIT = 10          # random initial points
N_TRIALS = 50        # total GP-guided evaluations
N_EI_CANDIDATES = 2000  # candidates for EI maximization

OUT_DIR = os.path.join(os.path.dirname(__file__))
os.makedirs(VMD_CACHE_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

# ══════════════════════════════════════════════
# Parameter search space
# ══════════════════════════════════════════════
SPACE = {
    "path_a_dropout":  ("float", 0.1, 0.5),
    "path_b_dropout":  ("float", 0.1, 0.5),
    "dropout":         ("float", 0.1, 0.5),
    "warmup_epochs":   ("int",   0, 30),
    "warmup_start_factor": ("float", 0.05, 0.5),
    "lr":              ("log",   1e-5, 1e-3),
    "weight_decay":    ("log",   1e-6, 1e-2),
}

def sample_params(rng=None):
    if rng is None:
        rng = np.random
    x = np.empty(len(SPACE))
    for j in range(len(SPACE)):
        x[j] = rng.uniform(0, 1)
    return x

def decode_params(x_norm):
    params = {}
    for j, (name, spec) in enumerate(SPACE.items()):
        v_norm = float(np.clip(x_norm[j], 0, 1))
        spec_type, lo, hi = spec
        if spec_type == "float":
            val = lo + v_norm * (hi - lo)
        elif spec_type == "int":
            val = int(np.round(lo + v_norm * (hi - lo)))
        elif spec_type == "log":
            lo_l, hi_l = np.log10(lo), np.log10(hi)
            val = 10.0 ** (lo_l + v_norm * (hi_l - lo_l))
        else:
            raise ValueError(f"Unknown spec: {spec_type}")
        params[name] = val
    return params


# ══════════════════════════════════════════════
# Data loading (once)
# ══════════════════════════════════════════════
print("=" * 60)
print("  Bayesian Optimization for Dropout & Warmup")
print(f"  Device: {DEVICE}")
print("=" * 60)

df = load_wind_data()
times = df["valid_time"].values

target_cols = [f"power_t{h}" for h in range(1, OUTPUT_DIM + 1)]

# ── Labels ──
y_raw = df[target_cols].values.astype(np.float32)
train_years_mask = pd.DatetimeIndex(times).year.isin([2024, 2025])
scaler_y = StandardScaler()
scaler_y.fit(y_raw[train_years_mask])
y_scaled = scaler_y.transform(y_raw)

# ── Weather ──
weather_raw = df[WEATHER_COLS].values.astype(np.float32)
weather_scaler = StandardScaler()
weather_scaler.fit(weather_raw[train_years_mask])
weather_scaled = weather_scaler.transform(weather_raw)

# ── VMD per-domain (shared cache) ──
dom_tr, dom_va, dom_te = get_sample_domain_masks(times)

print(f"\n[VMD] alpha={VMD_ALPHA} per-domain (shared cache={VMD_CACHE_DIR}) ...")
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
# Dataset + training
# ══════════════════════════════════════════════
class SeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.y[idx])


def train_eval(params):
    """Train a model with given hyperparameters; return val RMSE (kW)."""
    pa_drop = float(params["path_a_dropout"])
    pb_drop = float(params["path_b_dropout"])
    drp     = float(params["dropout"])
    wu_ep   = int(params["warmup_epochs"])
    wu_fac  = float(params["warmup_start_factor"])
    lr      = float(params["lr"])
    wd      = float(params["weight_decay"])

    model = VMDLSTMHybrid(
        weather_dim=WEATHER_DIM, n_imfs=N_IMFS,
        cnn_out=CNN_OUT, output_dim=OUTPUT_DIM,
        trend_hidden=TREND_HIDDEN, fluct_hidden=FLUCT_HIDDEN,
        n_layers=N_LAYERS, dropout=drp,
        fc_hidden=FC_HIDDEN, path_b_dropout=pb_drop,
        capacity=CAPACITY,
        conv1_filters=CONV1_FILTERS, conv2_filters=CONV2_FILTERS,
        conv_kernel=CONV_KERNEL, pool_size=POOL_SIZE,
        cnn_lstm_hidden=CNN_LSTM_HIDDEN, cnn_lstm_layers=CNN_LSTM_LAYERS,
        path_a_dropout=pa_drop,
    ).to(DEVICE)

    train_loader = DataLoader(SeqDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(SeqDataset(X_val, y_val), BATCH_SIZE)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_ctr = 0
    best_epoch = 0

    for epoch in range(1, TRIAL_EPOCHS + 1):
        # Linear LR warmup
        if wu_ep > 0 and epoch <= wu_ep:
            wu_frac = wu_fac + (1.0 - wu_fac) * (epoch - 1) / max(wu_ep - 1, 1)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * wu_frac

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
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                break

    # Evaluate on validation set (unscaled RMSE, kW)
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

    del model
    torch.cuda.empty_cache()
    return rmse_val, best_epoch


# ══════════════════════════════════════════════
# Bayesian Optimization — GP + EI
# ══════════════════════════════════════════════
def expected_improvement(x, gp, y_best, xi=0.01):
    x = np.atleast_2d(x)
    mu, sigma = gp.predict(x, return_std=True)
    sigma = np.maximum(sigma, 1e-9)
    with np.errstate(divide="ignore"):
        z = (y_best - mu - xi) / sigma
        ei = (y_best - mu - xi) * sp.stats.norm.cdf(z) + sigma * sp.stats.norm.pdf(z)
        ei[sigma < 1e-9] = 0.0
    return ei

def sample_next_point(gp, y_best, n_candidates=N_EI_CANDIDATES):
    X_cand = np.array([sample_params(np.random) for _ in range(n_candidates)])
    ei_vals = expected_improvement(X_cand, gp, y_best)
    best_idx = np.argmax(ei_vals)
    return X_cand[best_idx], float(ei_vals[best_idx])

import scipy as sp

print("=" * 60)
print(f"  Bayesian Optimization — {N_INIT} init + {N_TRIALS} trials")
print(f"  {len(SPACE)} parameters: {list(SPACE.keys())}")
print("=" * 60)

X_evaluated = []
y_evaluated = []
best_epochs  = []
results_rows = []

# ── Phase 1: Random initial points ──
print(f"\n--- Phase 1: Initial random sampling ({N_INIT} points) ---")
for i in range(N_INIT):
    x_norm = sample_params(np.random)
    params = decode_params(x_norm)
    desc = (f"pa_drop={params['path_a_dropout']:.3f} pb_drop={params['path_b_dropout']:.3f} "
            f"drop={params['dropout']:.3f} wu_ep={params['warmup_epochs']} "
            f"wu_fac={params['warmup_start_factor']:.3f} lr={params['lr']:.2e} wd={params['weight_decay']:.2e}")
    t0 = time.time()
    rmse, epoch = train_eval(params)
    elapsed = time.time() - t0

    X_evaluated.append(x_norm)
    y_evaluated.append(rmse)
    best_epochs.append(epoch)

    row = {k: v for k, v in params.items()}
    row["val_rmse"] = round(rmse, 2)
    row["best_epoch"] = epoch
    row["time_s"] = round(elapsed, 1)
    results_rows.append(row)

    print(f"  Init {i+1:2d}/{N_INIT} | RMSE={rmse:.1f} kW  ep={epoch}  {elapsed:.0f}s")

# ── Phase 2: GP-guided optimisation ──
print(f"\n--- Phase 2: GP Bayesian optimisation ({N_TRIALS} trials) ---")
for i in range(N_TRIALS):
    X_arr = np.array(X_evaluated)
    y_arr = np.array(y_evaluated)

    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(len(SPACE)),
                                         length_scale_bounds=(1e-3, 1e3)) + \
             WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-5, 1e0))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5,
                                  normalize_y=True, random_state=SEED + i)
    gp.fit(X_arr, y_arr)

    y_best = np.min(y_arr)
    x_next, ei_val = sample_next_point(gp, y_best)
    params = decode_params(x_next)

    desc = (f"pa_drop={params['path_a_dropout']:.3f} pb_drop={params['path_b_dropout']:.3f} "
            f"drop={params['dropout']:.3f} wu_ep={params['warmup_epochs']} "
            f"wu_fac={params['warmup_start_factor']:.3f} lr={params['lr']:.2e} wd={params['weight_decay']:.2e}")
    t0 = time.time()
    rmse, epoch = train_eval(params)
    elapsed = time.time() - t0

    X_evaluated.append(x_next)
    y_evaluated.append(rmse)
    best_epochs.append(epoch)

    row = {k: v for k, v in params.items()}
    row["val_rmse"] = round(rmse, 2)
    row["best_epoch"] = epoch
    row["time_s"] = round(elapsed, 1)
    results_rows.append(row)

    best_so_far = min(y_evaluated)
    print(f"  Trial {i+1:2d}/{N_TRIALS} | RMSE={rmse:.1f}  best={best_so_far:.1f}  "
          f"EI={ei_val:.3f}  ep={epoch}  {elapsed:.0f}s  | {desc}")

    if (i + 1) % 10 == 0:
        pd.DataFrame(results_rows).to_csv(
            os.path.join(OUT_DIR, "results_partial.csv"), index=False)


# ══════════════════════════════════════════════
# Save results
# ══════════════════════════════════════════════
df_res = pd.DataFrame(results_rows).sort_values("val_rmse").reset_index(drop=True)
csv_path = os.path.join(OUT_DIR, "results.csv")
df_res.to_csv(csv_path, index=False)
print(f"\n  Results saved -> {csv_path}")

best_row = df_res.iloc[0]
print(f"\n{'=' * 60}")
print(f"  BEST Configuration (val RMSE={best_row['val_rmse']:.1f} kW):")
print(f"    path_a_dropout      = {best_row['path_a_dropout']:.4f}")
print(f"    path_b_dropout      = {best_row['path_b_dropout']:.4f}")
print(f"    dropout             = {best_row['dropout']:.4f}")
print(f"    warmup_epochs       = {int(best_row['warmup_epochs'])}")
print(f"    warmup_start_factor = {best_row['warmup_start_factor']:.4f}")
print(f"    lr                  = {best_row['lr']:.6e}")
print(f"    weight_decay        = {best_row['weight_decay']:.6e}")
print(f"    best_epoch          = {int(best_row['best_epoch'])}")
print(f"{'=' * 60}")


# ══════════════════════════════════════════════
# Plots
# ══════════════════════════════════════════════
cumulative_best = np.minimum.accumulate(y_evaluated)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Bayesian Optimization — Dropout & Warmup Tuning", fontsize=14, fontweight="bold")

# (0,0) Convergence
ax = axes[0, 0]
trial_nums = np.arange(1, len(y_evaluated) + 1)
ax.plot(trial_nums, y_evaluated, "o", color="#3498DB", markersize=4, alpha=0.5, label="Observed RMSE")
ax.plot(trial_nums, cumulative_best, "-", color="#E74C3C", linewidth=2, label="Best so far")
ax.axvline(x=N_INIT + 0.5, color="gray", linestyle="--", alpha=0.5, label="GP starts")
ax.set_xlabel("Trial"); ax.set_ylabel("Val RMSE (kW)")
ax.set_title(f"Convergence (best={cumulative_best[-1]:.0f} kW)"); ax.legend(fontsize=8); ax.grid(True, alpha=0.2)

# (0,1) Per-parameter RMSE scatter
param_scatter = ["path_a_dropout", "path_b_dropout", "warmup_epochs", "lr"]
titles_scatter = ["Path A Dropout", "Path B Dropout", "Warmup Epochs", "Learning Rate"]
ax = axes[0, 1]
for pk, title in zip(param_scatter, titles_scatter):
    vals = df_res[pk].values
    corr = np.corrcoef(vals, df_res["val_rmse"].values)[0, 1]
    print(f"  Corr({pk}, RMSE) = {corr:.3f}")

# (1,0) Dropout vs RMSE
ax = axes[1, 0]
x = df_res["path_a_dropout"].values
y = df_res["path_b_dropout"].values
sc = ax.scatter(x, y, c=df_res["val_rmse"].values, cmap="RdYlGn_r", s=50, edgecolors="white", linewidth=0.5)
ax.set_xlabel("Path A Dropout"); ax.set_ylabel("Path B Dropout")
ax.set_title("Dropout Space → Val RMSE")
cbar = plt.colorbar(sc, ax=ax); cbar.set_label("Val RMSE (kW)")
ax.grid(True, alpha=0.2)

# (1,1) LR vs weight_decay
ax = axes[1, 1]
sc2 = ax.scatter(np.log10(df_res["lr"].values), np.log10(df_res["weight_decay"].values),
                 c=df_res["val_rmse"].values, cmap="RdYlGn_r", s=50, edgecolors="white", linewidth=0.5)
ax.set_xlabel("log10(lr)"); ax.set_ylabel("log10(weight_decay)")
ax.set_title("LR vs Weight Decay → Val RMSE")
cbar2 = plt.colorbar(sc2, ax=ax); cbar2.set_label("Val RMSE (kW)")
ax.grid(True, alpha=0.2)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bayesian_dropout_warmup.png"), dpi=150)
plt.close(fig)
print(f"  Plot saved -> {os.path.join(OUT_DIR, 'bayesian_dropout_warmup.png')}")

print("\nDone.")
