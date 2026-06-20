"""VMD-LSTM Hybrid — Standalone Evaluation Dashboard.

Reads vmd_hybrid_predictions.csv and produces per-horizon charts + metrics summary.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_DIM = 24
CAPACITY = 2000.0


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


def per_horizon_metrics(y_true, y_pred):
    mae_h = np.zeros(OUTPUT_DIM)
    rmse_h = np.zeros(OUTPUT_DIM)
    r2_h = np.zeros(OUTPUT_DIM)
    for h in range(OUTPUT_DIM):
        mae_h[h] = mean_absolute_error(y_true[:, h], y_pred[:, h])
        rmse_h[h] = np.sqrt(mean_squared_error(y_true[:, h], y_pred[:, h]))
        r2_h[h] = r2_score(y_true[:, h], y_pred[:, h])
    return mae_h, rmse_h, r2_h


def load_eval_data():
    df = pd.read_csv("data/wind_nc/output/wind_data.csv")
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df = df[df["point_id"] == 1].sort_values("valid_time").reset_index(drop=True)

    df_shifted = df[["valid_time"]].copy()
    for h in range(1, OUTPUT_DIM + 1):
        df_shifted[f"power_t{h}"] = compute_power(
            df["wind_speed_100m"].shift(-h).values,
            df["air_density"].shift(-h).values,
        )
        df_shifted[f"ws_t{h}"] = df["wind_speed_100m"].shift(-h)
    df_shifted = df_shifted.dropna().reset_index(drop=True)

    pred_path = os.path.join("outputs", "vmd_hybrid_predictions.csv")
    if not os.path.exists(pred_path):
        print(f"  [WARN] {pred_path} not found. Run forecast_vmd_hybrid.py first.")
        return None, None, None

    pred = pd.read_csv(pred_path)
    pred["valid_time"] = pd.to_datetime(pred["valid_time"])
    merged = pred.merge(df_shifted, on="valid_time", how="inner").dropna().reset_index(drop=True)

    y_pred = np.zeros((len(merged), OUTPUT_DIM))
    y_true = np.zeros((len(merged), OUTPUT_DIM))
    ws_true = np.zeros((len(merged), OUTPUT_DIM))
    for h in range(OUTPUT_DIM):
        y_pred[:, h] = merged[f"power_h{h+1}"].values
        y_true[:, h] = merged[f"power_t{h+1}"].values
        ws_true[:, h] = merged[f"ws_t{h+1}"].values

    return merged, y_true, y_pred, ws_true


def main():
    print("=" * 60)
    print("  VMD-LSTM Hybrid — Evaluation Dashboard")
    print("=" * 60)

    df, y_true, y_pred, ws_true = load_eval_data()
    if df is None:
        return

    n = len(df)
    print(f"  Samples: {n} x {OUTPUT_DIM}h")

    mae_all = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    rmse_all = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
    r2_all = r2_score(y_true.flatten(), y_pred.flatten())
    mae_h, rmse_h, r2_h = per_horizon_metrics(y_true, y_pred)

    print(f"\n  VMD-Hybrid: MAE={mae_all:.2f} kW  RMSE={rmse_all:.2f} kW  R2={r2_all:.4f}")
    print(f"  NMAE={mae_all / CAPACITY * 100:.2f}%  NRMSE={rmse_all / CAPACITY * 100:.2f}%")
    print(f"  Per-horizon R2 range: {r2_h.min():.4f} ~ {r2_h.max():.4f}")

    horizons = np.arange(1, OUTPUT_DIM + 1)

    # ── Figure 1: Per-horizon error + R2 + scatter + residuals ──
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("VMD-LSTM Hybrid 24h Wind Power Forecast — Evaluation",
                 fontsize=15, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(horizons, mae_h, "o-", label="MAE (kW)", color="#E74C3C", markersize=5)
    ax.plot(horizons, rmse_h, "s-", label="RMSE (kW)", color="#2C3E50", markersize=5)
    ax.set_xlabel("Horizon (hours ahead)"); ax.set_ylabel("Error (kW)")
    ax.set_title(f"Error by Horizon | Mean MAE={mae_all:.0f}kW  RMSE={rmse_all:.0f}kW")
    ax.legend(); ax.grid(True, alpha=0.25); ax.set_xticks(range(1, 25, 3))

    ax = axes[0, 1]
    colors_r2 = ["#2ECC71" if v > 0 else "#E74C3C" for v in r2_h]
    ax.bar(horizons, r2_h, color=colors_r2, edgecolor="white")
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=r2_all, color="#3498DB", linewidth=1.5, linestyle="--",
               label=f"Overall R2={r2_all:.3f}")
    ax.set_xlabel("Horizon (hours ahead)"); ax.set_ylabel("R2")
    ax.set_title("R2 by Horizon"); ax.legend(); ax.set_xticks(range(1, 25, 3))

    ax = axes[1, 0]
    ax.scatter(y_true.flatten(), y_pred.flatten(), alpha=0.04, s=3, color="#7B1FA2")
    pmax = max(y_true.max(), y_pred.max()) * 1.05
    ax.plot([0, pmax], [0, pmax], "r--", linewidth=1)
    ax.set_xlim(0, pmax); ax.set_ylim(0, pmax)
    ax.set_xlabel("Actual (kW)"); ax.set_ylabel("Predicted (kW)")
    ax.set_title(f"Scatter | MAE={mae_all:.0f}kW  RMSE={rmse_all:.0f}kW  R2={r2_all:.3f}")

    ax = axes[1, 1]
    res = (y_true - y_pred).flatten()
    ax.hist(res, bins=80, color="#95A5A6", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#E74C3C", linewidth=2, linestyle="--")
    ax.axvline(res.mean(), color="black", linewidth=1.5, linestyle="-",
               label=f"bias={res.mean():.1f}kW")
    ax.set_xlabel("Residual (kW)")
    ax.set_title(f"Error Dist | std={res.std():.0f}kW  NMAE={mae_all/CAPACITY*100:.1f}%")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "vmd_evaluation_dashboard.png"), dpi=180, bbox_inches="tight")
    plt.close()
    print("  Saved -> outputs/vmd_evaluation_dashboard.png")

    # ── Figure 2: Sample profiles + power curve + metrics table ──
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("VMD-LSTM Hybrid — Sample Profiles & Power Curve", fontsize=14, fontweight="bold")

    ax = axes[0]
    for s in [0, n // 4, n // 2, n * 3 // 4]:
        ax.plot(horizons, y_true[s], "o-", markersize=4, alpha=0.7, label=f"True #{s}")
        ax.plot(horizons, y_pred[s], "s--", markersize=4, alpha=0.7, label=f"Pred #{s}")
    ax.set_xlabel("Hours ahead"); ax.set_ylabel("Power (kW)")
    ax.set_title("Sample 24h Profiles"); ax.legend(fontsize=6, ncol=2)

    ax = axes[1]
    ws_range = np.linspace(0, 28, 200)
    p_std = power_curve_v90(wind_at_hub(ws_range))
    ax.plot(ws_range, p_std, "b-", linewidth=2.5, label="Ideal power curve")
    ax.fill_between(ws_range, p_std * 0.8, p_std * 1.2, alpha=0.08, color="blue")
    mask = min(3000, n * OUTPUT_DIM)
    ax.scatter(ws_true.flatten()[::3][:mask], y_true.flatten()[::3][:mask],
               s=2, alpha=0.15, color="red")
    ax.set_xlabel("Wind Speed 100m (m/s)"); ax.set_ylabel("Power (kW)")
    ax.set_title("Power Curve"); ax.legend(fontsize=7)

    ax = axes[2]
    ax.axis("off")
    tbl_data = [
        ["Metric", "Value"],
        ["MAE (overall)",  f"{mae_all:.1f} kW"],
        ["RMSE (overall)", f"{rmse_all:.1f} kW"],
        ["R2  (overall)",  f"{r2_all:.4f}"],
        ["NMAE",           f"{mae_all/CAPACITY*100:.2f}%"],
        ["NRMSE",          f"{rmse_all/CAPACITY*100:.2f}%"],
        ["MAE (h=1)",  f"{mae_h[0]:.1f} kW"],
        ["MAE (h=6)",  f"{mae_h[5]:.1f} kW"],
        ["MAE (h=12)", f"{mae_h[11]:.1f} kW"],
        ["MAE (h=24)", f"{mae_h[23]:.1f} kW"],
        ["R2 (h=1)",  f"{r2_h[0]:.4f}"],
        ["R2 (h=6)",  f"{r2_h[5]:.4f}"],
        ["R2 (h=12)", f"{r2_h[11]:.4f}"],
        ["R2 (h=24)", f"{r2_h[23]:.4f}"],
    ]
    tbl = ax.table(cellText=tbl_data, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.1, 1.4)
    for i in [0, 6, 10]:
        tbl[i, 0].set_facecolor("#ECF0F1")
        tbl[i, 0].set_text_props(fontweight="bold")
    for j in range(2):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Metrics Summary", fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "vmd_evaluation_profiles.png"), dpi=180, bbox_inches="tight")
    plt.close()
    print("  Saved -> outputs/vmd_evaluation_profiles.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
