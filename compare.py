import numpy as np, pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

# Power curve utils
def power_curve_v90(v_hub):
    curve = np.array([[0,0],[1,0],[2,0],[3,0],[4,35],[5,80],[6,150],[7,260],[8,410],[9,610],
        [10,870],[11,1180],[12,1540],[13,1850],[14,1970],[15,2000],[16,2000],[17,2000],
        [18,2000],[19,2000],[20,2000],[21,2000],[22,2000],[23,2000],[24,2000],[25,2000],[26,0]], dtype=float)
    return np.interp(v_hub, curve[:,0], curve[:,1])
def wind_at_hub(v100, z_ref=100, z_hub=90, z0=0.03):
    return v100 * (np.log(z_hub/z0)/np.log(z_ref/z0))
def compute_power(v100, rho, rho_ref=1.225):
    return power_curve_v90(wind_at_hub(v100)) * (rho/rho_ref)

# Load data
df = pd.read_csv('data/wind_nc/output/wind_data.csv')
df['valid_time'] = pd.to_datetime(df['valid_time'])
df = df[df['point_id']==1].sort_values('valid_time').reset_index(drop=True)

hard = pd.read_csv('outputs/hard_split_predictions.csv')
hard['valid_time'] = pd.to_datetime(hard['valid_time'])
soft = pd.read_csv('outputs/vmd_hybrid_predictions.csv')
soft['valid_time'] = pd.to_datetime(soft['valid_time'])

merged = hard[['valid_time']].merge(df[['valid_time','wind_speed_100m','air_density']], on='valid_time')

# Rebuild true power labels
true = np.zeros((len(merged), 24))
ws = df['wind_speed_100m'].values
rho = df['air_density'].values
for i, t in enumerate(merged['valid_time'].values):
    idx = df[df['valid_time']==t].index[0]
    for h in range(24):
        fi = idx + h + 1
        if fi < len(df):
            true[i,h] = compute_power(ws[fi], rho[fi])

hard_pred = hard[[f'power_h{h}' for h in range(1,25)]].values
soft_pred = soft[[f'power_h{h}' for h in range(1,25)]].values

# Per-horizon metrics
mae_hard, rmse_hard, r2_hard = np.zeros(24), np.zeros(24), np.zeros(24)
mae_soft, rmse_soft, r2_soft = np.zeros(24), np.zeros(24), np.zeros(24)
for h in range(24):
    mae_hard[h] = mean_absolute_error(true[:,h], hard_pred[:,h])
    rmse_hard[h] = np.sqrt(mean_squared_error(true[:,h], hard_pred[:,h]))
    r2_hard[h] = r2_score(true[:,h], hard_pred[:,h])
    mae_soft[h] = mean_absolute_error(true[:,h], soft_pred[:,h])
    rmse_soft[h] = np.sqrt(mean_squared_error(true[:,h], soft_pred[:,h]))
    r2_soft[h] = r2_score(true[:,h], soft_pred[:,h])

# Overall
mae_h = mean_absolute_error(true.flatten(), hard_pred.flatten())
rmse_h = np.sqrt(mean_squared_error(true.flatten(), hard_pred.flatten()))
r2_h = r2_score(true.flatten(), hard_pred.flatten())
mae_s = mean_absolute_error(true.flatten(), soft_pred.flatten())
rmse_s = np.sqrt(mean_squared_error(true.flatten(), soft_pred.flatten()))
r2_s = r2_score(true.flatten(), soft_pred.flatten())

# Plot
horizons = np.arange(1, 25)
fig, axes = plt.subplots(2, 3, figsize=(22, 13))
fig.suptitle('VMD-LSTM Hybrid: Hard Split vs Soft Fusion', fontsize=15, fontweight='bold')

CH = '#E74C3C'
CS = '#2C3E50'

# (0,0) MAE
ax = axes[0,0]
ax.plot(horizons, mae_hard, 'o-', color=CH, label='Hard Split (CNN 1-8h / VMD 9-24h)', markersize=5)
ax.plot(horizons, mae_soft, 's-', color=CS, label='Soft Fusion (learned gate)', markersize=5)
ax.axvline(x=8.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Horizon (hours)'); ax.set_ylabel('MAE (kW)')
ax.set_title(f'MAE | Hard={mae_h:.0f}kW  Soft={mae_s:.0f}kW')
ax.legend(fontsize=8); ax.grid(True, alpha=0.25); ax.set_xticks(range(1,25,2))

# (0,1) R2
ax = axes[0,1]
x = np.arange(24); w = 0.35
ax.bar(x-w/2, r2_hard, w, color=CH, alpha=0.85, label='Hard Split')
ax.bar(x+w/2, r2_soft, w, color=CS, alpha=0.85, label='Soft Fusion')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axvline(x=7.5, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Horizon (hours)'); ax.set_ylabel('R2')
ax.set_title(f'R2 | Hard={r2_h:.3f}  Soft={r2_s:.3f}')
ax.legend(fontsize=8); ax.set_xticks(range(0,24,2)); ax.set_xticklabels(range(1,25,2))

# (0,2) R2 delta
ax = axes[0,2]
dr2 = r2_soft - r2_hard
colors = ['#2ECC71' if v>0 else '#E74C3C' for v in dr2]
ax.bar(horizons, dr2, color=colors, edgecolor='white')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axvline(x=8.5, color='gray', linestyle=':', alpha=0.5)
n_better = sum(dr2>0); n_worse = sum(dr2<0)
ax.set_xlabel('Horizon (hours)'); ax.set_ylabel('Delta R2')
ax.set_title(f'R2 Gain | Improved: {n_better}h  Degraded: {n_worse}h')
ax.set_xticks(range(1,25,2))

# (1,0) MAE delta
ax = axes[1,0]
dmae = mae_hard - mae_soft
colors = ['#2ECC71' if v>0 else '#E74C3C' for v in dmae]
ax.bar(horizons, dmae, color=colors, edgecolor='white')
ax.axhline(y=0, color='black', linewidth=0.8)
ax.axvline(x=8.5, color='gray', linestyle=':', alpha=0.5)
n_better = sum(dmae>0); n_worse = sum(dmae<0)
ax.set_xlabel('Horizon (hours)'); ax.set_ylabel('Delta MAE (kW, positive=better)')
ax.set_title(f'MAE Gain | Improved: {n_better}h  Degraded: {n_worse}h')
ax.set_xticks(range(1,25,2))

# (1,1) Scatter
ax = axes[1,1]
ax.scatter(hard_pred.flatten(), soft_pred.flatten(), alpha=0.06, s=2, color='#7B1FA2')
pmax = max(hard_pred.max(), soft_pred.max()) * 1.05
ax.plot([0, pmax], [0, pmax], 'gray', linewidth=0.8, linestyle='--')
ax.set_xlim(0, pmax); ax.set_ylim(0, pmax)
ax.set_xlabel('Hard Split (kW)'); ax.set_ylabel('Soft Fusion (kW)')
corr = np.corrcoef(hard_pred.flatten(), soft_pred.flatten())[0,1]
ax.set_title(f'Predictions Scatter | N={len(true)*24:,}  r={corr:.4f}')

# (1,2) Summary
ax = axes[1,2]
ax.axis('off')
rows = [
    ['Metric', 'Hard Split', 'Soft Fusion', 'Delta'],
    ['MAE (kW)', f'{mae_h:.1f}', f'{mae_s:.1f}', f'{mae_h-mae_s:+.1f}'],
    ['RMSE (kW)', f'{rmse_h:.1f}', f'{rmse_s:.1f}', f'{rmse_h-rmse_s:+.1f}'],
    ['R2', f'{r2_h:.4f}', f'{r2_s:.4f}', f'{r2_s-r2_h:+.4f}'],
    ['NMAE', f'{mae_h/2000*100:.2f}%', f'{mae_s/2000*100:.2f}%', ''],
    ['', '', '', ''],
    ['h=1 MAE', f'{mae_hard[0]:.1f}', f'{mae_soft[0]:.1f}', f'{mae_hard[0]-mae_soft[0]:+.1f}'],
    ['h=4 MAE', f'{mae_hard[3]:.1f}', f'{mae_soft[3]:.1f}', f'{mae_hard[3]-mae_soft[3]:+.1f}'],
    ['h=8 MAE', f'{mae_hard[7]:.1f}', f'{mae_soft[7]:.1f}', f'{mae_hard[7]-mae_soft[7]:+.1f}'],
    ['h=12 MAE', f'{mae_hard[11]:.1f}', f'{mae_soft[11]:.1f}', f'{mae_hard[11]-mae_soft[11]:+.1f}'],
    ['h=16 MAE', f'{mae_hard[15]:.1f}', f'{mae_soft[15]:.1f}', f'{mae_hard[15]-mae_soft[15]:+.1f}'],
    ['h=20 MAE', f'{mae_hard[19]:.1f}', f'{mae_soft[19]:.1f}', f'{mae_hard[19]-mae_soft[19]:+.1f}'],
    ['h=24 MAE', f'{mae_hard[23]:.1f}', f'{mae_soft[23]:.1f}', f'{mae_hard[23]-mae_soft[23]:+.1f}'],
    ['', '', '', ''],
    ['h=1 R2', f'{r2_hard[0]:.3f}', f'{r2_soft[0]:.3f}', f'{r2_soft[0]-r2_hard[0]:+.3f}'],
    ['h=8 R2', f'{r2_hard[7]:.3f}', f'{r2_soft[7]:.3f}', f'{r2_soft[7]-r2_hard[7]:+.3f}'],
    ['h=16 R2', f'{r2_hard[15]:.3f}', f'{r2_soft[15]:.3f}', f'{r2_soft[15]-r2_hard[15]:+.3f}'],
    ['h=24 R2', f'{r2_hard[23]:.3f}', f'{r2_soft[23]:.3f}', f'{r2_soft[23]-r2_hard[23]:+.3f}'],
]
tbl = ax.table(cellText=rows, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1.1, 1.25)
for j in range(4):
    tbl[0,j].set_facecolor('#2C3E50')
    tbl[0,j].set_text_props(color='white', fontweight='bold')
for i in range(1, len(rows)):
    val = rows[i][3]
    if val and val.startswith('+'):
        tbl[i,3].set_text_props(color='#2ECC71', fontweight='bold')
    elif val and val.startswith('-'):
        tbl[i,3].set_text_props(color='#E74C3C', fontweight='bold')
ax.set_title('Summary', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/hard_vs_soft_fusion.png', dpi=150, bbox_inches='tight')
plt.close()
print('Saved -> outputs/hard_vs_soft_fusion.png')
print(f'Hard Split: MAE={mae_h:.1f}  RMSE={rmse_h:.1f}  R2={r2_h:.4f}')
print(f'Soft Fusion: MAE={mae_s:.1f}  RMSE={rmse_s:.1f}  R2={r2_s:.4f}')
