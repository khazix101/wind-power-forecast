# 模型调优计划

## 当前状态

- **模型**：VMD-LSTM Hybrid，双路径架构（CNN-LSTM 1-8h + VMD-LSTM 9-24h）
- **基线性能**：MAE=258.94 kW, RMSE=368.79 kW, R²=0.4313, NMAE=12.95%
- **纯 LSTM 基线**：MAE=294.30 kW, RMSE=426.37 kW, R²=0.2399
- **核心问题**：
  1. CNN/VMD 路径在 h=8 处硬切换，h=16 时 R² 跌至 0.10 以下
  2. MSE 损失对各风速段一视同仁，10-15 m/s 爬坡区误解代价大
  3. 验证集仅 3 个月（2025.10-12），不代表测试集（2026.1-5），调参方向可能偏离
  4. 缺少风向、功率变化率等时序衍生特征
  5. LSTM 仅取最后时刻 hidden state，120 小时信息压缩损耗大

---

## Phase 1：软融合 + 功率感知损失（核心改进）

预期收益：R² +0.05~0.10，爬坡区 MAE 显著降低

### 1.1 软融合替代硬切换

当前两条路径各输出一部分 horizon，在边界处硬拼接。改为两条路径各输出完整 24 小时，通过可学习门控融合。

**修改文件**：`forecast_tsp/vmd_hybrid_model.py`

**架构变更**：

```
输入: (B, 120, 12)

Path A (CNN-LSTM): 天气特征 → 输出 (B, 24)
Path B (VMD-LSTM): IMF特征   → 输出 (B, 24)

FusionGate: weight = sigmoid(Linear(concat([a_last, b_last])))  → (B, 24)
output = weight * a + (1 - weight) * b
```

**改动要点**：
- `self.cnn_fc` 输出从 `cnn_out` 改为 24
- `self.trend_fc2` 和 `self.fluct_fc2` 输出从 `vmd_out` 改为 24
- 新增 `self.gate_fc = nn.Linear(cnn_lstm_hidden + fc_hidden, 24)`
- forward: `gate = torch.sigmoid(self.gate_fc(torch.cat([a_hidden, b_hidden], dim=-1)))`

移除 `cnn_out` 超参数，模型自动学习最优分工。

### 1.2 功率曲线感知损失

对 MSE 按功率曲线局部梯度加权，使模型更关注爬坡区。

**修改文件**：`forecast_tsp/forecast_vmd_hybrid.py`

**实现**：

```python
def power_curve_sensitivity_loss(pred, true, ws_current, alpha=1.0):
    """按功率曲线局部梯度的归一化值加权 MSE"""
    # 计算当前风速在功率曲线上的敏感度
    v_hub = wind_at_hub(ws_current)
    dp = power_curve_v90(v_hub + 0.5) - power_curve_v90(v_hub - 0.5)  # 局部梯度近似
    weight = 1.0 + alpha * np.abs(dp) / (np.abs(dp).max() + 1e-8)
    se = (pred - true) ** 2
    return (weight * se).mean()
```

`ws_current` 来自输入特征的 `wind_speed_100m` 列（已做反缩放）。

---

## Phase 2：验证策略修复

阻止调参误入歧途。

### 2.1 拉长验证集

**当前**：验证集为 2025.10-12（3 个月，~2,208 样本）

**改进**：验证集扩展为 2025 全年（~8,760 样本），覆盖四季风况

**修改文件**：`forecast_tsp/forecast_vmd_hybrid.py`

```python
# 旧
val_mask = (seq_years == 2025) & (seq_months >= 10)
# 新
val_mask = seq_years == 2025
```

训练集相应缩减为仅 2024 年。

### 2.2 测试集保持 2026 年不变

数据划分方案：
| 用途 | 时间范围 | 约样本数 |
|------|---------|:--------:|
| 训练 | 2024 全年 | ~8,760 |
| 验证 | 2025 全年 | ~8,760 |
| 测试 | 2026.1-5 | ~3,500 |

---

## Phase 3：特征增强

### 3.1 风向特征

`u100` 和 `v100` 是风速的笛卡尔分量，模型需自行学习风向信息。显式注入可降低学习难度。

**修改文件**：`forecast_tsp/forecast_vmd_hybrid.py`

**新增特征**：
| 特征 | 公式 | 维度 |
|------|------|:--:|
| `wind_dir_sin` | sin(arctan2(v100, u100)) | 1 |
| `wind_dir_cos` | cos(arctan2(v100, u100)) | 1 |

`WEATHER_DIM` 从 8 增加到 10。

### 3.2 功率变化率（爬坡率）

短期预测中，当前功率的变动趋势是强信号。

| 特征 | 公式 | 维度 |
|------|------|:--:|
| `power_ramp_1h` | power_current(t) - power_current(t-1) | 1 |
| `power_ramp_6h` | power_current(t) - power_current(t-6) | 1 |

`WEATHER_DIM` 从 10 增加到 12。

注意：构建序列后前几行 ramp 为 NaN，用 0 填充即可。

---

## Phase 4：Attention 机制（中长期预测增强）

当前 LSTM 输出仅取最后时间步，120 小时信息经过长距离压缩。加入自注意力让模型自主选择关键历史时刻。

**修改文件**：`forecast_tsp/vmd_hybrid_model.py`

**实现**：在 Path B（VMD-LSTM）的 LSTM 之后加一层 Multi-Head Self-Attention：

```python
# 替代 a_out[:, -1, :]
self.vmd_attn = nn.MultiheadAttention(embed_dim=trend_hidden, num_heads=4, batch_first=True)
# forward:
lstm_out, _ = self.trend_lstm(imf_low)       # (B, 120, trend_hidden)
attn_out, _ = self.vmd_attn(lstm_out, lstm_out, lstm_out)
t_out = attn_out.mean(dim=1)                  # 注意力加权池化
```

Path A 的 CNN 已有局部感受野 + MaxPool 压缩，暂不加 Attention，避免过度复杂化。

---

## Phase 5：损失函数扩展

### 5.1 Huber Loss 选项

Huber Loss 对离群点更鲁棒（如弃风限电导致的突发零功率，或传感器异常尖峰）。

```python
huber = nn.SmoothL1Loss(beta=100.0)  # beta 设为大值，接近 MSE 但抗离群
```

### 5.2 多任务损失

联合优化短期和长期预测：

```python
loss = mse(all_24h) + 0.5 * mse(h1_h4) + 0.3 * mse(h21_h24)
```

鼓励模型在不过分牺牲短期精度的前提下改善长期预报。

---

## Phase 6（可选）：多站点联合训练

当前仅使用 `point_id=1`，数据中还有 4 个邻近站点。

**方案 A**：多站点联合训练（扩大样本量 5 倍）

- 每个站点作为独立样本，模型输入增加 `point_id` 嵌入
- VMD 需要逐站点独立分解（频谱特性不同）

**方案 B**：迁移学习

- 在其他站点预训练 LSTM 编码器
- 在目标站点微调全连接层

---

## 依赖关系

```
Phase 1.1 (软融合)  ──独立──→  可单独验证
Phase 1.2 (加权损失) ──依赖──→  Phase 1.1（需新的 24h 输出结构）
Phase 2 (验证修复)   ──独立──→  可随时做
Phase 3 (特征增强)   ──独立──→  需同步修改 model 输入维度
Phase 4 (Attention)  ──独立──→  需调整 model 结构
Phase 5 (损失扩展)   ──依赖──→  Phase 1.1
Phase 6 (多站点)     ──独立──→  工作量大，低优先级
```

```
Phase 2 (验证修复)
    ↓
Phase 1.1 (软融合) ──→ Phase 1.2 (加权损失)
    │
    ├──→ Phase 3 (特征增强)
    │
    └──→ Phase 4 (Attention)
              │
              └──→ Phase 5 (多任务损失)
```

## 建议执行顺序

1. **Phase 2 先行**：验证策略修复，成本最低，确保后续调参不跑偏
2. **Phase 1.1**：软融合架构改造，是收益最大的单一改动
3. **Phase 1.2 + 3 + 4 并行**：损失函数、特征、Attention 三者独立，可同时开发
4. **Phase 5**：在前述改动稳定后微调损失函数
5. **Phase 6 按需**：如果前 5 个 Phase 达到 NMAE < 10%，可暂缓
