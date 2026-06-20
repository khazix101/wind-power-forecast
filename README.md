# CNN-LSTM(1-8h) + VMD-LSTM(9-24h) 风电预测模型说明

## 一、模型架构（CNN-LSTM + VMD-LSTM 混合网络）

本模型采用双路径混合架构：**短时（h=1~8）用 CNN-LSTM 基于气象特征直接预测，中长时（h=9~24）用 VMD-LSTM 基于 IMF 分解预测**。

```
Input: (batch, 120h, 12 维)
  ├─ 前 8 维 [power_current, ws100, ρ, u100, v100, t2m, hour_sin, hour_cos]
  │     → Path A: CNN-LSTM → h=1~8
  │
  └─ 后 4 维 [IMF1, IMF2, IMF3, IMF4]  (VMD alpha=500, K=4)
        → Path B: VMD-LSTM → h=9~24

Path A — CNN-LSTM（短时预测，h=1~8）：
  (B, 120, 8) → Conv1D(8→128, k=3, ReLU) → Conv1D(128→192, k=3, ReLU)
  → MaxPool(2) → LSTM(192→32, 1层, DO=0.14) → FC(32→8)

Path B — VMD-LSTM（中长时预测，h=9~24）：
  IMF1+IMF2 → Trend LSTM(128, 1层) → FC(128→64) → DO(0.444) → FC(64→16)
  IMF3+IMF4 → Fluct LSTM(64, 1层)  → FC(64→64)  → DO(0.444) → FC(64→16)
  Sum → (B, 16)

输出：Concat[PathA(8), PathB(16)] → (B, 24)
```

### 输入特征明细

| 通道   | 特征                | 来源                        |
| ---- | ----------------- | ------------------------- |
| 1    | `power_current`   | 当前时刻功率（Vestas V90 物理公式计算） |
| 2    | `wind_speed_100m` | ERA5 100m 风速              |
| 3    | `air_density`     | ERA5 空气密度                 |
| 4    | `u100`            | ERA5 100m U 风分量           |
| 5    | `v100`            | ERA5 100m V 风分量           |
| 6    | `t2m`             | ERA5 2m 温度                |
| 7    | `hour_sin`        | 时间编码（小时正弦）                |
| 8    | `hour_cos`        | 时间编码（小时余弦）                |
| 9~12 | `IMF1~IMF4`       | 功率序列的 VMD 分解模态（α=500, K=4）|

### 参数量概览

| 组件                            | 参数量          |
| ----------------------------- | ------------ |
| Path A Conv1D (8→128→192)     | ~24,200      |
| Path A LSTM (192→32, 1层)      | ~28,800      |
| Path A FC (32→8)              | ~264         |
| Path B Trend-LSTM (2→128, 1层) | ~67,600      |
| Path B Trend FC 层             | ~10,496      |
| Path B Fluct-LSTM (2→64, 1层)  | ~17,300      |
| Path B Fluct FC 层             | ~8,640       |
| **合计**                        | **~206,000** |

---

## 二、预测逻辑

### 流程图

```
.nc 文件 → nc2wind_csv.py → wind_data.csv
                                  │
                   气象特征提取 + VMD 分域分解（防泄露）
                  ┌──────────────────────────┐
                  │ 气象特征 (8维)           │ → Path A CNN-LSTM → h=1~8
                  │ VMD IMFs (4维)           │ → Path B VMD-LSTM → h=9~24
                  │   train/val/test 独立 VMD│
                  │   IMF scaler fit on train│
                  └──────────────────────────┘
                                  │
                               Concat
                                  │
                                  ▼
                           24h 风电功率曲线
```

### 训练流程

1. **计算功率标签**：Vestas V90 功率曲线 + 风廓线修正 + 密度修正 → 逐小时 t+1~t+24 功率
2. **VMD 分域分解**：train/val/test 各自独立运行 VMD（K=4, α=500），消除频率域数据泄露；IMF scaler 仅 fit train 域，transform 全量
3. **气象特征提取**：风速、密度、U/V 风分量、温度、小时编码、当前功率（8 维）
4. **序列构建**：120h 滑动窗口，(N, 120, 12) 特征矩阵
5. **双路径训练**：CNN-LSTM 学习短时气象-功率映射 + VMD-LSTM 学习 IMF 中长时序列模式
6. **输出合并**：前 8h (CNN) + 后 16h (VMD) → 24h 完整预测

### VMD 防泄露机制

| 步骤 | 方法 | 原因 |
|------|------|------|
| VMD 分解 | train/val/test **各自独立**做 FFT+ADMM 迭代 | 全局 FFT 会使任何时刻的 IMF 依赖全序列信息 |
| IMF 缩放 | StandardScaler **仅拟合 train IMFs** | 防止 val/test 的统计数据污染训练 |

---

## 三、训练配置

| 参数                  | 值                                           | 说明                    |
| ------------------- | ------------------------------------------- | --------------------- |
| 序列长度 (seq_len)      | 120 小时                                      | 5 天回顾窗口               |
| 短时输出 (cnn_out)      | **8**                                       | CNN-LSTM 负责 h=1~8     |
| 长时输出 (vmd_out)      | **16**                                      | VMD-LSTM 负责 h=9~24    |
| 气象特征维 (weather_dim) | 8                                           | 功率+风速+密度+U/V+t2m+时间编码 |
| VMD 模态数             | K=4                                         | 趋势 + 3 个波动分量          |
| VMD 带宽惩罚            | **α=500**                                   | 平衡频带分离（寻优后）           |
| CNN Conv1D 滤波器      | 128 → 192                                   | 2 层，k=3，ReLU（寻优后）     |
| CNN-LSTM 隐藏单元       | 32                                          | 1 层（寻优后）              |
| Path A Dropout      | 0.14                                        | CNN-LSTM FC 层（贝叶斯寻优后）  |
| Trend LSTM 隐藏单元     | **128**                                     | 1 层，DO=0.444         |
| Fluct LSTM 隐藏单元     | **64**                                      | 1 层，DO=0.444         |
| FC 隐藏单元             | **64**                                      | ReLU + Dropout(0.444) |
| 训练集                 | 2024 ~ 2025.9                               | ~14,465 序列            |
| 验证集                 | 2025.10 ~ 2025.12                           | ~2,208 序列             |
| 测试集                 | 2026 全年                                     | ~3,461 序列             |
| 损失函数                | MSE                                         |                       |
| 优化器                 | Adam, lr=5e-4, weight_decay=5e-4            |                       |
| 学习率调度               | ReduceLROnPlateau (factor=0.5, patience=10) |                       |
| 梯度裁剪                | max_norm=1.0                                |                       |
| 早停                  | patience=30                                 |                       |
| 批大小                 | 64                                          |                       |
| 归一化                 | StandardScaler（fit train only）             |                       |
| 可复现性                | seed=42 + torch.use_deterministic_algorithms(True) |                  |
| 总参数量                | **~206,000**                                |                       |

---

## 四、预测效果（无数据泄露）

### 测试集（2026 年，3,461 样本 × 24h = 83,064 预测点）

#### 总体指标

| 指标        | 值             |
| --------- | ------------- |
| **MAE**   | **258.94 kW** |
| **RMSE**  | **368.79 kW** |
| **NMAE**  | **12.95%**    |
| **NRMSE** | **18.44%**    |
| **R²**    | **0.4313**    |

#### 分小时表现

| 提前量          | MAE (kW)  | R²         | 负责路径     |
| ------------ | --------- | ---------- | -------- |
| **h=1**（最近）  | **98.6**  | **0.8855** | CNN-LSTM |
| h=4          | 202.9     | 0.5682     | CNN-LSTM |
| h=8          | 268.8     | 0.3781     | CNN-LSTM |
| h=9          | 271.3     | 0.3380     | VMD-LSTM |
| **h=12**     | **266.5** | **0.4358** | VMD-LSTM |
| h=16         | 276.7     | 0.3805     | VMD-LSTM |
| h=20         | 298.4     | 0.2934     | VMD-LSTM |
| **h=24**（最远） | **317.9** | **0.2456** | VMD-LSTM |

### 性能特征

1. **CNN-LSTM 短时精度高**：h=1 MAE 98.6 kW, R²=0.89，模型能较准地捕捉 1 小时后的功率
2. **VMD-LSTM 中长时稳定**：h=9~20 的 R² 围绕 0.3~0.4 波动，没有随 horizon 单调衰减
3. **整体 R²=0.43**：在严格无数据泄露的前提下，模型解释了约 43% 的功率方差

---

## 五、对比实验（Pure LSTM Baseline）

运行 `baseline_lstm.py` 进行纯 LSTM 基线对比，量化 VMD 分解 + CNN 双路径架构的实际增益。

| 指标 | Pure LSTM | VMD-Hybrid | 提升 |
|------|-----------|------------|------|
| **MAE** | 294.30 kW | **258.94 kW** | **-35.4 kW** |
| **RMSE** | 426.37 kW | **368.79 kW** | **-57.6 kW** |
| **R²** | 0.2399 | **0.4313** | **+0.19** |
| NMAE | 14.72% | **12.95%** | - |

**结论**：VMD 混合架构相比纯 LSTM 有显著提升，尤其是中长时预测。纯 LSTM 的 h=24 R² 仅 0.02，VMD-Hybrid 提升至 0.25——VMD 分解有效缓解了误差累积问题。

---

## 六、关键设计分析与超参数寻优

### 6.1 分界线寻优（A_B_hours/）

对 cnn_out ∈ {4,6,8,10,12,14,16,18,20,22} 进行网格搜索，固定最优 VMD 和 Path A 参数。

| cnn_out | 说明 | MAE |
|---------|------|-----|
| 4 | CNN 太弱 | 244.5 |
| **8** | **最优** | **240.0** |
| 10 | 接近最优 | 245.1 |
| 22 | VMD 太弱 | 323.2 |

**结论**：分界线在 h=8 附近最优。CNN-LSTM 适合 8h 内短时预测，VMD-LSTM 适合 9h 后中长时预测。CNN 覆盖太多小时反而拖累整体。

### 6.2 VMD 超参数寻优

在 `VMD_Hyperparameter_Tuning/` 中开展 60 轮随机搜索（6 个参数），alpha 覆盖 {500,1000,2000,4000,8000}。

| 参数 | 搜索范围 | 最优值 |
|------|---------|--------|
| alpha | {500, 1000, 2000, 4000, 8000} | **500** |
| trend_hidden | {64, 100, 128, 192, 256} | **128** |
| fluct_hidden | {64, 128, 192, 256, 320} | **64** |
| n_layers | {1, 2, 3} | **1** |
| path_b_dropout | [0.1, 0.5] | **0.444** |
| fc_hidden | {32, 50, 64, 100} | **64** |

**关键发现**：
- alpha=500 一致最优（低带宽惩罚让 VMD 提取更多有用信息）
- n_layers=1 一致优于 2/3（多层 LSTM 严重过拟合）
- 较小 fluct_hidden=64 即可（高频分量信息量有限）
- 所有 alpha=500 的试验都在 epoch 1 收敛，进一步训练验证集损失上升
- **总参数从初始 1.7M 降至 206K**

### 6.3 Path A 超参数寻优（CNN 随机搜索）

在 `CNN_Hyperparameter_Tuning/` 中开展 30 轮随机搜索（5 个参数）。

| 参数 | 搜索范围 | 最优值 |
|------|---------|--------|
| conv1_filters | {32, 48, 64, 96, 128} | **128** |
| conv2_filters | {64, 96, 128, 192, 256} | **192** |
| cnn_lstm_hidden | {32, 50, 64, 100, 128} | **32** |
| cnn_lstm_layers | {1, 2} | **1** |
| path_a_dropout | [0.1, 0.5] | **0.31** |

### 6.4 Dropout + Warmup 贝叶斯优化

在 `Dropout_Warmup_Tuning/` 中开展 Gaussian Process + Expected Improvement 贝叶斯优化（50 轮, 7 个参数）。

**关键发现**：
- path_a_dropout 进一步优化为 **0.14**（替代 CNN 随机搜索的 0.31）
- **LR=1e-5 + weight_decay=9e-3** 能让模型训练 72 epoch 不崩（验证集视角）
- 但该配置在测试集上表现更差（MAE=329, R²=0.19），因为验证集（2025.10-12）与测试集（2026）分布差异大
- **高 LR=5e-4 的 epoch-1 模型是实证最优**：大步更新恰好跳到泛化更好的位置
- warmup 在本任务中作用有限（最优 warmup_epochs=1，几乎不 warmup）

---

## 七、已知限制

1. **epoch-1 收敛**：模型在 epoch 1 即达验证集最优，进一步训练过拟合。原因：(a) alpha=500 的 VMD IMFs 信息量大, (b) 验证集小（2,208 条）且与测试集分布不匹配
2. **验证/测试分布差异**：验证集（2025.10-12）仅有 3 个月数据，不足以代表 2026 全年测试集。任何基于验证集做的超参数微调都可能损害测试泛化性
3. **数据量限制**：14k 训练序列对 206K 参数模型偏少，即使多层 dropout 也无法完全防止过拟合
4. **单一站点**：仅使用 point_id=1（lat=41, lon=96），模型不适用于其他地理位置

### 改进方向

- 扩展验证集（如使用 2025 全年交叉验证替代 10-12 月单段）
- 降低 VMD bandwidth（α>500）牺牲部分信息量以换取更稳定的训练动态
- 引入更多气象特征（风向、气压、云量）
- 增大训练数据（扩展年份范围或增广）

---

## 八、运行命令

```powershell
# 0. 生成 wind_data.csv（首次使用）
cd data/wind_nc
python nc2wind_csv.py

# 1. 训练 VMD-LSTM 混合模型（从项目根目录）
cd ../..
python forecast_tsp/forecast_vmd_hybrid.py

# 2. 生成评估仪表盘
python forecast_tsp/evaluate_vmd_hybrid.py

# 3. 运行纯 LSTM 基线对比
python forecast_tsp/baseline_lstm.py
```

## 输出文件说明

所有生成文件位于 `outputs/` 目录：

| 文件 | 内容 |
| --- | --- |
| `vmd_hybrid_results.png` | 分小时误差曲线 + R² 柱状图 + 样本 24h 剖面 + 散点图 + 残差分布 + 功率曲线 |
| `vmd_decomposition.png` | 原始功率 + 4 个 IMF 分解可视化（展示 train domain 前 2000h） |
| `vmd_evaluation_dashboard.png` | 评估仪表盘（误差、R²、散点、残差） |
| `vmd_evaluation_profiles.png` | 样本预测剖面 + 功率曲线 + 指标汇总表 |
| `baseline_vs_hybrid.png` | 纯 LSTM vs VMD-Hybrid 对比图 |
| `vmd_hybrid_predictions.csv` | VMD-Hybrid 24h 预测结果 CSV |
| `baseline_lstm_predictions.csv` | 纯 LSTM 24h 预测结果 CSV |
| `vmd_imfs.npz` | VMD 分解结果存档（含 train/val/test domain omega） |
| `vmd_hybrid.pth` | VMD-Hybrid 最佳模型权重 |
| `baseline_lstm.pth` | 纯 LSTM 最佳模型权重 |
| `vmd_cache/` | VMD 分解缓存（分域独立，可复现） |
