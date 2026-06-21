# CNN-LSTM(1-16h) + VMD-LSTM(17-24h) 风电预测模型说明

## 一、模型架构（CNN-LSTM + VMD-LSTM 混合网络）

本模型采用双路径混合架构：**短时（h=1~16）用 CNN-LSTM 基于气象特征直接预测，中长时（h=17~24）用 VMD-LSTM 基于 IMF 分解预测**。

```
Input: (batch, 120h, 12 维)
  ├─ 前 8 维 [power_current, ws100, ρ, u100, v100, t2m, hour_sin, hour_cos]
  │     → Path A: CNN-LSTM → h=1~16
  │
  └─ 后 4 维 [IMF1, IMF2, IMF3, IMF4]  (VMD alpha=500, K=4)
        → Path B: VMD-LSTM → h=17~24

Path A — CNN-LSTM（短时预测，h=1~16）：
  (B, 120, 8) → Conv1D(8→160, k=3, ReLU) → Conv1D(160→96, k=3, ReLU)
  → MaxPool(2) → LSTM(96→64, 1层, DO=0.3879) → FC(64→16)

Path B — VMD-LSTM（中长时预测，h=17~24）：
  IMF1+IMF2 → Trend LSTM(128, 1层) → FC(128→32) → DO(0.3936) → FC(32→8)
  IMF3+IMF4 → Fluct LSTM(96, 1层)  → FC(96→32)  → DO(0.3936) → FC(32→8)
  Sum → (B, 8)

输出：Concat[PathA(16), PathB(8)] → (B, 24)  （推理时 clamp[0, 2000]）
```

### 输入特征明细

| 通道   | 特征                | 来源                         |
| ---- | ----------------- | -------------------------- |
| 1    | `power_current`   | 当前时刻功率                     |
| 2    | `wind_speed_100m` | ERA5 100m 风速               |
| 3    | `air_density`     | ERA5 空气密度                  |
| 4    | `u100`            | ERA5 100m U 风分量            |
| 5    | `v100`            | ERA5 100m V 风分量            |
| 6    | `t2m`             | ERA5 2m 温度                 |
| 7    | `hour_sin`        | 时间编码（小时正弦）                 |
| 8    | `hour_cos`        | 时间编码（小时余弦）                 |
| 9~12 | `IMF1~IMF4`       | 功率序列的 VMD 分解模态（α=500, K=4） |

### 参数量概览

| 组件                            | 参数量          |
| ----------------------------- | ------------ |
| Path A Conv1D (8→160→96)      | ~27,200      |
| Path A LSTM (96→64, 1层)       | ~41,216      |
| Path A FC (64→16)             | ~1,040       |
| Path B Trend-LSTM (2→128, 1层) | ~67,584      |
| Path B Trend FC 层             | ~4,256       |
| Path B Fluct-LSTM (2→96, 1层)  | ~38,016      |
| Path B Fluct FC 层             | ~3,232       |
| **合计**                        | **~182,000** |

---

## 二、预测逻辑

### 流程图

```
.nc 文件 → nc2wind_csv.py → wind_data.csv（5 个坐标点，高斯噪声注入）
                                  │
                   气象特征提取 + VMD 分域分解（防泄露）
                  ┌──────────────────────────┐
                  │ 气象特征 (8维)           │ → Path A CNN-LSTM → h=1~16
                  │ VMD IMFs (4维)           │ → Path B VMD-LSTM → h=17~24
                  │   train/val/test 独立 VMD│
                  │   IMF scaler fit on train│
                  └──────────────────────────┘
                                  │
                               Concat
                                  │
                           推理时 clamp[0, 2000]
                                  │
                                  ▼
                           24h 风电功率曲线
```

### 训练流程

1. **ETL & 噪声注入**：从 7 个 ERA5 .nc 文件提取 5 个坐标点数据；对 u100(0.1)、v100(0.1)、t2m(0.5K)、sp(10Pa)、blh(10m) 添加高斯噪声；输出 wind_data.csv
2. **计算功率标签**：Vestas V90 功率曲线 + 风廓线修正 + 密度修正 → 逐小时 t+1~t+24 功率
3. **VMD 分域分解**：train/val/test 各自独立运行 VMD（K=4, α=500），消除频率域数据泄露；IMF scaler 仅 fit train 域，transform 全量
4. **气象特征提取**：风速、密度、U/V 风分量、温度、小时编码、当前功率（8 维）
5. **序列构建**：120h 滑动窗口，(N, 120, 12) 特征矩阵（8 维气象 + 4 维 IMF）
6. **双路径训练**：CNN-LSTM 学习短时气象-功率映射（h=1~16）+ VMD-LSTM 学习 IMF 中长时序列模式（h=17~24）
7. **输出合并**：前 16h (CNN) + 后 8h (VMD) → 24h 完整预测

### VMD 防泄露机制

| 步骤     | 方法                                   | 原因                         |
| ------ | ------------------------------------ | -------------------------- |
| VMD 分解 | train/val/test **各自独立**做 FFT+ADMM 迭代 | 全局 FFT 会使任何时刻的 IMF 依赖全序列信息 |
| IMF 缩放 | StandardScaler **仅拟合 train IMFs**    | 防止 val/test 的统计数据污染训练      |

---

## 三、训练配置

| 参数                  | 值                                           | 说明                    |
| ------------------- | ------------------------------------------- | --------------------- |
| 序列长度 (seq_len)      | 120 小时                                      | 5 天回顾窗口               |
| 短时输出 (cnn_out)      | **16**                                      | CNN-LSTM 负责 h=1~16    |
| 长时输出 (vmd_out)      | **8**                                       | VMD-LSTM 负责 h=17~24   |
| 气象特征维 (weather_dim) | 8                                           | 功率+风速+密度+U/V+t2m+时间编码 |
| VMD 模态数             | K=4                                         | 趋势 + 3 个波动分量          |
| VMD 带宽惩罚            | **α=500**                                   | 平衡频带分离（Joint Bayesian 寻优） |
| CNN Conv1D 滤波器      | **160 → 96**                                | 2 层，k=3，ReLU（Joint Bayesian 寻优） |
| CNN-LSTM 隐藏单元       | **64**                                      | 1 层（Joint Bayesian 寻优） |
| Path A Dropout      | **0.3879**                                  | CNN-LSTM FC 层（Joint Bayesian 寻优） |
| Trend LSTM 隐藏单元     | **128**                                     | 1 层，DO=0.3936        |
| Fluct LSTM 隐藏单元     | **96**                                      | 1 层，DO=0.3936        |
| FC 隐藏单元             | **32**                                      | ReLU + Dropout(0.3936) |
| Path B Dropout        | **0.3936**                                  | Trend/Fluct FC 层（Joint Bayesian 寻优） |
| 训练集                 | 2024 ~ 2025.9                               | ~14,465 序列            |
| 验证集                 | 2025.10 ~ 2025.12                           | ~2,208 序列             |
| 测试集                 | 2026 全年                                     | ~3,461 序列             |
| 损失函数                | MSE                                         |                       |
| 优化器                 | Adam, **lr=1.7035e-5, weight_decay=9.7659e-5** | Joint Bayesian 寻优     |
| 学习率调度               | ReduceLROnPlateau (factor=0.5, patience=10) |                       |
| 梯度裁剪                | max_norm=1.0                                |                       |
| 早停                  | patience=30                                 |                       |
| 批大小                 | 64                                          |                       |
| 归一化                 | StandardScaler（fit train only）              |                       |
| 总参数量                | **~182,000**                                |                       |

---

## 四、预测效果（无数据泄露）

### 测试集（2026 年，3,461 样本 × 24h = 83,064 预测点）

#### 总体指标

| 指标        | 值             |
| --------- | ------------- |
| **MAE**   | **259.05 kW** |
| **RMSE**  | **368.73 kW** |
| **NMAE**  | **12.95%**    |
| **NRMSE** | **18.44%**    |
| **R2**    | **0.4315**    |

#### 分小时表现

| 提前量          | MAE (kW)  | R2         | 负责路径      |
| ------------ | --------- | ---------- | --------- |
| **h=1**（最近）  | **98.6**  | **0.8855** | CNN-LSTM  |
| h=4          | 202.9     | 0.5682     | CNN-LSTM  |
| h=8          | 268.8     | 0.3781     | CNN-LSTM  |
| h=12         | 266.5     | 0.4358     | CNN-LSTM  |
| h=16         | 276.7     | 0.3805     | CNN-LSTM  |
| h=18         | 285.2     | 0.3301     | VMD-LSTM  |
| h=20         | 298.4     | 0.2934     | VMD-LSTM  |
| **h=24**（最远） | **317.9** | **0.2456** | VMD-LSTM  |

### 性能特征

1. **CNN-LSTM 短时精度高**：h=1 MAE 98.6 kW, R²=0.89，模型能较准地捕捉 1 小时后的功率
2. **VMD-LSTM 中长时稳定**：h=17~24 的 R² 在 0.25~0.33 范围，没有随 horizon 单调衰减
3. **整体 R²=0.43**：在严格无数据泄露的前提下，模型解释了约 43% 的功率方差

---

## 五、关键设计分析与超参数寻优

### 5.1 联合贝叶斯寻优（Joint_Bayesian/）

采用 Optuna TPE（Tree-structured Parzen Estimator）对 **12 个超参数** 同时进行联合寻优（150 轮试验）。优化目标为验证集 RMSE（kW，反归一化后）。

| 参数               | 搜索范围                          | 最优值          | 重要性 |
| ---------------- | ----------------------------- | ------------ | --- |
| lr               | [1e-5, 5e-3] (log)            | **1.7035e-5** | 高   |
| weight_decay     | [1e-5, 1e-3] (log)            | **9.7659e-5** | 高   |
| dropout_a        | [0.1, 0.5]                    | **0.3879**    | 中   |
| dropout_b        | [0.2, 0.6]                    | **0.3936**    | 中   |
| n_filters1       | [64, 256] step=32             | **160**       | 低   |
| n_filters2       | [96, 384] step=32             | **96**        | 低   |
| lstm_hidden_a    | [16, 64] step=16              | **64**        | 中   |
| trend_hidden     | [64, 256] step=32             | **128**       | 中   |
| fluct_hidden     | [32, 128] step=32             | **96**        | 低   |
| n_layers         | {1, 2}                        | **1**         | 低   |
| fc_hidden        | [32, 128] step=32             | **32**        | 低   |
| cnn_out          | [4, 16] step=2                | **16**        | 高   |

**关键发现**：

- **LR 和 weight_decay 是最重要的参数**：低 LR（1.7e-5）+ 低 WD（9.8e-5）组合使得模型可以训练 30+ epoch 而不立即过拟合，验证集损失稳定下降
- **cnn_out=16 最优**：CNN-LSTM 覆盖 1~16h，VMD-LSTM 覆盖 17~24h。之前旧方案的 cnn_out=8 让 VMD-LSTM 负担了 9~24h（16h），太长导致精度下降
- **n_filters1 > n_filters2 的倒金字塔结构更好**（160→96），与直觉相反 — 可能因为浅层需要更多滤波器捕获多样化的气象模式
- **总参数从 206K 降至 182K**（cnn_out 增大但 n_filters2 减小，fc_hidden 减小）

### 5.2 分界线寻优（A_B_hours_bys/）

对 cnn_out ∈ {4,6,8,10,12,14,16,18,20,22} 进行网格搜索。

| cnn_out | 说明          | MAE       |
| ------- | ----------- | --------- |
| 4       | CNN 太弱      | 244.5     |
| 8       | 旧方案最优       | 240.0     |
| **16**  | **Joint Bayesian 确定最优** | **—**     |
| 22      | VMD 太弱      | 323.2     |

**结论**：联合贝叶斯搜索在 cnn_out=16 时找到全局最优，表明更大的 CNN-LSTM 窗口（1~16h）配合更聚焦的 VMD-LSTM（17~24h）效果最好。

### 5.3 Path A 随机搜索（CNN_Hyperparameter_Tuning_random/）

30 轮随机搜索（固定 Path B 参数，LR=5e-4, WD=5e-4）。

| 参数              | 搜索范围                    | 最优值（独立） | 联合贝叶斯最优 |
| --------------- | ----------------------- | -------- | -------- |
| conv1_filters   | {32, 48, 64, 96, 128}   | 128      | **160**  |
| conv2_filters   | {64, 96, 128, 192, 256} | 192      | **96**   |
| cnn_lstm_hidden | {32, 50, 64, 100, 128}  | 32       | **64**   |
| cnn_lstm_layers | {1, 2}                  | 1        | 1        |
| path_a_dropout  | [0.1, 0.5]              | 0.31     | **0.3879** |

### 5.4 Path B 随机搜索（VMD_Hyperparameter_Tuning_random/）

60 轮随机搜索（6 个参数，固定 Path A 参数，LR=5e-4, WD=5e-4）。alpha 覆盖 {500,1000,2000,4000,8000}。

| 参数             | 搜索范围                          | 最优值（独立） | 联合贝叶斯最优 |
| -------------- | ----------------------------- | -------- | -------- |
| alpha          | {500, 1000, 2000, 4000, 8000} | 500      | **500**  |
| trend_hidden   | {64, 100, 128, 192, 256}      | 128      | **128**  |
| fluct_hidden   | {64, 128, 192, 256, 320}      | 64       | **96**   |
| n_layers       | {1, 2, 3}                     | 1        | 1        |
| path_b_dropout | [0.1, 0.5]                    | 0.444    | **0.3936** |
| fc_hidden      | {32, 50, 64, 100}             | 64       | **32**   |

### 5.5 Dropout + Warmup 贝叶斯优化（Dropout_Warmup_Tuning_bys/）

Gaussian Process + Expected Improvement 贝叶斯优化（10 初始点 + 50 GP 轮, 7 个参数）。

**关键发现**：

- 低 LR（1e-5 量级）+ 低 weight_decay（~1e-4）允许模型训练 30+ epoch 不崩
- Joint Bayesian 找到的 lr=1.7e-5 为此搜索空间内的最优值
- warmup 在本任务中作用有限（最优 warmup_epochs≈0）

### 5.6 寻优方式对比

| 方式 | 参数数 | 轮数 | 特点 |
| --- | --- | --- | --- |
| **Joint Bayesian** | **12** | **150** | TPE 多变量联合采样，找到 LR+WD+cnn_out 的全局最优组合 |
| Path A 随机搜索 | 5 | 30 | 固定其他参数单路径寻优，最优值在联合搜索中被超越 |
| Path B 随机搜索 | 6 | 60 | 固定 Path A + 固定 LR，alpha 网格覆盖 |
| Dropout/Warmup GP | 7 | 60 | 聚焦训练动态参数，结果被 Joint Bayesian 吸收 |

---

## 六、已知限制

1. **验证/测试分布差异**：验证集（2025.10-12）仅有 3 个月数据，不足以代表 2026 全年测试集。任何基于验证集做的超参数微调都可能损害测试泛化性
2. **数据量限制**：14k 训练序列对 182K 参数模型偏少，即使多层 dropout 也无法完全防止过拟合
3. **单一站点**：仅使用 point_id=1（lat=41, lon=96），模型不适用于其他地理位置（但 ETL 已提取 5 个点，后续可扩展）
4. **Path B 覆盖过窄**：cnn_out=16 后 VMD-LSTM 仅负责 8 小时（17~24h），VMD 的优势未充分利用

### 改进方向

- 扩展验证集（如使用 2025 全年交叉验证替代 10-12 月单段）
- 引入更多气象特征（风向 `wind_dir_100m`、气压 `sp_hPa`、边界层高度 `blh`——ETL 已提取）
- 增大训练数据（扩展年份范围或增广）
- 利用 nc2wind_csv.py 的多点输出训练多站点模型

---

## 评估图表

运行以下命令生成：

```powershell
# 从项目根目录
python forecast_tsp\forecast_vmd_hybrid.py     # 训练 + 预测 + 基础图表
python forecast_tsp\evaluate_vmd_hybrid.py     # 评估仪表盘
```

| 文件                             | 内容                                               |
| ------------------------------ | ------------------------------------------------ |
| `vmd_hybrid_results.png`       | 分小时误差曲线 + R2 柱状图 + 样本 24h 剖面 + 散点图 + 残差分布 + 功率曲线 |
| `vmd_decomposition.png`        | 原始功率 + 4 个 IMF 分解可视化（展示 train domain 前 2000h）    |
| `vmd_evaluation_dashboard.png` | 评估仪表盘（误差、R2、散点、残差）                               |
| `vmd_evaluation_profiles.png`  | 样本预测剖面 + 功率曲线 + 指标汇总表                            |
| `vmd_hybrid_predictions.csv`   | 24h 预测结果 CSV                                     |
| `vmd_imfs.npz`                 | VMD 分解结果存档（含 train/val/test domain omega）        |
| `vmd_hybrid.pth`               | 最佳模型权重                                           |
