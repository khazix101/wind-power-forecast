## 项目背景

- **当前任务**：风电功率 24 小时 ahead 多步预测（每小时预测，output_dim=24）
- **现有模型**：纯 LSTM（输入：过去 7 天 168 小时滑动窗口，输出：未来 24 小时功率）
- **数据特征**：风速、功率、风向、温度等（feature_dim ≈ 4-5），已归一化
- **痛点**：单步误差累积导致 horizon 越长精度下降严重（24h 预测 MAE 可能是 1h 的 2-3 倍）

## 目标架构（基于论文："Wind power forecasting with a VMD-LSTM-Informer hybrid deep learning model"）

将原始功率序列分解为多个频率分量，对不同分量采用不同的深度学习模型预测，最后重构得到最终结果。

### 核心架构设计

原始功率序列 P(t) ↓ VMD 分解（k=4 个 IMF 分量） ↓ ┌─────────────┬─────────────┬─────────────┬─────────────┐ │ IMF 1 │ IMF 2 │ IMF 3 │ IMF 4 │ │ (低频趋势) │ (低频趋势) │ (中频波动) │ (高频噪声) │ └──────┬──────┘──────┬──────┘──────┬──────┘──────┬──────┘ ↓ ↓ ↓ ↓ LSTM_A LSTM_A LSTM_B LSTM_B (慢速学习) (慢速学习) (快速学习) (快速学习) ↓ ↓ ↓ ↓ 预测 IMF1 预测 IMF2 预测 IMF3 预测 IMF4 └─────────────┴─────────────┴─────────────┘ ↓ Sum() 重构 ↓ 最终 24h 功率预测

### 模型分工策略（关键创新点）

根据论文第 3.5 节及第 4.4 节：

- **IMF1-2（低频/趋势分量）**：使用 **LSTM** 预测（擅长捕捉长期趋势和日周期）
- **IMF3-4（高频/波动分量）**：使用 **LSTM-Informer** 或 **增强型 LSTM** 预测（擅长捕捉快速波动和突变）

**简化实现建议**（如果 Informer 实现复杂）：

- 方案 A（推荐）：IMF1-2 用标准 LSTM，IMF3-4 用另一组 LSTM（但增加 hidden_size 或层数）
- 方案 B（进阶）：IMF3-4 用 LSTM-Informer（需额外实现 ProbSparse Attention）

## 具体实现要求

### 1. VMD 分解模块

- **参数**：IMF 数量 k=4，惩罚因子 α=2000（论文 Table 2-3 验证的最优值）
- **实现**：使用 `vmdpy` 库或 `torch-vmd`（如不可用则用 PyTorch 实现变分模态分解，或使用替代的 EMD/VMD 库）
- **输入**：原始功率序列（训练集+测试集，统一分解以保持一致性）
- **输出**：4 个 IMF 分量 + 1 个残差（residual，可忽略或单独处理）

### 2. 双路径预测模型

#### 路径 A：低频模型（Trend-LSTM）

- **输入**：IMF1 + IMF2 的滑动窗口（lookback=168）
- **架构**：LSTM(hidden_size=100, num_layers=2, dropout=0.2) → Dense(50) → Dense(24)
- **作用**：捕捉功率的日变化趋势和系统性波动

#### 路径 B：高频模型（Fluctuation-LSTM 或 LSTM-Informer）

- **输入**：IMF3 + IMF4 的滑动窗口（lookback=168）
- **架构**（简化版）：
  - LSTM(hidden_size=128, num_layers=2, dropout=0.2)
  - 或增加 Attention 层增强对突变的敏感度
- **作用**：捕捉小时级的湍流、阵风引起的快速波动

### 3. 重构模块

```python
final_prediction = (
    model_trend(imf1_imf2_input) + 
    model_fluctuation(imf3_imf4_input)
)
**注意**：如果包含残差（residual），也需预测并加入重构。

### 4. 数据流改造

保持现有数据管道不变，**在模型训练前增加 VMD 分解步骤**：

# 现有流程

X, y = create_sequences(data, lookback=168, horizon=24)

# 新增流程

imfs = vmd_decompose(data['power'].values, k=4, alpha=2000)

# imfs.shape = (n_samples, 4)  # 4 个 IMF

# 为每个 IMF 创建独立的 sliding window

X_imf1_2, y = create_sequences(imfs[:, :2], lookback=168, horizon=24)  # 低频
X_imf3_4, y = create_sequences(imfs[:, 2:], lookback=168, horizon=24)  # 高频

### 5. 损失函数与训练

- **损失**：MSE（训练） + MAE（监控）
- **优化器**：Adam(lr=1e-4)
- **训练策略**：   - 先分别预训练 Trend-LSTM 和 Fluctuation-LSTM 若干 epoch
  - 再联合微调（可选，为简化可直接端到端训练）
- **Early Stopping**：patience=15，基于验证集 MAE

## 代码结构要求

请提供以下完整代码：

1. **VMD 分解工具类**（`VMDDecomposer`）：
   
   - 支持参数 k=4, alpha=2000
   - 包含 `fit_transform`（对训练集分解）和 `transform`（对测试集分解）
   - **关键**：必须保存训练集的 IMF 用于测试集的分解基准（避免数据泄露）

2. **双路径模型定义**（`VMD_LSTM_Hybrid`）：
   
   - `__init__`：定义 trend_lstm 和 fluctuation_lstm
   - `forward`：接收原始功率序列（batch, lookback, 1），自动分解后分别送入两个子模型，最后求和

3. **修改后的训练循环**：
   
   - 数据加载时集成 VMD 分解
   - 批量训练时，每个 batch 的功率序列先分解为 IMF，再分别送入对应模型

4. **对比实验框架**：
   
   - 保留原始纯 LSTM 作为 baseline
   - 在同一测试集上对比：原始 LSTM vs VMD-LSTM-Hybrid
   - 输出指标：MAE, RMSE, R²，按 horizon（1h, 6h, 12h, 24h）分别统计

## 关键约束与验证清单

- **物理一致性**：重构后的预测功率必须在 [0, 额定容量] 范围内（使用 Clip 或 Sigmoid）
- **维度检查**：VMD 分解后 IMF 数量必须为 4，输入 LSTM 的维度需适配（IMF1-2 拼接后 channel=2，IMF3-4 拼接后 channel=2）
- **无数据泄露**：VMD 分解必须在滑动窗口构建之后、训练之前进行；测试集的 VMD 分解不能使用测试集信息（如使用训练集的中心/边界）
- **计算效率**：VMD 分解是离线操作（训练前一次性完成），在线预测时直接使用预计算的 IMF 分量
- **可复现性**：固定 random_seed=42，保存 VMD 分解的 IMF 分量供后续分析

## 预期效果（基于论文）

- 相比纯 LSTM，24h 预测 MAE 应降低 **20-40%**
- 长 horizon（12-24h）精度提升应比短 horizon（1-3h）更显著（因为 VMD 主要解决误差累积问题）
- R² 提升至 0.93-0.95

## 输出格式

请提供：

1. 完整的 `VMDDecomposer` 类（含参数选择和分解逻辑）
2. 完整的 `VMD_LSTM_Hybrid` 模型类（PyTorch）
3. 修改后的数据准备函数（create_sequences_vmd）
4. 训练脚本的关键修改部分（对比 baseline 和 hybrid 的循环）
5. 评估脚本（按 horizon 分段统计误差）
   注意：如果 `vmdpy` 库不可用，请使用 `PyEMD` 的 EEMD 作为替代方案（添加白噪声的 EMD），或实现简化的经验小波分解（EWT）。