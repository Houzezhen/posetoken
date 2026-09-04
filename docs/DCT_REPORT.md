# DCT 频域 Pose Token 编码报告

## 1. 背景

### 1.1 标准时域分桶的问题

传统的机器人 state/action 编码采用**逐维逐帧独立分桶**（per-dimension, per-timestep binning）：将每个时间步的每个维度独立映射到一个离散桶。Pi0FAST 论文（arXiv:2501.09747）指出这种方法在高频控制场景下存在根本缺陷：

> **相邻时间步的动作高度相关，导致 next-token prediction 退化为"复制上一个 token"，模型陷入局部最优。**

具体来说，机器人 state 在 30Hz 采样下，相邻两帧的位姿差异极小（约 0.03 秒）。如果逐帧独立分桶：
- 相邻 token 的值几乎相同（落入同一桶或相邻桶）
- 模型只需学会"复制上一个 token"就能获得很低的 loss
- 真正的运动趋势和细节差异被淹没在冗余信息中
- 训练效率低下，难以学习精细技能

### 1.2 Pi0FAST 的解决思路

Pi0FAST 提出 **FAST（Frequency-space Action Sequence Tokenization）**，核心洞察：

> **机器人信号在编码前需要先压缩，以减少相邻 token 之间的相关性。**

方法：对时间序列做 DCT（离散余弦变换），将信号从时域转到频域。DCT 的正交性保证频域系数之间**不相关**——每个 token 承载不同频率的信息，模型无法通过"复制"获得低 loss，被迫学习真实的运动特征。

FAST 的完整流程（用于 action 编码）：
1. 归一化 → 2. DCT → 3. 量化（取整 + 丢弃高频）→ 4. 展平 → 5. BPE 压缩

### 1.3 本项目的应用

本项目将 FAST 的核心思想（DCT + 分桶）应用于 PiPER 架构的 **state 编码**，设计 `PoseDCTTokenizer`：

- 对最近 30 帧密集连续 state 的**平移维度**沿时间轴做 DCT
- 丢弃高频系数（只保留前 8 个），减少冗余
- 对频域系数分桶，每个 token 代表一个频率分量
- 相邻 token 承载不同频率信息，**模型能学习到帧间运动差异**

与 FAST 的区别：FAST 编码 action（输出端），本项目编码 state（输入端）；FAST 用 BPE 进一步压缩，本项目用百分位分桶。

## 2. 原理

### 2.1 DCT（离散余弦变换）

DCT-II 将长度为 N 的实数序列变换为 N 个频域系数：

```
X[k] = Σ_{n=0}^{N-1} x[n] · cos(πk(2n+1) / (2N))
```

- `k=0`：直流分量（DC），即序列均值
- `k=1`：基频，一个完整周期/窗口
- `k` 越大：频率越高，对应越细的时序波动
- `k=N-1`：Nyquist 频率，逐点抖动

DCT 是正交变换，频域→时域完全可逆，能量集中在低频系数。

### 2.2 为什么用 DCT 编码机器人 state

**解决相邻帧冗余**：30Hz 采样下，30 帧平移数据在时域高度相关（相邻帧差异极小）。DCT 将其变换为频域系数后：
- `k=0`（DC）：30 帧平均位置 —— 一个 token 概括"在哪"
- `k=1`（基频）：整体运动方向 —— 一个 token 概括"往哪走"
- `k=2~7`：中高频细节 —— 6 个 token 概括"怎么走"
- `k≥8`：丢弃（冗余的逐帧抖动）

30 帧 × 3 维 = 90 个时域值 → 8 个系数 × 3 维 = 24 个频域 token，**压缩 73%**。每个 token 承载不同频率信息，相邻 token 不相关，模型能学习到帧间运动差异。

### 2.3 分桶编码

DCT 系数是连续浮点数，无法直接作为 token。采用百分位分桶：
- **低频系数**（信息量高，动态范围大）→ 稠密桶（256 bin），精细分辨位置和方向
- **高频系数**（信息量低，动态范围小）→ 稀疏桶（64 bin），粗粒度分辨细节

分桶边界由训练数据的 DCT 系数百分位统计生成（`compute_pose_dct_bin_stats.py`），确保每个桶内样本均匀分布。

**分桶的意义**（参考 Pi0FAST）：
- 频域系数已去相关，分桶后的每个 token 承载独立的频率信息
- 模型无法通过"复制相邻 token"获得低 loss，被迫学习运动特征
- 低频稠密桶让模型精细区分"在哪/往哪走"，高频稀疏桶让模型粗粒度感知"运动细节"

## 3. 设计决策

### 3.1 平移走 DCT，旋转不走 DCT

| 维度 | 编码方式 | 原因 |
|------|---------|------|
| 平移 (xyz) | DCT 频域分桶 | 欧氏空间，三轴独立，线性叠加有物理意义，DCT 完全适用 |
| 旋转 (四元数) | 时域逐帧欧拉角分桶 | 四元数不适合频域分析（见下） |

#### 四元数不适合 DCT 的三个原因

1. **单位约束**：`w²+x²+y²+z²=1`，四分量不独立。对每分量独立做 DCT 会得到"非单位"的频谱系数，物理上不存在。

2. **双覆盖**：`q` 和 `-q` 表示同一旋转。位姿序列里某帧符号翻转，旋转本身完全连续，但 DCT 看到阶跃 → 虚假高频爆炸。

3. **非线性叠加**：`0.5*(q₁+q₂)` 不是 q₁ 和 q₂ 之间的中间旋转。DCT 系数（加权和）在四元数空间里没有插值意义。

#### 旋转处理：四元数 → 欧拉角 → 时域分桶

逐帧将四元数转为欧拉角 `[roll, pitch, yaw]`（弧度，范围 [-π, π]），均匀分桶 360 bin（分辨率约 1°），与 `PoseStateTokenizer` 的旋转分桶完全一致。

### 3.2 反归一化时机

训练数据 state 经过 z-score 归一化。`encode_state` 内部在 DCT 之前先反归一化，恢复原始位姿，再转欧拉角。这保证：
- 四元数单位约束得到恢复（归一化会破坏范数）
- DCT 作用在真实物理量上，系数有明确物理意义
- 欧拉角转换基于原始位姿，结果正确

数据流：
```
归一化 state → _denormalize → 原始 state → 平移做 DCT / 旋转转欧拉角 → 分桶 → token
```

### 3.3 密集连续采样

DCT 要求均匀采样。早期版本使用 `PIPER_HISTORY_OFFSETS`（间隔 20 帧的稀疏采样）做 DCT，存在两个问题：
1. 4 个稀疏点做 DCT 无频域意义（Nyquist 频率仅 0.75 Hz）
2. 分桶统计脚本用 stride=1 密集滑窗，与训练时稀疏采样分布不一致

最终方案改为取最近 30 帧密集连续 state（30Hz 下约 1 秒），DCT 有真实的频域分辨率。

## 4. 版本演进

### V0：时域分词（`pose_token` 分支）

- 4 帧稀疏历史 state，逐帧独立分桶
- 平移 2048 bin，旋转 360 bin
- 无频域分析

### V1：全 DCT 频域分词（`pose_token_dct` 初版）

- 4 帧稀疏历史（绑定 `PIPER_HISTORY_OFFSETS`）做 DCT
- 平移 + 旋转都走 DCT
- 低频 2048 bin，高频 360 bin
- **问题**：稀疏采样无频域意义；四元数不适合 DCT；分桶统计与训练采样不一致

### V2：密集 20 帧 DCT

- 改为最近 20 帧密集连续 state
- 仍对平移 + 旋转都做 DCT
- **问题**：四元数耦合问题依然存在；旋转高频系数是噪声

### V3：平移 DCT + 旋转时域（当前版本）

- 平移 (xyz)：最近 30 帧密集 DCT，只保留前 8 个系数
  - 低频 k=0,1 → 256 bin 稠密桶
  - 高频 k=2..7 → 64 bin 稀疏桶
- 旋转 (四元数)：退回时域，逐帧转欧拉角 → 360 bin 均匀分桶
- 反归一化兼容单帧 (D,) 与 per-frame (T*D,) 两种统计量布局

### 版本对比

| 版本 | 窗口 | 平移 | 旋转 | 新增 token 种类 | 每 sample token 数 |
|------|------|------|------|----------------|-------------------|
| V0 时域 | 4 帧稀疏 | 时域 2048 bin | 时域 360 bin | 7224 | 28 |
| V1 全DCT | 4 帧稀疏 | DCT 2048/360 | DCT 2048/360 | 5176 | 24 |
| V2 密集DCT | 20 帧密集 | DCT 2048/360 | DCT 2048/360 | 5176 | 120 |
| V3 当前 | 30 帧密集 | DCT 256/64 (保留8系数) | 时域 360 | 1000 | 114 |

## 5. 当前版本（V3）参数

### 配置（`pi05_piper_pose_dct`）

| 参数 | 值 | 说明 |
|------|-----|------|
| `dct_window_size` | 30 | DCT 时间窗口（帧数） |
| `dct_state_window` | 30 | 密集连续采样帧数 |
| `dct_n_low_freq` | 2 | 低频系数个数（k=0,1） |
| `dct_n_keep_coeffs` | 8 | 平移保留的 DCT 系数总数 |
| `dct_low_bins` | 256 | 低频桶数 |
| `dct_high_bins` | 64 | 高频桶数 |
| `dct_state_dim` | 7 | 单帧 state 维度（xyz + quat_xyzw） |

### Token 构成

每个 30 帧 state 输入编码出 **114 个 pose token**：

| 部分 | 计算 | token 数 |
|------|------|---------|
| 平移低频 | 3 维 × 2 系数（k=0,1） | 6 |
| 平移高频 | 3 维 × 6 系数（k=2..7） | 18 |
| 旋转欧拉角 | 30 帧 × 3 维（roll/pitch/yaw） | 90 |
| **合计** | | **114** |

### Embedding 表

| token 族 | 桶数 | 用途 |
|---------|------|------|
| `dctXYlow` | 256 | 平移 x/y 低频系数 |
| `dctZlow` | 256 | 平移 z 低频系数 |
| `dctXYhigh` | 64 | 平移 x/y 高频系数 |
| `dctZhigh` | 64 | 平移 z 高频系数 |
| `rot` | 360 | 欧拉角（roll/pitch/yaw 共用） |
| **合计** | **1000** | |

- base vocab: 257152
- 扩展后 vocab: 258152（+1000）

## 6. DCT 系数物理意义（window=30, 30Hz）

| 系数 | 频率 | 物理含义 |
|------|------|---------|
| k=0 | DC（0 Hz） | 30 帧平均位置 |
| k=1 | 1 Hz | 整体运动方向（接近/远离） |
| k=2 | 2 Hz | 中等变化 |
| k=5 | 5 Hz | 可见平滑运动 |
| k=7 | 7 Hz | 中频细节（保留的最高频） |
| k≥8 | ≥8 Hz | 丢弃（逐帧抖动级别） |

实际机械臂运动功率谱通常集中在 k<5，保留前 8 个系数已覆盖主要信息。

## 7. 文件清单

| 文件 | 作用 |
|------|------|
| `src/openpi/models/pose_tokenizer.py` | `PoseDCTTokenizer` 实现 |
| `src/openpi/models/pi0_config.py` | DCT 配置字段 |
| `src/openpi/models/pi0.py` | JAX 版 tokenizer 构造 |
| `src/openpi/models_pytorch/pi0_pytorch.py` | PyTorch 版 tokenizer 构造 |
| `src/openpi/training/config.py` | `pi05_piper_pose_dct` 训练配置 |
| `src/openpi/policies/piper_policy.py` | `PiperInputs`（deque 累积 + DCT 窗口提取） |
| `scripts/compute_pose_dct_bin_stats.py` | 分桶边界统计脚本 |
| `bin_stats_dct_w30.pkl` | 当前版本分桶统计（window=30） |

## 8. 使用方式

### 生成分桶统计

```bash
python scripts/compute_pose_dct_bin_stats.py \
    --data-dir=./data \
    --output=./bin_stats_dct_w30.pkl \
    --window 30 --n-low-freq 2 --n-keep-coeffs 8 \
    --low-bins 256 --high-bins 64
```

### 训练

```bash
python scripts/train.py --config=pi05_piper_pose_dct
```

### 部署

```bash
python scripts/serve_policy.py \
    --policy.config=pi05_piper_pose_dct \
    --policy.dir=checkpoints/pi05_piper_pose_dct/<exp_name>/<step>
```

注意：`--policy.config` 必须是 `pi05_piper_pose_dct`（配置名），不是实验目录名。

## 9. 注意事项

1. **旧 checkpoint 不兼容**：V3 的 token 序列（114 个）与 V1（24 个）/V2（120 个）完全不同，旧权重不可用，需重新训练。

2. **推理 deque 累积**：客户端逐帧传 state，`PiperInputs` 内部 deque(maxlen=61) 自然累积，DCT 取最近 30 帧。不足 30 帧时首帧补齐。

3. **主干 state 不受影响**：主干 state 仍按 `PIPER_HISTORY_OFFSETS` 稀疏下采样 4 帧（28 维 + pad = 32 维），2 秒跨度语义不变。DCT 是独立通路。

4. **norm_stats 兼容**：`_denormalize` 兼容单帧 (D,) 与 per-frame (T*D,) 两种统计量布局，旧 norm_stats（28/32 维）可直接使用。
