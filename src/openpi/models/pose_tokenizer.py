"""PoseVLA 风格的 State 离散化编码器。

提供两种编码方式:
  1) PoseStateTokenizer (时域): 对单帧 state [x,y,z,qw,qx,qy,qz] 做语义化分桶
     - 平移 (x, y): 百分位非均匀分桶 (2048 bins) → <transXY...>
     - 平移 (z):      百分位非均匀分桶 (2048 bins) → <transZ...>
     - 旋转 (quaternion → euler): 均匀分桶 (360 bins, 1° 精度) → <rot...>

  2) PoseDCTTokenizer (频域, pose_token_dct 分支): 参考 pi0fast 思路,
     对 state 时间序列 [T, D] 在时间轴做 DCT, 得到 T 个频域系数后再分桶。
     先逐帧把四元数转欧拉角 (4→3 维), 再对欧拉角时序做 DCT。
     - 低频系数 (整体运动趋势, 信息量高) → 复用 transXY/transZ 2048-bin 稠密分桶
     - 高频系数 (细节抖动, 信息量低)     → 复用 rot 360-bin 稀疏分桶
     这样保留原有 "稀疏(360) + 稠密(2048)" 双分布结构, 同时把信号转到频域。

由于 SentencePiece 不支持运行时添加 token, pose token 被映射为
> base_vocab_size 的整数 ID, 直接绕过 SentencePiece 编码。
"""

import logging
import os
import pickle

import numpy as np
from scipy.fft import dct, idct

logger = logging.getLogger("openpi")

# PaliGemma 原始词表大小
BASE_VOCAB_SIZE = 257152

# state 结构: [x, y, z, qx, qy, qz, qw]  (xyzw 四元数顺序)
DEFAULT_POSE_SPLIT = {
    "trans_xyz": [0, 1, 2],
    "quat_xyzw": [3, 4, 5, 6],
    "other": [],
}


class PoseStateTokenizer:
    """将 7D 位姿 state 编码为 PoseVLA 风格的语义化 token。

    编码流程:
        x → percentiles_txy 分桶 → <transXY1024>
        y → percentiles_txy 分桶 → <transXY0512>
        z → percentiles_tz  分桶 → <transZ0890>
        [qw,qx,qy,qz] → euler [roll,pitch,yaw] → 360bin → <rot045><rot090><rot180>

    输出 token 字符串: "<transXY1024><transXY0512><transZ0890><rot045><rot090><rot180>"
    """

    def __init__(
        self,
        bin_stats_path: str | None = None,
        base_vocab_size: int = BASE_VOCAB_SIZE,
        # 训练数据中 state 经过 Normalize (z-score), 分桶边界与欧拉角都基于原始值,
        # 因此编码前需用这些统计量反归一化, 恢复原始位姿。None 表示跳过 (原样使用)。
        state_norm_mean: np.ndarray | None = None,
        state_norm_std: np.ndarray | None = None,
        state_norm_q01: np.ndarray | None = None,
        state_norm_q99: np.ndarray | None = None,
    ):
        """
        Args:
            bin_stats_path: 预计算的分桶边界 pkl 文件路径。
                            包含:
                              - percentiles_txy: (2049,) 平移 XY 分桶边界
                              - percentiles_tz:  (2049,) 平移 Z 分桶边界
            base_vocab_size: PaliGemma 原始词表大小，pose token ID 从此值开始。
            state_norm_mean/std: 归一化统计量 (z-score), 用于编码前反归一化。
            state_norm_q01/q99:  归一化统计量 (quantile), 用于编码前反归一化。
        """
        self._base_vocab_size = base_vocab_size
        self._state_mean = np.asarray(state_norm_mean, dtype=np.float64) if state_norm_mean is not None else None
        self._state_std = np.asarray(state_norm_std, dtype=np.float64) if state_norm_std is not None else None
        self._state_q01 = np.asarray(state_norm_q01, dtype=np.float64) if state_norm_q01 is not None else None
        self._state_q99 = np.asarray(state_norm_q99, dtype=np.float64) if state_norm_q99 is not None else None

        # --- 生成 token 字符串列表 ---
        self.TRANS_XY_TOKENS = self._gen_tokens("transXY", 2048)
        self.TRANS_Z_TOKENS = self._gen_tokens("transZ", 2048)
        self.ROT_TOKENS = self._gen_tokens("rot", 360)

        # --- 加载非均匀分桶边界 ---
        if bin_stats_path and os.path.exists(bin_stats_path):
            with open(bin_stats_path, "rb") as f:
                data = pickle.load(f)
            self.percentiles_txy = np.asarray(data["percentiles_txy"], dtype=np.float64)
            self.percentiles_tz = np.asarray(data["percentiles_tz"], dtype=np.float64)
            logger.info(
                "Loaded pose bin stats from %s (txy: %d bins, tz: %d bins)",
                bin_stats_path,
                len(self.percentiles_txy) - 1,
                len(self.percentiles_tz) - 1,
            )
        else:
            logger.warning("Bin stats not found at %s, falling back to uniform bins [-1, 1].", bin_stats_path)
            self.percentiles_txy = np.linspace(-1, 1, 2049, dtype=np.float64)
            self.percentiles_tz = np.linspace(-1, 1, 2049, dtype=np.float64)

        # --- 构建 token → ID 映射 (ID 从 base_vocab_size 开始) ---
        all_tokens = self.all_special_tokens
        self._token_to_id = {tok: base_vocab_size + i for i, tok in enumerate(all_tokens)}
        self._vocab_size = base_vocab_size + len(all_tokens)

    @staticmethod
    def _gen_tokens(prefix: str, num: int) -> list[str]:
        """生成 token 字符串列表: <prefix0000>, <prefix0001>, ..."""
        width = len(str(num - 1))
        return [f"<{prefix}{str(i).zfill(width)}>" for i in range(num)]

    # ===== 编码核心 =====

    def _encode_value(self, value: float, percentiles: np.ndarray, token_list: list[str]) -> str:
        """单个标量值 → 语义化 token 字符串"""
        bin_idx = int(np.digitize(value, percentiles, right=False)) - 1
        bin_idx = int(np.clip(bin_idx, 0, len(token_list) - 1))
        return token_list[bin_idx]

    @staticmethod
    def _quaternion_to_euler(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        """四元数 [w, x, y, z] → 欧拉角 [roll, pitch, yaw] (弧度, 范围 [-π, π])"""
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.array([roll, pitch, yaw], dtype=np.float64)

    def encode_state(self, state: np.ndarray, pose_split: dict | None = None) -> str:
        """将完整 state 向量编码为 token 字符串。

        Args:
            state: (D,) float, 已归一化到合适的值域 (通常 [-1, 1])
            pose_split: state 结构描述。若为 None, 使用 DEFAULT_POSE_SPLIT。

        Returns:
            token_str: 如 "<transXY1024><transXY0512><transZ0890><rot045><rot090><rot180>"
        """
        if pose_split is None:
            pose_split = DEFAULT_POSE_SPLIT

        state = np.asarray(state, dtype=np.float64).flatten()
        # 反归一化: 训练/推理时 state 先经过 Normalize, 分桶与欧拉角基于原始值,
        # 因此先恢复原始位姿再编码。与 Normalize 一致, 统计量按前 N 维切片
        # (统计量维度 >= state 维度时使用前 state.shape[0] 维)。
        if self._state_mean is not None and self._state_std is not None and state.shape[0] <= self._state_mean.shape[0]:
            state = state * (self._state_std[: state.shape[0]] + 1e-6) + self._state_mean[: state.shape[0]]
        elif self._state_q01 is not None and self._state_q99 is not None and state.shape[0] <= self._state_q01.shape[0]:
            state = (state + 1.0) / 2.0 * (self._state_q99[: state.shape[0]] - self._state_q01[: state.shape[0]] + 1e-6) + self._state_q01[: state.shape[0]]
        parts = []

        # --- 平移维度 ---
        trans_idx = pose_split.get("trans_xyz", [])
        if len(trans_idx) >= 3:
            tx, ty, tz = state[trans_idx[0]], state[trans_idx[1]], state[trans_idx[2]]
            parts.append(self._encode_value(tx, self.percentiles_txy, self.TRANS_XY_TOKENS))
            parts.append(self._encode_value(ty, self.percentiles_txy, self.TRANS_XY_TOKENS))
            parts.append(self._encode_value(tz, self.percentiles_tz, self.TRANS_Z_TOKENS))

        # --- 旋转维度: 四元数 → 欧拉角 → 360 bin ---
        # 支持 xyzw 和 wxyz 两种顺序
        quat_idx = pose_split.get("quat_xyzw", [])
        if len(quat_idx) >= 4:
            qx, qy, qz, qw = state[quat_idx[0]], state[quat_idx[1]], state[quat_idx[2]], state[quat_idx[3]]
            euler = self._quaternion_to_euler(qw, qx, qy, qz)
            rot_bins = ((euler + np.pi) / (2 * np.pi) * 359).astype(int)
            rot_bins = np.clip(rot_bins, 0, 359)
            for b in rot_bins:
                parts.append(self.ROT_TOKENS[int(b)])
        else:
            quat_idx = pose_split.get("quat_wxyz", [])
            if len(quat_idx) >= 4:
                qw, qx, qy, qz = state[quat_idx[0]], state[quat_idx[1]], state[quat_idx[2]], state[quat_idx[3]]
                euler = self._quaternion_to_euler(qw, qx, qy, qz)
                rot_bins = ((euler + np.pi) / (2 * np.pi) * 359).astype(int)
                rot_bins = np.clip(rot_bins, 0, 359)
                for b in rot_bins:
                    parts.append(self.ROT_TOKENS[int(b)])

        # --- 其他维度 (如 gripper) ---
        other_idx = pose_split.get("other", [])
        for idx in other_idx:
            if idx < len(state):
                # 其他维度也用 XY 分桶 (可扩展)
                parts.append(self._encode_value(state[idx], self.percentiles_txy, self.TRANS_XY_TOKENS))

        return "".join(parts)

    def encode_to_ids(self, token_str: str) -> list[int]:
        """将 pose token 字符串编码为 ID 序列。

        将 "<transXY1024><rot045>" 解析为 [base+0, base+45, ...] 的整数 ID 列表。
        这些 ID > base_vocab_size, 绕过 SentencePiece。

        Args:
            token_str: encode_state() 返回的 token 字符串

        Returns:
            ID 列表, 每个 ID >= base_vocab_size
        """
        ids = []
        i = 0
        while i < len(token_str):
            if token_str[i] == "<":
                end = token_str.find(">", i)
                if end == -1:
                    break
                token = token_str[i : end + 1]
                if token in self._token_to_id:
                    ids.append(self._token_to_id[token])
                    i = end + 1
                    continue
            i += 1
        return ids

    # ===== 属性 =====

    @property
    def all_special_tokens(self) -> list[str]:
        """返回所有需要注册的特殊 token"""
        return self.TRANS_XY_TOKENS + self.TRANS_Z_TOKENS + self.ROT_TOKENS

    @property
    def num_special_tokens(self) -> int:
        """新增 token 数量"""
        return len(self.all_special_tokens)

    @property
    def extended_vocab_size(self) -> int:
        """扩展后的词表大小"""
        return self._vocab_size

    @property
    def token_to_id(self) -> dict[str, int]:
        """token → ID 映射 (只读)"""
        return dict(self._token_to_id)


class PoseDCTTokenizer:
    """频域 Pose Token 编码器 (pose_token_dct 分支)。

    平移 (xyz) 沿时间轴做 DCT, 只保留前 n_keep_coeffs 个低频系数再分桶:
        trans[T, 3]  --DCT(时间轴)-->  coeffs[T, 3]  --取前 n_keep_coeffs-->  --分桶-->  token IDs

    旋转 (四元数) 不适合频域分析 (单位约束/双覆盖/非线性叠加), 退回时域:
    逐帧把四元数转欧拉角 [roll, pitch, yaw], 均匀分桶 360 bin —— 与
    PoseStateTokenizer 的 rot 分桶完全一致。

    平移分桶结构 (compute_pose_dct_bin_stats.py 生成):
        - 低频系数 (前 n_low_freq 个, 整体运动趋势) → 256-bin 稠密桶
        - 高频系数 (第 n_low_freq..n_keep_coeffs 个, 细节)   → 64-bin 稀疏桶

    编码流程 (以 T=30, 输入 D=7 [xyz + quat_xyzw] 为例):
        1) state[30,7] 反归一化到原始位姿
        2) 平移: 每个 xyz 分量沿时间轴做 DCT → coeffs[30,3]
           只保留前 8 个系数: k=0,1 → 256 bin; k=2..7 → 64 bin
           (低频 k=0: 均值/直流; k=1: 基频/整体运动方向)
        3) 旋转: 逐帧四元数 → 欧拉角, 均匀分桶 360 bin (30 帧 × 3 维 = 90 tokens)
        4) 每个 (dim, freq) / (frame, euler_dim) 的 bin_idx → 对应 token 字符串 → ID
    """

    # 平移 DCT 系数: 低频 256 bin (稠密), 高频 64 bin (稀疏); 旋转(欧拉角)均匀 360 bin
    LOW_FREQ_BINS = 256
    HIGH_FREQ_BINS = 64
    ROT_BINS = 360

    def __init__(
        self,
        bin_stats_path: str | None = None,
        base_vocab_size: int = BASE_VOCAB_SIZE,
        # DCT 窗口长度 (时间帧数). 与训练时 dense state 帧数一致。
        dct_window_size: int = 4,
        # 低频系数数量 (前 n_low_freq 个系数用稠密桶, 其余用稀疏桶)。
        # None 表示用一半 (ceil(window/2))。
        n_low_freq: int | None = None,
        # 平移维度保留的 DCT 系数个数 (前 n_keep_coeffs 个, 丢弃更高频).
        n_keep_coeffs: int = 8,
        # 低频/高频桶数 (与 bin_stats pkl 一致).
        low_freq_bins: int = LOW_FREQ_BINS,
        high_freq_bins: int = HIGH_FREQ_BINS,
        state_dim: int = 7,
        # 训练数据中 state 经过 Normalize (z-score), 编码前需反归一化恢复原始位姿。
        state_norm_mean: np.ndarray | None = None,
        state_norm_std: np.ndarray | None = None,
        state_norm_q01: np.ndarray | None = None,
        state_norm_q99: np.ndarray | None = None,
    ):
        """
        Args:
            bin_stats_path: 预计算的 DCT 系数分桶边界 pkl, 包含:
                              - percentiles_low_xy:  (low_bins+1,) 低频 XY 系数边界
                              - percentiles_low_z:   (low_bins+1,) 低频 Z  系数边界
                              - percentiles_high_xy: (high_bins+1,) 高频 XY 系数边界
                              - percentiles_high_z:  (high_bins+1,) 高频 Z  系数边界
                           旋转 (欧拉角) 用均匀 360 bin, 无需统计.
            base_vocab_size: PaliGemma 原始词表大小, pose token ID 从此值开始。
            dct_window_size: DCT 时间窗口长度 (帧数), 需与 dense state 帧数一致。
            n_low_freq:      平移低频系数个数 (用稠密桶); 其余为高频 (用稀疏桶)。
            n_keep_coeffs:   平移保留的 DCT 系数总数 (前 n_keep_coeffs 个).
            low_freq_bins:   低频桶数 (默认 256).
            high_freq_bins:  高频桶数 (默认 64).
            state_dim:       state 维度 (默认 7 = xyz + quat_xyzw)。
            state_norm_*:    归一化统计量, 用于编码前反归一化。
        """
        self._base_vocab_size = base_vocab_size
        self._window = int(dct_window_size)
        self._state_dim = int(state_dim)
        self._n_keep_coeffs = int(n_keep_coeffs)
        self._low_bins = int(low_freq_bins)
        self._high_bins = int(high_freq_bins)
        if n_low_freq is None:
            n_low_freq = max(1, (self._window + 1) // 2)
        # 低频系数数量不能超过保留的系数总数
        self._n_low_freq = int(min(n_low_freq, self._n_keep_coeffs))
        self._n_high_freq = self._n_keep_coeffs - self._n_low_freq

        self._state_mean = np.asarray(state_norm_mean, dtype=np.float64) if state_norm_mean is not None else None
        self._state_std = np.asarray(state_norm_std, dtype=np.float64) if state_norm_std is not None else None
        self._state_q01 = np.asarray(state_norm_q01, dtype=np.float64) if state_norm_q01 is not None else None
        self._state_q99 = np.asarray(state_norm_q99, dtype=np.float64) if state_norm_q99 is not None else None

        # --- 生成 token 字符串列表 ---
        # 平移: 低频 256 bin (稠密), 高频 64 bin (稀疏)
        self.LOW_XY_TOKENS = self._gen_tokens("dctXYlow", self._low_bins)
        self.LOW_Z_TOKENS = self._gen_tokens("dctZlow", self._low_bins)
        self.HIGH_XY_TOKENS = self._gen_tokens("dctXYhigh", self._high_bins)
        self.HIGH_Z_TOKENS = self._gen_tokens("dctZhigh", self._high_bins)
        # 旋转: 欧拉角均匀 360 bin (与 PoseStateTokenizer 一致)
        self.ROT_TOKENS = self._gen_tokens("rot", self.ROT_BINS)

        # --- 加载平移 DCT 系数分桶边界 ---
        if bin_stats_path and os.path.exists(bin_stats_path):
            with open(bin_stats_path, "rb") as f:
                data = pickle.load(f)
            self.pct_low_xy = np.asarray(data["percentiles_low_xy"], dtype=np.float64)
            self.pct_low_z = np.asarray(data["percentiles_low_z"], dtype=np.float64)
            self.pct_high_xy = np.asarray(data["percentiles_high_xy"], dtype=np.float64)
            self.pct_high_z = np.asarray(data["percentiles_high_z"], dtype=np.float64)
            logger.info(
                "Loaded DCT bin stats from %s (low_xy:%d low_z:%d high_xy:%d high_z:%d bins, "
                "window=%d, n_low=%d, n_keep=%d)",
                bin_stats_path,
                len(self.pct_low_xy) - 1,
                len(self.pct_low_z) - 1,
                len(self.pct_high_xy) - 1,
                len(self.pct_high_z) - 1,
                self._window,
                self._n_low_freq,
                self._n_keep_coeffs,
            )
        else:
            logger.warning("DCT bin stats not found at %s, falling back to uniform bins [-1, 1].", bin_stats_path)
            self.pct_low_xy = np.linspace(-1, 1, self._low_bins + 1, dtype=np.float64)
            self.pct_low_z = np.linspace(-1, 1, self._low_bins + 1, dtype=np.float64)
            self.pct_high_xy = np.linspace(-1, 1, self._high_bins + 1, dtype=np.float64)
            self.pct_high_z = np.linspace(-1, 1, self._high_bins + 1, dtype=np.float64)

        # --- 构建 token → ID 映射 (ID 从 base_vocab_size 开始) ---
        all_tokens = self.all_special_tokens
        self._token_to_id = {tok: base_vocab_size + i for i, tok in enumerate(all_tokens)}
        self._vocab_size = base_vocab_size + len(all_tokens)

    @staticmethod
    def _gen_tokens(prefix: str, num: int) -> list[str]:
        """生成 token 字符串列表: <prefix0000>, <prefix0001>, ..."""
        width = len(str(num - 1))
        return [f"<{prefix}{str(i).zfill(width)}>" for i in range(num)]

    # ===== 编码核心 =====

    def _denormalize(self, state: np.ndarray) -> np.ndarray:
        """反归一化: 与 PoseStateTokenizer 一致的 z-score / quantile 反归一化。

        支持两种统计量布局:
          - per-frame 统计量 (T*D,): 训练时对堆叠历史帧逐帧统计, 逐帧切片应用
          - 单帧统计量 (D,): 普通归一化统计量, 广播应用到每一帧
        统计量长度不足 D 时跳过该帧的反归一化。

        Args:
            state: (T, D) 已归一化的 state
        Returns:
            (T, D) 原始位姿
        """
        T, D = state.shape

        def _denorm_frame(t: int, params: np.ndarray) -> np.ndarray | None:
            n = params.shape[0]
            if n >= T * D:
                # per-frame 统计量: 逐帧切片
                return params[t * D : t * D + D]
            if n >= D:
                # 单帧统计量: 广播到所有帧
                return params[:D]
            return None  # 统计量不足, 跳过

        # --- z-score 反归一化 ---
        if self._state_mean is not None and self._state_std is not None:
            mean = np.asarray(self._state_mean, dtype=np.float64).reshape(-1)
            std = np.asarray(self._state_std, dtype=np.float64).reshape(-1)
            out = state.copy()
            for t in range(T):
                m = _denorm_frame(t, mean)
                s = _denorm_frame(t, std)
                if m is not None and s is not None and np.all(np.abs(s) > 1e-12):
                    out[t] = state[t] * (s + 1e-6) + m
            return out
        # --- quantile 反归一化 ---
        if self._state_q01 is not None and self._state_q99 is not None:
            q01 = np.asarray(self._state_q01, dtype=np.float64).reshape(-1)
            q99 = np.asarray(self._state_q99, dtype=np.float64).reshape(-1)
            out = state.copy()
            for t in range(T):
                lo = _denorm_frame(t, q01)
                hi = _denorm_frame(t, q99)
                if lo is not None and hi is not None and np.all(np.abs(hi - lo) > 1e-12):
                    out[t] = (state[t] + 1.0) / 2.0 * (hi - lo + 1e-6) + lo
            return out
        return state

    def _encode_coeff(self, coeff: float, percentiles: np.ndarray, token_list: list[str]) -> str:
        """单个 DCT 系数 → token 字符串 (digitize 分桶)"""
        bin_idx = int(np.digitize(coeff, percentiles, right=False)) - 1
        bin_idx = int(np.clip(bin_idx, 0, len(token_list) - 1))
        return token_list[bin_idx]

    @staticmethod
    def _quaternion_to_euler(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
        """四元数 [w, x, y, z] → 欧拉角 [roll, pitch, yaw] (弧度, 范围 [-π, π])"""
        roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
        pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return np.array([roll, pitch, yaw], dtype=np.float64)

    def encode_state(self, state: np.ndarray, pose_split: dict | None = None) -> str:
        """将 state 时间序列编码为 pose token 字符串。

        - 平移 (xyz): 沿时间轴做 DCT, 只保留前 n_keep_coeffs 个系数。
          前 n_low_freq 个 (k=0,1: 均值/基频) → 低频稠密桶;
          其余 (k=2..n_keep_coeffs-1: 高频细节) → 高频稀疏桶。
        - 旋转 (四元数): 不走 DCT, 逐帧转欧拉角后均匀分桶 360 bin
          (与 PoseStateTokenizer.rot 一致)。

        Args:
            state: (T, D) 或 (T*D,) 或 (>=T*D,) 历史帧堆叠向量。
                   - (T, D): T 帧 × D 维, 按时间从旧到新排列
                   - (>=T*D,): 一维堆叠 (可能含 padding), 会取前 T*D 个值按 D 重塑为 (T, D),
                            frame-major 顺序 [frame0_d0, ..., frame0_d(D-1), frame1_d0, ...]。
                   若实际 T < window, 会在时间轴首部重复首帧补齐; 若 T > window, 取最后 window 帧。
            pose_split: 维度划分, 仅用其中的索引决定 XY/Z/Q 分组。
                        若为 None, 使用 DEFAULT_POSE_SPLIT (xyzw 顺序)。

        Returns:
            token_str: 拼接的 token 字符串, 形如
                       "<dctXYlow0123><dctZlow0456><dctXYhigh007><dctZhigh063>...<rot090><rot180>..."
            token 顺序: 平移低频 (XY→Z) → 平移高频 (XY→Z) → 旋转 (逐帧 roll/pitch/yaw)。
        """
        if pose_split is None:
            pose_split = DEFAULT_POSE_SPLIT

        state = np.asarray(state, dtype=np.float64)
        D = self._state_dim
        # 统一重塑为 (T, D), 处理可能的 padding
        if state.ndim == 1:
            total = state.shape[0]
            if total < D:
                raise ValueError(f"1D state length {total} < state_dim {D}")
            # 只取前 floor(total/D)*D 个值 (丢弃尾部 padding), 再重塑
            usable = (total // D) * D
            T = usable // D
            state = state[:usable].reshape(T, D)
        elif state.ndim == 2:
            D = state.shape[1]
        else:
            raise ValueError(f"state must be 1D or 2D, got {state.ndim}D")

        T, D = state.shape
        # 对齐窗口长度
        if T < self._window:
            pad = np.repeat(state[:1], self._window - T, axis=0)
            state = np.concatenate([pad, state], axis=0)
        elif T > self._window:
            state = state[-self._window :]
        T = self._window

        # 反归一化 (按维度)
        state = self._denormalize(state)

        # 维度分组索引
        trans_idx = pose_split.get("trans_xyz", [0, 1, 2])
        quat_xyzw = pose_split.get("quat_xyzw", [])
        quat_wxyz = pose_split.get("quat_wxyz", [])
        xy_idx = trans_idx[:2] if len(trans_idx) >= 2 else []
        z_idx = [trans_idx[2]] if len(trans_idx) >= 3 else []
        if len(quat_xyzw) >= 4:
            q_idx = list(quat_xyzw)
            quat_order = "xyzw"
        elif len(quat_wxyz) >= 4:
            q_idx = list(quat_wxyz)
            quat_order = "wxyz"
        else:
            q_idx = []
            quat_order = None

        parts: list[str] = []

        # ===== 平移: 时间轴 DCT (type-II 正变换, orthogonally normalized) =====
        # 只保留前 n_keep_coeffs 个系数 (更低频), 丢弃更高频细节。
        trans_dim = len(trans_idx)
        if trans_dim > 0:
            trans_state = state[:, [i for i in trans_idx if i < D]]  # (T, n_trans)
            coeffs = dct(trans_state, type=2, norm="ortho", axis=0)  # (T, n_trans)
            # 低频系数 (k=0..n_low_freq-1): 256-bin 稠密桶
            xy_idx_valid = [i for i, _ in enumerate(xy_idx) if i < trans_state.shape[1]]
            z_idx_valid = [i for i, _ in enumerate(z_idx) if i < trans_state.shape[1]]
            for grp_idx, pct, tokens in (
                (xy_idx_valid, self.pct_low_xy, self.LOW_XY_TOKENS),
                (z_idx_valid, self.pct_low_z, self.LOW_Z_TOKENS),
            ):
                for d in grp_idx:
                    for t in range(self._n_low_freq):
                        parts.append(self._encode_coeff(coeffs[t, d], pct, tokens))
            # 高频系数 (k=n_low_freq..n_keep_coeffs-1): 64-bin 稀疏桶
            if self._n_high_freq > 0:
                for grp_idx, pct, tokens in (
                    (xy_idx_valid, self.pct_high_xy, self.HIGH_XY_TOKENS),
                    (z_idx_valid, self.pct_high_z, self.HIGH_Z_TOKENS),
                ):
                    for d in grp_idx:
                        for t in range(self._n_low_freq, self._n_keep_coeffs):
                            parts.append(self._encode_coeff(coeffs[t, d], pct, tokens))

        # ===== 旋转: 四元数 → 欧拉角 (逐帧), 均匀分桶 360 bin (时域, 不做 DCT) =====
        # 四元数有单位约束/双覆盖/非线性叠加, 不适合频域分析, 退回时域分桶,
        # 与 PoseStateTokenizer.rot 完全一致。
        if len(q_idx) >= 4:
            # 统一成 (qw, qx, qy, qz) 索引, 兼容 xyzw / wxyz 两种顺序
            if quat_order == "wxyz":
                iw, ix, iy, iz = (q_idx[0], q_idx[1], q_idx[2], q_idx[3])
            else:  # xyzw
                ix, iy, iz, iw = (q_idx[0], q_idx[1], q_idx[2], q_idx[3])
            for t in range(T):
                qw, qx, qy, qz = state[t, iw], state[t, ix], state[t, iy], state[t, iz]
                euler = self._quaternion_to_euler(qw, qx, qy, qz)
                rot_bins = ((euler + np.pi) / (2 * np.pi) * (self.ROT_BINS - 1)).astype(int)
                rot_bins = np.clip(rot_bins, 0, self.ROT_BINS - 1)
                for b in rot_bins:
                    parts.append(self.ROT_TOKENS[int(b)])

        return "".join(parts)

    def encode_to_ids(self, token_str: str) -> list[int]:
        """将 pose token 字符串编码为 ID 序列 (与 PoseStateTokenizer 相同的解析逻辑)。"""
        ids = []
        i = 0
        while i < len(token_str):
            if token_str[i] == "<":
                end = token_str.find(">", i)
                if end == -1:
                    break
                token = token_str[i : end + 1]
                if token in self._token_to_id:
                    ids.append(self._token_to_id[token])
                    i = end + 1
                    continue
            i += 1
        return ids

    # ===== 属性 =====

    @property
    def all_special_tokens(self) -> list[str]:
        """返回所有需要注册的特殊 token"""
        return (
            self.LOW_XY_TOKENS
            + self.LOW_Z_TOKENS
            + self.HIGH_XY_TOKENS
            + self.HIGH_Z_TOKENS
            + self.ROT_TOKENS
        )

    @property
    def num_special_tokens(self) -> int:
        """新增 token 数量"""
        return len(self.all_special_tokens)

    @property
    def extended_vocab_size(self) -> int:
        """扩展后的词表大小"""
        return self._vocab_size

    @property
    def token_to_id(self) -> dict[str, int]:
        """token → ID 映射 (只读)"""
        return dict(self._token_to_id)
