"""PoseVLA 风格的 State 离散化编码器。

将机器人状态向量 [x, y, z, qw, qx, qy, qz] 编码为语义化 token：
  - 平移 (x, y): 百分位非均匀分桶 (2048 bins) → <transXY...>
  - 平移 (z):      百分位非均匀分桶 (2048 bins) → <transZ...>
  - 旋转 (quaternion → euler): 均匀分桶 (360 bins, 1° 精度) → <rot...>

由于 SentencePiece 不支持运行时添加 token，pose token 被映射为
> base_vocab_size 的整数 ID，直接绕过 SentencePiece 编码。
"""

import logging
import os
import pickle

import numpy as np

logger = logging.getLogger("openpi")

# PaliGemma 原始词表大小
BASE_VOCAB_SIZE = 257152

# 默认 state 结构: [x, y, z, qw, qx, qy, qz]
DEFAULT_POSE_SPLIT = {
    "trans_xyz": [0, 1, 2],
    "quat_wxyz": [3, 4, 5, 6],
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

    def __init__(self, bin_stats_path: str | None = None, base_vocab_size: int = BASE_VOCAB_SIZE):
        """
        Args:
            bin_stats_path: 预计算的分桶边界 pkl 文件路径。
                            包含:
                              - percentiles_txy: (2049,) 平移 XY 分桶边界
                              - percentiles_tz:  (2049,) 平移 Z 分桶边界
            base_vocab_size: PaliGemma 原始词表大小，pose token ID 从此值开始。
        """
        self._base_vocab_size = base_vocab_size

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
        parts = []

        # --- 平移维度 ---
        trans_idx = pose_split.get("trans_xyz", [])
        if len(trans_idx) >= 3:
            tx, ty, tz = state[trans_idx[0]], state[trans_idx[1]], state[trans_idx[2]]
            parts.append(self._encode_value(tx, self.percentiles_txy, self.TRANS_XY_TOKENS))
            parts.append(self._encode_value(ty, self.percentiles_txy, self.TRANS_XY_TOKENS))
            parts.append(self._encode_value(tz, self.percentiles_tz, self.TRANS_Z_TOKENS))

        # --- 旋转维度: 四元数 → 欧拉角 → 360 bin ---
        quat_idx = pose_split.get("quat_wxyz", [])
        if len(quat_idx) >= 4:
            qw, qx, qy, qz = state[quat_idx[0]], state[quat_idx[1]], state[quat_idx[2]], state[quat_idx[3]]
            euler = self._quaternion_to_euler(qw, qx, qy, qz)
            # [-π, π] → [0, 359]
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
