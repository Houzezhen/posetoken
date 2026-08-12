"""统计 Pose Token DCT 频域系数的分桶边界 (pose_token_dct 分支)。

对每个 episode 的 state 时间序列在时间轴做 DCT, 得到 DCT 系数, 然后按
"低频/高频 × 维度组 (XY/Z/Q)" 分别计算百分位分桶边界, 生成 bin_stats_dct.pkl。

分桶结构 (与 PoseDCTTokenizer 对齐):
  - 低频 (前 n_low_freq 个系数): XY/Z/Q 各 2048 bin (稠密)
  - 高频 (剩余系数):             XY/Z/Q 各 360  bin (稀疏)
即保留原 PoseStateTokenizer 的 "稠密(2048) + 稀疏(360)" 双分布, 但作用在频域。

DCT 用 scipy.fft.dct(type=2, norm="ortho"), 与 PoseDCTTokenizer 一致。

用法:
    python scripts/compute_pose_dct_bin_stats.py \
        --data-dir=./data --output=./bin_stats_dct.pkl \
        --window 20 --n-low-freq 2
"""
import glob
import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import tqdm
import tyro
from scipy.fft import dct


def compute_percentiles(values: np.ndarray, n_bins: int) -> np.ndarray:
    """对一组值计算 n_bins+1 个百分位边界。

    Args:
        values: (N,) 一维数值数组
        n_bins: 桶数

    Returns:
        (n_bins+1,) 分桶边界, 单调递增
    """
    if len(values) == 0:
        return np.linspace(-1, 1, n_bins + 1, dtype=np.float32)
    p = np.percentile(values, np.linspace(0, 100, n_bins + 1))
    for i in range(1, len(p)):
        if p[i] <= p[i - 1]:
            p[i] = p[i - 1] + 1e-10
    return p.astype(np.float32)


def main(
    data_dir: str = "./data",
    output: str = "./bin_stats_dct.pkl",
    state_col: str = "observation.state",
    pose_split: str = '{"trans_xyz": [0, 1, 2], "quat_xyzw": [3, 4, 5, 6], "other": []}',
    # === DCT 参数 ===
    window: int = 20,
    n_low_freq: int | None = None,
    # === 桶数 (与 PoseDCTTokenizer 对齐) ===
    low_bins: int = 2048,
    high_bins: int = 360,
    # === 滑窗步长 ===
    stride: int = 1,
):
    """统计 DCT 系数的分桶边界。

    Args:
        data_dir: 存放 episode_*.parquet 的目录
        output: 输出 pkl 路径
        state_col: parquet 中 state 列名
        pose_split: JSON 格式的 state 结构描述
        window: DCT 时间窗口长度 (帧数)
        n_low_freq: 低频系数个数 (None → ceil(window/2))
        low_bins: 低频系数桶数 (稠密)
        high_bins: 高频系数桶数 (稀疏)
        stride: 滑窗步长 (1 = 每帧都采样一个窗口)
    """
    pose_split_dict = json.loads(pose_split)
    print(f"Pose split: {pose_split_dict}")

    trans_idx = pose_split_dict.get("trans_xyz", [0, 1, 2])
    assert len(trans_idx) >= 3, "trans_xyz 必须是 3 个索引"
    xy_idx = trans_idx[:2]
    z_idx = [trans_idx[2]]
    quat_xyzw = pose_split_dict.get("quat_xyzw", [])
    quat_wxyz = pose_split_dict.get("quat_wxyz", [])
    if len(quat_xyzw) >= 4:
        q_idx = list(quat_xyzw)
        quat_order = "xyzw"
    elif len(quat_wxyz) >= 4:
        q_idx = list(quat_wxyz)
        quat_order = "wxyz"
    else:
        q_idx = []
        quat_order = None

    # --- 旋转分组: 四元数 → 欧拉角 [roll, pitch, yaw] (与 PoseDCTTokenizer 一致) ---
    # 预计算转换后的维度索引: 平移维度保持不变, 欧拉角 3 维接在后面
    if len(q_idx) >= 4:
        if quat_order == "wxyz":
            iw, ix, iy, iz = (q_idx[0], q_idx[1], q_idx[2], q_idx[3])
        else:  # xyzw
            ix, iy, iz, iw = (q_idx[0], q_idx[1], q_idx[2], q_idx[3])
        # DCT 前会把每帧的 quat(4) 转成 euler(3), 平移维度保持原位
        keep_idx = [i for i in range(len(trans_idx) + len(q_idx)) if i not in q_idx]
        n_xy, n_z = len(xy_idx), len(z_idx)
        xy_dct_idx = list(range(n_xy))
        z_dct_idx = [n_xy] if n_z else []
        rpy_dct_idx = list(range(n_xy + n_z, n_xy + n_z + 3))
        print(f"Quaternion → Euler: {q_idx} ({quat_order}) → roll/pitch/yaw, "
              f"keep_idx={keep_idx}, xy={xy_dct_idx}, z={z_dct_idx}, rpy={rpy_dct_idx}")
    else:
        iw = ix = iy = iz = None
        xy_dct_idx = list(xy_idx)
        z_dct_idx = list(z_idx)
        rpy_dct_idx = []

    if n_low_freq is None:
        n_low_freq = max(1, (window + 1) // 2)
    n_low_freq = int(min(n_low_freq, window))
    n_high_freq = window - n_low_freq
    print(f"DCT window={window}, n_low_freq={n_low_freq}, n_high_freq={n_high_freq}")

    # ============ 读取所有 episode ============
    # 支持平铺和 chunk-XXX/ 子目录两种布局 (lerobot 格式)
    data_p = pathlib.Path(data_dir)
    files = sorted(data_p.glob("**/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到 episode_*.parquet")
    print(f"Found {len(files)} episodes")

    episodes = []
    for f in tqdm.tqdm(files, desc="Loading parquet"):
        df = pd.read_parquet(f)
        s = np.stack(df[state_col].values).astype(np.float64)
        episodes.append(s)
    print(f"Total episodes: {len(episodes)}, state dim D = {episodes[0].shape[1]}")

    # ============ 滑窗 + DCT, 收集系数 ============
    # 分别收集: 低频/高频 × XY/Z/Q
    low_xy, low_z, low_q = [], [], []
    high_xy, high_z, high_q = [], [], []
    n_windows = 0

    for ep in tqdm.tqdm(episodes, desc="DCT per episode"):
        T = len(ep)
        if T < window:
            # 不足一个窗口: 首帧补齐到 window
            pad = np.repeat(ep[:1], window - T, axis=0)
            ep = np.concatenate([pad, ep], axis=0)
            T = window

        for start in range(0, T - window + 1, stride):
            chunk = ep[start : start + window]  # (window, D)

            # 四元数 → 欧拉角 (逐帧), 与 PoseDCTTokenizer.encode_state 一致
            if iw is not None:
                qw, qx, qy, qz = chunk[:, iw], chunk[:, ix], chunk[:, iy], chunk[:, iz]
                roll = np.arctan2(2 * (qw * qx + qy * qz), 1 - 2 * (qx * qx + qy * qy))
                pitch = np.arcsin(np.clip(2 * (qw * qy - qz * qx), -1.0, 1.0))
                yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                euler = np.stack([roll, pitch, yaw], axis=-1)  # (window, 3)
                # 重构: 保留非四元数维度 + 欧拉角
                chunk = np.concatenate([chunk[:, keep_idx], euler], axis=-1)

            coeffs = dct(chunk, type=2, norm="ortho", axis=0)  # (window, D')

            # 低频系数 (前 n_low_freq 行)
            for d in xy_dct_idx:
                low_xy.append(coeffs[:n_low_freq, d].ravel())
            for d in z_dct_idx:
                low_z.append(coeffs[:n_low_freq, d].ravel())
            for d in rpy_dct_idx:
                low_q.append(coeffs[:n_low_freq, d].ravel())

            # 高频系数 (第 n_low_freq..window 行)
            if n_high_freq > 0:
                high_block = coeffs[n_low_freq:window]
                for d in xy_dct_idx:
                    high_xy.append(high_block[:, d].ravel())
                for d in z_dct_idx:
                    high_z.append(high_block[:, d].ravel())
                for d in rpy_dct_idx:
                    high_q.append(high_block[:, d].ravel())

            n_windows += 1

    print(f"Total DCT windows: {n_windows}")

    low_xy_vals = np.concatenate(low_xy) if low_xy else np.array([], dtype=np.float64)
    low_z_vals = np.concatenate(low_z) if low_z else np.array([], dtype=np.float64)
    low_q_vals = np.concatenate(low_q) if low_q else np.array([], dtype=np.float64)
    high_xy_vals = np.concatenate(high_xy) if high_xy else np.array([], dtype=np.float64)
    high_z_vals = np.concatenate(high_z) if high_z else np.array([], dtype=np.float64)
    high_q_vals = np.concatenate(high_q) if high_q else np.array([], dtype=np.float64)

    def _log(name, vals):
        if len(vals):
            print(f"{name}: n={len(vals)}, range=[{vals.min():.4f}, {vals.max():.4f}], "
                  f"mean={vals.mean():.4f}, std={vals.std():.4f}")
        else:
            print(f"{name}: n=0 (empty)")

    _log("low_xy", low_xy_vals)
    _log("low_z", low_z_vals)
    _log("low_q", low_q_vals)
    _log("high_xy", high_xy_vals)
    _log("high_z", high_z_vals)
    _log("high_q", high_q_vals)

    # ============ 计算分桶边界 ============
    pct_low_xy = compute_percentiles(low_xy_vals, low_bins)
    pct_low_z = compute_percentiles(low_z_vals, low_bins)
    pct_low_q = compute_percentiles(low_q_vals, low_bins)
    pct_high_xy = compute_percentiles(high_xy_vals, high_bins)
    pct_high_z = compute_percentiles(high_z_vals, high_bins)
    pct_high_q = compute_percentiles(high_q_vals, high_bins)

    print(f"low_xy:  {len(pct_low_xy)-1} bins, range [{pct_low_xy[0]:.4f}, {pct_low_xy[-1]:.4f}]")
    print(f"low_z:   {len(pct_low_z)-1} bins, range [{pct_low_z[0]:.4f}, {pct_low_z[-1]:.4f}]")
    print(f"low_q:   {len(pct_low_q)-1} bins, range [{pct_low_q[0]:.4f}, {pct_low_q[-1]:.4f}]")
    print(f"high_xy: {len(pct_high_xy)-1} bins, range [{pct_high_xy[0]:.4f}, {pct_high_xy[-1]:.4f}]")
    print(f"high_z:  {len(pct_high_z)-1} bins, range [{pct_high_z[0]:.4f}, {pct_high_z[-1]:.4f}]")
    print(f"high_q:  {len(pct_high_q)-1} bins, range [{pct_high_q[0]:.4f}, {pct_high_q[-1]:.4f}]")

    # ============ 保存 (与 PoseDCTTokenizer 对齐) ============
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "percentiles_low_xy": pct_low_xy,
        "percentiles_low_z": pct_low_z,
        "percentiles_low_q": pct_low_q,
        "percentiles_high_xy": pct_high_xy,
        "percentiles_high_z": pct_high_z,
        "percentiles_high_q": pct_high_q,
        "pose_split": pose_split_dict,
        "window": window,
        "n_low_freq": n_low_freq,
        "low_bins": low_bins,
        "high_bins": high_bins,
    }
    with open(output_path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"Saved DCT bin stats to: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
