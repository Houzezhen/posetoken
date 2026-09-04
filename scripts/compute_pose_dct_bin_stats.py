"""统计 Pose Token 平移 DCT 频域系数的分桶边界 (pose_token_dct 分支)。

对每个 episode 的 state 时间序列中**平移维度 (xyz)** 在时间轴做 DCT, 得到
DCT 系数, 只保留前 n_keep_coeffs 个系数 (丢弃更高频), 然后按
"低频/高频 × 维度组 (XY/Z)" 分别计算百分位分桶边界, 生成 bin_stats_dct.pkl。

旋转 (四元数) 不做 DCT —— 退回时域: 逐帧转欧拉角后均匀分桶 360 bin
(与 PoseStateTokenizer.rot 一致), 无需统计分桶边界。

分桶结构 (与 PoseDCTTokenizer 对齐):
  - 低频 (前 n_low_freq 个系数): XY/Z 各 256 bin (稠密)
  - 高频 (第 n_low_freq..n_keep_coeffs 个系数): XY/Z 各 64 bin (稀疏)

DCT 用 scipy.fft.dct(type=2, norm="ortho"), 与 PoseDCTTokenizer 一致。

用法:
    python scripts/compute_pose_dct_bin_stats.py \
        --data-dir=./data --output=./bin_stats_dct.pkl \
        --window 30 --n-low-freq 2 --n-keep-coeffs 8 \
        --low-bins 256 --high-bins 64
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
    window: int = 30,
    n_low_freq: int | None = None,
    # 平移维度保留的 DCT 系数个数 (只统计前 n_keep_coeffs 个系数).
    n_keep_coeffs: int = 8,
    # === 桶数 (与 PoseDCTTokenizer 对齐) ===
    low_bins: int = 256,
    high_bins: int = 64,
    # === 滑窗步长 ===
    stride: int = 1,
):
    """统计平移 DCT 系数的分桶边界。

    Args:
        data_dir: 存放 episode_*.parquet 的目录
        output: 输出 pkl 路径
        state_col: parquet 中 state 列名
        pose_split: JSON 格式的 state 结构描述
        window: DCT 时间窗口长度 (帧数)
        n_low_freq: 低频系数个数 (None → ceil(window/2))
        n_keep_coeffs: 平移保留的 DCT 系数总数 (只统计前 n_keep_coeffs 个)
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

    if n_low_freq is None:
        n_low_freq = max(1, (window + 1) // 2)
    n_low_freq = int(min(n_low_freq, n_keep_coeffs))
    n_high_freq = n_keep_coeffs - n_low_freq
    print(f"DCT window={window}, n_low_freq={n_low_freq}, n_keep_coeffs={n_keep_coeffs}, "
          f"n_high_freq={n_high_freq}")

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

    # ============ 滑窗 + DCT, 收集平移系数 ============
    # 只统计平移维度 (xy/z); 旋转 (四元数) 退回时域 360 bin, 无需统计。
    low_xy, low_z = [], []
    high_xy, high_z = [], []
    n_windows = 0

    for ep in tqdm.tqdm(episodes, desc="DCT per episode"):
        T = len(ep)
        if T < window:
            # 不足一个窗口: 首帧补齐到 window
            pad = np.repeat(ep[:1], window - T, axis=0)
            ep = np.concatenate([pad, ep], axis=0)
            T = window

        # 只取平移维度
        trans = ep[:, trans_idx]  # (T, 3)

        for start in range(0, T - window + 1, stride):
            chunk = trans[start : start + window]  # (window, 3)
            coeffs = dct(chunk, type=2, norm="ortho", axis=0)  # (window, 3)
            # 只保留前 n_keep_coeffs 个系数
            keep = coeffs[:n_keep_coeffs]

            # 低频系数 (前 n_low_freq 行)
            for d in range(2):  # xy
                low_xy.append(keep[:n_low_freq, d].ravel())
            low_z.append(keep[:n_low_freq, 2].ravel())

            # 高频系数 (第 n_low_freq..n_keep_coeffs 行)
            if n_high_freq > 0:
                high_block = keep[n_low_freq:n_keep_coeffs]
                for d in range(2):
                    high_xy.append(high_block[:, d].ravel())
                high_z.append(high_block[:, 2].ravel())

            n_windows += 1

    print(f"Total DCT windows: {n_windows}")

    low_xy_vals = np.concatenate(low_xy) if low_xy else np.array([], dtype=np.float64)
    low_z_vals = np.concatenate(low_z) if low_z else np.array([], dtype=np.float64)
    high_xy_vals = np.concatenate(high_xy) if high_xy else np.array([], dtype=np.float64)
    high_z_vals = np.concatenate(high_z) if high_z else np.array([], dtype=np.float64)

    def _log(name, vals):
        if len(vals):
            print(f"{name}: n={len(vals)}, range=[{vals.min():.4f}, {vals.max():.4f}], "
                  f"mean={vals.mean():.4f}, std={vals.std():.4f}")
        else:
            print(f"{name}: n=0 (empty)")

    _log("low_xy", low_xy_vals)
    _log("low_z", low_z_vals)
    _log("high_xy", high_xy_vals)
    _log("high_z", high_z_vals)

    # ============ 计算分桶边界 ============
    pct_low_xy = compute_percentiles(low_xy_vals, low_bins)
    pct_low_z = compute_percentiles(low_z_vals, low_bins)
    pct_high_xy = compute_percentiles(high_xy_vals, high_bins)
    pct_high_z = compute_percentiles(high_z_vals, high_bins)

    print(f"low_xy:  {len(pct_low_xy)-1} bins, range [{pct_low_xy[0]:.4f}, {pct_low_xy[-1]:.4f}]")
    print(f"low_z:   {len(pct_low_z)-1} bins, range [{pct_low_z[0]:.4f}, {pct_low_z[-1]:.4f}]")
    print(f"high_xy: {len(pct_high_xy)-1} bins, range [{pct_high_xy[0]:.4f}, {pct_high_xy[-1]:.4f}]")
    print(f"high_z:  {len(pct_high_z)-1} bins, range [{pct_high_z[0]:.4f}, {pct_high_z[-1]:.4f}]")

    # ============ 保存 (与 PoseDCTTokenizer 对齐) ============
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "percentiles_low_xy": pct_low_xy,
        "percentiles_low_z": pct_low_z,
        "percentiles_high_xy": pct_high_xy,
        "percentiles_high_z": pct_high_z,
        "pose_split": pose_split_dict,
        "window": window,
        "n_low_freq": n_low_freq,
        "n_keep_coeffs": n_keep_coeffs,
        "low_bins": low_bins,
        "high_bins": high_bins,
    }
    with open(output_path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"Saved DCT bin stats to: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
