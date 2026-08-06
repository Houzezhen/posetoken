"""从训练数据统计 Pose Token 的非均匀分桶边界 (平移 xyz)。

读取 data 目录下的 parquet 文件 (observation.state = [x, y, z, qx, qy, qz, qw])，
为平移维度计算非均匀分桶边界，生成 bin_stats.pkl，包含:
  - percentiles_txy: (2049,) X/Y 合并的平移分桶边界 (共用)
  - percentiles_tz:  (2049,) Z 平移分桶边界
  - pose_split: 维度划分配置

旋转维度 (四元数) 不参与分桶 —— 由 pose_tokenizer.py 转欧拉角后均匀 360 bin。

分桶策略 (基于"首尾位姿相同"的特性):
  1. 用每帧到 home 位置(episode 首帧)的 xyz 欧氏距离识别停滞段
     - 停滞段: 距 home < home_radius 的帧 (首尾 home 位置附近)
     - 动作中段: 距 home 较远的帧 (含精细操作)
  2. 停滞段只占少量桶 (粗分辨率)
  3. 动作中段占绝大多数桶，且对精细操作(帧间速度低但距 home 远)加权采样
     —— 关键: 精细操作速度低但欧氏距离大，不会被判为停滞

用法:
    python scripts/compute_pose_bin_stats.py --data-dir=./data --output=./bin_stats.pkl
"""
import glob
import json
import pathlib
import pickle

import numpy as np
import pandas as pd
import tqdm
import tyro


def detect_stall_segments(
    states: np.ndarray,
    home_radius: float = 2.0,
) -> np.ndarray:
    """识别一个 episode 中靠近 home 位置的停滞帧。

    用每帧的 xyz 到 home (episode 首帧 xyz) 的欧氏距离判断:
      - 距离 < home_radius → 停滞帧 (首尾 home 位置附近)
      - 距离 >= home_radius → 动作中段 (含远离 home 的精细操作)

    相比"帧间速度"判据,这样能正确区分:
      - 首尾停滞: 速度低 + 距 home 近
      - 中间精细操作: 速度低 + 距 home 远 → 不会被误判为停滞

    Args:
        states: (T, D) 单个 episode 的 state 序列,前 3 维为 xyz
        home_radius: 距 home 的半径阈值,小于此值视为停滞 (与 state 单位一致)

    Returns:
        is_stall: (T,) bool 数组,True=停滞帧,False=动作中段
    """
    T = len(states)
    if T < 2:
        return np.zeros(T, dtype=bool)

    home_xyz = states[0, :3]  # 以首帧 xyz 作为 home 位置
    dist_to_home = np.linalg.norm(states[:, :3] - home_xyz, axis=1)  # (T,)
    return dist_to_home < home_radius


def compute_segmented_percentiles(
    stall_values: np.ndarray,
    mid_values: np.ndarray,
    n_bins: int = 2048,
    stall_bin_ratio: float = 0.1,
) -> np.ndarray:
    """分段计算分桶边界: 停滞段占少数桶，中段占多数桶。

    两段各自用百分位法计算内部边界，然后拼接。

    Args:
        stall_values: 停滞段的数值
        mid_values: 动作中段的数值
        n_bins: 总桶数
        stall_bin_ratio: 分配给停滞段的桶数比例 (0~1)

    Returns:
        (n_bins+1,) 分桶边界
    """
    n_bins = int(n_bins)
    n_stall_bins = max(1, int(n_bins * stall_bin_ratio))
    n_mid_bins = max(1, n_bins - n_stall_bins)

    def _pct(vals: np.ndarray, nb: int) -> np.ndarray:
        if len(vals) == 0:
            return np.array([], dtype=np.float32)
        p = np.percentile(vals, np.linspace(0, 100, nb + 1))
        for i in range(1, len(p)):
            if p[i] <= p[i - 1]:
                p[i] = p[i - 1] + 1e-10
        return p.astype(np.float32)

    p_stall = _pct(stall_values, n_stall_bins)
    p_mid = _pct(mid_values, n_mid_bins)

    if len(p_stall) == 0:
        out = p_mid
        if len(out) < n_bins + 1:
            pad = np.full(n_bins + 1 - len(out), out[-1] if len(out) else 0.0, dtype=np.float32)
            out = np.concatenate([out, pad])
        return out
    if len(p_mid) == 0:
        out = p_stall
        if len(out) < n_bins + 1:
            pad = np.full(n_bins + 1 - len(out), out[-1], dtype=np.float32)
            out = np.concatenate([out, pad])
        return out

    merged = np.concatenate([p_stall, p_mid[1:]])
    if len(merged) < n_bins + 1:
        pad = np.full(n_bins + 1 - len(merged), merged[-1], dtype=np.float32)
        merged = np.concatenate([merged, pad])
    elif len(merged) > n_bins + 1:
        merged = merged[: n_bins + 1]

    for i in range(1, len(merged)):
        if merged[i] <= merged[i - 1]:
            merged[i] = merged[i - 1] + 1e-10
    return merged.astype(np.float32)


def main(
    data_dir: str = "./data",
    output: str = "./bin_stats.pkl",
    state_col: str = "observation.state",
    pose_split: str = '{"trans_xyz": [0, 1, 2], "quat_xyzw": [3, 4, 5, 6], "other": []}',
    # === 停滞检测参数 ===
    home_radius: float = 2.0,
    # === 分桶分配参数 ===
    stall_bin_ratio: float = 0.1,
    n_bins: int = 2048,
    # === 精细操作保护 ===
    fine_op_speed_max: float = 0.5,
    fine_op_weight: int = 2,
):
    """统计平移 xyz 的分桶边界，避开开头/结尾停滞段，密集覆盖动作中段。

    Args:
        data_dir: 存放 episode_*.parquet 的目录
        output: 输出 pkl 文件路径
        state_col: parquet 中 state 列名
        pose_split: JSON 格式的 state 结构描述
        home_radius: 距 home(首帧 xyz) 的欧氏距离阈值,小于此值视为停滞帧 (与 state 单位一致)
        stall_bin_ratio: 停滞段分配的桶数比例 (0~1)，其余给动作中段
        n_bins: 每个维度的总桶数
        fine_op_speed_max: 精细操作速度上限 (帧间速度),中段内低速帧会被加权采样
        fine_op_weight: 精细操作帧的采样权重倍数
    """
    pose_split_dict = json.loads(pose_split)
    print(f"Pose split: {pose_split_dict}")

    trans_idx = pose_split_dict.get("trans_xyz", [0, 1, 2])
    assert len(trans_idx) == 3, "trans_xyz 必须是 3 个索引"
    ix, iy, iz = trans_idx

    # ============ 读取所有 episode ============
    files = sorted(glob.glob(pathlib.Path(data_dir).joinpath("episode_*.parquet").as_posix()))
    if not files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到 episode_*.parquet")
    print(f"Found {len(files)} episodes")

    episodes = []  # list of (T, D)
    for f in tqdm.tqdm(files, desc="Loading parquet"):
        df = pd.read_parquet(f)
        s = np.stack(df[state_col].values).astype(np.float64)  # (T, D)
        episodes.append(s)
    N_total = sum(len(e) for e in episodes)
    print(f"Total frames: {N_total}, state dim D = {episodes[0].shape[1]}")

    # ============ 对每个 episode 做停滞检测，划分 stall / mid / fine_op ============
    # X 和 Y 合并共用 txy 边界；Z 单独
    txy_stall = []
    txy_mid = []
    tz_stall = []
    tz_mid = []
    n_stall_frames = 0
    n_mid_frames = 0
    n_fine_op_boosted = 0

    print("Detecting stall segments per episode...")
    for ep_states in tqdm.tqdm(episodes, desc="Per-episode stall detect"):
        T = len(ep_states)
        if T < 2:
            txy_mid.append(ep_states[:, [ix, iy]])
            tz_mid.append(ep_states[:, iz])
            n_mid_frames += T
            continue

        # 用到 home 的欧氏距离划分停滞/中段
        is_stall = detect_stall_segments(ep_states, home_radius=home_radius)
        stall_states = ep_states[is_stall]
        mid_states = ep_states[~is_stall]
        n_stall_frames += len(stall_states)
        n_mid_frames += len(mid_states)

        # 停滞段
        if len(stall_states) > 0:
            txy_stall.append(stall_states[:, [ix, iy]])
            tz_stall.append(stall_states[:, iz])

        # 中段: 精细操作 (低速但距 home 远) 加权
        if len(mid_states) > 1:
            diffs = np.diff(mid_states, axis=0, prepend=mid_states[:1])
            speeds = np.linalg.norm(diffs, axis=1)
            fine_mask = speeds < fine_op_speed_max
            txy_mid.append(mid_states[:, [ix, iy]])
            tz_mid.append(mid_states[:, iz])
            if fine_mask.any():
                txy_mid.append(np.repeat(mid_states[fine_mask][:, [ix, iy]], fine_op_weight - 1, axis=0))
                tz_mid.append(np.repeat(mid_states[fine_mask, iz], fine_op_weight - 1))
            n_fine_op_boosted += int(fine_mask.sum())
        elif len(mid_states) == 1:
            txy_mid.append(mid_states[:, [ix, iy]])
            tz_mid.append(mid_states[:, iz])

    print(f"Stall frames: {n_stall_frames} ({n_stall_frames/N_total*100:.1f}%)")
    print(f"Mid frames:   {n_mid_frames} ({n_mid_frames/N_total*100:.1f}%)")
    print(f"Fine-op boosted frames (x{fine_op_weight}): {n_fine_op_boosted}")

    # ============ 分段计算分桶边界 ============
    # X、Y 合并 (共用 txy)
    txy_stall_vals = np.concatenate(txy_stall, axis=0).reshape(-1) if txy_stall else np.array([], dtype=np.float64)
    txy_mid_vals = np.concatenate(txy_mid, axis=0).reshape(-1) if txy_mid else np.array([], dtype=np.float64)
    tz_stall_vals = np.concatenate(tz_stall, axis=0) if tz_stall else np.array([], dtype=np.float64)
    tz_mid_vals = np.concatenate(tz_mid, axis=0) if tz_mid else np.array([], dtype=np.float64)

    print(f"txy: stall range [{txy_stall_vals.min():.4f}, {txy_stall_vals.max():.4f}], n={len(txy_stall_vals)}")
    print(f"txy: mid   range [{txy_mid_vals.min():.4f}, {txy_mid_vals.max():.4f}], n={len(txy_mid_vals)}")
    print(f"tz:  stall range [{tz_stall_vals.min():.4f}, {tz_stall_vals.max():.4f}], n={len(tz_stall_vals)}")
    print(f"tz:  mid   range [{tz_mid_vals.min():.4f}, {tz_mid_vals.max():.4f}], n={len(tz_mid_vals)}")

    pct_txy = compute_segmented_percentiles(txy_stall_vals, txy_mid_vals, n_bins=n_bins, stall_bin_ratio=stall_bin_ratio)
    pct_tz = compute_segmented_percentiles(tz_stall_vals, tz_mid_vals, n_bins=n_bins, stall_bin_ratio=stall_bin_ratio)

    print(f"txy: {len(pct_txy)-1} bins, range [{pct_txy[0]:.4f}, {pct_txy[-1]:.4f}]")
    print(f"tz:  {len(pct_tz)-1} bins, range [{pct_tz[0]:.4f}, {pct_tz[-1]:.4f}]")

    # ============ 保存 (与 pose_tokenizer.py 对齐) ============
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "percentiles_txy": pct_txy,
        "percentiles_tz": pct_tz,
        "pose_split": pose_split_dict,
        "n_bins": n_bins,
        "config": {
            "home_radius": home_radius,
            "stall_bin_ratio": stall_bin_ratio,
            "fine_op_speed_max": fine_op_speed_max,
            "fine_op_weight": fine_op_weight,
        },
    }
    with open(output_path, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"Saved bin stats to: {output_path}")


if __name__ == "__main__":
    tyro.cli(main)
