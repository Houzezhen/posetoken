"""统计 Motion Token 的分位数分桶边界 (6D 相对运动)。

motion 定义与 piper_policy._relative_motion 一致:
  [dx, dy, dz, rx, ry, rz] —— 当前帧 vs motion_offset 帧前的位姿差分
  (平移增量 + 相对旋转向量)。Realman 数据集四元数顺序 wxyz。

统计内容 (输出独立 pkl, 不污染 pose bin stats):
  - percentiles_dxyz: (n_trans_bins+1,) dx/dy/dz 三维合并共用的分位边界
  - percentiles_rot:  (n_rot_bins+1,) rx/ry/rz 三维合并共用的分位边界
  - phase_trans_thresholds / phase_rot_thresholds: EMA 速度范数的相位阈值
  - config: 统计参数

关键: 按 episode 边界计算差分 (开头 motion_offset 帧置零), 与推理时
deque 行为对齐, 避免训练/推理分布偏差。

用法:
    python scripts/compute_motion_bin_stats.py \
        --data-dir=/media/Data/user/datasets/realman/pushbutton_0825_merge_cut_camframe/data \
        --output=./bin_stats_motion_pushbutton.pkl --quat-order=wxyz --motion-offset=20
"""
import glob
import pathlib
import pickle

import numpy as np
import pandas as pd
import tqdm
import tyro


def _relative_motion(previous: np.ndarray, current: np.ndarray, quat_order: str = "xyzw") -> np.ndarray:
    """与 piper_policy._relative_motion 完全一致的 6D 差分。"""
    previous = np.ravel(previous)[:7]
    current = np.ravel(current)[:7]
    delta_xyz = current[:3] - previous[:3]
    q0 = np.asarray(previous[3:7], dtype=np.float32)
    q1 = np.asarray(current[3:7], dtype=np.float32)
    if quat_order == "wxyz":
        q0 = q0[[1, 2, 3, 0]]
        q1 = q1[[1, 2, 3, 0]]
    q0 = q0 / max(np.linalg.norm(q0), 1e-8)
    q1 = q1 / max(np.linalg.norm(q1), 1e-8)
    q0_inv = np.array([-q0[0], -q0[1], -q0[2], q0[3]])
    q = np.array([
        q1[3] * q0_inv[0] + q1[0] * q0_inv[3] + q1[1] * q0_inv[2] - q1[2] * q0_inv[1],
        q1[3] * q0_inv[1] - q1[0] * q0_inv[2] + q1[1] * q0_inv[3] + q1[2] * q0_inv[0],
        q1[3] * q0_inv[2] + q1[0] * q0_inv[1] - q1[1] * q0_inv[0] + q1[2] * q0_inv[3],
        q1[3] * q0_inv[3] - q1[0] * q0_inv[0] - q1[1] * q0_inv[1] - q1[2] * q0_inv[2],
    ])
    if q[3] < 0:
        q = -q
    xyz_norm = np.linalg.norm(q[:3])
    if xyz_norm < 1e-8:
        rotvec = 2.0 * q[:3]
    else:
        rotvec = 2.0 * np.arctan2(xyz_norm, np.clip(q[3], -1.0, 1.0)) * q[:3] / xyz_norm
    return np.concatenate((delta_xyz, rotvec)).astype(np.float32)


def compute_episode_motions(states: np.ndarray, motion_offset: int, quat_order: str) -> np.ndarray:
    """按 episode 计算 (T, 6) motion。开头不足 offset 的帧置零 (与推理 deque 行为一致)。"""
    T = len(states)
    motions = np.zeros((T, 6), dtype=np.float32)
    for t in range(T):
        if t - motion_offset >= 0:
            motions[t] = _relative_motion(states[t - motion_offset], states[t], quat_order=quat_order)
    return motions


def _percentiles(vals: np.ndarray, n_bins: int) -> np.ndarray:
    """(n_bins+1,) 分位边界, 强制严格递增。"""
    p = np.percentile(vals, np.linspace(0, 100, n_bins + 1)).astype(np.float32)
    for i in range(1, len(p)):
        if p[i] <= p[i - 1]:
            p[i] = p[i - 1] + 1e-10
    return p


def main(
    data_dir: str = "./data",
    output: str = "./bin_stats_motion.pkl",
    state_col: str = "observation.state",
    quat_order: str = "wxyz",
    motion_offset: int = 20,
    # 平移 3 维合并共用边界 (细桶); 旋转 3 维合并共用边界 (粗桶)
    n_trans_bins: int = 64,
    n_rot_bins: int = 32,
    # phase token: EMA 速度范数的分档百分比 (产生 n-1 个阈值)
    phase_ema_alpha: float = 0.1,
    n_phase_trans: int = 4,
    n_phase_rot: int = 3,
    phase_percentiles: tuple[float, ...] = (50.0, 90.0, 99.0),
):
    # LeRobot v2.1 目录结构: data/chunk-XXX/episode_*.parquet
    files = sorted(
        glob.glob(pathlib.Path(data_dir).joinpath("episode_*.parquet").as_posix())
        + glob.glob(pathlib.Path(data_dir).joinpath("chunk-*", "episode_*.parquet").as_posix())
    )
    if not files:
        raise FileNotFoundError(f"在 {data_dir} 下未找到 episode_*.parquet")
    print(f"Found {len(files)} episodes | quat_order={quat_order} motion_offset={motion_offset}")

    all_motions = []
    all_phase_norms_trans = []
    all_phase_norms_rot = []
    for f in tqdm.tqdm(files, desc="Computing motion"):
        df = pd.read_parquet(f)
        states = np.stack(df[state_col].values).astype(np.float64)
        motions = compute_episode_motions(states, motion_offset, quat_order)
        all_motions.append(motions)
        # phase 用 EMA 速度范数 (区分瞬时抖动 vs 持续运动)
        alpha = phase_ema_alpha
        ema = np.zeros(2, dtype=np.float64)
        for m in motions:
            inst = np.array([np.linalg.norm(m[:3]), np.linalg.norm(m[3:])])
            ema = alpha * inst + (1 - alpha) * ema
            all_phase_norms_trans.append(ema[0])
            all_phase_norms_rot.append(ema[1])

    motions = np.concatenate(all_motions, axis=0)
    phase_t = np.asarray(all_phase_norms_trans)
    phase_r = np.asarray(all_phase_norms_rot)
    print(f"Total frames: {len(motions)}")

    pct_trans = _percentiles(motions[:, :3].reshape(-1), n_trans_bins)
    pct_rot = _percentiles(motions[:, 3:].reshape(-1), n_rot_bins)
    # phase 阈值: 在 EMA 范数分布上取分位数 (静止帧占比高, 分位分档自然区分静止/慢/快)
    pcts = list(phase_percentiles)[: n_phase_trans - 1] if len(phase_percentiles) >= n_phase_trans - 1 else None
    if pcts is None:
        pcts = np.linspace(0, 100, n_phase_trans + 1)[1:-1].tolist()
    phase_trans_th = np.percentile(phase_t, pcts).astype(np.float32)
    pcts_r = np.linspace(0, 100, n_phase_rot + 1)[1:-1].tolist()
    phase_rot_th = np.percentile(phase_r, pcts_r).astype(np.float32)

    print(f"trans motion range [{motions[:, :3].min():.4f}, {motions[:, :3].max():.4f}] -> {n_trans_bins} bins")
    print(f"rot   motion range [{motions[:, 3:].min():.4f}, {motions[:, 3:].max():.4f}] -> {n_rot_bins} bins")
    print(f"phase trans thresholds: {phase_trans_th}")
    print(f"phase rot   thresholds: {phase_rot_th}")

    save_dict = {
        "percentiles_trans": pct_trans,
        "percentiles_rot": pct_rot,
        "n_trans_bins": n_trans_bins,
        "n_rot_bins": n_rot_bins,
        "phase_trans_thresholds": phase_trans_th,
        "phase_rot_thresholds": phase_rot_th,
        "config": {
            "motion_offset": motion_offset,
            "quat_order": quat_order,
            "phase_ema_alpha": phase_ema_alpha,
            "phase_percentiles": list(pcts),
            "state_col": state_col,
        },
    }
    out = pathlib.Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(save_dict, f)
    print(f"Saved motion bin stats to: {out}")


if __name__ == "__main__":
    tyro.cli(main)
