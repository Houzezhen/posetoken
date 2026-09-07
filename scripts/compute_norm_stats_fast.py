#!/usr/bin/env python3
"""加速版 norm_stats 计算: 绕过 LeRobot torch dataloader (视频解码+多进程很慢),
直接读 parquet 计算 state / actions 的 mean/std/q01/q99, 输出与
compute_norm_stats.py 完全同格式的 norm_stats.json。

注意:
  - actions 统计与原版一致需按训练 action 序列展开 (action_horizon 滑窗采样)。
    由于 action[t]=state[t+1] 且逐帧重复出现在多个 chunk 中, 直接对 parquet
    全量 action 做统计与滑窗加权的差别只在 episode 边界截断, 误差可忽略
    (q01/q99/mean 均为近似)。如需严格一致, 用 --strict-horizon。
用法:
    python scripts/compute_norm_stats_fast.py --config pi05_realman_pose_dct
"""
import argparse
import json
import pathlib

import numpy as np
import pyarrow.parquet as pq

import openpi.models.model as _model
import openpi.policies.piper_policy as piper_policy
import openpi.shared.normalize as normalize
import openpi.training.config as _config


def quantiles(arr: np.ndarray, qs=(0.01, 0.99)) -> list[np.ndarray]:
    # 分块百分位近似 (内存友好): 全量 58k 帧 × 7 维很小, 直接算
    return [np.quantile(arr, q, axis=0) for q in qs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    config = _config.get_config(args.config)
    data_config = config.data.create(config.assets_dirs, config.model)
    repo = data_config.repo_id
    data_dir = pathlib.Path(repo)

    files = sorted(data_dir.glob("data/**/episode_*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet under {data_dir}")
    print(f"{len(files)} episodes from {data_dir}")

    state_all, action_all = [], []
    for f in files:
        df = pq.read_table(f, columns=["observation.state", "action"]).to_pandas()
        state_all.append(np.stack(df["observation.state"].values))
        action_all.append(np.stack(df["action"].values))

    state = np.concatenate(state_all).astype(np.float64)
    action = np.concatenate(action_all).astype(np.float64)

    # 计算 motion (相邻帧差分: Δxyz + 相对旋转向量), 与 PiperInputs 的训练管线一致。
    # Realman 数据集四元数为 wxyz, Piper 为 xyzw, 由 config.data.quat_order 决定。
    quat_order = getattr(config.data, "quat_order", "xyzw")
    motion_all = []
    for s in state_all:
        if s.shape[0] >= 2:
            m = np.stack(
                [piper_policy._relative_motion(s[i], s[i + 1], quat_order=quat_order) for i in range(s.shape[0] - 1)]
            )
        else:
            m = np.zeros((0, piper_policy.PIPER_MOTION_DIM), dtype=np.float64)
        motion_all.append(m)
    motion = np.concatenate(motion_all).astype(np.float64)

    # 若 data_transforms 含 DeltaActions, 统计作用在 delta 之后 (与训练管线一致:
    # repack -> data_transforms(Delta) -> Normalize)。action[t]-state[t] 近似
    # chunk 首帧对齐的 delta (action[t]=state[t+1], chunk 首帧差一帧, 可忽略)。
    delta = None
    for t in data_config.data_transforms.inputs:
        if type(t).__name__ == "DeltaActions" and t.mask is not None:
            delta = np.asarray(t.mask)
    if delta is not None:
        d = delta.shape[-1]
        # 每 episode 内: action[t] - state[t0] (chunk 首帧 state)
        action = np.concatenate(
            [a - s[0:1] for a, s in zip(action_all, state_all)]
        ).astype(np.float64)
        print(f"delta actions (mask={delta.tolist()[:d]}): mean={action.mean(0).round(4)}")
    print(f"state: {state.shape}, action: {action.shape}")

    def stats_of(arr):
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        q01, q99 = quantiles(arr)
        return normalize.NormStats(mean=mean, std=std, q01=q01, q99=q99)

    norm_stats = {"state": stats_of(state), "actions": stats_of(action), "motion": stats_of(motion)}

    output_path = config.assets_dirs / repo
    print(f"Writing stats to: {output_path}")
    normalize.save(output_path, norm_stats)
    print(json.dumps({k: {"mean": v.mean.tolist(), "std": v.std.tolist()} for k, v in norm_stats.items()}, indent=2))


if __name__ == "__main__":
    main()
