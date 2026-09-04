#!/usr/bin/env python3
"""把 LeRobot 数据集的末端位姿（base 系，法兰，四元数）转换到相机坐标系。

眼在手外（eye-to-hand）: T_cam_base = [R|t]（hand_eye_result_eye_to_hand_eco63f.json），
p_cam = R @ p_base + t，q_cam = quat(R @ R_base)。

- 只改 parquet 中的 observation.state / action（7 维 xyz+qwqxqyqz），
  重新计算转换后的统计量并写入 meta/stats.json、meta/episodes_stats.jsonl。
- videos / episodes.jsonl / tasks.jsonl / norm_stats.json 等其他文件原样复制。
- 输出为新数据集文件夹。
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

STATE = "observation.state"
ACTION = "action"
QUAT_COLS = 4  # state/action 第 4 列起为四元数，共 4 列


def quat_to_mat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
    ], -2)


def mat_to_quat(R):
    # R: (...,3,3) -> (...,4) wxyz
    m = R
    t = m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]
    q = np.zeros(m.shape[:-2] + (4,))
    for idx in np.ndindex(m.shape[:-2]):
        M = m[idx]
        tr = M.trace()
        if tr > 0:
            s = np.sqrt(tr + 1.0) * 2
            q[idx] = [0.25 * s, (M[2, 1] - M[1, 2]) / s, (M[0, 2] - M[2, 0]) / s, (M[1, 0] - M[0, 1]) / s]
        elif M[0, 0] > M[1, 1] and M[0, 0] > M[2, 2]:
            s = np.sqrt(1.0 + M[0, 0] - M[1, 1] - M[2, 2]) * 2
            q[idx] = [(M[2, 1] - M[1, 2]) / s, 0.25 * s, (M[0, 1] + M[1, 0]) / s, (M[0, 2] + M[2, 0]) / s]
        elif M[1, 1] > M[2, 2]:
            s = np.sqrt(1.0 + M[1, 1] - M[0, 0] - M[2, 2]) * 2
            q[idx] = [(M[0, 2] - M[2, 0]) / s, (M[0, 1] + M[1, 0]) / s, 0.25 * s, (M[1, 2] + M[2, 1]) / s]
        else:
            s = np.sqrt(1.0 + M[2, 2] - M[0, 0] - M[1, 1]) * 2
            q[idx] = [(M[1, 0] - M[0, 1]) / s, (M[0, 2] + M[2, 0]) / s, (M[1, 2] + M[2, 1]) / s, 0.25 * s]
    return q


def transform_pose(pose7, R_cb, t_cb):
    """pose7: (N,7) [x,y,z,qw,qx,qy,qz] base 系 -> 相机系."""
    p = pose7[:, 0:3]
    R_b = quat_to_mat(pose7[:, 3:7])
    p_c = p @ R_cb.T + t_cb          # (N,3)
    R_c = R_cb @ R_b                 # (N,3,3): R_cam = R_cb @ R_b
    q_c = mat_to_quat(R_c)
    # 与上一帧符号对齐，保证连续
    flip = np.cumprod(np.concatenate([np.ones(1), np.sign(np.sum(q_c[1:] * q_c[:-1], axis=1))])) < 0
    q_c[flip] *= -1
    out = np.concatenate([p_c, q_c], axis=1)
    return out.astype("float32")


def update_stats_entry(entry, arr):
    entry["min"] = arr.min(axis=0).tolist()
    entry["max"] = arr.max(axis=0).tolist()
    entry["mean"] = arr.mean(axis=0).tolist()
    entry["std"] = arr.std(axis=0).tolist()
    entry["count"] = [arr.shape[0]]  # lerobot 要求 count 是长度 1 的数组


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/media/Data/user/datasets/realman/pushbutton_0825_merge_cut")
    ap.add_argument("--dst", default="/media/Data/user/datasets/realman/pushbutton_0825_merge_cut_camframe")
    ap.add_argument("--handeye", default="/home/user/pi05_poseToken/hand_eye_result_eye_to_hand_eco63f.json")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    he = json.loads(Path(args.handeye).read_text())
    R_cb = np.array(he["rotation_matrix"], dtype="float64")
    t_cb = np.array(he["translation_vector"], dtype="float64").reshape(3)

    assert dst != src and not dst.exists(), "目标目录必须不存在"
    dst.mkdir(parents=True)

    # 复制 meta / videos / norm_stats 等所有内容（data/ 之后单独转换）
    for item in src.iterdir():
        if item.is_dir():
            shutil.copytree(item, dst / item.name)
        else:
            shutil.copy2(item, dst / item.name)
    shutil.rmtree(dst / "data")

    data_dir = dst / "data" / "chunk-000"
    data_dir.mkdir(parents=True)

    files = sorted((src / "data" / "chunk-000").glob("episode_*.parquet"))
    print(f"{len(files)} episodes")

    for i, f in enumerate(files):
        table = pq.read_table(f)
        df = table.to_pandas()
        for key in (STATE, ACTION):
            if key not in df.columns:
                continue
            n = len(df)
            arr = np.stack(df[key].tolist()).astype("float64")  # (N,7)
            arr = arr[:, :7]
            new = transform_pose(arr, R_cb, t_cb)
            df[key] = [row for row in new]
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), data_dir / f.name)
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(files)}")

    # 重算 stats
    stats = json.loads((dst / "meta" / "stats.json").read_text())
    for key in (STATE, ACTION):
        if key in stats:
            allv = []
            for f in files:
                df = pq.read_table(dst / "data" / "chunk-000" / f.name).to_pandas()
                allv.append(np.stack(df[key].tolist()))
            update_stats_entry(stats[key], np.concatenate(allv))
    (dst / "meta" / "stats.json").write_text(json.dumps(stats, indent=4))

    # episodes_stats.jsonl 逐 episode 重算
    eps_stats_path = dst / "meta" / "episodes_stats.jsonl"
    lines_out = []
    for line in eps_stats_path.read_text().strip().splitlines():
        rec = json.loads(line)
        ep = rec["episode_index"]
        df = pq.read_table(dst / "data" / "chunk-000" / f"episode_{ep:06d}.parquet").to_pandas()
        for key in (STATE, ACTION):
            if key in rec["stats"]:
                update_stats_entry(rec["stats"][key], np.stack(df[key].tolist()))
        lines_out.append(json.dumps(rec))
    eps_stats_path.write_text("\n".join(lines_out) + "\n")

    # info.json 标注坐标系
    info_path = dst / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    info["robot_type"] = "realman_eco63f"
    (info_path).write_text(json.dumps(info, indent=4))

    print("done:", dst)


if __name__ == "__main__":
    import pyarrow as pa
    main()
