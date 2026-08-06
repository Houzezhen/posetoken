"""从训练数据统计 Pose Token 的非均匀分桶边界。

生成 bin_stats.pkl，包含:
  - percentiles_<dim>: (N+1,) 每个维度的百分位分桶边界
  - pose_split: 维度划分配置

分桶策略:
  1. 基于帧间欧氏速度识别每个 episode 的开头/结尾停滞段
  2. 停滞段(初始/结束 home 位置附近)只占用少量桶(粗分辨率)
  3. 动作中段占用绝大多数桶,且对精细操作(低速但非停滞)加权采样,
     避免精细操作被误判为停滞导致分辨率不足
  4. 支持全部 state 维度分别分桶

用法:
    python scripts/compute_pose_bin_stats.py --config-name=<your-config> --output=./bin_stats.pkl

注意: 需要先运行 scripts/compute_norm_stats.py 确保数据集可加载。
"""

import json
import pathlib
import pickle

import numpy as np
import tqdm
import tyro

import openpi.models.model as _model
import openpi.shared.normalize as normalize
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as transforms

import datasets
from datasets.features import Features, Sequence

try:
    import datasets.features.features as features_mod

    features_mod._FEATURE_TYPES["List"] = Sequence
except Exception:
    if hasattr(Features, "_FEATURE_TYPES"):
        Features._FEATURE_TYPES["List"] = Sequence


class RemoveStrings(transforms.DataTransformFn):
    def __call__(self, x: dict) -> dict:
        return {k: v for k, v in x.items() if not np.issubdtype(np.asarray(v).dtype, np.str_)}


def create_torch_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    model_config: _model.BaseModelConfig,
    num_workers: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    if data_config.repo_id is None:
        raise ValueError("Data config must have a repo_id")
    dataset = _data_loader.create_torch_dataset(data_config, action_horizon, model_config)
    dataset = _data_loader.TransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
        shuffle = True
    else:
        num_batches = len(dataset) // batch_size
        shuffle = False
    data_loader = _data_loader.TorchDataLoader(
        dataset,
        local_batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        num_batches=num_batches,
    )
    return data_loader, num_batches


def create_rlds_dataloader(
    data_config: _config.DataConfig,
    action_horizon: int,
    batch_size: int,
    max_frames: int | None = None,
) -> tuple[_data_loader.Dataset, int]:
    dataset = _data_loader.create_rlds_dataset(data_config, action_horizon, batch_size, shuffle=False)
    dataset = _data_loader.IterableTransformedDataset(
        dataset,
        [
            *data_config.repack_transforms.inputs,
            *data_config.data_transforms.inputs,
            RemoveStrings(),
        ],
        is_batched=True,
    )
    if max_frames is not None and max_frames < len(dataset):
        num_batches = max_frames // batch_size
    else:
        num_batches = len(dataset) // batch_size
    data_loader = _data_loader.RLDSDataLoader(dataset, num_batches=num_batches)
    return data_loader, num_batches


def detect_stall_segments(
    states: np.ndarray,
    speed_thresh: float = 0.01,
    window: int = 5,
) -> tuple[int, int]:
    """识别一个 episode 的开头/结尾停滞段索引。

    用滑动窗口平滑帧间欧氏速度,再从两端向内寻找第一个非停滞帧。

    Args:
        states: (T, D) 单个 episode 的 state 序列
        speed_thresh: 速度阈值,小于此值视为停滞
        window: 滑动窗口大小(帧),用于平滑速度避免抖动误判

    Returns:
        (start, end): 动作中段 [start, end),开头 [0,start) 与结尾 [end,T) 为停滞段
    """
    T = len(states)
    if T < 3:
        return 0, T

    # 帧间欧氏速度
    diffs = np.diff(states, axis=0, prepend=states[:1])
    speeds = np.linalg.norm(diffs, axis=1)  # (T,)

    # 滑动窗口平均,抑制单帧抖动
    if window > 1:
        kernel = np.ones(window) / window
        smooth = np.convolve(speeds, kernel, mode="same")
    else:
        smooth = speeds

    is_stall = smooth < speed_thresh

    # 从开头向右找第一个非停滞帧
    start = 0
    while start < T and is_stall[start]:
        start += 1
    # 从结尾向左找第一个非停滞帧
    end = T
    while end > start and is_stall[end - 1]:
        end -= 1

    # 兜底:中段至少保留 50%,防止极端情况全部判为停滞
    min_mid = T // 2
    if end - start < min_mid:
        # 退化方案: 用前后各 10% 作为停滞段
        start = max(1, T // 10)
        end = T - max(1, T // 20)
        if end <= start:
            start, end = 0, T
    return start, end


def compute_segmented_percentiles(
    stall_values: np.ndarray,
    mid_values: np.ndarray,
    n_bins: int = 2048,
    stall_bin_ratio: float = 0.1,
) -> np.ndarray:
    """分段计算分桶边界: 停滞段占少数桶,中段占多数桶。

    两段各自用百分位法计算内部边界,然后拼接,保证每段内部仍有非均匀分辨率。

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
            return np.array([])
        p = np.percentile(vals, np.linspace(0, 100, nb + 1))
        for i in range(1, len(p)):
            if p[i] <= p[i - 1]:
                p[i] = p[i - 1] + 1e-10
        return p.astype(np.float32)

    p_stall = _pct(stall_values, n_stall_bins)
    p_mid = _pct(mid_values, n_mid_bins)

    if len(p_stall) == 0:
        # 无停滞数据,直接用中段(补齐到 n_bins+1)
        out = p_mid
        if len(out) < n_bins + 1:
            pad = np.full(n_bins + 1 - len(out), out[-1], dtype=np.float32)
            out = np.concatenate([out, pad])
        return out
    if len(p_mid) == 0:
        out = p_stall
        if len(out) < n_bins + 1:
            pad = np.full(n_bins + 1 - len(out), out[-1], dtype=np.float32)
            out = np.concatenate([out, pad])
        return out

    # 拼接: 去掉中段的第一个点(与停滞段最后一个点重叠或邻近)
    merged = np.concatenate([p_stall, p_mid[1:]])
    # 确保总数为 n_bins+1
    if len(merged) < n_bins + 1:
        pad = np.full(n_bins + 1 - len(merged), merged[-1], dtype=np.float32)
        merged = np.concatenate([merged, pad])
    elif len(merged) > n_bins + 1:
        merged = merged[: n_bins + 1]

    # 最终严格递增检查
    for i in range(1, len(merged)):
        if merged[i] <= merged[i - 1]:
            merged[i] = merged[i - 1] + 1e-10
    return merged.astype(np.float32)


def main(
    config_name: str,
    output: str = "./bin_stats.pkl",
    max_frames: int | None = None,
    pose_split: str = '{"trans_xyz": [0, 1, 2], "quat_wxyz": [3, 4, 5, 6], "other": []}',
    # === 停滞检测参数 ===
    speed_thresh: float = 0.01,
    smooth_window: int = 5,
    trim_head_ratio: float = 0.0,
    trim_tail_ratio: float = 0.0,
    # === 分桶分配参数 ===
    stall_bin_ratio: float = 0.1,
    n_bins: int = 2048,
    # === 精细操作保护 ===
    fine_op_speed_max: float = 0.1,
    fine_op_weight: int = 2,
    # === 分桶模式 ===
    bin_all_dims: bool = True,
):
    """统计 state 的分桶边界,避开开头/结尾停滞段,密集覆盖动作中段。

    Args:
        config_name: 训练配置名称 (如 "pi05_aloha")
        output: 输出 pkl 文件路径
        max_frames: 最大采样帧数 (None 表示全部)
        pose_split: JSON 格式的 state 结构描述
        speed_thresh: 帧间欧氏速度阈值,小于此值视为停滞 (单位与 state 一致)
        smooth_window: 速度平滑窗口 (帧),抑制抖动
        trim_head_ratio: 额外按时间比例剔除每个 episode 开头的比例 (0=不剔除)
        trim_tail_ratio: 额外按时间比例剔除每个 episode 结尾的比例 (0=不剔除)
        stall_bin_ratio: 停滞段分配的桶数比例 (0~1),其余给动作中段
        n_bins: 每个维度的总桶数
        fine_op_speed_max: 精细操作速度上限 (低速但 > speed_thresh),会被加权采样
        fine_op_weight: 精细操作帧的采样权重倍数
        bin_all_dims: True=对 state 所有维度分别分桶; False=只按 pose_split 的 XYZ 分桶
    """
    pose_split_dict = json.loads(pose_split)
    print(f"Pose split: {pose_split_dict}")

    config = _config.get_config(config_name)
    data_config = config.data.create(config.assets_dirs, config.model)

    if data_config.rlds_data_dir is not None:
        data_loader, num_batches = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, num_batches = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    # ============ 第一遍: 收集所有帧 + episode 边界,用于停滞检测 ============
    print(f"Collecting states from {num_batches} batches...")
    all_states_list = []
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Collecting states"):
        states = np.asarray(batch["state"])  # (B, D) or (D,)
        if states.ndim == 1:
            states = states[None, :]
        all_states_list.append(states)
    all_states = np.concatenate(all_states_list, axis=0)  # (N_total, D)
    N_total, D = all_states.shape
    print(f"Total frames: {N_total}, state dim D = {D}")

    # ============ 识别 episode 边界 (通过 episode_index 字段或窗口检测) ============
    # 由于上面的循环已合并所有 batch,这里用一个简单的启发式:
    # 把数据视为单个长序列做停滞检测会更不稳,因此改为在 batch 内按 episode 分组.
    # 但为保持简单可靠,这里重新做一次遍历,按 episode_index 分段处理.
    # --- 重新按 episode 分段 ---
    episode_segments = []  # list of (start_idx, end_idx) in all_states
    # 用 batch 再遍历一次获取 episode_index
    seg_start = 0
    prev_ep = None
    # 重新迭代 (data_loader 是生成器,需要重建)
    if data_config.rlds_data_dir is not None:
        data_loader, _ = create_rlds_dataloader(
            data_config, config.model.action_horizon, config.batch_size, max_frames
        )
    else:
        data_loader, _ = create_torch_dataloader(
            data_config, config.model.action_horizon, config.batch_size, config.model, config.num_workers, max_frames
        )

    idx_cursor = 0
    for batch in tqdm.tqdm(data_loader, total=num_batches, desc="Segmenting episodes"):
        ep_idx = np.asarray(batch.get("episode_index", np.zeros(len(batch["state"])))).reshape(-1)
        states = np.asarray(batch["state"])
        if states.ndim == 1:
            states = states[None, :]
        B = len(states)
        # 找出 episode 变化点
        for i in range(B):
            cur_ep = int(ep_idx[i]) if i < len(ep_idx) else prev_ep
            if prev_ep is None:
                prev_ep = cur_ep
            if cur_ep != prev_ep:
                episode_segments.append((seg_start, idx_cursor))
                seg_start = idx_cursor
                prev_ep = cur_ep
            idx_cursor += 1
    if seg_start < idx_cursor:
        episode_segments.append((seg_start, idx_cursor))
    print(f"Found {len(episode_segments)} episodes (avg len={N_total/max(1,len(episode_segments)):.1f})")

    # ============ 对每个 episode 做停滞检测,划分 stall/mid/fine_op ============
    # 收集每个维度的三段数据
    stall_per_dim = [[] for _ in range(D)]
    mid_per_dim = [[] for _ in range(D)]
    n_stall_frames = 0
    n_mid_frames = 0
    n_fine_op_boosted = 0

    print("Detecting stall segments per episode...")
    for (s, e) in tqdm.tqdm(episode_segments, desc="Per-episode stall detect"):
        ep_states = all_states[s:e]  # (T, D)
        T = len(ep_states)
        if T < 3:
            # 太短,全部当中段
            for d in range(D):
                mid_per_dim[d].append(ep_states[:, d])
            n_mid_frames += T
            continue

        # 时间比例剔除 (可选,默认 0,作为兜底)
        head_trim = int(T * trim_head_ratio)
        tail_trim = int(T * trim_tail_ratio)
        inner_s = head_trim
        inner_e = T - tail_trim

        # 在 [inner_s, inner_e) 内做速度检测
        start, end = detect_stall_segments(
            ep_states[inner_s:inner_e], speed_thresh=speed_thresh, window=smooth_window
        )
        # 转回原 episode 索引
        start += inner_s
        end += inner_s

        # 停滞段: [0, start) + [end, T)
        for d in range(D):
            stall_per_dim[d].append(ep_states[:start, d])
            stall_per_dim[d].append(ep_states[end:, d])
        n_stall_frames += start + (T - end)

        # 中段: [start, end),精细操作加权
        mid_states = ep_states[start:end]
        if len(mid_states) > 1:
            diffs = np.diff(mid_states, axis=0, prepend=mid_states[:1])
            speeds = np.linalg.norm(diffs, axis=1)
            fine_mask = (speeds > speed_thresh) & (speeds < fine_op_speed_max)
            for d in range(D):
                mid_per_dim[d].append(mid_states[:, d])
                # 精细操作帧加权
                if fine_mask.any():
                    mid_per_dim[d].append(np.repeat(mid_states[fine_mask, d], fine_op_weight - 1))
            n_mid_frames += len(mid_states)
            n_fine_op_boosted += int(fine_mask.sum())
        else:
            for d in range(D):
                mid_per_dim[d].append(mid_states[:, d])
            n_mid_frames += len(mid_states)

    print(f"Stall frames: {n_stall_frames} ({n_stall_frames/N_total*100:.1f}%)")
    print(f"Mid frames:   {n_mid_frames} ({n_mid_frames/N_total*100:.1f}%)")
    print(f"Fine-op boosted frames (x{fine_op_weight}): {n_fine_op_boosted}")

    # ============ 决定要分桶的维度 ============
    if bin_all_dims:
        dims_to_bin = list(range(D))
    else:
        trans_idx = pose_split_dict.get("trans_xyz", [0, 1, 2])
        dims_to_bin = sorted(set(trans_idx))
    print(f"Dimensions to bin: {dims_to_bin}")

    # ============ 分段计算分桶边界 ============
    result_percentiles = {}
    for d in dims_to_bin:
        stall_vals = np.concatenate(stall_per_dim[d]) if stall_per_dim[d] else np.array([], dtype=np.float64)
        mid_vals = np.concatenate(mid_per_dim[d]) if mid_per_dim[d] else np.array([], dtype=np.float64)

        if len(stall_vals) > 0:
            print(f"  dim {d}: stall range [{stall_vals.min():.4f}, {stall_vals.max():.4f}], n={len(stall_vals)}")
        if len(mid_vals) > 0:
            print(f"  dim {d}: mid   range [{mid_vals.min():.4f}, {mid_vals.max():.4f}], n={len(mid_vals)}")

        pct = compute_segmented_percentiles(
            stall_vals, mid_vals, n_bins=n_bins, stall_bin_ratio=stall_bin_ratio
        )
        key = f"percentiles_dim{d}"
        result_percentiles[key] = pct
        print(f"  dim {d}: {len(pct)-1} bins, range [{pct[0]:.4f}, {pct[-1]:.4f}]")

    # ============ 保存 ============
    output_path = pathlib.Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        **result_percentiles,
        "pose_split": pose_split_dict,
        "bin_all_dims": bin_all_dims,
        "dims_to_bin": dims_to_bin,
        "n_bins": n_bins,
        "config": {
            "speed_thresh": speed_thresh,
            "smooth_window": smooth_window,
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
