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
    home_thresh: float | None = None,
    window: int = 5,
    speed_dims: int = 3,
    speed_thresh: float = 0.01,
) -> tuple[int, int]:
    """识别一个 episode 的开头/结尾停滞段索引。

    主判据: 帧 XYZ 位置到 home (初始帧) 的欧氏距离。
      - 距离 < home_thresh 的帧视为 home 附近 (停滞)
      - 距离 >= home_thresh 的帧视为动作中段
    这能可靠区分"停滞"和"精细操作":精细操作虽然速度小,但位置远离 home。

    然后从两端向内扫描,找出最长的"开头 home 段"和"结尾 home 段",
    中间 (即使穿过 home) 都当作动作中段。

    Args:
        states: (T, D) 单个 episode 的 state 序列
        home_thresh: 距 home 的 XYZ 距离阈值,小于此值视为停滞。
            None = 自动取 max(0.5, 全程最大距离 * 5%) 作为自适应阈值
        window: 滑动窗口大小(帧),用于平滑"是否在 home 附近"避免单帧抖动
        speed_dims: 用于位置/速度计算的维度数(默认前 3 维 = XYZ 平移)
        speed_thresh: 速度阈值 (兜底用,当 home 距离无法区分时退化到速度判断)

    Returns:
        (start, end): 动作中段 [start, end),开头 [0,start) 与结尾 [end,T) 为停滞段
    """
    T = len(states)
    if T < 3:
        return 0, T

    # 只用前 speed_dims 维 (默认 XYZ) 判断位置
    xyz = states[:, :speed_dims]
    home = xyz[0]
    dist_to_home = np.linalg.norm(xyz - home, axis=1)  # (T,)

    # 自适应阈值: 取整个 episode 最大距离的一个比例
    if home_thresh is None:
        max_dist = float(dist_to_home.max())
        home_thresh = max(0.5, max_dist * 0.05)

    # "是否在 home 附近"的逐帧布尔序列,用滑动窗口多数投票平滑
    near_home = (dist_to_home < home_thresh).astype(np.float32)
    if window > 1:
        kernel = np.ones(window) / window
        smooth = np.convolve(near_home, kernel, mode="same")
        is_stall = smooth >= 0.6  # 窗口内 >=60% 帧在 home 附近才算停滞
    else:
        is_stall = near_home > 0.5

    # 从开头向右找第一个非停滞帧
    start = 0
    while start < T and is_stall[start]:
        start += 1
    # 从结尾向左找第一个非停滞帧
    end = T
    while end > start and is_stall[end - 1]:
        end -= 1

    # 兜底 1: 中段至少保留 50%,防止极端情况全部判为停滞
    min_mid = T // 2
    if end - start < min_mid:
        # 退化方案: 用速度阈值再试一次
        diffs = np.diff(xyz, axis=0, prepend=xyz[:1])
        speeds = np.linalg.norm(diffs, axis=1)
        if window > 1:
            sp_smooth = np.convolve(speeds, np.ones(window) / window, mode="same")
        else:
            sp_smooth = speeds
        is_stall2 = sp_smooth < speed_thresh
        start = 0
        while start < T and is_stall2[start]:
            start += 1
        end = T
        while end > start and is_stall2[end - 1]:
            end -= 1

    # 兜底 2: 仍然太短就用前后固定比例
    if end - start < min_mid:
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
    min_gap: float = 1e-6,
) -> np.ndarray:
    """加权百分位分桶: 通过重复采样让中段在统计中占更大权重,自动去重死桶。

    停滞段和动作中段的数值范围通常会重叠 (机器人在动作过程中会经过 home 位置附近),
    因此不能分段独立计算边界再拼接 (会导致拼接处不单调)。

    策略:
      1. 用加权采样合并数据: mid_values 重复 N 次 (N 由 stall_bin_ratio 反推),
         使中段数据密度更高,从而在百分位分桶时获得更多更窄的桶。
      2. 用合并后的数据计算 n_bins+1 个百分位边界 (天然单调)。
      3. 停滞段在 home 位置有大量相同值,会产生大量零宽"死桶"。
         用 min_gap 合并相邻过近的边界,得到真正有效的桶 (可能 < n_bins)。

    Args:
        stall_values: 停滞段的数值
        mid_values: 动作中段的数值
        n_bins: 总桶数上限
        stall_bin_ratio: 停滞段数据占总数据的期望比例 (0~1),越小则 mid 段分辨率越高
        min_gap: 相邻边界的最小允许间距,小于此值会被合并去重 (float32 精度友好)

    Returns:
        (M+1,) 严格递增的分桶边界, M <= n_bins
    """
    n_bins = int(n_bins)
    stall_values = np.asarray(stall_values, dtype=np.float64).ravel()
    mid_values = np.asarray(mid_values, dtype=np.float64).ravel()

    # 处理某一段为空的退化情况
    if len(stall_values) == 0 and len(mid_values) == 0:
        return np.linspace(0.0, 1.0, n_bins + 1, dtype=np.float32)
    if len(stall_values) == 0:
        values = mid_values
    elif len(mid_values) == 0:
        values = stall_values
    else:
        # 反推 mid 段的重复倍数:
        # 目标: stall 占总数据的 stall_bin_ratio
        #   len(stall) / (len(stall) + k*len(mid)) = stall_bin_ratio
        #   => k = len(stall)*(1-r) / (r*len(mid))
        r = max(1e-4, min(0.999, stall_bin_ratio))
        k = (len(stall_values) * (1.0 - r)) / (r * len(mid_values))
        # 至少重复 1 次,上限 50 防止内存爆炸
        k = max(1.0, min(50.0, k))
        n_repeat = max(1, int(round(k)))
        values = np.concatenate(
            [stall_values, np.repeat(mid_values, n_repeat)]
        )

    # 用合并后的数据计算百分位边界 (天然单调非递减)
    percentiles = np.percentile(values, np.linspace(0, 100, n_bins + 1))

    # 合并过近/重复的边界 (解决 home 位置死桶问题)
    # 保留第一个边界,后续边界必须比上一个至少大 min_gap
    out = [percentiles[0]]
    for v in percentiles[1:]:
        if v - out[-1] >= min_gap:
            out.append(v)
        # 否则丢弃 (与上一个边界太近,合并为一个桶)
    # 如果去重后只剩 1 个点 (所有值相同),添加一个微小偏移作为第二边界
    if len(out) == 1:
        out.append(out[0] + max(min_gap, abs(out[0]) * 1e-6))

    # 转 float32 后再次去重 (float32 精度可能导致相邻值塌缩为同一个值)
    arr32 = np.asarray(out, dtype=np.float32)
    # 用相邻差值 (float32 域内) 做最终去重
    keep = np.concatenate([[True], np.diff(arr32) > 0])
    arr32 = arr32[keep]
    if len(arr32) == 1:
        arr32 = np.array([arr32[0], np.nextafter(arr32[0], np.inf)], dtype=np.float32)
    return arr32


def main(
    config_name: str,
    output: str = "./bin_stats.pkl",
    max_frames: int | None = None,
    pose_split: str = '{"trans_xyz": [0, 1, 2], "quat_wxyz": [3, 4, 5, 6], "other": []}',
    # === 停滞检测参数 ===
    home_thresh: float | None = None,
    speed_thresh: float = 0.01,
    smooth_window: int = 5,
    speed_dims: int = 3,
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

    停滞段判据: 帧 XYZ 位置到 home (初始帧) 的欧氏距离 < home_thresh。
    这能可靠区分"停滞"(在 home 附近)和"精细操作"(远离 home 但速度小)。

    Args:
        config_name: 训练配置名称 (如 "pi05_aloha")
        output: 输出 pkl 文件路径
        max_frames: 最大采样帧数 (None 表示全部)
        pose_split: JSON 格式的 state 结构描述
        home_thresh: 距 home 的 XYZ 距离阈值。None=自适应 max(0.5, max_dist*5%)
        speed_thresh: 速度阈值 (兜底用,当 home 距离无法区分时退化到速度判断)
        smooth_window: 平滑窗口 (帧),抑制"是否在 home 附近"的单帧抖动
        speed_dims: 计算位置/速度的维度数 (默认前 3 维 = XYZ)
        trim_head_ratio: 额外按时间比例剔除每个 episode 开头的比例 (0=不剔除)
        trim_tail_ratio: 额外按时间比例剔除每个 episode 结尾的比例 (0=不剔除)
        stall_bin_ratio: 停滞段数据在统计中的期望占比 (0~1),越小则动作中段分辨率越高 (默认 0.1)
        n_bins: 每个维度分桶数的上限 (实际桶数可能更少,会自动去除零宽死桶)
        fine_op_speed_max: 精细操作速度上限 (中段里速度低于此值的帧会被加权采样)
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

        # 在 [inner_s, inner_e) 内做停滞检测 (基于 XYZ 位置到 home 的距离)
        start_rel, end_rel = detect_stall_segments(
            ep_states[inner_s:inner_e],
            home_thresh=home_thresh,
            window=smooth_window,
            speed_dims=speed_dims,
            speed_thresh=speed_thresh,
        )
        # 转回原 episode 索引
        start = start_rel + inner_s
        end = end_rel + inner_s

        # 停滞段: [0, start) + [end, T)
        for d in range(D):
            stall_per_dim[d].append(ep_states[:start, d])
            stall_per_dim[d].append(ep_states[end:, d])
        n_stall_frames += start + (T - end)

        # 中段: [start, end),对其中精细操作帧 (XYZ 速度小) 加权采样
        # 这里用 XYZ 速度识别精细操作:它们远离 home 但运动缓慢,需要更高分辨率
        mid_states = ep_states[start:end]
        if len(mid_states) > 1:
            mid_xyz = mid_states[:, :speed_dims]
            diffs = np.diff(mid_xyz, axis=0, prepend=mid_xyz[:1])
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
            "home_thresh": home_thresh,
            "speed_thresh": speed_thresh,
            "smooth_window": smooth_window,
            "speed_dims": speed_dims,
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
