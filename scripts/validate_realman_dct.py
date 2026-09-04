"""Validate a trained pi05_realman_pose_dct policy by replaying dataset observations.

验证思路 (DCT 频域 pose token 版本):
    1. 从 LeRobot 数据集加载某个 episode, 用 delta_timestamps 取:
       - observation.state 的最近 30 帧密集序列 [30, 7] (供 DCT 频域分析)
       - action 的未来 30 步绝对位姿 [30, 7] (作为 GT)
    2. 把 state 的 30 帧堆叠作为 observation/state 直接发给策略服务器,
       服务器侧 PiperInputs 走训练路径 (ndim+1), DCT 拿到完整 30 帧窗口。
    3. 策略返回 [30, 7] 的绝对 TCP 位姿预测 (x,y,z + qw,qx,qy,qz)。
    4. 与数据集未来 30 步真实 action 逐维逐 horizon step 对比。

前置条件:
    终端 A 已启动策略服务器 (realman DCT checkpoint):
        CUDA_VISIBLE_DEVICES=1 uv run --python python3.11 scripts/serve_policy.py \
            policy:checkpoint \
            --policy.config=pi05_realman_pose_dct \
            --policy.dir=/home/user/pi05_poseToken/checkpoints/pi05_realman_pose_dct/realman_dct_camframe_delta/15000 \
            --port=8123

运行本脚本 (CPU 即可, 不占 GPU):
        CUDA_VISIBLE_DEVICES="" uv run --python python3.11 scripts/validate_realman_dct.py \
            --host=localhost --port=8123 \
            --repo-id=/media/Data/user/datasets/realman/liftseat_0819_0820_merged_v2.1_cut_camframe \
            --episodes=0,1,2,3,4 \
            --sample-steps=20
"""

# ===== 强制禁用 GPU =====
# 必须在 import jax/torch/lerobot 之前设置, 否则这些库会初始化 GPU 占用显存.
# 本脚本只做数据加载 + WebSocket 通信 + 统计计算, 全部在 CPU 上完成即可.
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # 对 JAX 和 PyTorch 都生效
os.environ["JAX_PLATFORMS"] = "cpu"              # 强制 JAX 用 CPU
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# =========================

import dataclasses
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tqdm
import tyro

# LeRobot 数据集 (与 openpi 训练时使用同一个类)
import lerobot.common.datasets.lerobot_dataset as lerobot_dataset

# openpi-client, 与策略服务器通信
from openpi_client import websocket_client_policy

logger = logging.getLogger(__name__)

# ----- 常量 (与 pi05_realman_pose_dct 训练配置一致) -----
STATE_DIM = 7             # x,y,z + qw,qx,qy,qz (相机系绝对位姿, 无 gripper)
ACTION_HORIZON = 30       # pi05_realman_pose_dct: action_horizon=30
DCT_WINDOW = 30           # dct_state_window=30 (DCT 频域分析窗口)
FPS = 30                  # 数据集采样率 (info.json)

# 数据集 state/action 维度顺序: x,y,z,qw,qx,qy,qz (见 meta/info.json)
ACTION_DIM_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz"]


@dataclasses.dataclass
class Args:
    # 服务器地址
    host: str = "localhost"
    port: int = 8123

    # LeRobot 数据集路径 (与 config.py 里 LeRobotRealmanDataConfig.repo_id 一致)
    repo_id: str = "/media/Data/user/datasets/realman/liftseat_0819_0820_merged_v2.1_cut_camframe"

    # 要验证的 episode 列表 (逗号分隔)
    episodes: str = "0,1,2"

    # 每个 episode 均匀采样的时刻数
    sample_steps: int = 10

    # 要验证的 horizon 步数 (模型总是输出 30 步, 这里截取前 N 步来评估)
    horizon: int = 30

    # 输出图和原始数据的路径
    out_png: pathlib.Path = pathlib.Path("validation_realman_dct.png")
    out_npz: pathlib.Path | None = None

    # 覆盖默认 prompt (默认从数据集 task 表读取)
    prompt: str | None = None


def load_episode(repo_id: str, ep_idx: int) -> lerobot_dataset.LeRobotDataset:
    """加载单个 episode 的 LeRobot 数据集, 带 delta_timestamps。

    - observation.state: 取最近 DCT_WINDOW 帧密集序列 (含当前帧),
      与训练数据 pipeline (data_loader.create_torch_dataset) 的 dense_ts 一致,
      这样服务器侧 PiperInputs 走训练路径, DCT 拿到完整 30 帧窗口。
    - action: 取未来 ACTION_HORIZON 步绝对位姿, 作为 GT。
    其它 key (图像等) 仍为当前帧单帧。
    """
    delta_timestamps = {
        "action": [i / FPS for i in range(ACTION_HORIZON)],
        "observation.state": [-(i) / FPS for i in range(DCT_WINDOW - 1, -1, -1)],
    }
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        episodes=[ep_idx],
        delta_timestamps=delta_timestamps,
    )
    return dataset


def get_prompt(repo_id: str, fallback: str | None) -> str:
    """从数据集 task 表读取 prompt (prompt_from_task=True)。

    realman liftseat 数据集只有 1 个 task, 取该 task 文本即可。
    """
    if fallback is not None:
        return fallback
    meta = lerobot_dataset.LeRobotDatasetMetadata(repo_id)
    tasks = meta.tasks  # dict {task_index: task_text}
    if not tasks:
        raise ValueError("Dataset has no tasks; pass --prompt explicitly.")
    # 单 task 数据集, 取唯一的 task 文本
    prompt = next(iter(tasks.values()))
    logger.info("Using prompt from dataset task table: %r", prompt)
    return prompt


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """把图像统一成 (H, W, 3) uint8 格式。

    LeRobot 视频解码后通常是 (C, H, W) float32 或 uint8,
    piper_policy._parse_image 内部也会做同样转换, 客户端先压缩成 uint8 能省带宽。
    """
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def build_obs(frame: dict, prompt: str) -> dict:
    """把 LeRobot 的一帧 (带 delta_timestamps) 转成客户端发给服务器的 obs 格式。

    键名严格匹配 piper_policy.PiperInputs 读取的键:
        - observation/state     [30, 7] 密集 DCT 窗口 (最后一帧 = 当前帧)
        - observation/image     (HWC uint8 第三人称相机 cam_bottom)
        - observation/wrist_image (HWC uint8 腕部相机 cam_wrist)
        - prompt

    注意: 不发 observation/target_image —— realman 无第三路相机,
    PiperInputs 会自动零填充 + mask 掉 (训练/推理一致)。
    """
    # state: delta_timestamps 取了最近 30 帧, 形状 [30, 7], 直接整段发出
    state = np.asarray(frame["observation.state"], dtype=np.float32)  # [30, 7]
    assert state.shape == (DCT_WINDOW, STATE_DIM), (
        f"observation.state shape {state.shape}, expected ({DCT_WINDOW}, {STATE_DIM}). "
        f"可能 delta_timestamps 在 episode 边界被 padding, 请避开边界采样。"
    )

    img_bottom = _to_hwc_uint8(np.asarray(frame["observation.images.cam_bottom"]))
    img_wrist = _to_hwc_uint8(np.asarray(frame["observation.images.cam_wrist"]))

    return {
        "observation/state": state,
        "observation/image": img_bottom,
        "observation/wrist_image": img_wrist,
        "prompt": prompt,
    }


def get_gt_action_chunk(frame: dict, horizon: int) -> np.ndarray:
    """从带 delta_timestamps 的 frame 取未来 horizon 步绝对 action [horizon, 7]。

    delta_timestamps['action'] 已让 LeRobot 把未来 30 步 action 堆叠好,
    这里只需截取前 horizon 步即可。
    """
    actions = np.asarray(frame["action"], dtype=np.float32)  # [30, 7]
    return actions[:horizon]


def main(args: Args) -> None:
    # 1. 连接策略服务器 (会阻塞直到服务器就绪)
    logger.info("Connecting to policy server at %s:%d ...", args.host, args.port)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logger.info("Server metadata: %s", client.get_server_metadata())

    prompt = get_prompt(args.repo_id, args.prompt)
    ep_indices = [int(x) for x in args.episodes.split(",") if x.strip() != ""]

    all_pred, all_gt = [], []

    # 2. 遍历每个 episode
    for ep_idx in ep_indices:
        dataset = load_episode(args.repo_id, ep_idx)
        n = len(dataset)
        # 留出 DCT 窗口 (前) 和 action horizon (后), 保证样本不被 padding 污染
        valid_start = DCT_WINDOW - 1          # t >= 29, 保证 state 有完整 30 帧
        valid_end = n - ACTION_HORIZON        # t <= n-30, 保证 action 有完整 30 步
        valid_range = max(1, valid_end - valid_start + 1)
        steps = np.linspace(valid_start, valid_end, num=min(args.sample_steps, valid_range), dtype=int)
        steps = [int(s) for s in steps]
        logger.info("Episode %d: %d frames, sampling %d steps at %s", ep_idx, n, len(steps), steps)

        # 3. 第一次推理通常会触发 JIT 编译, 单独跑一次热身
        if ep_idx == ep_indices[0]:
            warmup_obs = build_obs(dataset[steps[0]], prompt)
            _ = client.infer(warmup_obs)
            logger.info("Warmup inference done.")

        # 4. 采样并推理
        for t in tqdm.tqdm(steps, desc=f"ep{ep_idx}"):
            frame = dataset[t]
            obs = build_obs(frame, prompt)

            pred = client.infer(obs)["actions"]            # [30, 7] 绝对 TCP 位姿
            pred = np.asarray(pred, dtype=np.float32)
            pred = pred[: args.horizon]                     # 截取前 horizon 步

            gt = get_gt_action_chunk(frame, args.horizon)   # [horizon, 7]

            all_pred.append(pred)
            all_gt.append(gt)

    # 5. 汇总
    all_pred = np.stack(all_pred, axis=0)   # (S, H, 7)
    all_gt = np.stack(all_gt, axis=0)       # (S, H, 7)
    diff = all_pred - all_gt
    num_samples = all_pred.shape[0]

    # per-dim 误差 (所有 horizon step 平均)
    mse_per_dim = np.mean(diff ** 2, axis=(0, 1))
    mae_per_dim = np.mean(np.abs(diff), axis=(0, 1))
    mse_total = float(np.mean(diff ** 2))

    # per-horizon-step 误差 (所有 dim 平均): 模型能准确预测多远的未来
    mse_per_step = np.mean(diff ** 2, axis=(0, 2))   # (horizon,)
    mae_per_step = np.mean(np.abs(diff), axis=(0, 2))

    # per-step-per-dim 的 MAE: 对样本平均, 保留 step 和 dim
    mae_per_step_per_dim = np.mean(np.abs(diff), axis=0)   # (horizon, 7)

    # 6. 打印
    print("\n" + "=" * 60)
    print(f"Validation Summary  (config=pi05_realman_pose_dct, samples={num_samples}, "
          f"horizon={args.horizon}, dim={STATE_DIM})")
    print("=" * 60)
    print(f"Overall MSE: {mse_total:.6f}")
    print(f"\nPer-dimension error:")
    print(f"  {'dim':<6} {'MSE':<12} {'MAE':<12}")
    for i, name in enumerate(ACTION_DIM_NAMES):
        print(f"  {name:<6} {mse_per_dim[i]:<12.6f} {mae_per_dim[i]:<12.6f}")

    # 找出误差仍小于阈值的 horizon 长度 (经验阈值)
    threshold = max(1e-4, mse_per_step[0] * 5)
    effective_steps = int(np.sum(mse_per_step < threshold))
    print(f"\nEffective prediction horizon (MSE < {threshold:.4f}): {effective_steps} steps "
          f"(~{effective_steps / FPS:.2f}s @ {FPS}fps)")

    # 7. 画图: 3 个子图
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))

    horizon_axis = np.arange(args.horizon)

    # 子图 1: per-horizon-step 误差曲线 (所有 dim 平均)
    ax = axes[0]
    ax.plot(horizon_axis, mse_per_step, "r-", label="MSE", linewidth=2)
    ax.plot(horizon_axis, mae_per_step, "b-", label="MAE", linewidth=2)
    ax.axvline(effective_steps, color="g", linestyle="--", alpha=0.5,
              label=f"effective horizon ≈ {effective_steps}")
    ax.set_xlabel("Action horizon step (0 = next frame)")
    ax.set_ylabel("Error")
    ax.set_title("Error grows with prediction horizon (avg over 7 dims)")
    ax.legend()
    ax.grid(alpha=0.3)

    # 子图 2: 每个维度的 MSE 柱状图
    ax = axes[1]
    bars = ax.bar(ACTION_DIM_NAMES, mse_per_dim, color="salmon", edgecolor="darkred")
    ax.set_ylabel("MSE")
    ax.set_title(f"Per-dimension MSE  (overall={mse_total:.5f})")
    ax.grid(alpha=0.3, axis="y")
    for bar, val in zip(bars, mse_per_dim):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # 子图 3: 7 个维度各自的 MAE-vs-step 折线图
    ax = axes[2]
    colors = plt.cm.tab10(np.linspace(0, 1, STATE_DIM))
    for i, (name, color) in enumerate(zip(ACTION_DIM_NAMES, colors)):
        ax.plot(horizon_axis, mae_per_step_per_dim[:, i],
                color=color, label=name, linewidth=1.5)
    ax.set_xlabel("Action horizon step (0 = next frame)")
    ax.set_ylabel("MAE")
    ax.set_title(f"Per-dimension MAE across {args.horizon} steps")
    ax.legend(ncol=3, fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)

    fig.suptitle(f"pi05_realman_pose_dct validation on {args.repo_id}  |  samples={num_samples}",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=120)
    logger.info("Saved figure to %s", args.out_png)

    # 8. 可选: 保存原始数据
    if args.out_npz is not None:
        np.savez(args.out_npz, pred=all_pred, gt=all_gt,
                 mse_per_step=mse_per_step, mse_per_dim=mse_per_dim)
        logger.info("Saved raw arrays to %s", args.out_npz)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))
