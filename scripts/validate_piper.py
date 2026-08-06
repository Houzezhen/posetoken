"""Validate a trained pi05_piper policy by replaying dataset observations.

验证思路:
    1. 从 LeRobot 数据集加载某个 episode
    2. 在每个采样时刻 t, 用数据集 frame t 的 obs 调用策略服务器
    3. 策略返回 (50, 7) 的绝对 TCP 位姿预测 (x,y,z + qw,qx,qy,qz)
    4. 对比数据集 frame t~t+49 的真实 action (前 7 维)
    5. 输出 per-horizon-step 误差曲线 + per-dim 误差

前置条件:
    终端 A 已启动策略服务器 (四元数版本 checkpoint):
        CUDA_VISIBLE_DEVICES=1 uv run scripts/serve_policy.py policy:checkpoint \
            --policy.config=pi05_piper \
            --policy.dir=checkpoints/pi05_piper/data_toiletButton_0626_quaternion/15000 \
            --port=8123

运行本脚本:
    CUDA_VISIBLE_DEVICES="" uv run scripts/validate_piper.py \
        --host=localhost --port=8123 \
        --repo-id=TCP/data_toiletButton_0626_quaternion \
        --episodes=0,1,2,3,4 \
        --sample-steps=20

输出:
    - 终端打印每个维度的 MSE/MAE
    - 保存 per-horizon-step 误差曲线 PNG
    - 可选保存原始预测和真值 npz
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

# ----- 常量 (与 pi05_piper 训练配置一致) -----
STATE_DIM = 7          # SelectFirstDims(7): x,y,z + quaternion(4), 去掉 gripper
ACTION_HORIZON = 50    # pi05 模型 action_horizon
FPS = 30               # 数据集采样率 (info.json)
DEFAULT_PROMPT = "press the toilet button"

ACTION_DIM_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]


@dataclasses.dataclass
class Args:
    # 服务器地址
    host: str = "localhost"
    port: int = 8123

    # LeRobot 数据集路径 (与 config.py 里 LeRobotPiperDataConfig.repo_id 一致)
    # 支持两种写法:
    #   1. 完整绝对路径: "/home/user/.cache/huggingface/lerobot/TCP/data_toiletButton_0626_quaternion"
    #   2. HuggingFace 短格式: "TCP/data_toiletButton_0626_quaternion"
    #      (会自动加上 ~/.cache/huggingface/lerobot/ 前缀)
    repo_id: str = "/home/user/.cache/huggingface/lerobot/TCP/data_toiletButton_0626_quaternion"

    # 要验证的 episode 列表 (逗号分隔)
    episodes: str = "0,1,2"

    # 每个 episode 均匀采样的时刻数
    sample_steps: int = 10

    # 要验证的 horizon 步数 (模型总是输出 50 步, 这里截取前 N 步来评估)
    # 例如 horizon=10 表示只看模型预测的前 10 步精度
    horizon: int = 10

    # 输出图和原始数据的路径
    out_png: pathlib.Path = pathlib.Path("validation_result.png")
    out_npz: pathlib.Path | None = None

    # 覆盖默认 prompt (数据集里是 "press the toilet button")
    prompt: str = DEFAULT_PROMPT


def load_episode(repo_id: str, ep_idx: int) -> lerobot_dataset.LeRobotDataset:
    """加载单个 episode 的 LeRobot 数据集.

    注意: 这里不传 delta_timestamps, 因为我们手动从 dataset[t+i] 取未来 action,
    这样能完全控制 GT 的取法, 避免 LeRobot 边界 padding 行为影响判断.
    """
    dataset = lerobot_dataset.LeRobotDataset(
        repo_id,
        episodes=[ep_idx],
    )
    return dataset


def build_obs(frame: dict, prompt: str) -> dict:
    """把 LeRobot 的一帧转成客户端发给服务器的 obs 格式.

    键名严格匹配 piper_policy.PiperInputs 读取的键:
        - observation/state     (前 7 维, 绝对 TCP 位姿)
        - observation/image     (HWC uint8 第三人称相机)
        - observation/wrist_image (HWC uint8 腕部相机)
        - prompt
    """
    # state: 数据集里是 8 维 (含 gripper), 客户端发前 7 维即可
    # (服务器侧 PiperInputs 会用 pad_to_dim 把它补到 action_dim=32)
    state = np.asarray(frame["observation.state"], dtype=np.float32)[:STATE_DIM]

    # 图像: LeRobot 默认存为 CHW float32 (或 uint8), 这里转成 HWC uint8
    # piper_policy._parse_image 内部也会做同样的转换,
    # 但客户端先把图压缩成 uint8 能减少 WebSocket 带宽
    img_primary = _to_hwc_uint8(np.asarray(frame["observation.images.primary"]))
    img_wrist = _to_hwc_uint8(np.asarray(frame["observation.images.wrist"]))
    img_target = _to_hwc_uint8(np.asarray(frame["observation.images.target"]))

    return {
        "observation/state": state,
        "observation/image": img_primary,
        "observation/wrist_image": img_wrist,
        "observation/target_image": img_target,
        "prompt": prompt,
    }


def _to_hwc_uint8(img: np.ndarray) -> np.ndarray:
    """把图像统一成 (H, W, 3) uint8 格式."""
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    # LeRobot 视频解码后通常是 (C, H, W), 第 0 维=3
    if img.ndim == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0))
    return img


def get_gt_action_chunk(
    dataset: lerobot_dataset.LeRobotDataset,
    t: int,
    horizon: int,
) -> np.ndarray:
    """从数据集取 ground-truth 的未来 horizon 步 action (前 7 维).

    每一帧的 action 是数据集记录的绝对 TCP 位姿 (8 维, 含 gripper),
    与模型输出的绝对位姿 (7 维, 无 gripper) 直接逐位对比即可.

    末尾不足时用最后一帧补齐 (避免越界), 这种样本在评估时应尽量避免.
    """
    n = len(dataset)
    t = int(t)   # 确保是 Python 原生 int, 避免 numpy.int64 索引报错
    actions = []
    for i in range(t, min(t + horizon, n)):
        actions.append(np.asarray(dataset[i]["action"], dtype=np.float32)[:STATE_DIM])
    arr = np.stack(actions, axis=0)
    # 末尾不足 horizon 时用最后一帧补齐
    if arr.shape[0] < horizon:
        pad = np.tile(arr[-1:], (horizon - arr.shape[0], 1))
        arr = np.concatenate([arr, pad], axis=0)
    return arr


def main(args: Args) -> None:
    # 1. 连接策略服务器 (会阻塞直到服务器就绪)
    logger.info("Connecting to policy server at %s:%d ...", args.host, args.port)
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)
    logger.info("Server metadata: %s", client.get_server_metadata())

    ep_indices = [int(x) for x in args.episodes.split(",") if x.strip() != ""]

    all_pred, all_gt = [], []

    # 2. 遍历每个 episode
    for ep_idx in ep_indices:
        dataset = load_episode(args.repo_id, ep_idx)
        n = len(dataset)
        # 留出最后 args.horizon 帧, 保证每个采样点都有完整的 GT
        valid_range = max(1, n - args.horizon)
        steps = np.linspace(0, valid_range - 1, num=min(args.sample_steps, valid_range), dtype=int)
        # 转成 Python 原生 int, 避免 numpy.int64 索引 LeRobot dataset 报错
        steps = [int(s) for s in steps]
        logger.info("Episode %d: %d frames, sampling %d steps at %s", ep_idx, n, len(steps), steps)

        # 3. 第一次推理通常会触发 JIT 编译, 单独跑一次热身
        if ep_idx == ep_indices[0]:
            warmup_obs = build_obs(dataset[steps[0]], args.prompt)
            _ = client.infer(warmup_obs)
            logger.info("Warmup inference done.")

        # 4. 采样并推理
        for t in tqdm.tqdm(steps, desc=f"ep{ep_idx}"):
            frame = dataset[t]
            obs = build_obs(frame, args.prompt)

            pred = client.infer(obs)["actions"]            # (50, 7) 绝对 TCP 位姿
            pred = np.asarray(pred, dtype=np.float32)
            pred = pred[: args.horizon]                     # 截取前 horizon 步

            gt = get_gt_action_chunk(dataset, t, args.horizon)   # (horizon, 7)

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

    # per-horizon-step 误差 (所有 dim 平均)
    # 这条曲线能告诉你 "模型能准确预测多远的未来"
    mse_per_step = np.mean(diff ** 2, axis=(0, 2))   # (50,)
    mae_per_step = np.mean(np.abs(diff), axis=(0, 2))

    # per-step-per-dim 的 MAE: 对样本平均, 保留 step 和 dim
    # shape = (50, 7), 用于画 7 条维度各自的 MAE-vs-step 曲线
    mae_per_step_per_dim = np.mean(np.abs(diff), axis=0)   # (50, 7)

    # 6. 打印
    print("\n" + "=" * 60)
    print(f"Validation Summary  (samples={num_samples}, horizon={args.horizon}, dim={STATE_DIM})")
    print("=" * 60)
    print(f"Overall MSE: {mse_total:.6f}")
    print(f"\nPer-dimension error:")
    print(f"  {'dim':<6} {'MSE':<12} {'MAE':<12}")
    for i, name in enumerate(ACTION_DIM_NAMES):
        print(f"  {name:<6} {mse_per_dim[i]:<12.6f} {mae_per_dim[i]:<12.6f}")

    # 找出误差仍小于阈值的 horizon 长度 (经验阈值)
    threshold = max(1e-4, mse_per_step[0] * 5)   # 用第 0 步的 5 倍作为参考
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
    # 在柱顶标注数值
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

    fig.suptitle(f"pi05_piper validation on {args.repo_id}  |  samples={num_samples}", fontsize=12)
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
