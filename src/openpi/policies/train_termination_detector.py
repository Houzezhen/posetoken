"""
TerminationDetector 训练脚本

数据集格式 (HDF5 或 LeRobot):
  每条轨迹包含:
    - observation/image:       (T, H, W, 3) uint8
    - observation/wrist_image: (T, H, W, 3) uint8  (可选)
    - observation/state:       (T, state_dim)       TCP 位姿
    - motion_enhanced:         (T, 10)               由 PiperInputs 预先计算
    - termination_label:       (T,) int{0,1}        1=任务已完成帧, 0=中间帧

标注策略:
  正样本: 轨迹末尾 done_tail_frames 帧 (默认 10)
  负样本: 其余所有帧（含歧义帧）
  类别不平衡通过 pos_weight 处理，无需下采样。

使用方法:
  python -m openpi.policies.train_termination_detector \
      --data_dir /path/to/dataset \
      --output_dir /path/to/checkpoints \
      --epochs 30 --batch_size 32
"""

from __future__ import annotations

import argparse
import logging
import pathlib
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from openpi.policies.termination_detector import TerminationDetector

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class TerminationDataset(Dataset):
    """
    从预处理好的 numpy 文件列表加载。
    每个 .npz 文件对应一条轨迹，包含以下字段:
        base_image:       (T, H, W, 3) uint8
        wrist_image:      (T, H, W, 3) uint8  (可选)
        state:            (T, state_dim)
        motion_enhanced:  (T, 10)
        done_at:          int  任务完成的帧索引 (含)，之后的帧均为正样本
    """

    def __init__(
        self,
        data_dir: str | pathlib.Path,
        done_tail_frames: int = 10,
        image_size: tuple[int, int] = (224, 224),
        augment: bool = True,
    ) -> None:
        self.data_dir = pathlib.Path(data_dir)
        self.done_tail_frames = done_tail_frames
        self.image_size = image_size
        self.augment = augment

        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        # 预先建立 (file_idx, frame_idx) → label 索引
        self._index: list[tuple[int, int, int]] = []  # (file_idx, frame_idx, label)
        self._trajectory_indices: list[list[int]] = []
        for fi, path in enumerate(self.files):
            data = np.load(path, allow_pickle=False)
            T = data["state"].shape[0]
            done_at = int(data["done_at"]) if "done_at" in data else max(0, T - done_tail_frames)
            trajectory_indices = []
            for ti in range(T):
                label = 1 if ti >= done_at else 0
                trajectory_indices.append(len(self._index))
                self._index.append((fi, ti, label))
            self._trajectory_indices.append(trajectory_indices)

        logger.info(
            "Dataset: %d samples (%d pos / %d neg) from %d trajectories",
            len(self._index),
            sum(1 for *_, l in self._index if l == 1),
            sum(1 for *_, l in self._index if l == 0),
            len(self.files),
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        fi, ti, label = self._index[idx]
        data = np.load(self.files[fi], allow_pickle=False)

        base_img = data["base_image"][ti]   # (H, W, 3) uint8
        state    = data["state"][ti].astype(np.float32)
        motion   = data["motion_enhanced"][ti].astype(np.float32)

        wrist_img: Optional[np.ndarray] = None
        if "wrist_image" in data:
            wrist_img = data["wrist_image"][ti]

        # 简单数据增强: 随机水平翻转 (颜色抖动可按需加)
        if self.augment and random.random() < 0.5:
            base_img = np.ascontiguousarray(base_img[:, ::-1])
            if wrist_img is not None:
                wrist_img = np.ascontiguousarray(wrist_img[:, ::-1])

        def to_tensor(img: np.ndarray) -> torch.Tensor:
            # (H,W,3) uint8 → (3,H,W) float32 [0,1]
            t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            if t.shape[-2:] != self.image_size:
                import torch.nn.functional as F
                t = F.interpolate(t.unsqueeze(0), size=self.image_size, mode="bilinear", align_corners=False).squeeze(0)
            return t

        sample = {
            "base_image":      to_tensor(base_img),
            "state":           torch.from_numpy(state),
            "motion_enhanced": torch.from_numpy(motion),
            "label":           torch.tensor(label, dtype=torch.float32),
        }
        if wrist_img is not None:
            sample["wrist_image"] = to_tensor(wrist_img)
        return sample


def _collate(batch: list[dict]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    result = {}
    for k in keys:
        tensors = [b[k] for b in batch]
        result[k] = torch.stack(tensors)
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    data_dir: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    done_tail_frames: int = 10,
    use_wrist_image: bool = True,
    val_split: float = 0.1,
    device: str = "cuda",
) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("Training on %s", dev)

    # 按轨迹分割，避免相邻帧泄漏到验证集
    full_ds = TerminationDataset(data_dir, done_tail_frames=done_tail_frames)
    trajectory_order = list(range(len(full_ds._trajectory_indices)))
    random.Random(42).shuffle(trajectory_order)
    n_val_trajectories = max(1, int(len(trajectory_order) * val_split))
    val_trajectory_ids = set(trajectory_order[:n_val_trajectories])
    train_indices = [i for ti, indices in enumerate(full_ds._trajectory_indices) if ti not in val_trajectory_ids for i in indices]
    val_indices = [i for ti, indices in enumerate(full_ds._trajectory_indices) if ti in val_trajectory_ids for i in indices]
    train_ds = torch.utils.data.Subset(full_ds, train_indices)
    val_ds = torch.utils.data.Subset(full_ds, val_indices)

    # 处理类别不平衡: 对训练集用加权采样器
    train_labels = [full_ds._index[i][2] for i in train_ds.indices]
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Training set must contain both positive and negative samples.")
    w_pos = n_neg / n_pos  # 正样本权重放大
    sample_weights = [w_pos if l == 1 else 1.0 for l in train_labels]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(train_labels), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, sampler=sampler,
        collate_fn=_collate, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=2, pin_memory=True
    )

    # 模型 + 损失 + 优化器
    model = TerminationDetector(use_wrist_image=use_wrist_image).to(dev)
    # pos_weight: 额外补偿训练集中正负比例
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], device=dev)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        # --- 训练 ---
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            base_img = batch["base_image"].to(dev)
            state    = batch["state"].to(dev)
            motion   = batch["motion_enhanced"].to(dev)
            labels   = batch["label"].to(dev)
            wrist    = batch.get("wrist_image")
            wrist    = wrist.to(dev) if wrist is not None else None

            optimizer.zero_grad()
            # 用 logit (去掉 sigmoid) 配合 BCEWithLogitsLoss 更稳定
            vis = model.base_encoder(base_img)
            if model.use_wrist_image and wrist is not None:
                vis = torch.cat([vis, model.wrist_encoder(wrist)], dim=-1)
            elif model.use_wrist_image:
                vis = torch.cat([vis, torch.zeros_like(vis)], dim=-1)
            proprio = model.proprio_enc(torch.cat([state, motion], dim=-1))
            logit = model.fusion(torch.cat([vis, proprio], dim=-1)).squeeze(-1)

            loss = criterion(logit, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        scheduler.step()

        # --- 验证 ---
        model.eval()
        val_loss = 0.0
        tp = fp = tn = fn = 0
        with torch.no_grad():
            for batch in val_loader:
                base_img = batch["base_image"].to(dev)
                state    = batch["state"].to(dev)
                motion   = batch["motion_enhanced"].to(dev)
                labels   = batch["label"].to(dev)
                wrist    = batch.get("wrist_image")
                wrist    = wrist.to(dev) if wrist is not None else None

                probs = model(base_img, state, motion, wrist)
                logits = torch.logit(probs.clamp(1e-6, 1 - 1e-6))
                val_loss += criterion(logits, labels).item()

                preds = (probs >= 0.9).float()
                tp += ((preds == 1) & (labels == 1)).sum().item()
                fp += ((preds == 1) & (labels == 0)).sum().item()
                tn += ((preds == 0) & (labels == 0)).sum().item()
                fn += ((preds == 0) & (labels == 1)).sum().item()

        val_loss /= len(val_loader)
        precision = tp / max(tp + fp, 1)
        recall    = tp / max(tp + fn, 1)
        f1        = 2 * precision * recall / max(precision + recall, 1e-8)

        logger.info(
            "Epoch %3d/%d  train_loss=%.4f  val_loss=%.4f  "
            "prec=%.3f  rec=%.3f  F1=%.3f",
            epoch, epochs, train_loss, val_loss, precision, recall, f1,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt = out / "best.pt"
            model.save(ckpt)
            logger.info("  -> Saved best checkpoint to %s", ckpt)

    # 保存最终 checkpoint
    model.save(out / "final.pt")
    logger.info("Training done. Best val_loss=%.4f", best_val_loss)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train TerminationDetector")
    parser.add_argument("--data_dir",       required=True,  help="目录下包含 .npz 轨迹文件")
    parser.add_argument("--output_dir",     required=True,  help="检查点保存目录")
    parser.add_argument("--epochs",         type=int,   default=30)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--lr",             type=float, default=1e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--done_tail_frames", type=int, default=10,
                        help="轨迹末尾多少帧标记为正样本 (任务已完成)")
    parser.add_argument("--no_wrist",       action="store_true",
                        help="不使用腕部图像 (仅基础视角)")
    parser.add_argument("--val_split",      type=float, default=0.1)
    parser.add_argument("--device",         default="cuda")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        done_tail_frames=args.done_tail_frames,
        use_wrist_image=not args.no_wrist,
        val_split=args.val_split,
        device=args.device,
    )


if __name__ == "__main__":
    main()
