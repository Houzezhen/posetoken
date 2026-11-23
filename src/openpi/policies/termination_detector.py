"""
TerminationDetector: 轻量级任务终止检测器

解决机械臂按按钮时"视觉歧义导致提前返回"的问题。
融合视觉特征、本体感知状态和运动趋势三路信号，作为主 policy 之外的独立验证器。

推理时集成方式:
    detector = TerminationDetector.load(checkpoint_path)
    if policy_wants_to_return and detector(obs) > threshold:
        return_to_init()
    else:
        continue_task()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

if TYPE_CHECKING:
    from pathlib import Path


# 输入维度常量
_STATE_DIM = 7       # TCP 位姿: xyz + quat(4)
_MOTION_ENH_DIM = 10  # 扩展 motion: 6D + cumulative(2) + inst_norm(2)
_IMAGE_FEAT_DIM = 512  # 视觉编码器输出 (MobileNetV3-Small)


class _ImageEncoder(nn.Module):
    """轻量级视觉编码器，基于 MobileNetV3-Small。"""

    def __init__(self, out_dim: int = _IMAGE_FEAT_DIM) -> None:
        super().__init__()
        try:
            from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
            backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        except ImportError:
            from torchvision.models import mobilenet_v3_small
            backbone = mobilenet_v3_small(pretrained=True)

        # 去掉分类头，保留特征提取部分
        self.features = backbone.features
        self.avgpool = backbone.avgpool
        feat_dim = backbone.classifier[0].in_features  # 576
        self.proj = nn.Linear(feat_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, H, W) float32, 已归一化到 [0, 1]
        Returns:
            (B, out_dim)
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.proj(x)


class TerminationDetector(nn.Module):
    """
    融合视觉、本体感知和运动趋势的二分类器。

    正样本: 任务真正完成后的末尾帧
    负样本: 任务中间帧，包括视觉上"看起来像完成"的歧义帧

    输出 sigmoid 概率，> threshold 时允许返回初始位置。
    """

    def __init__(
        self,
        state_dim: int = _STATE_DIM,
        motion_enhanced_dim: int = _MOTION_ENH_DIM,
        image_feat_dim: int = _IMAGE_FEAT_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        use_wrist_image: bool = True,
    ) -> None:
        super().__init__()
        self._state_dim = state_dim
        self._motion_enhanced_dim = motion_enhanced_dim
        self._image_feat_dim = image_feat_dim
        self._hidden_dim = hidden_dim
        self._dropout = dropout
        self.use_wrist_image = use_wrist_image

        # 视觉编码器 (基础视角 + 可选腕部视角)
        self.base_encoder = _ImageEncoder(image_feat_dim)
        if use_wrist_image:
            self.wrist_encoder = _ImageEncoder(image_feat_dim)
            visual_dim = image_feat_dim * 2
        else:
            visual_dim = image_feat_dim

        # 本体感知 + 运动特征编码
        proprio_dim = state_dim + motion_enhanced_dim
        self.proprio_enc = nn.Sequential(
            nn.Linear(proprio_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 128),
        )

        # 融合 MLP
        fused_dim = visual_dim + 128
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        base_image: torch.Tensor,
        state: torch.Tensor,
        motion_enhanced: torch.Tensor,
        wrist_image: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            base_image:      (B, 3, H, W) float32 [0,1]
            state:           (B, state_dim)
            motion_enhanced: (B, motion_enhanced_dim)
            wrist_image:     (B, 3, H, W) float32 [0,1]，可选

        Returns:
            (B,) sigmoid 概率，接近 1.0 表示"任务真正完成"
        """
        # 视觉特征
        vis = self.base_encoder(base_image)
        if self.use_wrist_image and wrist_image is not None:
            vis = torch.cat([vis, self.wrist_encoder(wrist_image)], dim=-1)
        elif self.use_wrist_image:
            # 推理时没有腕部图像时用零填充
            vis = torch.cat([vis, torch.zeros_like(vis)], dim=-1)

        # 本体感知 + 运动特征
        proprio = self.proprio_enc(torch.cat([state, motion_enhanced], dim=-1))

        # 融合并预测
        logit = self.fusion(torch.cat([vis, proprio], dim=-1)).squeeze(-1)
        return torch.sigmoid(logit)

    def predict(
        self,
        base_image: np.ndarray,
        state: np.ndarray,
        motion_enhanced: np.ndarray,
        wrist_image: np.ndarray | None = None,
        threshold: float = 0.9,
    ) -> tuple[bool, float]:
        """
        推理接口，接受 numpy 输入，返回 (is_done, probability)。

        Args:
            base_image:      (H, W, 3) uint8
            state:           (state_dim,)
            motion_enhanced: (motion_enhanced_dim,)
            wrist_image:     (H, W, 3) uint8，可选
            threshold:       判定为"完成"的概率阈值

        Returns:
            is_done:     是否允许返回初始位置
            probability: 完成概率
        """
        self.eval()
        device = next(self.parameters()).device

        def _prep_image(img: np.ndarray) -> torch.Tensor:
            t = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0  # (3,H,W)
            return t.unsqueeze(0).to(device)  # (1,3,H,W)

        with torch.no_grad():
            b_img = _prep_image(base_image)
            w_img = _prep_image(wrist_image) if wrist_image is not None else None
            s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(device)
            m = torch.from_numpy(motion_enhanced.astype(np.float32)).unsqueeze(0).to(device)
            prob = self(b_img, s, m, w_img).item()

        return prob >= threshold, prob

    def save(self, path: str | Path) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self._config()}, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "TerminationDetector":
        ckpt = torch.load(str(path), map_location="cpu")
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        return model

    def _config(self) -> dict:
        return {
            "state_dim": self._state_dim,
            "motion_enhanced_dim": self._motion_enhanced_dim,
            "image_feat_dim": self._image_feat_dim,
            "hidden_dim": self._hidden_dim,
            "dropout": self._dropout,
            "use_wrist_image": self.use_wrist_image,
        }


class TerminationGate:
    """要求连续多帧满足完成条件后才允许策略返回。"""

    def __init__(self, detector: TerminationDetector, threshold: float = 0.9, confirm_frames: int = 5) -> None:
        if confirm_frames < 1:
            raise ValueError("confirm_frames must be at least 1")
        self.detector = detector
        self.threshold = threshold
        self.confirm_frames = confirm_frames
        self._positive_frames = 0

    def reset(self) -> None:
        self._positive_frames = 0

    def update(
        self,
        base_image: np.ndarray,
        state: np.ndarray,
        motion_enhanced: np.ndarray,
        wrist_image: np.ndarray | None = None,
    ) -> tuple[bool, float]:
        """返回 (confirmed_done, probability)，每次调用对应一个控制周期。"""
        _, probability = self.detector.predict(
            base_image, state, motion_enhanced, wrist_image, threshold=self.threshold
        )
        if probability >= self.threshold:
            self._positive_frames += 1
        else:
            self._positive_frames = 0
        return self._positive_frames >= self.confirm_frames, probability
