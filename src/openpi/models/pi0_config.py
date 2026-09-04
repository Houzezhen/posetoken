import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils

if TYPE_CHECKING:
    from openpi.models.pi0 import Pi0


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant = "gemma_2b"          # 2B parameters
    action_expert_variant: _gemma.Variant = "gemma_300m"    # 300M parameters

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None  # type: ignore
    # Pi05 has two differences from Pi0:
    # - the state input is part of the discrete language tokens rather than a continuous input that is part of the suffix
    # - the action expert uses adaRMSNorm to inject the flow matching timestep
    pi05: bool = False
    # This config option is not used directly by the model, but it is read by the ModelTransformFactory.
    discrete_state_input: bool = None  # type: ignore

    # ===== Pose Token State 编码配置 (方案 B) =====
    # 是否启用 PoseVLA 风格的 state 编码 (非均匀分桶 + 欧拉角)
    use_pose_state_encoding: bool = False
    # 分桶边界 pkl 文件路径 (由 compute_pose_bin_stats.py 生成)
    bin_stats_path: str | None = None
    # state 结构描述: {"trans_xyz": [0,1,2], "quat_wxyz": [3,4,5,6], "other": []}
    # 用参数化类型, 否则 tyro 解析 config 时无法匹配 dict 默认值
    pose_split: dict[str, list[int]] | None = None
    # ==============================================

    # ===== DCT 频域分词配置 (pose_token_dct 分支) =====
    # 启用频域分词: 对 state 时间序列中**平移维度 (xyz)** 在时间轴做 DCT, 再对系数分桶
    # (参考 pi0fast)。旋转 (四元数) 不适合频域分析, 退回时域逐帧转欧拉角分桶 360 bin。
    # 低频系数用 256-bin 稠密桶, 高频系数用 64-bin 稀疏桶。
    # 启用后会覆盖 use_pose_state_encoding 的单帧时域编码路径。
    use_dct_pose_encoding: bool = False
    # DCT 时间窗口长度 (帧数), 对应密集连续采样帧数。
    # 当 dct_state_window > 0 时, 与数据 pipeline 取的连续帧数一致,
    # 不再依赖 PIPER_HISTORY_OFFSETS 的稀疏采样。
    dct_window_size: int = 4
    # 密集连续采样帧数 (用于 DCT 频域分析)。设为 0 表示沿用旧行为:
    # 从 history 堆叠里取 dct_window_size 帧 (稀疏, 不推荐用于频域)。
    # 设为 > 0 时, 数据 pipeline 会额外取连续 N 帧 state 专门给 DCT。
    dct_state_window: int = 0
    # 低频系数个数 (前 n_low_freq 个系数用稠密桶, 其余用稀疏桶)。
    # None 表示取 ceil(window/2)
    dct_n_low_freq: int | None = None
    # 平移维度保留的 DCT 系数总数 (只保留前 n_keep_coeffs 个系数, 丢弃更高频).
    dct_n_keep_coeffs: int = 8
    # 平移 DCT 系数的低频/高频桶数 (与 bin_stats pkl 对齐).
    dct_low_bins: int = 256
    dct_high_bins: int = 64
    # 单帧 state 的实际维度 (不含 pad_to_dim 的 padding)。
    # 例: PIPER_HISTORY_STATE_DIM = 7 (xyz + quat_xyzw)。
    # 当 state 经过 PadStatesAndActions 填充到 action_dim 时, DCT tokenizer 用此值
    # 从填充后的向量中取出前 history*dct_state_dim 个值重塑为 (history, dct_state_dim)。
    dct_state_dim: int = 7
    # ==================================================

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

        # 启用 DCT 频域分词时, 强制 discrete_state_input=True 并增大 max_token_len
        if self.use_dct_pose_encoding:
            object.__setattr__(self, "discrete_state_input", True)
            # DCT token 数估算: 平移 n_keep_coeffs*3 + 旋转 window*3 (欧拉角逐帧);
            # 前缀 + 后缀约 30 token。取一个较宽松的上界, 避免频繁截断告警
            if self.max_token_len < 300:
                object.__setattr__(self, "max_token_len", 300)
        # 启用 Pose Token 编码时, 强制 discrete_state_input=True 并增大 max_token_len
        elif self.use_pose_state_encoding:
            object.__setattr__(self, "discrete_state_input", True)
            if self.max_token_len < 300:
                object.__setattr__(self, "max_token_len", 300)

    @property
    @override
    def model_type(self) -> _model.ModelType:
        if self.pi05:
            return _model.ModelType.PI05
        return _model.ModelType.PI0

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0":
        from openpi.models.pi0 import Pi0

        return Pi0(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)   # IMAGE_RESOLUTION = (224, 224)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
            # Pose Token: 新增的 pose embedding 表 (llm/embedder/pose_embedding)
            # 需要保持可训练, 不随 gemma 基础参数一起被冻结。
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*pose_embedding.*")),
            )
        if not filters:
            return nnx.Nothing
        return nnx.All(*filters)
