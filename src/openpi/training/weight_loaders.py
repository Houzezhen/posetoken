import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*lora.*")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            ref_v = flat_ref[k]
            # ===== Pose Token: 处理 embedding 扩展 =====
            # 当从预训练 checkpoint 恢复时, 旧 checkpoint 的 embedding 行数 < 新模型行数。
            # 用已有 embedding 的均值初始化新增的 pose token embedding, 保证向后兼容。
            k_str = "/".join(k) if isinstance(k, tuple) else str(k)
            if (
                "input_embedding" in k_str
                and hasattr(v, "shape")
                and hasattr(ref_v, "shape")
                and v.shape != ref_v.shape
                and v.ndim == 2
                and ref_v.ndim == 2
                and v.shape[1] == ref_v.shape[1]
                and v.shape[0] < ref_v.shape[0]
            ):
                mean_emb = v.mean(axis=0, keepdims=True)
                num_new = ref_v.shape[0] - v.shape[0]
                extended_v = np.concatenate([v, np.tile(mean_emb, (num_new, 1))], axis=0)
                result[k] = extended_v.astype(ref_v.dtype)
                logger.info(
                    "Extended embedding '%s' from %s to %s (mean initialization for %d new tokens)",
                    k_str,
                    v.shape,
                    ref_v.shape,
                    num_new,
                )
                continue
            # ============================================
            result[k] = v.astype(ref_v.dtype) if v.dtype != ref_v.dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    # ===== Pose Token: checkpoint 缺失独立 pose 表 → 保留 ref 结构占位 =====
    # 拆分方案下 pose_embedding 是新参数, 预训练/旧 checkpoint 中不存在。
    # 这里用 ref (通常为 ShapeDtypeStruct) 占位保证结构与模型一致; 训练代码
    # 后续会移除占位并回退到 train_state 的随机初始化 (pose 表不被冻结, 可正常训练)。
    for k, ref_v in flat_ref.items():
        if k in result:
            continue
        k_str = "/".join(k) if isinstance(k, tuple) else str(k)
        if "pose_embedding" in k_str:
            result[k] = ref_v
            logger.info(
                "[weight loader] Pose embedding '%s' missing from checkpoint; using train-state init (%s)",
                k_str,
                getattr(ref_v, "shape", None),
            )
    # ============================================

    return flax.traverse_util.unflatten_dict(result, sep="/")
