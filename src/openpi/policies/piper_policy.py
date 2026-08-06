import dataclasses
import math
from collections import deque

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model


# History samples are ordered from oldest to current. With a policy request every
# 20 control frames, these offsets give the model about two seconds of context at
# 30 Hz while keeping the existing model input shapes unchanged.
PIPER_HISTORY_STRIDE_FRAMES = 20
PIPER_HISTORY_OFFSETS = (-60, -40, -20, 0)
PIPER_HISTORY_DATASET_KEYS = (
    "observation.state",
    "observation.images.primary",
    "observation.images.wrist",
)
PIPER_HISTORY_INPUT_SPECS = {
    "observation/state": 1,
    "observation/image": 3,
    "observation/wrist_image": 3,
}
PIPER_HISTORY_STATE_DIM = 7


def make_libero_example() -> dict:
    """Creates a random input example for the Libero policy."""
    return {
        "observation/state": np.random.rand(8),
        "observation/image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "prompt": "do something",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


def _make_history_mosaic(frames: list[np.ndarray]) -> np.ndarray:
    """Place chronological frames in a fixed grid without changing channels."""
    if len(frames) == 1:
        return frames[0]

    height, width, channels = frames[0].shape
    if any(frame.shape != (height, width, channels) for frame in frames):
        raise ValueError("All image history frames must have the same shape")

    columns = math.ceil(math.sqrt(len(frames)))
    rows = math.ceil(len(frames) / columns)
    mosaic = np.zeros((rows * height, columns * width, channels), dtype=frames[0].dtype)
    for index, frame in enumerate(frames):
        row, column = divmod(index, columns)
        mosaic[row * height : (row + 1) * height, column * width : (column + 1) * width] = frame
    return mosaic


@dataclasses.dataclass
class PiperInputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """

    # The action dimension of the model. Will be used to pad state and actions for pi0 model (not pi0-FAST).
    # Do not change this for your own dataset.
    action_dim: int

    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType = _model.ModelType.PI0

    # Number of past observations (including the current one) supplied to the
    # policy. A value of 1 preserves the original single-observation behavior.
    history_length: int = 1

    _history_buffers: dict[str, deque[np.ndarray]] = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.history_length < 1:
            raise ValueError("history_length must be at least 1")

    def reset(self) -> None:
        self._history_buffers.clear()

    def _history(self, key: str, value: np.ndarray, base_ndim: int) -> list[np.ndarray]:
        value = np.asarray(value)
        if value.ndim == base_ndim + 1:
            if value.shape[0] != self.history_length:
                raise ValueError(
                    f"{key} has {value.shape[0]} history frames, expected {self.history_length}"
                )
            frames = [np.asarray(frame) for frame in value]
            self._history_buffers[key] = deque((frame.copy() for frame in frames), maxlen=self.history_length)
            return frames
        if value.ndim != base_ndim:
            raise ValueError(
                f"{key} must have {base_ndim} dimensions, or {base_ndim + 1} with history; got {value.shape}"
            )

        buffer = self._history_buffers.setdefault(key, deque(maxlen=self.history_length))
        buffer.append(value.copy())
        frames = list(buffer)
        return [frames[0]] * (self.history_length - len(frames)) + frames

    def __call__(self, data: dict) -> dict:
        if bool(np.asarray(data.get("observation/history_reset", False))):
            self.reset()

        # We pad the proprioceptive input to the action dimension of the model.
        # For pi0-FAST, we don't pad the state. For Libero, we don't need to differentiate
        # since the pi0-FAST action_dim = 7, which is < state_dim = 8, so pad is skipped.
        # Keep this for your own dataset, but if your dataset stores the proprioceptive input
        # in a different key than "observation/state", you should change it below.
        state_history = self._history(
            "observation/state", data["observation/state"], PIPER_HISTORY_INPUT_SPECS["observation/state"]
        )
        # The current state stays first so DeltaActions and AbsoluteActions keep
        # using the current TCP pose. Earlier states fill the remaining slots.
        state_frames = [np.ravel(frame) for frame in state_history]
        if self.history_length > 1:
            if any(frame.shape[0] < PIPER_HISTORY_STATE_DIM for frame in state_frames):
                raise ValueError(f"Piper state history frames must contain at least {PIPER_HISTORY_STATE_DIM} values")
            state_frames = [frame[:PIPER_HISTORY_STATE_DIM] for frame in state_frames]
        state = np.concatenate(list(reversed(state_frames)))
        if state.shape[0] > self.action_dim:
            raise ValueError(
                f"Flattened state history has {state.shape[0]} values, exceeding action_dim={self.action_dim}"
            )
        state = transforms.pad_to_dim(state, self.action_dim)

        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_history = self._history(
            "observation/image", data["observation/image"], PIPER_HISTORY_INPUT_SPECS["observation/image"]
        )
        wrist_history = self._history(
            "observation/wrist_image",
            data["observation/wrist_image"],
            PIPER_HISTORY_INPUT_SPECS["observation/wrist_image"],
        )
        base_image = _make_history_mosaic([_parse_image(frame) for frame in base_history])
        wrist_image = _make_history_mosaic([_parse_image(frame) for frame in wrist_history])
        # 数据集可能没有第三路图像 (target)。有则作为 right_wrist, 无则用零填充并 mask 掉。
        target_image = _parse_image(data["observation/target_image"]) if "observation/target_image" in data else None
        if target_image is not None:
            right_wrist_image = target_image
            right_wrist_mask = np.True_
        else:
            right_wrist_image = np.zeros_like(base_image)
            right_wrist_mask = np.False_  # 缺失的第三路相机永远 mask 掉 (训练/推理一致)
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # Mask any non-existent images with False (if ``mask_padding`` is True).
                "right_wrist_0_rgb": right_wrist_mask,
            },
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        if "actions" in data:
            # We are padding to the model action dim.
            # For pi0-FAST, this is a no-op (since action_dim = 7).
            actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            inputs["actions"] = actions

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class PiperOutputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.
        return {"actions": np.asarray(data["actions"][:, :7])}  
