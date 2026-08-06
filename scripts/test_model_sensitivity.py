"""测试模型对输入的敏感度：用随机噪声图片代替真实图片，看输出是否崩掉"""
import numpy as np
import jax
from openpi.training import config as _config
from openpi.policies import policy_config as _policy_config
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="/home/user/openpi05/checkpoints/pi05_piper/data_toiletButton_0609_quaternion/29999")
    parser.add_argument("--sample-idx", type=int, default=100)
    args = parser.parse_args()

    print("加载模型...")
    train_config = _config.get_config("pi05_piper")
    policy = _policy_config.create_trained_policy(train_config, args.checkpoint)

    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    action_std = data_config.norm_stats["actions"].std[:7]
    action_mean = data_config.norm_stats["actions"].mean[:7]

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(data_config.repo_id)
    raw = ds[args.sample_idx]

    state = raw["observation.state"].numpy()
    true_action = np.asarray(raw["action"])[:7]
    image = raw["observation.images.primary"].numpy()
    wrist = raw["observation.images.wrist"].numpy()

    print(f"\n真实 action: {true_action.round(4)}")
    print(f"state:       {state[:7].round(4)}")

    # 测试1：正常输入
    obs_normal = {
        "observation/state": state,
        "observation/image": image,
        "observation/wrist_image": wrist,
        "prompt": "push the button",
    }
    out = policy.infer(obs_normal)
    print(f"\n=== 测试1: 正常图片和 state ===")
    print(f"输出: {out['actions'][0, :7].round(4)}")
    print(f"误差: {(out['actions'][0, :7] - true_action).round(4)}")

    # 测试2：随机噪声图片，正常 state
    obs_noise_img = {
        "observation/state": state,
        "observation/image": np.random.randint(0, 256, image.shape, dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 256, wrist.shape, dtype=np.uint8),
        "prompt": "push the button",
    }
    out2 = policy.infer(obs_noise_img)
    err2 = out2['actions'][0, :7] - true_action
    bias2 = err2 / (action_std + 1e-6)
    print(f"\n=== 测试2: 随机噪声图片 + 正常 state ===")
    print(f"输出: {out2['actions'][0, :7].round(4)}")
    print(f"误差: {err2.round(4)}")
    print(f"模型归一化偏差: {bias2.round(2)}")

    # 测试3：全黑图片，正常 state
    obs_black = {
        "observation/state": state,
        "observation/image": np.zeros(image.shape, dtype=np.uint8),
        "observation/wrist_image": np.zeros(wrist.shape, dtype=np.uint8),
        "prompt": "push the button",
    }
    out3 = policy.infer(obs_black)
    err3 = out3['actions'][0, :7] - true_action
    bias3 = err3 / (action_std + 1e-6)
    print(f"\n=== 测试3: 全黑图片 + 正常 state ===")
    print(f"输出: {out3['actions'][0, :7].round(4)}")
    print(f"误差: {err3.round(4)}")
    print(f"模型归一化偏差: {bias3.round(2)}")

    # 测试4：正常图片，state 偏移一点
    state_shifted = state.copy()
    state_shifted[:7] = state[:7] + np.array([5, 0, 5, 0.1, 0.1, 0.1, 0.1])
    obs_shifted_state = {
        "observation/state": state_shifted,
        "observation/image": image,
        "observation/wrist_image": wrist,
        "prompt": "push the button",
    }
    out4 = policy.infer(obs_shifted_state)
    err4 = out4['actions'][0, :7] - true_action
    bias4 = err4 / (action_std + 1e-6)
    print(f"\n=== 测试4: 正常图片 + state 偏移 [5,0,5,0.1,0.1,0.1,0.1] ===")
    print(f"输出: {out4['actions'][0, :7].round(4)}")
    print(f"误差: {err4.round(4)}")
    print(f"模型归一化偏差: {bias4.round(2)}")

    # 测试5：正常图片，state 大幅偏移
    state_big_shift = state.copy()
    state_big_shift[:7] = state[:7] + np.array([20, 5, 20, 0.5, 0.5, 0.5, 0.3])
    obs_big_shift = {
        "observation/state": state_big_shift,
        "observation/image": image,
        "observation/wrist_image": wrist,
        "prompt": "push the button",
    }
    out5 = policy.infer(obs_big_shift)
    err5 = out5['actions'][0, :7] - true_action
    bias5 = err5 / (action_std + 1e-6)
    print(f"\n=== 测试5: 正常图片 + state 大幅偏移 [20,5,20,0.5,0.5,0.5,0.3] ===")
    print(f"输出: {out5['actions'][0, :7].round(4)}")
    print(f"误差: {err5.round(4)}")
    print(f"模型归一化偏差: {bias5.round(2)}")

    print(f"\n=== 总结 ===")
    print(f"如果噪声/偏移图片导致归一化偏差暴增，说明模型对输入分布非常敏感")
    print(f"这就是真实机器人推理时输出异常的原因")


if __name__ == "__main__":
    main()
