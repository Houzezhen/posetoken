## CAN激活
```bash
mamba activate lerobot
cd lerobot_xiazhi
bash piper_scripts/can_activate.sh  # CAN激活
```


## Camer_index修改
打开lerobot/common/robot_devices/robots/configs.py
修改camera_index

```python
@RobotConfig.register_subclass("piper")
@dataclass
class PiperRobotConfig(RobotConfig):
    inference_time: bool
    
    follower_arm: dict[str, MotorsBusConfig] = field(
        default_factory=lambda: {
            "main": PiperMotorsBusConfig(
                can_name="can_left",
                motors={
                    # name: (index, model)
                    "joint_1": [1, "agilex_piper"],
                    "joint_2": [2, "agilex_piper"],
                    "joint_3": [3, "agilex_piper"],
                    "joint_4": [4, "agilex_piper"],
                    "joint_5": [5, "agilex_piper"],
                    "joint_6": [6, "agilex_piper"],
                    "gripper": (7, "agilex_piper"),
                },
            ),
        }
    )

    cameras: dict[str, CameraConfig] = field(
        default_factory=lambda: {
            "primary": OpenCVCameraConfig(
                # camera_index=10,
                camera_index=6, 
                # camera_index=22,
                fps=30,
                width=640,
                height=480,
            ),
            "wrist": OpenCVCameraConfig(
                # camera_index=22,
                camera_index=8,
                fps=30,
                width=640,
                height=480,
            ),
            # "front": OpenCVCameraConfig(
            #     camera_index=10,
            #    # camera_index=28,
            #     fps=30,
            #     width=640,
            #     height=480,
            # ),
        }
    )
```




## CAN激活
```bash
mamba activate lerobot
cd lerobot_xiazhi
bash piper_scripts/can_activate.sh  # CAN激活
```


## 数据采集

cd ~/lerobot_xiazhi/lerobot/common/robot_devices/motors
gedit piper.py
# Comment out the following lines when teleop is used

```bash


python lerobot/scripts/control_robot.py \
    --robot.type=piper \
    --robot.inference_time=false \
    --control.type=record \
    --control.fps=30 \
    --control.single_task="pick up the bottles one by one and place the bottles from right to left" \ # 数据集语言指令
    --control.num_episodes=30 \ # 录制数量; 
    --control.warmup_time_s=2 \ 
    --control.episode_time_s=90 \ # 每一条录制时间（可以提早结束）
    --control.reset_time_s=10 \ # 右键后间隔时间
    --control.play_sounds=true \
    --control.push_to_hub=false \
    --control.repo_id test/0912
```

```bash
'''sink to table'''  
- Default folder: 
 cd /home/user/.cache/huggingface/lerobot/test/

 bash piper_scripts/can_activate.sh
 
- prompt:  pick up the bottles one by one and place the bottles from sink to table

python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="pick up the bottles one by one and place the bottles from sink to table" --control.num_episodes=25 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_sink2table_left_0120

python lerobot/scripts/visualize_dataset_html.py --repo-id test/data_sink2table_left_0120
python lerobot/scripts/visualize_dataset_html.py --repo-id test/data_sink2table_0429
    
## 以上命令中，control.num_episodes设置为25，每种bottle的左右分别采集25条，共50条数据。

```


##  Press the toilet button
python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="press the toilet button" --control.num_episodes=25 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_toiletButton_01_0605


data_toiletButton_01_0605
data_toiletButton_02_0605
data_toiletButton_03_0605
data_toiletButton_04_0605



python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=true --control.type=replay --control.fps=30 --control.episode=0 --control.repo_id=test/data_toiletButton_test


python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="press the toilet button" --control.num_episodes=25 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_toiletButton_01_0603

python lerobot/scripts/visualize_dataset_html.py --repo-id test/data_toiletButton_0603





python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="clean the countertop" --control.num_episodes=30 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_countertop_left_0106
## 以上命令中，control.num_episodes设置为30，清洁countertop左右分别采集30条，共60条数据。




python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="pick up the bottles one by one and place the bottles from sink to table" --control.num_episodes=40 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_sink2table_121101

python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="pick up the bottles one by one and place the bottles from sink to table" --control.num_episodes=20 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_sink2table_0429

python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="pick up the bottles one by one and place the bottles from sink to table" --control.num_episodes=10 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_sink2table_0401


python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="pick up the bottles one by one and place the bottles from sink to table" --control.num_episodes=10 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_sink2table_0409

python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="pick up the bottles one by one and place the bottles from sink to table" --control.num_episodes=10 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_sink2table_0411

以上命令中，control.num_episodes设置为40，收集xx条数据。



```
python -m lebot.scripts.control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="clean the sink" --control.num_episodes=10 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_cleanSink_1113

### 以上命令中，control.num_episodes设置为10，
```

```bash
'''wipe the countertop'''  
- Default folder: 
 cd /home/user/.cache/huggingface/lerobot/test/
 
- prompt:  wipe the countertop

python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="wipe the countertop" --control.num_episodes=30 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_wipecountertop_1110

## 以上命令中，control.num_episodes设置为30，我收集30条数据。

```

```bash
''' pick up the brush and clean the stains on the countertop'''  
- Default folder: 
 cd /home/user/.cache/huggingface/lerobot/test/
 
- prompt:  wipe the countertop

python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="pick up the brush and clean the stains on the countertop" --control.num_episodes=10 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_brushstains_011801

## 以上命令中，control.num_episodes设置为10，每次收集10条数据。

python lerobot/scripts/visualize_dataset_html.py --repo-id /home/user/data/data_wipecountertop_left_success_041601

data_wipecountertop_left_success_041601
data_wipecountertop_left_success_041603


python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=false --control.type=record --control.fps=30 --control.single_task="wipe the countertop" --control.num_episodes=15 --control.warmup_time_s=2 --control.episode_time_s=90 --control.reset_time_s=10 --control.play_sounds=true --control.push_to_hub=false --control.repo_id test/data_wipecountertop_0515_left

```




```bash  Visualize data stored on a local machin 检查数据
python lerobot/scripts/visualize_dataset_html.py --repo-id test/data_toiletButton_0605


python lerobot/scripts/visualize_dataset_html.py --repo-id test/data_sink2table_0429

python lerobot/scripts/visualize_dataset_html.py --repo-id test/data_wipecountertop_left_success_041601

python lerobot/scripts/visualize_dataset_html.py --repo-id /home/user/data/data_toiletButton_0605


python lerobot/scripts/visualize_dataset_html.py --repo-id /home/user/data/mergedData_wipecountertop_left_success/data_wipecountertop_left_success_merged

# lebot采数据
python -m lerobot.scripts.visualize_dataset_html --repo-id test/data_sink2table_0330



python -m lerobot.scripts.visualize_dataset_html --repo-id test/data_sink2table_0401
python -m lerobot.scripts.visualize_dataset_html --repo-id test/data_sink2table_041101


### lerobot_force
python -m lerobot.scripts.visualize_dataset_html --repo-id test/data_wipecountertop_left_merged


python lerobot/scripts/visualize_dataset.py --repo-id test/data_sink2table_0509 --episode-index 0

http://localhost:9090      # 查看数据是否正确

```

## Replay
python lerobot/scripts/control_robot.py --robot.type=piper --robot.inference_time=true --control.type=replay --control.fps=30 --control.episode=0 --control.repo_id=test/data_sink2table_0509


```bash

- 合并数据
python lerobot/scripts/merge.py --sources /home/user/data/data_toiletButton_0603 /home/user/data/data_toiletButton_0605  --output /home/user/data/data_toiletButton_0603_merged_0605/  --max_dim 7

python lerobot/scripts/merge.py --sources /home/demo/data/mergedData/data_sink2table_1015_01/ /home/demo/data/mergedData/data_sink2table_1015_02/ /home/demo/data/mergedData/data_sink2table_1015_03/ --output /home/demo/data/mergedData/data_sink2table_1015/ --max_dim 7



python lerobot/scripts/merge.py --sources /home/user/data/data_sink2table_left_012001 /home/user/data/data_sink2table_right_012001 /home/user/data/data_sink2table_left_012002 /home/user/data/data_sink2table_right_012002 --output /home/user/data/data_sink2table_0120 --max_dim 7

    

python lerobot/scripts/merge.py --sources /home/user/data/mergedData1211/data_sink2table_bacter_left_1211 /home/user/data/mergedData1211/data_sink2table_bacter_right_1211 /home/user/data/mergedData1211/data_sink2table_foaming_left_1211 /home/user/data/mergedData1211/data_sink2table_foaming_right_1211 /home/user/data/mergedData1211/data_sink2table_left_1211 /home/user/data/mergedData1211/data_sink2table_right_1211 --output /home/user/data/mergedData1211/data_sink2table_3bottles_1211/ --max_dim 7



python lerobot/scripts/merge.py --sources /home/user/data/mergedData/data_sink2table_bothside_1105 /home/user/data/mergedData/data_sink2table_bothside_1203 --output /home/user/data/mergedData/data_sink2table_bothside_merged/ --max_dim 7


python lerobot/scripts/merge.py --sources /home/user/data/mergedData/data_sink2table_bothside_1105 /home/user/data/mergedData/data_sink2table_bothside_1203 --output /home/user/data/mergedData/data_sink2table_bothside_merged/ --max_dim 7



python lerobot/scripts/merge.py --sources /home/user/data/data_wipecountertop_left_success_041601 /home/user/data/data_wipecountertop_left_success_041603 --output /home/user/data/data_wipecountertop_left_merged/ --max_dim 7


python lerobot/scripts/merge.py --sources /home/user/data/data_wipecountertop_0515_left/ /home/user/data/data_wipecountertop_0515_right/ --output /home/user/data/data_wipecountertop_merged_0515/ --max_dim 7



```






```bash
**注意**：您可以使用键盘来控制数据记录流程。
- 在录制一集时，按右箭头键“->”可提前退出并进入环境重置。
- 在重置环境时，按右箭头键“->”可提前退出并开始录制下一集。
- 按左箭头键“<-”可提前退出并重新录制当前集。
- 按“esc”键可停止数据记录。
这可能需要超级用户权限，以允许您的终端监控键盘事件。

**注意**：您可以通过运行相同的数据记录命令并添加 `--control.resume=true` 来恢复/继续数据记录。


在录制剧集期间，随时按下右箭头（->），可以提前停止并进入重置阶段。在重置阶段，按下右箭头（->）也可以提前停止并进入下一集录制。
在录制剧集或重置期间，随时按下左箭头（<-），可以提前停止、取消当前剧集，然后重新录制。
'''
```


# 计算数据的标准统计
```bash

1. 修改openpi下的/home/user/openpi05/src/openpi/training/config.py，将pi05_piper的repo_id设置为当前要计算的数据目录。比如：
   repo_id="/home/user/.cache/huggingface/lerobot/test/data_sink2table_bothside", 

   repo_id="/home/user/.cache/huggingface/lerobot/test/data_wipecountertop_1110",

   repo_id="/home/user/.cache/huggingface/lerobot/test/data_cleanSink_1113",

   data_countertop_cola_left_0106

   
2. 计算Normalized Statistics
JAX_PLATFORM_NAME=cpu uv run scripts/compute_norm_stats.py --config-name pi05_piper

    
'''
'''


# 训练
```bash
--overwrite 覆盖；   --resume 继续训练

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=wipecountertop  

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=cleanSink  


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=test  

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=sink2table_1203  

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=sink2table_80_1205  


XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 uv run scripts/train.py pi05_piper --exp-name=toiletButton_0603
  
XLA_PYTHON_CLIENT_MEM_FRACTION=0.7 uv run scripts/train.py pi05_piper --exp-name=toiletButton_0605  

3 bottles 数据在pi05_piper模型上fiinetune

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=sink2table_3bottles_180FTpi05base_1212 --overwrite


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=clean_countertop_cola_0106 

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=brushstains_0118


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=sink2table_1bottle_0120 

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=sink2table_1bottle_0125

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=data_cleanSink_test

XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=sink2table_180FT241_1219_0330



XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_piper --exp-name=wipecountertop_0515 --resume

wandb API Key:
zengfanyu:
1448c591ccd16c572fbe607a1e2edb159cfbd741

lirenjun:
97f1dfdec185312b228904f4b8ccac6f5e28d634
'''
'''


# Openpi05 部署
```bash
- 5880
uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/toiletButton_0603/25000
```

uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/.cache/openpi/openpi-assets/checkpoints/pi05_base

uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/cleanSink/10000

uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/sink2table_1203/14000


uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/sink2table_80finetune241_1206/15000

uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_noforce/pi05_noforce/5000

uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/sink2table_3bottles_180FTpi05base_1212/29999


uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/clean_countertop_cola_0106/14000

uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/wipecountertop_0515/29999





uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/brushstains_0118/15000



uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/sink2table_1bottle_0125/16000



uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/sink2table_1bottle_0125/16000


# grasp various bottles
uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/sink2table_1bottle_0125/16000


uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/sink2table_bothside/29999



uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/sink2table_180FT241_1219_0330/10000

 uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_force_lora_finetune --policy.dir=/home/user/openpi05_force/checkpoints/pi05_force_lora_finetune/pi05_force/10000


uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/data_sink2table_0509_FT241_0513/5000

uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi05_piper --policy.dir=/home/user/openpi05/checkpoints/pi05_piper/sink2table_0507/14999




```bash

cd ~/lerobot_xiazhi/lerobot/common/robot_devices/motors
gedit piper.py
# Uncomment the following lines when manipulation is conducted.



# ------------------------------------------------------------------------------


# lerobot
python -m lerobot.scripts.run_piper_with_openpi_EMA --host=localhost --port=8000 --device=cuda --n_action_steps 40


python lerobot/scripts/run_piper_with_openpi_EMA.py     --host=localhost     --port=8000     --device=cuda --n_action_steps 40



# lebot
python -m lerobot.scripts.run_piper_with_openpi_EMA --host=localhost --port=8000 --device=cuda --n_action_steps 40

python -m lerobot.scripts.run_piper_with_openpi_impedance --host=localhost --port=8000 --device=cuda --n_action_steps 40

python -m lerobot.scripts.run_piper_with_openpi_impedance_EMA --host=localhost --port=8000 --device=cuda --n_action_steps 40


python -m lerobot.scripts.run_piper_with_openpi_force  --host=localhost --port=8000 --device=cuda --n_action_steps 40

python lerobot/scripts/run_piper_with_openpi.py     --host=localhost     --port=8000     --device=cuda --n_action_steps 40


##### pi0部署 #####
```bash
- 5880
conda activte pi05
uv run --python python3.11 scripts/serve_policy.py policy:checkpoint --policy.config=pi0_piper --policy.dir=/home/user/checkpoints/27000

```bash
conda activate lerobot
python lerobot/scripts/run_piper_with_openpi.py     --host=localhost     --port=8000     --device=cuda --n_action_steps 40

