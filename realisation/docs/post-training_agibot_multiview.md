<a id="agibot-multiview-post-training"></a>

# Agibot 多视角后训练
> **⚠️ 注意**
> 本文介绍的模型和功能目前**仅限内部使用**，尚未公开。这包括文中引用的所有检查点，例如 `Cosmos-Transfer2.5-2B/robot-multiview-control`。
>
> 因此，目前无法在 NVIDIA 外部执行本指南中的步骤。未来可能会开放这些模型；如有变化，本文将相应更新。

本指南介绍如何使用本地数据，对 Cosmos-Transfer2.5 Agibot 三视角控制模型（head_color、hand_left、hand_right）进行后训练，支持 edge、depth、seg 或 vis 控制。

<a id="table-of-contents"></a>

## 目录
<!--TOC-->

- [目录](#table-of-contents)
- [前置条件](#prerequisites)
- [1. 准备数据](#1-preparing-data)
  - [1.1 数据集布局](#11-dataset-layout)
  - [1.2 布局示例](#12-example-layout)
  - [1.3 用于准备后训练数据的示例数据集](#13-example-dataset-for-preparing-post-train-data)
- [2. 后训练](#2-post-training)
  - [2.1 环境变量](#21-environment-variables)
  - [2.2 运行命令](#22-run-command)
  - [2.3 覆盖训练超参数](#23-overriding-training-hyperparameters)
- [3. 另请参阅](#3-see-also)

<!--TOC-->

<a id="prerequisites"></a>

## 前置条件
开始前，请阅读[后训练指南](./post-training.md)，了解环境配置、Hugging Face 配置和检查点管理。此外：

- 设置 `COSMOS_EXPERIMENTAL_CHECKPOINTS=1`，以便正确解析基础检查点位置（例如，在非内部环境中从 Hugging Face 获取）。若未设置，训练会**从头开始**，而不会加载预训练检查点。

<a id="1-preparing-data"></a>

## 1. 准备数据
<a id="11-dataset-layout"></a>

### 1.1 数据集布局
使用一个目录存放以下内容：

- **视频：**`videos/head_color/*.mp4`、`videos/hand_left/*.mp4`、`videos/hand_right/*.mp4`。不同视角使用相同的样本 ID，例如 `episode_000000.mp4`。
- **描述文本：**`captions/head_color/*.json`、`captions/hand_left/*.json`、`captions/hand_right/*.json`。每个 JSON 必须包含 `{"caption": "..."}`。`head_color` 的描述文本用作训练提示词，其他视角可以共享或复制该描述。如有需要，可使用 `scripts/write_agibot_captions.py` 生成占位描述文本。
- **控制信号（可选）：**仅对 **depth** 或 **seg** 控制，需要添加 `control_input_depth/` 或 `control_input_seg/`，其视角子目录和 `.mp4` 文件布局与视频相同，文件主名与 `videos/` 中对应文件一致。对于 **edge** 或 **vis**，控制信号在运行时根据视频计算。

<a id="12-example-layout"></a>

### 1.2 布局示例
```
your_dataset/
├── captions/
│   ├── head_color/
│   │   ├── episode_000000.json
│   │   └── ...
│   ├── hand_left/
│   │   └── ...
│   └── hand_right/
│       └── ...
├── videos/
│   ├── head_color/
│   │   ├── episode_000000.mp4
│   │   └── ...
│   ├── hand_left/
│   │   └── ...
│   └── hand_right/
│       └── ...
└── (optional) control_input_depth/ or control_input_seg/
    ├── head_color/
    ├── hand_left/
    └── hand_right/
```

<a id="13-example-dataset-for-preparing-post-train-data"></a>

### 1.3 用于准备后训练数据的示例数据集
使用 Cosmos-Assets 数据集时，仓库在 **`assets/robot_multiview_control_posttrain-agibot`** 提供了布局示例，可作为准备自有后训练数据的参考：

- **videos/**：三视角视频（head_color、hand_left、hand_right），样本 ID 相互对应。
- **captions/**：每个视角的描述文本 JSON（`{"caption": "..."}`）；head_color 的描述文本用作训练提示词。
- **control_input_depth/** 和 **control_input_seg/**：可选的预计算 depth 和 seg 控制视频，其视角结构与文件主名均与 `videos/` 一致。

对于 **edge** 和 **vis**，控制信号在运行时根据视频计算，因此不需要控制信号目录。文档测试脚本 `tests/docs_test/post-training_agibot_multiview.sh` 使用此数据集运行全部四种控制类型的后训练。

<a id="2-post-training"></a>

## 2. 后训练
<a id="21-environment-variables"></a>

### 2.1 环境变量
```bash
export COSMOS_EXPERIMENTAL_CHECKPOINTS=1
```

<a id="22-run-command"></a>

### 2.2 运行命令
```shell exclude=true
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
  -- \
  experiment=transfer2_agibot_posttrain_edge_example \
  dataloader_train.dataset.dataset_dir=assets/robot_multiview_control_posttrain-agibot \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  job.wandb_mode=disabled
```

将 `/path/to/your/dataset` 替换为实际数据集目录（示例命令将其设置为 `assets/robot_multiview_control_posttrain-agibot`）。

- **实验配置：**根据需要的控制类型，使用 `transfer2_agibot_posttrain_edge_example`、`transfer2_agibot_posttrain_depth_example`、`transfer2_agibot_posttrain_seg_example` 或 `transfer2_agibot_posttrain_vis_example`。
- **数据集路径：**将 `dataloader_train.dataset.dataset_dir` 设置为实际数据集目录，可以是绝对路径，也可以是相对于当前工作目录的路径。除非该目录确实存在，否则不要直接使用字面路径 `datasets/your_dataset`。
- **采样器：**`'dataloader_train.sampler.dataset=${dataloader_train.dataset}'` 确保采样器指向覆盖后的数据集。
- **GPU：**将 `--nproc_per_node` 设置为 GPU 数量，例如 4 或 8。

<a id="23-overriding-training-hyperparameters"></a>

### 2.3 覆盖训练超参数
可以在命令行中覆盖以下任意参数，与上面的参数放在同一个 `--` 参数块之后：

| 参数 | 覆盖键 | 示例 | 后训练默认值 |
|-------|--------------|---------|----------------------|
| 最大迭代次数 | `trainer.max_iter` | `trainer.max_iter=30000` | 5_000 |
| 日志记录间隔 | `trainer.logging_iter` | `trainer.logging_iter=100` | 50 |
| 学习率 | `optimizer.lr` | `optimizer.lr=1e-4` | 5e-5（来自基础配置） |
| 检查点保存间隔 | `checkpoint.save_iter` | `checkpoint.save_iter=1000` | 500 |
| 学习率预热步数 | `scheduler.warm_up_steps` | `scheduler.warm_up_steps=[500]` | [1000]（来自基础配置） |
| 学习率周期长度 | `scheduler.cycle_lengths` | `scheduler.cycle_lengths=[10000]` | [400000]（来自基础配置） |

自定义学习率和迭代次数的示例：

```shell exclude=true
torchrun --nproc_per_node=8 -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py \
  -- \
  experiment=transfer2_agibot_posttrain_edge_example \
  dataloader_train.dataset.dataset_dir=assets/robot_multiview_control_posttrain-agibot \
  'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
  trainer.max_iter=10000 \
  optimizer.lr=1e-4 \
  job.wandb_mode=disabled
```

<a id="3-see-also"></a>

## 3. 另请参阅
- [推理：机器人多视角控制（Agibot）](./inference_robot_multiview_control.md)：使用相同模型进行推理。
- [后训练指南](./post-training.md)：通用配置与检查点管理。
