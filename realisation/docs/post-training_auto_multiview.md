<a id="auto-multiview-post-training-for-hdmap"></a>

# HDMap 自动驾驶多视角后训练
本指南介绍如何使用 Cosmos-Transfer2.5 自动驾驶多视角 2B 模型进行后训练。

**有关自动驾驶多视角模型后训练的更多信息和示例，请参阅 Cosmos Cookbook 中的[自动驾驶多视角后训练](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/post_training/transfer2_5/av_world_scenario_maps/post_training.html)教程。**

<a id="table-of-contents"></a>

## 目录
<!--TOC-->

- [目录](#table-of-contents)
- [前置条件](#prerequisites)
- [1. 准备数据](#1-preparing-data)
  - [1.1 准备 Transfer 多视角训练数据集](#11-prepare-transfer-multiview-training-dataset)
  - [1.2 检查数据集目录格式](#12-verify-the-dataset-folder-format)
- [2. 后训练](#2-post-training)
- [3. 使用后训练检查点进行推理](#3-inference-with-the-post-trained-checkpoint)
  - [3.1 将 DCP 检查点转换为合并的 PyTorch 格式](#31-converting-dcp-checkpoint-to-consolidated-pytorch-format)
  - [3.2 运行推理](#32-running-inference)

<!--TOC-->

<a id="prerequisites"></a>

## 前置条件
开始前，请阅读[后训练指南](./post-training.md)，了解详细配置步骤和重要后训练说明，包括检查点管理与最佳实践，为使用 Cosmos-Transfer2.5 进行后训练做好准备。

<a id="1-preparing-data"></a>

## 1. 准备数据
<a id="11-prepare-transfer-multiview-training-dataset"></a>

### 1.1 准备 Transfer 多视角训练数据集
第一步是准备包含视频的数据集。

需要提供一个包含 **MP4 格式**视频的目录，建议使用 720p，同时提供相应的目录存放 **MP4 格式**的 HDMap 控制输入视频。每个样本的各个视角应进一步按相机名称分入子目录。`assets/multiview_hdmap_posttrain_dataset` 提供了可用的示例数据集。

<a id="12-verify-the-dataset-folder-format"></a>

### 1.2 检查数据集目录格式
数据集目录格式如下：

```
assets/multiview_hdmap_posttrain_dataset/
├── captions/
│   └── ftheta_camera_front_wide_120fov/
│       └── *.json
├── control_input_hdmap_bbox/
│   ├── ftheta_camera_cross_left_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_cross_right_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_wide_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_tele_30fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_left_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_right_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_tele_30fov/
│   │   └── *.mp4
├── videos/
│   ├── ftheta_camera_cross_left_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_cross_right_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_wide_120fov/
│   │   └── *.mp4
│   ├── ftheta_camera_front_tele_30fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_left_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_right_70fov/
│   │   └── *.mp4
│   ├── ftheta_camera_rear_tele_30fov/
│   │   └── *.mp4
```

<a id="2-post-training"></a>

## 2. 后训练
运行以下命令，使用多视角数据执行示例后训练任务：

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train --config=cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/config.py -- experiment=transfer2_auto_multiview_post_train_example job.wandb_mode=disabled
```

模型将使用该多视角数据集进行后训练。数据加载器的定义请参阅[数据配置](../cosmos_transfer2/_src/transfer2_multiview/configs/vid2vid_transfer/defaults/dataloader_local.py)。

检查点保存在 `${IMAGINAIRE_OUTPUT_ROOT}/PROJECT/GROUP/NAME/checkpoints`。`IMAGINAIRE_OUTPUT_ROOT` 默认值为 `/tmp/imaginaire4-output`。强烈建议将 `IMAGINAIRE_OUTPUT_ROOT` 设置为具有足够存储空间的位置。

在上述示例中，`PROJECT` 为 `cosmos_transfer_v2p5`，`GROUP` 为 `auto_multiview`，`NAME` 为 `2b_cosmos_multiview_post_train_example`。

这些值的具体来源请参阅任务配置：

```python
transfer2_auto_multiview_post_train_example = dict(
    dict(
        ...
        job=dict(
            project="cosmos_transfer_v2p5",
            group="auto_multiview",
            name="2b_cosmos_multiview_post_train_example"
        ),
        ...
    )
)
```

<a id="3-inference-with-the-post-trained-checkpoint"></a>

## 3. 使用后训练检查点进行推理
<a id="31-converting-dcp-checkpoint-to-consolidated-pytorch-format"></a>

### 3.1 将 DCP 检查点转换为合并的 PyTorch 格式
训练期间的检查点以 DCP 格式保存，因此推理前需要将其转换为合并的 PyTorch 格式（.pt）。使用 `convert_distcp_to_pt.py` 脚本：

```bash
# Get path to the latest checkpoint
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_transfer_v2p5/auto_multiview/2b_cosmos_multiview_post_train_example/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

# Convert DCP checkpoint to PyTorch format
python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

转换后会创建三个文件：

- `model.pt`：完整检查点，同时包含普通权重和 EMA 权重。
- `model_ema_fp32.pt`：仅包含 float32 精度的 EMA 权重。
- `model_ema_bf16.pt`：仅包含 bfloat16 精度的 EMA 权重，推荐用于推理。

<a id="32-running-inference"></a>

### 3.2 运行推理
转换检查点后，可以使用后训练模型进行推理，并通过 JSON 配置文件指定推理参数，示例见 `assets/multiview_example/multiview_spec.json`。

```bash
export NUM_GPUS=8
torchrun --nproc_per_node=$NUM_GPUS --master_port=12341 -m examples.multiview -i assets/multiview_example/multiview_spec.json -o outputs/postrained-auto-mv --checkpoint_path $CHECKPOINT_DIR/model_ema_bf16.pt --experiment transfer2_auto_multiview_post_train_example
```

生成的视频将保存到输出目录，例如 `outputs/postrained-auto-mv/`。

更多推理选项和高级用法请参阅 [docs/inference.md](./inference.md)。
