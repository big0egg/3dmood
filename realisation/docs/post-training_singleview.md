<a id="post-training-cosmos-transfer2-models-with-local-single-view-data"></a>

# 使用本地单视角数据对 Cosmos-Transfer2 模型进行后训练
本指南介绍如何使用自有本地视频数据，对 Cosmos-Transfer2 模型进行后训练。

<a id="table-of-contents"></a>

## 目录
<!--TOC-->

- [目录](#table-of-contents)
- [前置条件](#prerequisites)
- [支持的模型](#model-support)
- [控制模态](#control-modalities)
- [快速开始](#quick-start)
  - [步骤 1：准备数据集](#step-1-prepare-dataset)
  - [步骤 2：设置环境](#step-2-set-environment)
  - [步骤 3：开始训练](#step-3-start-training)
- [自定义数据准备](#custom-data-preparation)
  - [1. 准备自有视频和描述文本](#1-prepare-your-own-videos-and-captions)
  - [2. 检查数据集目录格式](#2-verify-dataset-folder-format)
  - [3. 预计算控制输入（可选）](#3-optional-pre-compute-control-inputs)
  - [4. 最终检查](#4-final-verification)
- [训练配置](#training-configuration)
  - [关键参数](#key-parameters)
  - [通过命令行自定义](#customizing-via-command-line)
- [启动训练](#launch-training)
  - [1. 运行训练](#1-run-training)
- [监控](#monitoring)
  - [生成样本](#sample-generations)
  - [训练耗时（2000 次迭代）](#training-time-2000-iterations)
- [检查点管理](#checkpoint-management)
  - [将 DCP 转换为 PyTorch 格式](#convert-dcp-to-pytorch)
  - [目录结构](#directory-structure)
  - [恢复训练](#resume-training)
- [推理](#inference)
- [常见问答](#faq)
  - [从哪种控制类型开始？](#which-control-type-to-start-with)
  - [需要多少数据？](#how-much-data-do-i-need)
  - [GPU 显存不足怎么办？](#gpu-out-of-memory)
  - [`context_parallel_size` 是什么？](#what-is-context_parallel_size)
  - [如何判断训练是否正常？](#how-to-know-if-training-is-working)
- [故障排查](#troubleshooting)
  - [出现 "CUDA out of memory"](#cuda-out-of-memory)
  - [出现 "Video has only X frames, need Y"](#video-has-only-x-frames-need-y)
  - [训练损失不下降](#training-loss-not-decreasing)
  - [生成视频存在伪影](#generated-videos-have-artifacts)
- [补充资源](#additional-resources)
- [引用](#citation)

<!--TOC-->

<a id="prerequisites"></a>

## 前置条件
1. **环境**：开始前，请阅读[后训练指南](./post-training.md)，了解详细配置步骤和重要后训练说明，包括检查点管理与最佳实践，为使用 Cosmos-Transfer2.5 进行后训练做好准备。
2. **硬件**：2B 模型需要 8 块 H100/A100（80GB）。
3. **存储**：为数据集、检查点和输出预留足够空间。
4. **缓存配置**：对于磁盘空间有限的容器，请通过环境变量重定向缓存位置。

<a id="model-support"></a>

## 支持的模型
| 模型 | 参数量 | 控制类型 | GPU 要求 |
|-------|------------|---------------|------------------|
| **Cosmos-Transfer2-2B** | 2B | edge、depth、seg、vis | 8 块 H100/A100 80GB |

**建议**：从 2B 模型和 edge 控制开始，无需预处理。

<a id="control-modalities"></a>

## 控制模态
| 控制类型 | 预处理 | 适用场景 |
|---------|--------------|----------|
| **edge** ⚡ | 无，实时计算 | 快速入门、通用场景 |
| **vis** ⚡ | 无，实时计算 | 风格迁移、去噪 |
| **depth** | VideoDepthAnything | 三维感知、机器人 |
| **seg** | SAM2 | 物体控制、合成 |

**所有控制类型使用相同默认值**：`state_t=24`、`num_frames=93`、`context_parallel_size=8`。

<a id="quick-start"></a>

## 快速开始
使用 VideoUFO 数据集，只需三个步骤即可开始：

**关于 VideoUFO：**

- **数据集**：Hugging Face 上的 [WenhaoWang/VideoUFO](https://huggingface.co/datasets/WenhaoWang/VideoUFO)。
- **规模**：超过 100 万个视频，带有详细描述文本。
- **论文**：[VideoUFO: A Million-Scale User-Focused Dataset](https://openreview.net/forum?id=wwlwRuKle7)（NeurIPS 2025）。
- **许可证**：CC BY 4.0。

<a id="step-1-prepare-dataset"></a>

### 步骤 1：准备数据集
使用自动化脚本下载并准备 VideoUFO 数据集，无需外部依赖：

```bash
# Download and prepare 128 videos from VideoUFO (recommended for testing)
python scripts/prepare_videoufo_dataset.py \
    --storage_dir assets/videoufo \
    --num_videos 128
```

脚本将依次执行：

1. 下载 VideoUFO 元数据 CSV，约 1.1 GB。
2. 下载包含视频的 tar 文件，每个约 4 GB。
3. 解压视频并实时筛选，只保留帧数 ≥93 的视频。
4. 持续解压，直到获得指定数量的有效视频。
5. 创建描述文本 JSON 文件，将简短描述与详细描述合并。
6. 按要求整理目录结构。

**脚本选项：**

- `--storage_dir`：数据集保存位置，必填。
- `--num_videos`：保留的视频数量，默认 8。
- `--num_tars`：下载的 tar 文件数量，默认 1；每个包含约 5,400 个视频。
- `--min_frames`：每个视频所需的最少帧数，默认 93，与训练要求一致。
- `--use_brief_caption`：仅使用简短描述，不合并详细描述。
- `--no_concat_brief`：仅使用详细描述，不拼接简短描述。
- `--skip_download`：tar 文件已存在时跳过下载。
- `--skip_metadata`：元数据 CSV 已存在时跳过下载。

**注意：**脚本会在解压时自动筛选视频，仅保留帧数 ≥93 的视频，并持续解压直到达到指定数量，确保所有视频满足 `num_frames=93` 的训练要求。

**存储需求：**128 个视频约需 5～7 GB，1,000 个视频约需 30 GB，5,000 个视频约需 130 GB，数据来自一个 tar 文件。

<a id="step-2-set-environment"></a>

### 步骤 2：设置环境
```bash
# Set output directory for checkpoints and samples
export IMAGINAIRE_OUTPUT_ROOT=/path/to/outputs
```

<a id="step-3-start-training"></a>

### 步骤 3：开始训练
使用 edge 控制进行训练，推荐采用这种方式，因为无需预处理：

```bash
# Train on 128 videos with edge control
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=assets/videoufo \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=2000 \
    checkpoint.save_iter=500 \
    job.wandb_mode=disabled
```

**上述命令会：**

- 使用 VideoUFO 中的 128 个视频训练。
- 使用实时计算的 edge 控制，无需预处理。
- 执行 2000 次迭代，在 8 块 A100 上约需 0.8 小时。
- 每 500 次迭代保存检查点。
- 将检查点保存到 `${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer2_posttrain/local_single_view/transfer2_singleview_posttrain_edge_example_*/checkpoints/`。

至此，模型已经开始训练。

**后续步骤：**

- 在终端输出中监控训练进度。
- 训练结束后，将检查点转换为 PyTorch 格式，参见[检查点管理](#checkpoint-management)。
- 使用微调后的模型进行推理，参见[推理](#inference)。
- 自定义数据集和高级选项请参阅下文。

---

<a id="custom-data-preparation"></a>

## 自定义数据准备
<a id="1-prepare-your-own-videos-and-captions"></a>

### 1. 准备自有视频和描述文本
按以下结构组织视频：

```
datasets/your_dataset/
├── videos/
│   ├── video1.mp4 (720p, ≥3s, 10-60fps)
│   ├── video2.mp4
│   └── ...
└── captions/
    ├── video1.json  ({"caption": "Your description here"})
    ├── video2.json
    └── ...
```

**视频要求：**

- 格式：MP4。
- 分辨率：建议 720p。
- 时长：≥3 秒。
- 帧率：10～60 fps。

**描述文本格式：**

每个 JSON 文件应包含：

```json
{
    "caption": "A detailed description of the video content"
}
```

<a id="2-verify-dataset-folder-format"></a>

### 2. 检查数据集目录格式
数据集应采用以下结构：

```
datasets/your_dataset/
├── videos/
│   └── *.mp4
└── captions/
    └── *.json
```

对于带有预计算控制输入的 depth/seg 数据集：

```
datasets/your_dataset/
├── videos/
│   └── *.mp4
├── captions/
│   └── *.json
├── depth/
│   └── *.mp4 (optional, for depth control)
└── seg/
    └── *.mp4 (optional, for segmentation control)
```

**注意：**训练过程中，描述文本由模型内置文本编码器实时编码，与多视角训练类似。

<a id="3-optional-pre-compute-control-inputs"></a>

### 3. 预计算控制输入（可选）
**edge/vis 可跳过此步骤**，它们会在训练时实时计算。

**depth 控制：**

深度控制需要预计算深度图。使用内置 DepthAnything V2 流程：

```bash

# Generate depth video for a single video
python cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py \
    --input_video datasets/your_dataset/videos/video1.mp4 \
    --output_video datasets/your_dataset/depth/video1.mp4 \
    --encoder vits

# Process multiple videos (example loop)
for video in datasets/your_dataset/videos/*.mp4; do
    basename=$(basename "$video")
    python cosmos_transfer2/_src/transfer2/auxiliary/depth_anything/depth_pipeline.py \
        --input_video "$video" \
        --output_video "datasets/your_dataset/depth/$basename" \
        --encoder vits
done
```

**参数：**

- `--input_video`：输入 RGB 视频路径。
- `--output_video`：深度视频保存路径，MP4 格式。
- `--encoder`：模型大小，`vits` 表示较小、较快的模型，`vitl` 表示较大、较准确的模型。

**输出：**MP4 格式的灰度深度视频，分辨率与帧数均与输入一致。

**seg 控制：**

分割控制需要预计算分割掩码。使用内置 SAM2 流程：

```bash
# Generate segmentation video using text prompts (recommended)
python cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py \
    --input_video datasets/your_dataset/videos/video1.mp4 \
    --output_video datasets/your_dataset/seg/video1.mp4 \
    --mode prompt \
    --prompt "person, car, vehicle, building, tree, road, sky" \
    --visualize

# Process multiple videos (example loop)
for video in datasets/your_dataset/videos/*.mp4; do
    basename=$(basename "$video")
    python cosmos_transfer2/_src/transfer2/auxiliary/sam2/sam2_pipeline.py \
        --input_video "$video" \
        --output_video "datasets/your_dataset/seg/$basename" \
        --mode prompt \
        --prompt "person, car, vehicle, building, tree, road, sky" \
        --visualize
done
```

**分割模式：**

1. **提示词模式**（推荐）：`--mode prompt --prompt "person, car, building"`。
2. **框选模式**：`--mode box --box "300,0,500,400"`。
3. **点选模式**：`--mode points --points "200,300" --labels "1"`。

**参数：**

- `--input_video`：输入 RGB 视频路径。
- `--output_video`：分割视频保存路径，MP4 格式。
- `--mode`：分割模式，可选 `prompt`、`box` 或 `points`。
- `--prompt`：待分割物体的文本描述，用于提示词模式。
- `--visualize`：启用视频输出所必需的标志。

**输出：**MP4 格式的彩色分割视频，每个物体实例具有唯一颜色。

<a id="4-final-verification"></a>

### 4. 最终检查
训练前，确认数据集包含全部必需内容：

```
datasets/your_dataset/
├── videos/
│   └── *.mp4
├── captions/
│   └── *.json
└── depth/  (optional, for depth control only)
    └── *.mp4
```

确保：

- `videos/` 中的每个视频在 `captions/` 中都有对应的 JSON 文件。
- 文件名相互对应，例如 `video1.mp4` → `video1.json`。
- 描述文本 JSON 文件包含有效的文本描述。

<a id="training-configuration"></a>

## 训练配置
`cosmos_transfer2.experiments.singleview.cosmos_singleview_example` 提供四个实验：

1. **边缘控制**（推荐）：`experiment=transfer2_singleview_posttrain_edge_example`。
2. **深度控制**：`experiment=transfer2_singleview_posttrain_depth_example`。
3. **分割控制**：`experiment=transfer2_singleview_posttrain_seg_example`。
4. **视觉模糊控制**：`experiment=transfer2_singleview_posttrain_vis_example`。

**文本编码：**所有实验均使用 Qwen2.5-VL-7B（`reason1p1_7B`）实时编码描述文本，以匹配预训练模型的训练分布。

<a id="key-parameters"></a>

### 关键参数
所有控制类型（edge、depth、seg、vis）共享相同默认参数：

```python
# Dataset
dataset_dir="datasets/your_dataset"          # Your dataset path
num_frames=93                                 # (state_t-1)*4+1 = (24-1)*4+1 = 93
video_size=(704, 1280)                        # (H, W) - 720p, 16:9 aspect ratio
hint_key="control_input_edge"                 # Control type: edge, depth, seg, or vis

# Model
state_t=24                                    # Temporal latent size
context_parallel_size=8                       # Must divide state_t (adjust based on GPUs)

# Text Encoder (matches pretrained model)
text_encoder_class="reason1p1_7B"             # Qwen2.5-VL-7B
embedding_concat_strategy="FULL_CONCAT"       # All 28 layers
crossattn_proj_in_channels=100352             # 3584 * 28
crossattn_emb_channels=1024                   # Projected dimension

# Training
max_iter=5000
save_iter=1000                                # Checkpoint save frequency
lr=5e-5                                       # Learning rate
warm_up_steps=1000                            # LR warmup to prevent gradient spikes
grad_accum_iter=4

# Checkpoint (auto-downloaded from HuggingFace)
load_path=get_checkpoint_path(...)            # Auto-downloads on first run
dcp_async_mode_enabled=False                  # Disabled for stability
```

<a id="customizing-via-command-line"></a>

### 通过命令行自定义
```bash
# Example with custom dataset
torchrun --nproc_per_node=8 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/your_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=5000 \
    checkpoint.save_iter=500

# Example with VideoUFO dataset
torchrun --nproc_per_node=8 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=assets/videoufo \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    trainer.max_iter=5000 \
    checkpoint.save_iter=500
```

**必填参数：**

- `dataloader_train.dataset.dataset_dir`：数据集目录路径，例如 `datasets/your_dataset` 或 `assets/videoufo`。
- `'dataloader_train.sampler.dataset=${dataloader_train.dataset}'`：将采样器关联到数据集，确保多 GPU 数据划分正确。

**可选参数：**

- `trainer.max_iter`：训练总迭代次数，默认 5000。
- `checkpoint.save_iter`：检查点保存频率，默认 1000。
- `optimizer.lr`：学习率，默认 5e-5。
- `scheduler.warm_up_steps`：学习率预热步数，默认 [1000]。
- `job.wandb_mode`：W&B 模式，可选 online/offline/disabled。
- `checkpoint.load_path`：使用自定义检查点路径覆盖自动下载。

<a id="launch-training"></a>

## 启动训练
**注意：**训练前请设置输出目录：`export IMAGINAIRE_OUTPUT_ROOT=/path/to/outputs`。

<a id="1-run-training"></a>

### 1. 运行训练
**8 块 GPU（2B 模型）：**

```bash
# Edge control
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_edge_example \
    dataloader_train.dataset.dataset_dir=datasets/your_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    job.wandb_mode=disabled

# Depth control (requires pre-computed depth videos)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_depth_example \
    dataloader_train.dataset.dataset_dir=datasets/your_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    job.wandb_mode=disabled

# Seg control (requires pre-computed segmentation masks)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_seg_example \
    dataloader_train.dataset.dataset_dir=datasets/your_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    job.wandb_mode=disabled

# Visual blur control (requires pre-computed visual blur videos)
torchrun --nproc_per_node=8 --master_port=12345 -m scripts.train \
    --config=cosmos_transfer2/singleview_config.py \
    -- experiment=transfer2_singleview_posttrain_vis_example \
    dataloader_train.dataset.dataset_dir=datasets/your_dataset \
    'dataloader_train.sampler.dataset=${dataloader_train.dataset}' \
    job.wandb_mode=disabled
```

**关键参数：**

- `experiment`：选择控制类型，edge、depth、seg 或 vis。
- `dataloader_train.dataset.dataset_dir`：数据集路径。
- `'dataloader_train.sampler.dataset=${dataloader_train.dataset}'`：将采样器关联到数据集，以支持多 GPU 训练。
- `job.wandb_mode=disabled`：禁用 W&B 日志，可选。

**注意：**采样器参数使用 `${...}` 语法引用数据集对象，确保数据在各 GPU 之间正确划分。

**注意：**首次运行时会从 Hugging Face **自动下载**预训练检查点，无需手动下载。

**检查点输出：**

检查点保存在 `${IMAGINAIRE_OUTPUT_ROOT}/PROJECT/GROUP/NAME/checkpoints`。`IMAGINAIRE_OUTPUT_ROOT` 默认为 `/tmp/imaginaire4-output`。强烈建议将 `IMAGINAIRE_OUTPUT_ROOT` 设置为具有足够检查点存储空间的位置。

在上述 `edge` 实验示例中，`PROJECT`、`GROUP` 和 `NAME` 来自实验配置的 `job` 字典：

- `PROJECT` = `cosmos_transfer2_posttrain`。
- `GROUP` = `local_single_view`。
- `NAME` = `transfer2_singleview_posttrain_edge_example_2025-11-21_16-30-45`，时间戳会自动添加。

因此，完整检查点路径为：

```
${IMAGINAIRE_OUTPUT_ROOT}/cosmos_transfer2_posttrain/local_single_view/transfer2_singleview_posttrain_vis_example_2025-11-21_16-30-45/checkpoints/
```

<a id="monitoring"></a>

## 监控
```
[Iteration 100/2000] Loss: 0.234, LR: 4.95e-05, Time: 1.23s/it  # After warmup
[Iteration 200/2000] Saving checkpoint...
```

<a id="sample-generations"></a>

### 生成样本
- 默认每 50 次迭代生成一次。
- 保存到 `${IMAGINAIRE_OUTPUT_ROOT}/<project>/<group>/<name>/samples/`。

<a id="training-time-2000-iterations"></a>

### 训练耗时（2000 次迭代）
| 数据集 | 模型 | GPU | 每次迭代耗时 | 总耗时 |
|---------|-------|------|-----------|-------|
| 128 个视频 | 2B | 8 块 A100 | 约 1.2 秒 | 约 0.7 小时 |
| 1000 个视频 | 2B | 8 块 A100 | 约 1.5 秒 | 约 0.8 小时 |
| 128 个视频 | 14B | 16 块 A100 | 约 2.0 秒 | 约 1.1 小时 |

<a id="checkpoint-management"></a>

## 检查点管理
<a id="convert-dcp-to-pytorch"></a>

### 将 DCP 转换为 PyTorch 格式
```bash
CHECKPOINTS_DIR=${IMAGINAIRE_OUTPUT_ROOT:-/tmp/imaginaire4-output}/cosmos_transfer2_posttrain/local_single_view/2B_edge_posttrain_*/checkpoints
CHECKPOINT_ITER=$(cat $CHECKPOINTS_DIR/latest_checkpoint.txt)
CHECKPOINT_DIR=$CHECKPOINTS_DIR/$CHECKPOINT_ITER

python scripts/convert_distcp_to_pt.py $CHECKPOINT_DIR/model $CHECKPOINT_DIR
```

生成以下文件：

- `model_ema_bf16.pt` ← **推理时使用此文件**。
- `model_ema_fp32.pt`。
- `model.pt`，完整检查点。

<a id="directory-structure"></a>

### 目录结构
```
${IMAGINAIRE_OUTPUT_ROOT}/
└── cosmos_transfer2_posttrain/
    └── local_single_view/
        └── 2B_edge_posttrain_2025-11-19_10-30-00/
            ├── checkpoints/
            │   ├── iter_000000200/model_ema_bf16.pt ← Use this
            │   └── latest_checkpoint.txt
            └── samples/
```

<a id="resume-training"></a>

### 恢复训练
重新运行相同命令即可，系统会从最新检查点自动恢复。

<a id="inference"></a>

## 推理
```bash
# Using the inference script
torchrun --nproc_per_node=8 examples/inference.py \
    -i assets/edge.jsonl \
    -o outputs/ \
    --checkpoint-path $CHECKPOINT_DIR/model_ema_bf16.pt \
    --experiment transfer2_singleview_posttrain_edge_example
```

<a id="faq"></a>

## 常见问答
<a id="which-control-type-to-start-with"></a>

### 从哪种控制类型开始？
**边缘控制**：无需预处理、迭代快，并采用优化的内存配置。

<a id="how-much-data-do-i-need"></a>

### 需要多少数据？
- 最低：50～100 个视频。
- 推荐：200～500 个视频。
- 理想：1000 个以上视频。

<a id="gpu-out-of-memory"></a>

### GPU 显存不足怎么办？
1. 增大 `context_parallel_size`，该值必须能整除 `state_t`，例如 12 或 24。
2. 同时减小 `num_frames` 和 `state_t`，二者必须满足 `num_frames = (state_t-1)*4+1`：
   - 93 帧 → 77 帧：`dataloader_train.dataset.num_frames=77 model.config.state_t=20`。
   - 93 帧 → 61 帧：`dataloader_train.dataset.num_frames=61 model.config.state_t=16`。
3. 启用 `dcp_async_mode_enabled=True`，出于稳定性考虑，默认禁用此选项。
4. 降低保存频率，即增大 `save_iter`。
5. 使用更多 GPU 分摊内存负载。

<a id="what-is-context_parallel_size"></a>

### `context_parallel_size` 是什么？
它将序列拆分到多块 GPU 上，以减少每块 GPU 的显存占用，必须能整除 `state_t`：

- 默认：`state_t=24` ÷ `context_parallel_size=8` = 每块 GPU 处理 3 个潜在帧。
- 示例：`state_t=24` ÷ `context_parallel_size=4` = 每块 GPU 处理 6 个潜在帧。

潜在帧数 `state_t` 与像素帧数 `num_frames` 的关系如下：

```
num_frames = (state_t - 1) * 4 + 1
# e.g., state_t=24 → num_frames = (24-1)*4+1 = 93
```

<a id="how-to-know-if-training-is-working"></a>

### 如何判断训练是否正常？
- 损失从约 0.5 降至约 0.1～0.2。
- 生成样本的质量随迭代改善。
- 日志中没有 NaN/Inf。
- W&B 曲线平滑。

<a id="troubleshooting"></a>

## 故障排查
<a id="cuda-out-of-memory"></a>

### 出现 "CUDA out of memory"
1. 增大 `context_parallel_size`，该值必须能整除 `state_t`，例如 12 或 24。
2. 同时减小 `num_frames` 和 `state_t`：
   - `dataloader_train.dataset.num_frames=77 model.config.state_t=20`。
   - `dataloader_train.dataset.num_frames=61 model.config.state_t=16`。
3. 使用更多 GPU。

<a id="video-has-only-x-frames-need-y"></a>

### 出现 "Video has only X frames, need Y"
- 使用更长的视频；93 帧在 24 FPS 下要求视频时长 ≥4 秒。
- 同时减小 `num_frames` 和 `state_t`：
  - `dataloader_train.dataset.num_frames=77 model.config.state_t=20`，要求时长 ≥3.2 秒。
  - `dataloader_train.dataset.num_frames=61 model.config.state_t=16`，要求时长 ≥2.5 秒。
- 预处理时过滤过短的视频。

<a id="training-loss-not-decreasing"></a>

### 训练损失不下降
- 检查学习率，尝试不同数值。
- 检查数据质量。
- 确认检查点已正确加载。
- 增加数据量。

<a id="generated-videos-have-artifacts"></a>

### 生成视频存在伪影
- 延长训练时间。
- 使用更高质量的训练数据。
- 调整推理时的引导强度。
- 比较 EMA 检查点与普通检查点的效果。

---

<a id="additional-resources"></a>

## 补充资源
**关键文件：**

- 训练实验：`cosmos_transfer2/experiments/singleview/cosmos_singleview_example.py`。
- 配置封装：`cosmos_transfer2/singleview_config.py`。
- 数据集加载器：`projects/cosmos/transfer2/datasets/local_datasets/singleview_dataset.py`。
- 数据加载器配置：`projects/cosmos/transfer2/configs/vid2vid_transfer/defaults/dataloader_local.py`。
- VideoUFO 数据准备脚本：`scripts/prepare_videoufo_dataset.py`。

**相关文档：**

- [安装指南](setup.md)
- [推理指南](inference.md)
- [多视角训练](post-training_auto_multiview.md)
- [后训练概览](post-training.md)

**示例数据集：**

- [VideoUFO](https://huggingface.co/datasets/WenhaoWang/VideoUFO)：超过 100 万个视频，带有详细描述文本，可使用 `scripts/prepare_videoufo_dataset.py` 快速准备。

<a id="citation"></a>

## 引用
```bibtex
@article{cosmos2025,
  title={Cosmos World Foundation Models},
  author={NVIDIA Research},
  year={2025}
}
```
