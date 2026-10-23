<a id="robot-multiview-control-conditioned-inference"></a>

# 机器人多视角控制条件推理
> **⚠️ 注意**
> 本文介绍的模型和功能目前**仅限内部使用**，尚未公开。这包括文中引用的所有检查点，例如 `Cosmos-Transfer2.5-2B/robot-multiview-control`。
>
> 因此，目前无法在 NVIDIA 外部执行本指南中的步骤。未来可能会开放这些模型；如有变化，本文将相应更新。

建议先阅读[推理指南](inference.md)。
本指南介绍机器人多视角场景中的控制条件视频生成，具体针对具有三个相机视角（头部、左手、右手）的 Agibot 数据集。

<a id="prerequisites"></a>

## 前置条件
[安装指南](setup.md)

<a id="overview"></a>

## 概述
控制条件推理允许在机器人自身视角的视频输入之外，使用多种控制信号引导视频生成。通过加入结构信息，可以更精确地控制生成结果。

<a id="supported-control-types"></a>

### 支持的控制类型
Agibot 多视角模型支持四种控制类型：

| 控制类型 | 说明 | 生成方式 | 用途 |
|--------------|-------------|-------------------|----------|
| **edge** | Canny 边缘检测图 | 根据输入视频实时生成 | 保留清晰边界和结构边缘 |
| **vis** | 视觉模糊图 | 根据输入视频实时生成 | 控制视觉清晰度和焦点 |
| **depth** | 深度估计图 | 预先计算，从磁盘加载 | 保留三维场景结构 |
| **seg** | 分割图 | 预先计算，从磁盘加载 | 保持物体边界和语义区域 |

<a id="camera-configuration"></a>

### 相机配置
Agibot 使用三个同步相机：

- **视角 0**：`head_color`，头部相机。
- **视角 1**：`hand_left`，左手相机。
- **视角 2**：`hand_right`，右手相机。

<a id="example-visualizations"></a>

## 可视化示例
以下为各控制类型的输出示例。每行左侧是控制信号，右侧是对应的生成结果：

<a id="edge-control"></a>

### 边缘控制
<table>
<tr>
<td width="50%">

**控制输入**

![边缘控制输入](https://github.com/user-attachments/assets/2d4adef2-66fa-49ed-9d02-5a8698c1cc6d)

</td>
<td width="50%">

**生成输出**

![边缘控制生成结果](https://github.com/user-attachments/assets/40f204ba-d8e0-4b10-aa3a-82678ca18b7c)

</td>
</tr>
</table>

<a id="vis-visual-blur-control"></a>

### Vis（视觉模糊）控制
<table>
<tr>
<td width="50%">

**控制输入**

![视觉模糊控制输入](https://github.com/user-attachments/assets/ba9ef039-595b-4fb1-b655-aa4521350565)

</td>
<td width="50%">

**生成输出**

![视觉模糊控制生成结果](https://github.com/user-attachments/assets/bf0bbc5e-e57d-412b-999d-43e9abfff7b2)

</td>
</tr>
</table>

<a id="depth-control"></a>

### 深度控制
<table>
<tr>
<td width="50%">

**控制输入**

![深度控制输入](https://github.com/user-attachments/assets/c4b5360a-6cbb-4d61-bc30-25e96f8992ae)

</td>
<td width="50%">

**生成输出**

![深度控制生成结果](https://github.com/user-attachments/assets/f38a1f74-b1ec-4eff-bc5b-9f6a69ebd1d4)

</td>
</tr>
</table>

<a id="seg-segmentation-control"></a>

### Seg（分割）控制
<table>
<tr>
<td width="50%">

**控制输入**

![分割控制输入](https://github.com/user-attachments/assets/f2b1aae5-08af-49a0-8dc8-6f7a826600a4)

</td>
<td width="50%">

**生成输出**

![分割控制生成结果](https://github.com/user-attachments/assets/b46361e5-ff20-4460-be42-82c8ccb9eb26)

</td>
</tr>
</table>

<a id="image-to-video-i2v-vs-text-to-video-t2v"></a>

## 图像到视频（I2V）与文本到视频（T2V）
Agibot 模型支持两种推理模式：

- **图像到视频（I2V）**：同时以控制信号和输入帧为条件生成视频，设置 `num_conditional_frames=1`，这是默认模式。
- **文本到视频（T2V）**：仅以控制信号和文本描述为条件生成视频，设置 `num_conditional_frames=0`。

**上述示例均使用默认的 I2V 模式。**如需使用 T2V 模式，请在 JSON 配置中添加 `"num_conditional_frames": 0`。

**使用 Edge/Vis 控制的 T2V 模式**（需要输入视频）：

```json
{
    "name": "my_t2v_edge_example",
    "seed": 42,
    "guidance": 3.0,
    "num_conditional_frames": 0,
    "prompt": "A robotic arm assembling circuit boards in a well-lit factory",
    "head_color": {
        "input_path": "videos/296_656371_chunk0_head_color_rgb.mp4"
    },
    "hand_left": {
        "input_path": "videos/296_656371_chunk0_hand_left_rgb.mp4"
    },
    "hand_right": {
        "input_path": "videos/296_656371_chunk0_hand_right_rgb.mp4"
    }
}
```

**使用 Depth/Seg 控制的 T2V 模式**（输入视频可选）：

```json
{
    "name": "my_t2v_depth_example",
    "seed": 42,
    "guidance": 3.0,
    "num_conditional_frames": 0,
    "prompt": "A robotic arm carefully picking up delicate electronic components",
    "head_color": {
        "control_path": "depth/296_656371_chunk0_head_color_depth.mp4"
    },
    "hand_left": {
        "control_path": "depth/296_656371_chunk0_hand_left_depth.mp4"
    },
    "hand_right": {
        "control_path": "depth/296_656371_chunk0_hand_right_depth.mp4"
    }
}
```

**T2V 模式的输入视频要求：**

- **Edge/Vis 控制**：**必须**在 `input_path` 中提供输入视频，用于实时生成控制信号，但这些视频不会作为生成的条件帧。
- **Depth/Seg 控制**：`input_path` 中的输入视频为**可选项**；若不提供，将使用控制视频作为占位输入。

<a id="setup-and-data-preparation"></a>

## 配置与数据准备
<a id="using-example-data"></a>

### 使用示例数据
全部四种控制类型的示例数据均可在 `assets/robot_multiview_control-agibot/` 中获取。可以直接使用这些示例测试模型，无需准备自有数据。

<a id="preparing-your-own-data"></a>

### 准备自有数据
使用自有机器人多视角数据时，请按以下方式组织：

```
your_data_root/
├── videos/                          # Required: Input RGB videos
│   ├── {video_id}_head_color_rgb.mp4
│   ├── {video_id}_hand_left_rgb.mp4
│   └── {video_id}_hand_right_rgb.mp4
├── captions/                        # Recommended: Text captions (see note below)
│   ├── {video_id}_head_color.txt
│   ├── {video_id}_hand_left.txt
│   └── {video_id}_hand_right.txt
├── depth/                           # Required for depth control only
│   ├── {video_id}_head_color_depth.mp4
│   ├── {video_id}_hand_left_depth.mp4
│   └── {video_id}_hand_right_depth.mp4
└── seg/                             # Required for seg control only
    ├── {video_id}_head_color_seg.mp4
    ├── {video_id}_hand_left_seg.mp4
    └── {video_id}_hand_right_seg.mp4
```

<a id="control-specific-requirements"></a>

### 各控制类型的数据要求
- **边缘控制**：仅需 `videos/` 目录中的输入视频，边缘图会自动生成。
- **视觉控制**：仅需 `videos/` 目录中的输入视频，视觉模糊图会自动生成。
- **深度控制**：同时需要 `videos/` 中的输入视频，以及 `depth/` 中预先计算的深度图。
- **分割控制**：同时需要 `videos/` 中的输入视频，以及 `seg/` 中预先计算的分割图。

<a id="inference-configuration-json-format"></a>

### 推理配置 JSON 格式
JSON 配置按相机分别指定输入路径和控制路径。每个相机视角需要：

- `input_path`：输入 RGB 视频路径，必填。
- `control_path`：控制视频路径，depth/seg 必填，edge/vis 不需要。

<a id="edgevisual-blur-controls-on-the-fly-generation"></a>

#### 边缘／视觉模糊控制（实时生成）
边缘和视觉模糊控制只需提供 `input_path`，控制图会根据输入视频自动计算。

```json
{
    "name": "my_edge_example",
    "seed": 42,
    "guidance": 3.0,
    "head_color": {
        "input_path": "videos/296_656371_chunk0_head_color_rgb.mp4"
    },
    "hand_left": {
        "input_path": "videos/296_656371_chunk0_hand_left_rgb.mp4"
    },
    "hand_right": {
        "input_path": "videos/296_656371_chunk0_hand_right_rgb.mp4"
    }
}
```

<a id="depthsegmentation-controls-pre-computed"></a>

#### 深度／分割控制（预计算）
深度和分割控制需要同时提供 `input_path` 和 `control_path`。

```json
{
    "name": "my_depth_example",
    "seed": 42,
    "guidance": 3.0,
    "head_color": {
        "input_path": "videos/296_656371_chunk0_head_color_rgb.mp4",
        "control_path": "depth/296_656371_chunk0_head_color_depth.mp4"
    },
    "hand_left": {
        "input_path": "videos/296_656371_chunk0_hand_left_rgb.mp4",
        "control_path": "depth/296_656371_chunk0_hand_left_depth.mp4"
    },
    "hand_right": {
        "input_path": "videos/296_656371_chunk0_hand_right_rgb.mp4",
        "control_path": "depth/296_656371_chunk0_hand_right_depth.mp4"
    }
}
```

**必填字段：**

- `name`：本次推理运行的名称，将用作输出文件名，例如 `outputs/my_test_example.mp4`。
- `head_color`、`hand_left`、`hand_right`：各相机配置，包含 `input_path`，以及可选的 `control_path`。
- `seed`：用于复现结果的随机种子。
- `guidance`：无分类器引导强度，通常为 3.0～7.0。

**路径说明：**`input_path` 和 `control_path` 均相对于命令行中指定的 `--input-root` 目录。

**可选字段：**

- `num_conditional_frames`：T2V 模式设为 `0`，I2V 模式设为 `1`，默认值为 `1`。
- `num_steps`：扩散步数，默认值为 `35`。
- `fps`：每秒帧数，默认值为 `10`。
- `target_height`：输出高度，默认值为 `720`。
- `target_width`：输出宽度，默认值为 `1280`。
- `prompt`：覆盖所有相机视角的文本提示词，优先级高于描述文本文件和相机前缀。**建议**提供详细描述以改善生成质量。
- `preset_edge_threshold`：边缘控制的检测强度，可选 `very_low`、`low`、`medium`、`high`、`very_high`，默认值为 `medium`。
- `preset_blur_strength`：vis 控制的模糊强度，可选 `very_low`、`low`、`medium`、`high`、`very_high`，默认值为 `medium`。
- `control_weight`：控制信号强度，范围为 `0.0`～`1.0`，默认值为 `1.0`。数值越大，生成结果越严格地遵循控制信号。
- `save_combined_views`：为 `true`（默认）时，仅保存一个横向拼接视频；为 `false` 时，保存三个独立视角视频和一个横向拼接视频。
- `enable_autoregressive`：启用自回归模式以生成更长的视频，默认值为 `false`。
- `num_chunks`：自回归模式下生成的片段数量，默认值为 `2`，仅在 `enable_autoregressive=true` 时使用。
- `chunk_overlap`：相邻片段之间重叠的视频帧数，默认值为 `1`，仅在 `enable_autoregressive=true` 时使用。

<a id="running-inference"></a>

## 运行推理
<a id="basic-command-template"></a>

### 基本命令模板
```shell exclude=true
torchrun \
    --nproc_per_node=<NUM_GPUS> \
    examples/robot_multiview_agibot_control.py \
    -i <PATH_TO_TEST_JSON> \
    -o <OUTPUT_DIR> \
    --input-root <DATA_ROOT> \
    --control-type <CONTROL_TYPE>
```

**注意：**所有命令都应在 `packages/cosmos-transfer2/` 目录中运行。

<a id="parameters"></a>

### 参数
- `--nproc_per_node`：使用的 GPU 数量，**至少需要 4 块**。
- `-i`：测试 JSON 文件路径。
- `-o`：生成视频的输出目录。
- `--input-root`：数据所在的根目录。
- `--control-type`：`edge`、`vis`、`depth` 或 `seg` 中的一种。

**默认输出分辨率：**模型默认生成 720×1280（高×宽）的视频。可以在推理 JSON 中通过 `target_height` 和 `target_width` 为每个样本单独设置。

> **GPU 要求：**
>
> 由于上下文并行限制，模型**至少需要 4 块 GPU**：
>
> **有效 GPU 数量：4、8 或 16。**
>
> - 4 块 GPU：推荐用于开发和测试。
> - 8 块 GPU：在速度与资源占用之间取得平衡。
> - 16 块 GPU：用于生产环境的最大并行规模。

<a id="example-commands"></a>

### 命令示例
<a id="edge-control-on-the-fly-generation"></a>

#### 边缘控制（实时生成）
```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 \
    examples/robot_multiview_agibot_control.py \
    -i assets/robot_multiview_control-agibot/edge_test.json \
    -o outputs/agibot_edge_test \
    --input-root assets/robot_multiview_control-agibot \
    --control-type edge
```

<a id="visual-blur-control-on-the-fly-generation"></a>

#### 视觉模糊控制（实时生成）
```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 \
    examples/robot_multiview_agibot_control.py \
    -i assets/robot_multiview_control-agibot/vis_test.json \
    -o outputs/agibot_vis_test \
    --input-root assets/robot_multiview_control-agibot \
    --control-type vis
```

<a id="depth-control-pre-computed"></a>

#### 深度控制（预计算）
```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 \
    examples/robot_multiview_agibot_control.py \
    -i assets/robot_multiview_control-agibot/depth_test.json \
    -o outputs/agibot_depth_test \
    --input-root assets/robot_multiview_control-agibot \
    --control-type depth
```

<a id="segmentation-control-pre-computed"></a>

#### 分割控制（预计算）
```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 \
    examples/robot_multiview_agibot_control.py \
    -i assets/robot_multiview_control-agibot/seg_test.json \
    -o outputs/agibot_seg_test \
    --input-root assets/robot_multiview_control-agibot \
    --control-type seg
```

<a id="autoregressive-mode-longer-videos"></a>

#### 自回归模式（更长视频）
生成时长超过模型原生时间容量的视频时，请使用自回归模式。创建包含自回归参数的 JSON 配置：

```json
{
    "name": "agibot_edge_test_autoregressive",
    "seed": 42,
    "guidance": 3.0,
    "enable_autoregressive": true,
    "num_chunks": 2,
    "chunk_overlap": 1,
    "head_color": {
        "input_path": "videos/296_656371_chunk0_long_head_color_rgb.mp4"
    },
    "hand_left": {
        "input_path": "videos/296_656371_chunk0_long_hand_left_rgb.mp4"
    },
    "hand_right": {
        "input_path": "videos/296_656371_chunk0_long_hand_right_rgb.mp4"
    }
}
```

然后运行推理：

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 \
    examples/robot_multiview_agibot_control.py \
    -i assets/robot_multiview_control-agibot/edge_test_autoregressive.json \
    -o outputs/agibot_edge_test_autoregressive \
    --input-root assets/robot_multiview_control-agibot \
    --control-type edge
```

模型会生成多个有重叠部分的片段，再将它们拼接为更长的视频。输出总帧数为：`num_video_frames_per_chunk + (num_video_frames_per_chunk - chunk_overlap) * (num_chunks - 1)`。

<a id="output-format"></a>

## 输出格式
生成的视频可以采用两种保存模式：

<a id="combined-view-mode-default-save_combined_views-true"></a>

### 合并视角模式（默认，`save_combined_views: true`）
**仅保存一个视频**，将三个相机视角横向排列：

```
<OUTPUT_DIR>/
└── <name>.mp4  # Combined video with all 3 views (head_color | hand_left | hand_right)
```

例如，设置 `"name": "my_test_example"` 时，输出为 `outputs/my_test_example.mp4`。

<a id="split-view-mode-save_combined_views-false"></a>

### 独立视角模式（`save_combined_views: false`）
保存**四个视频**：三个独立相机视角视频，以及一个横向拼接视频：

```
<OUTPUT_DIR>/
├── <name>_head_color.mp4      # Individual head camera view
├── <name>_hand_left.mp4       # Individual left hand camera view
├── <name>_hand_right.mp4      # Individual right hand camera view
└── <name>_grid.mp4            # Combined horizontal view (head_color | hand_left | hand_right)
```

例如，设置 `"name": "my_test_example"` 时，输出包含 `my_test_example_head_color.mp4` 等文件。

网格视频即合并视角视频，将三个视角横向排列在同一画面中。

<a id="technical-details"></a>

## 技术细节
<a id="control-signal-processing"></a>

### 控制信号处理
**实时控制信号（Edge、Visual）：**

- 边缘图：对输入视频进行 Canny 边缘检测得到。
- 视觉模糊图：对输入视频进行模糊检测得到。
- 在推理时处理，无需预先计算。
- 对整个输入序列的所有帧使用一致的处理方式。

**预计算控制信号（Depth、Seg）：**

- 以视频文件形式从磁盘加载。
- 时间长度必须与输入视频一致。
- 应与输入视频逐帧同步。
- 通常使用外部深度估计或分割模型生成。

<a id="model-architecture"></a>

### 模型架构
控制条件模型基于 Transfer2.5 多视角架构，并增加控制条件层，具有以下特点：

- 接收三个视角的同步输入视频。
- 将控制信号作为附加条件。
- 生成三个视角的同步输出视频。
- 支持上下文并行，以高效执行多 GPU 推理。

<a id="text-prompts-and-captions"></a>

### 文本提示词与描述文本
**描述文本的工作方式**

推理系统支持多种文本提示词提供方式，优先级如下：

1. **JSON 配置中的 `prompt` 字段**（最高优先级）：对所有相机视角使用相同提示词。
2. **描述文本文件**：从 `{input_root}/captions/{video_id}_{camera}.txt` 加载每个相机的描述。
3. **相机前缀回退**（最低优先级）：使用各相机专属的描述前缀。

**描述文本前缀**

从文件加载描述文本或使用回退方式时，可以自动添加相机专属前缀，其含义如下：

- 视角 0（head_color）：“视频由安装在主体头部、朝向前方的相机拍摄。”
- 视角 1（hand_left）：“视频由安装在主体左手的相机拍摄。”
- 视角 2（hand_right）：“视频由安装在主体右手的相机拍摄。”

**在 JSON 中覆盖提示词**

可以通过 `prompt` 字段为所有相机提供统一的覆盖提示词：

```json
{
    "name": "my_example",
    "seed": 42,
    "guidance": 3.0,
    "prompt": "A robotic arm carefully manipulating delicate electronic components on a workbench under bright LED lighting",
    "head_color": {
        "input_path": "videos/296_656371_chunk0_head_color_rgb.mp4"
    },
    "hand_left": {
        "input_path": "videos/296_656371_chunk0_hand_left_rgb.mp4"
    },
    "hand_right": {
        "input_path": "videos/296_656371_chunk0_hand_right_rgb.mp4"
    }
}
```

**建议：**为获得最佳质量，应始终提供详细的描述文本，涵盖：

- 场景背景和环境。
- 正在执行的动作或运动。
- 光照条件和视觉细节。
- 重要物体及其交互。

<a id="related-documentation"></a>

## 相关文档
- [后训练](post-training.md)：在自定义数据集上微调。
- [故障排查](troubleshooting.md)：通用故障排查指南。
