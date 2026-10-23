<a id="cosmos-transfer25-2b-world-generation-with-adaptive-multimodal-control"></a>

# Cosmos-Transfer2.5-2B：具有自适应多模态控制的世界生成
本指南介绍如何使用 Cosmos-Transfer2.5 通用模型进行推理。

![模型架构](../assets/Cosmos-Transfer2-2B-Arch.png)

<a id="pre-requisites"></a>

### 前置条件
1. 按照[安装指南](setup.md)完成环境配置、检查点下载，并确认满足硬件要求。

<a id="hardware-requirements"></a>

### 硬件要求
下表列出了不同 Cosmos-Transfer2.5 模型进行单 GPU 推理时的显存需求：

| 模型 | 所需 GPU 显存 |
|-------|-------------------|
| Cosmos-Transfer2.5-2B | 65.4 GB |

<a id="inference-performance"></a>

### 推理性能
<a id="segmentation"></a>

#### 分割控制
下表列出了不同 NVIDIA GPU 硬件进行单 GPU 推理时的生成耗时（*）：

| GPU 硬件 | Cosmos-Transfer2.5-2B 生成 93 帧的耗时 | Cosmos-Transfer2.5-2B 端到端耗时（**） |
|--------------|---------------|---------------|
| NVIDIA B200 | 92.25 秒 | 186.92 |
| NVIDIA H100 NVL | 445.52 秒 | 895.33 |
| NVIDIA H100 PCIe | 264.13 秒 | 533.58 |
| NVIDIA H20 | 683.65 秒 | 1370.39 |

\* 生成耗时对应 720P、16 FPS 视频，输入为分割控制信号，并关闭安全护栏。 \
\** 端到端耗时使用 121 帧输入视频测量，需要生成两个各 93 帧的视频片段。

<a id="edge"></a>

#### 边缘控制
下表比较了不同 GPU 架构上 Transfer 2.5 边缘控制基础模型与蒸馏模型的推理性能。

| 指标 | GPU 数量 | RTX PRO 6000 Blackwell SE | H20 | H100 NVL | H200 NVL | B200 | B300 |
|--------|------|----------------------------|-----|----------|----------|------|------|
| **蒸馏模型平均扩散耗时（秒）** | 1 | 78.5 | 176.4 | 64.5 | 49.8 | 24.2 | 53.2 |
| | 4 | 33.7 | 62.7 | 27.4 | 20.4 | 12.6 | 25.6 |
| | 8 | 25.0 | 44.0 | 20.4 | 16.9 | 11.1 | 19.9 |
| **基础模型平均扩散耗时（秒）** | 1 | 605.7 | 1374.6 | 502.6 | 374.4 | 179.7 | 415.5 |
| | 4 | 196.1 | 373.4 | 154.5 | 117.0 | 62.3 | 127.7 |
| | 8 | 118.8 | 201.5 | 92.5 | 82.4 | 41.8 | 76.1 |
| **平均性能提升** | 1 | 7.7x | 7.8x | 7.8x | 7.5x | 7.4x | 7.8x |
| | 4 | 5.8x | 6.0x | 5.6x | 5.7x | 5.0x | 5.0x |
| | 8 | 4.7x | 4.6x | 4.5x | 4.9x | 3.8x | 3.8x |

<a id="inference-with-pre-trained-cosmos-transfer25-models"></a>

## 使用预训练 Cosmos-Transfer2.5 模型进行推理
**有关控制模态的详细说明和示例，请参阅 Cosmos Cookbook 中的[控制模态](https://nvidia-cosmos.github.io/cosmos-cookbook/core_concepts/control_modalities/overview.html)教程。**

各个控制变体均可在单 GPU 上运行：

```bash
python examples/inference.py -i assets/robot_example/depth/robot_depth_spec.json -o outputs/depth
```

若要对单个控制变体进行多 GPU 推理，或运行多个控制变体，请使用 [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)：

```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py -i assets/robot_example/depth/robot_depth_spec.json -o outputs/depth
```

我们为各个控制变体以及多控制变体提供了参数文件示例：

| 变体 | 参数文件 |
| --- | --- |
| 深度 | `assets/robot_example/depth/robot_depth_spec.json` |
| 边缘 | `assets/robot_example/edge/robot_edge_spec.json` |
| 分割 | `assets/robot_example/seg/robot_seg_spec.json` |
| 模糊 | `assets/robot_example/vis/robot_vis_spec.json` |
| 多控制 | `assets/robot_example/multicontrol/robot_multicontrol_spec.json` |
| 蒸馏／边缘 | `assets/robot_example/distilled/edge/robot_edge_spec.json` |

运行以下命令可查看所有可用参数的说明：

```bash
python examples/inference.py --help

python examples/inference.py control:edge --help # for information specific to edge control
```

参数可以使用 JSON 格式指定：

```jsonc
{
    // Path to the prompt file, use "prompt" to directly specify the prompt
    "prompt_path": "assets/robot_example/robot_prompt.json",

    // Directory to save the generated video
    "output_dir": "outputs/robot_multicontrol",

    // Path to the input video
    "video_path": "assets/robot_example/robot_input.mp4",

    // Inference settings:
    "guidance": 3,

    // Depth control settings
    "depth": {
        // Path to the control video
        // If a control is not provided, it will be computed on the fly.
        "control_path": "assets/robot_example/depth/robot_depth.mp4",

        // Control weight for the depth control
        "control_weight": 0.5
    },

    // Edge control settings
    "edge": {
        // Path to the control video
        "control_path": "assets/robot_example/edge/robot_edge.mp4",
        // Default control weight of 1.0 for edge control
    },

    // Seg control settings
    "seg": {
        // Path to the control video
        "control_path": "assets/robot_example/seg/robot_seg.mp4",

        // Control weight for the seg control
        "control_weight": 1.0
    },

    // Blur control settings
    "vis":{
        // Control video computed on the fly
        "control_weight": 0.5
    }
}
```

如果只希望在部分区域使用控制输入，可以为相应控制输入模态定义 MP4 格式的时空二值掩码。白色像素表示在对应区域使用控制信号，黑色像素表示不使用。示例如下：

```jsonc
{
    "depth": {
        "control_path": "assets/robot_example/depth/robot_depth.mp4",
        "mask_path": "/path/to/depth/mask.mp4",
        "control_weight": 0.5
    }
}
```

使用蒸馏模型进行推理时，需要在 Transfer 2.5 推理的基础上做两项修改：（1）在 JSON 文件中指定采样步数 `num_steps`（或使用命令行参数 `--num-steps`），蒸馏模型采用 4 个采样步训练；（2）在推理命令中指定 `--model=edge/distilled`。注意，蒸馏模型面向短视频，采样帧数严格为 93 帧。

Transfer 2.5 边缘控制蒸馏模型的 JSON 文件示例：

```jsonc
{
    "name": "robot_edge",
    "prompt_path": "/path/to/prompt/robot_prompt.txt",
    "video_path": "/path/to/input/robot_input.mp4",
    "guidance": 3,
    "num_steps": 4,
    "edge": {
        "control_path": "/path/to/edge/robot_edge.mp4",
        "control_weight": 1.0
    }
}
```

运行 Transfer 2.5 边缘控制蒸馏模型推理的命令示例：

```
# 8 GPUs
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py \
    -i assets/robot_example/distilled/edge/robot_edge_spec.json \
    -o outputs/distilled/edge \
    --model=edge/distilled

# 1 GPU
python examples/inference.py \
    -i assets/robot_example/distilled/edge/robot_edge_spec.json \
    -o outputs/distilled/edge \
    --model=edge/distilled
```

<a id="outputs"></a>

## 输出
<a id="multi-control"></a>

### 多控制
https://github.com/user-attachments/assets/337127b2-9c4e-4294-b82d-c89cdebfbe1d
