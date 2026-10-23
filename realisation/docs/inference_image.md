<a id="image-inference-guide"></a>

# 图像推理指南
Cosmos-Transfer2.5 图像推理可以将模型应用于单帧图像，也可以使用图像作为风格参考，对控制视频进行生成。本指南介绍前置配置、图像到图像与风格参考工作流示例、相关 JSON 参数，以及用于多 GPU 扩展的 torchrun 命令。

<a id="prerequisites"></a>

## 前置条件
1. 按照[安装指南](setup.md)完成环境配置、检查点下载，并确认满足硬件要求。

<a id="image-to-image"></a>

## 图像到图像
使用控制信号和文本提示词变换单张图像或视频帧：

```bash
python examples/inference.py -i assets/image_example/image2image.json -o outputs/image2image
```

<a id="image-prompt-using-an-image-as-a-style-reference"></a>

## 图像提示：将图像作为风格参考
**有关图像提示的详细说明和示例，请参阅 Cosmos Cookbook 中的[风格引导推理](https://nvidia-cosmos.github.io/cosmos-cookbook/recipes/inference/transfer2_5/inference-image-prompt/inference.html)教程。**

使用图像作为风格参考，引导视频生成特定的视觉风格。

```bash
python examples/inference.py -i assets/image_example/image_style.json -o outputs/image_style
```

也可以使用 torchrun 进行多 GPU 推理：

```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py -i assets/image_example/image_style.json -o outputs/image_style/
```

运行以下命令可查看所有可用参数的说明：

```bash
python examples/inference.py --help

python examples/inference.py control:edge --help # for information specific to edge control
```

<a id="configuration"></a>

## 配置
<a id="image-to-image-1"></a>

### 图像到图像
```jsonc
{
    "name": "image_to_image",
    "prompt": "A scenic drive unfolds along a coastal highway...",

    // The input video. We'll extract the {max_frames} frames from the video.
    "video_path": "coastal_highway.mp4",
    "max_frames": 1,

    // Generate only the first frame
    "num_video_frames_per_chunk": 1,

    "seed": 1,
    "edge": {}  // Control computed on the fly
}
```

<a id="image-prompt"></a>

### 图像提示
```jsonc
{
    "name": "image_style",
    "prompt": "The camera moves steadily forward...",

    // Input video that determines the control signals for the generation
    "video_path": "calm_street.mp4",

    // Reference image that determines the style of the generated video
    "image_context_path": "sunset.jpg",

    "seed": 1,
    "edge": {}
}
```

<a id="examples"></a>

## 示例
<a id="image-prompt-1"></a>

### 图像提示
<table>
  <tr>
    <th>输入视频</th>
    <th>参考图像</th>
    <th>输出视频</th>
  </tr>
  <tr>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/3d7e8e93-b190-470c-8eb7-8b1ccf38904b" width="100%" controls></video>
    </td>
    <td valign="middle" width="33%">
      <img src="https://github.com/user-attachments/assets/1c00a2c2-99a6-4fd7-a0bf-79333ac227c6" width="100%" alt="参考图像">
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/a925ec8e-6005-4e14-9c05-58597a9be61c" width="100%" controls></video>
    </td>
  </tr>
</table>
