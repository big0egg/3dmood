<a id="cosmos-transfer25-2b-world-generation-with-guided-generation"></a>

# Cosmos-Transfer2.5-2B：通过引导生成实现世界生成
本指南介绍如何使用引导生成进行推理。引导生成无需额外训练模型，就能将仿真视频转换为具有真实感的画面，实现域随机化，并通过多种控制输入保持结构一致性。具体而言，先将仿真帧编码到模型的潜在空间，再在去噪过程中施加空间约束，选择性地固定手术器械、机械臂或人体仿真模型等重要区域，同时让场景其余部分自由生成。这样，模型可以改善整体真实感（光照、纹理、背景复杂度），并保留关键前景元素的几何结构和身份特征。

<a id="pre-requisites"></a>

### 前置条件
按照[安装指南](setup.md)完成环境配置、检查点下载，并确认满足硬件要求。

<a id="guided-generation-inference-with-pre-trained-cosmos-transfer25-models"></a>

## 使用预训练 Cosmos-Transfer2.5 模型进行引导生成推理
各个控制变体均可在单 GPU 上运行：

```bash
python examples/inference.py -i assets/humanoid_example/seg/humanoid_seg_guided_spec.json -o outputs/seg_guided_generation
```

若要对单个控制变体进行多 GPU 推理，或运行多个控制变体，请使用 [torchrun](https://docs.pytorch.org/docs/stable/elastic/run.html)：

```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/inference.py -i assets/humanoid_example/seg/humanoid_seg_guided_spec.json -o outputs/seg_guided_generation
```

我们为各个控制变体提供了参数文件示例：

| 变体 | 参数文件 |
| --- | --- |
| 分割 | `assets/humanoid_example/seg/humanoid_seg_guided_spec.json` |
| 深度 | `assets/humanoid_example/depth/humanoid_depth_guided_spec.json` |
| 边缘 | `assets/humanoid_example/edge/humanoid_edge_guided_spec.json` |
| 模糊 | `assets/humanoid_example/vis/humanoid_vis_guided_spec.json` |

运行以下命令可查看所有可用参数的说明：

```bash
python examples/inference.py --help

python examples/inference.py control:seg --help # for information specific to seg control
```

参数可以使用 JSON 格式指定：

```jsonc
{
    // Path to the prompt file, use "prompt" to directly specify the prompt
    "prompt_path": "assets/humanoid_example/humanoid_prompt.json",

    // Directory to save the generated video
    "output_dir": "outputs/humanoid_multicontrol",

    // Path to the input video
    "video_path": "assets/humanoid_example/humanoid_input.mp4",

    // Path to the binary mask video indicating the foreground elements for guided generation
    "guided_generation_mask": "assets/humanoid_example/humanoid_guided_mask.mp4",

    // Number of steps for guided generation. Using more guidance steps provides stronger guidance of foreground
    // elements in the generated videos. By default, 25 steps are used for guided generation.
    "guided_generation_step_threshold": 25,

    // Inference settings:
    "guidance": 3,

    // Depth control settings
    "depth": {
        // Path to the control video
        // If a control is not provided, it will be computed on the fly.
        "control_path": "assets/humanoid_example/depth/humanoid_depth.mp4",

        // Control weight for the depth control
        "control_weight": 0.5
    },

    // Seg control settings
    "seg": {
        // Path to the control video
        "control_path": "assets/humanoid_example/seg/humanoid_seg.mp4",

        // Control weight for the seg control
        "control_weight": 0.5
    },

    // Edge control settings
    "edge": {
        // Control video computed on the fly
        "control_weight": 0.0
    },

    // Blur control settings
    "vis":{
        // Control video computed on the fly
        "control_weight": 0.0
    }
}
```

<a id="example-input"></a>

### 输入示例
<table>
  <tr>
    <th>输入视频</th>
    <th>输入分割控制信号</th>
    <th>引导生成掩码</th>
  </tr>
  <tr>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/9fb7f172-97a9-4ab1-92e6-cf4ba1b6656f" width="100%" controls></video>
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/55ebd525-6226-4dfd-9219-956e33f988e1" width="100%" controls></video>
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/7caf916b-59ea-490e-a63a-484dd29cbdbe" width="100%" controls></video>
    </td>
  </tr>
</table>

<a id="example-output"></a>

### 输出示例
<table>
  <tr>
    <th>输出视频</th>
  </tr>
  <tr>
    <td valign="middle" width="60%">
      <video src="https://github.com/user-attachments/assets/7ee3e5ff-1b36-47ee-99e2-45c77041f555" width="60%" controls></video>
    </td>
  </tr>
</table>

<a id="image-to-image"></a>

## 图像到图像
我们还提供了图像到图像的参数文件示例：

```bash
python examples/inference.py -i assets/image_example/image2image_guided.json -o outputs/image2image_guided
```

<a id="example-input-1"></a>

### 输入示例
<table>
  <tr>
    <th>输入图像</th>
    <th>输入边缘控制信号</th>
    <th>引导生成掩码</th>
  </tr>
  <tr>
    <td valign="middle" width="33%">
      <image src="https://github.com/user-attachments/assets/31a836d1-2dcc-4b3f-bb88-e6320b62c9c2" width="100%" controls></image>
    </td>
    <td valign="middle" width="33%">
      <image src="https://github.com/user-attachments/assets/1a8f526d-baa8-4b53-b2fe-f203f75c56c0" width="100%" controls></image>
    </td>
    <td valign="middle" width="33%">
      <image src="https://github.com/user-attachments/assets/1272086d-504a-426c-9ebb-a890ea12cbe5" width="100%" controls></image>
    </td>
  </tr>
</table>

<a id="example-output-1"></a>

### 输出示例
<table>
  <tr>
    <th>输出图像</th>
  </tr>
  <tr>
    <td valign="middle" width="60%">
      <image src="https://github.com/user-attachments/assets/36e66752-5492-47d7-9c0e-28c725c53cc7" width="60%" controls></image>
    </td>
  </tr>
</table>
