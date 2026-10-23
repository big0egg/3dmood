<a id="auto-multiview-inference-guide"></a>

# 自动驾驶多视角推理指南
<a id="prerequisites"></a>

## 前置条件
1. 按照[安装指南](setup.md)完成环境配置、检查点下载，并确认满足硬件要求。

<a id="exploratory-using-less-gpus"></a>

## 使用更少的 GPU（探索性支持）
GPU 数量（上下文并行规模）必须**大于或等于配置中的活动视角数量**。
活动视角是指提供了 `control_path` 的相机条目（例如 `front_wide`、`rear_left` 等）。
默认示例配置启用了七个视角，因此使用七块 GPU 运行。
减少 JSON 配置中的视角数量后，可以使用更少的 GPU。
执行下列命令前，请相应调整 `--nproc_per_node`（或总进程数 world size）。

<a id="examples"></a>

## 示例
多视角示例需要 8 块 GPU。

运行 multiview2world：

```bash
torchrun --nproc_per_node=8 --master_port=12341 examples/multiview.py -i assets/multiview_example/multiview_spec.json -o outputs/multiview/
```

默认输出为一个将所有视角并排拼接的视频。在参数 JSON 中设置 `"save_combined_views": false`，则会为每个相机视角分别保存 MP4 文件，并额外保存一个将所有视角组合在一起的 3×3 网格视频。

运行以下命令可查看所有可用参数的说明：

```bash
python examples/multiview.py --help

python examples/multiview.py control:view-config --help # for information specific to view configuration
```

运行自回归多视角生成（用于生成更长的视频）：

```bash
torchrun --nproc_per_node=8 --master_port=12341 -m examples.multiview -i assets/multiview_example/multiview_autoregressive_spec.json -o outputs/multiview_autoregressive
```

<a id="end-to-end-multiview-example"></a>

## 端到端多视角示例
本示例展示从三维场景标注到多视角视频输出的完整流程。首先将场景标注（物体位置、相机标定、车辆轨迹）渲染为世界场景视频，再将其作为多视角生成的条件。本示例仅使用渲染得到的控制视频，不使用原始实拍视频。

**步骤 1：下载示例数据，准备生成世界场景视频**

详细步骤请参阅世界场景视频生成指南中的[完整示例](world_scenario_video_generation.md#complete-example)：

```bash exclude=true
# Download example data
wget -P assets https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/multiview_example1.zip && unzip -oq assets/multiview_example1.zip -d assets
```

**步骤 2：生成世界场景视频**

```bash exclude=true
# See world_scenario_video_generation.md for detailed instructions
python scripts/generate_control_videos.py -i assets/multiview_example1/scene_annotations -o outputs/multiview_example1_world_scenario_videos
```

**步骤 3：使用生成的世界场景视频进行推理**

```bash exclude=true
# Run inference (num_conditional_frames=0 since we're not using raw footage)
torchrun --nproc_per_node=8 --master_port=12341 -m examples.multiview -i assets/multiview_example1/multiview_spec.json -o outputs/multiview_e2w/
```
