<a id="world-scenario-video-generation"></a>

# 世界场景视频生成
本工具根据三维场景标注为 Cosmos Transfer2.5 生成控制视频。它将三维元素（折线、多边形和长方体）投影到相机视角，将世界模型渲染为视频。

**支持的输入格式：**

- **Parquet 格式**：保存在 Parquet 文件中的结构化场景标注，参见[世界场景 Parquet 文件结构](world_scenario_parquet.md)。
- **RDS-HQ 格式**：[Cosmos-Drive-Dreams 数据集](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams)使用的 NVIDIA 内部格式。

<a id="table-of-contents"></a>

## 目录
<!--TOC-->

- [目录](#table-of-contents)
- [快速入门](#getting-started)
  - [运行要求](#requirements)
  - [环境配置](#setup)
  - [快速开始](#quick-start)
  - [示例 1：Parquet 输入](#example-1-parquet-input)
  - [示例 2：RDS-HQ 输入](#example-2-rds-hq-input)
- [使用方法](#usage)
  - [基本命令](#basic-commands)
  - [选项](#options)
  - [可用相机](#available-cameras)
- [数据格式](#data-format)
  - [输入结构](#input-structure)
  - [输出结构](#output-structure)
- [渲染内容](#what-gets-rendered)
- [故障排查](#troubleshooting)
  - [常见问题](#common-issues)
- [集成](#integration)
- [渲染规范](#rendering-specifications)
  - [配色](#color-palette)
  - [动态物体](#dynamic-objects)
  - [车道线](#lanelines)
  - [交通信号灯](#traffic-lights)
  - [地图元素](#map-elements)
- [流程概览](#pipeline-overview)
  - [帧率配置](#frame-rate-configuration)
  - [1. 插值](#1-interpolation)
  - [2. 坐标系](#2-coordinate-systems)
  - [3. 渲染过程](#3-rendering-process)

<!--TOC-->

---

<a id="getting-started"></a>

## 快速入门
<a id="requirements"></a>

### 运行要求
- Python 3.10+
- UV（用于依赖管理）
- 支持 EGL 的 GPU（用于无显示器环境下的 OpenGL 渲染）
- Parquet 或 RDS-HQ 格式的三维场景标注数据

<a id="setup"></a>

### 环境配置
首先配置环境：

```bash
uv sync
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

<a id="quick-start"></a>

### 快速开始
根据三维场景标注生成世界场景视频：

```bash
python scripts/generate_control_videos.py -i /path/to/input_root -o /path/to/save_root
```

脚本会自动判断输入采用 Parquet 还是 RDS-HQ 格式。

<a id="example-1-parquet-input"></a>

<a id="complete-example"></a>

### 示例 1：Parquet 输入
以下是可直接复制运行的完整示例：

```bash
# Download example data
wget -P assets https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/multiview_example1.zip && unzip -oq assets/multiview_example1.zip -d assets

# Generate control videos for the example scene
python scripts/generate_control_videos.py -i assets/multiview_example1/scene_annotations -o outputs/multiview_example1_world_scenario_videos
```

还可以下载以下示例：

```bash
wget https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/multiview_example2.zip
wget https://github.com/nvidia-cosmos/cosmos-dependencies/releases/download/assets/multiview_example3.zip
```

<a id="example-2-rds-hq-input"></a>

### 示例 2：RDS-HQ 输入
使用 [Cosmos-Drive-Dreams 数据集](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams)中的数据，RDS-HQ 格式详情请参阅数据集文档：

```bash
wget -P scripts https://raw.githubusercontent.com/nv-tlabs/Cosmos-Drive-Dreams/main/scripts/download.py
python scripts/download.py --odir ./assets/rdshq-data --limit 1
python scripts/generate_control_videos.py -i assets/rdshq-data -o outputs/rdshq-generated
```

<a id="usage"></a>

## 使用方法
<a id="basic-commands"></a>

### 基本命令
```bash
# All cameras (default)
python scripts/generate_control_videos.py -i {input_root}/ -o {save_root}/

# Specific cameras
python scripts/generate_control_videos.py -i {input_root}/ -o {save_root}/ \
    --cameras "camera:front:wide:120fov,camera:cross:right:120fov"
```

<a id="options"></a>

### 选项
| 选项 | 默认值 | 说明 |
|--------|---------|-------------|
| `--cameras` | `all` | 相机名称；使用 "all" 可处理全部 7 个相机 |

<a id="available-cameras"></a>

### 可用相机
- `camera:front:wide:120fov`
- `camera:front:tele:sat:30fov`
- `camera:cross:right:120fov`
- `camera:cross:left:120fov`
- `camera:rear:left:70fov`
- `camera:rear:right:70fov`
- `camera:rear:tele:30fov`

<a id="data-format"></a>

## 数据格式
<a id="input-structure"></a>

### 输入结构
工具支持两种输入格式：

**1. Parquet 格式**：在 Parquet 文件中存储结构化场景标注：

```
scene_annotations_directory/
├── uuid.obstacle.parquet              (required)
├── uuid.calibration_estimate.parquet  (required)
├── uuid.egomotion_estimate.parquet    (required)
├── uuid.lane.parquet                  (optional)
├── uuid.lane_line.parquet             (optional)
└── ... (other optional parquet files)
```

详细数据结构和要求请参阅[世界场景 Parquet 文件结构](world_scenario_parquet.md)。

**2. RDS-HQ 格式**：NVIDIA 的录制格式，包含传感器数据和标注。脚本会自动提取所需场景信息。格式详情请参阅 [Cosmos-Drive-Dreams 数据集文档](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicle-Cosmos-Drive-Dreams)。

<a id="output-structure"></a>

### 输出结构
两种输入格式生成相同的输出结构：

```
save_root/
└── uuid/
    ├── uuid.camera_front_wide_120fov.mp4
    ├── uuid.camera_front_tele_sat_30fov.mp4
    ├── uuid.camera_cross_right_120fov.mp4
    ├── uuid.camera_cross_left_120fov.mp4
    ├── uuid.camera_rear_left_70fov.mp4
    ├── uuid.camera_rear_right_70fov.mp4
    ├── uuid.camera_rear_tele_30fov.mp4
```

<a id="what-gets-rendered"></a>

## 渲染内容
**始终渲染：**车辆、行人和其他动态物体的三维包围框。

**有数据时可选渲染：**

- 车道线、车道和道路边界。
- 人行横道、道路标记和停止线。
- 杆体、交通信号灯和交通标志。

<a id="troubleshooting"></a>

## 故障排查
<a id="common-issues"></a>

### 常见问题
| 问题 | 解决方法 |
|-------|----------|
| **ModernGL/EGL 错误** | 安装 GPU 驱动和 EGL 库。Ubuntu/Debian：`apt install libegl1-mesa-dev libgl1-mesa-dri` |
| **缺少 Parquet 文件** | 确保所需文件存在：`obstacle`、`calibration_estimate`、`egomotion_estimate` |
| **内存问题** | 使用 `--cameras` 减少同时处理的相机数量 |
| **相机名称无效** | 使用 `--help` 查看有效选项 |

<a id="integration"></a>

## 集成
生成的控制视频作为 Cosmos Transfer2.5 模型推理的条件输入，通过高精地图可视化提供空间上下文，实现可控视频生成。

<a id="rendering-specifications"></a>

## 渲染规范
本渲染规范基于 [Cosmos Drive Dreams](https://github.com/nv-tlabs/Cosmos-Drive-Dreams)。

<p align="center">
  <img src="../assets/docs/world_scenario_image.png" alt="世界场景图像" width="800">
</p>

<p align="center">
  <img src="../assets/docs/rendering_diagram.png" alt="渲染示意图" width="800">
</p>

<a id="color-palette"></a>

### 配色
<p align="center">
  <img src="../assets/docs/color_palette.png" alt="配色方案" width="800">
</p>

<a id="dynamic-objects"></a>

### 动态物体
**配置：**[`config_color_bbox.json`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/color_scheme/config_color_bbox.json)

动态物体渲染为实心三维长方体，带浅灰色边缘，并沿前后方向使用颜色渐变。

**标签映射：**[`bbox_utils.py`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/utils/bbox_utils.py) 将 DCP 标签映射为五类：

1. **汽车**：automobile、other_vehicle、vehicle。
2. **卡车**：heavy_truck、bus、train_or_tram_car、trailer。
3. **行人**：person。
4. **骑行者**：rider。
5. **其他**：protruding_object、animal、stroller。

<p align="center">
  <img src="../assets/docs/dynamic_objects.png" alt="动态物体" width="800">
</p>

<a id="lanelines"></a>

### 车道线
**配置：**[`config_color_geometry_laneline.json`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/color_scheme/config_color_geometry_laneline.json)

车道线根据以下属性分为 15 种类型：

- **颜色：**黄色、白色、其他。
- **样式：**实线、虚线、点线、实线与虚线组合。

**示例：**`yellow solid dashed` 表示沿折线方向，右侧为黄色实线，左侧为黄色虚线。

<p align="center">
  <img src="../assets/docs/lanelines.png" alt="车道线" width="800">
</p>

<a id="traffic-lights"></a>

### 交通信号灯
**配置：**[`config_color_traffic_light.json`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/color_scheme/config_color_traffic_light.json)

交通信号灯渲染为长方体，具有红、黄、绿和未知四种状态。

<p align="center">
  <img src="../assets/docs/traffic_lights.png" alt="交通信号灯" width="800">
</p>

<a id="map-elements"></a>

### 地图元素
**配置：**[`config_color_hdmap.json`](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/cosmos_transfer2/_src/imaginaire/auxiliary/world_scenario/color_scheme/config_color_hdmap.json)

地图元素使用三种几何类型：

- **折线**：杆体、道路边界、停止线。
- **多边形**：人行横道、道路标记。
- **长方体**：交通标志。

<p align="center">
  <img src="../assets/docs/map_elements.png" alt="地图元素" width="800">
</p>

<a id="pipeline-overview"></a>

## 流程概览
<a id="frame-rate-configuration"></a>

### 帧率配置
流程使用两项可配置的帧率：

- **`INPUT_POSE_FPS`**（默认 30fps）：插值处理帧率，决定生成多少帧。
- **`TARGET_RENDER_FPS`**（默认 30fps）：输出视频播放帧率，决定播放速度。

源数据通常为 10Hz，需要插值到处理帧率。

<a id="1-interpolation"></a>

### 1. 插值
- **障碍物数据**：从源频率（通常为 10Hz）插值到处理帧率。
- **自车运动数据**：插值到相同的处理帧率。
- **地图数据**：静态元素，只加载一次，不进行插值。

处理帧率 `INPUT_POSE_FPS` 决定时间分辨率，输出帧率 `TARGET_RENDER_FPS` 决定播放速度。两者通常均默认为 30fps。

<a id="2-coordinate-systems"></a>

### 2. 坐标系
- **世界坐标**：右手坐标系，x 向前、y 向左、z 向上。
- **相机坐标**：相机沿 z 轴正方向观察。
  - x 轴：向右。
  - y 轴：向下。
  - z 轴：向前，即指向场景内部。
- **FLU 约定**：前、左、上（Forward-Left-Up），用于车辆到相机的坐标变换。

<a id="3-rendering-process"></a>

### 3. 渲染过程
1. 加载相机标定。
2. 解析自车运动轨迹，并插值到处理帧率。
3. 对障碍物轨迹进行插值，使其与自车运动时间戳一致。
4. 将所有几何体从世界坐标转换到相机坐标。
5. 使用 OpenGL 渲染每一帧。
6. 将输出编码为 MP4 视频。
