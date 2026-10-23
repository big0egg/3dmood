<a id="plenoptic-multiview-inference-guide"></a>

# 全光场多视角推理指南
本指南介绍如何运行全光场多视角相机推理，从单个输入视频生成新的相机视角。

<a id="prerequisites"></a>

## 前置条件
1. 按照[安装指南](setup.md)完成环境配置、检查点下载，并确认满足硬件要求。

<a id="overview"></a>

## 概述
全光场多视角推理根据单个输入视频，生成新的相机视角视频。模型采用自回归生成方式，以之前生成的视角为条件，逐步创建新的相机视角。

<a id="supported-camera-motions"></a>

### 支持的相机运动
支持以下相机运动类型：

| 运动类型 | 说明 |
|-------------|-------------|
| `static` | 相机不移动（输入视角） |
| `rot_left` / `rot_right` | 绕竖直轴旋转 |
| `arc_left` / `arc_right` | 沿圆弧运动 |
| `azimuth_left` / `azimuth_right` | 方位角旋转 |
| `tilt_up` / `tilt_down` | 竖直方向俯仰 |
| `translate_up_rot` / `translate_down_rot` | 平移并旋转 |
| `elevation_up_1` / `elevation_up_2` | 改变高度 |
| `zoom_in` / `zoom_out` | 拉近／拉远 |
| `distance_away_1` / `distance_away_2` | 改变与主体的距离 |

<a id="input-format"></a>

## 输入格式
创建具有以下结构的 JSON 文件：

```json
{
    "name": "sample_001",
    "input_path": "videos/input.mp4",
    "prompt": "A scenic outdoor landscape with mountains and trees",
    "camera_sequence": ["static", "rot_left", "arc_right", "azimuth_right"],
    "seed": 1,
    "guidance": 7
}
```

<a id="parameters"></a>

### 参数
| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `name` | string | 当前样本的唯一标识符 |
| `input_path` | string | 输入视频路径（相对于 base_path） |
| `prompt` | string | 场景的文本描述 |
| `camera_sequence` | list[string] | 按生成顺序排列的相机运动列表 |
| `seed` | int | 用于复现结果的随机种子（默认：0） |
| `guidance` | int | 引导强度，范围 0～7（默认：7） |

<a id="cli-override-parameters"></a>

### 命令行覆盖参数
以下参数可以通过命令行覆盖：

| 参数 | 类型 | 说明 |
|-----------|------|-------------|
| `--focal_length` | int | 相机内参中的焦距，单位毫米（默认：24） |
| `--extrinsic_scale` | float | 相机外参的缩放系数（默认：1.5） |
| `--input_video_res` | string | 输入视频分辨率（仅支持 `480p`） |
| `--guidance` | int | 覆盖所有样本的引导强度 |
| `--seed` | int | 覆盖所有样本的随机种子 |

<a id="camera-parameter-ranges"></a>

### 相机参数范围
自定义相机参数时，以下范围已经过测试验证：

| 参数 | 最小值 | 最大值 | 说明 |
|-----------|-----|-----|-------------|
| **焦距（内参）** | 12 | 100 | 相机焦距，单位毫米（默认：24）。超出此范围可能产生伪影。 |
| **外参缩放系数** | 0.5 | 3.0 | 作用于相机平移和旋转的缩放系数，用于控制相对于输入模板的相机运动幅度。 |

> **注意：**使用 `focal_length` 参数时，`cameras/` 目录中必须存在对应的 `intrinsics_focal{focal_length}.txt` 文件。默认资产提供了 `intrinsics_focal24.txt` 和 `intrinsics_focal50.txt`。使用自定义焦距时，需要创建相应的内参文件。

<a id="examples"></a>

## 示例
<a id="single-gpu-inference"></a>

### 单 GPU 推理
```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 python examples/plenoptic.py \
    -i assets/plenoptic_example/sample.json \
    -o outputs/plenoptic/ \
    --base-path assets/plenoptic_example/
```

<a id="multi-gpu-inference-8-gpus"></a>

### 多 GPU 推理（8 块 GPU）
为获得最佳性能，请使用 8 块 GPU 进行上下文并行推理（从 `torchrun` 自动检测）：

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1  torchrun --nproc_per_node=8 examples/plenoptic.py \
    -i assets/plenoptic_example/sample.json \
    -o outputs/plenoptic/ \
    --base-path assets/plenoptic_example/
```

<a id="batch-inference"></a>

### 批量推理
使用 JSONL 文件处理多个样本：

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 examples/plenoptic.py \
    -i assets/plenoptic_example/batch.jsonl \
    -o outputs/plenoptic_batch/ \
    --base-path assets/plenoptic_example/
```

<a id="custom-parameters"></a>

### 自定义参数
通过命令行覆盖默认参数：

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 torchrun --nproc_per_node=8 examples/plenoptic.py \
    -i assets/plenoptic_example/sample.json \
    -o outputs/plenoptic_custom/ \
    --base-path assets/plenoptic_example/ \
    --guidance=5 \
    --seed=42 \
    --focal_length=24
```

<a id="data-preparation"></a>

## 数据准备
<a id="directory-structure"></a>

### 目录结构
数据目录应具有以下结构：

```
base_path/
├── videos/
│   ├── input1.mp4
│   └── input2.mp4
├── cameras/
│   ├── static.txt (or static.pt)
│   ├── rot_left.txt
│   ├── rot_right.txt
│   ├── arc_left.txt
│   ├── arc_right.txt
│   ├── azimuth_left.txt
│   ├── azimuth_right.txt
│   ├── tilt_up.txt
│   ├── tilt_down.txt
│   ├── translate_up_rot.txt
│   ├── translate_down_rot.txt
│   ├── elevation_up_1.txt
│   ├── elevation_up_2.txt
│   ├── zoom_in.txt
│   ├── zoom_out.txt
│   ├── distance_away_1.txt
│   ├── distance_away_2.txt
│   ├── intrinsics_focal24.txt
│   └── intrinsics_focal50.txt
├── metadata.csv (optional)
└── sample.json
```

<a id="camera-file-formats"></a>

### 相机文件格式
相机外参支持两种格式（`.pt` 或 `.txt`）。代码优先尝试 `.pt`，若不存在则使用 `.txt`。

**外参（`.txt` 格式）：**每行包含一帧展平后的 3×4 相机位姿矩阵，共 12 个值：

```
R11 R12 R13 Tx R21 R22 R23 Ty R31 R32 R33 Tz
```

对于 24 个潜在帧，文件应有 24 行。`static.txt` 示例（单位位姿，无运动）：

```
1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 1.0 0.0
... (24 lines total)
```

**外参（`.pt` 格式）：**包含相机位姿矩阵、形状为 `[24, 4, 4]` 的 PyTorch 张量。

**内参（`intrinsics_focal*.txt`）：**单行包含 4 个值：

```
fx fy cx cy
```

其中，`fx`、`fy` 为焦距，`cx`、`cy` 为主点坐标。

<a id="creating-custom-camera-trajectories"></a>

### 创建自定义相机轨迹
`assets/plenoptic_example/cameras/` 中的示例资产包含预定义相机轨迹。创建自定义轨迹的方法如下：

1. **使用已有示例作为模板**：复制文件并修改其中的值。
2. **旋转轨迹**：逐帧改变旋转矩阵元素（R11、R13、R31、R33）。
3. **平移轨迹**：逐帧改变平移分量（Tx、Ty、Tz）。

参考轨迹请查看 `assets/plenoptic_example/cameras/` 中的示例文件。

<a id="sample-outputs"></a>

## 输出示例
以下示例展示了输入视频，以及不同相机运动对应的生成视频：

<table>
  <tr>
    <th>相机运动</th>
  </tr>
  <tr>
    <td valign="middle" width="33%">
      <code>rot_left</code><br>绕竖直轴向左旋转
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/de384ce4-b58e-4f86-82c3-3111e466689f" width="100%" controls></video>
    </td>
  </tr>
  <tr>
    <td valign="middle" width="33%">
      <code>rot_right</code><br>绕竖直轴向右旋转
    </td>
    <td valign="middle" width="33%">
      <video src="https://github.com/user-attachments/assets/8f65a028-b8f2-4533-a466-aabf841ab3eb" width="100%" controls></video>
    </td>
  </tr>
</table>

<a id="getting-help"></a>

### 获取帮助
查看可用参数的更多信息：

```bash exclude=true
COSMOS_EXPERIMENTAL_CHECKPOINTS=1 python examples/plenoptic.py --help
```

<a id="related-documentation"></a>

## 相关文档
- [安装指南](setup.md)
- [故障排查](troubleshooting.md)
- [自动驾驶多视角推理](inference_auto_multiview.md)
