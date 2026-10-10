# ioutest：ARKitScenes 三维检测评测模块

模块根目录：`/mnt/workspace/3dmood/ioutest`。

模块已提供数据准备、3D-MOOD 推理适配器、三维 IoU AP、米制三维中心距离 AP、断点续跑和离线预测导入。采样清单与 GT 只生成一次，各模型分别保存预测和报告。首版只内置 3D-MOOD；其他模型可按 [模型接口](#模型接口)自行接入。

数据和指标协议、类别提示词及训练重叠说明见 [评测协议](PLAN.md)。当前真实模型 AP 尚未生成；不能将测试中的模拟预测结果当作模型效果。

## 目录与依赖

```text
ioutest/
├── README.md / PLAN.md
├── schema.py                     # ModelInput / Detection / ModelAdapter
├── dataset.py                    # ARKitScenes ZIP、采样和逐帧 GT
├── geometry.py / metrics.py      # 完整旋转框 IoU 与两类 AP
├── pipeline.py                   # 缓存、恢复、导入和报告
├── __main__.py                   # python -m ioutest
├── adapters/mood3d.py             # 3D-MOOD 适配器
├── visualize.py / tests/
└── runs/                         # 本地结果；已设置 Git 忽略
```

`prepare`、`evaluate`、`visualize` 和预测导入只需 Python、NumPy、SciPy、Pillow；`infer --adapter 3dmood` 使用现有项目环境。核心评测不会导入 torch、Vis4D 或 `opendet3d`。当前虚拟环境已具备所需依赖，无需重建或升级它。

所有以下命令在项目根目录执行：

```bash
source /mnt/workspace/venvs/3d-mood-ppu/bin/activate
cd /mnt/workspace/3dmood
export MPLBACKEND=Agg
export MPLCONFIGDIR=/mnt/workspace/3dmood/ioutest/cache/matplotlib
mkdir -p "$MPLCONFIGDIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python -m ioutest --help
```

沿用成功运行 Demo 时的 Hugging Face 缓存配置。不要将 `HF_HOME` 改到尚未准备 BERT 的空目录。检测权重默认直接读取 `/mnt/workspace/.cache/3d-mood/torch/hub/checkpoints/gdino3d_swin-b_120e_omni3d_834c97.pt`，不在线下载。

## 首次执行顺序

### 1. 检查设备与离线模型

在可访问 PPU 的 **DSW 原生 Terminal** 中执行：

```bash
python -m ioutest doctor
```

确认 `ready: true`、`cuda_available: true`，权重和 BERT 缓存检查通过。设备不可用时程序会提前停止，不会自动切换到 CPU 做长时间推理。仅检查 CPU 数据/几何环境可使用 `doctor --device cpu`；这不能证明 PPU 推理通过。

### 2. 先准备小规模清单并看 GT 投影

```bash
python -m ioutest prepare \
  --run-dir ioutest/runs/arkit-smoke-v1 \
  --max-scans 10 \
  --frames-per-scan 10

python -m ioutest visualize \
  --run-dir ioutest/runs/arkit-smoke-v1 \
  --max-frames 10
```

扫描子集使用固定种子，并覆盖图像方向和不同 `visit_id`。图片输出到对应运行目录的 `visualizations/groundtruth/`，绿色为有效 GT。检查图像转正与框投影是否符合画面；可用 `--show-ignore` 显示灰色忽略框。

### 3. 小规模推理并计算 AP

```bash
python -m ioutest infer \
  --run-dir ioutest/runs/arkit-smoke-v1 \
  --model-id 3dmood-swin-b

python -m ioutest evaluate \
  --run-dir ioutest/runs/arkit-smoke-v1 \
  --model-id 3dmood-swin-b

python -m ioutest visualize \
  --run-dir ioutest/runs/arkit-smoke-v1 \
  --model-id 3dmood-swin-b \
  --max-frames 10
```

红色为预测。可视化默认只画分数 ≥0.1 的预测；这只影响显示，AP 使用完整预测缓存。`timing.json` 给出首帧之后的平均预测耗时及大规模预测时间估计。

推理时只加载一次模型，每完成一帧就原子写入对应扫描分片。中断后重跑同一条 `infer` 命令，会复用已完成帧。空检测必须保存 `status: "ok"` 与 `detections: []`；失败帧不作为空检测，未完成的预测不允许进入正式报告。

### 4. 正式评测全部验证扫描

```bash
python -m ioutest prepare \
  --run-dir ioutest/runs/arkit-validation-v1 \
  --frames-per-scan 20

python -m ioutest infer \
  --run-dir ioutest/runs/arkit-validation-v1 \
  --model-id 3dmood-swin-b

python -m ioutest evaluate \
  --run-dir ioutest/runs/arkit-validation-v1 \
  --model-id 3dmood-swin-b
```

数据准备可断点续跑；已有相同协议的完整清单会校验后复用。改变场景、切帧、可见性或实现版本时，使用新的 `--run-dir`。改变模型、权重或适配器配置时，使用新的 `--model-id`。

也可用 `run` 合并三步，例如：

```bash
python -m ioutest run \
  --run-dir ioutest/runs/arkit-pilot-v1 \
  --max-scans 10 \
  --frames-per-scan 10 \
  --model-id 3dmood-swin-b
```

原始 `/mnt/data/openBenchmark/ARKitScenes` 只读，直接读取 ZIP 中所选帧，不整体解压验证集，也不修改原始文件。

## 指标与结果

默认报告：

| 指标 | 匹配方式 | 阈值 |
| --- | --- | --- |
| AP3D | 完整有向框的三维体积 IoU | 0.25、0.50 |
| AP_center | 三维中心欧氏距离，单位米 | 0.25m、0.50m、1.00m |

报告中 `summary.iou.map` / `summary.center.map` 是类别与对应阈值的平均值，`map_per_threshold` 是单个阈值的类别平均值。AP 范围为 0～1；无 GT 类别为 `null` / CSV 空值，报告列明实际参与平均的类别数。

指标采用 101 个召回位置的插值精度；同帧、同类、按分数降序一对一匹配。有效 GT 优先，随后匹配忽略 GT；ignore 匹配也保持一对一，重复预测不会无限吸收。距离采用固定米制门槛，不使用原项目的物体尺寸归一化阈值。

目录示例：

```text
ioutest/runs/arkit-validation-v1/
├── protocol.json                 # 固定协议与 benchmark_id
├── dataset_audit.json            # 数量、配对问题与官方 Training visit 重叠
├── frames.jsonl                  # 共享的、包含空 GT 帧的采样清单
├── groundtruth/<video_id>.json
├── models/3dmood-swin-b/
│   ├── model.json                # 权重/代码摘要、预处理与模型配置
│   ├── predictions/<video_id>.json
│   ├── timing.json
│   └── reports/
│       ├── metrics.json
│       ├── metrics_per_class.csv
│       └── metrics_per_scan.csv
└── visualizations/
```

`metrics.json` 同时报告完整范围和 `official_training_visit_disjoint` 子集。后者只排除官方 metadata 中与 Training 共享 visit 的扫描，**不等于已经审计模型全部训练数据**。默认 `training_audit.status=not_supplied`，不会宣称未见场景泛化。

已有模型训练所用的 Omni3D ARKitScenes train/val 标注时，在 `infer` 额外传入 `--training-annotations /path/ARKitScenes_train.json /path/ARKitScenes_val.json`。模块审计扫描与 visit，并增加 `declared_model_training_disjoint` 子集。声明仅覆盖所提供的训练清单。

修改阈值只需重新计算指标：

```bash
python -m ioutest evaluate \
  --run-dir ioutest/runs/arkit-validation-v1 \
  --model-id 3dmood-swin-b \
  --iou-thresholds 0.15 0.25 0.50 \
  --distance-thresholds 0.25 0.50 1.00 \
  --report-name alternative-thresholds
```

## 模型接口

接口定义在 [schema.py](schema.py)，模型依赖只放在自己的适配器中。后续接入其他模型无需修改数据准备和 AP 代码。

```python
from ioutest.schema import Detection, ModelInput

class YourAdapter:
    def __init__(self, options: dict):
        self.options = options

    def load(self) -> dict:
        # 一次加载模型，返回可 JSON 序列化的模型/权重/预处理信息。
        ...

    def predict(self, sample: ModelInput) -> list[Detection]:
        # 模型自身 resize/pad 与输出坐标转换在此完成。
        ...

    def close(self) -> None:
        ...
```

`ModelInput` 只有 `frame_id`、转正后的 `rgb`（RGB uint8 H×W×3）、对应的 `intrinsics`、统一 `class_names` 与 `prompts`。不提供深度、轨迹、GT 或当前帧实际类别列表。

每个 `Detection` 必须满足：

- `class_id`：官方 17 类在 `schema.CLASSES` 中的整数索引，适配器自行映射源模型类别。
- `score`：有限数值，越大越可信；不强制不同模型分数校准一致。
- `center`：三维中心，单位米，坐标为**转正后的 RGB 相机坐标**，x 右、y 下、z 前。
- `size`：沿框局部 x/y/z 的完整边长；不是半长，也不隐含模型自己的 w/l/h 顺序。
- `rotation`：3×3 右手正交矩阵，将框局部列向量变换到上述相机坐标。

框角点由评测器统一生成：`corner = rotation @ local_corner + center`。适配器必须把世界坐标、四元数、yaw、毫米或原图旋转等源约定转换到此接口。模型自身 NMS、分数过滤和候选上限须记录在 `load()` 信息中。

自定义适配器保持其 Python 模块可导入，通过 `--adapter your_module:YourAdapter --adapter-options /path/options.json` 接入。`options.json` 是传给构造函数的字典。各模型用不同 `model-id`，共享同一个 `run-dir` 和 benchmark 清单。

### 离线预测导入

也可以让模型在自己的环境中推理，再输出 JSONL，每帧一行，必须覆盖 `frames.jsonl` 全部帧，包括空预测：

```json
{"frame_id":"41069021:41069021_305.377","status":"ok","detections":[{"class_id":13,"score":0.9,"center":[0.0,0.0,2.0],"size":[1.0,1.0,1.0],"rotation":[[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]}]}
```

这里的帧 ID 仅展示格式；实际必须逐字使用当前清单的 ID。从 `protocol.json` 取 `benchmark_id` 后导入：

```bash
BENCHMARK_ID=$(python -c 'import json; print(json.load(open("ioutest/runs/arkit-validation-v1/protocol.json"))["benchmark_id"])')
python -m ioutest import-predictions \
  --run-dir ioutest/runs/arkit-validation-v1 \
  --model-id my-model \
  --path /absolute/path/predictions.jsonl \
  --benchmark-id "$BENCHMARK_ID"

python -m ioutest evaluate \
  --run-dir ioutest/runs/arkit-validation-v1 \
  --model-id my-model
```

同一份清单、GT、类别和指标阈值才能比较模型。导入会检查签名、每帧状态、重复/缺失 ID、有限坐标、正尺寸和旋转矩阵，不会悄悄跳过不合格预测。

## 验证与当前状态

```bash
python -B -m pytest -q --disable-warnings -p no:cacheprovider ioutest/tests
```

已验证的内容包括解析 IoU、任意三轴旋转、刚体变换不变性、与已有 Vis4D IoU 算子的数值对照、四种图像方向的投影一致性、3D-MOOD w/l/h 尺寸转换、米制距离匹配、空/重复预测、ignore、同分排序、清单完整性和断点恢复。

真实数据的 Up/Left/Down 三个扫描已准备 15 帧，生成并检查了 GT 投影图：`runs/geometry-smoke-v1/visualizations/groundtruth/`。该清单用于几何检查，正式评测另用 `arkit-validation-v1`。

正式清单 `runs/arkit-validation-v1/` 已完成准备并通过全部摘要校验和重复执行验证：549 个扫描、183 个 visit、10,980 帧、19,147 个有效逐帧 GT，覆盖全部 17 类；3,166 帧没有有效 GT，仍保留用于误检评测。采样前无文件/轨迹配对问题，无排除扫描。目录约 156 MB，原始 ZIP 保持不变。详细计数见该目录的 `preparation_summary.json`。正式清单已就绪，可以直接执行第 4 步中的 `infer` 与 `evaluate`；再次执行 `prepare` 会验证并复用它。

20 项测试已通过；完整旋转 CPU IoU 与现有 Vis4D 算子的随机框对照通过。3D-MOOD 本地模型/权重加载和一个真实 ARKitScenes 帧的 CPU 推理也已验证，输出 28 个符合统一接口的三维预测，记录在 `runs/geometry-smoke-v1/cpu_adapter_check.json`。这用于验证适配器，并非完整评测。

当前工具进程无法访问 PPU；PPU 批量推理及正式 AP 还需在 DSW 原生 Terminal 执行上面的 `doctor → infer → evaluate`。模块没有用模拟预测填写真实模型报告。
