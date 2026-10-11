# ioutest：ARKitScenes 三维检测评测模块

模块根目录：`/mnt/workspace/3dmood/ioutest`。

模块提供冻结数据清单、本地 RGB/深度缓存、3D-MOOD、WildDet3D、CuTR RGB/RGB-D 适配器、两类 AP、断点续跑和中文对比报告。采样清单与 GT 只生成一次，各模型在各自虚拟环境中顺序运行；不需要切换阿里云镜像。

数据和指标协议、类别提示词及训练重叠说明见 [评测协议](PLAN.md)。当前真实模型 AP 尚未生成；不能将测试中的模拟预测结果当作模型效果。

## 当前实例：四组统一评测

**数据已就绪：549 个扫描、183 个 visit、10,980 帧 RGB、同帧 ARKit 深度和逐帧 GT。不要重新运行 `prepare`。** RGB 原图与转正图片位于 `cache/arkit-validation-v1/raw_images/`、`images/`；深度位于 `depth_images/`，约 281 MiB。缓存摘要在每次使用时检查，输入不需要访问原始 ZIP。

默认运行配置见 [benchmark_config.json](benchmark_config.json)：

| 组别 | 项目 / 环境 | 输入 | 比较范围 |
|---|---|---|---|
| 3D-MOOD | `/mnt/workspace/3dmood` / `3d-mood-ppu` | RGB、内参、固定 17 类提示词 | RGB 几何 AP、17 类语义 AP |
| WildDet3D | `/mnt/workspace/wilddet3d` / `wilddet3d-ppu` | RGB、内参、固定 17 类提示词 | RGB 几何 AP、17 类语义 AP |
| CuTR RGB | `/mnt/workspace/CA1M` / `ca1m-ppu` | RGB、内参 | RGB 几何 AP |
| CuTR RGB-D | 同上 | RGB、内参、同帧 ARKit LiDAR 深度 | 单独与 CuTR RGB 对比 |

所有组使用相同帧 ID。RGB 主表采用类别无关匹配；17 类语义表只比较 3D-MOOD 与 WildDet3D。CuTR 的未知类别保存为 `class_id: null`，不从 GT 推断类别。IoU 阈值 0.25/0.50，中心距离阈值 0.25/0.50/1.00 米，均采用 101 点插值 AP。

在 **DSW 原生 Terminal** 运行：

```bash
cd /mnt/workspace/3dmood
/mnt/workspace/venvs/3d-mood-ppu/bin/python -B -m ioutest benchmark \
  --report-root "$HOME/report/arkitscenes-three-models-v1"
```

这条命令先检查四组环境、设备、权重、本地缓存，再完成固定 10 个不同 visit 扫描的 200 帧试评测并生成报告；全部成功后继续 10,980 帧全量评测。各模型在独立子进程中加载一次、顺序运行，上一模型退出后释放资源。整个流程共四组，不把 RGB-D 纳入纯 RGB 排名。当前配置沿用实际已有的 `/root/.cache/huggingface` BERT 缓存，不指向空缓存目录。

需要先查看试评测结果时，添加 `--phase pilot`；随后以相同报告目录执行 `--phase full`。中断后重复原命令，已完成帧会校验并复用，试评测的预测也能在全量阶段复用。模型/源码/环境/缓存摘要变化时拒绝混用旧预测，使用新的 `--report-root`。

可选的传感器重力实验：添加 `--with-gravity`，第五组为 **CuTR RGB + 已知重力**。只从相机姿态推导重力旋转，不读取 GT 框；结果单独输出 `comparison_gravity.csv`。默认四组 CuTR 不提供重力，沿用相机直立假设。

已经有完整预测时，可以在原配置与相同设备参数下仅重算评分：

```bash
/mnt/workspace/venvs/3d-mood-ppu/bin/python -B -m ioutest benchmark \
  --report-root "$HOME/report/arkitscenes-three-models-v1" \
  --phase full --score-only
```

### 输出位置

当前用户为 root，目标根目录为 `/root/report/arkitscenes-three-models-v1`：

```text
arkitscenes-three-models-v1/
├── README.md / experiment.json / status.json
├── models/<model-id>/
│   ├── model.json                    # 模型、权重、源码、输入、环境摘要
│   └── predictions/<scan>/*.json.gz   # 原子写入数值预测，续跑复用
├── pilot/                            # 200 帧试评测
└── full/                             # 10,980 帧正式评测
    ├── report.md / summary.json       # 中文报告和完成状态
    ├── comparison_rgb.csv            # 三模型 RGB 几何 AP
    ├── comparison_rgbd.csv           # CuTR RGB/RGB-D
    ├── comparison_17class.csv        # 两模型语义 AP
    ├── comparison_visit_disjoint.csv # 排除官方训练 visit 重叠的 RGB 几何 AP
    ├── runtime.csv
    ├── metrics/                      # 完整指标、逐类 AP、逐扫描计数
    ├── figures/                      # AP 图，PNG/SVG
    ├── visualizations/               # 同帧 GT/预测投影图
    └── logs/                         # 每组环境的运行日志
```

运行期间可查看 `status.json` 和 `pilot/logs/`、`full/logs/`。报告只对完整阶段生成；失败或缺失帧不会当作空预测计分。显示图片使用分数阈值 0.1，AP 使用全部保留候选。CSV 的 AP 单位是 0～1，中文报告为百分数。

### 模型输入与候选约定

3D-MOOD 沿用 Demo 的 800×1333 预处理；WildDet3D 沿用自身 1008 尺寸预处理。CuTR 将长边缩放至 1024，RGB 使用双线性插值，深度用最近邻，同步更新像素中心内参，再使用原生归一化/填充；RGB 与 RGB-D 使用完全相同的尺寸设置。原始低分辨率图像并不会因放大增加细节，因此不能直接对照论文 AP。

不使用 Demo 的展示分数阈值提前删框。3D-MOOD/WildDet3D 保留原生 NMS 与最多 300 个候选；CuTR 在 top-100 前只保留监督前景通道 0，另一通道不作为语义类别或物体预测。这些规则随模型摘要保存。

CuTR 构建阶段固定单个 CPU 线程，随后恢复推理线程数，严格加载全部模型参数；适配当前 PPU 镜像的 CPU 初始化行为。该设置不修改模型结构、权重或原项目文件。

RGB-D 使用 `lowres_depth` 的 uint16 毫米深度，读取后除以 1000 并与 RGB 一同转正；没有有效深度的像素保留为 0，后续交给 CuTR 原生深度预处理。评测器的 GT 可见性筛选与 RGB-D 的输入深度角色分开，模型不接收 GT 框、ignore 标记或当前帧真实类别。

官方训练 visit 与验证集有两扫描、40 帧重叠，报告另给出 10,940 帧的排除子集。各模型具体训练集重叠尚未完成审计，不将此子集宣称为所有模型未见场景。CuTR 检出的物体也可能超出 ARKitScenes 17 类标注范围，在本协议中可能计为误检，报告会保留这一边界说明。

整个 `cache/` 和 `runs/` 已被 Git 忽略；正式报告写入仓库外的 `~/report`。GitHub 只提交评测代码、配置和说明，不提交图片、深度、权重或数值预测。

## 目录与依赖

```text
ioutest/
├── README.md / PLAN.md
├── schema.py                     # ModelInput / Detection / ModelAdapter
├── dataset.py                    # ARKitScenes ZIP、采样和逐帧 GT
├── image_cache.py                # 提取所选图片、本地读取与完整性校验
├── depth_cache.py                # RGB-D 深度缓存与输入隔离
├── benchmark*.py / benchmark_config.json  # 多环境执行与对比报告
├── geometry.py / metrics.py      # 完整旋转框 IoU 与两类 AP
├── pipeline.py                   # 缓存、恢复、导入和报告
├── __main__.py                   # python -m ioutest
├── adapters/{mood3d,wilddet,cutr}.py  # 三项目四组适配器
├── visualize.py / tests/
├── cache/                        # 本地图片；已设置 Git 忽略
└── runs/                         # 本地结果；已设置 Git 忽略
```

`prepare`、`cache-images`、`evaluate`、`visualize` 和预测导入只需 Python、NumPy、SciPy、Pillow；`infer --adapter 3dmood` 使用现有项目环境。核心评测不会导入 torch、Vis4D 或 `opendet3d`。当前虚拟环境已具备所需依赖，无需重建或升级它。

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

## 单模型接口与新数据集的执行顺序

本节保留底层命令的用法。当前已冻结清单的四组评测直接使用上面的 `benchmark` 命令；新建数据清单才需要下面的 `prepare`。

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

python -m ioutest cache-images \
  --run-dir ioutest/runs/arkit-smoke-v1

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

**当前实例已有完整的 10,980 帧清单、GT 和本地图片，直接执行以下命令，无需重新 prepare 或提取图片。** `infer` 自动使用 `ioutest/cache/arkit-validation-v1/`：

```bash
python -m ioutest infer \
  --run-dir ioutest/runs/arkit-validation-v1 \
  --model-id 3dmood-swin-b

python -m ioutest evaluate \
  --run-dir ioutest/runs/arkit-validation-v1 \
  --model-id 3dmood-swin-b
```

数据准备可断点续跑；已有相同协议的完整清单会校验后复用。改变场景、切帧、可见性或实现版本时，使用新的 `--run-dir`。改变模型、权重或适配器配置时，使用新的 `--model-id`。

仅在新部署、尚无清单和图片的情况下，先执行以下准备命令，再执行上面的推理与评分：

```bash
python -m ioutest prepare \
  --run-dir ioutest/runs/arkit-validation-v1 \
  --frames-per-scan 20

python -m ioutest cache-images \
  --run-dir ioutest/runs/arkit-validation-v1
```

已有完整清单不因本次读取优化而重建。`prepare` 将代码版本纳入签名；现有冻结数据继续直接用于 `infer` / `evaluate`，需要重新准备时选择新的 `--run-dir`。

也可用 `run` 合并三步，例如：

```bash
python -m ioutest run \
  --run-dir ioutest/runs/arkit-pilot-v1 \
  --max-scans 10 \
  --frames-per-scan 10 \
  --model-id 3dmood-swin-b
```

原始 `/mnt/data/openBenchmark/ARKitScenes` 只读，准备 GT 和首次提取图片时读取 ZIP；缓存就绪后，推理和可视化直接读取本地 PNG，不再打开远程 ZIP。不整体解压验证集，也不修改原始文件。

## 本地图片缓存与 GitHub 提交

当前正式清单的 10,980 帧已保存于 `/mnt/workspace/3dmood/ioutest/cache/arkit-validation-v1/`，原图与转正图片合计约 1.39 GiB：

```text
cache/arkit-validation-v1/
├── raw_images/<video_id>/*.png    # 原始方向，用于模块推理
├── images/<video_id>/*.png        # 转正后的 RGB，可供其他项目使用
├── images.jsonl                  # 帧编号、两种路径、转正后的内参和像素摘要
└── cache_summary.json            # benchmark_id、完整性、清单摘要与体积
```

`infer`、`run` 和 `visualize` 默认检测 `ioutest/cache/<run-dir 名称>/`。存在缓存时，先检查 benchmark ID、清单摘要、全部帧编号和原图文件是否完整，再逐帧校验原图像素与冻结清单一致；读取 `raw_images` 后只转正一次，使用冻结清单的对应内参。`timing.json` 的 `input_source.kind` 为 `local_png` 时表示使用了本地缓存。

没有默认缓存目录时仍可读取原始 ZIP；已有但损坏或不完整的缓存会停止执行，不会悄悄改用其他数据。可以重新运行 `cache-images` 补齐中断时未完成的图片；已存在的图片会校验后复用。若需要自定义路径或明确使用 ZIP：

```bash
python -m ioutest infer \
  --run-dir ioutest/runs/arkit-validation-v1 \
  --image-cache /mnt/workspace/3dmood/ioutest/cache/arkit-validation-v1

python -m ioutest visualize \
  --run-dir ioutest/runs/arkit-validation-v1 \
  --no-image-cache \
  --max-frames 3
```

向其他模型提供 `images/` 时，使用 `images.jsonl` 中的配套内参，不要再按 `rotation_k` 转正；`rotation_k` 描述的是从 `raw_images/` 到 `images/` 的变换。

`ioutest/.gitignore` 忽略整个 `/cache/` 和 `/runs/`，包括数据图片、缓存清单、预测和可视化文件。它们保留在机器上，普通 `git add` 不会提交到 GitHub。仓库自带的 `assets/` 示例图不受影响。新克隆仓库需自行准备数据、生成评测清单，再运行 `cache-images`；不要使用 `git add -f` 强制加入缓存。

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

- `class_id`：语义模型使用官方 17 类的整数索引；类别无关模型可以为 `None`，仅在 `benchmark` 几何 AP 中使用。旧的单模型 `infer/evaluate/import-predictions` 仍要求 17 类语义 ID。
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

正式清单 `runs/arkit-validation-v1/` 已完成准备并通过全部摘要校验：549 个扫描、183 个 visit、10,980 帧、19,147 个有效逐帧 GT，覆盖全部 17 类；3,166 帧没有有效 GT，仍保留用于误检评测。采样前无文件/轨迹配对问题，无排除扫描。目录约 156 MB，原始 ZIP 保持不变。详细计数见该目录的 `preparation_summary.json`。正式清单与本地图片已就绪，可以直接执行第 4 步中的 `infer` 与 `evaluate`，无需重跑 `prepare`。

35 项测试已通过，覆盖完整旋转 CPU IoU 与现有 Vis4D 算子的随机框对照、本地缓存/ZIP 输入一致性、四种旋转、源数据不可访问时的推理与可视化、缓存损坏和缺失检测。真实正式缓存的全部 10,980 帧已逐帧校验原图摘要、转正结果和内参，检查期间禁止 ZIP 访问；15 帧几何清单的缓存重建命令与 3 帧可视化也已通过。3D-MOOD 本地模型/权重加载和一个真实 ARKitScenes 帧的 CPU 推理此前已验证，输出 28 个符合统一接口的三维预测，记录在 `runs/geometry-smoke-v1/cpu_adapter_check.json`。这些验证不是完整模型评测。

本次全部 46 项测试通过，包括类别无关匹配、深度米单位与旋转、CuTR 原生角点与统一坐标的对照、逐帧压缩预测身份校验、分项目环境预检、续跑复用和报告完整性。3D-MOOD、CuTR RGB、CuTR RGB-D 已使用同一真实 ARKitScenes 帧在各自环境完成 CPU 推理、数值预测保存、两类 AP 和报告生成，验证目录为 `runs/benchmark-validation/real-cpu-smoke-v2/`；CuTR 采用正式配置的 1024 长边，分别输出 100 个前景候选。WildDet3D 本地完整模型构建通过，前向使用 CUDA Stream，无法在本工具进程进行 CPU 验证。这些单帧检查不代表完整模型效果评测。

工具进程无法访问 PPU，且可写范围为 `/mnt/workspace` 与 `/tmp`，无法直接向 `/root/report` 写正式报告；200 帧试评测、10,980 帧全量评测和目标路径报告生成需在 DSW 原生 Terminal 执行上面的 `benchmark` 命令。模块没有用模拟预测填写真实模型报告。
