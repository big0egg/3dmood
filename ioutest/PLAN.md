# ARKitScenes 多场景单帧三维检测评测方案

本文记录已经确认的评测协议及数据调查。实现统一放在 `/mnt/workspace/3dmood/ioutest`；可执行操作、接口与验证状态见 [模块使用说明](README.md)。设备端批量推理及真实模型 AP 是否完成，以使用说明和运行报告为准。

## 1. 目标与数据现状

首个适配器使用现有 Swin-B 权重，对 ARKitScenes 的 RGB 帧进行单目三维检测。数据准备与指标计算采用通用接口，分别用三维有向框 IoU 和三维中心欧氏距离匹配预测与真值，输出各类别 AP 和类别平均 mAP。

数据根目录：`/mnt/data/openBenchmark/ARKitScenes`。2026-10-10 的只读检查结果如下：

| 项目 | 已核对内容 |
| --- | --- |
| 数据格式 | 官方 3DOD 原始 ZIP，尚未整体解压 |
| 全部扫描 | Training 4,498，Validation 549，共 5,047 |
| 验证集范围 | 549 个扫描，183 个不同 `visit_id`；扫描数不能直接称为独立房间数 |
| 验证集 RGB | 下载清单记录共 475,833 帧；单扫描最少 113 帧、中位数 691 帧、最多 4,106 帧 |
| 验证集标注 | 下载清单记录共 5,962 个场景级框；逐帧有效框数需另行计算 |
| 验证集体积 | ZIP 总计约 51.25 GiB，完整解压约 61.20 GiB |
| 文件检查 | 549 个验证 ZIP 均存在；9 个扫描抽查包含标注和所需帧资源 |
| 原始图像 | 抽查 RGB 与深度均为 256×192；实际逐帧读取尺寸与内参，不将尺寸写死 |
| 图像方向 | 验证集 Up 189、Left 240、Down 120 个扫描 |
| 当前工作盘 | 检查时 `/mnt/workspace` 约余 14 GiB；不在该盘完整解压验证集 |

一个扫描压缩包包含：

```text
<video_id>/
├── <video_id>_3dod_annotation.json
├── <video_id>_3dod_mesh.ply
└── <video_id>_frames/
    ├── lowres_wide/                 # RGB 帧
    ├── lowres_depth/                # 深度帧
    ├── lowres_wide_intrinsics/      # 每帧 .pincam
    └── lowres_wide.traj             # 带时间戳的相机轨迹
```

原始数据有场景级三维真值，不直接提供可套用本项目评测器的逐帧 Omni3D 标注。需要将场景真值转换到相机坐标，再确定当前帧可见的目标。[Apple 3DOD 数据说明](https://github.com/apple-aiml-research/ARKitScenes/blob/main/threedod/README.md)

## 2. 固定评测协议

### 场景范围与切帧

建议正式评测覆盖全部 549 个 Validation 扫描，每个扫描最多选 20 帧，理论上限 10,980 帧。实际数量以完整性检查和采样清单为准；若用户选择先评测 100 个扫描，按 `visit_id` 和图像方向分组抽样，固定随机种子 20261010，并记录完整 ID 列表。

切帧直接使用 ZIP 中已有 PNG，无需先转视频再截图：

1. 按数字时间戳排序 RGB 帧，检查 RGB、深度、内参和轨迹可配对。
2. 保留时间间隔至少 0.5 秒的候选帧，再在整个扫描时间范围内等间隔分层选取最多 20 帧，帧少则全取。
3. 使用同时间戳内参；允许与官方加载器一致的 ±0.001 秒内参回退。轨迹采用最近时间戳，误差必须小于 0.005 秒。
4. 采样依据时间与文件完整性，不依据模型预测或目标类别。没有有效目标的采样帧也保留，用于统计误检。
5. 保存 `video_id`、`visit_id`、时间戳、ZIP 内文件名、方向和配对误差。一次冻结清单，后续模型使用同一批帧。

同一物体在不同帧出现时，属于不同的单帧检测样本；本方案不执行跨帧融合或跟踪。每个扫描设相同帧数上限，避免长视频主导结果。

### 真值类别与文本提示词

采用官方 17 类作为目标类别，固定类别 ID 和提示词映射，不根据每帧真值改变模型提示词。官方类别参考 [Apple taxonomy](https://github.com/apple-aiml-research/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/taxonomy.py)。

| GT 类别 | 模型提示词 |
| --- | --- |
| cabinet | cabinet |
| refrigerator | refrigerator |
| shelf | shelf |
| stove | stove |
| bed | bed |
| sink | sink |
| washer | washing machine |
| toilet | toilet |
| bathtub | bathtub |
| oven | oven |
| dishwasher | dishwasher |
| fireplace | fireplace |
| stool | stool |
| chair | chair |
| table | table |
| tv_monitor | television |
| sofa | sofa |

当前 `opendet3d/data/datasets/omni3d/arkitscenes.py` 使用另一套经过整理的类别映射，不能直接替代原始 17 类。主结果不合并 `stool` 与 `chair`，也不合并洗衣机与洗碗机。先统计全部验证标注中的原始标签，处理官方加载器使用的分隔符归一化，并报告未知标签；未知标签不得悄悄映射。

这是固定类别提示词下的单帧检测评测。开放集泛化能力需要另外定义已见/未见类别和独立测试范围。

### 真值坐标与可见性

读取 `segments.obbAligned` 的 `centroid`、`axesLengths`、`normalizedAxes`，使用米单位的有向框，保留完整三维旋转。抽查数据的 `obb` 和 `obbAligned` 数值尺度与坐标不同，不能混用。[官方真值与轨迹加载代码](https://github.com/apple-aiml-research/ARKitScenes/blob/main/threedod/benchmark_scripts/utils/tenFpsDataLoader.py)

转换与验证顺序：

1. 按官方约定解析轴角轨迹，明确 `T_world_to_camera` 与其逆矩阵；将场景框中心和 8 个角点转换到 RGB 相机坐标。
2. 根据 `sky_direction` 将图片转正，同时旋转深度、更新内参、变换框和相机坐标；不能只旋转图片。统一为模型使用的 OpenCV 坐标：x 向右、y 向下、z 向前。
3. 模型预处理继续使用项目现有 resize/normalize/pad 逻辑，并同步更新内参。保存原始与转正后的图像尺寸、内参和变换矩阵。
4. 将深度从毫米转为米，剔除零值；建议按 2 像素步长采样，在完整有向框内统计观测点。
5. 初始可见性协议使用 **框内观测点数 >20**，同时检查投影与图像相交、存在正深度部分。对穿越相机近平面的框按近平面裁剪边后投影，避免负深度角点导致错误投影。
6. 保存每个框的点数和有效/忽略原因。几何有效但因可见性被忽略的框保留为 ignore：预测优先匹配有效 GT，之后匹配到同类 ignore 框的预测不计 FP。处理规则对两类指标分别按对应匹配阈值执行。

点数门槛参考官方单帧准备脚本实际使用的 `pts_cnt > 20`；该脚本使用默认 2 像素步长生成深度点。深度在本方案中用于建立逐帧 GT 可见性，**不作为模型输入**。[官方单帧准备脚本](https://github.com/apple-aiml-research/ARKitScenes/blob/main/threedod/benchmark_scripts/data_prepare_online.py)

已用三个方向各一个扫描的中间帧进行只读几何检查：轨迹配对误差均小于 0.0004 秒，内参时间戳匹配；场景框数/深度支持的可见框数分别为 18/4、6/2、5/1。还需通过 GT 投影图核对转正后的图片与框，再进入批量推理。

### 两类 AP 的定义

以下为建议初版，实施前随方案确认后冻结。

| 指标族 | 匹配条件 | 必报阈值 | 汇总值 |
| --- | --- | --- | --- |
| 三维 IoU AP | 同类预测与 GT 的完整有向框体积 IoU ≥ 阈值 | AP3D@0.25、AP3D@0.50 | mAP_IoU：对有 GT 的类别及两个阈值取平均 |
| 三维中心距离 AP | 同类预测与 GT 的三维中心欧氏距离 ≤ 阈值，单位米 | AP_center@0.25m、@0.50m、@1.00m | mAP_center：对有 GT 的类别及三个阈值取平均 |

三维 IoU 使用完整旋转框，不替换为二维、BEV 或轴对齐框 IoU。模块使用独立 CPU 凸多面体求交实现，并与解析值及已有 Vis4D 算子对照验证；离线指标计算不依赖模型、torch 或 PPU。

AP 统一规则：同类别、同帧一对一匹配；预测按置信度降序处理，IoU 选满足阈值的最大 IoU 未匹配 GT，距离选满足阈值的最近未匹配 GT；重复预测计 FP。各类别汇总全部采样帧的预测，计算精度—召回曲线，用 101 个召回位置的插值精度平均得到 AP。同分预测使用固定 frame/prediction ID 排序，保证可复现。

报告含每类 AP、GT 数量和预测数量；无 GT 类别显示 N/A 并列明，不作为 0 纳入 mAP。空预测且有 GT 的类别 AP 为 0；空 GT 帧上的未忽略预测计 FP。两类指标使用同一批预测、同一套 GT 和类别映射。

现有 `Detect3DEvaluator(iou_type="dist")` 使用 **GT 半对角线长度 × 0.50～1.00** 的尺寸归一化距离门槛，不是固定米制距离。因此本方案的米制 AP 需要新增明确的匹配实现；若同时报告项目原有归一化 AP，使用独立名称，不能混为同一指标。现有 IoU AP 默认平均 0.05～0.50 的阈值，与本方案的两个阈值汇总也不同。

本方案使用原始低分辨率帧、自定义切帧和可见性协议；结果应标为 `ARKitScenes Validation sampled-frame`，不能直接对照 README 中的 Omni3D AP 或官方整扫描点云检测结果。

### 训练重叠审计

当前权重对应的训练配置包含 Omni3D `ARKitScenes_train` 与 `ARKitScenes_val`。这些名称不能直接等同于原始 3DOD 的 Training/Validation，正式解释泛化结果前要获取 Omni3D 标注，按 `video_id`、时间戳和 `visit_id` 审计训练重叠。

本地 metadata 还显示一个 `visit_id=381879` 同时出现在官方 Training 与 Validation：验证扫描为 `41159519`、`41159525`，训练扫描为 `41159521`、`41159526`。主表可保留官方完整验证范围，同时输出排除这两个验证扫描及其他审计发现重叠扫描的严格子集结果，明确两个范围各自的扫描数、visit 数和帧数。尚未完成 Omni3D 训练重叠审计时，不将结果描述为完全未见场景测试。

## 3. 推理与数据存储

复用现有 Swin-B 模型与已下载权重：

```text
/mnt/workspace/.cache/3d-mood/torch/hub/checkpoints/
gdino3d_swin-b_120e_omni3d_834c97.pt
```

新入口一次加载模型，逐帧推理，初始 batch size=1。初版冻结为：与 Demo 相同的预处理和 NMS 策略，最多保留 300 个候选，置信度过滤为 0.0，避免 Demo 的 0.1 门槛提前截断 AP 召回。上述设置记录在运行配置，不通过改动模型架构实现。

原始 `/mnt/data` 数据只读。准备 GT 和首次提取所选图片时读取 ZIP；不读取评测不需要的场景 mesh，不整体解压 61 GiB 验证集。选定帧缓存到 `ioutest/cache/<run_id>/`，保留原图和转正图片。推理与可视化默认优先读取本地原图，逐帧校验冻结的像素摘要并只转正一次；无缓存时可读取 ZIP。缓存的 benchmark ID、帧清单和完整性必须匹配当前运行目录。图片缓存与读取来源不改变冻结清单、GT 或指标定义。整个 `cache/` 与 `runs/` 已设置 Git 忽略，不提交数据图片到 GitHub。

采样清单、预测和报告持久保存到模块下独立运行目录；数据/GT 由各模型共享，模型预测与报告分开：

```text
ioutest/runs/<run_id>/
├── protocol.json                 # 阈值、提示词、预处理、可见性、版本与权重摘要
├── dataset_audit.json            # 类别、skipped、完整性、训练重叠与排除统计
├── frames.jsonl                  # 冻结的逐帧清单
├── groundtruth/                  # 各扫描逐帧有效/ignore GT
├── models/<model_id>/
│   ├── model.json                # 模型、权重、适配器和训练清单摘要
│   ├── predictions/              # 按扫描分片，含空预测帧与完成状态
│   ├── reports/                  # 两类 AP、每类/每扫描 CSV
│   └── timing.json
└── visualizations/               # 固定抽查帧的 GT/预测投影
```

预测缓存保存类别、分数、相机中心、完整尺寸/旋转、8 个角点、二维框和帧 ID。计算指标时直接读取缓存，修改指标阈值不必重新推理。断点续跑校验协议、模型和采样清单摘要；配置变化不能混入旧运行。每帧均须有完成状态，不能把运行失败的帧当作空预测或悄悄跳过。

## 4. 分阶段实施与验收

| 阶段 | 规模 | 验收结果 |
| --- | --- | --- |
| 数据审计 | 全部 549 个验证 ZIP 的目录、标注与元信息 | 17 类映射、skipped/异常扫描、可配对帧、训练重叠、采样清单和空间预算 |
| 几何与指标验证 | 覆盖 Up/Left/Down 的至少 3 个扫描，每个 3～5 帧 | GT 投影对齐；恒等框 IoU=1、分离框=0；已知平移对应正确距离；空预测、重复预测、ignore 和同分排序正确 |
| 小规模模型验证 | 10 个扫描，每个最多 10 帧 | 最多 100 帧完成；生成两类 AP 和类别表；检查点位、尺寸及方向；测量模型预热后的帧耗时与峰值内存 |
| 正式评测 | 549 个扫描，每个最多 20 帧，或用户确认的子集 | 全部计划帧有状态；总表、类别表、严格子集表、样例图和协议摘要齐全 |

几何单元检查还包括相机/世界坐标往返一致、转正前后投影一致、内参随缩放更新、角点拓扑一致。至少构造一个倾斜三维框验证完整旋转 IoU，避免把只支持水平框的实现误用于相机坐标。

运行时间不预先猜测。100 帧阶段分别测量数据读取、预处理、设备推理、后处理和指标计算；用实测吞吐估算正式评测。大批量推理在可访问 PPU 的 DSW 原生终端运行。

## 5. 模块实现范围

已按用户确认将所有新增评测代码放在 `ioutest` 内：

- `dataset.py`：只读 ZIP 适配、时间戳配对、图像转正、坐标转换与逐帧 GT。
- `image_cache.py`：选定帧缓存、本地读取、原图像素和缓存完整性校验。
- `geometry.py` / `metrics.py`：完整旋转框 IoU、米制中心距离 AP、ignore 和一对一匹配。
- `schema.py`：不含真值的模型输入与统一预测接口。
- `adapters/mood3d.py`：唯一的内置模型适配器，复用现有模型与权重。
- `pipeline.py` / `__main__.py`：命令入口、批量推理、断点续跑、通用预测导入和离线报告。
- `visualize.py` / `tests/`：投影检查及几何、指标、数据完整性验证。

默认复用现有模型和权重，不修改模型结构与已有数据集评测逻辑，也不需要重新训练。模型只接收 RGB、相机内参和固定类别文本；原始深度、轨迹、三维框用于评测数据适配。
