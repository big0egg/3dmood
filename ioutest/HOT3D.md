# HOT3D benchmarkv2 使用说明

本流程用 HOT3D-Clips `train_aria` 的 RGB，对比以下五组配置。数据、预测和报告默认位于 `/root/benchmarkv2`，独立于已有 ARKitScenes 结果。

| 任务 | 配置 | 流程 | 主要指标 |
|---|---|---|---|
| RGB＋GT二维框 | `wilddet3d-box` | 原生 geometric 提示逐目标估计，不输入类别 | 成功率、失败率，补充AP |
| RGB＋GT二维框 | `unipose9d-box-rgb` | 相同框→SAM2预测mask；RGB→MoGe→UniPose9D | 同上 |
| RGB＋类别文本 | `3dmood` | RGB＋相机内参＋固定全词表→原生三维检测 | 几何AP、语义AP、0.7工作点 |
| RGB＋类别文本 | `wilddet3d` | 相同RGB、内参、词表→原生三维检测 | 同上 |
| RGB＋类别文本 | `unipose9d-text-rgb` | GroundingDINO→SAM2→MoGe→UniPose9D，保留所有达标实例 | 同上 |

没有传感器深度，所以不运行RGB-D组。MoGe深度属于RGB预测流程，GT渲染深度不作为模型输入。

## 1. 直接启动full

在 **DSW原生Terminal** 执行，使用现成PPU环境，无需切换镜像。当前配置直接复用`/root/model-assets`的权重，不复制或重新下载。只有缺失权重才恢复到`<root>/assets`：WildDet3D优先从`/mnt/lvyifan`复制，UniPose9D和辅助权重使用官方来源。恢复步骤校验已知SHA256，不安装或升级模型环境；缺失时需要联网。

当前实际目录有一层同名子目录，权重完整路径为`/root/model-assets/model-assets/{unipose9d,wilddet3d}/...`，配置已按实际位置填写。

```bash
mkdir -p /root/benchmarkv2
nohup bash /mnt/workspace/3dmood/ioutest/run_hot3d_benchmark.sh \
  --root /root/benchmarkv2 \
  --phase full \
  > /root/benchmarkv2/launch.log 2>&1 &
```

顺序：恢复缺失权重→环境/权重/PPU检查→抽帧和几何检查→第一类两组推理及评分→第二类三组推理及评分→报告。不执行smoke或pilot。抽帧和几何检查是数据准备，不是模型小批量试评测。所有HOT3D预测重新生成，不复用ARKitScenes预测。

本会话工具对`/root`只读、不可见PPU时，只能在可写目录验证实现；实际向`/root/benchmarkv2`落盘和PPU前向由以上原生终端命令执行。

查看进度：

```bash
cat /root/benchmarkv2/status.json
cat /root/benchmarkv2/prepare_status.json
cat /root/benchmarkv2/runs/hot3d-aria-v1/worker_status.json
tail -n 30 /root/benchmarkv2/launch.log
```

准备尚未开始时没有`prepare_status.json`；模型进程尚未启动时没有`worker_status.json`。子进程详细日志在`runs/hot3d-aria-v1/<gt-box或text>/full/logs/`。

**中断后重新执行原命令即可续跑**，日志重定向可改为`>>`追加。数据按完整片段提交；推理按完整帧提交，中断片段/帧重新处理。已经记录的目标估计失败保留，不自动反复尝试到成功。源码、输入、权重、环境或参数不一致会拒绝混用缓存；改变实验设置使用新`--root`。

## 2. 可填写的命令参数

脚本和`python -B -m ioutest hot3d-benchmark`接受相同参数。带值的参数写`--参数 值`，开关只写名字。

| 参数 | 作用 | 默认值 | 可填内容 |
|---|---|---|---|
| `--root` | 数据、缓存、报告根目录 | `/root/benchmarkv2` | 可写的本地目录 |
| `--config` | 数据/模型/指标配置 | [hot3d_config.json](hot3d_config.json) | 已存在的JSON文件 |
| `--phase` | 正式执行阶段 | `full` | 目前仅`full` |
| `--tracks` | 执行哪类任务 | `gt-box text` | `gt-box`、`text`，或两者 |
| `--prepare-only` | 只生成本地数据、GT和几何检查图 | 不启用 | 开关，不运行模型、不需要PPU |
| `--score-only` | 核验已有预测、重算需要更新的指标及报告 | 不启用 | 开关，不运行模型、不需要PPU |
| `--check` | 只读检查模型环境、权重和PPU | 不启用 | 开关，不写入数据 |

最后三个开关互斥。例如只续跑文本组：

```bash
bash /mnt/workspace/3dmood/ioutest/run_hot3d_benchmark.sh \
  --root /root/benchmarkv2 --phase full --tracks text
```

## 3. 配置字段

| 字段 | 意义 | 当前值 / 可填内容 |
|---|---|---|
| `dataset_root` | 只读原始HOT3D位置 | `/mnt/data/opendatasets/HOT3D` |
| `data.name` | 本次数据版本名，也是数据和run子目录名 | `hot3d-aria-v1` |
| `data.split` / `stream_id` | 划分与彩色相机流 | 本协议只允许`train_aria` / `214-1` |
| `data.clip_ids` | 选择的片段整数编号 | `null`表示官方train Aria全部1516个；也可填官方划分中的整数列表 |
| `data.frame_indices` | 每片段内选帧位置 | `[7,22,37,52,67,82,97,112,127,142]`；不重复整数，范围0～149 |
| `data.camera.width` / `height` | 模型输入针孔图尺寸 | 768×768，整数≥16 |
| `data.camera.horizontal_fov_degrees` | 虚拟针孔水平视场角 | 90度；需在10～150度内且源相机有效像素占比≥95% |
| `data.camera.clockwise_quarter_turns` | 同时旋转图像和相机坐标 | `1`即顺时针90度；可填0、1、2、3 |
| `data.minimum_amodal_pixels` | 虚拟图内GT完整投影轮廓最小面积 | 64像素；不得根据模型结果调整 |
| `data.minimum_native_modeled_visibility` | 原鱼眼图中几何可见性下限 | 0.1；来自官方几何可见性，不是SAM2估计 |
| `object_semantics` | 33个资产ID映射到通用语义提示词 | 当前合并成25类；全局词表每帧相同 |
| `metrics.iou_thresholds` | 三维框IoU匹配阈值 | `[0.25,0.5]`，0～1之间的正数，不重复 |
| `metrics.center_thresholds` | 米制中心距离阈值 | `[0.02,0.05,0.1,0.25,0.5,1.0]`，正数、不重复 |
| `display_score` | 文本可视化及附加P/R/F1工作点 | 0.7，不过滤主AP |
| `seed` | 帧/目标派生随机种子的基础值 | 0 |
| `device` | 模型执行设备 | 正式配置为`cuda`；PPU不可见时停止，不自动回退CPU |
| `frontend` | SAM2、GroundingDINO、MoGe解释器和本地权重 | 沿用现有隔离前处理环境；GroundingDINO框0.4/文本0.3、最多300候选 |
| `models` | 五组模型各自解释器、适配器和权重 | 沿用当前已经适配的PPU环境，修改后使用新root |

配置中显式填写且确实存在的权重路径优先使用；缺失路径自动转到`<root>/assets`。实际生效路径和全部权重SHA记录到`<root>/config/hot3d_aria.json`及`<root>/assets/manifest.json`。GroundingDINO只下载safetensors版本，避免再存一份相同模型的bin格式。`--prepare-only`不恢复模型权重，只准备数据。

UniPose9D使用完整预算：1024点、1000 tuples、32步、10 repeats、2个RANSAC hops、1000次RANSAC、scale clustering启用；不为加速降低正式采样预算。

原始片段包括150帧，完整官方划分按固定10帧抽样共15,160帧。准备时按原始录制＋时间戳检查重复，核验150张RGB/物体标注、标定、源图尺寸、UID映射和选中内容摘要；不宣称完成了所有tar完整文件的SHA校验。源目录的`.part`不是正式tar，不参与抽帧。

## 4. 鱼眼与GT规则

相机使用FISHEYE624完整15参数：单焦距、主点、6项径向、2项切向、4项薄棱镜。按Meta官方相机公式直接将虚拟针孔射线投影到源鱼眼图，用OpenCV做图像重采样；不把它当作OpenCV四参数鱼眼模型。所用NumPy/SciPy/OpenCV/Pillow均已在现有驱动环境中，无需安装新的相机SDK或改动模型框架。

GLB读取应用所有场景节点的矩阵/旋转/缩放，包含0.001单位缩放。在物体局部坐标计算固定包围盒，使用逐帧物体位姿和相机位姿转到相机，再按图像转正应用同一旋转；单位米，x右y下z前，尺寸是局部轴的完整边长。模型原点不等于包围盒中心。

GT二维提示框由最终三维框重新投影。准备时将官方amodal轮廓重投影到同一针孔图，检查有足够面积的轮廓至少95%落在三维框投影内（容差3像素），以发现坐标、旋转和单位错误；失败即停止。鱼眼转针孔会损失周边视野，这些排除量在审计和报告中展示。

有效GT采用**虚拟视野投影面积＋原鱼眼几何可见性**，没有重新计算针孔下完整场景的遮挡比例，不能将二者混称。官方SAM2生成的modal mask/可见性不用于主样本筛选和模型输入。数据集只标注建模物体；未标注背景同类物体可能产生FP，报告披露这一限制。

源格式及几何依据：[HOT3D-Clips](https://github.com/facebookresearch/hot3d/tree/main/hot3d/clips)、[官方相机实现](https://github.com/facebookresearch/hand_tracking_toolkit/blob/main/hand_tracking_toolkit/camera.py)、[FISHEYE624方程](https://github.com/facebookresearch/hand_tracking_toolkit/blob/main/hand_tracking_toolkit/camera_distortion.py)。

## 5. 输入与评分

框提示组不给GT三维几何、类别、掩码，只输入统一RGB/K及原目标二维框。模型预测固定对应原目标，失败留在成功率分母。补充AP按帧匹配，WildDet3D用原生分数，UniPose9D用inlier_ratio代理分数。

文本组不输入GT位置、该帧类别、物体数量、相机世界位姿或实测深度。全词表→各模型原生流程。几何AP忽略类别；语义AP要求语义一致，对有GT类别等权平均。以完整旋转框计算IoU，中心距离不评价尺寸/方向。按置信度降序、同帧一对一匹配，优先有效GT，再匹配未占用ignore GT；其他预测计FP。AP使用101召回点插值。

主表是IoU 0.25/0.50和中心距离2/5/10厘米；25厘米/50厘米/1米放补充表。GT来自官方训练划分，属于自定义外部评测，不是官方隐藏测试/BOP榜单；没有证明所有权重无训练重叠，不标为严格零样本。

## 6. 输出位置

```text
/root/benchmarkv2/
├── assets/                         # 缺失权重的恢复位置及SHA清单
├── config/{hot3d_aria,categories}.json
├── data/hot3d-aria-v1/
│   ├── protocol.json / frames.jsonl / dataset_audit.json
│   ├── rgb_fisheye/ / rgb_pinhole/ / calibration/
│   ├── groundtruth/ / object_models/
│   ├── geometry_checks/             # 每片段首个抽样帧的GT投影图
│   └── completed_clips/             # 数据准备断点
├── runs/hot3d-aria-v1/
│   ├── worker_status.json
│   ├── aux/                        # 预测mask、二维候选、共享MoGe深度
│   ├── gt-box/{models,full}/
│   └── text/{models,full}/
└── report.md                       # 已完成任务的报告索引
```

每个`full/`生成`report.md`、`report_context.json`、`summary.json`、对比CSV、模型指标JSON、`runtime.csv`、`speed.json`和带框图。框提示组有逐目标`*_targets.csv`；文本组有逐类`*_classes.csv`和逐物体几何召回`*_object_recall_at_0.7.csv`，不把资产ID当作语义类别。

速度：完整流程FPS=全部帧数÷（模型＋定位/分割＋预测深度秒数）；平均秒/帧为倒数。框提示另算全部输入目标的平均秒/目标。包含预处理、首次推理、无目标帧，排除初始化、读写和CPU评分。共享缓存复用也保留生成时耗时，不拿读缓存速度冒充推理速度。

建议目标盘预留80GB左右。仅15,160张768×768 float32预测深度约33.3GiB，另需图片、掩码和预测空间。本地大文件在`/root/benchmarkv2`，不进入项目Git仓库。
