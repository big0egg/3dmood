# ioutest 使用说明

HOT3D 的五组 full 评测使用独立入口，数据和结果默认写到 `/root/benchmarkv2`。操作命令、全部参数、鱼眼/GT规则及文件说明见 [HOT3D 使用说明](HOT3D.md)。原 ARKitScenes 入口继续使用下面的命令。

当前推荐使用 `unipose-benchmark` 评测 3D-MOOD、WildDet3D 和 UniPose9D 的两类任务。旧的 `benchmark` 命令保留上一轮 CuTR 实验的复现入口。

## 当前任务：UniPose9D 两类 benchmark

沿用冻结的 549 个扫描、183 个 visit、10,980 帧，不重新准备数据。第一类共有 19,147 个有效帧—物体目标；同一个物体跨帧会重复计数。

| 任务 | 配置 ID | 输入与流程 | 主要指标 |
|---|---|---|---|
| RGB＋GT 二维框 | `wilddet3d-box` | RGB、内参、GT 投影框；原生 geometric 提示，无目标类别 | 五项成功率、失败率；补充 AP |
| RGB＋GT 二维框 | `unipose9d-box-rgb` | 同一框→SAM2 掩码；MoGe 预测深度→UniPose9D | 同上 |
| RGB-D＋GT 二维框 | `unipose9d-box-rgbd` | 同一框、共享 SAM2 掩码、同一姿态权重；ARKit 实测深度 | 同上，单列额外深度输入 |
| RGB＋类别文本 | `3dmood` | 模型项目自行根据固定 17 类文本定位 | 五项 AP、17 类语义 AP、≥0.7 工作点 |
| RGB＋类别文本 | `wilddet3d` | 模型项目自行根据固定 17 类文本定位 | 同上 |
| RGB＋类别文本 | `unipose9d-text-rgb` | 官方 GroundingDINO＋SAM2＋MoGe＋UniPose9D 组件，保留全部达标实例 | 同上；AP 分数继承前置检测器 |

框提示组允许提供真实二维位置，但不提供 GT 三维框或类别；文本组不提供 GT 位置、物体数量或实测深度。GT 投影框不等于实例掩码，掩码由 SAM2 根据图片预测。两个任务的分数分别报告，不能混合排名。

成功率以全部输入目标为分母，失败记为不达标，并固定关联到原输入目标。补充 AP 按帧一对一匹配；UniPose9D 框提示组使用 `inlier_ratio` 质量代理排序，不能当作检测概率。文本组主 AP 使用全部导出候选，另算分数≥0.7 的精确率、召回率和 F1。

### 在 DSW 原生 Terminal 启动

```bash
bash /mnt/workspace/3dmood/ioutest/run_unipose_benchmark.sh \
  --phase full \
  --report-root /root/report/arkitscenes-unipose9d-v2
```

命令先准备独立的前处理环境与 SAM2/检测器权重，然后直接执行 `full`（10,980帧），两类任务共六组配置，跳过 smoke 和 pilot。UniPose9D 使用你已验证的 clean 环境和真实权重；前处理环境复用框架，不升级共享 PyTorch。正式推理需要原生终端能访问 PPU，运行时不自动改用 CPU。

PPU 前处理关闭 SAM2 的可选 NVIDIA 扩展构建，GroundingDINO 使用 `disable_custom_kernels=true` 的 PyTorch 参考算子；这些兼容设置写入模型来源记录。

需要关闭终端后继续运行时：

```bash
nohup bash /mnt/workspace/3dmood/ioutest/run_unipose_benchmark.sh \
  --phase full \
  --report-root /root/report/arkitscenes-unipose9d-v2 \
  > /mnt/workspace/3dmood/ioutest/runs/unipose-v2-launch.log 2>&1 &
```

查看总进度、当前子进程进度和启动日志：

```bash
cat /root/report/arkitscenes-unipose9d-v2/status.json
cat /root/report/arkitscenes-unipose9d-v2/worker_status.json
tail -n 30 /mnt/workspace/3dmood/ioutest/runs/unipose-v2-launch.log
```

如在下载/环境准备阶段，`worker_status.json` 尚不存在属于正常情况。子进程详细日志位于各任务阶段的 `logs/`。

### 参数和配置

这些参数用于 `python -B -m ioutest unipose-benchmark`；上面的脚本已经加上 `--prepare`。

| 参数 | 作用 | 默认值 | 可填写内容 |
|---|---|---|---|
| `--config` | 数据、环境、权重、采样与阈值配置 | [unipose_suite_config.json](unipose_suite_config.json) | 已存在的 JSON 文件 |
| `--report-root` | 新实验根目录 | `~/report/arkitscenes-unipose9d-v2` | 可写目录，不是文件名；配置变化使用新目录 |
| `--phase` | 执行阶段 | `all` | `smoke`、`pilot`、`full`、`all`；all 按三阶段顺序执行 |
| `--tracks` | 执行任务 | `gt-box text` | `gt-box`、`text`，或两个值；每个任务固定三组 |
| `--prepare` | 准备前处理依赖和辅助权重 | 不启用 | 只写名字；首次使用需联网，后续复用本地文件 |
| `--check` | 只检查环境、权重和设备 | 不启用 | 只写名字；不运行模型 |
| `--reuse-baselines-only` | 核验并评分已有文本组历史预测 | 不启用 | 只写名字；此模式不完成六组评测，不运行 PPU |

配置中的 `models.<ID>.python` 是虚拟环境 Python 路径；`options.checkpoint` 是真实权重文件；`frontend` 指定 SAM2、GroundingDINO、MoGe 和辅助解释器。当前 UniPose9D 核心权重为 `/mnt/workspace/model-assets/unipose9d/last.ckpt`，仓库内同名文件只是 LFS 指针。

UniPose9D 正式采样预算为 `num_points=1024`、`num_tuples=1000`、`num_steps=32`、`num_repeats=10`、`ransac_hops=2`、`ransac_iterations=1000`、`scale_clustering=true`。固定种子按帧/目标派生，断点续跑不因执行顺序改变采样；RGB/RGB-D 对应目标共享种子。修改这些值必须使用新报告目录。

`frontend.box_threshold=0.4`、`text_threshold=0.3` 沿用官方前置检测默认值，`max_detections=300` 为候选上限；文本类别通过固定提示 token 范围确定。`display_score=0.7` 用于附加工作点和文本图片展示，不用于过滤主 AP。各模型分数尺度没有统一校准。

### 输出与续跑

```text
<report-root>/
├── gt-box/{smoke,pilot,full}/   # 条件成功率、补充 AP、逐目标明细和带框图
├── text/{smoke,pilot,full}/     # 自主检测 AP、语义 AP、0.7 工作点和带框图
├── gt-box/models/              # 新逐帧/逐目标框提示预测
├── text/models/                # 文本预测，或明确记录的历史预测引用
└── aux/                        # 预测掩码、二维候选和共享预测深度
```

每个完整阶段自动生成详细 `report.md`、`report_context.json`、对比 CSV、模型指标 JSON、逐类 AP、性能 CSV、日志和检查图。框提示组还保存 `*_targets.csv`，逐行说明原目标的 IoU、中心距离、失败原因和五项达标情况。全部配置完成后才生成最终对比，不将缺失预测算成“无检出”。

推理速度写入各任务阶段的 `report.md`、`runtime.csv` 和 `speed.json`：

| 速度指标 | 意义与计算 | 如何看 |
|---|---|---|
| 完整流程 FPS（`fps_pipeline`） | 全部帧数÷（模型预测＋定位/掩码＋预测深度总秒数） | 主要速度指标，越高越快 |
| 完整流程平均秒/帧（`mean_pipeline_seconds_per_frame`） | 完整流程总秒数÷全部帧数 | 越低越快，与 FPS 互为倒数 |
| 模型 FPS / 平均秒/帧（`fps_predict` / `mean_predict_seconds_per_frame`） | 只使用模型预测计时 | UniPose9D 此列不含外置检测、分割和深度预测；跨模型优先比较完整流程 |
| 模型 / 完整流程平均秒/目标（`mean_predict_seconds_per_target` / `mean_pipeline_seconds_per_target`） | 对应总时间÷全部输入目标数，只用于框提示组 | 含失败目标，共享掩码/深度成本按目标分摊，需结合成功率判断 |
| 历史测量标志（`historical_reuse`） | `true` 表示复用之前预测时记录的时间 | 不能作为本次硬件重新测速的结果 |

这些速度包含首次推理和无目标帧，排除模型加载、数据读写和 AP 评分；缓存复用仍计入生成缓存的原始耗时。RGB-D 组不计 MoGe 预测深度时间。例：100 帧的完整流程累计耗时 200 秒，平均 2 秒/帧、0.5 FPS；这不等于整个 benchmark 的墙钟耗时。没有有效计时时留空，不显示无限 FPS。

中断后使用原命令续跑。模型、代码、输入或环境变化会阻止混用缓存。旧文本基线只有通过权重内容、源码、适配器、数据和环境版本核验才复用，报告明确标注历史预测和计时来源。该复用不适用于新的 WildDet3D 框提示实验。

图像与运行缓存已由 Git 忽略规则排除。指标 CSV/JSON 的成功率与 AP 使用 0～1，中文报告使用百分数。当前工具进程不可见 PPU、不能写 `/root/report` 时，可以完成代码、CPU验证和历史评分；正式模型运行需上述原生终端入口。

## 1. 先了解这几个词

| 名称 | 含义 |
|---|---|
| RGB / RGB-D | RGB 是彩色图片；RGB-D 还输入同一帧的深度图，深度表示距离 |
| GT / ground truth | 数据集提供的真实物体框，用来判断模型是否检测正确 |
| 扫描 / scan | 一段场景采集记录；一个扫描可以抽取多张图片 |
| visit | 一次场景访问的编号；同一个 visit 可以包含多个扫描 |
| 推理 / inference | 模型读取图片并输出预测框 |
| AP | 综合漏检、误检和预测分数的指标，越高越好；不是“检测正确率” |
| 三维 IoU | 预测框与真实框的交集体积除以并集体积，范围 0～1，越大越重合 |
| 三维中心距离 | 预测框中心与真实框中心之间的距离，单位米，越小越接近 |

## 2. 上一轮 CuTR 流程（历史复现）

以下为旧 `benchmark` 的四组流程，重新运行需要对应 CuTR 项目与权重。本次六组任务使用上面的 `unipose-benchmark` 入口。

当前已有 **549 个扫描、183 个 visit、10,980 帧**的图片、深度和真实标注，无需重新切帧或下载。

| 内容 | 当前路径 |
|---|---|
| 项目 | `/mnt/workspace/3dmood` |
| 原始数据 | `/mnt/data/openBenchmark/ARKitScenes` |
| 评测清单和 GT | `/mnt/workspace/3dmood/ioutest/runs/arkit-validation-v1` |
| 本地 RGB 和深度 | `/mnt/workspace/3dmood/ioutest/cache/arkit-validation-v1` |
| 模型配置 | [benchmark_config.json](benchmark_config.json) |
| 默认报告目录 | `~/report/arkitscenes-three-models-v1`；当前 root 用户对应 `/root/report/arkitscenes-three-models-v1` |

在阿里云 DSW 的 **Terminal（终端）**中执行。先进入项目并激活已有环境：

```bash
cd /mnt/workspace/3dmood
source /mnt/workspace/venvs/3d-mood-ppu/bin/activate
```

下面所有命令都在这个终端、这个目录运行。启动程序会自动调用其他模型各自的环境，不需要手动切换镜像或环境。

### 第一步：试评测 200 帧

```bash
python -B -m ioutest benchmark \
  --phase pilot \
  --report-root "$HOME/report/arkitscenes-three-models-v1"
```

从 10 个不同 visit 各选一个扫描，使用每个扫描已有的 20 帧。四组模型依次运行，完成后查看 `pilot/report.md`。

### 第二步：全量评测 10,980 帧

```bash
python -B -m ioutest benchmark \
  --phase full \
  --report-root "$HOME/report/arkitscenes-three-models-v1"
```

会复用试评测已完成的预测，完成后查看 `full/report.md`。如果希望一次连续完成两步，将 `--phase` 填成 `all`；省略该参数也使用 `all`。

### 第三步：查看进度和结果

```bash
cat "$HOME/report/arkitscenes-three-models-v1/status.json"
tail -n 20 "$HOME/report/arkitscenes-three-models-v1/pilot/logs/3dmood.log"
```

`status` 为 `running` 表示运行中，`complete` 表示记录中的 `phase` 阶段完成，`failed` 表示该阶段未完成。查看全量阶段日志时，将路径中的 `pilot` 换成 `full`；其他模型的日志文件名见下表。

**中断续跑：重新执行原命令即可。** 保持数据目录、配置文件、设备和试评测扫描数一致；更换权重、配置或代码时使用新的报告目录。

命令写法：`--参数名 值`；多个值用空格分隔。`--score-only` 这类开关只写名字，后面不填 `true`。路径有空格时加引号。多行命令末尾的 `\` 表示下一行继续，后面不要加其他文字。`-B` 表示不生成 Python 字节码缓存，`-m ioutest` 表示启动本模块，都不会改变评测指标。

## 3. `benchmark` 全部参数

这些参数写在 `python -B -m ioutest benchmark` 后面。

| 参数 | 作用 | 默认值 | 可以填写的内容 / 示例 |
|---|---|---|---|
| `--phase` | 选择运行阶段 | `all` | `pilot`：只试评测；`full`：只全量；`all`：先试评测，再全量 |
| `--report-root` | 保存报告、日志和预测的目录，程序会创建目录 | `~/report/arkitscenes-three-models-v1` | 可写目录，例如 `"$HOME/report/test-v2"`；不是文件名 |
| `--models` | 选择模型及执行顺序 | `3dmood wilddet3d cutr-rgb cutr-rgbd` | 填下面模型表中的一个或多个 ID，用空格分隔，不重复 |
| `--pilot-scans` | 试评测使用多少个扫描，每个扫描来自不同 visit | `10` | 正整数；当前可填 `1`～`183`，例如 `5` 对应 100 帧。只改变试评测规模，不减少全量帧数 |
| `--run-dir` | 读取已准备好的帧清单和 GT | `/mnt/workspace/3dmood/ioutest/runs/arkit-validation-v1` | 已包含 `protocol.json`、`frames.jsonl`、`groundtruth/` 的目录；不能填原始数据根目录 |
| `--config` | 读取模型环境和权重配置 | 本模块的 `benchmark_config.json` | 已存在的 JSON 配置文件路径，例如 `/mnt/workspace/3dmood/ioutest/benchmark_config.json` |
| `--with-gravity` | 额外加入 CuTR RGB + 已知重力实验 | 不启用 | 只写 `--with-gravity`；没有后续值。增加的是 `cutr-rgb-gravity`，不会替换原有四组 |
| `--device` | 模型在哪个设备运行 | `cuda` | 当前单卡正式评测用 `cuda` 或 `cuda:0`；`cpu` 仅用于部分模型验证，不能运行包含 WildDet3D 的评测 |
| `--score-only` | 从已有完整预测生成或复用指标与报告，不运行模型 | 不启用 | 只写 `--score-only`；所选模型必须已有对应阶段全部帧的预测 |
| `-h` / `--help` | 显示命令帮助后退出 | 不启用 | 例如 `python -m ioutest benchmark --help` |

**续跑配置要一致：** 如果第一次使用 `--pilot-scans 5`，后续执行 `full` 或 `--score-only` 时也填写 `--pilot-scans 5`。`--run-dir`、`--config`、`--device` 同样保持首次设置。评测对比时保持相同模型列表；需要不同实验设置时换一个 `--report-root`。

### 模型 ID 怎么填

| `--models` 可填值 | 对应模型 | 输入 | 所用环境 / 日志名 |
|---|---|---|---|
| `3dmood` | 3D-MOOD | RGB、相机内参、固定 17 类提示词 | `3d-mood-ppu` / `3dmood.log` |
| `wilddet3d` | WildDet3D | RGB、相机内参、固定 17 类提示词 | `wilddet3d-ppu` / `wilddet3d.log` |
| `cutr-rgb` | CuTR RGB | RGB、相机内参 | `ca1m-ppu` / `cutr-rgb.log` |
| `cutr-rgbd` | CuTR RGB-D | RGB、相机内参、同帧 ARKit 深度 | `ca1m-ppu` / `cutr-rgbd.log` |
| `cutr-rgb-gravity` | CuTR RGB + 已知重力 | RGB、相机内参、从传感器姿态计算的重力方向 | `ca1m-ppu` / `cutr-rgb-gravity.log`；默认不运行 |

例如，只评测 CuTR 的两种输入，并保存在另一个报告目录：

```bash
python -B -m ioutest benchmark \
  --models cutr-rgb cutr-rgbd \
  --phase pilot \
  --report-root "$HOME/report/cutr-comparison-v1"
```

已有完整的四组全量预测时，只生成报告：

```bash
python -B -m ioutest benchmark \
  --phase full \
  --score-only \
  --report-root "$HOME/report/arkitscenes-three-models-v1"
```

## 4. 报告怎么看

报告目录下有 `pilot/`（试评测）和 `full/`（全量）。以下文件在阶段全部完成后生成：

| 文件或目录 | 用途 |
|---|---|
| `report.md` | 中文评测报告，优先查看这个文件 |
| `report_context.json` | 当前阶段的数据规模、类别分布、忽略原因、输入位置和带框图索引；与报告一起自动生成 |
| `comparison_rgb.csv` | 3D-MOOD、WildDet3D、CuTR RGB 的几何 AP；不要求预测类别相同 |
| `comparison_rgbd.csv` | CuTR RGB 和 RGB-D 的几何 AP，单独观察深度输入组的效果 |
| `comparison_17class.csv` | 3D-MOOD 和 WildDet3D 的 17 类语义 AP，要求物体类别匹配；CuTR 不提供这 17 类标签 |
| `comparison_visit_disjoint.csv` | 排除与官方训练集共享 visit 的扫描后，RGB 三组的几何 AP |
| `comparison_gravity.csv` | CuTR RGB 与 RGB + 已知重力对比，仅启用重力组时生成 |
| `runtime.csv` | 平均耗时、95% 分位耗时（95% 的计时帧不超过此值）、每秒处理帧数、峰值已分配显存；耗时单位秒，显存单位字节 |
| `metrics/` | 完整指标 JSON、逐类 AP 和各扫描的 GT / 预测数量 |
| `figures/` | AP 图表，PNG 和 SVG 格式 |
| `visualizations/` | 绿色为真实框，红色为预测框；只显示分数 ≥0.1 的预测，显示过滤不影响 AP |
| `logs/` | 各模型的运行日志 |

报告根目录中的 `models/` 保存逐帧数值预测，供续跑和评分使用；`experiment.json` 保存实验设置，`status.json` 保存最新阶段状态。只选择部分模型时，对比表只包含实际运行的对应组。

性能计时包括模型预处理、推理和预测导出，不包括图片读取、结果写入文件和模型加载；排除每次模型加载后的首帧。`fps` 越高越快，耗时越低越快。

| 指标列 | 判断一次匹配是否正确 | 怎样理解 |
|---|---|---|
| `AP3D@0.25` | 三维 IoU ≥0.25 | 框的位置、尺寸与方向都参与判断 |
| `AP3D@0.50` | 三维 IoU ≥0.50 | 比 0.25 更严格 |
| `APcenter@0.25m` | 框中心距离 ≤0.25 米 | 只判断中心位置，不判断尺寸和方向 |
| `APcenter@0.50m` | 框中心距离 ≤0.50 米 | 允许更大的位置误差 |
| `APcenter@1.00m` | 框中心距离 ≤1.00 米 | 允许的位置误差最大 |

AP 越高越好。**CSV 使用 0～1，中文报告使用百分数**，例如 CSV 的 `0.42` 对应报告的 `42.00`。没有有效 GT 的类别不参与类别平均值，在 JSON 中为 `null`，在 CSV 中为空。`mAP` 表示对参与评测的类别取平均；JSON 中 `map` 还对相应的多个阈值取平均。

各模型使用同一批帧。RGB-D 和已知重力组得到额外传感器信息，单独比较。当前原始图片为 256×192，结果按本模块协议计算，不能直接当作论文设置下的 AP。CuTR 检出的物体可能超出数据集的 17 类标注范围，未标注物体可能计为误检。官方训练 visit 排除子集为 10,940 帧，但不代表已经查清每个模型全部训练数据。详细规则见 [PLAN.md](PLAN.md)。

`benchmark` 的五个指标阈值固定为上表数值，目前没有修改它们的命令参数。下面的 `--iou-thresholds`、`--distance-thresholds` 仅适用于单模型 `evaluate` / `run`。

### 已完成评测，只补全报告说明

后续 benchmark 会自动生成包含数据与文件用途、评测流程、指标原理、模型输入差异、公平性限制、逐类成绩、性能和图片索引的报告。已有完整指标时，使用下面的命令更新报告，不加载模型，不重新计算 AP，不需要 PPU：

```bash
python -B -m ioutest.benchmark_report \
  --report-root "$HOME/report/arkitscenes-three-models-v1" \
  --phase full
```

| 参数 | 作用 | 默认值 | 可以填写的内容 |
|---|---|---|---|
| `--report-root` | 已完成 benchmark 的报告根目录 | `~/report/arkitscenes-three-models-v1` | 包含 `experiment.json`、阶段目录和 `models/` 的已有目录 |
| `--phase` | 更新哪个阶段的报告 | `full` | `pilot`、`full`、`cpu-smoke`；该阶段必须已经完成 |
| `--run-dir` | 冻结数据目录 | 从该次实验的 `experiment.json` 读取 | 已包含清单和 GT 的目录；内容须与原评测一致 |
| `--output` | 将补全报告保存到其他文件，供查看或后续复制 | 更新 `<report-root>/<phase>/report.md` | 例如 `/mnt/workspace/report-review/report.md`；同目录生成 `report_context.json`，图片链接仍相对原阶段目录 |
| `-h` / `--help` | 查看帮助后退出 | 不启用 | 只写参数名字 |

命令依据历史推理记录描述模型配置，依据当前阶段的帧清单统计数据，不会将全量数量写入试评测报告。更新报告后，既有预测和指标继续保留；如果修改代码后需要重新执行模型推理，仍使用新的报告目录。

## 5. 模型配置怎么填写

当前路径已经配置好，直接运行不需要编辑。需要换环境、项目路径或权重时，复制 [benchmark_config.json](benchmark_config.json)，修改副本，并通过 `--config` 指定副本；使用新的报告目录。

配置是 JSON：路径和字符串加双引号，整数不加引号，布尔值填小写 `true` / `false`，不能写注释。下表中的 `options.xxx` 表示模型条目内 `options` 对象里的字段。

| 字段 | 意义 | 可填内容 / 当前设置 |
|---|---|---|
| `default_models` | 不写 `--models` 时运行的模型列表 | 模型 ID 数组；当前为 `["3dmood", "wilddet3d", "cutr-rgb", "cutr-rgbd"]` |
| `environment.HF_HOME` | Hugging Face 模型缓存位置 | 已有完整 BERT 文件的目录；当前 `/root/.cache/huggingface` |
| `environment.TORCH_HOME` | PyTorch 权重缓存位置 | 目录路径；当前 `/mnt/workspace/.cache/3d-mood/torch` |
| `models` | 按模型 ID 保存配置 | 使用上面的五个 ID，各条目填写下面这些字段 |
| `python` | 该模型虚拟环境中的 Python 程序 | 可执行文件的绝对路径，例如 `/mnt/workspace/venvs/ca1m-ppu/bin/python`；不能只填环境名称 |
| `adapter` | 把模型输出转换为统一三维框的适配器 | 3D-MOOD 填 `3dmood`；WildDet3D 填 `ioutest.adapters.wilddet:WildDetAdapter`；CuTR 三组填 `ioutest.adapters.cutr:CuTRAdapter` |
| `semantic_labels` | 模型是否输出官方 17 类标签 | 3D-MOOD / WildDet3D 为 `true`，CuTR 三组为 `false`；不随意改动 |
| `options.project_root` | 模型项目根目录 | 3D-MOOD：`/mnt/workspace/3dmood`；WildDet3D：`/mnt/workspace/wilddet3d`；CuTR：`/mnt/workspace/CA1M` |
| `options.checkpoint` | 模型权重文件 | 已存在且匹配模型的 `.pt` / `.pth` 文件路径，具体当前值见配置文件 |
| `options.lingbot_checkpoint` | WildDet3D 的 LingBot 文件，用来构建深度分支结构 | 仅 WildDet3D 使用；当前 `/mnt/workspace/TEMP-FILE-STATION/model.pt`；不表示输入实测深度 |
| `options.max_detections` | 每帧最多保留多少个预测框 | 3D-MOOD / WildDet3D 当前 `300`；填正整数，3D-MOOD 限制 `1`～`900`；CuTR 不使用此字段，保留前景 top-100 |
| `options.score_threshold` | 推理时提前过滤低分框 | 仅 3D-MOOD 使用；可填 `0` ≤ 数值 < `1`，当前 `0.0`。正式 AP 保持 0，避免提前删框 |
| `options.depth_model` | 是否使用 CuTR RGB-D 模型 | 仅 CuTR；`cutr-rgbd` 为 `true` 并配 `cutr_rgbd.pth`，其余 CuTR 组为 `false` 并配 `cutr_rgb.pth` |
| `options.use_gravity` | 是否给 CuTR 输入已知重力方向 | 仅 CuTR；`cutr-rgb-gravity` 为 `true`，RGB / RGB-D 组为 `false` |
| `options.input_long_edge` | CuTR 模型输入图片的长边像素数 | 可选字段，未写时为 `1024`；可填整数 `32`～`1024`。RGB 与 RGB-D 对比时保持相同值 |

配置字段不是命令行参数，例如应在 JSON 的 `options` 内填写 `input_long_edge`，不能把 `--input-long-edge` 加到 `benchmark` 命令。

## 6. 更换评测数据或重新切帧时使用

当前 10,980 帧不需要执行本节。改变扫描范围、每扫描帧数或 GT 筛选设置时，准备一份**新的**清单，并使用新的 `--run-dir`。数据准备只支持当前 ARKitScenes 格式，不支持直接填任意图片文件夹。

例如：最多选 10 个扫描，每扫描最多抽 10 帧，生成 RGB 和深度缓存：

```bash
python -B -m ioutest prepare \
  --run-dir ioutest/runs/arkit-custom-v1 \
  --max-scans 10 \
  --frames-per-scan 10

python -B -m ioutest cache-images --run-dir ioutest/runs/arkit-custom-v1
python -B -m ioutest cache-depth --run-dir ioutest/runs/arkit-custom-v1

python -B -m ioutest benchmark \
  --run-dir ioutest/runs/arkit-custom-v1 \
  --phase full \
  --report-root "$HOME/report/arkit-custom-v1"
```

这里 `full` 表示评测新清单中的全部帧，最多 100 帧，不是原清单的 10,980 帧。

### `prepare` 参数

| 参数 | 意义 | 默认值 | 可填内容 / 示例 |
|---|---|---|---|
| `--dataset-root` | 原始 ARKitScenes 根目录 | `/mnt/data/openBenchmark/ARKitScenes` | 包含 `source/metadata.csv`、`3dod/Validation/` 的目录 |
| `--run-dir` | 保存新帧清单和 GT 的目录 | `ioutest/runs/arkit-validation-v1` | 新目录，例如 `ioutest/runs/arkit-custom-v1`；不能位于原始数据目录内 |
| `--scans` | 明确指定扫描编号 | 所有 Validation 扫描 | metadata 中的验证扫描 ID，空格分隔，例如 `41069021 41069042`；与 `--max-scans` 同用时在这些扫描中再抽样 |
| `--max-scans` | 最多选择多少个扫描 | 不限制 | 正整数，例如 `10`、`100`；实际数量不超过可选扫描数 |
| `--frames-per-scan` | 每扫描最多抽多少帧 | `20` | 正整数，例如 `10`、`20`；可配对帧不足时使用实际帧数 |
| `--minimum-interval` | 相邻候选帧的最小时间间隔，单位秒 | `0.5` | 非负有限数值，例如 `0`、`0.5`、`1.0` |
| `--depth-stride` | 判断 GT 可见性时，深度图每隔多少像素取样 | `2` | 正整数；`1` 为逐像素，`2` 为每两像素取样；只影响 GT 筛选，不降低 RGB-D 输入深度的分辨率 |
| `--visible-points` | 框内观测深度点数必须严格大于这个数，才作为有效 GT | `20` | 非负整数，例如 `0`、`20`、`50`；当前 `20` 表示至少 21 个点 |

扫描抽样种子固定为 `20261010`，没有种子参数。每次对比使用同一份清单和 GT。

### `cache-images` / `cache-depth` 参数

| 参数 | 意义 | 默认值 | 可填内容 / 示例 |
|---|---|---|---|
| `--run-dir` | 读取哪个已完成的帧清单 | `ioutest/runs/arkit-validation-v1` | 与 `prepare` 使用的目录一致 |
| `--image-cache` | 将图片或深度保存到哪里；两个命令都使用这个参数名 | `ioutest/cache/<run-dir目录名>` | 可写目录；RGB 与深度应保存到同一个缓存根目录 |
| `--workers` | 同时处理多少个扫描 | `4` | 正整数，例如 `1`、`4`、`8`；增大可能加快提取，也会增加资源使用 |

`benchmark` 自动读取 `ioutest/cache/<run-dir目录名>`，没有指定缓存路径的命令参数。要用于 `benchmark`，按默认位置生成缓存。

缓存中 `raw_images/` 是原始方向 RGB，`images/` 是转正 RGB，`depth_images/` 是原始方向深度；`images.jsonl` 保存转正 RGB 的配套内参，`depths.jsonl` 保存深度路径和摘要。其他项目直接使用 `images/` 时不要再次转正，并使用配套内参。`cache/` 和 `runs/` 已被 Git 忽略，图片、深度、预测不会随普通 `git add` 提交到 GitHub。

## 7. 单模型命令参数（按需使用）

日常四组评测使用 `benchmark`。本节是原有单模型工具，只支持带官方 17 类标签的预测，输出到 `--run-dir` 内；其预测文件格式与 `benchmark` 不同，不能直接互相代用。CuTR RGB / RGB-D 通过 `benchmark` 运行。

| 命令 | 用途 |
|---|---|
| `doctor` | 检查 3D-MOOD 环境、设备和离线权重；不是全部模型的检查 |
| `infer` | 运行一个模型并保存预测 |
| `evaluate` | 读取完整预测，计算 AP，无需设备推理 |
| `visualize` | 绘制真实框或真实框加预测框 |
| `import-predictions` | 导入外部生成的带 17 类标签的 JSONL 预测 |
| `run` | 顺序执行 `prepare → infer → evaluate`；不自动执行 `cache-images` / `cache-depth` |

以下表格中的“适用命令”指参数可以加在哪个命令后面。`run` 还可以使用上面的全部 `prepare` 参数。

| 参数 | 适用命令 | 作用、可填内容与默认值 |
|---|---|---|
| `--run-dir` | 除 `doctor` 外全部单模型命令 | 数据清单及结果目录，默认 `ioutest/runs/arkit-validation-v1` |
| `--model-id` | `infer`、`evaluate`、`visualize`、`import-predictions`、`run` | 单模型结果名称，默认 `3dmood-swin-b`；`visualize` 不填则只画 GT，导入时必须填。用英文字母或数字开头，后续允许字母、数字、`_`、`.`、`-`，例如 `mood-test-v2`；不是选择模型的 `--models` |
| `--device` | `doctor`、`infer`、`run` | 设备名，如 `cuda`、`cuda:0`、`cpu`；3D-MOOD 默认 `cuda`。CPU 只适合小规模验证 |
| `--checkpoint` | `doctor`、`infer`、`run` | 已存在的权重文件路径；3D-MOOD 默认 `/mnt/workspace/.cache/3d-mood/torch/hub/checkpoints/gdino3d_swin-b_120e_omni3d_834c97.pt` |
| `--adapter` | `infer`、`run` | 默认 `3dmood`；开发者可填可导入的 `模块名:类名`，例如 `my_adapter:YourAdapter` |
| `--adapter-options` | `infer`、`run` | 适配器选项 JSON 文件路径，默认不提供；文件内容是选项对象，不是整份 `benchmark_config.json`。同名命令行选项优先于 JSON |
| `--max-detections` | `infer`、`run` | 3D-MOOD 每帧候选上限，默认 `300`，可填整数 `1`～`900` |
| `--score-threshold` | `infer`、`run` | 3D-MOOD 推理分数下限，默认 `0.0`，可填 `0` ≤ 数值 < `1`；正式 AP 保持 0 |
| `--training-annotations` | `infer`、`run` | 模型训练所用 Omni3D/COCO 标注 JSON 文件，一个或多个路径用空格分隔；默认不提供。用于额外排除模型已训练场景，文件必须有 `images` 字段 |
| `--image-cache` | `infer`、`visualize`、`run` | 指定已完成的 RGB 缓存目录，默认自动检测 `ioutest/cache/<run-dir目录名>`；找不到默认缓存时读取 ZIP |
| `--no-image-cache` | `infer`、`visualize`、`run` | 只写开关，强制读取原始 ZIP；不能与 `--image-cache` 同时使用 |
| `--iou-thresholds` | `evaluate`、`run` | 三维 IoU 匹配阈值，默认 `0.25 0.50`；每个数必须满足 `0` < 数值 ≤ `1`，空格分隔且不重复，例如 `0.15 0.25 0.50` |
| `--distance-thresholds` | `evaluate`、`run` | 三维中心距离阈值，单位米，默认 `0.25 0.50 1.00`；填正的有限数值，空格分隔且不重复，例如 `0.5 1.0 2.0` |
| `--report-name` | `evaluate`、`run` | 指标文件名的前缀，默认 `metrics`；例如 `threshold-test`，字符规则与 `--model-id` 相同，不填 `.json` 后缀 |
| `--max-frames` | `visualize` | 最多绘制多少帧，默认 `12`；填正整数，例如 `3`、`12` |
| `--score-threshold` | `visualize` | 只显示分数达到该值的预测，默认 `0.1`；一般填 `0`～`1`，不影响已经计算的 AP |
| `--show-ignore` | `visualize` | 只写开关，额外显示灰色忽略 GT 框；默认不显示 |
| `--path` | `import-predictions` | 必填，外部预测 JSONL 文件的路径 |
| `--benchmark-id` | `import-predictions` | 必填，填写该清单 `protocol.json` 中 `benchmark_id` 的完整值，不能自己取名 |

所有命令支持 `-h` / `--help`。单模型命令还有 `--debug` 开关，用于显示完整异常信息，写在子命令前：`python -m ioutest --debug infer ...`；`benchmark` 不支持该开关。

### 单模型示例：推理后修改评分阈值

```bash
python -B -m ioutest infer --model-id 3dmood-swin-b

python -B -m ioutest evaluate \
  --model-id 3dmood-swin-b \
  --iou-thresholds 0.15 0.25 0.50 \
  --distance-thresholds 0.25 0.50 1.00 \
  --report-name threshold-test
```

结果位于 `ioutest/runs/arkit-validation-v1/models/3dmood-swin-b/reports/`。已有完整单模型预测时，只需执行 `evaluate`。

### 外部预测的填写格式

`import-predictions` 的 JSONL 文件每帧一行，必须覆盖清单全部帧；无检测也要写 `"detections": []`。下面的帧 ID 仅为格式示例，实际填写 `frames.jsonl` 中的 ID：

```json
{"frame_id":"41069021:41069021_305.377","status":"ok","detections":[{"class_id":13,"score":0.9,"center":[0,0,2],"size":[1,1,1],"rotation":[[1,0,0],[0,1,0],[0,0,1]]}]}
```

`class_id` 按 [官方类别表](PLAN.md#真值类别与文本提示词)从上到下编号为 `0`～`16`，例如 `cabinet=0`、`chair=13`、`sofa=16`；`score` 是有限分数；`center` 是米制三维中心；`size` 是框局部 x/y/z 方向的完整边长；`rotation` 是局部坐标到转正相机坐标的右手正交旋转矩阵。相机坐标 x 向右、y 向下、z 向前。开发者接口见 [schema.py](schema.py)。
