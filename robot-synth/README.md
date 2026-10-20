# LIBERO ground-truth 采集

工程目录：`/mnt/workspace/3dmood/robot-synth`。本阶段只获取仿真真值，不接入模型训练，不使用模型预测或伪标签。

目标是独立的 **11,000 张相机图像：train 10,000 张、eval 1,000 张**。默认同一状态采集 `agentview`、`robot0_eye_in_hand` 两个视角，所以目标为 5,500 个仿真状态。小批量限制和完成数量均按相机图像计数；相机对一起提交。

## 当前验证状态

采集代码、离线清点和冻结计划可以在现有 Python 环境运行。完整 GT 采集需要单独安装 MuJoCo 3.2.7、OSMesa 和 LIBERO/robosuite 资产。**依赖检查通过仍不等于渲染验证通过**：请先运行真实渲染测试和 200 张试采。`doctor.json` 和实际清点／计划输出位于 `runs/libero_gt_v1/`。

当前执行终端最初没有上述仿真依赖和资产，网络代理不可访问。工程不会自动下载，不会以测试数据填充真实输出，也不会用成功的模拟单元测试声称真实渲染已通过。实际运行证据见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。

## 模块结构

```text
configs/libero_gt_v1.yaml       输入、输出、配额、相机和 GT 参数
robot_synth/
  sources/libero_hdf5.py       原始 HDF5 只读清点／状态读取
  planning/                   场景归组、划分、状态选择和计划冻结
  simulation/                 XML 资产映射、MuJoCo 状态恢复和同步采集
  groundtruth/                实例层级、完整几何框和坐标转换
  storage/                    暂存、校验、最后提交和断点续采
  validation/                 几何、深度、分割、数量与划分检查
  reporting/                  GT 预览和质量汇总
  collection.py               采集流程与配额控制
  cli.py                      命令行
tests/                        合成夹具测试及独立真实渲染测试
```

## 存储与配置

默认配置：

| 用途 | 路径 |
|---|---|
| 原始数据（只读） | `/mnt/data/opendatasets/LIBERO` |
| 最终数据 | `/mnt/data/opendatasets/LIBERO_GT_v1` |
| 清点、计划、日志和本地锁 | 工程下 `runs/libero_gt_v1` |
| 本地暂存 | `/tmp/robot-synth/libero_gt_v1` |
| LIBERO 资产 | 工程下 `third_party/LIBERO/libero/libero/assets` |
| robosuite 资产 | 工程下 `third_party/robosuite/robosuite/models/assets` |

源码、生成结果和原始数据的目录不得互相包含。可以通过 YAML 调整所有路径；`--input`、`--output` 也可覆盖，但覆盖参数必须在所有命令中一致，且不能改变已有冻结计划的身份。`--config` 默认指向工程内的配置文件。

初期采用 512×512、无损 PNG RGB、uint16 实例 PNG、float32 米制深度 NPZ。先试采测量实际大小，再扩展；不要把全量结果堆在剩余空间有限的源码盘。

## 先运行无需仿真的步骤

现有 `wilddet3d-ppu` venv 已有 HDF5/NumPy/Pillow/YAML，可只借它的解释器运行只读数据处理，**无需安装或改动该 venv**：

```bash
cd /mnt/workspace/3dmood/robot-synth
/mnt/workspace/venvs/wilddet3d-ppu/bin/python -B -m robot_synth inspect
/mnt/workspace/venvs/wilddet3d-ppu/bin/python -B -m robot_synth plan
/mnt/workspace/venvs/wilddet3d-ppu/bin/python -B -m robot_synth doctor
```

首次清点读取每段示范的元数据，不读取所有 RGB；OSS 上可能需要数分钟。之后在源文件大小／修改时间未变化时复用清点缓存；`inspect --refresh` 可重读。计划额外读取选中示范的状态数组，对选中状态及 XML 保存 SHA256。

`doctor` 为只读依赖和资产检查，缺失时返回退出码 2，不会创建最终数据集。不能以它代替实际渲染测试。

## 独立采集环境

无需策略执行或机器学习训练包：直接加载记录中的 MJCF/XML，用原始 `[time,qpos,qvel]` 恢复 MuJoCo，再调用 `mj_forward` 更新几何。编译模型必须满足 `na == 0` 且状态维度完全匹配；否则停止。`mujoco==3.2.7` 是本工程目标渲染版本，其他版本需要重新验证，不能与已有采集混用。

在有网络的专用采集镜像中执行：

```bash
cd /mnt/workspace/3dmood/robot-synth
bash scripts/bootstrap_runtime.sh
source .venv/bin/activate
python -m robot_synth doctor
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  python -m pytest -q tests/test_native_renderer.py
```

安装脚本仅在工程 `.venv` 安装依赖，在 `third_party` 准备资产；不安装 LIBERO 的策略训练依赖，也不修改现有 PPU 模型环境。系统需要 OSMesa 图形库；脚本不会自动改动系统包。可用提供的 `Dockerfile` 构建独立镜像：

```bash
docker build -t robot-synth:libero-gt-v1 .
```

容器内工程路径与主机一致，以复用已冻结的计划；不要用工程根目录挂载覆盖镜像内已经安装的 `.venv` 和资产。挂载原始数据为只读、结果和运行记录为可写：

```bash
mkdir -p /mnt/data/opendatasets/LIBERO_GT_v1 /tmp/robot-synth
docker run --rm \
  -v /mnt/data/opendatasets/LIBERO:/mnt/data/opendatasets/LIBERO:ro \
  -v /mnt/data/opendatasets/LIBERO_GT_v1:/mnt/data/opendatasets/LIBERO_GT_v1 \
  -v "$PWD/runs:/mnt/workspace/3dmood/robot-synth/runs" \
  -v /tmp/robot-synth:/tmp/robot-synth \
  robot-synth:libero-gt-v1 doctor
```

实际采集时将最后的 `doctor` 换为 `collect --limit 200`；检查时换为 `validate --allow-partial` 或 `report`。在同一挂载方式下运行后续命令，以保留计划和进度。容器本身不要求访问 PPU。

Dockerfile 和安装脚本在本终端未进行联网构建验证。资产默认取 LIBERO master 和 robosuite v1.4.0；首次验证后记录实际 Git 提交，后续用 `LIBERO_REF`、`ROBOSUITE_REF` 指定相同提交重建。采集前会对所有记录资产计算内容 SHA256，并将实际运行版本、源码和资产摘要保存到数据集身份中。已有资产目录不会被安装脚本自动更新或覆盖。

## 场景划分和数量

默认仅使用 `libero_90`、`libero_10`，合并为候选池后重新按场景组划分，不沿用策略学习的原始目录划分。

| 类型 | 计划评测场景组数 | train 图像配额 | eval 图像配额 |
|---|---:|---:|---:|
| KITCHEN | 2 | 5,000 | 500 |
| LIVING_ROOM | 1 | 3,000 | 250 |
| STUDY | 1 | 2,000 | 250 |
| 合计 | 4 | 10,000 | 1,000 |

同一 `SCENE` 的所有任务归同组。发现跨编号重复的标准化示例 XML 时合并分组；此检查不证明不同编号等于不同独立房间，也不保证资产互不重叠。`scene_split.json` 固定组名单、别名和配额；可在首次计划前用 `eval_scene_ids` 指定评测组。

每个训练场景默认选择 20 段示范，评测场景 10 段，先在任务之间轮流选择示范，避免被单一任务占满。用 XML 的关节定义取得真实 nq/nv，仅将 qpos 变化作为静态帧初筛，时间和速度不作为变化信号；它不是图像感知去重。轨迹内均匀选择状态，默认额外冻结 50% 候补，质量拒绝后按同场景配额补选。采集还会剔除两个相机图像同时完全相同的状态。

对近似图像的感知去重、语义动作阶段分类不作已实现承诺。若候补池耗尽，任务会报错并保留已有结果，不复制帧凑数。调整采样配置需建立新 run_dir 和输出目录，以保持原计划可追溯。

## 试采、续采、检查

以下在独立采集环境中执行：

```bash
cd /mnt/workspace/3dmood/robot-synth
source .venv/bin/activate

# 一共选择 200 张，尽量均匀覆盖场景；不是额外新增 200 张。
python -m robot_synth collect --limit 200
python -m robot_synth validate --allow-partial
python -m robot_synth report --max-previews 40

# 先检查质量报告和 GT 预览；验收后再继续到完整数量。
python -m robot_synth collect --resume
python -m robot_synth validate
python -m robot_synth report
```

`--limit` 必须为相机数的整数倍。已达到该限制的结果会复用，不自动继续到下一张；去掉限制才扩大到完整配额。相机对必须同时通过质量检查。首个状态恢复／渲染运行异常会停止；连续质量拒绝达到配置阈值也停止，并保存原因。已拒绝状态在续跑时跳过，按冻结候补补足。

每个状态先写本地暂存，复制到输出后逐文件校验，最后写入 `COMMIT.json`。没有有效 COMMIT 的目录不计入数据集。断点续采会检查原计划、运行身份及已有文件 SHA256；损坏的已提交结果会报错，不静默覆盖。只清理本工具明确拥有且未提交的半成品目录。

同一输出目录只允许单个写入进程。本地 `flock` 防止同一 run_dir 重复启动；这不是跨主机分布式锁，勿在多个实例同时写入同一 OSS 前缀。OSS 汇总文件是进度快照，逐状态 COMMIT 是样本完整性的依据。

## 数据格式与 GT 约定

```text
LIBERO_GT_v1/
  dataset_info.json              配置、实际版本、资产和源码摘要
  scene_split.json               冻结划分
  sampling_plan.json             完整冻结状态／候补清单
  categories.json                稳定类别名称、原始类别与实例类型
  manifest.json                  完成状态、已接收／拒绝状态和错误
  provenance/<episode>.xml.gz    原始场景 XML（保留来源）
  train/<scene>/<state_id>/
    state.npy                    实际恢复的 float64 仿真状态
    agentview/
      rgb.png
      depth.npz                  z_m: float32；valid: bool
      instance.png               uint16；0 为未标注世界背景
      annotation.json
    robot0_eye_in_hand/...
    PENDING.json                 本工具写入所有权记录
    COMMIT.json                  最后提交及文件 SHA256
  eval/<scene>/<state_id>/...
```

`annotation.json` 包含逐实例的 `category`、`raw_category`、`instance_id`、`parent_instance_id`、`body_name`、`geom_ids`、`segmentation_ids`、`center_world`、`center_camera`、`size_xyz`、`R_world_from_box`、`R_camera_from_box` 和可见像素／2D 框／来源。

- 长度单位米；世界坐标 MuJoCo Z 向上；相机坐标 OpenCV x 右、y 下、z 前。
- 内参以连续图像边缘坐标定义，像素 `(v,u)` 的中心为 `(u+0.5,v+0.5)`；2D 框为右下端点不包含的 xyxy。
- 深度是相机 Z，而非到相机中心的欧氏距离；无几何背景深度设 0，并用 valid 排除。
- 完整物体框在当前根 body 的轴系内计算，对组 1 的完整可视 mesh 顶点取范围；球、椭球、圆柱、胶囊和方盒使用解析支持范围。只用视觉几何，不用碰撞简化框或可见深度点拟合。它不是最小体积 OBB。
- 包围框中心与仿真 body 原点的偏移单独保存。运动部件按当前变换计算；整柜框随活动部件的当前外形范围变化，这一口径明确记录在 box_definition 中。
- 活动部件是子实例，类别默认 `articulated_part`，不猜测它一定是抽屉。根实例的 `segmentation_ids` 包括其子部件 ID，可见像素按层级聚合。机器人独立保留；预览默认只画物体，避免机械臂和部件框淹没画面。
- worldbody 下的背景道具／贴图不推断成语义家电；其像素可有有效深度，但实例 ID 为 0。可视几何组可配置，改变后需新数据集。
- `occlusion_fraction` 目前为 null，因为没有额外做无遮挡重渲染，不用框面积估算伪装成真实遮挡率。`box_truncated` 指完整 3D 框投影的截断，不等于精确 mesh 轮廓截断率。
- 所有实例均保留，包括不可见实例；`visible` 和 `visible_pixels` 区分当前相机的可见性。类别字符串稳定，别名规则记录在配置，不预先绑定训练框架类别 ID。

质量检查包含合法旋转／正尺寸、标定互逆、可见框与实例掩码一致、深度反投影点落在完整框内、双相机恢复的世界几何一致、完整文件、原状态摘要、完全重复图像对和场景隔离。3D 框投影通常大于可见掩码，不要求二者边缘重合。

## 测试

```bash
python -m pytest -q
```

单元测试使用隔离的临时 HDF5／明确标识的测试回放器，验证场景划分、配额、状态索引、几何、异常提交、续跑与损坏拒绝。真实 MuJoCo 渲染测试缺依赖时明确 skip，不能把单元测试通过当成真实采集成功。

输出检查图位于 `runs/libero_gt_v1/quality_report/`。`validate` 默认要求达到完整配额；200 张试采使用 `--allow-partial`，完整验收不加此参数。后续 RoboCasa 只需增加来源和回放适配，不涉及本阶段模型训练。
