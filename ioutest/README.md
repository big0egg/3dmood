# 三维框评测

只读取已处理的测试集，支持原有 AP 和中心、尺寸、方向、检出／漏检、完整感知延迟五项指标。微调由独立训练模块负责；本模块不制作数据、不下载权重。

## 运行步骤

在 DSW 原生 Terminal 执行。先修改 [配置文件](configs/robot_boxing_v1.json)，填写已有测试集、模型环境和权重的实际位置。

```bash
cd /mnt/workspace/3dmood
source /mnt/workspace/venvs/3d-mood-ppu-clean/bin/activate

# 1. 只读检查；不生成评测文件
python -B -m ioutest --config ioutest/configs/robot_boxing_v1.json --stage check

# 2. 默认：3D-MOOD、WildDet3D 官方权重，HOT3D 已冻结的 3000 帧测试集，TEXT 输入
python -B -m ioutest --config ioutest/configs/robot_boxing_v1.json --stage all
```

选择模型、权重和输入：

```bash
python -B -m ioutest --config ioutest/configs/robot_boxing_v1.json \
  --models wilddet3d --variants official \
  --datasets hot3d_eval --tracks text gt-box --stage all
```

微调版本使用 `finefune export-eval` 导出的配置，其中版本名为 `finetuned`；随附配置仅保留官方版本。中断后重跑同一命令，完整预测会复用。补算指标用 `--stage score`，随后用 `--stage report`；重新实测延迟用 `--stage latency`，需已有完整预测。复用预测时的延迟来自原测量，不是本次测速。

## 命令参数

多个模型、版本、数据集和输入条件按组合执行。

| 参数 | 意义 | 可以填写的内容／默认值 |
|---|---|---|
| `--config` | 配置文件 | JSON 路径；默认 `ioutest/configs/robot_boxing_v1.json` |
| `--models` | 模型 ID，可填多个 | 默认配置包含 `3dmood`、`wilddet3d`、`unipose9d`、`unipose9d-rgbd`、`cutr-rgb`、`cutr-rgbd`；不填使用配置的 `selection.models` |
| `--variants` | 权重版本，可填多个 | 随附配置为 `official`；微调导出配置为 `finetuned`。填写所用配置实际存在的版本 ID；默认取 `selection.variants` |
| `--datasets` | 测试集 ID，可填多个 | `hot3d_eval`、`libero_original_eval`、`libero_realistic_eval`、`arkit_eval`；默认 `hot3d_eval` |
| `--tracks` | 输入条件，可填多个 | `text`：自主定位，3D-MOOD／WildDet3D／UniPose 使用固定类别文本，CuTR 仅做无类别几何检测；`gt-box`：RGB＋GT 二维框，仅 WildDet3D／UniPose 支持；默认 `text` |
| `--stage` | 执行阶段 | `check` 检查；`infer` 推理；`score` 离线评分；`latency` 重新实测并评分、出报告；`report` 汇总；`all` 完整评测，默认值 |
| `--debug` | 显示完整错误堆栈 | 开关，不需要填写值 |
| `--help` | 查看命令帮助 | 开关，不需要填写值 |

## 配置参数

相对路径：数据文件相对于对应的 `datasets.<ID>.root`；模型、训练来源及输出路径相对于配置文件所在目录。随附配置以绝对路径为主。未恢复的环境／权重和测试集请先更新路径，检查只要求实际选中的模型及依赖。

| 配置项 | 意义 | 可以填写的内容 |
|---|---|---|
| `output.root` | 预测、日志和报告目录 | 可写目录，默认 `/root/robot-boxing-eval-v1`；不能位于源数据目录内 |
| `selection` | 不传命令参数时的选择 | `models`、`variants`、`datasets`、`tracks`，均为 ID 列表 |
| `datasets.<ID>.format` | 已处理数据格式 | `hot3d_prepared`、`arkit_prepared`、`robot_synth_prepared`、`robot_realism_prepared` |
| `datasets.<ID>.root` | 数据根目录 | 已有目录 |
| `datasets.<ID>.manifest` | 测试清单 | HOT3D 的 `eval.jsonl`；ARKit 的 `frames.jsonl`；仿真 `manifest.json`；生成数据 `manifests/eval_general.jsonl` |
| `datasets.<ID>.protocol` | 冻结协议与类别 | HOT3D／ARKit 的 `protocol.json` 路径 |
| `datasets.<ID>.image_cache` | 已有本地 RGB 缓存 | ARKit 缓存目录，包含完整 `images.jsonl`、`cache_summary.json` 和图片 |
| `datasets.<ID>.categories_file` / `classes` | 仿真冻结类别表 | `categories.json` 路径，或显式类别名称列表 |
| `datasets.<ID>.eval_split` / `camera` | 仿真测试划分／视角 | 默认 `eval`／`agentview` |
| `datasets.<ID>.target_kinds` | 仿真目标类型 | 默认 `["object"]`；跳过子实例，避免父子重复 |
| `datasets.<ID>.minimum_visible_pixels` | 仿真有效目标可见像素下限 | 非负整数，默认 `32`；不达标目标设为 ignore |
| `datasets.<ID>.symmetries` | 可选对称性文件 | JSON 路径；`by_class.<类别名>` 或 `by_object.<物体ID>` 下填写 `rotations`（SO(3) 矩阵列表）、`continuous_axis`（`0/1/2`）；不填仅处理框轴等价 |
| `datasets.<ID>.depth_manifest` | 可选的已有实测深度清单 | JSONL；每行包含 `frame_id`、`depth_path`、`depth_kind: "sensor"`、`depth_scale`（到米的倍率，默认 `1`）。RGB-D 配置必须提供；不使用仿真 GT 深度替代 |
| `models.<ID>.family` | 模型类型 | `3dmood`、`wilddet3d`、`unipose9d`、`cutr` |
| `models.<ID>.python` | 模型解释器 | 已存在的虚拟环境 Python 路径 |
| `environment` / `models.<ID>.environment` | 全局／模型缓存目录，模型设置优先 | `HF_HOME`、`HF_HUB_CACHE`、`TRANSFORMERS_CACHE`、`TORCH_HOME`、`MPLCONFIGDIR` 的本地路径；3D-MOOD 已固定使用现有 BERT 缓存 |
| `models.<ID>.options` | 原生推理参数 | `project_root`、`checkpoint`；WildDet3D 加 `lingbot_checkpoint`；UniPose 加 `config` 与采样参数；RGB-D 配置 `depth_model: true` |
| `models.<ID>.variants.<版本>` | 权重版本及训练来源 | 空对象表示使用基础权重；`checkpoint` 可指定完整权重；`head_checkpoint` 指定三维头增量；`training_manifest`、`training_root` 用于重叠检查；可指定 `epochs_completed`、`training_identity`、`training_benchmark_id` 核验来源 |
| `frontend` | UniPose 的 RGB 前端 | `python`、`project_root`、`sam2_checkpoint`、`sam2_config`、`grounding_model`、`moge_checkpoint`；`box_threshold`、`text_threshold`、`max_detections` 为前端检测设置 |
| `metrics.iou_thresholds` | AP3D／达标阈值 | `(0,1]` 内不重复数值，默认 `[0.25,0.5]` |
| `metrics.center_thresholds` | 中心距离 AP／达标阈值 | 不重复正数，单位米，默认 `[0.02,0.05,0.1,0.25,0.5,1.0]` |
| `metrics.association_iou` | TEXT 几何诊断的二维投影关联阈值 | `(0,1]`，默认 `0.5`；不改变 AP 标准匹配 |
| `metrics.score_floor` | 诊断与检出工作点 | 有限数值，默认 `0`；主 AP 不按此值过滤 |
| `timing.warmup_frames` | 正式计时前重复首个待测帧的次数 | 非负整数，默认 `1`；不计入主延迟 |
| `execution.device` / `seed` / `threads` | 推理设备／随机种子／CPU 线程数 | `cuda` 或显式 `cpu`；整数种子；正整数线程数。实际模型须支持所选设备，不自动回退 CPU |

运行时主终端显示当前任务、推理／测速进度、复用帧数、已用时及预计剩余时间，详细输出仍保留在 worker.log。加载／预热阶段先显示等待状态；缓存复用和加载时间不用于推算新帧速度。非交互终端约每 5 秒输出一次进度。

report.md 分表展示 AP、中心、尺寸、方向、检出／条件定位和延迟。误差包含均值、中位数、P90，中心及尺寸另有分轴统计。JSON 几何误差单位为米，方向为度，相对尺寸及 AP／检出比例为 0～1；报告与 CSV 按列标注 cm／%。漏检不填零误差，覆盖率和全目标达标率同时输出。GT-BOX 显示输出失败率和目标绑定的定位达标率，输出覆盖率不代表自主检测召回率。方向评价框几何方向。实际延迟包含必要进程通信，排除图片解码、模型初始化、预热及结果写盘；阶段耗时之和另存，GT-BOX 不包含外部检测器获取二维框的耗时。

## 输出结构

```text
/root/robot-boxing-eval-v1/
├── resolved_config.json                 # 实际使用的配置
├── experiment.json                      # 本次选择和数据标识
├── readiness.json                       # 选中环境及设备检查
├── status.json                          # 总状态
├── audits/<测试集>_dataset.json         # 数据来源与标签状态
├── runs/<测试集>/<模型>/<权重版本>/<输入条件>/
│   ├── request.json                     # 冻结输入和权重摘要
│   ├── model.json                       # 实际加载的模型与环境
│   ├── training_audit.json              # 训练／测试重叠检查
│   ├── predictions/*.json.gz            # 逐帧预测与原始计时
│   ├── metrics.json                     # AP 和五项指标
│   ├── targets.csv                      # 逐目标误差、缺失和关联状态
│   ├── classes.csv                      # 逐类 AP
│   ├── latency.json                     # 实测延迟与阶段耗时
│   ├── latency_records/*.json.gz        # 独立测速时产生
│   ├── progress.json                    # 当前推理进度
│   └── logs/                            # 子进程任务与 worker.log
└── reports/
    ├── comparison.csv                   # 模型／版本／数据集对照
    ├── per_class.csv                    # 各任务逐类结果
    ├── results.json                     # 汇总及结果来源
    └── report.md                        # 概览及 AP、五项诊断的详细分表
```
