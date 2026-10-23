# HOT3D微调与两类输入评测：完整运行说明

按以下顺序完成一次实验：**准备数据 → 官方权重两类评测 → 微调 → 微调权重两类评测 → 最终报告**。
详细参数、W&B和数据选择规则见 [README.md](README.md)。所有模型命令在能访问PPU的DSW原生Terminal执行。

## 实验包含哪些对比

| 输入条件 | 模型 | 微调前与微调后分别测什么 | 主要指标 |
| --- | --- | --- | --- |
| RGB＋固定类别文本＋相机内参 | 3D-MOOD、WildDet3D | 自行定位目标并估计三维框 | geometry/semantic AP3D、APcenter |
| RGB＋GT二维框＋相机内参 | WildDet3D | 给定目标图像位置后估计三维框 | 逐目标成功率；辅助geometry AP |

一共六组结果：文本两模型×两种权重为四组，GT二维框一模型×两种权重为两组。
两类评测使用同一批3,000张图片、14,373个有效目标，以及同一套微调权重。
微调采用文本训练流程，不为二维框模式再训练一次；框提示结果反映几何微调向该输入模式的迁移。
3D-MOOD当前原生推理入口不消费框提示，所以不参加GT二维框条件。

## 步骤1：设置路径并准备数据环境

在当前Terminal设置变量；重新打开Terminal后需要重新设置：

```bash
FT=/mnt/workspace/3dmood/hot3d-finetune
MOOD_PY=/mnt/workspace/venvs/3d-mood-ppu/bin/python
WILD_PY=/mnt/workspace/venvs/wilddet3d-ppu/bin/python
DATA_PY=/mnt/workspace/venvs/hot3d-data/bin/python
ARTIFACTS=/root/hot3d-finetune-v1

bash "$FT/setup_data_env.sh"
```

FT是代码目录，ARTIFACTS是产物目录，两者不是同一个路径。
首次依赖安装需要网络；两个模型已有各自的PPU环境，数据处理使用独立hot3d-data环境。

## 步骤2：检查环境和源数据

```bash
"$MOOD_PY" -B "$FT/run.py" check --root "$ARTIFACTS"
"$MOOD_PY" -B "$FT/run.py" audit --root "$ARTIFACTS" --output /tmp/hot3d-source-audit.json
```

环境检查须为ready=true，两个模型都能访问PPU，官方权重路径均存在。
源数据审计只统计候选，不代表图片已经抽取完成。

## 步骤3：准备训练集和评测集

```bash
"$DATA_PY" -B "$FT/run.py" prepare --root "$ARTIFACTS"
"$DATA_PY" -B "$FT/run.py" verify --images --root "$ARTIFACTS"
```

完成后应有15,000张训练图、3,000张评测图，清单在
`$ARTIFACTS/data/hot3d-aria-18k-v1/train.jsonl`、`eval.jsonl`。
数据状态为data_ready，并且完整图片校验成功；当前阶段先处理GT候选，再读取VRS保存RGB。

训练图保存在该数据目录的rgb_pinhole下，评测清单引用
`/root/benchmarkv2/data/hot3d-aria-v1/rgb_pinhole/`中的选定图片。
GT二维框取同一份GT中的projected_bbox，无需另行生成第二套评测数据。

## 步骤4：评测官方权重，两类输入都运行

依次执行三条命令：

```bash
"$MOOD_PY" -B "$FT/run.py" evaluate --track text --model 3dmood --variant baseline --root "$ARTIFACTS"
"$WILD_PY" -B "$FT/run.py" evaluate --track text --model wilddet3d --variant baseline --root "$ARTIFACTS"
"$WILD_PY" -B "$FT/run.py" evaluate --track gt-box --model wilddet3d --variant baseline --root "$ARTIFACTS"
```

前两条对应RGB＋文本，第三条对应RGB＋GT二维框；每条都包含完整推理和评分。
完成标志：text/full/metrics下有两份官方权重指标，gt-box/full/metrics下有一份官方权重指标，
这些目录都位于 `$ARTIFACTS/evaluation/` 下。三组均成功结束后再开始微调。

## 步骤5：登录W&B并微调两个模型

使用W&B SDK所在的3D-MOOD环境登录一次，再顺序训练：

```bash
"$MOOD_PY" -m wandb login
"$MOOD_PY" -B "$FT/run.py" train --model 3dmood --root "$ARTIFACTS"
"$WILD_PY" -B "$FT/run.py" train --model wilddet3d --root "$ARTIFACTS"
```

默认每个模型训练4轮，共7,500次参数更新。训练完成应同时满足：
`runs/<模型>/status.json`为stage=complete，以及 `checkpoints/final_head.pt`存在。
last.pt只是恢复点，不代表已训练完。GT框评测直接使用本次微调权重，不新增训练任务。

## 步骤6：评测微调权重，两类输入都运行

保持与步骤4相同的配置、数据和提示规则，仅把variant切换为finetuned：

```bash
"$MOOD_PY" -B "$FT/run.py" evaluate --track text --model 3dmood --variant finetuned --root "$ARTIFACTS"
"$WILD_PY" -B "$FT/run.py" evaluate --track text --model wilddet3d --variant finetuned --root "$ARTIFACTS"
"$WILD_PY" -B "$FT/run.py" evaluate --track gt-box --model wilddet3d --variant finetuned --root "$ARTIFACTS"
```

完成标志：两种输入下，官方和微调权重共六组预测与指标全部完整。
框提示状态可看 `$ARTIFACTS/evaluation/gt-box/status.json`。

## 步骤7：统一生成最终报告

确认上述六组评分全部完成，再执行：

```bash
"$DATA_PY" -B "$FT/run.py" report --root "$ARTIFACTS"
```

打开 `$ARTIFACTS/report/report.md`，分表查看两类输入下微调前后的结果。
文本结果包含AP、分类细表、速度和训练曲线；框提示结果包含成功率、辅助AP、误差、失败、速度、
同目标改善/退步及二维框与三维框对照图。完整输出文件见下方“输出文件”。
未完成的评测会明确标为待完成，不算完整最终实验，也不会用另一类输入的分数替代。

## 一条命令运行全部步骤

完成W&B登录后执行：

```bash
nohup bash /mnt/workspace/3dmood/hot3d-finetune/run_full.sh \
  > /mnt/workspace/3dmood/hot3d-finetune/full-launch.log 2>&1 &
```

全流程通过workflow.py按与上面一致的顺序执行，根目录suite_status.json记录当前阶段、模型及输入条件；
workflow.json保存任务计划。最终stage=complete表示两类评测及报告全部完成。
已在分步运行时，不要同时启动全流程。中断后保持配置和源码，重跑对应阶段命令即可校验并恢复。

只查看计划，不启动任务：

```bash
"$MOOD_PY" -B "$FT/workflow.py" --dry-run
```

## 参数在哪里改、可以填什么

| 参数/命令 | 意义与可填内容 |
| --- | --- |
| `evaluate --track gt-box` | 单独进行框提示推理和评分 |
| `--track text` | RGB＋类别文本评测；省略track也默认为文本 |
| `--model` | 框提示只能填 `wilddet3d`；文本可填 `3dmood` 或 `wilddet3d` |
| `--variant` | `baseline`官方权重，`finetuned`官方权重＋原实验增量三维头；evaluate必填 |
| `--config` | 完整配置文件的绝对路径；省略时用configs/default.json。恢复时可使用ARTIFACTS/resolved_config.json快照 |
| `--root` | 产物根目录，默认/root/hot3d-finetune-v1；同一次实验各步骤使用同一目录 |
| `--score-only` | 用于evaluate --track gt-box；完整预测已存在时仅重评分，可用hot3d-data环境且不需要PPU |
| `--report-dir` | 用于report，填写报告输出目录；省略时为ROOT/report，可生成独立预览 |
| `--output` | 可选保存命令返回状态的JSON文件，与逐帧预测目录不同 |

基础模型路径取主配置的 `models.wilddet3d.checkpoint`、`lingbot_checkpoint` 和 `python`。
微调组固定加载ROOT下的 `runs/wilddet3d/checkpoints/final_head.pt`，并检查模型、基础权重摘要、
完成轮数和数据协议。指标阈值取 `metrics.iou_thresholds`、`metrics.center_thresholds`。
不要通过修改训练参数来切换评测模式，也不要直接修改原训练配置快照。

当前配置的成功率阈值为三维IoU≥0.25/0.50，中心距离≤0.02/0.05/0.10/0.25/0.50/1.00米。
框提示不会把0.7作为删除预测的阈值。

## 框从哪里来，GT是否泄露

框来自同批评测GT中的 `projected_bbox`：完整三维GT框在虚拟针孔图像上的投影，
经图像边界裁剪，坐标为像素xyxy。它不是重新运行二维检测器得到的框，也不是mask可见区域的紧框。
框先在原图坐标中提供，再由原生推理接口同步缩放到模型坐标。

每次模型调用仅接收RGB、内参和一个二维框；不提供该框的类别、三维中心、尺寸、旋转、GT mask或深度。
目标uid只在模型外用于恢复和评分。每个目标独立调用，避免原包装器把同图多个框误当成多张图转换坐标。

提供GT框本身明确给出了理想二维定位与目标数量，属于**已知定位条件下的三维几何评测**。
它能帮助判断已有目标的三维恢复能力，不能等同于无需定位提示的RGB＋文本完整检测任务。

## 成功率、AP和失败如何计算

主要指标：每个预测只与输入框绑定的GT目标比较，
成功率＝通过指定三维IoU或中心距离阈值的目标数／全部有效GT目标数。
无返回、无效几何等明确目标失败计入分母，不因为分数低而删除目标。

辅助指标：保留原生置信度排序，计算geometry AP3D/APcenter；AP允许帧内一对一重新匹配，
因此可能匹配到同图其他目标。目标关联错位时，辅助AP和严格成功率可能不同，应以成功率为主。
没有类别提示，不计算semantic AP。

额外输出：失败数与原因、含失败的平均IoU（失败记0）、返回框目标的中心误差中位数。
中心误差中位数只统计有框返回，不等同于已经通过几何阈值。

两份权重按frame_id＋target_id对齐，输出每个阈值下：两者通过、仅微调后通过（改善）、
仅原来通过（退步）、两者未通过，以及成功率百分点变化。

## 断点和速度

每个完成目标立即保存，再汇总为完整帧。中断后重跑相同命令，已经校验通过的目标和帧会复用，
不必重算一整张多目标图片。明确几何失败也会保留；设备错误、OOM、未知异常会停止任务，
不伪装成正常目标失败。缺失预测的评分会报错，不以“没有检测到”补齐。

若仅评分阶段中断，可在CPU环境重评分：

```bash
/mnt/workspace/venvs/hot3d-data/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py \
  evaluate --track gt-box --model wilddet3d --variant finetuned --score-only \
  --config /root/hot3d-finetune-v1/resolved_config.json
```

秒/目标包含每次目标调用的预处理、设备传输、推理和导出，包含失败尝试与首次调用；
不含图片读取、模型初始化、恢复校验与CPU评分。FPS＝图片数／全部逐目标推理耗时。
同图多个目标会重复调用，所以FPS不是一次整图前向的速度，也不是系统端到端速度。

## 输出文件

```text
/root/hot3d-finetune-v1/
├── evaluation/gt-box/
│   ├── input_manifest.json             # RGB路径、内参、纯二维框提示及摘要
│   ├── resolved_config.json            # 框提示任务配置（使用独立批量入口时保存）
│   ├── status.json / suite_status.json
│   ├── models/wilddet3d-<variant>/
│   │   ├── model.json                  # 模型/权重、框提示和源码身份
│   │   ├── objects/                    # 逐目标断点，包含明确失败的目标
│   │   └── predictions/                # 完整逐帧预测、目标关联、失败及耗时
│   └── full/metrics/
│       ├── wilddet3d-<variant>.json
│       ├── wilddet3d-<variant>_targets.csv
│       └── wilddet3d-<variant>_audit.json
└── report/
    ├── report.md                       # 两种输入下的微调前后结果
    ├── comparison.csv                  # RGB＋文本的指标
    ├── box_comparison.csv              # 成功率、失败、误差、速度
    ├── box_ap_comparison.csv           # 辅助AP
    ├── box_paired_comparison.csv        # 同目标改善/退步
    ├── results.json                    # models为文本结果，gt_box为框提示结果
    └── visualizations/gt-box/          # 两份权重各12张固定帧结果图
```

框图中青色为输入二维框，绿色为GT三维框，红色为模型预测；展示全部有效返回，
不使用0.7过滤。这些固定样例不代替全量指标，也不作为随机代表性抽样。

## 与原实验兼容

框提示实现位于 `hot3d_ft/boxeval/`，由根目录 `run.py` 分发。原训练和文本预测源码摘要范围没有改变，
框提示子模块不改变已有训练恢复点和文本缓存的身份。原 `train`、默认文本 `evaluate` 与W&B命令保持兼容。
标准全流程入口为 `run_full.sh`，由workflow.py统一调度；官方权重的两类评测均在训练前，微调后再完成两类评测，最后生成报告。

具体实施验证见 [VALIDATION.md](VALIDATION.md)。日常运行以状态文件和完整评分结果为准。
