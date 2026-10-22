# HOT3D微调与两类输入评测：操作说明

本模块训练3D-MOOD和WildDet3D的三维几何头，并比较官方权重与微调权重。
完整数据、训练参数与W&B设置见 [README.md](README.md)。

## 两类评测分别做什么

| 评测输入 | 参加模型 | 主要任务 | 主要结果 |
| --- | --- | --- | --- |
| RGB＋固定类别文本＋相机内参 | 3D-MOOD、WildDet3D | 自行定位物体并估计三维框 | geometry/semantic AP3D、APcenter |
| RGB＋GT二维框＋相机内参 | WildDet3D | 给定目标图像位置后估计三维框 | 逐目标成功率，辅助geometry AP |

3D-MOOD训练函数里的 `boxes2d` 是监督参数，当前原生推理路径不消费框提示，所以不参加第二类。
WildDet3D使用原生 `input_boxes` 和 `prompt_text="geometric"`；`geometric`是固定模式标记，不包含目标类别。

当前两类评测使用的是**同一套在文本流程下微调好的权重**。增加框提示评测不重新训练，
结果表示这次几何微调向框提示任务的迁移，不应写成“完成了框提示专项微调”。

## 已完成文本微调，接下来执行什么

默认已有：

- 训练/评测清单：`/root/hot3d-finetune-v1/data/hot3d-aria-18k-v1/`。
- WildDet3D微调权重：`/root/hot3d-finetune-v1/runs/wilddet3d/checkpoints/final_head.pt`。
- 原文本报告：`/root/hot3d-finetune-v1/report/report.md`。

保持这些目录和原配置，**在能访问PPU的DSW原生Terminal**运行：

```bash
nohup /mnt/workspace/venvs/wilddet3d-ppu/bin/python -B \
  /mnt/workspace/3dmood/hot3d-finetune/run.py box-supplement \
  --config /root/hot3d-finetune-v1/resolved_config.json \
  > /mnt/workspace/3dmood/hot3d-finetune/box-launch.log 2>&1 &
```

命令顺序完成：官方权重框提示推理与评分 → 微调权重框提示推理与评分 → 合并原报告。
不会重新抽帧、训练或推理原文本实验。默认数据是3,000张图片、14,373个有效目标。

查看日志与状态：

```bash
tail -n 60 -f /mnt/workspace/3dmood/hot3d-finetune/box-launch.log
```

```bash
cat /root/hot3d-finetune-v1/evaluation/gt-box/status.json
cat /root/hot3d-finetune-v1/evaluation/gt-box/suite_status.json
```

完成标志：补充流程的 `suite_status.json` 为 `stage=complete`、`complete=true`，
返回状态的 `box_evaluation_complete=true`。原根目录的 `suite_status.json` 是旧流程记录，
不能用它判断新增框提示评测是否结束。

报告内分两部分展示输入条件；两组完整评分尚未完成时，会明确显示待完成，不补零或借用文本分数。

## 分步运行

第一步，官方权重：

```bash
/mnt/workspace/venvs/wilddet3d-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py \
  evaluate --track gt-box --model wilddet3d --variant baseline \
  --config /root/hot3d-finetune-v1/resolved_config.json
```

第二步，微调权重：

```bash
/mnt/workspace/venvs/wilddet3d-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py \
  evaluate --track gt-box --model wilddet3d --variant finetuned \
  --config /root/hot3d-finetune-v1/resolved_config.json
```

两条命令都包含推理和评分，应顺序执行，不与训练或其他PPU评测并发。
模型、数据或代码发生变化会拒绝混用旧断点。

第三步，合并原报告：

```bash
/mnt/workspace/venvs/hot3d-data/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py \
  report --config /root/hot3d-finetune-v1/resolved_config.json
```

`report`不启动模型、不训练。已有文本报告时保留其内容，更新框提示补充章节；
多次运行不会重复追加同一章节。若原文本报告不存在，先按原流程从完整文本结果生成它。

## 参数在哪里改、可以填什么

| 参数/命令 | 意义与可填内容 |
| --- | --- |
| `box-supplement` | 自动补跑两份权重并合并报告，固定使用WildDet3D |
| `evaluate --track gt-box` | 单独进行框提示推理和评分 |
| `--track text` | 原有文本评测；省略track也默认为文本 |
| `--model` | 框提示只能填 `wilddet3d`；文本可填 `3dmood` 或 `wilddet3d` |
| `--variant` | `baseline`官方权重，`finetuned`官方权重＋原实验增量三维头；evaluate必填 |
| `--config` | 原实验配置文件绝对路径，建议使用上面的resolved_config.json |
| `--root` | 覆盖产物根目录；补跑当前实验必须沿用 `/root/hot3d-finetune-v1` |
| `--score-only` | 用于evaluate或box-supplement；完整预测已存在时仅重评分，可使用hot3d-data环境且不需要PPU |
| `--report-dir` | 用于report或box-supplement，填写报告输出目录；省略时为ROOT/report，可生成独立预览 |
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
│   ├── resolved_config.json            # 补跑配置；不覆盖原训练配置快照
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
    ├── report.md                       # 原文本内容＋框提示补充章节
    ├── comparison.csv                  # 原文本结果保持不变
    ├── box_comparison.csv              # 成功率、失败、误差、速度
    ├── box_ap_comparison.csv           # 辅助AP
    ├── box_paired_comparison.csv        # 同目标改善/退步
    ├── results.json                    # 保留原models，新增gt_box
    └── visualizations/gt-box/          # 两份权重各12张固定帧结果图
```

框图中青色为输入二维框，绿色为GT三维框，红色为模型预测；展示全部有效返回，
不使用0.7过滤。这些固定样例不代替全量指标，也不作为随机代表性抽样。

## 与原实验兼容

新增实现位于 `hot3d_ft/boxeval/`，由根目录 `run.py` 分发。原训练和文本预测源码摘要范围没有改变，
新增子模块不使既有训练恢复点和文本缓存失效。原 `train`、默认文本 `evaluate` 与W&B命令保持兼容。
从零运行 `run_full.sh` 时，在原文本流程之后会自动补跑框提示并合并报告。

本轮工具环境不能访问PPU，也不能写入 `/root`。自动测试与独立报告预览不等于全量PPU实验已经完成；
请在原生Terminal执行顶部补跑命令，完整结果会写入指定的原报告。
