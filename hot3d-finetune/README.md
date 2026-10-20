# HOT3D 微调：15,000 帧训练＋3,000 帧评测

本模块微调 **3D-MOOD、WildDet3D**，只运行 **RGB＋相机内参＋固定类别文本**。
从现有官方权重开始，冻结原二维检测、语言和主要特征网络，更新三维预测头。
正式训练固定4轮，使用第四轮权重评测，不用评测集挑权重。

源码：`/mnt/workspace/3dmood/hot3d-finetune`。
默认产物：`/root/hot3d-finetune-v1`。图片、GT、预测、日志和权重不进入 Git。

## 当前完成到哪里

代码已提供数据准备、校验、训练恢复、原权重／微调权重评测和自动报告入口。
原始数据审计确认六位训练参与者有83段录制、29,561个质量标记合格的约4Hz候选。
候选还要检查三维目标、RGB解码与时间同步，**不代表18,000帧已经准备完成**。

已验证：数据隔离与来源校验、原生三维编码转换、断点恢复；两个官方权重在CPU完整构建；
3D-MOOD真实帧的CPU训练前向／反向。
当前工具会话不能访问PPU，缺少VRS读取库且网络安装受限，因此VRS解码、18,000帧正式准备、
PPU反向传播和正式AP须在DSW原生Terminal执行。详见 [VALIDATION.md](VALIDATION.md)。

## 直接执行全流程

在 **DSW 原生 Terminal** 执行，可以从任意目录启动：

```bash
nohup bash /mnt/workspace/3dmood/hot3d-finetune/run_full.sh \
  > /mnt/workspace/3dmood/hot3d-finetune/full-launch.log 2>&1 &
```

流程依次执行：

1. 建立独立的数据环境 `hot3d-data`，安装VRS读取和数据处理依赖；不修改两个模型环境。
2. 检查两个模型环境的PPU可见性、基础权重和数据环境。
3. 准备并校验精确15,000帧训练集、3,000帧评测集。
4. 3D-MOOD原权重评测 → 微调4轮 → 微调权重评测。
5. WildDet3D原权重评测 → 微调4轮 → 微调权重评测。
6. 生成最终报告、CSV、JSON、固定帧三维框对比图。

查看日志：

```bash
tail -n 60 -f /mnt/workspace/3dmood/hot3d-finetune/full-launch.log
```

中断后重新执行同一条命令。程序会验证数据、源码和配置，再复用完整图片、预测以及训练恢复点。
不要同时启动两份；也不要在任务运行时修改源码或配置。

首次依赖安装需要能访问pip源。使用系统的pip源配置；如镜像缺少projectaria-tools，
可在启动前设置自己可用的 `PIP_INDEX_URL`。安装结果记录在
`/mnt/workspace/venvs/hot3d-data/hot3d-data-lock.txt`。

## 数据怎么选

| 集合 | 参与者 | 帧数 | 来源 |
| --- | --- | ---: | --- |
| 训练 | P0001、P0002、P0003、P0009、P0010、P0011 | 15,000 | 原始Aria VRS＋CSV GT |
| 评测 | P0014 | 1,610 | 原benchmark针孔RGB与GT |
| 评测 | P0015 | 1,390 | 原benchmark针孔RGB与GT |

P0012不计入本次18,000帧。训练与评测的参与者、录制、帧ID和图像SHA256不重叠。

训练候选须同时满足人工QA、曝光、相机位姿、物体位姿和物体可见性标记。
每段录制按至少0.25秒间隔生成候选，再检查虚拟视野中的有效GT。
优先覆盖各录制、25类和33个物体，之后按录制轮流补足，避免长视频占满配额。
此版本不宣称已经基于姿态或遮挡特征完成最优去重；时间间隔和录制均衡是主要控制方式。

原始CSV的时间戳为TIME_CODE；使用官方映射表找到DEVICE_TIME后读取VRS RGB，
图像捕获时间误差超过1ms即停止。GT位姿按完全相同的TIME_CODE关联，不按帧号猜测。

图像使用768×768、90°水平视场、顺时针转正90°的针孔相机。
坐标为OpenCV相机坐标：右、下、前；长度为米；保留完整三维旋转。
训练时缩放和中心填充通过内参推导二维监督的同步变换。

新增训练帧的有效性使用原生可见性及实际GLB三角形在虚拟相机中的amodal覆盖像素。
达到64像素后提前停止栅格化，记录的是**面积下界**，字段
`virtual_amodal_pixels_is_lower_bound=true`；未达到阈值时完成全部三角形检查。
评测帧保持旧GT和ignore标记，避免因新增数据准备方式改变评测标注。

候选不足、类别/物体覆盖不完整、GT或RGB摘要变化时明确停止，不重复图片或跳过异常凑数。
训练数据只覆盖33个已建模物体；背景没有穷尽标注。

## 配置参数

默认配置：[configs/default.json](configs/default.json)。修改配置应使用新的产物根目录。
本实验固定15,000／3,000和4轮；更改规模、参与者、相机或轮数属于新实验。

| 参数 | 默认值 | 意义与可填内容 |
| --- | --- | --- |
| `root` / `--root` | `/root/hot3d-finetune-v1` | 产物根目录，填写有足够容量、可写的绝对路径 |
| `source` | `/mnt/data/opendatasets/HOT3D` | 原始数据目录，内含 `Aria/<sequence>/*main_recording.vrs` 和 `*ground_truth.zip` |
| `existing_data` | `/root/benchmarkv2/data/hot3d-aria-v1` | 已准备的旧数据，包含protocol、frames、GT、RGB和object_models |
| `dataset_name` | `hot3d-aria-18k-v1` | 新数据版本名；不得复用不同内容的旧版本目录 |
| `seed` | `20261020` | 选帧、训练打乱及增强的整数种子 |
| `train_participants` | 六位训练参与者 | 参与者ID列表；不能与评测参与者重叠 |
| `eval_quotas` | P0014=1610、P0015=1390 | 每位评测参与者的精确帧数 |
| `train_count` / `eval_count` | `15000` / `3000` | 正整数，正式实验固定这两个值 |
| `candidate_interval_ns` | `250000000` | 同一录制候选最小时间间隔，单位纳秒；本版约4Hz |
| `minimum_amodal_pixels` | `64` | 物体在虚拟视野至少覆盖的像素数，正整数 |
| `minimum_native_modeled_visibility` | `0.1` | 原生GT可见比例，0～1；CSV列名虽含%，值实际是比例 |
| `data_python` | `.../hot3d-data/bin/python` | 安装projectaria-tools、numpy、scipy、OpenCV、Pillow、matplotlib的数据环境Python |
| `models.<模型>.python` | 各自现有venv的Python | 模型专用Python；模型名只能为 `3dmood` 或 `wilddet3d` |
| `models.<模型>.project_root` | 模型项目路径 | 已适配源码的绝对路径 |
| `models.<模型>.checkpoint` | 当前官方权重 | 本地文件绝对路径；不自动联网下载 |
| `lingbot_checkpoint` | WildDet3D的 `model.pt` | 仅WildDet3D需要，用于本地模型结构构建 |
| `training.micro_batch` | `1` | 每次前向的图片数，首版仅支持1 |
| `training.accumulate` | `8` | 累积多少张图片更新一次，正整数；默认有效批量8 |
| `training.epochs` | `4` | 正式实验固定4轮，不依据评测分数延长 |
| `training.lr` | `0.0001` | 三维头学习率，正数 |
| `training.weight_decay` | `0.0001` | AdamW权重衰减，非负数 |
| `training.warmup_updates` | `300` | 学习率预热的更新次数；之后余弦下降到初值的10% |
| `training.clip_grad` | `0.1` | 梯度范数裁剪上限，正数 |
| `training.checkpoint_updates` | `250` | 每多少次更新持久保存恢复点，正整数 |
| `training.threads` | `1` | CPU线程数，正整数；保留已验证的单线程模型初始化路径 |
| `device` | `cuda` | 当前PPU由Torch CUDA兼容接口访问；正式任务不回退CPU |
| `max_detections` | `300` | 每张图保留的原生预测上限；训练前后相同 |
| `display_score` | `0.7` | 展示框的分数阈值，不用它提前删除AP计算的预测 |
| `metrics.iou_thresholds` | `[0.25,0.5]` | 三维IoU匹配阈值列表 |
| `metrics.center_thresholds` | `[0.02,0.05,0.1,0.25,0.5,1.0]` | 三维中心距离阈值列表，单位米 |

## 分步骤操作

所有命令都可以附加 `--config /绝对路径/config.json`、`--root /绝对路径/产物目录`。
`--root`覆盖配置的root，默认值见上表。

先安装数据环境：

```bash
bash /mnt/workspace/3dmood/hot3d-finetune/setup_data_env.sh
```

只读检查两个模型的PPU、权重和数据环境：

```bash
/mnt/workspace/venvs/3d-mood-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py check
```

统计源录制及质量标记候选，不解码全部图片：

```bash
/mnt/workspace/venvs/3d-mood-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py audit \
  --output /tmp/hot3d-source-audit.json
```

准备数据和完整图片核验：

```bash
/mnt/workspace/venvs/hot3d-data/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py prepare
/mnt/workspace/venvs/hot3d-data/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py verify --images
```

单独训练；同一命令也用于恢复：

```bash
/mnt/workspace/venvs/3d-mood-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py train --model 3dmood
/mnt/workspace/venvs/wilddet3d-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py train --model wilddet3d
```

`evaluate`必须指定 `--model`（`3dmood` / `wilddet3d`）和
`--variant`（`baseline` 原始权重 / `finetuned` 第四轮权重）。例如：

```bash
/mnt/workspace/venvs/3d-mood-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py evaluate --model 3dmood --variant baseline
/mnt/workspace/venvs/3d-mood-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py evaluate --model 3dmood --variant finetuned
/mnt/workspace/venvs/wilddet3d-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py evaluate --model wilddet3d --variant baseline
/mnt/workspace/venvs/wilddet3d-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py evaluate --model wilddet3d --variant finetuned
/mnt/workspace/venvs/hot3d-data/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py report
```

## 训练、恢复与评测约定

训练监督只使用原生目标匹配之后最终层的三维中心偏移、目标深度、尺寸和旋转回归，
四项权重均为1；暂不加入原训练的中间层、O2M和编码器辅助损失。
这是第一阶段的几何适配实验，不是完整复现原论文训练配方。
没有逐像素实测深度，不构造假深度GT；冻结的几何后端维持推理模式。
未匹配预测不产生分类、presence或三维置信度负样本训练损失。

15,000张、有效批量8、4轮共约7,500次更新。
`last.pt`包含三维头、优化器、随机状态、已完成轮数和下一张图片位置。
正常异常/终止在最近完成的参数更新边界保存；突然断电或强制SIGKILL最多回退到最近持久恢复点。
恢复时丢弃尚未完成的梯度累积，并重算这些图片；训练日志同步回退，不重复计数。
CPU测试已验证恢复后与连续训练的权重一致；PPU上的浮点数值一致性尚未验证。

`final_head.pt`是**增量权重**，需要配合固定SHA256的原权重使用，不能直接传给原项目的通用checkpoint参数。
本模块的 `evaluate --variant finetuned` 会加载原权重，再严格加载增量权重。
增量权重也包含数据协议和第四轮完成标记，拒绝错误基础权重或错误数据版本。

每张评测图固定给完整25类文本词表；不提供GT目标数、该帧类别、GT位置、物体CAD、GT mask或深度。
二维投影框仅在训练时作为模型原生目标匹配的监督，不是评测提示。
原始与微调权重使用同一相机内参路径、预处理、原生NMS和预测上限。
原权重评测在这3,000帧重新执行，不把旧15,160帧总分拿来直接比较。

AP3D：完整旋转三维框IoU匹配。APcenter：三维中心欧氏距离匹配。
geometry忽略类别；semantic要求类别一致并按类平均，两者分别报告。
模型结构、底座预训练和原生候选规则不同；本实验比较相同输入下各模型的微调收益。

## 产物在哪里

```text
/root/hot3d-finetune-v1/
├── resolved_config.json
├── suite_status.json
├── data/hot3d-aria-18k-v1/
│   ├── preparation.json       # 数据来源、配置与源码摘要
│   ├── candidate_cache/       # 完整GT候选缓存，按录制恢复
│   ├── completed_frames/      # 每张已完成图片的校验记录
│   ├── train.jsonl            # 15,000行
│   ├── eval.jsonl             # 3,000行
│   ├── protocol.json
│   ├── dataset_audit.json
│   ├── rgb_pinhole/           # 新训练图片；评测清单引用旧图片
│   ├── groundtruth/
│   └── calibration/
├── runs/<3dmood或wilddet3d>/
│   ├── training.json          # 配方和环境来源
│   ├── parameters.json        # 可训练参数清单
│   ├── train.jsonl            # 损失、学习率、梯度、时间和显存
│   ├── status.json
│   └── checkpoints/
│       ├── last.pt            # 可恢复的完整训练状态
│       └── final_head.pt      # 第四轮三维头增量权重
├── evaluation/text/
│   ├── models/<模型-权重>/predictions/
│   └── full/metrics/          # AP、分类细表、工作点和速度
└── report/
    ├── report.md
    ├── comparison.csv
    ├── results.json
    ├── 3dmood-training.png
    ├── wilddet3d-training.png
    └── visualizations/
```

产物根目录建议预留至少40GiB，并与另一个终端的仿真数据生成协调容量。
两个模型顺序占用PPU；代码使用 `/mnt/workspace/.hot3d-ppu.lock` 协作锁，
只有同样遵守这把锁的其他任务才会互斥，不会强行终止其他进程。

## 检查代码

```bash
cd /mnt/workspace/3dmood
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/venvs/3d-mood-ppu/bin/python -B -m pytest -q --disable-warnings -p no:cacheprovider hot3d-finetune/tests/test_data.py hot3d-finetune/tests/test_training.py hot3d-finetune/tests/test_evaluation.py
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/venvs/3d-mood-ppu/bin/python -B -m pytest -q --disable-warnings -p no:cacheprovider hot3d-finetune/tests/test_native_coders.py -k 3dmood
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/venvs/wilddet3d-ppu/bin/python -B -m pytest -q --disable-warnings -p no:cacheprovider hot3d-finetune/tests/test_native_coders.py -k wilddet3d
```

参考：[HOT3D数据格式与质量标记](https://github.com/facebookresearch/hot3d)、
[Aria VRS读取API](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/data_provider)、
[3D-MOOD](https://github.com/cvg/3D-MOOD)、[WildDet3D训练代码](https://github.com/allenai/WildDet3D/blob/main/docs/TRAINING.md)。
