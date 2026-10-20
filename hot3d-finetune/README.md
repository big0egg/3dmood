# HOT3D 微调：15,000 帧训练＋3,000 帧评测

本模块微调 **3D-MOOD、WildDet3D**，只运行 **RGB＋相机内参＋固定类别文本**。
从现有官方权重开始，冻结原二维检测、语言和主要特征网络，更新三维预测头。
默认训练4轮，使用预先设定的最后一轮权重评测，不用评测集挑权重。
训练 loss 实时上传到 W&B；同时保留本地逐次更新记录和报告中的静态曲线。

源码：`/mnt/workspace/3dmood/hot3d-finetune`。
默认产物：`/root/hot3d-finetune-v1`。图片、GT、预测、日志和权重不进入 Git。

## 当前完成到哪里

代码已提供数据准备、校验、训练恢复、原权重／微调权重评测和自动报告入口。
原始数据审计确认六位训练参与者有83段录制、29,561个质量标记合格的约4Hz候选。
候选还要检查三维目标、RGB解码与时间同步，**不代表18,000帧已经准备完成**。

已验证：数据隔离与来源校验、原生三维编码转换、断点恢复；两个官方权重在CPU完整构建；
3D-MOOD真实帧的CPU训练前向／反向。
正式任务在DSW原生Terminal运行；当前执行到哪里，以产物根目录的 `suite_status.json`、
`status.json` 和模型的 `runs/<模型>/status.json` 为准，不能仅凭启动日志判断完成。
本次添加W&B时，全流程仍在准备GT，尚未训练；已有进程会在启动训练子进程时读取新增的
`configs/wandb.json`，无需重启数据准备。已完成的验证见 [VALIDATION.md](VALIDATION.md)。

## 直接执行全流程

在 **DSW 原生 Terminal** 执行，可以从任意目录启动：

先按下文“W&B登录与曲线”完成一次登录，再启动。**如果全流程已在运行，不要重复启动。**

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
不要同时启动两份；不要在任务运行时修改源码或训练/数据配置。
W&B单独配置在每次训练启动时读取，可在尚未训练时设置。

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

**训练、数据、评测参数改 [configs/default.json](configs/default.json)；W&B参数改
[configs/wandb.json](configs/wandb.json)。** 两个模型共用同一份 `training` 配方。
当前实验使用15,000／3,000和4轮；改训练配方或数据规则应新建产物目录。
本版本数据模块固定15,000／3,000，不能仅改帧数就切换到其他规模。

例如学习率减半并训练6轮，将主配置的对应部分改为：

```json
"training": {
  "epochs": 6,
  "micro_batch": 1,
  "accumulate": 8,
  "lr": 0.00005,
  "weight_decay": 0.0001,
  "warmup_updates": 300,
  "clip_grad": 0.1,
  "checkpoint_updates": 250,
  "threads": 1
}
```

这是主配置中的一个字段片段，不能作为完整配置文件。建议先复制完整配置，再编辑副本：

```bash
cp /mnt/workspace/3dmood/hot3d-finetune/configs/default.json \
  /mnt/workspace/3dmood/hot3d-finetune/configs/experiment-lr5e5.json
```

编辑副本后，用 `--config /mnt/workspace/3dmood/hot3d-finetune/configs/experiment-lr5e5.json`
和新的 `--root /root/hot3d-finetune-lr5e5` 启动。
`resolved_config.json` 是程序保存的实际配置快照，**不要把它当作参数编辑入口**。
已启动的全流程不会自动读取主配置的新值；恢复旧实验必须保留原训练参数和源码。

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
| `training.epochs` | `4` | 正整数；完整遍历训练集的次数，新实验启动前设定，不依据评测分数延长 |
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
| `camera.width` / `height` | `768` / `768` | 虚拟针孔图像宽高，单位像素；更改须重新准备数据 |
| `camera.horizontal_fov_degrees` | `90` | 虚拟相机水平视场角，单位度；须在0～180之间 |
| `camera.clockwise_quarter_turns` | `1` | 顺时针转正次数，每次90°，可填0、1、2、3；应与采集相机方向一致 |

`accumulate=8` 表示8张图进行一次优化器更新；默认每轮 `ceil(15000/8)=1875` 次，
4轮7,500次。显存不足时，提高梯度累积不能降低单张前向所需显存；本版单张批量已经为1。
`warmup_updates`、`checkpoint_updates`、W&B的 `log_every_updates` 都以**参数更新次数**计，
不是图片数。最后不足8张的批次仍会更新一次。

主配置会绑定实验身份；已绑定目录内更改参数会拒绝复用。即使只改AP阈值，
本版本也应使用独立实验目录，避免将不同协议的结果混放。
`camera.*` 必须与 `existing_data/protocol.json` 中的虚拟相机一致。
若要改分辨率、视场或转正方向，须先准备同相机配置的评测数据；不能只改数值就复用现有图片。

## W&B登录与曲线

W&B使用3D-MOOD环境中的SDK独立上传两个模型的日志，不往WildDet3D环境安装W&B，
也不调整它的protobuf。上传内容为训练指标、训练配方和实验标识；不自动上传图片或checkpoint。

**1. 在DSW原生Terminal登录一次。** 按提示输入你自己的W&B API key：

```bash
/mnt/workspace/venvs/3d-mood-ppu/bin/python -m wandb login
```

不要将key填入JSON或Git。登录后，在 `configs/wandb.json` 设置项目与团队：

| 参数 | 默认值 | 含义与可填内容 |
| --- | --- | --- |
| `enabled` | `true` | `true` 开启、`false` 关闭上传；本地 `train.jsonl` 始终记录 |
| `python` | `.../3d-mood-ppu/bin/python` | 已安装W&B的Python绝对路径，独立于正在训练的模型环境 |
| `project` | `hot3d-finetune` | W&B项目名，如 `hot3d-finetune`，非空字符串 |
| `entity` | `null` | W&B账户或团队slug字符串；`null` 使用账户默认归属，不填写显示昵称 |
| `group` | `null` | 同一实验的分组名；`null` 自动使用产物根目录名 |
| `name_prefix` | `hot3d` | 运行名称前缀，实际名称含模型名和第几次启动，如 `hot3d-wilddet3d-a0001` |
| `mode` | `online` | `online` 联网上传；`offline` 只存本地；`disabled` 关闭 |
| `fallback_offline` | `true` | 登录缺失或在线初始化失败时退为本地离线记录；`false` 则上传进程报失败，模型训练仍继续 |
| `log_every_updates` | `10` | 正整数，每多少次参数更新上传一次；第1次与结束时最后一次也记录。填1可逐次上传 |
| `init_timeout` | `30` | 初始化W&B的等待秒数，正整数；在独立进程等待，不阻塞每次训练更新 |

配置文件是严格JSON：不能写注释，布尔值用小写 `true/false`，空值用 `null`。
`WANDB_MODE` 环境变量会覆盖 `mode`。若自定义主配置含 `wandb` 对象，它的同名字段
会覆盖 `configs/wandb.json`，便于不同实验使用不同项目。

**2. 训练开始后查看网页地址和上传状态。**

```bash
/mnt/workspace/venvs/3d-mood-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py wandb-status
```

也可加 `--model 3dmood` 或 `--model wilddet3d`。使用自定义实验时加相同的 `--config`、`--root`。
输出中 `mode=online` 且有 `url` 时，打开 `url` 查看实时曲线；`starting` 表示SDK正在初始化。
`mode=offline` 只在服务器保存，不会实时出现在网页。初始化或上传失败查看该次目录下的
`uploader.log`；W&B状态与训练状态分别保存，不能用上传状态判断训练是否成功。

在W&B页面以 `train/update` 为横轴：

| 曲线 | 意义 |
| --- | --- |
| `train/loss_total` | 本次更新的四项回归损失之和，按累积批次内图片取平均 |
| `train/loss_delta_2d` | 三维中心投影相对二维目标中心的偏移损失 |
| `train/loss_depth` | 目标中心深度损失 |
| `train/loss_dim` | 三维尺寸损失 |
| `train/loss_rot` | 三维旋转损失 |
| `train/epoch` | 含小数的训练轮次，例如1.5表示第二轮进行到一半 |
| `optim/lr` / `optim/grad_norm` | 学习率／裁剪前的梯度范数 |
| `runtime/seconds_per_update` | 一次更新的耗时，含前向、反向、参数更新与设备同步，不含保存checkpoint |
| `runtime/peak_allocated_gib` | 当前训练进程累计峰值张量显存，GiB；不是设备总占用 |

loss是训练目标，不是AP；不同模型原生编码和损失尺度不同，不能仅以loss大小判断哪个模型更好。
本地每次更新都记录；W&B默认每10次记录该次值，不做10次平均。网页平滑只改变展示。

**3. 断点恢复与补传。** 每次恢复训练生成新的W&B运行，同组展示，并从已经保留的
`train.jsonl` 回放历史。这样可避免突然中断后回退到checkpoint，使旧网页中尚未持久保存的更新
与新曲线混在一起。看最新一次启动的曲线；本地日志仍以恢复后的有效历史为准。

若之前未登录、网络失败或离线运行，先登录、确认 `mode=online`，在该模型训练结束或退出后执行：

```bash
/mnt/workspace/venvs/3d-mood-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py wandb-sync --model 3dmood
/mnt/workspace/venvs/3d-mood-ppu/bin/python -B /mnt/workspace/3dmood/hot3d-finetune/run.py wandb-sync --model wilddet3d
```

它从本地日志新建W&B运行补传，不重新训练、不加载模型、不占用PPU。
省略 `--model` 会依次处理两个模型，须确保两份训练记录都已存在。
如果网络仍不可用且允许离线回退，状态仍会是 `mode=offline`，这不代表已上传成功。
W&B原生同步也可使用 `wandb sync <offline-run目录>`，准确目录可从状态里的
`sdk_directory` 去掉末尾 `/files` 获取。在线训练中途掉网通常由SDK在后台重试；
最终是否可用以网页和SDK状态为准，需要时用上述命令从本地日志补传。

## 分步骤操作

所有命令都可以附加 `--config /绝对路径/config.json`、`--root /绝对路径/产物目录`。
`--root`覆盖配置的root，默认值见上表。

以下步骤用于手动分阶段执行；已经运行 `run_full.sh` 时只查看进度，不要同时运行训练或数据命令。
Shell变量只为缩短命令，先在当前Terminal设置：

```bash
FT=/mnt/workspace/3dmood/hot3d-finetune
MOOD_PY=/mnt/workspace/venvs/3d-mood-ppu/bin/python
WILD_PY=/mnt/workspace/venvs/wilddet3d-ppu/bin/python
DATA_PY=/mnt/workspace/venvs/hot3d-data/bin/python
ARTIFACTS=/root/hot3d-finetune-v1
```

**步骤1：安装独立数据环境。** 只用于VRS解码、选帧、校验与报告：

```bash
bash "$FT/setup_data_env.sh"
```

完成标志：命令退出码为0，存在 `$DATA_PY` 及该环境的 `hot3d-data-lock.txt`。
首次需要pip网络；已安装时脚本复用环境。

**步骤2：检查环境与源数据。** `check` 不训练；`audit` 不解码全部图片：

```bash
"$MOOD_PY" -B "$FT/run.py" check --root "$ARTIFACTS"
```

完成标志：顶层 `ready=true`，两个模型 `ppu=true`，数据环境 `ready=true`。
这只确认环境可访问，不等同于完整训练已经通过。

```bash
"$MOOD_PY" -B "$FT/run.py" audit --root "$ARTIFACTS" \
  --output /tmp/hot3d-source-audit.json
```

候选审计保存在 `/tmp/hot3d-source-audit.json`。候选多于15,000仍须经过后面的GT和解码筛选。

**步骤3：准备18,000帧并逐图校验。** 这是CPU与磁盘任务，可能先长时间生成GT，再出现RGB文件：

```bash
"$DATA_PY" -B "$FT/run.py" prepare --root "$ARTIFACTS"
"$DATA_PY" -B "$FT/run.py" verify --images --root "$ARTIFACTS"
```

查看进度：`cat "$ARTIFACTS/status.json"`。`prepare_gt` 是GT检查，
`prepare_rgb` 是读取并保存图片，`data_ready` 表示准备结束；随后 `verify --images`
还需成功结束。最终 `train.jsonl` 15,000行、`eval.jsonl` 3,000行，审计无交集。
中断后重新执行步骤3，按已完成录制和图片恢复；不删除 `candidate_cache/` 或 `completed_frames/`。

**步骤4：运行原权重基线。** 两个模型分别在同一3,000帧上生成完整预测和AP：

```bash
"$MOOD_PY" -B "$FT/run.py" evaluate --model 3dmood --variant baseline --root "$ARTIFACTS"
"$WILD_PY" -B "$FT/run.py" evaluate --model wilddet3d --variant baseline --root "$ARTIFACTS"
```

每个命令都需成功结束，结果位于 `evaluation/text/full/metrics/`，逐帧预测在
`evaluation/text/models/`。断点后重新执行对应命令，完整预测会校验后复用。

**步骤5：登录W&B并训练。** 先完成上节登录，再顺序运行下面两个命令。
不要同时训练；同一命令也用于恢复：

```bash
"$MOOD_PY" -B "$FT/run.py" train --model 3dmood --root "$ARTIFACTS"
"$WILD_PY" -B "$FT/run.py" train --model wilddet3d --root "$ARTIFACTS"
```

默认每模型7,500次更新。完成标志：`runs/<模型>/status.json` 为 `stage=complete`，
且 `checkpoints/final_head.pt` 存在。`last.pt` 只表示存在恢复点，不表示训练完毕。
通过 `wandb-status` 获取曲线网址；本地 `train.jsonl` 每行是一次已完成的参数更新。

**步骤6：评测微调权重。** `--variant baseline` 表示原权重，`finetuned` 表示完整训练后的增量头：

```bash
"$MOOD_PY" -B "$FT/run.py" evaluate --model 3dmood --variant finetuned --root "$ARTIFACTS"
"$WILD_PY" -B "$FT/run.py" evaluate --model wilddet3d --variant finetuned --root "$ARTIFACTS"
```

必须与基线使用相同配置和评测集，不先按0.7删除预测。两个命令成功结束后，
应有两个模型×两种权重共四组完整预测与指标。

**步骤7：生成最终报告。**

```bash
"$DATA_PY" -B "$FT/run.py" report --root "$ARTIFACTS"
```

打开 `$ARTIFACTS/report/report.md`，同时得到 `comparison.csv`、`results.json`、
训练曲线PNG及固定12帧的三维框对比图。报告会核验训练日志更新数，缺记录时拒绝生成完整报告。
若已产生W&B状态，报告还包含最新一次运行的状态和网页链接。

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
增量权重也包含数据协议和预设轮数完成标记，拒绝错误基础权重或错误数据版本。

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
│   ├── wandb/attempt-0001/    # 每次训练启动／补传独立一个目录
│   │   ├── task.json         # 上传配置和实验标识，不包含API key
│   │   ├── status.json       # 在线/离线、网页URL、已记录更新位置
│   │   ├── uploader.log      # SDK初始化、上传信息
│   │   └── wandb/            # SDK自己的在线／离线文件
│   └── checkpoints/
│       ├── last.pt            # 可恢复的完整训练状态
│       └── final_head.pt      # 预设最后一轮的三维头增量权重
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
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/venvs/3d-mood-ppu/bin/python -B -m pytest -q --disable-warnings -p no:cacheprovider hot3d-finetune/tests/test_data.py hot3d-finetune/tests/test_training.py hot3d-finetune/tests/test_evaluation.py hot3d-finetune/tests/test_wandb_log.py
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/venvs/3d-mood-ppu/bin/python -B -m pytest -q --disable-warnings -p no:cacheprovider hot3d-finetune/tests/test_native_coders.py -k 3dmood
PYTHONDONTWRITEBYTECODE=1 /mnt/workspace/venvs/wilddet3d-ppu/bin/python -B -m pytest -q --disable-warnings -p no:cacheprovider hot3d-finetune/tests/test_native_coders.py -k wilddet3d
```

参考：[HOT3D数据格式与质量标记](https://github.com/facebookresearch/hot3d)、
[Aria VRS读取API](https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/core_code_snippets/data_provider)、
[3D-MOOD](https://github.com/cvg/3D-MOOD)、[WildDet3D训练代码](https://github.com/allenai/WildDet3D/blob/main/docs/TRAINING.md)。
W&B：[自定义曲线横轴](https://docs.wandb.ai/models/track/log/customize-logging-axes)、
[离线记录与同步](https://docs.wandb.ai/models/track/offline)。
