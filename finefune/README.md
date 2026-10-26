# finefune：多数据集三维几何头微调

代码目录已由 hot3d-finetune 改为 `/mnt/workspace/3dmood/finefune`。支持 3D-MOOD、WildDet3D，以及已准备的 HOT3D、LIBERO 仿真和 Cosmos 生成数据。加载完整基础权重，只更新三维几何头；置信度分支及其他部分保持冻结。

finefune 负责数据接入、训练和实验调度；评测调用仓库中的 ioutest。当前实现为单设备 FP32、micro batch 1、梯度累积、AdamW、预热及余弦学习率。全参数微调、LoRA、混合精度和多卡训练不在当前支持范围，不会仅凭配置项启用。

按步骤操作见 [introduce.md](introduce.md)，验证范围见 [VALIDATION.md](VALIDATION.md)。

## 环境和入口

模型继续使用 `configs/models/*.yaml` 中各自的 PPU Python。WildDet3D 使用 `/root/wilddet3d-weights/` 下的基础和 LingBot 权重。3D-MOOD 的 `environment.HF_HOME` 固定到 `/mnt/workspace/.cache/3d-mood/huggingface`，检查、训练、评测均显式传入。控制程序可使用基础 Python 或独立数据环境；train 会自动切换到所选模型环境，不会自动安装或升级模型依赖。

需要独立数据环境时手动执行：

```bash
cd /mnt/workspace/3dmood/finefune
bash setup_data_env.sh
```

默认安装数据读取依赖到 `/mnt/workspace/venvs/finefune-data`，可以通过 FINEFUNE_DATA_ENV 覆盖。不在模型环境安装普通 PyPI torch。已处理数据读取不需要重新安装 VRS 解码器或 MuJoCo。

```bash
FT=/mnt/workspace/3dmood/finefune
PY=/usr/local/bin/python3

# 只读读取实际数据和配置，不检查 PPU、不写产物
"$PY" -B "$FT/run.py" train --config "$FT/configs/experiments/3dmood_hot3d.yaml" --dry-run

# 检查所选模型环境、基础权重路径和实际设备
"$PY" -B "$FT/run.py" check --config "$FT/configs/experiments/3dmood_hot3d.yaml"

# 实际训练
"$PY" -B "$FT/run.py" train --config "$FT/configs/experiments/3dmood_hot3d.yaml"
```

训练 dry-run 会读取并审计数据，可能耗时；suite dry-run 只解析配置和展示计划。二者都不能代替真实模型前后向验证。真实的一次更新检查入口是 `scripts/check_native_training.py`，默认只列计划，显式加 `--run` 才执行，详见 [VALIDATION.md](VALIDATION.md)。

## 命令与参数

| 命令 | 作用 |
|---|---|
| check | 只读检查数据、模型文件和实际运行环境 |
| prepare | 创建统一索引、规范化 GT，不复制原图、不重新抽帧或生成数据 |
| verify | 检查已建立的索引；--images 进一步解码图片 |
| train | 微调、显式恢复、在参数更新边界停止 |
| export-eval | 导出 ioutest 可单独执行的评测配置 |
| evaluate | 调用 ioutest 评测原始或微调权重 |
| suite | 多模型、多训练数据的基线→训练→微调后评测→汇总 |
| report | 汇总已有实验，缺失或损坏阶段列为待完成 |
| convert-config | 显式转换旧 HOT3D JSON，不改旧数据或权重 |

| 参数 | 含义 |
|---|---|
| --config | 实验 YAML/JSON，或保存的 resolved_config.yaml |
| --model | 3dmood、wilddet3d |
| --model-config | 模型解释器、工程、基础及辅助权重 |
| --origin-ckpt | 覆盖完整基础 checkpoint，不能填三维头增量 |
| --dataset-config | 数据格式、根目录、类别、划分和对齐规则 |
| --dataset-root | 覆盖数据根目录，保留显式外部类别表等引用 |
| --train-config | 完整训练配方 |
| --epochs、--lr、--grad-accum-steps | 覆盖轮数、学习率、梯度累积数 |
| --output-dir | 独立实验输出目录，不能与输入互相包含 |
| --resume [auto或路径] | 恢复本实验；auto 无恢复点时开始新训练 |
| --device、--seed | 显式设备和随机种子，不自动回退 CPU |
| --stop-after-updates N | 第 N 次更新后保存并停止，状态 stopped，不产生最终权重 |
| --dry-run | 只读展示配置或任务计划 |
| --variant | evaluate/export-eval 使用 official 或 finetuned |
| --finetuned-ckpt | 覆盖本实验评测用三维头，仍核验基础权重和训练来源 |
| --tests-config、--datasets | 从 ioutest 配置选择独立测试集 |
| --tracks | text；WildDet3D 还支持 gt-box |
| --stage | ioutest 的 check/infer/score/latency/report/all |
| suite --models/--datasets | 选择组合中的模型或训练数据 ID |
| report --suite-dir | 指定组合产物目录 |

参数化训练：

```bash
FT=/mnt/workspace/3dmood/finefune
/usr/local/bin/python3 -B "$FT/run.py" train \
  --model 3dmood \
  --origin-ckpt /mnt/workspace/.cache/3d-mood/torch/hub/checkpoints/gdino3d_swin-b_120e_omni3d_834c97.pt \
  --dataset-config "$FT/configs/datasets/hot3d.yaml" \
  --train-config "$FT/configs/training/geometry_head.yaml" \
  --epochs 4 --lr 0.0001 --grad-accum-steps 8 \
  --output-dir /root/finefune-runs/3dmood-hot3d-exp01
```

## 分层配置

```text
configs/
├── models/{3dmood,wilddet3d}.yaml
├── datasets/{hot3d,libero,libero_matched,cosmos_generic,cosmos_mixed}.yaml
├── training/geometry_head.yaml
├── experiments/{3dmood,wilddet3d}_hot3d.yaml
└── suites/model_data_compare.yaml
```

训练参数优先级：默认值→训练配方→数据配置的 training_epochs（仅轮数）→实验配置的 training 覆盖→显式命令行参数。HOT3D 的 training_epochs 为 4，LIBERO/Cosmos 为 12，单次训练和 suite 均自动选择；未设置该字段的数据使用训练配方轮数。--epochs 始终可手动覆盖。

保存的 resolved_config.yaml 自包含，恢复时保持快照中的轮数，不受新的数据默认值影响。显式选择新的 train-config 会重新加载配方并应用数据轮数默认值，单项命令行参数最后生效。更改已有实验的训练轮数需使用新输出目录。

配置引用的其他配置、模型与输出路径相对于该配置所在目录解析。数据配置中的 manifest、protocol、categories_file、align_to 相对于 dataset.root；绝对路径保持原义。命令行路径相对于当前工作目录。解释器保留 venv 路径，不解析成系统 Python。

```yaml
finetune_mode: geometry_head
precision: fp32
epochs: 4
micro_batch_size: 1
grad_accum_steps: 8
optimizer: adamw
lr: 0.0001
weight_decay: 0.0001
scheduler: cosine
warmup_updates: 300
max_grad_norm: 0.1
checkpoint_every_updates: 250
threads: 1
seed: 20261020
```

解析后的内部字段为 micro_batch/accumulate/clip_grad/checkpoint_updates。有效批量为 micro_batch_size × grad_accum_steps；每轮最后不足一个累积组时按实际组大小归一化。更新次数为 epochs × ceil(实际有效训练图片数 / grad_accum_steps)。预热从 0.1 倍学习率开始，余弦段降至 0.1 倍。

上面的配方以 4 轮作为回退值；当前 LIBERO/Cosmos 数据配置会将轮数改为 12。4,910 帧按 12 轮计算为 58,920 次图片训练量，约等于 HOT3D 15,000 帧 × 4 轮的 60,000 次。

## 数据接入与划分

| 配置 | 输入 | 隔离字段 |
|---|---|---|
| hot3d.yaml | /root/hot3d-finetune-v1/data/hot3d-aria-18k-v1 | participant_id |
| libero.yaml | /root/opendatasets/LIBERO_GT_v1，完整 agentview 划分 | scene_id |
| libero_matched.yaml | 同上，训练状态对齐 Cosmos train_mixed | scene_id |
| cosmos_generic.yaml | /root/robot-realism/agentview_prompt_v3，train_all_general 与 eval_general | scene_id |
| cosmos_mixed.yaml | 同上，train_mixed 与 eval_general | scene_id |

复用 ioutest 的坐标、类别和父子实例过滤规则。默认只训练顶层 object，跳过机器人、支撑物和子实例；低可见目标设为 ignore。无有效目标的训练图片排除并记录，评测图片保留。不再要求固定 25 类/33 个物体。

数据必须提供米制 OpenCV 相机坐标、有效内参和完整旋转框。训练/评测检查帧 ID、仿真 state_id、RGB 摘要，以及参与者/序列/场景隔离字段。同一仿真状态的原图、生成图、不同视角不能跨训练和评测。

LIBERO/Cosmos 比较模板使用共同可用训练状态。生成图按成功清单读取，拦截和缺失项不补占位图；继承 GT 的来源及待视觉审核状态保留到索引和评测报告。

可单独建立可复用索引：

```bash
FT=/mnt/workspace/3dmood/finefune
/usr/local/bin/python3 -B "$FT/run.py" prepare \
  --dataset-config "$FT/configs/datasets/cosmos_mixed.yaml" \
  --output-dir /root/finefune-data/cosmos-v3

/usr/local/bin/python3 -B "$FT/run.py" verify \
  --output-dir /root/finefune-data/cosmos-v3 --images
```

输出 dataset_config.yaml 可继续传给 train 的 dataset-config；原图保持外部引用。train 也会在自己的实验目录建立独立索引。抽帧、仿真采集和图像生成由原数据制作模块负责。

## 权重、断点与产物

```text
实验目录/
├── resolved_config.yaml
├── config_identity.json
├── model_identity.json
├── dataset_reference.json
├── dataset_config.yaml
├── dataset/                    统一清单、GT、协议和审计
├── train.jsonl                 给评测模块使用的训练来源清单
├── training.json
├── parameters.json
├── status.json
├── checkpoints/
│   ├── last.pt
│   └── final_head.pt
├── logs/
│   ├── train.jsonl             每行一次完整参数更新
│   └── wandb/
└── evaluation/
    ├── official_config.json
    ├── finetuned_config.json
    ├── official/
    └── finetuned/
```

原始 checkpoint 用于初始化。last.pt 保存三维头、优化器、更新位置、学习率状态和随机状态；final_head.pt 保存训练后的三维头，推理时加载原模型后覆盖对应参数，不能当成完整模型使用。

```bash
/usr/local/bin/python3 -B /mnt/workspace/3dmood/finefune/run.py train \
  --config /root/finefune-runs/3dmood-hot3d-exp01/resolved_config.yaml \
  --resume auto
```

已有恢复点时必须显式 resume。数据、权重、配方或训练代码改变需新建实验目录，错误恢复参数不会覆盖原快照。源码摘要使用稳定逻辑文件名；仅移动外层代码目录不改变训练参数身份。只在完整优化器更新边界保存，累积中断会重放上一完整边界后的工作。

W&B 保留独立 SDK 上传进程及离线回退。密钥通过现有登录机制提供，不写入配置。实验配置可设置 wandb.enabled: false、mode: offline 等。本地 train.jsonl 与网页上传状态分开。

## 评测与组合实验

```bash
FT=/mnt/workspace/3dmood/finefune
/usr/local/bin/python3 -B "$FT/run.py" evaluate \
  --config /root/finefune-runs/3dmood-hot3d-exp01/resolved_config.yaml \
  --variant finetuned --tracks text

/usr/local/bin/python3 -B "$FT/run.py" evaluate \
  --config /root/finefune-runs/3dmood-hot3d-exp01/resolved_config.yaml \
  --tests-config /mnt/workspace/3dmood/ioutest/configs/robot_boxing_v1.json \
  --datasets hot3d_eval libero_original_eval libero_realistic_eval \
  --variant finetuned --tracks text
```

指标、计时、评测执行参数和缓存设置继承 `--tests-config`（未填时用 ioutest 默认配置）；显式 `--device`／`--seed` 优先。模型及权重来自本次训练配置。默认使用本实验冻结的 eval；可显式跨数据集评测，训练身份和测试身份分开记录。每个测试集使用固定词表，原始与微调模型遵守相同输入条件。WildDet3D 支持 text/gt-box，3D-MOOD 只支持 text；已知二维框条件不与自主检测混合排名。

组合模板为两模型×HOT3D／对齐原始仿真／全通用生成／混合生成，共八组独立微调，另有每模型一次官方基线；统一使用三个固定测试集。当前三套仿真相关训练清单均为相同的 4,910 个状态；生成测试集为 499 帧，原始仿真测试集为 500 帧。

```bash
FT=/mnt/workspace/3dmood/finefune
/usr/local/bin/python3 -B "$FT/run.py" suite \
  --config "$FT/configs/suites/model_data_compare.yaml" --dry-run

# 审核权重、环境、数据与输出路径后真正启动
/usr/local/bin/python3 -B "$FT/run.py" suite \
  --config "$FT/configs/suites/model_data_compare.yaml"
```

可加 --models 3dmood、--datasets hot3d 缩小组合。顺序是准备→全部训练/测试来源审计→每个模型一次基线→各组训练→微调后评测→报告。各组从自己的原始权重开始，PPU 任务顺序运行。重复原 suite 命令会核验缓存并恢复已完成阶段。

组合产物在 output_dir/baselines、experiments/<模型--数据>、reports；进度是 suite_status.json，计划是 suite_plan.json。汇总复核最终权重、评测任务集合、指标摘要及 CSV 一致性，缺失阶段明确列为待完成。

```bash
/usr/local/bin/python3 -B /mnt/workspace/3dmood/finefune/run.py report \
  --suite-dir /root/finefune-runs/model-data-compare-v1
```
