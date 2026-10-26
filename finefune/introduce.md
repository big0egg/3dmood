# finefune 使用说明

**一次微调不需要把本文所有命令都执行一遍。** 根据目的选择：

| 目的 | 阅读哪部分 |
|---|---|
| 一个模型在一份数据上微调 | 路线 A：单次微调 |
| 两模型 × 四训练数据的完整对照 | 路线 B：全量对照实验 |
| 训练中断后继续 | 中断续训 |
| 比较官方与微调效果 | 单次训练后的评测 |
| 提前排查配置、环境或数据 | 可选检查 |

路线 A 和路线 B 是两种用法，不需要依次执行。`train` 自动建立索引并训练；单次训练不会自动评测。`suite` 调度准备、基线、训练、评测与汇总全流程。

## 公共环境：每次打开终端先设置

```bash
FT=/mnt/workspace/3dmood/finefune
PY=/usr/local/bin/python3
```

控制 Python 需要 NumPy、SciPy、Pillow、PyYAML。已有依赖可直接使用；缺依赖时才执行 `bash "$FT/setup_data_env.sh"`，然后设置 `PY=/mnt/workspace/venvs/finefune-data/bin/python`。

训练会自动切换到 `configs/models/` 中指定的模型 Python。模型配置中的工程、解释器、基础及辅助权重路径需有效，训练在可访问 PPU 的 DSW 原生 Terminal 中执行。

## 路线 A：单次微调

以下示例使用官方 3D-MOOD 权重，在现成的 HOT3D 上微调三维几何头：

```bash
CFG="$FT/configs/experiments/3dmood_hot3d.yaml"
EXP=/root/finefune-runs/3dmood-hot3d-exp01

"$PY" -B "$FT/run.py" train --config "$CFG" --output-dir "$EXP"
```

**这一条 `train` 命令就能完成一次微调，无需另跑 `prepare`、`verify` 或 `suite`。** 默认训练 4 轮，单张前向、累积 8 张，只更新三维几何头。

训练 WildDet3D 时，将 `CFG` 改为 `$FT/configs/experiments/wilddet3d_hot3d.yaml`，`EXP` 改为 `/root/finefune-runs/wilddet3d-hot3d-exp01`，再执行同一条训练命令。每次实验从所选模型配置中的完整基础权重初始化。

### 更换数据或训练参数

例如，用 3D-MOOD 在 Cosmos 全通用提示词数据上训练：

```bash
CFG="$FT/configs/experiments/3dmood_hot3d.yaml"
EXP=/root/finefune-runs/3dmood-cosmos-generic-exp01

"$PY" -B "$FT/run.py" train --config "$CFG" \
  --dataset-config "$FT/configs/datasets/cosmos_generic.yaml" \
  --output-dir "$EXP"
```

| 参数 | 含义与可填内容 |
|---|---|
| `--config` | 实验模板 `3dmood_hot3d.yaml`、`wilddet3d_hot3d.yaml`；续训和评测用保存的 `resolved_config.yaml` |
| `--dataset-config` | `configs/datasets/` 下的数据配置，见下表；不填则使用实验模板中的数据 |
| `--output-dir` | 独立实验目录，例如 `/root/finefune-runs/3dmood-cosmos-generic-exp01` |
| `--epochs` | 正整数，默认 `4` |
| `--lr` | 正学习率，默认 `0.0001` |
| `--grad-accum-steps` | 正整数，默认 `8` |

| 数据配置 | 用途 |
|---|---|
| `hot3d.yaml` | 已准备的 HOT3D：15,000 训练 / 3,000 评测 |
| `libero.yaml` | 完整 LIBERO agentview 划分 |
| `libero_matched.yaml` | 与 Cosmos 共用训练状态的 LIBERO 原图，用于公平对照 |
| `cosmos_generic.yaml` | 全通用：训练 `train_all_general`，评测 `eval_general` |
| `cosmos_mixed.yaml` | 混合：训练 `train_mixed`，评测 `eval_general` |

更换模型、数据、训练配方或基础权重时使用新输出目录。数据应已由原制作模块准备好；本模块只建立索引和规范化 GT，不重新抽帧或生成图片。现成 HOT3D 输入 `/root/hot3d-finetune-v1/data/hot3d-aria-18k-v1` 继续使用。

## 中断续训：仅中断后需要

将 `EXP` 设置为原实验目录，使用保存的完整配置恢复：

```bash
EXP=/root/finefune-runs/3dmood-hot3d-exp01

"$PY" -B "$FT/run.py" train \
  --config "$EXP/resolved_config.yaml" --resume auto
```

`--resume auto` 从本实验的 `checkpoints/last.pt` 恢复，没有恢复点时开始新训练。续训保持原目录和原配置；已有恢复点时必须显式加 `--resume`。

训练完成标志：`status.json` 的 `stage` 为 `complete`，且 `checkpoints/final_head.pt` 存在并通过来源校验。只有 `last.pt` 不表示训练完成。

## 单次训练后的评测：需要对比效果时执行

设置 `EXP` 为已完成的实验目录，分别评测官方和微调权重。默认使用本实验冻结的同一份评测集：

```bash
"$PY" -B "$FT/run.py" evaluate \
  --config "$EXP/resolved_config.yaml" --variant official --tracks text

"$PY" -B "$FT/run.py" evaluate \
  --config "$EXP/resolved_config.yaml" --variant finetuned --tracks text
```

`official` 加载基础权重，`finetuned` 加载同一基础权重并覆盖本次三维头，因此官方基线也可以在训练结束后补跑。若想提前做官方评测，先执行 `prepare --config "$CFG" --output-dir "$EXP"`（作为 `run.py` 的子命令），建立索引后执行官方评测；有数据覆盖参数时也须传给 `prepare`。

WildDet3D 可用 `--tracks text gt-box`，3D-MOOD 使用 `--tracks text`。改用统一的独立测试集时，在两条评测命令末尾都加上相同参数：

```text
--tests-config /mnt/workspace/3dmood/ioutest/configs/robot_boxing_v1.json --datasets hot3d_eval libero_original_eval libero_realistic_eval
```

指标、计时和执行设置继承 ioutest 配置。报告分别位于 `$EXP/evaluation/official/reports/report.md` 和 `$EXP/evaluation/finetuned/reports/report.md`。

## 路线 B：全量对照实验

需要整套实验时执行这一条命令，无需先执行路线 A：

```bash
"$PY" -B "$FT/run.py" suite \
  --config "$FT/configs/suites/model_data_compare.yaml"
```

默认运行 **两模型 × 四训练数据 = 八组独立微调**。四种数据为 HOT3D、对齐后的 LIBERO 原图、Cosmos 全通用、Cosmos 混合。每组均从对应官方基础权重开始。

程序依次执行：准备索引 → 来源重叠审计 → 两模型官方基线 → 八组训练及各自评测 → 汇总报告。所有组都在配置中的三个独立测试集上评测，PPU 子任务依次运行。

| 可选参数 | 作用与可填内容 |
|---|---|
| `--dry-run` | 只解析配置和展示计划，不启动训练，也不读取审计全部数据 |
| `--models` | `3dmood`、`wilddet3d`，可填一个或两个 |
| `--datasets` | 训练数据 ID：`hot3d`、`libero`、`cosmos_generic`、`cosmos_mixed`，可填多个 |

输出目录为 `/root/finefune-runs/model-data-compare-v1`。中断后重新执行同一条 `suite` 命令，程序会校验并复用已完成结果、恢复未完成训练。不要同时启动两份相同实验。此路线使用自己的实验目录，不接管路线 A 的训练目录。

## 可选检查：排查问题时使用

这些命令不属于每次训练的必经步骤。`CFG`、`EXP` 应与待训练实验一致；训练时覆盖了数据等参数，检查时也传入相同覆盖参数。

```bash
# 读取、审计实际数据，查看预计更新次数；不检查 PPU、不写产物
"$PY" -B "$FT/run.py" train --config "$CFG" --output-dir "$EXP" --dry-run

# 检查数据、模型文件和运行环境；ready: true 表示检查通过
"$PY" -B "$FT/run.py" check --config "$CFG" --output-dir "$EXP"
```

数据量大时训练 `--dry-run` 也可能耗时，通过不代表真实前后向已验证。`prepare` 用于提前建立索引，`verify` 用于检查已建立的索引，均可按需使用。独立索引及完整参数见 [README.md](README.md)，真实前后向验证说明见 [VALIDATION.md](VALIDATION.md)。

## 最终输出

```text
/root/finefune-runs/<单次实验目录>/
├── resolved_config.yaml          完整配置，供续训和评测使用
├── dataset/                      统一索引、GT、协议与审计
├── dataset_config.yaml
├── train.jsonl                   训练来源清单
├── status.json                   训练进度与完成状态
├── checkpoints/
│   ├── last.pt                   恢复点
│   └── final_head.pt             最终三维头，需搭配对应基础权重
├── logs/train.jsonl              训练更新日志
└── evaluation/                   执行评测后产生
    ├── official/reports/report.md
    └── finetuned/reports/report.md

/root/finefune-runs/model-data-compare-v1/   选择路线 B 时产生
├── suite_status.json             整套实验的进度
├── baselines/                    官方基线
├── experiments/                  各组独立实验
└── reports/report.md             对照汇总
```

以上为主要产物，完整说明见 [README.md](README.md)。
