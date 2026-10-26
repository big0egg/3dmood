# finefune 操作步骤

支持 3D-MOOD、WildDet3D，以及准备好的 HOT3D、LIBERO、Cosmos 数据，只微调三维几何头。参数与数据规则见 [README.md](README.md)，验证范围见 [VALIDATION.md](VALIDATION.md)。

## 1. 设置路径与检查模型配置

```bash
FT=/mnt/workspace/3dmood/finefune
PY=/usr/local/bin/python3
EXP=/root/finefune-runs/3dmood-hot3d-exp01
```

基础 Python 需有 NumPy、SciPy、Pillow、PyYAML。需要独立控制环境时运行 bash "$FT/setup_data_env.sh"，然后将 PY 改为 /mnt/workspace/venvs/finefune-data/bin/python。训练自动使用 configs/models 中所选模型的 PPU Python。

检查 configs/models/3dmood.yaml 或 wilddet3d.yaml 的工程、Python、完整基础权重路径。WildDet3D 还需要 LingBot 辅助权重。--origin-ckpt 可覆盖基础权重，不能填三维头增量文件。

## 2. 选择数据与训练配方

| 配置文件 | 用途 |
|---|---|
| configs/datasets/hot3d.yaml | 已准备的 15,000/3,000 HOT3D |
| configs/datasets/libero.yaml | 完整 LIBERO agentview 划分 |
| configs/datasets/libero_matched.yaml | 与 Cosmos 共用训练状态的 LIBERO 原图 |
| configs/datasets/cosmos_generic.yaml | 全通用：train_all_general、eval_general |
| configs/datasets/cosmos_mixed.yaml | 混合：train_mixed、eval_general |

默认配方 configs/training/geometry_head.yaml 为 4 轮、FP32、单张前向、累积 8 张。修改配方、数据或基础权重时新建输出目录。

单模型模板：configs/experiments/3dmood_hot3d.yaml、wilddet3d_hot3d.yaml。使用其他数据时复制模板，修改 dataset_config 和 output_dir，或通过命令行覆盖。

## 3. 只读检查

```bash
"$PY" -B "$FT/run.py" train \
  --config "$FT/configs/experiments/3dmood_hot3d.yaml" --dry-run

"$PY" -B "$FT/run.py" check \
  --config "$FT/configs/experiments/3dmood_hot3d.yaml"
```

确认数据数量、参数与预计更新次数正确；check 必须 ready: true。只检查所选模型，PPU 不可用不会自动使用 CPU。dry-run 成功不代表真实前后向已验证。

## 4. 启动一次微调

```bash
"$PY" -B "$FT/run.py" train \
  --config "$FT/configs/experiments/3dmood_hot3d.yaml"
```

完整原始 checkpoint 用于初始化，只更新三维几何头。训练日志在 $EXP/logs/train.jsonl，状态在 $EXP/status.json。

先验证一次真实参数更新时，使用独立目录：

```bash
"$PY" -B "$FT/run.py" train \
  --config "$FT/configs/experiments/3dmood_hot3d.yaml" \
  --output-dir /root/finefune-runs/3dmood-hot3d-native-check \
  --stop-after-updates 1
```

状态 stopped 表示按要求保存并停止，不是训练完成。真实前后向检查需在可访问 PPU 的 DSW 原生 Terminal 执行。

## 5. 中断恢复与完成标志

```bash
"$PY" -B "$FT/run.py" train \
  --config "$EXP/resolved_config.yaml" --resume auto
```

完成条件是 status.json 的 stage 为 complete，并且 checkpoints/final_head.pt 存在且来源校验成功。last.pt 只表示有恢复点。已有恢复点必须显式 resume；不一致配置会被拒绝，原快照保留。

## 6. 评测原始与微调权重

微调前基线应在第 4 步之前完成：先用模板 prepare 建立本实验索引，再使用保存的配置评测。手动单模型流程也可在训练后补跑；完整 suite 强制先完成所有基线。

```bash
# 尚未建立实验索引时先执行；已有同配置索引可跳过
"$PY" -B "$FT/run.py" prepare \
  --config "$FT/configs/experiments/3dmood_hot3d.yaml"

"$PY" -B "$FT/run.py" evaluate \
  --config "$EXP/resolved_config.yaml" --variant official --tracks text

"$PY" -B "$FT/run.py" evaluate \
  --config "$EXP/resolved_config.yaml" --variant finetuned --tracks text
```

评测调用 ioutest，默认读取冻结的 eval；WildDet3D 可传 --tracks text gt-box。报告位于 $EXP/evaluation/<official或finetuned>/reports/report.md。微调三维头必须搭配对应原始权重。

## 7. 一次运行两模型×四训练数据

```bash
"$PY" -B "$FT/run.py" suite \
  --config "$FT/configs/suites/model_data_compare.yaml" --dry-run
```

审核六个实验的数据、基础权重、配方、三个独立测试集与输出路径，然后启动：

```bash
"$PY" -B "$FT/run.py" suite \
  --config "$FT/configs/suites/model_data_compare.yaml"
```

可加 --models 3dmood --datasets hot3d 缩小组合。默认顺序为准备→来源重叠审计→两模型基线→八组训练→微调后评测→报告，PPU 子任务依次运行。

进度在 /root/finefune-runs/model-data-compare-v1/suite_status.json，报告在其 reports/report.md。中断后使用同一命令核验并恢复，不同时启动两份相同实验。

## 8. 现成数据输入

/root/hot3d-finetune-v1/data/hot3d-aria-18k-v1 继续作为现成数据输入，不重新抽帧。微调权重使用本次实验目录下 checkpoints/final_head.pt，并通过 export-eval 导出评测配置。

评测指标、计时及执行设置继承统一 ioutest 配置，显式评测 --device／--seed 优先。WildDet3D 权重统一在 /root/wilddet3d-weights；3D-MOOD BERT 缓存由模型配置 environment.HF_HOME 指定。
