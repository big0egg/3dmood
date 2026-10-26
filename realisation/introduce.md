# Cosmos-Transfer2.5：项目运行说明

**当前 v3 实验使用单视角全量入口：**

```bash
bash /mnt/workspace/3dmood/realisation/scripts/run_agentview_prompt_full_v3.sh --background
```

该入口自动准备两份运行配置，处理主数据 5,500 张和同帧对照 200 张，明确的内容拦截记录后继续。
所有产物位于 `/root/robot-realism/agentview_prompt_v3/`，旧 75 张不导入。
状态、拦截清单和校验方式见 [README 的 v3 说明](README.md)。

先按 [README.md](README.md) 完成环境安装和权重准备，再使用上面的 v3 全量入口。下面先说明当前 v3 的产物和输入目录；后面的六步保留为历史双视角流程参考。

## 当前 v3 的产物目录结构

`/root/robot-realism/agentview_prompt_v3/` 是整个实验的根目录。实际生成的图片和继承标注位于其 `datasets/` 子目录下。

```text
/root/robot-realism/agentview_prompt_v3/
├── experiment.json                    固定的实验配置与来源身份
├── configs/
│   ├── main.yaml                      主数据实际运行配置
│   └── paired_general.yaml            同帧通用提示词对照配置
├── datasets/
│   ├── main/
│   │   ├── dataset_info.json
│   │   ├── manifest.json              生成结果、成功和拦截清单
│   │   ├── categories.json
│   │   ├── sampling_plan.json
│   │   ├── scene_split.json
│   │   ├── runtime_identity.json
│   │   ├── provenance/                继承的场景 XML 来源文件
│   │   ├── train/
│   │   │   └── 场景/状态/
│   │   │       ├── state.npy
│   │   │       ├── COMMIT.json         成功提交记录与文件校验值
│   │   │       ├── PENDING.json        写入归属记录，提交后也可能保留
│   │   │       └── agentview/
│   │   │           ├── rgb.png         Cosmos 生成的彩色图片
│   │   │           ├── depth.npz       继承的仿真深度
│   │   │           ├── instance.png    继承的实例分割
│   │   │           ├── annotation.json 继承的相机和 3D 标注
│   │   │           └── render_meta.json 生成参数、提示词与模型来源
│   │   └── eval/                      样本结构与 train 相同
│   └── paired_general/                同帧对照数据，元数据结构同 main
│       └── train/场景/状态/agentview/  仅训练对照，没有 eval 目录
├── manifests/                         各提示词组及训练、评测样本清单
├── prompts/                           通用、负面及逐帧提示词
├── previews/
│   ├── source/                        源图预览
│   ├── paired/                        同帧两种提示词的对照图
│   └── eval/                          评测生成结果预览
├── reports/                           汇总报告、拦截清单和人工审核表
├── runs/                              启动日志、进度与模型详细日志
│   ├── main/
│   └── paired_general/
└── scratch/                           两组任务的临时控制图及辅助文件
```

v3 仅生成外部相机 `agentview`。主数据计划为 5,000 张训练图和 500 张评测图，训练图包含 4,800 张通用提示词图和 200 张逐帧提示词图；`paired_general/` 为这 200 个训练状态另生成通用提示词对照。因此实验合计计划处理 5,700 张图片。

| 要查看的内容 | 相对于实验根目录的位置 |
|---|---|
| 主训练图片 | `datasets/main/train/场景/状态/agentview/rgb.png` |
| 主评测图片 | `datasets/main/eval/场景/状态/agentview/rgb.png` |
| 同帧通用提示词对照 | `datasets/paired_general/train/场景/状态/agentview/rgb.png` |
| 各组实际可用样本 | `manifests/train_general.jsonl`、`train_specific.jsonl`、`eval_general.jsonl`、`paired_general.jsonl` |
| 使用共同可用状态的两套训练清单 | `manifests/train_mixed.jsonl`、`train_all_general.jsonl` |
| 缺少任一版本的对照状态 | `manifests/paired_missing.jsonl` |
| 生成数量与运行状态 | `reports/summary.json`、`runs/suite_status.json` |
| 拦截记录及对照报告 | `reports/blocked.csv`、`reports/blocked.jsonl`、`reports/comparison.md` |

实际成功图片数以清单和汇总报告为准。内容检查拦截项单独记录，不提交占位图片；存在 `PENDING.json` 也不能单独证明样本成功，应以有效的 `COMMIT.json` 和生成清单为准。生成图沿用仿真标注，图像与 GT 的视觉对齐仍需审核。

## 当前 v3 的输入 GT 数据结构

当前 v3 实际运行配置中的输入是 robot-synth 采集完成的 GT 数据集：

```text
/root/opendatasets/LIBERO_GT_v1
```

该输入目录记录了 5,500 个状态，每个状态有外部相机 `agentview` 和腕部相机 `robot0_eye_in_hand` 两张图片，共 11,000 张。v3 从中读取 `agentview`；腕部相机图片仍保留在输入数据中。

| 输入内容 | 文件位置 | 用途 |
|---|---|---|
| 原始彩色图片 | 每个相机目录下的 `rgb.png` | 作为单帧输入 |
| 仿真深度 | 每个相机目录下的 `depth.npz` | 生成深度控制图，约束物体的空间结构；其中 `z_m` 是米制深度，`valid` 是有效像素标记，尺寸须与 RGB 一致 |
| 文本提示词 | 配置中的 `generation.prompt`、`generation.negative_prompt` | 描述希望生成的真实外观；默认已经填写 |
| 原始实例分割 | 每个相机目录下的 `instance.png` | 随结果保存，并用于制作标注对照图 |
| 相机和 3D 标注 | 每个相机目录下的 `annotation.json` | 继承相机参数和物体标注，用于检查生成图与原标注的对应关系 |
| 仿真状态及来源 | 每个状态的 `state.npy`、`COMMIT.json`，以及根目录的 `provenance/` | 校验源文件完整性并保留仿真来源 |
| 数据集清单 | 根目录的 `dataset_info.json`、`manifest.json` | 读取样本、相机及训练／评测划分 |

默认深度控制流程会把 `depth.npz` 自动转换成模型需要的控制图。分割、3D 标注和仿真状态用于本工具的数据校验、保存和检查，不作为默认生成模型的控制信号。

输入 GT 数据的主要目录结构如下，完整目录还包含类别、采样计划等元数据：

```text
LIBERO_GT_v1/
├── dataset_info.json
├── manifest.json
├── provenance/                         场景 XML 来源文件
├── train/
│   └── 场景/状态/
│       ├── state.npy
│       ├── COMMIT.json
│       ├── agentview/
│       │   ├── rgb.png
│       │   ├── depth.npz
│       │   ├── instance.png
│       │   └── annotation.json
│       └── robot0_eye_in_hand/          包含同样的四个文件
└── eval/                               与 train 相同
```

**该 GT 目录作为输入保持原样，生成结果写入前述实验目录的 `datasets/`。** 原始 LIBERO HDF5 或单独的图片文件夹不能直接替代完整 GT 数据集。

## 历史双视角流程参考

以下第 1～6 步及参数说明保留原 `cosmos25_libero_v1.yaml` 的双视角流程和旧路径示例，包括 `LIBERO_GT_v1_plane_fix`。这些路径与当前 v3 运行配置不同；当前 v3 使用本文开头的全量入口，实际配置见 `/root/robot-realism/agentview_prompt_v3/configs/`。

## 第 1 步：进入工程并激活环境

每次打开新终端，先执行：

```bash
cd /mnt/workspace/3dmood/realisation
source /tmp/robot-realism/cosmos25/venv/bin/activate
export HF_HOME=/tmp/robot-realism/cosmos25/cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
which python
```

**完成标志：** 显示 `/tmp/robot-realism/cosmos25/venv/bin/python`。

如果按 README 手动准备了权重，再执行：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

如果按 README 选择自动下载，再执行：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
```

## 第 2 步：确认输入和结果保存位置

打开 `configs/realism/cosmos25_libero_v1.yaml`，确认下面这些字段。首次使用默认路径时可以保持原样；这里仅列出路径部分，其他配置保留。

```yaml
input:
  root: /root/opendatasets/LIBERO_GT_v1_plane_fix
  cameras: [agentview, robot0_eye_in_hand]
output:
  root: /root/opendatasets/LIBERO_REALISM_cosmos25_v1
  run_dir: ../../runs/realism/cosmos25_libero_v1
  scratch: /tmp/robot-realism/cosmos25/staging
```

`input.root` 是已经采集好的 GT 数据集；`output.root` 保存生成后的图片和继承的标注；`run_dir` 保存进度与对照报告；`scratch` 保存临时控制图。

```bash
ls /root/opendatasets/LIBERO_GT_v1_plane_fix/dataset_info.json
ls /root/opendatasets/LIBERO_GT_v1_plane_fix/manifest.json
```

**完成标志：** 两个文件存在。首次生成的结果目录应为空或尚未创建。

## 第 3 步：检查环境并建立任务计划


先检查运行环境：

```bash
python -m robot_realism doctor --config configs/realism/cosmos25_libero_v1.yaml
```

**完成标志：** 显示 `ready: true`。此检查会核对依赖、加载当前控制方式的模型配置并测试 PPU 基础算子，全部通过后再建立任务计划：

```bash
python -m robot_realism plan --config configs/realism/cosmos25_libero_v1.yaml
```

**完成标志：** 计划为 5,500 个状态、11,000 张图片。每个状态包含外部相机和机械臂腕部相机各一张。

## 第 4 步：先生成两张，查看效果


先用一个状态检查是否能完成生成。

```bash
python -m robot_realism render \
  --config configs/realism/cosmos25_libero_v1.yaml \
  --limit 2
```

**完成标志：** 显示 `accepted_images: 2`，生成结果目录中实际出现两张新图片。然后检查文件并生成对照图：

```bash
python -m robot_realism validate --config configs/realism/cosmos25_libero_v1.yaml --allow-partial
python -m robot_realism report --config configs/realism/cosmos25_libero_v1.yaml --max-previews 2
```

**完成标志：** 检查显示 `valid: true`。打开下面的目录查看对照图：

```text
/mnt/workspace/3dmood/realisation/runs/realism/cosmos25_libero_v1/quality_report/previews/
```

对照图从左到右是：原图、生成图、原标注叠加到生成图。确认物体数量、位置、轮廓和遮挡关系合适，再进入下一步。

生成图继承原仿真标注。生成模型可能改变物体形状，两个相机的生成内容也不保证严格一致，所以文件检查通过后仍需查看对照图。

## 第 5 步：生成到 200 张，再查看一批结果


继续生成到累计 200 张，前面完成的两张会复用。

```bash
python -m robot_realism render \
  --config configs/realism/cosmos25_libero_v1.yaml \
  --limit 200 \
  --resume

python -m robot_realism validate --config configs/realism/cosmos25_libero_v1.yaml --allow-partial
python -m robot_realism report --config configs/realism/cosmos25_libero_v1.yaml --max-previews 40
```

**完成标志：** 累计 200 张、检查显示 `valid: true`。在同一个对照图目录查看结果，效果符合要求后继续。

## 第 6 步：生成全部数据并完成检查


去掉张数限制，继续处理全部输入：

```bash
python -m robot_realism render \
  --config configs/realism/cosmos25_libero_v1.yaml \
  --resume
```

生成结束后执行：

```bash
python -m robot_realism validate --config configs/realism/cosmos25_libero_v1.yaml
python -m robot_realism report --config configs/realism/cosmos25_libero_v1.yaml --max-previews 40
python -m robot_realism status --config configs/realism/cosmos25_libero_v1.yaml
```

**完成标志：** 检查显示 `valid: true`、`complete: true`，训练集 10,000 张、评测集 1,000 张。

| 要查看的内容 | 位置 |
|---|---|
| 生成的图片 | `/root/opendatasets/LIBERO_REALISM_cosmos25_v1/train/场景/状态/相机/rgb.png`，评测图片在 `eval/` 下 |
| 深度、分割和标注 | 每张图片同目录下的 `depth.npz`、`instance.png`、`annotation.json` |
| 对照报告 | `runs/realism/cosmos25_libero_v1/quality_report/report.md` |
| 对照图片 | `runs/realism/cosmos25_libero_v1/quality_report/previews/` |

生成中断后，重新进入工程、激活环境并设置权重读取方式，再执行带 `--resume` 的生成命令即可。生成过程中保持配置、代码和权重不变。

## 参数说明（需要修改时再看）


默认配置对应上述流程。`--limit` 按图片数计数，两个相机时填正偶数；`--limit 200` 表示累计到 200 张。`--resume` 表示接着已有结果继续。

<details>
<summary>展开查看命令参数</summary>

| 参数 | 用在哪里 | 意义及可填内容 |
|---|---|---|
| `--config` | 所有运行命令 | YAML 配置文件路径，默认 `configs/realism/cosmos25_libero_v1.yaml` |
| `--limit` | `render`、`prepare` | 累计图片数量，如 `2`、`200`；不填则处理全部输入 |
| `--resume` | `render` | 接着已有结果继续，不需要填写数值 |
| `--allow-partial` | `validate` | 允许检查尚未达到全量的试生成结果 |
| `--max-previews` | `report` | 对照图的最多张数，默认 `40` |
| `--input` | 所有运行命令 | 覆盖输入 GT 数据目录 |
| `--output` | 所有运行命令 | 覆盖生成结果目录；不自动修改运行记录目录 |
| `--help` | 各命令 | 查看该命令的参数 |

</details>

<details>
<summary>展开查看配置文件参数</summary>

配置文件是 `configs/realism/cosmos25_libero_v1.yaml`。使用空格缩进；布尔值填写 `true` 或 `false`。文件中的相对路径从该 YAML 所在目录计算。

| 配置项 | 默认值 | 意义及可填内容 |
|---|---|---|
| `schema_version` | `1` | 当前配置格式版本，只填整数 `1` |
| `name` | `libero_cosmos25_v1` | 字符串，配置名称；修改名称不会自动改路径 |
| `input.root` | `/root/opendatasets/LIBERO_GT_v1_plane_fix` | 已提交的仿真 GT 数据根目录；不是原始 LIBERO HDF5 目录 |
| `input.cameras` | `[agentview, robot0_eye_in_hand]` | 非空且不可重复的相机名列表，必须存在于输入数据；名称可含字母、数字、下划线和短横线 |
| `output.root` | `/root/opendatasets/LIBERO_REALISM_cosmos25_v1` | 新生成数据目录，首次应为空或不存在 |
| `output.run_dir` | `../../runs/realism/cosmos25_libero_v1` | 计划、进度和报告目录 |
| `output.scratch` | `/tmp/robot-realism/cosmos25/staging` | 本地控制图、原生结果及日志目录；成功提交后清理对应控制图和原生 RGB，日志等辅助文件保留 |
| `runtime.python` | `/tmp/robot-realism/cosmos25/venv/bin/python` | 完成安装的 Cosmos 环境解释器完整路径；保留 venv 路径，不手动替换成它指向的系统 Python |
| `runtime.cache` | `/tmp/robot-realism/cosmos25/cache` | 模型及下载缓存根目录，需有足够本地空间 |
| `runtime.enable_guardrails` | `true` | 布尔值；是否运行上游文本／图像检查。被阻止的生成会报错，不用占位图补数 |
| `runtime.offload_guardrails` | `true` | 布尔值；是否将检查模型卸载到 CPU，节省显存并使用更多主机内存 |
| `generation.controls` | `[depth]` | 当前支持 `[depth]`、`[edge]`、`[depth, edge]`，不可重复。多控制会载入多个控制分支，显存、耗时和效果需单独验证 |
| `generation.resolution` | `'720'` | 模型内部目标分辨率档位，填字符串 `'480'` 或 `'720'`。最终恢复输入图片尺寸；不是自定义像素宽高 |
| `generation.steps` | `35` | 正整数，扩散采样步数；增加通常更慢，不能保证一定更好 |
| `generation.guidance` | `3` | 整数 `0`～`7`，提示词引导强度；过强可能加剧外观或内容变化 |
| `generation.prompt` | YAML 内的英文说明 | 非空文本。描述真实材质和照明，并要求保留物体、相机和遮挡；首版各状态共用，不自动根据 GT 猜测场景语义 |
| `generation.negative_prompt` | YAML 内的负向说明 | 非空文本。描述不希望出现的卡通、增删物体、变形等情况；不是避免几何漂移的保证 |
| `controls.near_m` | `0.05` | 正数，米制深度的近端截断值，须小于 far_m |
| `controls.far_m` | `5.0` | 大于 near_m 的正数，深度远端截断值。截断后取逆深度，再在有效像素内归一化到 0～255；近处较白 |
| `controls.depth_weight` | `1.0` | 数值大于 0 且小于等于 1，深度控制权重 |
| `controls.edge_weight` | `1.0` | 数值大于 0 且小于等于 1，边缘控制权重 |
| `controls.canny_low` | `100` | Canny 低阈值，0～255，须小于 high |
| `controls.canny_high` | `200` | Canny 高阈值，0～255，须大于 low |
| `selection.seed` | `20261022` | 整数，控制场景内样本顺序和每状态的生成随机种子；两个相机共享状态种子，但不保证生成内容一致 |

输入、结果、运行记录、临时目录必须互不包含。更改环境或缓存路径时，也要同步修改安装命令及权重准备路径。

README 中的手动权重清单对应默认 `[depth]`。如需更换输入、提示词或控制方式，应使用新配置及新的结果、运行记录和临时目录，并准备对应模型权重。

</details>
