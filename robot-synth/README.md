# LIBERO Ground Truth 数据集采集操作说明书

这份说明书用于从已有 LIBERO 示范中恢复仿真状态，生成 RGB 图片、深度、实例分割、相机标定和物体 3D 真值。本阶段的交付是 **11,000 张图片及对应 GT：训练集 10,000 张，评测集 1,000 张**。

**第一次使用，按第二部分的步骤 1～12 顺序执行即可。默认参数已经配好，不需要先修改 YAML。** 如果使用 Docker，请改按第七部分执行；两种环境方式选一种。

> 当前进度（2026-10-21）：环境已能真实渲染，旧版本试采保存了 20 张，随后在客厅场景的无限地面框计算处停止。无限地面处理已修复；这批旧结果保留，修复后请按 [3.4 节](#34-遇到无限地面报错后使用修复版本)使用独立的新配置重新试采，不直接续写旧版本。Docker 镜像尚未构建验证。详细记录见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。

## 目录

- [一、先了解数据数量和目录](#一先了解数据数量和目录)
- [二、按顺序执行：准备环境到完成采集](#二按顺序执行准备环境到完成采集)
- [三、中断后怎么继续、怎么重新做一版](#三中断后怎么继续怎么重新做一版)
- [四、命令和命令行参数说明](#四命令和命令行参数说明)
- [五、YAML 配置参数说明](#五yaml-配置参数说明)
- [六、生成的文件怎么看](#六生成的文件怎么看)
- [七、可选：使用 Docker 运行](#七可选使用-docker-运行)
- [八、常见问题处理](#八常见问题处理)

## 一、先了解数据数量和目录

### 1.1 “一帧”按什么计数

本工程以**一张相机图片**作为一帧。默认同一仿真状态拍摄两个视角：

| 相机名称 | 含义 |
|---|---|
| `agentview` | 从外部观察机械臂和物体的相机 |
| `robot0_eye_in_hand` | 安装在机械臂腕部的相机 |

因此，一个状态产生两张图片。数量目标如下：

| 划分 | 仿真状态数 | 相机数 | 最终图片数 |
|---|---:|---:|---:|
| `train`：训练集 | 5,000 | 2 | 10,000 |
| `eval`：评测集 | 500 | 2 | 1,000 |
| 合计 | 5,500 | 2 | 11,000 |

`--limit 200` 也按图片计数，对应最多选择 100 个双相机状态。

### 1.2 默认使用哪些目录

| 用途 | 默认目录 | 说明 |
|---|---|---|
| 程序代码 | `/mnt/workspace/3dmood/robot-synth` | 所有命令先进入这里执行 |
| 原始示范 | `/mnt/data/opendatasets/LIBERO` | 读取 `libero_90` 和 `libero_10`，保持只读 |
| 最终 GT 数据 | YAML 的 `output.root` | 原始方案为 `/mnt/data/opendatasets/LIBERO_GT_v1`；当前配置已由用户改为 `/root/opendatasets/LIBERO_GT_v1` |
| 计划和运行记录 | YAML 的 `output.run_dir` | 原始方案为 `../runs/libero_gt_v1`；当前配置为 `../runs/libero_gt_local_v1`，均以配置文件目录为基准 |
| 临时缓存 | `/tmp/robot-synth/libero_gt_v1` | 写完并核验上传的样本组会清理本地缓存 |
| 配置文件 | `configs/libero_gt_v1.yaml` | 管理输入输出、数量、相机等参数 |

**原始示范、最终结果、运行记录、临时缓存这四个目录必须互不包含。** 运行记录可以放在工程目录下，最终数据不要写进原始示范目录。

下文主流程展示原始方案的路径；以实际 YAML 为准。若使用修复版本，统一按 3.4 节操作。`/mnt/data` 是 OSS 挂载；`/root/opendatasets` 是否为挂载点需在实例上确认，不能仅凭目录名判断。`df` 显示的巨大容量不代表实际配额；临时缓存所在磁盘也需要保留足够空间。先用 200 张试采测量数据大小，再扩展到全量。

## 二、按顺序执行：准备环境到完成采集

本节适用于**专门用于仿真采集的 DSW 镜像／实例终端**，采用 OSMesa 软件离屏渲染。Python 依赖安装到本工程 `.venv`，模型项目原有的 PPU 虚拟环境不参与安装。

### 步骤 1：进入工程，检查数据和磁盘

**这一步做什么：** 确认代码和原始数据存在，确认暂存磁盘有空间。

```bash
cd /mnt/workspace/3dmood/robot-synth

ls configs/libero_gt_v1.yaml
ls /mnt/data/opendatasets/LIBERO/libero_90
ls /mnt/data/opendatasets/LIBERO/libero_10

df -h /mnt/workspace /tmp
python3 --version
```

**正常结果：** 能看到配置文件和 `.hdf5` 示范文件。建议本流程使用 Python 3.12，与提供的 Dockerfile 一致。

**不正常时：** 数据路径不存在，先确认 OSS 是否挂载；磁盘不足，先选择足够大的临时缓存磁盘，再调整 `output.scratch`。此时不要启动采集。

### 步骤 2：准备系统渲染库

**这一步做什么：** 安装离屏渲染需要的系统库，以及下载仿真资产需要的 Git。

以下命令需要管理员权限，在专用采集镜像中执行；非 root 用户需使用该环境允许的管理员方式。

```bash
apt-get update
apt-get install -y --no-install-recommends git libosmesa6 libgl1 ca-certificates
```

检查 OSMesa 是否可以找到：

```bash
python3 - <<'PY'
import ctypes.util
print("OSMesa:", ctypes.util.find_library("OSMesa"))
PY
```

**正常结果：** 显示类似 `libOSMesa.so.8` 的库名。

**不正常时：** 如果显示 `None`，说明系统库还没有准备好。没有系统安装权限时，使用第七部分的独立 Docker 镜像方案。

### 步骤 3：安装采集用 Python 环境和仿真资产

**这一步做什么：** 在工程下建立 `.venv`，安装固定依赖，并下载 LIBERO、robosuite 的资产源码。

需要能够访问 Python 软件包源和 GitHub。原始 HDF5 已在 OSS 上，本步骤不重新下载示范数据。

```bash
cd /mnt/workspace/3dmood/robot-synth
bash scripts/bootstrap_runtime.sh
```

脚本会依次执行：

1. 创建工程下的 `.venv`。
2. 安装 `requirements-runtime.txt` 中的依赖，包括 `mujoco==3.2.7`。
3. 安装本工程命令入口。
4. 将 LIBERO 下载到 `third_party/LIBERO`，默认引用 `master`。
5. 将 robosuite 下载到 `third_party/robosuite`，默认引用 `v1.4.0`。
6. 检查系统是否有 OSMesa，并输出资产仓库的实际 Git 提交。

**正常结果：** 最后显示 `Python dependencies/assets present`，并存在以下目录：

```text
.venv/
third_party/LIBERO/libero/libero/assets/
third_party/robosuite/robosuite/models/assets/
```

**不正常时：** 如果软件包或 Git 下载报网络错误，先修复该终端的网络，再重跑本步骤。若脚本最后提示缺少 OSMesa，回到步骤 2。

已有资产目录不会被脚本自动更新。要指定版本、使用其他 Python 或其他虚拟环境目录，见第五部分的“安装脚本参数”。

### 步骤 4：激活环境并检查版本

**这一步做什么：** 让后续命令使用刚建立的采集环境。

```bash
cd /mnt/workspace/3dmood/robot-synth
source .venv/bin/activate

which python
python -m pip show mujoco h5py numpy
python -m robot_synth --help
```

**正常结果：** Python 路径以工程下的 `.venv/bin/python` 结尾；MuJoCo 为 `3.2.7`；帮助中列出 `inspect`、`doctor`、`plan`、`collect`、`validate`、`report`。

**以后每次重新打开终端，先执行本步骤前两行。** 若安装时自定义了 `CAPTURE_VENV`，激活该目录中的 `bin/activate`，不要继续使用默认 `.venv` 路径。

### 步骤 5：清点原始数据

**这一步做什么：** 读取任务、轨迹、状态数组结构、场景 XML 和资产路径信息。

```bash
python -m robot_synth inspect --config configs/libero_gt_v1.yaml
```

**正常结果：** 当前默认数据应报告约以下内容：

```json
{
  "task_count": 100,
  "episode_count": 5000,
  "timesteps": 807133
}
```

结果保存到：

```text
runs/libero_gt_v1/inventory.json
```

首次通过 OSS 读取每段轨迹的元数据可能需要数分钟。它不会把全部 RGB 读取一遍；后续源文件大小和修改时间未变化时，会复用清点记录。

**不正常时：** `No HDF5 files found` 通常表示输入路径或 `source.suites` 不对；缺少 `states` 或 `model_file` 表示该文件不满足回放要求，应先检查数据版本。

### 步骤 6：生成并固定采样计划

**这一步做什么：** 先划分场景，再决定具体使用哪些任务、轨迹和状态，避免训练与评测场景组交叉。

```bash
python -m robot_synth plan --config configs/libero_gt_v1.yaml
```

**正常结果：** 输出包括：

```json
{
  "targets": {"train": 10000, "eval": 1000},
  "candidate_states": 8255
}
```

`8255` 是当前已生成计划的候选状态数，包含候补，不是最终图片数量。最终仍按 5,500 个状态、11,000 张图片的配额收集。

产生的文件：

| 文件 | 用途 |
|---|---|
| `runs/libero_gt_v1/plan.json` | 完整状态清单、相机、配额和来源摘要 |
| `runs/libero_gt_v1/scene_split.json` | 训练／评测场景组名单和各组配额 |

当前计划保留 `KITCHEN_SCENE2`、`KITCHEN_SCENE8`、`LIVING_ROOM_SCENE6`、`STUDY_SCENE3` 用于评测，其余组用于训练。

**计划固定后，继续采集应使用同一份配置。** 修改数量、相机、分辨率或采样规则，要按第三部分建立新版本，不能覆盖旧计划。`--refresh` 只刷新清点记录，不会解除固定计划。

### 步骤 7：检查运行环境和资产是否齐全

**这一步做什么：** 检查 Python 依赖、渲染库、资产目录，以及记录中引用的具体资产文件。

```bash
python -m robot_synth doctor --config configs/libero_gt_v1.yaml
```

**可以继续的条件：** 输出中出现：

```json
{"ready": true}
```

详细结果保存在 `runs/libero_gt_v1/doctor.json`。

**如果是 `ready: false`，先处理 `checks` 中 `ok: false` 的项目：**

| 缺失项目 | 去哪里处理 |
|---|---|
| `mujoco` | 检查是否激活 `.venv`，然后检查步骤 3 的安装结果 |
| `rendering:OSMesa` | 回到步骤 2 |
| `assets:libero`、`assets:robosuite` | 检查下载结果和 YAML 的 `simulation.asset_roots` |
| `recorded_scene_asset_files` | 查看报错中的具体原始路径和尝试映射的新路径，确认资产版本及文件是否匹配 |

`doctor` 不会下载依赖或生成 GT；通过它以后，还需要步骤 8 的真实渲染测试。

### 步骤 8：运行真实离屏渲染测试

**这一步做什么：** 在无地面和带无限地面的两个真实 MuJoCo 场景中验证 RGB、米制深度和实例分割接口。

```bash
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  python -m pytest -q tests/test_native_renderer.py
```

**可以继续的条件：** 结果包含 `2 passed`。

**如果显示 `skipped`，也不能继续试采。** 这表示缺少运行条件，测试没有执行，不能当作渲染成功。根据跳过原因检查 MuJoCo 版本或 OSMesa。

此测试使用简单几何体；实际 LIBERO 资产与标注是否正确，还需要通过下一步的真实试采检查。

### 步骤 9：先采集 200 张图片

**这一步做什么：** 在完整计划中选择一个小批量，实际回放状态并同步保存图像和 GT。

```bash
python -m robot_synth collect \
  --config configs/libero_gt_v1.yaml \
  --limit 200
```

**正常结果：** 首次在全新输出目录中运行，成功收集 200 张，即 100 个双相机状态。默认计划会尽量覆盖全部场景组。

本步骤会：

1. 检查运行条件，并计算资产和源码摘要。
2. 恢复记录中的场景和状态。
3. 同步获取两个相机的 RGB、深度、分割及相机标定。
4. 根据完整可视几何生成物体 3D 真值。
5. 检查质量，通过后保存；被拒绝的状态由候补补齐。

**`--limit 200` 是本次希望达到的总量，包含已有结果，不是再新增 200 张。** 已有 200 张时重复相同的续采命令不会增加到 400 张；已有更多结果也不会被删除。

首次使用空输出目录。若输出已经有本工程记录，应加 `--resume`，见第三部分。

### 步骤 10：检查这 200 张，并查看画框图

**这一步做什么：** 自动检查已保存文件和几何关系，然后生成供人工查看的 GT 预览。

```bash
python -m robot_synth validate \
  --config configs/libero_gt_v1.yaml \
  --allow-partial

python -m robot_synth report \
  --config configs/libero_gt_v1.yaml \
  --max-previews 40
```

**正常结果：** 验证输出 `valid: true`。试采尚未达到完整配额，此时 `complete: false` 是正常的。

在 VS Code 中打开：

```text
runs/libero_gt_v1/validation.json
runs/libero_gt_v1/quality_report/report.md
runs/libero_gt_v1/quality_report/previews/
```

重点检查：

- 图片是否正常，有没有上下颠倒或空白。
- 框是否对应正确物体，中心和尺度是否明显异常。
- 外部和腕部两个视角是否都合理。
- 遮挡后物体框是否仍代表完整物体，而不是缩成只包住可见部分。
- `validation.json` 的 `errors` 是否为空。

预览默认只画物体框，隐藏机械臂、支撑物和子部件框；相应 GT 仍保存在标注中。完整 3D 框的投影可能大于可见物体轮廓，不要求边缘重合。

`report` 负责生成检查图，不代替 `validate`，也不表示完整数据集已经通过验收。

### 步骤 11：继续采集到 11,000 张

**这一步做什么：** 复用已经完成的 200 张，继续完成所有场景组的配额。

```bash
python -m robot_synth collect \
  --config configs/libero_gt_v1.yaml \
  --resume
```

这里去掉 `--limit`，表示以计划中的完整数量为目标。

**正常结果：** 完成后输出 `status: complete`，`accepted_images: 11000`。其中 train 10,000 张、eval 1,000 张。

续采会检查已保存文件、源码、依赖和资产身份，开始时可能有较多读取。此时不要同时启动另一个写入同一结果目录的任务，也不要在采集过程中改源码、配置或更新依赖。

### 步骤 12：完成最终验收

**这一步做什么：** 按完整配额检查数据，并重新生成质量报告。

```bash
python -m robot_synth validate --config configs/libero_gt_v1.yaml
python -m robot_synth report --config configs/libero_gt_v1.yaml --max-previews 40
```

最终验收时不加 `--allow-partial`。应同时满足：

```json
{
  "valid": true,
  "complete": true,
  "counts": {"train": 10000, "eval": 1000},
  "errors": []
}
```

检查图和报告仍保存在 `runs/libero_gt_v1/quality_report/`；数据本体保存在 `/mnt/data/opendatasets/LIBERO_GT_v1/`。保留运行记录和配置，便于追溯。

## 三、中断后怎么继续、怎么重新做一版

### 3.1 200 张试采中断后继续

重新进入工程并激活环境，然后执行：

```bash
cd /mnt/workspace/3dmood/robot-synth
source .venv/bin/activate
python -m robot_synth collect --config configs/libero_gt_v1.yaml --limit 200 --resume
```

它继续达到总计 200 张。若希望扩大到 1,000 张试验，可将限制改为：

```bash
python -m robot_synth collect --config configs/libero_gt_v1.yaml --limit 1000 --resume
```

### 3.2 全量采集中断后继续

```bash
python -m robot_synth collect --config configs/libero_gt_v1.yaml --resume
```

保持原来的输入、输出和配置。已通过完整性校验的样本会复用。

每个状态目录最后才写入 `COMMIT.json`。没有完整提交标记的目录不计入结果；程序只清理自己拥有的未完成目录。已提交文件损坏时会报错，不自动覆盖。

### 3.3 想改分辨率、相机、数量或采样规则

建立新版本，保留旧结果：

```bash
cp configs/libero_gt_v1.yaml configs/libero_gt_v2.yaml
```

在 VS Code 打开新文件，先修改下列字段，再修改所需参数。以下是**局部修改示例，不是完整配置文件**：

```yaml
name: libero_gt_v2
output:
  root: /mnt/data/opendatasets/LIBERO_GT_v2
  run_dir: ../runs/libero_gt_v2
  scratch: /tmp/robot-synth/libero_gt_v2
```

后续所有命令都带新配置，例如：

```bash
python -m robot_synth inspect --config configs/libero_gt_v2.yaml
python -m robot_synth plan --config configs/libero_gt_v2.yaml
python -m robot_synth doctor --config configs/libero_gt_v2.yaml
python -m robot_synth collect --config configs/libero_gt_v2.yaml --limit 200
```

只调整 `--limit`、是否续采、允许小批量检查、预览图数量，不需要另建数据集。修改 YAML 中影响采样、渲染、GT 或质量规则的内容，需要新计划和新输出。

### 3.4 遇到无限地面报错后使用修复版本

**适用情况：** 已采到 20 张后，报 `Unsupported finite geom type 0`。类型 0 是无限平面，旧代码错误地尝试为它计算有限物体框；不是数据下载失败。

修复后地面仍出现在 RGB、深度中，但不生成地面物体框，实例编号为 0。有限的桌面、物体、机器人和部件仍按原配置标注；未知的其他几何类型仍会报错，不自动生成近似框。

**为什么要换版本：** 源码摘要属于数据集的身份。代码更新后，旧版本的 `--resume` 会拒绝不同的 `profile_id`。不篡改旧元数据，也不把旧的 20 张手工复制进新数据集。

已提供完整新配置 [configs/libero_gt_v1_plane_fix.yaml](configs/libero_gt_v1_plane_fix.yaml)，保留当前用户选择的 `/root/opendatasets` 父目录：

| 配置项 | 新版本的值 |
|---|---|
| `output.root` | `/root/opendatasets/LIBERO_GT_v1_plane_fix` |
| `output.run_dir` | `../runs/libero_gt_v1_plane_fix` |
| `output.scratch` | `/tmp/robot-synth/libero_gt_v1_plane_fix` |

旧目录和旧配置不变；新版本重新收集 200 张，之后目标仍是 train 10,000 张、eval 1,000 张。如果要换回 OSS 路径，应在运行新 `plan` **之前**修改新 YAML 的 `output.root`，后续保持一致。

在工程目录、已激活 `.venv` 的终端，依次执行；每一步通过后再执行下一步：

```bash
# 1. 确认两个真实渲染测试均通过。
MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa \
  python -m pytest -q tests/test_native_renderer.py

# 2. 为新配置生成独立计划。
python -m robot_synth plan --config configs/libero_gt_v1_plane_fix.yaml

# 3. 检查环境，要求 ready: true。
python -m robot_synth doctor --config configs/libero_gt_v1_plane_fix.yaml

# 4. 在新输出目录试采；第一次不加 --resume。
python -m robot_synth collect --config configs/libero_gt_v1_plane_fix.yaml --limit 200

# 5. 检查试采并生成预览。
python -m robot_synth validate --config configs/libero_gt_v1_plane_fix.yaml --allow-partial
python -m robot_synth report --config configs/libero_gt_v1_plane_fix.yaml --max-previews 40
```

人工查看 `runs/libero_gt_v1_plane_fix/quality_report/previews/`，确认正常后继续全量：

```bash
python -m robot_synth collect --config configs/libero_gt_v1_plane_fix.yaml --resume
python -m robot_synth validate --config configs/libero_gt_v1_plane_fix.yaml
python -m robot_synth report --config configs/libero_gt_v1_plane_fix.yaml --max-previews 40
```

如果只是新版本试采中断，使用 `collect --config configs/libero_gt_v1_plane_fix.yaml --limit 200 --resume`。之后所有命令都保持使用这份新配置。

## 四、命令和命令行参数说明

### 4.1 命令怎么写

```text
python -m robot_synth 子命令 [参数]
```

| 子命令 | 作用 | 什么时候运行 |
|---|---|---|
| `inspect` | 清点原始 HDF5 并保存清点缓存 | 准备计划前 |
| `plan` | 固定场景划分、状态清单和配额 | 采集前，每个数据版本建立一次 |
| `doctor` | 检查依赖、渲染库和资产 | 环境准备后、实际采集前 |
| `collect` | 回放状态、生成并保存 GT | 试采或全量采集 |
| `validate` | 检查已生成数据的文件、几何和数量 | 试采后、最终验收时 |
| `report` | 生成 GT 检查图和汇总报告 | 已经有样本时 |

### 4.2 所有子命令通用的参数

| 参数 | 默认值 | 含义 | 能填什么／示例 |
|---|---|---|---|
| `--config` | 工程内 `configs/libero_gt_v1.yaml` | 选择配置文件 | 存在的 YAML 文件路径，例如 `--config configs/libero_gt_v2.yaml` |
| `--input` | YAML 的 `source.root` | 覆盖原始 HDF5 根目录 | 文件夹路径，例如 `--input /mnt/data/opendatasets/LIBERO`；其下应有所选子集目录 |
| `--output` | YAML 的 `output.root` | 覆盖最终 GT 根目录 | 独立结果文件夹路径，例如 `--output /mnt/data/opendatasets/LIBERO_GT_v2` |
| `-h`、`--help` | 不填 | 查看帮助，不执行采集 | 单独作为开关，例如 `python -m robot_synth collect --help` |

`--input`、`--output` 只覆盖对应目录，不会自动修改 `run_dir`、`scratch` 或相机配置。若用了覆盖参数，应在整个流程的相关命令中保持一致。新版本优先复制 YAML 后统一配置，减少遗漏。

参数放在子命令后，例如 `robot_synth collect --config ...`，不要把 `--config` 放在 `collect` 前面。

### 4.3 各子命令专用参数

| 参数 | 用于哪些命令 | 默认值 | 含义与允许值 |
|---|---|---|---|
| `--refresh` | `inspect`、`plan` | 不启用 | 开关，不跟值。重新读取原始数据元信息；不会覆盖已有固定计划。例：`inspect --refresh` |
| `--limit` | `collect` | 不限制，使用完整计划配额 | 希望达到的总图片数量，包含已有结果。正整数，且必须能被相机数整除；默认双相机填 `200`、`1000` 等偶数。大于完整配额时按完整配额收集 |
| `--resume` | `collect` | 不启用 | 开关，不跟值。检查并复用已有结果。例：`collect --resume`；不是 `--resume true` |
| `--allow-partial` | `validate` | 不启用 | 开关，不跟值。允许检查尚未达到完整配额的试采结果；不跳过几何或文件检查 |
| `--max-previews` | `report` | `40` | 最多生成多少张检查图，填正整数，例如 `20`、`40`、`100`；不改变采集数量和 GT |

### 4.4 返回码怎么看

命令结束后可立即执行 `echo $?` 查看：

| 返回码 | 含义 |
|---:|---|
| `0` | 命令正常结束；仍需结合输出字段判断是否达到完整数量 |
| `1` | 执行错误，例如路径、配置、回放或采集异常 |
| `2` | `doctor` 检查不通过，或 `validate` 检查不通过；参数格式错误也可能返回 2 |
| `130` | 用户通过 Ctrl+C 中断 |

`collect --limit 200` 正常结束可以返回 0，但数据集状态仍是 `partial`，因为完整目标是 11,000 张。

## 五、YAML 配置参数说明

配置文件为 [configs/libero_gt_v1.yaml](configs/libero_gt_v1.yaml)，无限地面修复后的独立版本为 [configs/libero_gt_v1_plane_fix.yaml](configs/libero_gt_v1_plane_fix.yaml)。下表默认值按原始方案说明；当前用户已改动的路径见 1.2 节，新版本路径见 3.4 节。YAML 使用空格缩进，布尔值写 `true` 或 `false`；列表写 `[值1, 值2]`；路径建议使用绝对路径。

**YAML 中的相对路径以 YAML 文件所在目录为基准。** 例如配置在 `configs/`，`../runs/libero_gt_v1` 代表工程下的 `runs/libero_gt_v1`。命令行 `--input`、`--output` 的相对路径则以当前终端目录为基准。

### 5.1 基础信息和数据路径

| 配置项 | 默认值 | 含义 | 可填内容 |
|---|---|---|---|
| `schema_version` | `1` | 配置格式版本 | 当前只能填整数 `1` |
| `name` | `libero_gt_v1` | 本次数据配置的名称 | 字符串，例如 `libero_gt_v2`；改名不会自动更改任何路径 |
| `source.root` | `/mnt/data/opendatasets/LIBERO` | 原始数据根目录 | 已存在的文件夹，其下包含子集目录 |
| `source.suites` | `[libero_90, libero_10]` | 使用哪些子集 | 当前只支持名称 `libero_90` 和 `libero_10`，不可重复。单独选择时可能需要调整场景数和数量，建议保持默认 |
| `output.root` | `/mnt/data/opendatasets/LIBERO_GT_v1` | 最终数据保存位置 | 独立且可写的文件夹；首次应为空或尚不存在 |
| `output.run_dir` | `../runs/libero_gt_v1` | 计划、进度、检查报告的位置 | 独立且可写的文件夹；长期保留用于复现 |
| `output.scratch` | `/tmp/robot-synth/libero_gt_v1` | 临时文件的位置 | 空间足够的本地磁盘目录；不能作为唯一的最终数据副本 |

### 5.2 采样参数

“场景组”是归在一起的一组场景任务；“示范／episode”是一次操作记录；“状态”是该记录中的一个时间点。

| 配置项 | 默认值 | 含义 | 可填内容及影响 |
|---|---|---|---|
| `sampling.seed` | `20261020` | 随机种子，控制场景和示范选择 | 整数。相同输入、相同配置可复现相同计划；换种子需建立新版本 |
| `sampling.episodes_per_scene.train` | `20` | 每个训练场景希望选择的示范数量 | 正整数，例如 `20`、`30`。数量增大可增加候选状态；实际受可用示范数量限制 |
| `sampling.episodes_per_scene.eval` | `10` | 每个评测场景希望选择的示范数量 | 正整数，例如 `10`、`20` |
| `sampling.reserve_fraction` | `0.5` | 额外候补状态比例 | 数值 `0`～`3`。`0.5` 表示在目标状态数之外准备约 50% 候补；不会增加最终交付配额 |
| `sampling.min_qpos_distance` | `0.005` | 状态变化初筛阈值 | 大于等于 0 的数值。比较关节／物体广义位置向量的距离，包含不同类型参数，不能简单理解为米；越大筛选越严格。`0` 关闭这项阈值过滤，后续相同图像对去重仍生效 |
| `sampling.eval_scenes.KITCHEN` | `2` | 保留多少个厨房场景组用于评测 | 正整数，需小于可用厨房场景组数，给训练集留出组 |
| `sampling.eval_scenes.LIVING_ROOM` | `1` | 保留多少个客厅场景组用于评测 | 正整数，需小于可用客厅场景组数 |
| `sampling.eval_scenes.STUDY` | `1` | 保留多少个书房场景组用于评测 | 正整数，需小于可用书房场景组数 |
| `sampling.eval_scene_ids` | `[]` | 手动指定评测场景编号 | 空列表表示按种子选择。非空时需列全三类场景，并分别满足上面 `2/1/1` 的组数；编号必须存在 |

手动指定评测组的填写示例：

```yaml
sampling:
  # 其他 sampling 字段仍保留。
  eval_scene_ids:
    - KITCHEN_SCENE2
    - KITCHEN_SCENE8
    - LIVING_ROOM_SCENE6
    - STUDY_SCENE3
```

这是局部示例。不要只填一个厨房编号而遗漏其他类别，否则会因组数不符而报错。

### 5.3 图片配额参数

以下每一项都按**相机图片张数**计数，必须为正整数，并能被相机数整除。默认两个相机，所以各项填偶数。

| 配置项 | 默认值 | 含义 |
|---|---:|---|
| `sampling.quotas.train.KITCHEN` | `5000` | 训练集厨房图片数量 |
| `sampling.quotas.train.LIVING_ROOM` | `3000` | 训练集客厅图片数量 |
| `sampling.quotas.train.STUDY` | `2000` | 训练集书房图片数量 |
| `sampling.quotas.eval.KITCHEN` | `500` | 评测集厨房图片数量 |
| `sampling.quotas.eval.LIVING_ROOM` | `250` | 评测集客厅图片数量 |
| `sampling.quotas.eval.STUDY` | `250` | 评测集书房图片数量 |

训练三项相加为 10,000，评测三项相加为 1,000。程序把每类配额分配给该类场景组；填写的数量还需要有足够候选状态才能完成。当前配置不支持用 `0` 表示跳过某一项。

### 5.4 仿真与相机参数

| 配置项 | 默认值 | 含义 | 可填内容及注意事项 |
|---|---|---|---|
| `simulation.backend` | `mujoco_xml` | 回放方式说明 | 当前只实现 `mujoco_xml`。这是保留说明字段，填写其他值不会切换到其他仿真后端 |
| `simulation.mujoco_version` | `3.2.7` | 要求的 MuJoCo 版本 | 本工程当前目标版本为字符串 `3.2.7`。只改此字段不会安装新版本，也不代表新版本兼容，建议保持不变 |
| `simulation.gl` | `osmesa` | 渲染后端 | 支持名称 `osmesa`、`egl`、`glfw`。主流程只按 OSMesa 编写；EGL 需要可用 EGL 运行环境，GLFW 通常需要显示环境，切换后要单独验证 |
| `simulation.cameras` | `[agentview, robot0_eye_in_hand]` | 每个状态采集哪些相机 | 非空、不可重复的名称列表。必须是记录 XML 中真实存在的相机；推荐默认两项。相机名只能含英文字母、数字、`_`、`-` |
| `simulation.width` | `512` | 输出图片宽度，单位像素 | 整数 `32`～`4096`；例如 `512` 或 `640`。增大会增加计算和存储开销 |
| `simulation.height` | `512` | 输出图片高度，单位像素 | 整数 `32`～`4096`；例如 `512` 或 `480` |
| `simulation.visual_geom_groups` | `[1]` | 哪些几何组参与渲染和框计算 | 非空整数列表，组号范围 `0`～`5`。当前资产按组 1 取可视几何，不建议随意增加碰撞组 |
| `simulation.max_geom` | `20000` | 渲染场景允许容纳的几何项上限 | 正整数；通常保持默认。过小可能无法容纳场景，过大会增加资源占用 |
| `simulation.asset_roots.libero` | `../third_party/LIBERO/libero/libero/assets` | LIBERO 资产根目录 | 存在的资产目录；可改成已有兼容资产的位置 |
| `simulation.asset_roots.robosuite` | `../third_party/robosuite/robosuite/models/assets` | 机器人等 robosuite 资产根目录 | 存在的资产目录；不能把整个仓库目录误填为 assets 目录 |

更换相机数量后，图片与状态数量的换算、`--limit` 的整除要求以及六项配额的整除要求都会随之变化。

### 5.5 Ground truth 参数

| 配置项 | 默认值 | 含义 | 可填内容及影响 |
|---|---|---|---|
| `groundtruth.include_robot` | `true` | 是否把机器人作为独立标注实例保存 | 布尔值 `true`／`false`。`false` 时机器人仍可出现在图像中，但不作为标注实例保存 |
| `groundtruth.include_support` | `true` | 是否保存程序识别到的有限 table／floor 支撑实例 | 布尔值 `true`／`false`。仅对识别到的顶层 body 有效；无限平面始终作为背景，不生成物体框或实例 ID |
| `groundtruth.include_articulated_parts` | `true` | 是否把带关节的子部件单独保存 | 布尔值 `true`／`false`。子部件默认类别为 `articulated_part`，保留父子层级，不自动猜成抽屉或柜门 |
| `groundtruth.depth_background` | `invalid` | 无几何背景深度的规则说明 | 当前仅按 `invalid` 使用。这是保留说明字段；代码固定将无效深度置 0，并用 valid 掩码排除，改成其他字符串不会切换处理方式 |
| `groundtruth.category_aliases` | 配置中的名称映射表 | 将资产原始类别映射为统一类别 | YAML 字典，格式 `原始类别: 目标类别`；例如 `akita_black_bowl: bowl`。没有映射的类别保留原始名称 |

映射表示例：

```yaml
groundtruth:
  # 其他 groundtruth 字段仍保留。
  category_aliases:
    akita_black_bowl: bowl
    red_coffee_mug: mug
    wine_bottle: bottle
    wooden_cabinet: cabinet
```

### 5.6 质量检查参数

| 配置项 | 默认值 | 含义 | 可填内容及影响 |
|---|---|---|---|
| `validation.geometry_tolerance_m` | `0.002` | 检查深度点是否落在完整物体框内时允许的几何误差，单位米 | 正数；默认 0.002 米即 2 毫米。减小更严格；不要仅为消除报错而放大，应先检查标定和几何 |
| `validation.minimum_owned_pixels` | `32` | 每张相机图像要求的物体可见像素总数下限 | 正整数，例如 `32`、`64`。只计物体根实例，不计机器人和支撑物；默认要求每个视角均满足 |
| `validation.max_failures` | `3` | 连续质量拒绝多少个状态后停止 | 正整数，例如 `3`。拒绝后先尝试候补；达到阈值停止并记录原因。状态恢复或渲染异常会直接停止 |

### 5.7 安装脚本参数：环境变量

这些只作用于 `scripts/bootstrap_runtime.sh`，不属于采集命令行参数。

| 变量 | 默认值 | 含义 | 可填内容 |
|---|---|---|---|
| `CAPTURE_PYTHON` | `python3` | 用哪个 Python 创建虚拟环境 | Python 命令名或解释器完整路径；建议 Python 3.12 |
| `CAPTURE_VENV` | 工程下 `.venv` | 虚拟环境创建位置 | 独立目录，不要填写现有模型的 PPU venv |
| `LIBERO_REF` | `master` | 首次下载后检出的 LIBERO 引用 | 该仓库存在的分支、标签或提交哈希 |
| `ROBOSUITE_REF` | `v1.4.0` | 首次下载后检出的 robosuite 引用 | 该仓库存在的分支、标签或提交哈希，默认资产基线为 v1.4.0 |

使用例子：

```bash
CAPTURE_PYTHON=python3.12 \
CAPTURE_VENV=/mnt/workspace/3dmood/robot-synth/.venv \
LIBERO_REF=master \
ROBOSUITE_REF=v1.4.0 \
  bash scripts/bootstrap_runtime.sh
```

已有资产仓库不会因重跑脚本而自动切换引用；若它们已经存在，需核对脚本输出的实际提交。首次验证成功后记录提交哈希，用于后续复现。

本流程还有两个渲染环境变量：`MUJOCO_GL=osmesa`、`PYOPENGL_PLATFORM=osmesa`。真实渲染测试使用它们；实际 `collect` 会根据 YAML 的 `simulation.gl` 在导入 MuJoCo 前设置对应值。

## 六、生成的文件怎么看

### 6.1 最终数据目录

```text
LIBERO_GT_v1/
├── dataset_info.json              本次配置、依赖、源码和资产身份
├── scene_split.json               固定的训练／评测划分
├── sampling_plan.json             完整状态和候补清单
├── categories.json                类别名称、原始名称和实例类型
├── manifest.json                  采集进度、已接受／拒绝状态和错误
├── provenance/
│   └── <episode>.xml.gz           原始场景 XML
├── train/<scene>/<state_id>/
│   ├── state.npy                  实际恢复的仿真状态
│   ├── agentview/
│   │   ├── rgb.png                RGB 图像
│   │   ├── depth.npz              米制深度与有效性掩码
│   │   ├── instance.png           实例 ID 图
│   │   └── annotation.json        对应的相机标定和物体真值
│   ├── robot0_eye_in_hand/         腕部相机的同类文件
│   ├── PENDING.json               本工具的写入所有权记录
│   └── COMMIT.json                完整提交标记和文件校验值
└── eval/<scene>/<state_id>/        与 train 相同的结构
```

`state_id` 是程序生成的唯一标识，用于追溯和防止重复提交。不要通过手工复制状态目录来补足数量。

### 6.2 深度和实例分割怎么看

`depth.npz` 包含：

| 字段 | 类型 | 含义 |
|---|---|---|
| `z_m` | float32 二维数组 | 相机 Z 方向深度，单位米；不是到相机中心的欧氏距离 |
| `valid` | bool 二维数组 | `true` 表示深度有效；无效位置的 `z_m` 为 0 |

`instance.png` 是 uint16 实例编号图，不是普通彩色预览。编号 0 表示未标注的世界背景，也包括无限地面；这些位置仍可能有正常 RGB 和有效深度。各编号的对应实例记录在 `annotation.json` 中。

### 6.3 标注中的主要字段

| 字段 | 含义 |
|---|---|
| `intrinsics` | 当前图片的 3×3 相机内参 |
| `T_world_from_camera` | 将相机坐标点变换到世界坐标的 4×4 矩阵 |
| `T_camera_from_world` | 将世界坐标点变换到相机坐标的 4×4 矩阵 |
| `category`、`raw_category` | 映射后的类别名和资产原始类别名 |
| `instance_id`、`parent_instance_id` | 当前实例及其父实例；子部件保留层级 |
| `center_world`、`center_camera` | 完整物体框在世界／相机坐标下的中心 |
| `size_xyz` | 沿框自身三个轴的完整尺寸，单位米 |
| `R_world_from_box`、`R_camera_from_box` | 从框的局部轴系旋转到世界／相机轴系的矩阵 |
| `center_body` | 框中心相对于仿真 body 原点的偏移 |
| `segmentation_ids` | 组成该实例掩码的 ID；根物体可能包含子部件 ID |
| `visible`、`visible_pixels` | 是否可见及可见像素数；不可见物体也可保留完整几何记录 |
| `bbox_visible_xyxy` | 根据当前可见实例掩码得到的 2D 框 |
| `bbox_projected_xyxy` | 完整 3D 框的投影范围 |
| `bbox_projected_clipped_xyxy` | 将投影范围截到画面内的结果 |
| `box_truncated` | 完整 3D 框投影是否被画面或近裁剪面截断 |
| `occlusion_fraction` | 当前为 `null`：尚未做无遮挡重渲染，不输出推测的精确遮挡率 |
| `provenance` | 原始文件、轨迹、状态索引等来源信息 |

坐标约定：世界坐标为 MuJoCo Z 向上；相机坐标为 OpenCV x 向右、y 向下、z 向前。图片左上边缘为原点，像素 `(v,u)` 中心为 `(u+0.5,v+0.5)`；2D 框采用右下端点不包含的 xyxy。

完整物体框来自当前可视几何，在根 body 轴系内计算包围范围，不是最小体积 OBB；活动部件变换后按当前外形范围更新。世界背景贴图不会被自动推断为具有真实几何的家电或柜子。

## 七、可选：使用 Docker 运行

**选本节时，用下面的容器命令替代第二部分中的环境安装和直接 Python 命令。** 数据清点、计划、检查、试采、验收、全量采集的顺序仍相同。

### 7.1 检查 Docker 并构建镜像

```bash
cd /mnt/workspace/3dmood/robot-synth
docker version
docker build -t robot-synth:libero-gt-v1 .
```

`docker version` 需要能访问 Docker 服务；只有客户端、无法连接服务时不能在该终端运行容器。构建需要能够下载基础镜像、系统包、Python 包和 GitHub 资产。

镜像内工程路径与主机相同。镜像包含 Python 环境、OSMesa 和资产；不要用主机的整个工程目录挂载覆盖它。

### 7.2 准备挂载目录和两个快捷命令

下面整个代码块复制执行一次。每次新开终端需要重新定义函数。

```bash
cd /mnt/workspace/3dmood/robot-synth
mkdir -p runs /tmp/robot-synth /mnt/data/opendatasets/LIBERO_GT_v1

rsynth_python() {
  docker run --rm \
    -e MUJOCO_GL=osmesa \
    -e PYOPENGL_PLATFORM=osmesa \
    -v /mnt/data/opendatasets/LIBERO:/mnt/data/opendatasets/LIBERO:ro \
    -v /mnt/data/opendatasets/LIBERO_GT_v1:/mnt/data/opendatasets/LIBERO_GT_v1 \
    -v /mnt/workspace/3dmood/robot-synth/runs:/mnt/workspace/3dmood/robot-synth/runs \
    -v /tmp/robot-synth:/tmp/robot-synth \
    --entrypoint /mnt/workspace/3dmood/robot-synth/.venv/bin/python \
    robot-synth:libero-gt-v1 "$@"
}

rsynth() {
  rsynth_python -m robot_synth "$@"
}
```

这些函数只是终端快捷写法，没有修改采集程序。

| Docker 参数 | 作用／能填什么 |
|---|---|
| `-t robot-synth:libero-gt-v1` | 构建时给镜像命名；可换名称，但后续运行也要使用同名镜像 |
| `--rm` | 容器退出后删除临时容器；挂载出来的数据和运行记录仍保留 |
| `-e 名称=值` | 给容器设置环境变量，这里选择 OSMesa |
| `-v 主机目录:容器目录:ro` | 把原始数据以只读方式放进容器 |
| 其他三个 `-v` | 让最终结果、运行记录和缓存写回主机；默认路径保持内外一致 |
| `--entrypoint .../python` | 使用镜像内 Python，既能运行采集命令，也能运行 pytest |
| `--build-arg LIBERO_REF=...` | 构建时可指定 LIBERO 分支、标签或提交；例如放在 `docker build` 命令中 |
| `--build-arg ROBOSUITE_REF=...` | 同理，指定 robosuite 引用，默认 v1.4.0 |

这些挂载按默认配置编写。若更改输入、输出、缓存或运行目录，也要调整相应挂载；若使用自定义 YAML，还要确保该文件进入容器。镜像构建时会复制配置，主机修改后需要重建或单独挂载该配置文件。

### 7.3 按顺序执行容器内流程

```bash
# 1. 清点原始数据。
rsynth inspect

# 2. 固定采样计划。
rsynth plan

# 3. 检查运行条件；必须 ready: true。
rsynth doctor

# 4. 真实渲染测试；必须 2 passed，不能是 skipped。
rsynth_python -m pytest -q tests/test_native_renderer.py

# 5. 试采 200 张。
rsynth collect --limit 200

# 6. 检查试采数据并生成预览，人工查看后再继续。
rsynth validate --allow-partial
rsynth report --max-previews 40

# 7. 继续到完整配额。
rsynth collect --resume

# 8. 完整验收和报告。
rsynth validate
rsynth report --max-previews 40
```

每步的预期结果和检查内容与第二部分相同。不要不看中间结果就一次性执行到最后。

## 八、常见问题处理

| 现象／错误 | 原因和处理 |
|---|---|
| `No module named robot_synth` | 先进入工程目录并激活正确环境；检查安装脚本是否完成 |
| `No module named h5py`、`yaml` 等 | 当前解释器缺依赖，检查 `which python`，回到步骤 3～4 |
| 连接 `127.0.0.1:17890` 失败、DNS 或 Git 下载失败 | 终端网络／代理不可用。先恢复该环境网络，再重跑安装；不需要重下 OSS 原始 HDF5 |
| `OSMesa is missing` 或图形库找不到 | 系统库未安装、镜像不匹配；检查步骤 2 或改用 Docker 路线 |
| 渲染测试只有 `skipped` | 测试未执行。按跳过原因补依赖，不进入采集 |
| `Runtime is not ready` | 打开 `runs/libero_gt_v1/doctor.json`，逐项处理 `ok: false` |
| `Cannot uniquely resolve asset` | 检查具体资产文件、两个 asset_roots 和版本；不能用任意同名文件替换 |
| `Unsupported finite geom type 0` | 旧版尝试给无限地面计算物体框；更新后按 3.4 节使用新配置重新试采，旧的 20 张保留 |
| `Dataset identity differs (profile_id)` | 源码、依赖或资产与旧结果不同。不要绕过校验；无限地面修复后按 3.4 节使用独立新版本 |
| `Existing plan differs`、`Configuration differs` | 当前配置或原始数据与固定计划不一致。恢复原配置，或按第三部分建立新版本 |
| `Dataset already exists; use --resume` | 输出已有运行记录，使用相同配置加 `--resume` |
| 输出非空且没有本工具记录 | 换一个新的独立输出目录，避免混入其他文件 |
| `--limit` 提示必须整除 | 按相机数填写；默认两个相机应填偶数 |
| `Not enough non-static states` | 当前示范数量或变化状态不足；新版本可增加 episodes_per_scene，先检查阈值是否合理 |
| `Candidate pool exhausted` | 冻结候补已耗尽，当前配额未完成。保留结果，检查拒绝原因；需要改变采样规则时建立新版本 |
| `Too many consecutive GT quality failures` | 连续质量不通过。查看进度中的 rejected／errors，检查几何和图像，不能靠重复启动凑数量 |
| `validate` 报数量不足 | 试采时加 `--allow-partial`；最终验收则继续采集到完整配额 |
| 已提交样本校验失败 | 文件缺失或内容损坏，程序会拒绝复用；先查明并恢复损坏文件，不静默覆盖 |
| 开始续采时长时间读取 | 程序正在核对已有结果、文件摘要和资产身份；OSS 读取可能较慢 |
| `collect` 状态是 `partial` | 可能只是试采限制尚未覆盖全量；去掉 `--limit` 并加 `--resume` 才继续到完整目标 |

本机可单独运行 `python -m pytest -q` 检查程序。部分测试使用明确标记的临时合成夹具；只有真实渲染测试和实际 LIBERO 试采通过，才能说明真实采集链路可用。
