# Cosmos-Transfer2.5：LIBERO 仿真图片写实化操作说明书

本工程把已有的 LIBERO RGB 图片通过 Cosmos-Transfer2.5 转换成更接近真实相机的外观，并保留对应的仿真状态、深度、实例分割、相机标定和 3D 标注。输入是已经采集好的 GT 数据集，不需要重新录制机器人动作。

**第一次使用，请按第二部分的步骤 1～12 顺序执行。每一步成功后，再进入下一步。** 默认配置已对应当前实例和现有 LIBERO 数据，不必先改参数。

> 当前状态：上游源码已移植；适配层测试、计划及真实数据的 200 张控制图准备已完成。独立环境已创建，但依赖安装在 `hatchling` 下载处失败；后台进程也无法访问 PPU 驱动。**目前没有生成任何 Cosmos 写实化图片，不能视为完整推理已验证。** 用户原生终端已验证 PPU 的基础算子可运行。详细记录见 [IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md)。

## 目录

- [一、先了解输入、输出和环境](#一先了解输入输出和环境)
- [二、按顺序执行：安装到完成写实化](#二按顺序执行安装到完成写实化)
- [三、断点恢复和重新建版](#三断点恢复和重新建版)
- [四、命令及命令行参数](#四命令及命令行参数)
- [五、YAML 配置参数](#五yaml-配置参数)
- [六、输出文件与质量检查](#六输出文件与质量检查)
- [七、目录结构和上游项目](#七目录结构和上游项目)
- [八、常见问题](#八常见问题)

## 一、先了解输入、输出和环境

### 1.1 本工程做什么

```text
已采集的 RGB + 米制深度 + 实例分割 + 相机及 3D GT
                       ↓
                 准备控制图和生成参数
                       ↓
              Cosmos 基础模型单帧写实化
                       ↓
         新 RGB + 继承的仿真 GT + 来源和运行记录
```

默认使用公开的 **2B 基础模型、深度控制、单帧、单卡**。支持改为边缘控制或深度加边缘；首版不使用蒸馏模型、机器人多视角专用模型或视频连续生成。

**生成模型可能改变物体轮廓、数量或遮挡关系。** 输出中的 GT 是继承的仿真标注，不会因为文件检查通过就自动成为新 RGB 的严格物理真值。必须查看对照图，筛掉对应关系不可靠的生成结果。

### 1.2 数量按相机图片计数

默认两个相机：

| 名称 | 视角 |
|---|---|
| `agentview` | 外部观察机械臂和物体 |
| `robot0_eye_in_hand` | 机械臂腕部 |

同一状态的两个视角分别调用生成模型，共享状态随机种子和提示词；**不保证双视角生成内容严格一致**。程序在两个相机都成功后才提交整个状态。

- `--limit 2`：1 个状态、2 张图片，用于检查能否出图。
- `--limit 200`：100 个状态、200 张图片，用于小批量检查。
- 不填 `--limit`：处理输入计划中的全部状态。
- 当前输入为 5,500 个状态、11,000 张：train 10,000，eval 1,000。

### 1.3 默认路径

| 用途 | 路径 | 保存什么 |
|---|---|---|
| 工程 | `/mnt/workspace/3dmood/realisation` | 本说明书、上游源码和适配代码 |
| 输入 GT | `/root/opendatasets/LIBERO_GT_v1_plane_fix` | 只读的原 RGB、深度和仿真标注 |
| 最终结果 | `/root/opendatasets/LIBERO_REALISM_cosmos25_v1` | 生成 RGB 和继承的 GT |
| 运行记录 | `runs/realism/cosmos25_libero_v1` | 计划、进度、检查结果和对照图 |
| 临时文件 | `/tmp/robot-realism/cosmos25/staging` | 深度／边缘控制图、原生生成结果和日志 |
| 独立环境 | `/tmp/robot-realism/cosmos25/venv` | Python 依赖 |
| 权重和缓存 | `/tmp/robot-realism/cosmos25/cache` | Hugging Face 权重、Torch 缓存、pip 下载缓存 |
| 默认配置 | `configs/realism/cosmos25_libero_v1.yaml` | 输入输出及生成参数 |

输入、最终结果、运行记录、临时文件四个目录必须互不包含。环境和权重不写进输入或输出目录。

`/root/opendatasets` 是否挂载 OSS，请在当前实例确认，不能仅凭目录名判断。若要输出到 OSS，可在生成计划前更改 `output.root`。本地临时目录与权重缓存仍建议使用本地磁盘。

`/tmp` 的环境和缓存可能在实例重建后丢失。更换到其他持久化路径时，在第一次生成计划前同步修改安装参数及 YAML。当前 workspace 仅剩约 11 GB，不适合同时存环境和模型；`/tmp` 与 `/root` 所在的根盘是同一块空间，不能把两者空闲容量相加。

### 1.4 为什么必须单独安装环境

当前镜像是 PPU-ZW810E 96 GB，带有可工作的 PPU PyTorch 2.9.0、Transformer Engine、FlashAttention 和 Triton。它们与 NVIDIA 官方预编译包不能直接等同。

本安装器会：

1. 读取镜像现有 SDK 软件版本。
2. 建立可继承这些 SDK 包的专用虚拟环境。
3. 用约束文件锁定 SDK 包及已安装的 CUDA 运行库。
4. 把其余 Python 依赖装进专用环境。
5. 安装本地 Cosmos 包并检查受保护版本是否保持一致。

采集工程的 `.venv` 使用 NumPy 1.26；本工程安装 NumPy 2.2.6。**不要把本工程依赖安装到采集环境、模型微调环境或系统 Python 中。**

## 二、按顺序执行：安装到完成写实化

### 步骤 1：进入工程，确认数据和磁盘

**这一步做什么：** 确认路径存在，查看加速卡和磁盘。

```bash
cd /mnt/workspace/3dmood/realisation

ls configs/realism/cosmos25_libero_v1.yaml
ls /root/opendatasets/LIBERO_GT_v1_plane_fix/dataset_info.json
ls /root/opendatasets/LIBERO_GT_v1_plane_fix/manifest.json

ppu-smi
df -h /mnt/workspace /tmp /root
```

**正常结果：** 文件存在；原生 DSW 终端能看到 PPU-ZW810E。当前默认输入已经通过采集工程的完整验证。

**不正常时：** 数据不存在，先检查挂载和配置；原生终端看不到卡，先处理实例驱动或设备分配。不要通过安装 NVIDIA 驱动来替换当前 PPU 驱动。

### 步骤 2：只检查镜像 SDK

**这一步做什么：** 确认安装器能识别基础镜像的软件包，不下载或安装任何内容。

```bash
/usr/local/bin/python scripts/bootstrap_cosmos25_ppu.py --check
```

**正常结果：** 输出类似：

```json
{
  "base_python": "/usr/local/bin/python",
  "sdk": {
    "torch": "2.9.0",
    "torchvision": "0.24.0",
    "transformer-engine": "2.8.0+902761ed",
    "flash-attn": "2.7.4.post1",
    "triton": "3.5.0+gitcc5446cf"
  }
}
```

还可能列出镜像已有的其他 CUDA 包。版本以实际镜像为准。

**不正常时：** 提示缺少 SDK 包，应先选择与 PPU 匹配的镜像。不要用普通 PyPI 的 torch 替代。

### 步骤 3：建立并安装专用环境

**这一步做什么：** 创建默认虚拟环境，安装受 SDK 约束的 Python 依赖及本地 Cosmos 包。

```bash
/usr/local/bin/python scripts/bootstrap_cosmos25_ppu.py
```

默认使用通用包源 `https://pypi.org/simple`。若此实例需要指定其他可访问的通用 PyPI 镜像，使用：

```bash
/usr/local/bin/python scripts/bootstrap_cosmos25_ppu.py \
  --index-url https://mirrors.aliyun.com/pypi/simple/
```

**正常结果：** 结尾显示 `Installation complete`，并生成：

```text
/tmp/robot-realism/cosmos25/venv/realisation-owner.json
/tmp/robot-realism/cosmos25/venv/sdk-constraints.txt
/tmp/robot-realism/cosmos25/venv/requirements-installed.txt
```

**不正常时：** 当前后台安装曾报 `No matching distribution found for hatchling`，因此环境尚未完成。检查包源可访问性，换可用通用源后重跑本步骤；安装器会识别自己创建的环境。没有生成 `requirements-installed.txt`，不能据此认为依赖已经装完。

安装器必须由基础镜像 Python 执行。即使终端已经激活了其他 `.venv`，本步骤仍使用上面的 `/usr/local/bin/python`。

这里有意继承基础镜像 SDK；pip 可能提示某些系统包位于环境外，无法卸载。这通常表示它们被保留。以安装是否完成、SDK 版本核验和下一步 `doctor` 为准。

### 步骤 4：激活环境，准备权重下载权限

**这一步做什么：** 后续使用正确解释器，并准备 Hugging Face 模型访问权限。

```bash
cd /mnt/workspace/3dmood/realisation
source /tmp/robot-realism/cosmos25/venv/bin/activate

which python
python --version
python -m robot_realism --help
```

**正常结果：** `which python` 指向该虚拟环境，帮助中列出七个命令。

首次使用 Hugging Face 权重时，在浏览器登录对应账号并确认模型要求的访问条款。若需要登录命令：

```bash
export HF_HOME=/tmp/robot-realism/cosmos25/cache/huggingface
hf auth login
```

输入你自己的 Token；不需要把 Token 写进 YAML、README 或发给其他人。若已使用环境变量或已有有效登录，可以跳过重复登录。权重会在首次真实生成时按上游逻辑下载，还可能下载文本编码器、图像编码器和检查模型，不能只按“2B 主模型”的大小估算全部缓存。

**以后每次打开新终端，先进入工程并重新激活这个环境。**

### 步骤 5：检查实际运行条件

**这一步做什么：** 检查输入、依赖、设备及几个小型算子，不下载完整模型。

```bash
python -m robot_realism doctor \
  --config configs/realism/cosmos25_libero_v1.yaml
```

**可以继续的条件：** 输出 `ready: true`，包括 BF16、SDPA、TE RMSNorm、128 维头的 FlashAttention 和 Cosmos 配置导入通过。

详细结果在：

```text
runs/realism/cosmos25_libero_v1/doctor.json
```

**不正常时：** 查看 `checks` 和 `runtime.checks` 中 `ok: false` 的项目。`cuda_device: false` 代表该进程不可见加速卡；`cosmos-cuda`、`tyro` 等缺失代表步骤 3 没完成。

`ready: true` 只表示小算子和导入检查通过，**不代表已经加载权重或成功出图**。必须继续步骤 8。

### 步骤 6：固定源样本和输出版本

**这一步做什么：** 读取源数据的已提交状态，固定相机、样本顺序和配额。按场景交替取样，让小批量尽量覆盖多个场景。

```bash
python -m robot_realism plan \
  --config configs/realism/cosmos25_libero_v1.yaml
```

**正常结果：** `states: 5500`，`targets.train: 10000`，`targets.eval: 1000`。计划保存在 `runs/realism/cosmos25_libero_v1/plan.json`。

已有同一计划会复用，不自动覆盖。更改输入输出、相机、提示词或生成规则，需要按第三部分建立新版本。

### 步骤 7：先准备一个状态的控制图

**这一步做什么：** 核对源文件摘要，将米制深度转换成模型需要的近白远黑控制图，并保存单帧生成参数。

```bash
python -m robot_realism prepare \
  --config configs/realism/cosmos25_libero_v1.yaml \
  --limit 2
```

**正常结果：** `prepared_states: 1`、`prepared_images: 2`。本步骤不使用生成模型，也不产生写实化图片。

查看：

```text
runs/realism/cosmos25_libero_v1/prepared.json
/tmp/robot-realism/cosmos25/staging/<plan_id>/<state_id>/<camera>/depth.png
/tmp/robot-realism/cosmos25/staging/<plan_id>/<state_id>/<camera>/spec.json
```

`prepared.json` 记录具体路径。控制图按有效像素的截断逆深度逐图归一化；无效位置为黑色，不是新的米制深度 GT。原 `depth.npz` 保持不变。

### 步骤 8：真实生成两张图片

**这一步做什么：** 加载权重，在同一状态的两个相机视角上分别生成，成功后一起提交。

```bash
python -m robot_realism render \
  --config configs/realism/cosmos25_libero_v1.yaml \
  --limit 2
```

`render` 会重新检查运行条件并自动准备所需控制图，不要求手动操作上一步的 JSON。一个工作进程加载一次模型，顺序执行待处理状态。

**正常结果：** 两张图片实际落盘，输出 `accepted_images: 2`。它们在最终输出目录，`runs` 中保存运行记录。

首次运行包含下载、加载模型和计算权重摘要，耗时不能当作稳定的每图生成速度。观察日志和 PPU 占用，暂时不要直接启动 11,000 张。

**不正常时：** 先看 `runs/realism/cosmos25_libero_v1/worker.log`，其中也记录模型初始化前的错误。还可查看 `worker_errors.jsonl` 和临时目录的 `native_output/console.log`、`debug.log`。依赖或设备失败不会生成占位图片；检查模型阻止生成、权重下载失败或模型异常都会停止，不会悄悄补假图。

若第一次已创建输出记录但中途失败，处理原因后用同一命令加 `--resume` 重试。

### 步骤 9：检查文件，再看对照图

**这一步做什么：** 自动检查完整性和来源关系，生成可人工查看的对照图。

```bash
python -m robot_realism validate \
  --config configs/realism/cosmos25_libero_v1.yaml \
  --allow-partial

python -m robot_realism report \
  --config configs/realism/cosmos25_libero_v1.yaml \
  --max-previews 2
```

**正常结果：** `valid: true`、`complete: false`。打开：

```text
runs/realism/cosmos25_libero_v1/quality_report/report.md
runs/realism/cosmos25_libero_v1/quality_report/previews/
```

每张对照图分三列：原 RGB、生成 RGB、原实例边界及原物体框叠加生成 RGB。绿色是源分割边界，黄色是可见物体的源 3D 框投影。

重点看物体有没有增删、碗口和杯柄有没有变形、边界是否移动、遮挡是否改变，以及同一物体在两个视角是否仍一致。**`visual_gt_alignment_certified: false` 是有意保留的结果，自动文件检查不能证明生成 RGB 的三维真值正确。**

### 步骤 10：扩大到总计 200 张

**这一步做什么：** 复用已提交状态，扩大测试范围。

```bash
python -m robot_realism render \
  --config configs/realism/cosmos25_libero_v1.yaml \
  --limit 200 \
  --resume

python -m robot_realism validate --config configs/realism/cosmos25_libero_v1.yaml --allow-partial
python -m robot_realism report --config configs/realism/cosmos25_libero_v1.yaml --max-previews 40
```

**正常结果：** 最终累计 200 张，而不是另外新增 200 张。当前 20 个场景组按顺序交替取样，前 100 个状态会覆盖这 20 组。

这些对照图没有问题后，再进入全量。若要修改控制方式、深度范围或提示词，不续写旧版本，按第三部分建立新版本。

### 步骤 11：继续处理完整输入

**这一步做什么：** 去掉限制，使用同一计划继续到全部输入状态。

```bash
python -m robot_realism render \
  --config configs/realism/cosmos25_libero_v1.yaml \
  --resume
```

默认输入完整时，目标为 11,000 张。输入中没有的状态不会凭空合成。

不要同时运行两个写入同一输出目录的任务；生成过程中不要改源码、依赖、权重或生成参数。已提交文件会校验后复用，损坏文件会报错，不静默覆盖。

### 步骤 12：完整检查和保存报告

**这一步做什么：** 检查全部计划是否处理完毕，并更新对照报告。

```bash
python -m robot_realism validate --config configs/realism/cosmos25_libero_v1.yaml
python -m robot_realism report --config configs/realism/cosmos25_libero_v1.yaml --max-previews 40
python -m robot_realism status --config configs/realism/cosmos25_libero_v1.yaml
```

**正常结果：** `valid: true`、`complete: true`，`counts` 为 train 10,000、eval 1,000。最终检查不加 `--allow-partial`。

完整数量与文件检查通过以后，仍要按样本外观质量筛选。完整数据量不是模型效果提高的证明，也不是生成图与 GT 严格对应的证明。

## 三、断点恢复和重新建版

### 3.1 试生成中断后继续

假设目标仍是总计 200 张：

```bash
python -m robot_realism render --config configs/realism/cosmos25_libero_v1.yaml --limit 200 --resume
```

全量中断后：

```bash
python -m robot_realism render --config configs/realism/cosmos25_libero_v1.yaml --resume
```

`COMMIT.json` 是完整状态的提交标记。一个状态必须包含全部配置相机才计数。未完成目录只有携带本工具匹配的 `PENDING.json` 所有权记录，程序才会清理后重试；其他目录拒绝覆盖。

已经提交的状态不会因再次运行而重复计数。大于 200 张时使用 `--limit 200` 不会删除已有结果。

### 3.2 更换提示词、控制方式、输入或生成参数

先复制配置：

```bash
cp configs/realism/cosmos25_libero_v1.yaml configs/realism/cosmos25_libero_v2.yaml
```

编辑新 YAML，先改这些字段，再改所需参数。以下只是局部示例，不是完整配置：

```yaml
name: libero_cosmos25_v2
output:
  root: /root/opendatasets/LIBERO_REALISM_cosmos25_v2
  run_dir: ../../runs/realism/cosmos25_libero_v2
  scratch: /tmp/robot-realism/cosmos25/staging_v2
```

然后使用新配置依次 `plan`、`doctor`、`prepare --limit 2`、`render --limit 2`、检查和扩展。新输出第一次不加 `--resume`。

`--limit`、`--resume`、`--allow-partial` 和 `--max-previews` 不改变生成规则，可以按需要调整。源码、运行依赖、权重或生成规则变化后，不绕过身份校验混写旧结果。

## 四、命令及命令行参数

### 4.1 七个子命令

统一格式是 `python -m robot_realism 子命令 [参数]`。请在本工程目录执行；安装器安装的是上游 Cosmos 包，适配入口从当前工程目录加载。

| 子命令 | 作用 | 是否调用生成模型 |
|---|---|---|
| `doctor` | 检查输入、运行环境和小算子 | 否，不下载完整权重 |
| `plan` | 固定输入、相机、状态顺序和目标数量 | 否 |
| `prepare` | 核验输入并生成控制图、单帧 JSON | 否 |
| `render` | 自动准备待处理状态并生成、提交图片 | 是 |
| `validate` | 核对文件、来源及继承 GT 是否保持一致 | 否 |
| `report` | 原图、新图、源 GT 叠加对照 | 否 |
| `status` | 查看保存的进度，不启动推理 | 否 |

### 4.2 通用参数

| 参数 | 默认值 | 含义与可填内容 |
|---|---|---|
| `--config` | `configs/realism/cosmos25_libero_v1.yaml` | 已存在的 YAML 文件路径 |
| `--input` | YAML 的 `input.root` | 覆盖输入 GT 文件夹，必须有本采集器格式的 manifest 和 COMMIT |
| `--output` | YAML 的 `output.root` | 覆盖新结果文件夹，不会自动改变 run_dir 或 scratch |
| `-h`、`--help` | 不启用 | 查看帮助，例如 `python -m robot_realism render --help` |

参数放在子命令后。输入／输出覆盖值应在整个流程中保持一致；建立新版本优先复制 YAML 后统一编辑。

### 4.3 专用参数

| 参数 | 使用命令 | 默认值 | 可填内容／含义 |
|---|---|---|---|
| `--limit` | `prepare`、`render` | 不限制，使用完整计划 | 正整数，按相机图片张数计数；须能被相机数整除。双相机填 `2`、`40`、`200` 等偶数；大于输入总量时按输入总量处理 |
| `--resume` | `render` | 不启用 | 开关，不跟 `true`。检查并复用当前版本已提交结果 |
| `--allow-partial` | `validate` | 不启用 | 开关。允许未达到全量数量，仍检查已提交结果；没有生成图片时检查失败 |
| `--max-previews` | `report` | `40` | 正整数，最多生成多少张对照图；不影响采集数量 |

### 4.4 安装器参数

格式：`/usr/local/bin/python scripts/bootstrap_cosmos25_ppu.py [参数]`。

| 参数 | 默认值 | 含义与可填内容 |
|---|---|---|
| `--check` | 不启用 | 只输出 SDK 检查结果，不创建环境或下载包 |
| `--venv` | `/tmp/robot-realism/cosmos25/venv` | 专用环境目录；自定义后，YAML 的 `runtime.python` 必须指向其 `bin/python` |
| `--cache` | `/tmp/robot-realism/cosmos25/cache` | pip、模型缓存根目录；自定义后同步修改 YAML 的 `runtime.cache` |
| `--index-url` | `https://pypi.org/simple` | 可访问的通用 Python 包索引地址；不用于替换 PPU SDK 软件包 |
| `-h`、`--help` | 不启用 | 查看安装参数 |

不要将 `--venv` 指向已有采集或微调环境。安装器有所有权记录，拒绝接管其他环境；不提供自动删除其他环境的功能。

### 4.5 返回码

命令执行后立即 `echo $?` 可以查看：`0` 表示正常结束，`1` 表示执行失败，`2` 表示 doctor／validate 未通过或命令行格式错误，`130` 表示 Ctrl+C 中断。试生成正常结束不等于完整数量已达到。

## 五、YAML 配置参数

默认文件：[configs/realism/cosmos25_libero_v1.yaml](configs/realism/cosmos25_libero_v1.yaml)。缩进使用空格；布尔值写 `true`／`false`。

**YAML 相对路径以配置文件所在目录为基准。** 该文件在 `configs/realism/`，所以 `../../runs/...` 指向工程的 `runs/`。命令行 `--input`、`--output` 的相对路径以当前终端目录为基准。

### 5.1 基础信息和路径

| 配置项 | 默认值 | 含义及能填什么 |
|---|---|---|
| `schema_version` | `1` | 当前配置格式版本，只填整数 `1` |
| `name` | `libero_cosmos25_v1` | 字符串，配置名称；修改名称不会自动改路径 |
| `input.root` | `/root/opendatasets/LIBERO_GT_v1_plane_fix` | 已提交的仿真 GT 数据根目录；不是原始 LIBERO HDF5 目录 |
| `input.cameras` | `[agentview, robot0_eye_in_hand]` | 非空且不可重复的相机名列表，必须存在于输入数据；名称可含字母、数字、下划线和短横线 |
| `output.root` | `/root/opendatasets/LIBERO_REALISM_cosmos25_v1` | 新生成数据目录，首次应为空或不存在 |
| `output.run_dir` | `../../runs/realism/cosmos25_libero_v1` | 计划、进度和报告目录 |
| `output.scratch` | `/tmp/robot-realism/cosmos25/staging` | 本地控制图、原生结果及日志目录；成功提交后清理对应控制图和原生 RGB，日志等辅助文件保留 |

### 5.2 运行环境

| 配置项 | 默认值 | 含义及能填什么 |
|---|---|---|
| `runtime.python` | `/tmp/robot-realism/cosmos25/venv/bin/python` | 完成安装的 Cosmos 环境解释器完整路径；保留 venv 路径，不手动替换成它指向的系统 Python |
| `runtime.cache` | `/tmp/robot-realism/cosmos25/cache` | 模型及下载缓存根目录，需有足够本地空间 |
| `runtime.enable_guardrails` | `true` | 布尔值；是否运行上游文本／图像检查。被阻止的生成会报错，不用占位图补数 |
| `runtime.offload_guardrails` | `true` | 布尔值；是否将检查模型卸载到 CPU，节省显存并使用更多主机内存 |

适配器子进程会设置 `HF_HOME=<cache>/huggingface`、`HF_HUB_CACHE=<cache>/huggingface/hub`、`TORCH_HOME=<cache>/torch`，并关闭训练模式和 tokenizer 编译。`COSMOS_SMOKE=1` 会跳过真实权重加载，本工程禁止在该模式生成数据。

### 5.3 生成参数

| 配置项 | 默认值 | 含义及可填内容 |
|---|---|---|
| `generation.controls` | `[depth]` | 当前支持 `[depth]`、`[edge]`、`[depth, edge]`，不可重复。多控制会载入多个控制分支，显存、耗时和效果需单独验证 |
| `generation.resolution` | `'720'` | 模型内部目标分辨率档位，填字符串 `'480'` 或 `'720'`。最终恢复输入图片尺寸；不是自定义像素宽高 |
| `generation.steps` | `35` | 正整数，扩散采样步数；增加通常更慢，不能保证一定更好 |
| `generation.guidance` | `3` | 整数 `0`～`7`，提示词引导强度；过强可能加剧外观或内容变化 |
| `generation.prompt` | YAML 内的英文说明 | 非空文本。描述真实材质和照明，并要求保留物体、相机和遮挡；首版各状态共用，不自动根据 GT 猜测场景语义 |
| `generation.negative_prompt` | YAML 内的负向说明 | 非空文本。描述不希望出现的卡通、增删物体、变形等情况；不是避免几何漂移的保证 |

基础模型选择由适配器根据控制项确定。这里不接入蒸馏 Edge、汽车多视角或内部机器人多视角权重，也不通过拼接无关图片伪造视频输入。

### 5.4 控制图与采样

| 配置项 | 默认值 | 含义及可填内容 |
|---|---|---|
| `controls.near_m` | `0.05` | 正数，米制深度的近端截断值，须小于 far_m |
| `controls.far_m` | `5.0` | 大于 near_m 的正数，深度远端截断值。截断后取逆深度，再在有效像素内归一化到 0～255；近处较白 |
| `controls.depth_weight` | `1.0` | 数值大于 0 且小于等于 1，深度控制权重 |
| `controls.edge_weight` | `1.0` | 数值大于 0 且小于等于 1，边缘控制权重 |
| `controls.canny_low` | `100` | Canny 低阈值，0～255，须小于 high |
| `controls.canny_high` | `200` | Canny 高阈值，0～255，须大于 low |
| `selection.seed` | `20261022` | 整数，控制场景内样本顺序和每状态的生成随机种子；两个相机共享状态种子，但不保证生成内容一致 |

深度控制的编码是适配层从仿真深度得到的相对结构提示，需要通过小批量实际效果检查；不是模型精确遵守米制几何的证明。原始米制深度和有效掩码直接继承，不替换为生成模型估计的深度。

## 六、输出文件与质量检查

### 6.1 最终结果

```text
LIBERO_REALISM_cosmos25_v1/
├── dataset_info.json          当前生成版本、配置和标注继承说明
├── sampling_plan.json         固定状态及原 train/eval 划分
├── runtime_identity.json      实际软件版本、设备、下载权重及配置摘要
├── manifest.json              已提交数量、状态和最后一次错误
├── categories.json            从输入复制（输入存在时）
├── scene_split.json           从输入复制（输入存在时）
├── provenance/                继承的原始场景 XML
├── train/<scene>/<state_id>/
│   ├── state.npy              原始仿真状态
│   ├── agentview/
│   │   ├── rgb.png            生成后的 RGB
│   │   ├── depth.npz          原仿真米制深度及 valid 掩码
│   │   ├── instance.png       原实例编号图
│   │   ├── annotation.json    继承的相机和物体 GT，增加来源说明
│   │   └── render_meta.json   生成参数、原 RGB 摘要、runtime_id
│   ├── robot0_eye_in_hand/    相同结构
│   ├── PENDING.json           写入所有权记录
│   └── COMMIT.json            双相机完整提交及文件摘要
└── eval/<scene>/<state_id>/   与 train 相同
```

上游单帧推理原生保存 JPEG，适配层读取后保存为 `rgb.png`。保存成 PNG 不会消除此前 JPEG 编码，因此不称为全流程无损输出。原始 RGB 一直在输入数据集中保留，可通过来源摘要追溯。

`annotation.json` 保留原 `objects`、内外参、状态来源，并增加 `appearance_source`、`gt_status`、`source_gt_sha256`。`render_meta.json` 默认 `review_status: pending_visual_review`；本版本没有自动通过人工审核的功能。

### 6.2 runs 目录没有原始图片是正常的

`runs/realism/cosmos25_libero_v1/` 保存计划、doctor、prepared、进度和报告；真正的 RGB 在 `output.root`。对照图只有运行 `report` 后才出现在 `quality_report/previews/`。

`prepared.json` 中控制图路径属于临时文件；对应状态成功提交后可能已清理。需要重新查看控制图可以重跑 `prepare --limit ...`，它不会修改输入或生成结果。

### 6.3 自动检查能证明什么

`validate` 检查：输出提交及摘要、图像是否可读、尺寸是否匹配相机、原状态／深度／实例图是否保持一致、原标注的几何与相机字段是否保持一致，以及数量是否达到计划。

它不能自动证明：生成图中没有变形、每个物体都保留了原轮廓、两个相机的外观一致、或真实模型微调效果提高。对照报告专门用于人工检查这些问题。

## 七、目录结构和上游项目

上游源码直接保留在本工程根目录，适配入口单独放在 `robot_realism/`：

```text
realisation/
├── README.md                     本中文操作说明
├── README.upstream.md            原上游 README
├── UPSTREAM.json                 上游提交、原文件摘要和移植来源
├── IMPLEMENTATION_STATUS.md       本次实际执行和未完成事项
├── cosmos_transfer2/             上游模型与推理代码
├── packages/                     上游内部 Python 包
├── examples/                     上游原始推理入口
├── assets/                       上游示例，当前部分仍为 LFS 占位
├── docs/                         上游说明
├── robot_realism/                LIBERO 接入、控制图、调度、检查与报告
├── configs/realism/              本实例默认配置
├── runtime/requirements-ppu.txt   PPU 环境的 Python 依赖
├── scripts/bootstrap_cosmos25_ppu.py
├── adaptation_tests/             适配层独立测试
└── runs/                         本地运行记录，Git 忽略
```

固定上游提交为 `2ff49d0717af02057ae79bc75c00fbff9da1b4e7`。原 `/mnt/workspace/cosmos-transfer2.5` 保留，没有移动或删除原克隆目录；新目录已复制全部上游工作文件，不带嵌套 `.git`，避免父仓库把它当作嵌入式 Git 仓库。

本次上游调整：`.python-version` 改为当前镜像的 3.12；`pyproject.toml` 的本地 cosmos-oss 依赖版本由 0.1.0 对齐到其实际 1.5.0；权重下载改用当前环境的 Hugging Face Python 接口，保留上游固定 revision 和筛选规则，不依赖额外 `uvx`；推理关闭梯度改用 `torch.set_grad_enabled(False)`。PPU 安装不执行官方 CUDA extra 自动同步。

### 7.1 运行适配层测试

```bash
python -m pytest \
  -c adaptation_tests/pytest.ini \
  --confcutdir=adaptation_tests \
  -q adaptation_tests
```

这些测试使用明确标记的临时夹具，不调用真实 Cosmos，也不会生成可冒充真实结果的数据。原上游测试有自己的 GPU 分配和插件配置；不要直接用裸 `pytest` 代替上述适配层检查。

### 7.2 如果确实要运行上游原始示例

当前部分 `assets` 文件是 Git LFS 指针，不能直接当 JSON 或视频读取。推荐先用本工程的 LIBERO 适配流程，它不依赖这些示例媒体。

若要补齐官方样例，应在安装 Git LFS 后，回到仍保留 `.git` 的原始克隆目录，按需要拉取具体样例，再复制到新工程。不要在不含嵌套 `.git` 的新工程运行不加范围的 `git lfs pull`，以免操作到父仓库。

```bash
cd /mnt/workspace/cosmos-transfer2.5
git lfs pull --include='assets/image_example/**' --exclude=''
```

这需要 git-lfs 已安装、原仓库网络可用；这里只展示单类样例的拉取命令，不会下载 Hugging Face 模型权重。实际推理使用的模型缓存仍由 HF_HOME 控制。

## 八、常见问题

| 现象 | 原因和处理 |
|---|---|
| `No matching distribution found for hatchling` | 当前包源没有解析出候选版本，也可能是网络／索引配置问题。步骤 3 使用可访问的通用索引后重试；不要在系统环境补装依赖 |
| 安装器提示 `Use the image Python` | 用 `/usr/local/bin/python scripts/bootstrap_cosmos25_ppu.py`，不要用采集 `.venv` 的 Python 创建环境 |
| `Existing venv is not owned` | 指向了其他环境，或基础 SDK 已变化；使用独立新路径并同步配置，不自动删除旧环境 |
| `CUDA extra not installed` | 上游导入没有找到本地 cosmos-cuda 包；检查步骤 3 是否完整安装。不在 PPU 上直接执行提示中的 `uv sync --extra=cu130` |
| `No module named tyro/hydra/...` | 依赖未安装完成或激活了错误环境，回到步骤 3～4 |
| `cuda_device: false` | 当前进程看不到卡；在 DSW 原生终端检查设备与环境。后台受限进程的失败不代表原生终端一定失败 |
| `TE RMSNorm` 或 FlashAttention 检查失败 | SDK 算子或 ABI 与当前运行环境不兼容，先处理该错误，不启动全量 |
| `No module named natten` | 首版默认尝试普通注意力后端。若真实加载确实进入 NATTEN 路径，停止并做针对性适配，不随意装 NVIDIA 预编译扩展 |
| `401/403`、模型访问受限 | 检查 Hugging Face 登录、模型访问条款和该 Token 的读取权限 |
| 权重下载时网络／代理错误 | 修复当前终端联网方式后重跑；已缓存文件可复用，不需要重新生成源 GT |
| `Existing realism plan differs` | 当前配置或输入清单与固定计划不一致。恢复原配置，或建立独立的新配置、输出和 run_dir |
| `Realism profile differs` | 生成源码或配置已变化，不能混写旧版本。保持原版本或重新建版 |
| `Runtime packages/device changed` | 同一输出曾使用不同依赖或设备。不要篡改身份记录，使用原环境或新输出版本 |
| `Cached checkpoint/config changed` | 已记录的权重或配置缺失／变化，先恢复缓存内容；若确实要换版本，另建输出 |
| `Source file missing/corrupt` | 原 GT 文件缺失或摘要不符；先恢复源文件，不跳过检查 |
| `Output is nonempty`／`unowned incomplete` | 目录不是本工具当前版本拥有的输出，换独立目录，避免混入其他内容 |
| `use --resume` | 已有输出记录，后续用同一配置加 `--resume`，即使之前实际生成数为 0 |
| `--limit` 报整除错误 | 默认两个相机，填正偶数，如 2、40、200；不是状态数量 |
| `prepared_images: 200`，但没有新 RGB | prepare 只准备控制图；运行 render 才会真实生成 |
| `status: not_started` | 尚未保存生成进度；已有 plan 或 prepared 不算开始生成 |
| `complete: false` | 试生成未达到全量时正常。最终验收需去掉 limit 继续完成，然后不带 allow-partial 验证 |
| `visual_gt_alignment_certified: false` | 有意保留：生成图与继承的 GT 仍需人工检查，文件检查无法证明几何对应 |
| 图片看起来真实，但物体变形／增删 | 不应直接用于原 GT 监督。保留原数据，对照检查；新版本可尝试更强结构控制或更保守提示词，再小批量验证 |
| `/tmp` 环境或权重重启后消失 | 重建专用环境和缓存；源码、配置及已保存的最终结果另行保留。长期使用可在第一次计划前改到合适持久化目录 |
