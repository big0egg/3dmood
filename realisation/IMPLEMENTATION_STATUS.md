# 2026-10-22 移植及适配记录

## 已完成

- 从 `/mnt/workspace/cosmos-transfer2.5` 复制完整上游工作树到本目录，排除嵌套 `.git`；原克隆目录保留，未修改原文件。
- 上游提交固定为 `2ff49d0717af02057ae79bc75c00fbff9da1b4e7`；原始文件摘要记录在 `UPSTREAM.json`，英文说明保存在 `README.upstream.md`。
- 新增 `robot_realism`：环境检查、源数据计划、深度／边缘控制图、真实 Cosmos 工作进程、双相机提交、版本及权重来源记录、断点恢复、验证及对照报告。
- 安装器复用 PPU 镜像 SDK，锁定已安装的 PyTorch、torchvision、TE、FlashAttention、Triton 及 CUDA 库；不选择官方 cu128/cu130 extra。
- `.python-version` 改为 3.12；修复上游根包要求 cosmos-oss 0.1.0、而其实际版本为 1.5.0 的声明不一致。
- 权重下载改用本环境的 Hugging Face Python API，保留固定 revision 和 include/exclude 过滤，避免额外 uvx CLI 环境；推理禁用梯度调用改用 `torch.set_grad_enabled(False)`。
- 默认输入 `/root/opendatasets/LIBERO_GT_v1_plane_fix`，新输出 `/root/opendatasets/LIBERO_REALISM_cosmos25_v1`；未修改源数据或已有采集代码。
- README 重写为 12 步操作说明，包含每步目的、命令、预期结果、错误处理及全部适配参数。

## 实际验证

| 项目 | 结果 |
|---|---|
| 用户在原生终端验证的加速卡 | PPU-ZW810E，96 GB |
| 原生终端的小算子结果 | BF16、SDPA、TE Linear、TE Attention、FlashAttention 全部通过；不等于完整 Cosmos 前向通过 |
| 适配层自动测试 | 12 passed；使用临时、明确标注的测试夹具，不是模型生成的数据 |
| 源 GT 清单 | 5,500 个已提交状态、11,000 张，原 train/eval 划分保留 |
| 新写实化计划 | 5,500 状态，train 10,000 张，eval 1,000 张 |
| 真实源数据控制图准备 | 100 个状态、200 张，覆盖 20 个场景组 |
| SDK 安装前版本检查 | 通过，记录了镜像版本 |
| 独立虚拟环境 | 已创建 `/tmp/robot-realism/cosmos25/venv`，依赖未装完 |
| 依赖安装尝试 | 镜像 SDK 包索引和指定的通用 PyPI 索引均未解析到 hatchling 候选版本，安装退出 |
| 后台联网诊断 | 直接访问 PyPI 返回 `URLError: [Errno 1] Operation not permitted`，当前执行环境限制联网 |
| 后台设备可见性 | `torch.cuda.is_available()` 为 false；不能代替原生终端的设备检查结果 |
| doctor | 输入检查通过，运行环境未就绪，ready=false |
| 权重下载、完整模型加载 | 未完成 |
| 实际 Cosmos 生成图片 | **0 张**；没有用原图或测试夹具冒充生成结果 |
| 实际 render --limit 2 预检查 | 正确拒绝未就绪的运行环境，未创建最终输出目录 |

准备好的 `plan.json`、`prepared.json` 和 `doctor.json` 位于 `runs/realism/cosmos25_libero_v1/`。控制图位于默认 scratch，`prepared.json` 中有逐图路径。

## 后续执行条件

按 README 从步骤 3 完成专用环境安装，再激活环境运行 doctor。包源必须可访问，Hugging Face 权重需具备下载权限。doctor 通过后先真实生成一个双相机状态，并查看对照图，再扩大到 200 张和全量。

当前全流程尚未通过真实模型出图测试。后续仍可能遇到 PPU 专属算子、依赖 API、权重访问或显存兼容问题，需要按实际错误继续适配，不能据本次单元测试承诺完整模型可运行。

## 安装器与操作说明更新

- README 仅保留顺序操作步骤、参数、预期结果和完成标准；执行历史保存在本文。
- NumPy 固定为 2.2.6、Megatron-Core 固定为 0.14.0，沿用上游 NumPy 2 覆盖策略。普通依赖经 pip 解析安装，Megatron 单独按固定版本安装；安装器逐项核验其依赖，只允许明确的 NumPy 上限例外。
- 安装结束前核验 SDK 版本及 NumPy / Torch CPU 数组转换；完整模型兼容性仍由 doctor 和真实试生成确认。
- 适配层测试更新为 **18 passed**，包含版本组合及额外依赖错误不能被 NumPy 例外掩盖的检查。
- SDK 只读检查通过。后台 `pip install --dry-run` 仍未从包源获取到 hatchling 候选版本，未完成完整依赖解析和安装；该结果不等于原生终端的联网结果。
- 官方样例的说明使用独立临时克隆，不再依赖 `/mnt/workspace/cosmos-transfer2.5`。原目录未删除。
