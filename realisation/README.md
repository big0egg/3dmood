# Cosmos-Transfer2.5：环境部署说明

本说明用于在当前 PPU-ZW810E 96 GB 实例中安装环境、准备模型权重。**按下面 4 步顺序执行，每一步完成后再进行下一步。**

环境准备好以后，打开 [introduce.md](introduce.md)，按其中的步骤提供仿真数据并生成图片。

| 用途 | 默认位置 |
|---|---|
| 工程目录 | `/mnt/workspace/3dmood/realisation` |
| Python 环境 | `/tmp/robot-realism/cosmos25/venv` |
| 模型及下载缓存 | `/tmp/robot-realism/cosmos25/cache` |
| 配置文件 | `configs/realism/cosmos25_libero_v1.yaml` |

`/tmp` 中的环境和权重可能在实例重建后丢失。长期使用时，应在首次运行前选好持久化目录，并同步修改安装路径及配置中的 `runtime.python`、`runtime.cache`。

## 第 1 步：进入工程并查看实例

在 DSW 终端执行：

```bash
cd /mnt/workspace/3dmood/realisation
ppu-smi
df -h /mnt/workspace /tmp /root
```

**完成标志：** 能看到 PPU-ZW810E，准备保存环境和模型的磁盘有足够空间。

## 第 2 步：安装运行环境


复制下面的完整命令。它会使用当前实例的 PPU 专用源，将项目依赖安装到单独的环境中。地址中的 `pg1-pip` 使用数字 `1`。

安装器会自动安装 NumPy 2.2.6、兼容 NumPy 2 的 scikit-learn、视频解码库 `decord`、几何处理库 `Shapely` 等运行依赖，以及本地源码安装需要的 `hatchling`、`editables`。已有环境可以直接重复执行，已安装且满足版本要求的包会复用。

```bash
NO_PROXY="${NO_PROXY:+$NO_PROXY,}aiext-pypi.mirrors.aliyuncs.com" \
no_proxy="${no_proxy:+$no_proxy,}aiext-pypi.mirrors.aliyuncs.com" \
/usr/local/bin/python scripts/bootstrap_cosmos25_ppu.py \
  --index-url https://aiext-pypi.mirrors.aliyuncs.com/pg1-pip/ubuntu_cu130/simple/
```

安装器会在结束前核对依赖版本，检查视频解码、几何处理、文本处理和推理入口，并加载实际使用的模型配置、解析模型组件。检查不下载模型、不加载权重、不生成图片；全部通过后才显示安装完成。

**完成标志：** 检查结果中显示 `"ready": true`，最后显示 `NumPy / Torch CPU array bridge: PASS` 和 `Installation complete`，然后进入第 3 步。

程序启动渲染进程时会自动设置 `COSMOS_TRAINING=1`，让推理配置能够加载所需的公共组件；实际执行的仍是图片生成。

## 第 3 步：激活环境


让后续命令使用刚安装的环境，并指定模型保存位置。

```bash
source /tmp/robot-realism/cosmos25/venv/bin/activate
export HF_HOME=/tmp/robot-realism/cosmos25/cache/huggingface
export HF_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HUB_CACHE"
which python
```

**完成标志：** 显示 `/tmp/robot-realism/cosmos25/venv/bin/python`。

以后每次打开新终端，先进入工程，再重新执行本步骤。

## 第 4 步：准备模型权重


权重就是模型运行需要读取的文件。**优先使用手动下载；如果希望程序自己下载，可以选择后面的自动下载。两种方式选一种即可。**

**手动下载（推荐）**

在浏览器登录 Hugging Face，按模型页面要求取得访问权限。点击下表链接下载文件，再通过 DSW 文件浏览器上传到对应位置。文件较多的模型，要保留全部配置文件、权重分片及原有子目录。

先在 DSW 终端创建保存目录：

```bash
mkdir -p "$HF_HUB_CACHE/models--nvidia--Cosmos-Transfer2.5-2B/snapshots/dea7737ca29dd8d9086413c6dc5724b8250a0bb4/general/depth"
mkdir -p "$HF_HUB_CACHE/models--nvidia--Cosmos-Predict2.5-2B/snapshots/f176dc95b4a70f53ce01c4b302851595e7322b00"
mkdir -p "$HF_HUB_CACHE/models--nvidia--Cosmos-Reason1-7B/snapshots/3210bec0495fdc7a8d3dbb8d58da5711eab4b423"
mkdir -p "$HF_HUB_CACHE/models--nvidia--Cosmos-Guardrail1/snapshots/d6d4bfa899a71454a700907664f3e88f503950cf"
mkdir -p "$HF_HUB_CACHE/models--Qwen--Qwen3Guard-Gen-0.6B/snapshots/fada3b2f655b89601929198343c94cd2f64d93cc"
mkdir -p "$HF_HUB_CACHE/models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
```

下表的“保存位置”都位于 `/tmp/robot-realism/cosmos25/cache/huggingface/hub/` 下。下载链接已选择本流程使用的版本。

| 下载入口 | 要下载的内容 | 保存位置（相对于上述目录） |
|---|---|---|
| [深度主模型](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B/tree/dea7737ca29dd8d9086413c6dc5724b8250a0bb4/general/depth) | `626e6618-bfcd-4d9a-a077-1409e2ce353f_ema_bf16.pt` | `models--nvidia--Cosmos-Transfer2.5-2B/snapshots/dea7737ca29dd8d9086413c6dc5724b8250a0bb4/general/depth/` |
| [图像编解码模型](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B/tree/f176dc95b4a70f53ce01c4b302851595e7322b00) | 只下载 `tokenizer.pth` | `models--nvidia--Cosmos-Predict2.5-2B/snapshots/f176dc95b4a70f53ce01c4b302851595e7322b00/` |
| [文本编码器](https://huggingface.co/nvidia/Cosmos-Reason1-7B/tree/3210bec0495fdc7a8d3dbb8d58da5711eab4b423) | 页面列出的全部文件，包括 4 个权重分片、索引和配置文件 | `models--nvidia--Cosmos-Reason1-7B/snapshots/3210bec0495fdc7a8d3dbb8d58da5711eab4b423/` |
| [内容检查资源](https://huggingface.co/nvidia/Cosmos-Guardrail1/tree/d6d4bfa899a71454a700907664f3e88f503950cf) | 全部文件和子目录 | `models--nvidia--Cosmos-Guardrail1/snapshots/d6d4bfa899a71454a700907664f3e88f503950cf/` |
| [文本检查模型](https://huggingface.co/Qwen/Qwen3Guard-Gen-0.6B/tree/fada3b2f655b89601929198343c94cd2f64d93cc) | 页面列出的全部文件 | `models--Qwen--Qwen3Guard-Gen-0.6B/snapshots/fada3b2f655b89601929198343c94cd2f64d93cc/` |
| [图像检查模型](https://huggingface.co/google/siglip-so400m-patch14-384/tree/9fdffc58afc957d1a03a25b10dba0329ab15c2a3) | 页面列出的全部文件 | `models--google--siglip-so400m-patch14-384/snapshots/9fdffc58afc957d1a03a25b10dba0329ab15c2a3/` |

上传完成后，执行下面的完整命令，登记两个辅助模型的版本并启用本地读取：

```bash
mkdir -p "$HF_HUB_CACHE/models--Qwen--Qwen3Guard-Gen-0.6B/refs"
printf '%s' fada3b2f655b89601929198343c94cd2f64d93cc \
  > "$HF_HUB_CACHE/models--Qwen--Qwen3Guard-Gen-0.6B/refs/main"
mkdir -p "$HF_HUB_CACHE/models--google--siglip-so400m-patch14-384/refs"
printf '%s' 9fdffc58afc957d1a03a25b10dba0329ab15c2a3 \
  > "$HF_HUB_CACHE/models--google--siglip-so400m-patch14-384/refs/main"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

**完成标志：** 表中所有文件均已上传到对应目录。权重的实际加载与生成检查见 [introduce.md](introduce.md)。

手动下载的用户以后打开新终端时，在第 3 步之后再执行 `export HF_HUB_OFFLINE=1` 和 `export TRANSFORMERS_OFFLINE=1`。

**自动下载（备选）**

先在浏览器登录 Hugging Face，依次打开下面三个官方页面，阅读并同意 **NVIDIA Open Model License Agreement** 和页面上的信息共享授权，确认该账号已经获得文件访问权限后再继续：

- [Cosmos-Transfer2.5-2B](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)
- [Cosmos-Predict2.5-2B](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1)

然后在[同一账号的 Token 页面](https://huggingface.co/settings/tokens)创建用于这些模型下载的读取 Token。在已完成第 3 步的终端执行下面的命令，粘贴 Token 后按回车；输入不会显示：

```bash
unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_HUB_DISABLE_IMPLICIT_TOKEN
read -rsp '请输入 Hugging Face 读取 Token（输入不显示）: ' HF_TOKEN
printf '\n'
export HF_TOKEN
```

下载库会自动读取 `HF_TOKEN`，无需先执行命令行登录；输入 Token 时不会连接官网进行账号验证。本方式仅对当前终端及其启动的程序生效。**网页上的协议确认仍需提前完成。** 未获准访问对应仓库、Token 权限不足或仍在等待审核时，自动下载可能被 `401/403` 或 `GatedRepoError` 阻止。参见 [Token 环境变量说明](https://huggingface.co/docs/huggingface_hub/v0.36.0/package_reference/environment_variables#hftoken)和[模型授权说明](https://huggingface.co/docs/hub/models-gated)。

**从官方站点自动下载：** 输入 Token 后执行：

```bash
export HF_ENDPOINT=https://huggingface.co
unset HF_HUB_DISABLE_XET
```

**通过镜像自动下载：** 完成网页授权和 Token 输入后，在同一终端执行：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

这里使用第三方 [HF-Mirror](https://hf-mirror.com/)，通过 `HF_ENDPOINT` 指定下载入口，并关闭 Xet 下载通道以使用 HTTP 下载。镜像支持携带 Token 下载受限仓库，下载请求会将读取 Token 发给镜像服务；使用专门用于这些模型的读取 Token。**镜像同样要求官网已授权，不会跳过协议或审核。**

保留上表的模型名称和版本，缓存位置也保持第 3 步的设置。镜像是否能取得全部文件及固定版本，以实际下载结果为准；不能仅凭模型页面存在就认定权重齐全。

选好一种下载入口后，按 [introduce.md](introduce.md) 运行项目。首次生成时，程序会自动下载所需权重并复用已有缓存。以后打开新终端，在第 3 步之后重新输入 Token，并设置所选的 `HF_ENDPOINT`；镜像方式还需设置 `HF_HUB_DISABLE_XET=1`。

## 环境准备完成

安装器显示 `Installation complete`，并且已按第 4 步完成手动权重准备或设置自动下载所需的 Token 和入口后，打开 [introduce.md](introduce.md) 开始运行。

<details>
<summary>安装参数（需要修改时再看）</summary>

| 参数 | 默认值 | 含义 |
|---|---|---|
| `--index-url` | `https://pypi.org/simple` | Python 包下载源；本说明显式指定当前实例的 PPU 专用源，不影响模型权重下载 |
| `--venv` | `/tmp/robot-realism/cosmos25/venv` | 专用环境保存位置；修改后同步修改配置的 `runtime.python` 为新环境的 `bin/python` |
| `--cache` | `/tmp/robot-realism/cosmos25/cache` | 下载缓存位置；修改后同步修改配置的 `runtime.cache` 和权重保存路径 |
| `--check` | 不启用 | 只查看基础镜像的 SDK 版本 |
| `--help` | 不启用 | 查看安装器参数 |

</details>
