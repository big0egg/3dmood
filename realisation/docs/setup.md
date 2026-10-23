<a id="setup-guide"></a>

# 安装指南
<!--TOC-->

- [系统要求](#system-requirements)
- [安装](#installation)
- [下载检查点](#downloading-checkpoints)

<!--TOC-->

<a id="system-requirements"></a>

## 系统要求
* NVIDIA Ampere 架构（RTX 30 系列、A100）或更新架构的 GPU
* NVIDIA 驱动版本 >=570.124.06，并兼容 [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions)
* Linux x86-64
* glibc>=2.35（例如 Ubuntu >=22.04）

<a id="installation"></a>

## 安装
安装 [git lfs](https://git-lfs.com/)：

```bash
sudo apt install git-lfs
git lfs install
```

克隆仓库：

```bash
git clone git@github.com:nvidia-cosmos/<repository_name>.git
cd <repository_name>
git lfs pull
```

选择以下一种环境进行安装：

<details id="virtual-environment"><summary><b>虚拟环境</b></summary>

安装系统依赖：

```shell
sudo apt update && sudo apt -y install curl ffmpeg libx11-dev tree wget
```

* [uv](https://docs.astral.sh/uv/getting-started/installation/)

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

将软件包安装到新环境中：

```shell
uv python install
uv sync --extra=cu128
source .venv/bin/activate
```

或者，将软件包安装到当前已激活的环境中（例如 conda）：

```shell
uv sync --extra=cu128 --active --inexact
```

CUDA 版本选项：

| CUDA 版本 | 参数 | 说明 |
| --- | --- | --- |
| CUDA 12.8 | `--extra cu128` | [NVIDIA 驱动](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |
| CUDA 13.0 | `--extra cu130` | [NVIDIA 驱动](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) |

对于 DGX Spark 和 Jetson AGX，必须使用 CUDA 13.0。
</details>

<details id="docker-container"><summary><b>Docker 容器</b></summary>

请确保机器上可以使用 Docker，并已安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)。

构建容器：

```bash
# Ampere - Hopper
image_tag=$(docker build -f Dockerfile -q .)
# Blackwell
image_tag=$(docker build -f docker/nightly.Dockerfile -q .)
```

运行容器：

```bash
docker run -it --runtime=nvidia --ipc=host --rm -v .:/workspace -v /workspace/.venv -v /root/.cache:/root/.cache -e HF_TOKEN="$HF_TOKEN" $image_tag
```

可选参数：

* `--ipc=host`：使用宿主机的共享内存，因为并行 torchrun 会消耗大量共享内存。如果安全策略不允许这样做，请增大 `--shm-size`（[文档](https://docs.docker.com/engine/containers/run/#runtime-constraints-on-resources)）。
* `-v /root/.cache:/root/.cache`：挂载宿主机缓存，避免重复下载缓存内容。
* `-e HF_TOKEN="$HF_TOKEN"`：设置 Hugging Face 令牌，避免重复认证。

如果出现 `docker: Error response from daemon: unknown or invalid runtime name: nvidia`，需要[配置 Docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker)：

```shell
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

</details>

<a id="downloading-checkpoints"></a>

## 下载检查点
1. 获取具有 `Read`（读取）权限的 [Hugging Face 访问令牌](https://huggingface.co/settings/tokens)。
2. 安装 [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/en/guides/cli)：`uv tool install -U "huggingface_hub[cli]"`。
3. 登录：`hf auth login`。
4. 接受 [NVIDIA 开放模型许可协议](https://huggingface.co/nvidia/Cosmos-Guardrail1)。

检查点会在推理和后训练过程中自动下载。要修改检查点的缓存位置，请设置 [`HF_HOME`](https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables#hfhome) 环境变量。
