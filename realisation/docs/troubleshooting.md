<a id="troubleshooting"></a>

# 故障排查
<a id="issues"></a>

## 常见问题
也可以查看相关[仓库](https://github.com/orgs/nvidia-cosmos/repositories)的 GitHub Issues。

<a id="changing-the-cache-directory"></a>

### 更改缓存目录
默认将软件包和检查点下载到用户主目录。要更改这一位置，可以设置以下变量。
如果安装或下载检查点时出现 `No space left on device`，也需要调整这些设置。

```shell
export UV_CACHE_DIR=<new_dir>/.cache
export PIP_CACHE_DIR=<new_dir>/.cache
export HF_HOME=<new_dir>/checkpoints
```

<a id="missing-pythonh"></a>

### 缺少 Python.h
错误信息：`fatal error: Python.h: No such file or directory`

可以通过[安装由 uv 管理的 Python](https://docs.astral.sh/uv/guides/install-python/#installing-python)来解决：

```shell
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install --reinstall
```

重新安装软件包：

```shell
rm -rf .venv
uv sync --extra=cu128
```

<a id="cuda-driver-version-insufficient"></a>

### CUDA 驱动版本不足
**解决方法：**将 NVIDIA 驱动更新到兼容 [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) 的最新版本。

检查驱动兼容性：

```shell
nvidia-smi | grep "CUDA Version:"
```

<a id="pythonpath-conflicts-in-nvidia-containers"></a>

### NVIDIA 容器中的 PYTHONPATH 冲突
使用 `nvcr.io/nvidia/pytorch:25.xx-py3` 容器时，需要取消设置 `PYTHONPATH`，以兼容 Python 3.10：

```shell
unset PYTHONPATH
```

<a id="out-of-memory-oom-errors"></a>

### 内存或显存不足（OOM）
**解决方法：**使用 2B 模型代替 14B 模型、使用多 GPU，或者减小批量大小／分辨率。

<a id="guide"></a>

## 排查指南
<a id="logs"></a>

### 日志
日志保存在 `<output_dir>/*.log`。

<a id="profiling"></a>

### 性能分析
传入 `--profile` 标志可进行性能分析。[pyinstrument](https://pyinstrument.readthedocs.io/en/latest/guide.html) 分析结果将导出到 `<output_dir>/profile.pyisession`。

查看分析结果：

```shell
pyinstrument --load=<output_dir>/profile.pyisession
```

导出分析结果：

```shell
pyinstrument --load=<output_dir>/profile.pyisession -r html -o <output_dir>/profile.html
```

更多信息请参阅 [pyinstrument](https://pyinstrument.readthedocs.io/en/latest/guide.html)。
