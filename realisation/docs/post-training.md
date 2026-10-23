<a id="post-training-guide"></a>

# 后训练指南
<a id="prerequisites"></a>

## 前置条件
<a id="1-environment-setup"></a>

### 1. 环境配置
请按照[安装指南](./setup.md)完成通用环境配置，包括依赖安装。

<a id="2-hugging-face-configuration"></a>

### 2. Hugging Face 配置
如果本地没有模型检查点，后训练期间会自动下载。请按以下方式配置 Hugging Face：

```bash
# Login with your Hugging Face token (required for downloading models)
hf auth login

# Set custom cache directory for HF models
# Default: ~/.cache/huggingface
export HF_HOME=/path/to/your/hf/cache
```

> **💡 提示**：确保 `HF_HOME` 所在位置具有足够的磁盘空间。

<a id="3-training-output-directory"></a>

### 3. 训练输出目录
配置训练检查点和产物的保存位置：

```bash
# Set output directory for training checkpoints and artifacts
# Default: /tmp/imaginaire4-output
export IMAGINAIRE_OUTPUT_ROOT=/path/to/your/output/directory
```

> **💡 提示**：确保 `IMAGINAIRE_OUTPUT_ROOT` 所在位置具有足够的磁盘空间。

<a id="weights--biases-wb-logging"></a>

## Weights & Biases（W&B）日志
默认情况下，训练会尝试将指标记录到 Weights & Biases。可以选择以下方式：

<a id="option-1-enable-wb"></a>

### 方式 1：启用 W&B
若要使用 W&B 完整跟踪实验：

1. 在 [wandb.ai](https://wandb.ai) 创建免费账户。
2. 从 [https://wandb.ai/authorize](https://wandb.ai/authorize) 获取 API 密钥。
3. 设置环境变量：

    ```bash
    export WANDB_API_KEY=your_api_key_here
    ```

4. 使用以下命令启动训练：

    ```bash
    EXP=your_experiment_name_here

    torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
      --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py  -- \
      experiment=${EXP}
    ```

<a id="option-2-disable-wb"></a>

### 方式 2：禁用 W&B
在训练命令中添加 `job.wandb_mode=disabled`，即可禁用 W&B 日志：

```bash
EXP=your_experiment_name_here

torchrun --nproc_per_node=8 --master_port=12341 -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py -- \
  experiment=${EXP} \
  job.wandb_mode=disabled
```

<a id="checkpointing"></a>

## 检查点管理
训练使用两种检查点格式，分别针对不同用途进行了优化：

<a id="1-distributed-checkpoint-dcp-format"></a>

### 1. 分布式检查点（DCP）格式
**训练检查点的主要格式。**

- **结构**：由多个文件组成的目录，模型权重分片保存。
- **用途**：训练过程中保存检查点、恢复训练。
- **优势**：
  - 多 GPU 训练时可高效并行读写。
  - 支持 FSDP（完全分片数据并行）。
  - 针对分布式工作负载进行了优化。

**目录结构示例：**

```
checkpoints/
├── iter_{NUMBER}/
│   ├── model/
│   │   ├── .metadata
│   │   └── __0_0.distcp
│   ├── optim/
│   ├── scheduler/
│   └── trainer/
└── latest_checkpoint.txt
```

<a id="2-consolidated-pytorch-pt-format"></a>

### 2. 合并的 PyTorch（.pt）格式
**用于推理和分发的单文件格式。**

- **结构**：单个 `.pt` 文件，包含完整模型状态。
- **用途**：推理、模型共享、初始化后训练。
- **优势**：
  - 易于分发和版本管理。
  - 使用标准 PyTorch 格式。
  - 单 GPU 工作流更简单。

<a id="loading-checkpoints"></a>

### 加载检查点
训练系统**支持加载上述两种格式**：

**加载 DCP 检查点（用于恢复训练）：**

```python
load_path="checkpoints/nvidia/Cosmos-Transfer2.5-2B/dcp"
```

**加载合并检查点（用于启动后训练）：**

```python
load_path="checkpoints/nvidia/Cosmos-Transfer2.5-2B/consolidated/model.pt"
```

> **注意**：从 Hugging Face 下载的预训练模型通常采用合并的 `.pt` 格式。训练系统会自动加载该格式并开始训练。

<a id="saving-checkpoints"></a>

### 保存检查点
**训练期间保存的所有检查点均采用 DCP 格式**，以保证：

- 不同训练运行之间的检查点结构一致。
- 分布式训练获得最佳性能。

<a id="post-training-examples"></a>

## 后训练示例
详细的训练示例和配置选项请参阅：

- [单视角 Control2World 后训练](./post-training_singleview.md)
- [HDMap 多视角 Control2World 后训练](./post-training_auto_multiview.md)
- [Agibot 三视角（head_color、hand_left、hand_right）后训练](./post-training_agibot_multiview.md)
