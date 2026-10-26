# 验证记录（2026-10-26）

- `ioutest`：43 项测试通过，2 项因验证环境缺少可选 Vis4D 组件跳过。
- `finefune`：WildDet3D 环境通过 59 项测试；3D-MOOD 原生编码用例在其对应环境另行通过 1 项，共 60 项。
- 新增验证覆盖评测指标／计时／执行设置的完整导出、显式参数优先级、模型缓存环境传递、缓存文件缺失检查及缓存内容变更后的身份隔离。
- 两模型×四训练数据的八分支计划已解析验证；没有启动正式训练。
- 实际读取并核对：对齐原始仿真、全通用生成、混合生成均为相同的 4,910 个训练状态。原始仿真测试 500 帧，两套生成训练配置共用 499 帧生成测试集。
- 3D-MOOD 的本地 BERT 配置、分词器和权重检查通过；模型环境检查仍因当前工具不可见 PPU 返回 ready=false。
- WildDet3D 基础和 LingBot 权重均使用 `/root/wilddet3d-weights/`，不回退到 OSS。
- 旧产物 `runs/`、`evaluation/`、`report/` 已核对不存在；现成 HOT3D 数据和 agentview_prompt_v3 数据保留。
- Python 语法、JSON 配置和差异格式检查通过。

以上文件与接口验证不代表生成图片的几何标签已完成视觉审计，也不代表已完成八组 PPU 微调或真实机器人操作评测。

原生终端可先运行只读计划：

```bash
python -B /mnt/workspace/3dmood/finefune/scripts/check_native_training.py
```

显式加 `--run` 才会进行各组的一次真实参数更新；该检查不产生完整微调权重。正式全量运行步骤见 [README](README.md)。
