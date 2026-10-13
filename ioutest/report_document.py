"""Chinese benchmark documentation rendered from frozen data and saved results.

Rendering never loads a detector, changes predictions or recalculates AP.
"""

from collections import Counter
from pathlib import Path

from .schema import CLASSES


CLASS_NAMES_ZH = (
    "柜子", "冰箱", "架子", "炉灶", "床", "水槽", "洗衣机", "马桶", "浴缸",
    "烤箱", "洗碗机", "壁炉", "凳子", "椅子", "桌子", "电视/显示器", "沙发",
)
METRIC_NAMES = ("AP3D@0.25", "AP3D@0.50", "APcenter@0.25m", "APcenter@0.50m", "APcenter@1.00m")


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", "<br>")
    return "\n".join([
        "| " + " | ".join(map(cell, headers)) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(cell, row)) + " |" for row in rows),
    ])


def percent(value):
    return "—" if value is None else f"{100 * value:.2f}"


def values(metric):
    summary = metric["summary"]
    return [summary["iou"]["map_per_threshold"][k] for k in ("0.25", "0.5")] + [
        summary["center"]["map_per_threshold"][k] for k in ("0.25", "0.5", "1.0")]


def comparison(results, keys, mode="geometry", scope="full"):
    return table(["模型", *METRIC_NAMES], [
        [key, *map(percent, values(results[key]["metrics"][mode][scope]))]
        for key in keys if key in results and mode in results[key]["metrics"]
    ])


def report_context(run_dir, report_root, phase, protocol, audit, all_frames, gt, frames, results, images):
    """Count the actual phase, rather than repeating full-dataset totals in pilots."""
    valid, ignored, reasons = Counter(), Counter(), Counter()
    empty = 0
    for frame in frames:
        boxes = gt[frame["frame_id"]]
        empty += not any(not b.get("ignore", False) for b in boxes)
        for box in boxes:
            (ignored if box.get("ignore", False) else valid)[box["class_id"]] += 1
            reasons.update(box.get("ignore_reasons", []))
    overlap = audit["official_training_visit_overlap"]
    excluded = sorted(set(overlap["validation_scan_ids"]) & {f["video_id"] for f in frames})
    counts = Counter(f["video_id"] for f in frames)
    return {
        "document_schema": "ioutest.report.v2", "benchmark_id": protocol["benchmark_id"],
        "phase": phase, "run_dir": str(Path(run_dir).resolve()),
        "report_root": str(Path(report_root).resolve()), "settings": protocol["settings"],
        "frozen_frames": len(all_frames), "frozen_scans": len({f["video_id"] for f in all_frames}),
        "frozen_visits": len({f["visit_id"] for f in all_frames}),
        "frames": len(frames), "scans": len(counts), "visits": len({f["visit_id"] for f in frames}),
        "frames_per_scan_min": min(counts.values()), "frames_per_scan_max": max(counts.values()),
        "raw_image_hw": sorted({tuple(f["raw_image_hw"]) for f in frames}),
        "valid_gt": sum(valid.values()), "ignored_gt": sum(ignored.values()),
        "empty_valid_gt_frames": empty,
        "classes": [{"id": cid, "name": name, "name_zh": CLASS_NAMES_ZH[cid],
                     "valid_gt": valid[cid], "ignored_gt": ignored[cid]} for cid, name in enumerate(CLASSES)],
        "ignore_reasons": dict(reasons), "official_overlap": overlap,
        "excluded_scan_ids": excluded,
        "excluded_frames": sum(f["video_id"] in excluded for f in frames),
        "model_ids": list(results),
        "model_frame_results": sum(r["frames"] for r in results.values()),
        "model_prediction_counts": {key: sum(row["prediction_count"] for row in
            result["metrics"]["geometry"]["full"]["per_class"]) for key, result in results.items()},
        "visualizations": images,
        "visualization_policy": {"maximum_scans": 12, "frame": "middle sampled frame",
                                 "scan_order": "phase manifest order", "score_threshold": .1},
    }


def render_report(context, results, selection):
    """Use recorded inference provenance, never the current editable config."""
    c, settings = context, context["settings"]
    run, root = c["run_dir"], c["report_root"]
    total = c["valid_gt"]
    sections = []

    def add(title, body):
        sections.append(f"## {title}\n\n{body}")

    title = "CPU 链路验证" if c["phase"] == "cpu-smoke" else f"ARKitScenes 三项目评测：{c['phase']}"
    intro = (f"本阶段全部指定模型已完成：{c['frames']:,} 帧、{c['scans']} 个扫描、{c['visits']} 个 visit，"
             f"共 {len(results)} 组配置。结果来自实际预测。AP 表格使用百分数（0～100），指标 CSV/JSON 使用 0～1。\n\n"
             f"benchmark_id：`{c['benchmark_id']}`；selection_id：`{selection['selection_id']}`。\n\n"
             "本报告比较这些权重与配置在当前抽帧协议下的表现。输入、候选数量、预处理与训练经历仍有差异，"
             "不能直接当作论文原始设置的成绩或严格零样本排名。")
    if c["phase"] == "cpu-smoke":
        intro = "本报告只验证 CPU 执行链路，不作为正式效果或设备性能比较。\n\n" + intro

    add("数据集、抽帧与真值", f"""原始数据目录：`{settings['dataset_root']}`。使用 ARKitScenes 的 `3dod/Validation`，评测单帧图像中的三维物体检测。

{table(['项目', '冻结数据全集', '本报告阶段'], [
    ['扫描数', c['frozen_scans'], c['scans']], ['visit 数', c['frozen_visits'], c['visits']],
    ['帧数', f"{c['frozen_frames']:,}", f"{c['frames']:,}"],
    ['当前阶段每扫描帧数', f"协议目标 {settings['frames_per_scan']}", f"{c['frames_per_scan_min']}～{c['frames_per_scan_max']}"],
    ['有效帧—物体标注数', '按阶段统计', f"{c['valid_gt']:,}"],
    ['忽略帧—物体标注数', '按阶段统计', f"{c['ignored_gt']:,}"],
    ['无有效 GT 的帧数', '按阶段统计', f"{c['empty_valid_gt_frames']:,}"],
])}

扫描是一次采集序列，visit 是采集访问分组，同一 visit 可能包含多个扫描。同一物体在不同帧出现会重复计入“帧—物体”标注，上述数量不是独立物体数。

RGB 来自 `lowres_wide`，原始宽×高为 {', '.join(f'{w}×{h}' for h, w in c['raw_image_hw'])}；深度来自同帧 `lowres_depth`（原始毫米，读取时转换为米）。真值来自扫描内的 `<扫描ID>_3dod_annotation.json`，相机内参来自 `.pincam`，相机位姿来自 `lowres_wide.traj`。

先检查 RGB、深度、内参与位姿是否配对；位姿时间差须小于 0.005 秒。候选帧时间间隔至少 {settings['minimum_interval_seconds']} 秒，再覆盖序列时间范围均匀选取每扫描最多 {settings['frames_per_scan']} 帧。pilot 从冻结清单中按固定种子选择不同 visit 的扫描；full 使用整个冻结清单。

GT 从世界坐标变换到当前帧的转正相机坐标系，单位米，x 向右、y 向下、z 向前。框尺寸是沿自身三个轴的完整边长，旋转矩阵将框自身坐标转换到相机坐标。

有效 GT 要求投影与图像视野相交，且框内观测深度点严格大于 {settings['visible_points_strictly_greater_than']}（深度采样步长 {settings['depth_stride']}）。其他框标记为 ignore。无有效 GT 的帧仍保留，预测误报仍计入 AP。

{table(['忽略原因', '本阶段出现次数'], [
    ['框内观测深度点不足', c['ignore_reasons'].get('insufficient_observed_depth_points', 0)],
    ['框投影不在视野内', c['ignore_reasons'].get('outside_camera_view', 0)],
])}

一个框可能同时有多个忽略原因，原因次数不能直接相加作为忽略框总数。GT 的可见性筛选使用实测深度，RGB 模型不会收到这些深度或 GT 框。""")

    file_rows = [
        ("protocol.json", "数据源、扫描列表、抽帧参数、类别/文本提示、校验值和 benchmark_id", "推理与评分开始前检查协议"),
        ("frames.jsonl", "每行一帧；帧/扫描/visit ID、时间戳、RGB/深度条目、内参、姿态、转正参数和图像校验值", "抽帧选择、输入读取、图像/内参转正"),
        ("groundtruth/", "每扫描一个 JSON；场景信息和每帧三维真值框", "AP 匹配与绿色真值框绘制；不传入模型接口"),
        ("dataset_audit.json", "扫描、类别、配对异常、官方训练/验证 visit 重叠和 GT 校验值", "完整性检查与排除重叠 visit 的补充评分"),
        ("preparation_summary.json", "数据准备阶段汇总数量（若存在）", "人工查看；不是最终模型成绩，报告统计从当前阶段 GT 重新计数"),
    ]
    add("评测文件及其用途", f"""清单和 GT 目录：`{run}`。这里保存数据清单和真值，不保存 RGB 图片。

{table(['文件/目录', '内容', '使用阶段'], [[f'[{name}]({run}/{name})', content, usage] for name, content, usage in file_rows])}

GT 框字段：`uid` 是物体标识，`class_id` 是类别，`center` 是米制中心，`size` 是完整边长，`rotation` 是方向；`ignore`、`ignore_reasons`、`observed_depth_points` 记录可见性状态，`projected_bbox` 记录图像投影范围。

{table(['本地缓存文件', '用途'], [
    ['raw_images/<扫描ID>/*.png', '推理读取原始 RGB，再按冻结参数转正一次'],
    ['images/<扫描ID>/*.png', '已转正 RGB，便于查看；不得再次旋转'],
    ['images.jsonl', '帧 ID 对应图片路径、内参、尺寸和校验值'],
    ['depth_images/<扫描ID>/*.png', '原始毫米深度，RGB-D 读取时转换成米并转正'],
    ['depths.jsonl', '帧 ID 对应深度路径与校验值'],
    ['cache_protocol.json / *_summary.json', '缓存协议与完整性记录'],
])}

{table(['模型', '实际 RGB 输入来源', '深度来源'], [
    [key, '`' + str(r['model']['provenance'].get('input_source', {}).get('root', '未记录')) + '`',
     r['model']['provenance'].get('depth_source', {}).get('source', '无') if r['model']['provenance'].get('depth_source') else '无']
    for key, r in results.items()
])}

当前 benchmark 要求本地 RGB 缓存；推理不逐帧访问远程 ZIP。图像、运行产物与预测缓存通过 ioutest 的 Git 忽略规则排除。""")

    input_rows, weight_rows = [], []
    for key, result in results.items():
        p = result["model"]["provenance"]
        m, options = p.get("model", {}), p.get("options", {})
        modalities = [dict(rgb="RGB", intrinsics="相机内参", fixed_class_prompts="固定类别文本",
                           fixed_17_class_prompts="固定 17 类文本", ARKit_LiDAR_depth_meters="ARKit 实测深度",
                           trajectory_derived_gravity="传感器姿态推导重力").get(x, x)
                      for x in m.get("input_modalities", [])]
        semantic = "有 17 类标签" if "semantic_17class" in result["metrics"] else "类别无关"
        preprocess = m.get("preprocessing", "适配器使用原生预处理，尺寸未单独记录")
        if isinstance(preprocess, dict):
            preprocess = f"长边 {preprocess.get('long_edge', '未记录')}；RGB {preprocess.get('rgb_resize', '未记录')}；深度 {preprocess.get('depth_resize', '未记录')}；同步调整内参"
        input_rows.append([key, '、'.join(modalities) or '未记录', semantic,
                           m.get("max_candidates", m.get("native_topk", "未记录")),
                           m.get("nms", "未记录"), preprocess])
        weight_rows.append([key, f"`{m.get('checkpoint', options.get('checkpoint', '未记录'))}`",
                            f"`{m.get('checkpoint_sha256', '未记录')}`"])
    add("模型输入、权重与推理配置", table(
        ['模型', '输入信息', '语义输出', '候选框上限', 'NMS', '预处理'], input_rows) +
        "\n\n配置来自此次推理保存的 `models/<模型>/model.json`，不会用后来修改的配置文件替换历史信息。\n\n" +
        table(['模型', '检测权重路径', '检测权重 SHA-256'], weight_rows) +
        "\n\nWildDet3D 的 LingBot 辅助权重路径与校验值（若使用）也保存在该模型的 `model.json`。" +
        "\n\n3D-MOOD/WildDet3D 使用冻结协议中的固定文本提示，不根据当前帧 GT 选择类别。CuTR 使用前景通道 0，在原生 top-k 之前排除其他通道；不给 CuTR 从 GT 推导类别。已知重力仅在独立的重力辅助配置启用，是否启用以记录中的输入模态和 options 为准。")

    add("评测流程", """1. **冻结数据。** 固定扫描、帧、类别、GT 和可见性规则，记录 benchmark_id；所有模型使用同一阶段清单。
2. **缓存与转正。** 读取本地原始 RGB 并转正，同步变换相机内参；RGB-D 额外读取同帧实测深度。GT 的世界坐标转换在数据准备时完成。
3. **分环境推理。** 每组使用各自虚拟环境和权重，模型只加载一次，当前逐帧 batch=1、各组顺序运行；使用配置指定设备，不自动回退 CPU。
4. **统一框格式。** 适配器将尺寸顺序、旋转表示统一为米制相机坐标的中心、完整尺寸、旋转、类别和分数，并检查数值与旋转合法性。
5. **逐帧保存。** 原子写入压缩 JSON；协议、模型、输入和环境标识一致时可复用。失败或缺失预测不会当成“模型没有检出”计分。
6. **统一评分。** 汇集当前阶段所有预测，按置信度全局降序排序；只在同一帧内匹配真值。语义评测还要求类别相同。
7. **生成输出。** 保存完整指标、逐类 AP、逐扫描计数、计时、对比图与带框检查图。正式结果使用 full，pilot 用于检查链路。

适配器输入中不包含 GT 框。RGB-D 深度来自 ARKit 传感器，不使用 FARO 深度或 GT 框生成的深度。相机姿态用于转换 GT；显式重力实验只使用传感器姿态。""")

    add("指标定义与数学计算", r"""| 指标 | 正确匹配条件 | 判断内容 |
| --- | --- | --- |
| AP3D@0.25 | 三维 IoU ≥ 0.25 | 框的位置、尺寸与方向 |
| AP3D@0.50 | 三维 IoU ≥ 0.50 | 更严格的框重合要求 |
| APcenter@0.25m | 三维中心距离 ≤ 0.25 米 | 只判断中心位置 |
| APcenter@0.50m | 三维中心距离 ≤ 0.50 米 | 允许更大的位置误差 |
| APcenter@1.00m | 三维中心距离 ≤ 1.00 米 | 最宽松的中心位置要求 |

三维 IoU = 两框相交体积 / 两框并集体积；采用完整旋转框，不只比较地面投影。中心距离 d = √[(xp−xg)²+(yp−yg)²+(zp−zg)²]，单位米。中心距离不能从 IoU 阈值换算；二者只是替换了匹配条件，其余 AP 流程相同。

按分数降序处理预测，每个预测和每个 GT 最多匹配一次。IoU 选择达标未占用框中 IoU 最大的，中心距离选择距离最小的；优先有效 GT，其次匹配尚未占用的 ignore GT。匹配有效 GT 记 TP，匹配 ignore GT 不记 TP/FP，其他预测记 FP。重复检测算 FP。ignore GT 不计入召回率分母。

纳入排序前 k 个未忽略预测时，Pₖ = TPₖ/(TPₖ+FPₖ)，Rₖ = TPₖ/N_GT。令 P_interp(r) = max{Pₖ : Rₖ ≥ r}，无法达到的召回率取 0。项目使用：

AP = (1/101) × Σ[j=0…100] P_interp(j/100)。报告分数 = 100 × AP。

**数学例子：** 有 3 个有效 GT，5 个预测按置信度排列为 TP、FP、TP、FP、FP。累计精确率为 1、1/2、2/3、1/2、2/5，召回率为 1/3、1/3、2/3、2/3、2/3。101 个采样点中，0.00～0.33 共 34 点精确率取 1；0.34～0.66 共 33 点取 2/3；其余 34 点取 0。因此 AP=(34+33×2/3)/101=0.554455，报告显示 55.45，而不是最终精确率 40 或召回率 66.67。

例如表里 16.29 表示 AP=0.1629，不表示 16.29% 的框 IoU 达标，也不表示每 100 个物体找到 16.29 个。置信度影响排序，不直接乘进分数。

RGB 主表将所有类别合为 object 计算几何 AP；语义表分别计算每类 AP，再对有有效 GT 的类别等权平均。没有有效 GT 的类别 AP 为 null，不当成 0 参与类别平均。""")

    add("RGB 几何 AP：主要比较", "输入相同帧的转正 RGB 和相机内参，匹配忽略类别。前两个模型使用固定类别文本，CuTR 为类别无关检测。\n\n" +
        comparison(results, ("3dmood", "wilddet3d", "cutr-rgb")))
    if "cutr-rgbd" in results:
        add("CuTR RGB-D：深度输入补充实验", "RGB-D 额外使用同帧 ARKit lowres_depth 实测深度。两组权重不同，分数差异同时包含训练与输入变化，不能当作同一权重添加深度的纯消融。\n\n" +
            comparison(results, ("cutr-rgb", "cutr-rgbd")))
    semantic_keys = [key for key, result in results.items() if "semantic_17class" in result["metrics"]]
    if semantic_keys:
        add("17 类语义 AP", "类别必须一致；对有有效 GT 的类别取 AP 均值。CuTR 没有这 17 类标签，不纳入该表。\n\n" + comparison(results, semantic_keys, "semantic_17class"))

    rows = []
    for cls in c["classes"]:
        row = [cls['name'] + ' / ' + cls['name_zh'], cls['valid_gt'],
               f"{100 * cls['valid_gt'] / total:.2f}%" if total else '—']
        for key in semantic_keys:
            metric = next(r for r in results[key]['metrics']['semantic_17class']['full']['per_class']
                          if r['class_id'] == cls['id'])
            row.extend(percent(metric[kind][threshold]['ap']) for kind, threshold in
                       [('iou', '0.25'), ('iou', '0.5')])
        rows.append(row)
    add("类别分布与逐类成绩", table(['类别', '有效 GT', '有效 GT 占比', *[
        key + ' ' + metric for key in semantic_keys for metric in ('IoU AP25', 'IoU AP50')]], rows) +
        "\n\n类别分布不均衡，类别无关主表更受常见物体影响。逐类表有助于识别模型在哪些类别表现较好；全部五项逐类 AP 见 `metrics/*_classes.csv`。")

    add("官方训练 visit 排除后的 RGB 几何 AP", f"""已知官方训练/验证重叠 visit：{c['official_overlap']['visit_ids']}。当前阶段排除扫描 {c['excluded_scan_ids']} 的 {c['excluded_frames']} 帧，剩余 {c['frames'] - c['excluded_frames']:,} 帧，重新排序并计算 AP。

{comparison(results, ('3dmood', 'wilddet3d', 'cutr-rgb'), scope='official_training_visit_disjoint')}

该审计只依据 ARKitScenes 官方元数据，不能证明所有权重未训练过其他评测场景。完整 JSON 还保存其他模型/语义模式对应的排除重叠结果。""")
    if "cutr-rgb-gravity" in results:
        add("CuTR RGB + 已知重力辅助实验", "重力来自传感器相机姿态，额外信息单独比较，不混入纯 RGB 主表。\n\n" +
            comparison(results, ("cutr-rgb", "cutr-rgb-gravity")))

    add("公平性、适用范围与结果解读", table(['因素', '当前处理及影响'], [
        ['一致的部分', '相同阶段帧清单、GT、坐标/米制单位、可见性筛选、AP 与匹配规则；模型接口不接收 GT 框'],
        ['输入模态', 'RGB-D 使用额外深度，重力组使用额外姿态信息；均单独列出，不作无条件总排名'],
        ['类别范围', '有文本提示的模型目标范围为固定 17 类；CuTR 的通用物体可能超出标注范围，未标注物体可能计为误报'],
        ['候选数与后处理', '实际上限及 NMS 见配置表，没有统一成相同数量；会影响召回率，比较的是完整推理流程'],
        ['分辨率与预处理', '原始图像相同，模型缩放/填充不同；低分辨率放大不能补充真实细节，计算量也不同'],
        ['重力与框方向', '默认 CuTR 不输入已知重力；图片转正不等于相机无俯仰/侧倾，相机直立假设可能影响方向与 IoU'],
        ['可见性筛选', '统一深度可见性规则排除不满足条件的目标，成绩不覆盖所有遮挡/小目标；RGB-D 还额外接收同源深度'],
        ['训练与域差异', '各权重训练数据与训练任务不同，完整训练重叠尚未审计；低分不能直接归因于模型本身差'],
        ['统计不确定性', '未计算置信区间或显著性；相邻帧与同一 visit 相关，不能将帧当成完全独立样本；小分差不证明显著更好'],
        ['候选分数', 'AP 使用导出的完整候选，不使用显示阈值；不同模型的分数尺度不统一，统一显示阈值不代表相同置信程度'],
    ]) + "\n\n本报告是明确协议下的工程评测。若要比较模型能力的因果差异，需要另做输入、候选规则、训练重叠和重力等受控实验，不能由当前排名直接推出。")

    runtime_rows = []
    for key, result in results.items():
        runtime = result['runtime']
        device = result['model']['provenance'].get('device', {})
        peak = runtime.get('peak_memory_bytes')
        runtime_rows.append([key, device.get('name', device.get('device', '未记录')),
            f"{runtime['mean_predict_seconds']:.4f}", f"{runtime['p95_predict_seconds']:.4f}",
            f"{runtime['fps_predict']:.2f}", f"{peak / 1024**3:.2f}" if peak is not None else '—',
            c['model_prediction_counts'][key]])
    add("运行性能与计算位置", table(['模型', '推理设备', '平均秒/帧', 'P95 秒/帧', 'FPS', '峰值已分配显存 GiB', '预测框总数'], runtime_rows) +
        "\n\n模型前向在记录的设备上执行（本机 cuda 接口对应 PPU）；数据读取、部分预处理/后处理及 AP 评分在 CPU 上执行。"
        "单帧计时包含原生预处理、前向、三维框导出和数值校验，设备同步测量，不能当作纯 PPU 内核时间。"
        "输入解码独立记录，模型加载、预测文件写盘、AP 评分不计入上述 FPS。排除每次加载后的首个推理帧；如果只有首帧则在 JSON 标明回退。"
        "P95 表示约 95% 的样本耗时不超过该值；显存是 PyTorch 最大已分配量，不是设备总占用。")

    artifact_rows = [
        ('report.md', '中文报告；AP 数值为百分数'),
        ('report_context.json', '本阶段数据统计、类别分布、忽略原因、文件位置和检查图片索引'),
        ('comparison_rgb.csv / comparison_rgbd.csv / comparison_17class.csv / comparison_visit_disjoint.csv', '各组对比；AP 使用 0～1，不能与报告百分数混用'),
        ('metrics/<模型>.json', '完整几何/语义 AP、TP、FP、忽略预测数量、GT 数量、数据标识、模型记录与计时'),
        ('metrics/*_classes.csv', '逐类 GT/预测数量及五项 AP；无语义标签的模型只有 object 几何行'),
        ('metrics/*_scan_counts.csv', '逐扫描帧数、有效 GT、忽略 GT、预测数量；不是逐扫描 AP'),
        ('runtime.csv', '平均耗时、P95、FPS、峰值已分配显存'),
        ('figures/geometry_ap.png / .svg', 'RGB 主表五项 AP 柱状图'),
        ('visualizations/<模型>/*.png', 'RGB 上叠加三维框投影；绿色有效 GT、红色预测'),
        ('../models/<模型>/predictions/<扫描ID>/*.json.gz', '逐帧中心、尺寸、旋转、类别、分数、标识和计时，支持复查/续跑'),
        ('../models/<模型>/model.json', '检测/辅助权重摘要、输入配置、设备、环境版本、源码摘要'),
        ('logs/<模型>.log / *_task.json / *_inference.json', '日志、运行参数、完成状态、执行/复用帧数与总耗时'),
        ('selection.json / summary.json', '本阶段实际帧 ID 清单与完成汇总'),
        ('../experiment.json / ../status.json', '整次实验设置与运行状态；pilot/full 分别保存'),
    ]
    add("输出内容与文件索引", f"输出根目录：`{root}`；本阶段目录：`{root}/{c['phase']}`。pilot 为试评测，full 为全量，模型逐帧缓存由两阶段共享。\n\n" +
        table(['文件/目录（相对本阶段目录）', '内容'], artifact_rows) +
        f"\n\n本阶段 {len(results)} 组共完成 {c['model_frame_results']:,} 个模型—帧结果，各组 {c['frames']:,} 帧；"
        "记录空 detections 表示该帧合法地没有预测，缺失文件不等于空预测。"
        "当前没有生成逐帧 TP/FP 匹配明细、PR 曲线、逐扫描 AP、视频或交互式三维查看器，也没有自动进行失败原因诊断。")

    image_rows = []
    grouped = {}
    for path in c['visualizations']:
        p = Path(path)
        grouped.setdefault(p.stem, {})[p.parent.name] = path
    for frame_name, paths in grouped.items():
        image_rows.append([frame_name, *[f'[查看]({paths[key]})' if key in paths else '—' for key in results]])
    gallery = table(['帧', *results], image_rows) if image_rows else '本阶段没有保存检查图片。'
    add("带框图：查看方式与限制", f"""实际保存 {len(c['visualizations'])} 张图片，覆盖 {len(grouped)} 个相同帧，各模型逐帧对照。

选图规则：按本阶段清单的扫描顺序取前最多 12 个扫描，各取其采样帧列表中间位置。它们是检查样例，不是随机代表性样本或自动挑选的最佳/最差案例，也不覆盖所有评测帧。

绿色为有效 GT，红色为预测；红框只显示分数 ≥0.1 的候选。各模型分数尺度不同，统一显示阈值可能让某模型看起来框很少，不能据此直接比较召回率。该阈值不影响 AP。图片是三维框投影到 RGB 的结果，不是二维 IoU 的评分依据。

{gallery}
""")
    if c['visualizations']:
        examples = {}
        for path in c['visualizations']:
            examples.setdefault(Path(path).parent.name, path)
        sections.append("\n\n".join(f"### {key}：检查样例\n\n![{key} 带框检查图]({path})" for key, path in examples.items()))
    sections.append("![RGB 几何 AP](figures/geometry_ap.png)")
    return f"# {title}\n\n{intro}\n\n" + "\n\n".join(sections) + "\n"
