"""Detailed reports for supplied-box estimation and autonomous text detection."""

from collections import Counter
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .benchmark_report import write_csv
from .benchmark_worker import prediction_path, read_prediction
from .common import atomic_text, read_json, write_json
from .report_document import CLASS_NAMES_ZH, percent, table, values
from .schema import CLASSES
from .suite_io import GT_MODELS, TEXT_MODELS, frame_path, read_record
from .suite_score import NAMES, THRESHOLDS
from .visualize import _draw


def inference_speed(result):
    """Derive throughput from recorded compute time, without rerunning AP."""
    timing = result['timing']
    frames = result['frames']
    predict, frontend, depth = [float(timing[name]) for name in
        ('total_predict_seconds', 'total_frontend_seconds', 'total_depth_seconds')]
    if frames < 0 or any(not math.isfinite(t) or t < 0 for t in (predict, frontend, depth)):
        raise ValueError('速度统计要求非负帧数和有限的非负计时')
    pipeline = predict + frontend + depth
    targets = (result.get('conditional') or {}).get('targets')
    return dict(frames=frames, targets=targets, predict_seconds=predict,
                frontend_seconds=frontend, depth_seconds=depth, pipeline_seconds=pipeline,
                mean_predict_seconds_per_frame=predict/frames if frames and predict else None,
                fps_predict=frames/predict if frames and predict else None,
                mean_pipeline_seconds_per_frame=pipeline/frames if frames and pipeline else None,
                fps_pipeline=frames/pipeline if frames and pipeline else None,
                mean_predict_seconds_per_target=predict/targets if targets and predict else None,
                mean_pipeline_seconds_per_target=pipeline/targets if targets and pipeline else None,
                historical_reuse=timing['historical_reuse'], includes_warmup=True,
                excludes_io_initialization_and_scoring=True)


def speed_value(value):
    return '—' if value is None else f'{value:.4f}'


def visualizations(config, root, track, phase, frames, gt, results):
    grouped = {}
    for f in frames: grouped.setdefault(f['video_id'], []).append(f)
    chosen = [fs[len(fs)//2] for fs in grouped.values()][:12]
    images = []
    cache = {r['frame_id']: r for r in read_json(root/track/'input_manifest.json')}
    for frame in chosen:
        data = cache[frame['frame_id']]
        h,w = frame['image_hw']
        for key, result in results.items():
            model = result['model']
            if key in ('3dmood','wilddet3d'):
                origin = Path(model['reuse']['model_root']) if model.get('reuse') else root/track/'models'/key
                record = read_prediction(prediction_path(origin,frame),frame,model['signature'],result['identity']['benchmark_id'],True)
                detections = record['detections']
            else:
                record = read_record(frame_path(root/track/'models'/key/'predictions',frame),frame['frame_id'],model['signature'])
                detections = [i['detection'] for i in record['items'] if i['status']=='ok']
            with Image.open(data['rgb_path']) as f: image=f.convert('RGB').resize((w*3,h*3))
            draw=ImageDraw.Draw(image)
            for box in gt[frame['frame_id']]:
                if not box.get('ignore',False): _draw(draw,box,frame['intrinsics'],w,h,'lime',3,'GT '+CLASSES[box['class_id']])
            for box in detections:
                if track=='text' and box['score']<config['display_score']:continue
                name='object' if box['class_id'] is None else CLASSES[box['class_id']]
                _draw(draw,box,frame['intrinsics'],w,h,'red',3,f"{name} {box['score']:.2f}")
            path=root/track/phase/'visualizations'/key/(Path(frame['rgb_entry']).stem+'.png')
            path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
            images.append(str(path.relative_to(root/track/phase)))
    return images


def assemble(config, root, track, phase, frames, gt, protocol, audit):
    root=Path(root);directory=root/track/phase
    keys=GT_MODELS if track=='gt-box' else TEXT_MODELS
    results={key:read_json(directory/'metrics'/(key+'.json')) for key in keys}
    if any(not r['complete'] or r['frames']!=len(frames) for r in results.values()):
        raise ValueError('该阶段模型未全部完成，不能生成最终对比')
    images=visualizations(config,root,track,phase,frames,gt,results)
    valid=Counter(b['class_id'] for f in frames for b in gt[f['frame_id']] if not b.get('ignore',False))
    ignored=sum(b.get('ignore',False) for f in frames for b in gt[f['frame_id']])
    empty=sum(not any(not b.get('ignore',False) for b in gt[f['frame_id']]) for f in frames)
    scans=len({f['video_id'] for f in frames});visits=len({f['visit_id'] for f in frames})
    rows=[]
    for key,r in results.items():
        if track=='gt-box':rows.append([key,*r['conditional']['success'],r['conditional']['failure_rate']])
        else:rows.append([key,*values(r['metrics']['geometry']['full'])])
    ap_names=['AP3D@0.25','AP3D@0.50','APcenter@0.25m','APcenter@0.50m','APcenter@1.00m']
    headers=['model',*(NAMES if track=='gt-box' else ap_names)]+(['failure_rate'] if track=='gt-box' else [])
    write_csv(directory/'comparison.csv',headers,rows)
    write_csv(directory/'comparison_ap.csv',['model',*ap_names],[[k,*values(r['metrics']['geometry']['full'])] for k,r in results.items()])
    speeds = {key: inference_speed(result) for key, result in results.items()}
    speed_fields = ['frames','targets','predict_seconds','frontend_seconds','depth_seconds','pipeline_seconds',
                    'mean_predict_seconds_per_frame','fps_predict','mean_pipeline_seconds_per_frame','fps_pipeline',
                    'mean_predict_seconds_per_target','mean_pipeline_seconds_per_target','historical_reuse',
                    'includes_warmup','excludes_io_initialization_and_scoring']
    write_csv(directory/'runtime.csv',['model',*speed_fields],
              [[key,*[speed[field] for field in speed_fields]] for key,speed in speeds.items()])
    write_json(directory/'speed.json',speeds)
    context=dict(track=track,phase=phase,benchmark_id=protocol['benchmark_id'],frames=len(frames),scans=scans,visits=visits,
                 valid_gt=sum(valid.values()),ignored_gt=ignored,empty_valid_gt_frames=empty,
                 classes={CLASSES[k]:n for k,n in valid.items()},visualizations=images,
                 models=list(keys),settings=config,source_protocol=protocol['settings'],inference_speed=speeds)
    write_json(directory/'report_context.json',context)
    title='RGB＋GT 投影二维框：条件三维估计' if track=='gt-box' else 'RGB＋固定类别文本：自主三维检测'
    sections=[]
    def add(title,body):sections.append('## '+title+'\n\n'+body)
    add('数据范围与文件用途',f"""本阶段 {len(frames):,} 帧、{scans} 个扫描、{visits} 个 visit，有效帧—物体标注 {sum(valid.values()):,}，忽略标注 {ignored:,}，无有效 GT 的帧 {empty:,}。同一物体跨帧重复计数。

数据源为 `{protocol['settings']['dataset_root']}` 的 ARKitScenes 3dod 验证集。冻结清单 `{config['run_dir']}`，原始 RGB 为 256×192，按时间每扫描最多 {protocol['settings']['frames_per_scan']} 帧、候选间隔至少 {protocol['settings']['minimum_interval_seconds']} 秒；当前阶段选择见 selection.json。图像转正同时更新相机内参，GT 转换到相同米制相机坐标系。

{table(['文件','作用','使用阶段'],[
 ['protocol.json','冻结参数、类别、数据摘要和 benchmark_id','输入/评分核验'],
 ['frames.jsonl','帧编号、时间戳、RGB/深度条目、内参、转正与姿态参数','选帧与数据准备'],
 ['groundtruth/<扫描>.json','每帧 GT 中心、尺寸、旋转、类别、ignore 与 projected_bbox','评分；仅框提示组取二维投影'],
 ['dataset_audit.json','场景分布与官方训练/验证 visit 重叠','数据审计与排除重叠评分'],
 [config['image_cache'],'本地 RGB/毫米深度 PNG 与校验清单','模型输入读取'],
 ['../input_manifest.json','模型允许接收的输入元数据','逐帧输入；文本清单无 GT 框或目标数量'],
])}

有效 GT 要求视野内且框内观测深度点严格大于 {protocol['settings']['visible_points_strictly_greater_than']}，深度采样步长 {protocol['settings']['depth_stride']}。忽略规则统一，但成绩不覆盖所有遮挡/微小目标。GT 检查不代表向 RGB 模型提供深度。
""")
    if track=='gt-box':
        add('三组输入与流程',table(['配置','二维定位/掩码','深度'],[
            ['wilddet3d-box','GT 投影框；原生 geometric 模式逐目标估计；无类别提示','WildDet3D 原生 RGB 几何流程'],
            ['unipose9d-box-rgb','相同 GT 投影框→SAM2 预测掩码','MoGe 从 RGB 预测深度'],
            ['unipose9d-box-rgbd','与 RGB 组共享 SAM2 掩码、姿态权重及采样预算','ARKit lowres_depth 实测深度，毫米转米'],
        ])+'\n\n每个有效目标提供二维位置，不提供 GT 三维中心、尺寸、方向或类别。投影框来自 GT，因此这是明确允许真值二维位置的条件实验，不能当作自主寻找物体的成绩。3D-MOOD 当前推理不读取外部二维框，不参加本组。')
        add('主指标：成功率与失败率',table(['配置',*NAMES,'失败率'],[[k,*map(percent,r['conditional']['success']),percent(r['conditional']['failure_rate'])] for k,r in results.items()])+
            '\n\n成功率=指定目标的三维预测达标数/全部输入目标数。固定关联输入框对应的原 GT 目标，不允许改匹配到另一个更接近的目标提高成绩。空掩码、深度点不足、RANSAC/几何估计失败均保留在分母、记为不达标；缺失文件表示任务未完成，不能当作失败目标。均值 IoU 将失败记为 0，成功估计中心误差另行保存。')
        add('补充 AP：排序能力',table(['配置',*NAMES],[[k,*map(percent,values(r['metrics']['geometry']['full']))] for k,r in results.items()])+
            '\n\nWildDet3D 使用原生预测分数；UniPose9D 使用 inlier_ratio 作为明确披露的质量代理分数，不是校准的正确概率。补充 AP 在同一帧有效目标之间按分数降序一对一匹配，可能与原目标成功率不同。AP 评价几何与排序，成功率直接评价已给定位置的几何估计；不把两者混为一个指标，也不将代理分数 0.7 与检测置信度 0.7 等同。')
        failure_rows=[[k,reason,n] for k,r in results.items() for reason,n in r['conditional']['failure_reasons'].items()]
        add('失败原因',table(['配置','原因','次数'],failure_rows) if failure_rows else '本阶段未记录预期目标估计失败。')
        add('逐类成功率',table(['类别','有效目标',*keys],[[CLASSES[c]+' / '+CLASS_NAMES_ZH[c],valid[c],
            *[percent(results[k]['conditional']['per_class'][c]['success'][0]) for k in keys]] for c in range(len(CLASSES))])+
            '\n\n该表显示 IoU≥0.25 的逐类成功率；完整五项成功率及失败数见各模型指标 JSON，逐目标 IoU、中心距离和失败原因见 *_targets.csv。')
    else:
        add('三组输入与流程',table(['配置','模型项目内的定位流程','额外深度'],[
            ['3dmood','固定 17 类文本→原生 GroundingDINO/三维预测','无实测深度'],
            ['wilddet3d','固定 17 类文本→原生 SAM3/三维预测','无实测深度'],
            ['unipose9d-text-rgb','GroundingDINO 检出全部达标实例→SAM2 分割→MoGe→UniPose9D','仅 RGB 预测深度'],
        ])+f"\n\n所有图片都使用同一固定词表。文本 worker 输入清单没有 GT 位置、目标数量、类别真值或实测深度路径。UniPose9D 继承前置 GroundingDINO 的分数与预测类别；保留官方框阈值 {config['frontend']['box_threshold']}、文本阈值 {config['frontend']['text_threshold']}，最多 {config['frontend']['max_detections']} 个候选，而不是 Demo 的单个最高分物体。类别归属由固定文本的 token 范围确定，不从 GT 获取。无检出帧正常记录为空，依然参与漏检/误报评分。")
        add('主要指标：几何 AP',table(['配置',*NAMES],[[k,*map(percent,values(r['metrics']['geometry']['full']))] for k,r in results.items()])+
            '\n\n几何匹配忽略类别，按全阶段置信度排序、同帧一对一匹配，优先有效 GT，其次匹配尚未占用的 ignore GT；其他预测算 FP，重复检测算 FP。主 AP 使用全部导出候选，不额外按 0.7 删除。')
        add('17 类语义 AP',table(['配置',*NAMES],[[k,*map(percent,values(r['metrics']['semantic_17class']['full']))] for k,r in results.items()])+
            '\n\n语义匹配还要求类别一致，对有有效 GT 的类别 AP 等权平均。原生模型内部检测/NMS与候选规则不同，报告比较完整流程，不是统一二维候选的受控消融。')
        point_rows=[[k,NAMES[i],percent(v['precision']),percent(v['recall']),percent(v['f1']),v['tp'],v['fp']]
                    for k,r in results.items() for i,v in enumerate(r['workpoint']['values'])]
        add('置信度≥0.7 使用效果',table(['配置','匹配条件','精确率','召回率','F1','TP','FP'],point_rows)+
            '\n\n这是额外工作点，不替代标准 AP。不同模型的分数尺度未统一校准，同为 0.7 不代表同样的正确概率；删除低分候选可降低误报，也会降低召回。')
        add('逐类分布与成绩',table(['类别','有效 GT',*keys],[[CLASSES[c]+' / '+CLASS_NAMES_ZH[c],valid[c],
            *[percent(results[k]['metrics']['semantic_17class']['full']['per_class'][c]['iou']['0.25']['ap']) for k in keys]] for c in range(len(CLASSES))])+
            '\n\n该表为 IoU AP25。类别分布不均衡，全部五项逐类 AP 见 *_classes.csv。')
    add('AP 原理与阈值',"""3D IoU=相交体积/并集体积，计算完整旋转框；中心距离=√[(xp−xg)²+(yp−yg)²+(zp−zg)²]，单位米。两个 IoU 阈值是 0.25、0.50；中心距离阈值是 0.25、0.50、1.00 米。中心距离不判断尺寸和方向，也不能从 IoU 数值换算。

P=累计 TP/(TP+FP)，R=累计 TP/有效 GT 数。P_interp(r) 取召回率至少达到 r 时的最高精确率，达不到则为 0。AP 是 r=0、0.01…1 共 101 点的插值精确率平均。

例如 3 个有效目标、5 个排序预测 TP/FP/TP/FP/FP：34 个点取 1、33 个点取 2/3、34 个点取 0，AP=(34+33×2/3)/101=0.554455。报告显示 55.45，CSV/JSON 存 0.554455；这不是最终正确框比例。类别没有有效 GT 时 AP 为 null，不作为 0 参与平均。""")
    add('排除官方训练 visit 重叠后的补充 AP',table(['配置',*NAMES],[[k,*map(percent,values(r['metrics']['geometry']['official_training_visit_disjoint']))] for k,r in results.items()])+
        f"\n\n官方重叠 visit 为 {audit['official_training_visit_overlap']['visit_ids']}，扫描为 {audit['official_training_visit_overlap']['validation_scan_ids']}。本阶段排除 {next(iter(results.values()))['excluded_frames']} 帧。该审计不能证明各权重全部训练集与评测场景无重叠。")
    add('公平性与执行记录',"""两个任务独立报告：GT 框提示组已经知道目标位置，文本组必须自行寻找位置，不能横向混合分数。所有组使用同一冻结数据及可见性规则，但 RGB-D 有额外传感器深度，必须独立列出。RGB 与 RGB-D UniPose9D 共享权重、掩码和预算；相机内参统一使用数据集 K，MoGe 预测 K 只留作诊断。

原始图像分辨率低，放大或裁剪不能恢复细节；投影框不等于精确可见掩码。SAM2、预测深度、遮挡与有效点不足都会影响姿态结果。模型训练范围、候选规则及预处理不完全相同，成绩不能直接当作论文设置的结果。未计算置信区间，同一场景的相邻帧也不独立，微小分差不能证明显著更好。

每组模型加载一次，逐帧/逐目标顺序执行，缓存 SAM2 掩码和整帧 MoGe 深度，按输入、代码、权重、环境标识校验续跑。随机种子按帧和目标派生，RGB/RGB-D 相同目标共享种子。网络/设备/模型异常停止任务；明确的目标输入/估计失败记为失败并保留分母。"""+
        '\n\n'+table(['配置','预测框数','估计失败数','历史预测复用'],[[k,r['predictions'],r['failures'],r['timing']['historical_reuse']] for k,r in results.items()]))
    add('参数与权重',table(['配置','模型/参数记录'],[[k,'[model.json](../models/'+k+'/model.json)'] for k in keys]))
    speed_body = table(['配置','模型平均秒/帧','模型 FPS','完整流程平均秒/帧','完整流程 FPS','计时来源'],
        [[k,*[speed_value(s[field]) for field in ('mean_predict_seconds_per_frame','fps_predict',
            'mean_pipeline_seconds_per_frame','fps_pipeline')], '历史测量' if s['historical_reuse'] else '本次预测记录']
         for k,s in speeds.items()])
    if track == 'gt-box':
        speed_body += '\n\n'+table(['配置','输入目标数（含失败）','模型平均秒/目标','完整流程平均秒/目标'],
            [[k,s['targets'],speed_value(s['mean_predict_seconds_per_target']),
              speed_value(s['mean_pipeline_seconds_per_target'])] for k,s in speeds.items()])
    add('推理速度',speed_body+
        '\n\n'+table(['配置','模型预测秒','定位/掩码秒','预测深度秒'],[[k,f"{r['timing']['total_predict_seconds']:.2f}",
            f"{r['timing']['total_frontend_seconds']:.2f}",f"{r['timing']['total_depth_seconds']:.2f}"] for k,r in results.items()])+
            '\n\nFPS 表示平均每秒处理多少帧，越高越快；秒/帧表示平均处理一帧花多久，越低越快。完整流程时间 T=模型预测秒＋定位/掩码秒＋预测深度秒；平均秒/帧=T/全部帧数，FPS=全部帧数/T。例如 100 帧的模型预测花 40 秒、定位/掩码花 10 秒、深度花 50 秒：模型 FPS=2.5，完整流程 FPS=1.0，完整流程平均 1 秒/帧。这里使用总时间的比值，不对逐帧 FPS 取平均。'
            '\n\n框提示组还提供平均秒/目标=对应总时间/全部输入目标数，包含失败目标；掩码和整帧深度的共享成本也按全部目标分摊。每帧物体越多通常越慢；提前失败会缩短耗时，因此速度必须结合成功率和失败率判断。文本组不会用检出框数计算目标吞吐量，以免漏检较多的模型看起来更快。'
            '\n\n预测时间同步设备，包含模型预处理、前向和数值导出；读取、写盘、模型初始化和 CPU 指标计算不混入预测秒。模型 FPS 的处理边界见各组流程：3D-MOOD/WildDet3D 原生文本流程包含其内部定位，UniPose9D 模型列仅为三维估计，定位/掩码和深度另列；跨模型优先比较完整流程 FPS。SAM2/深度缓存复用时仍保留产生缓存的计时，作为对应完整流程成本，不等于本次墙钟执行时间。RGB-D 不计 MoGe 深度时间。所有帧（包括无目标帧）参与帧级统计；保留首次推理，不是排除预热后的纯前向速度。没有有效计时或目标时显示 —，不报告无限 FPS。历史基线保留原测量，不能当成此次重新测量的设备性能。PPU 负责模型前向，CPU 负责读写、部分处理和评分。')
    add('输出文件与检查图片',table(['文件/目录','内容'],[
        ['report.md / report_context.json','中文报告及数据/配置/图片索引'],
        ['comparison.csv / comparison_ap.csv','主指标与几何 AP，数值 0～1；本报告乘 100'],
        ['metrics/<配置>.json','完整 AP、成功率、工作点、失败原因、来源和计时'],
        ['metrics/*_targets.csv','框提示组逐目标对应的 IoU、中心误差、失败及达标情况'],
        ['metrics/*_classes.csv','逐类 AP、有效/忽略 GT 和预测数'],
        ['../models/<配置>/predictions/','逐帧三维预测和逐目标状态；历史复用基线位置见 model.json 的 reuse'],
        ['../../aux/','预测掩码、预测深度及二维候选缓存；来自模型，不是真值掩码'],
        ['logs/ / *_inference.json','子进程日志和执行/复用完成记录'],
        ['runtime.csv / speed.json','模型/完整流程 FPS、平均秒/帧、框提示组平均秒/目标及计时来源'],
        ['selection.json / summary.json','实际帧清单与阶段完成汇总'],
    ])+f"\n\n保存 {len(images)} 张带框检查图：按清单前最多 12 个扫描，各取采样列表中间帧；相同图片比较各配置。绿框为有效 GT，红框为预测。框提示组展示全部输出，文本组显示分数≥{config['display_score']}，显示过滤不影响主 AP。选图不是随机代表性抽样，不覆盖全部帧；不同模型分数尺度不同，不能通过红框数量直接比较召回。\n\n"+
        table(['图片'],[[f'[查看]({p})'] for p in images]))
    atomic_text(directory/'report.md',f"# {title}：{phase}\n\n全部三组已完成。benchmark_id：`{protocol['benchmark_id']}`。\n\n"+'\n\n'.join(sections)+'\n')
    write_json(directory/'summary.json',dict(complete=True,models=list(keys),frames=len(frames),track=track,phase=phase))
    atomic_text(root/'README.md','# 两类三维评测\n\n'+f'当前完成 [{track}/{phase}]({track}/{phase}/report.md)。运行状态见 status.json 和 worker_status.json。\n')
