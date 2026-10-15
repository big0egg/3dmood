"""Chinese, self-contained reports for both HOT3D tracks."""

import os
from collections import Counter
from pathlib import Path

from PIL import Image,ImageDraw

from ..benchmark_report import write_csv
from ..common import atomic_text,read_json,write_json
from ..report_document import percent,table
from ..visualize import _draw
from .common import TRACKS,read_record,record_path
from .dataset import paths
from .metrics import label,thresholds


def assemble(config,track,frames,gt,protocol,audit):
    base,data,root=paths(config);directory=root/track/'full';classes=protocol['classes']
    results={k:read_json(directory/'metrics'/(k+'.json')) for k in TRACKS[track]}
    if any(not r['complete'] or r['frames']!=len(frames) for r in results.values()):raise ValueError('配置未全部完成，不生成最终对比')
    limits=thresholds(config);ap_names=[label(k,t) for k,t in limits];success_names=[label(k,t,False) for k,t in limits]
    primary=[label(k,t,track!='gt-box') for k,t in limits if k=='iou' or t<=.1]
    comparison=[]
    for k,r in results.items():
        if track=='gt-box':comparison.append([k,*[r['conditional']['success'][n] for n in success_names],r['conditional']['failure_rate']])
        else:comparison.append([k,*[r['geometry']['summary'][n] for n in ap_names]])
    write_csv(directory/'comparison.csv',['model',*(success_names if track=='gt-box' else ap_names),*(['failure_rate'] if track=='gt-box' else [])],comparison)
    write_csv(directory/'comparison_ap.csv',['model',*ap_names],[[k,*[r['geometry']['summary'][n] for n in ap_names]] for k,r in results.items()])
    speeds={k:r['speed'] for k,r in results.items()};write_json(directory/'speed.json',speeds)
    fields=list(next(iter(speeds.values())))
    write_csv(directory/'runtime.csv',['model',*fields],[[k,*[r[n] for n in fields]] for k,r in speeds.items()])
    chosen={}
    for f in frames:chosen.setdefault(f['sequence_id'],[]).append(f)
    chosen=[group[len(group)//2] for group in chosen.values()][:12]
    images=[]
    for f in chosen:
        for key,result in results.items():
            record=read_record(record_path(root/track/'models'/key/'predictions',f),f,result['model']['signature'])
            with Image.open(data/f['rgb_relative']) as im:image=im.convert('RGB')
            draw=ImageDraw.Draw(image);h,w=f['image_hw']
            for b in gt[f['frame_id']]:
                if not b['ignore']:_draw(draw,b,f['intrinsics'],w,h,'lime',1,'GT '+b['object_name'])
            for b in record['detections']:
                if track=='text' and b['score']<config['display_score']:continue
                name='object' if b['class_id'] is None else classes[b['class_id']]['name']
                _draw(draw,b,f['intrinsics'],w,h,'red',1,f'{name} {b["score"]:.2f}')
            path=directory/'visualizations'/key/(f['clip_id']+'_'+f['frame_key']+'.png')
            path.parent.mkdir(parents=True,exist_ok=True);image.save(path)
            images.append(str(path.relative_to(directory)))
    title='RGB＋GT投影二维框：条件三维估计' if track=='gt-box' else 'RGB＋固定类别文本：自主三维检测'
    sections=[]
    def add(title,body):sections.append('## '+title+'\n\n'+body)
    add('数据范围',f"""来源：`{config['dataset_root']}/Clips/train_aria`。本阶段 {len(frames):,} 帧、{audit['clips']} 个5秒片段、{audit['source_sequences']} 段原始录制、{audit['participants']} 位参与者；33个物体资产合并为 {len(classes)} 个语义类别。有效帧—目标 {audit['valid_targets']:,}，忽略目标 {audit['ignored_targets']:,}，无有效目标帧 {audit['empty_valid_gt_frames']:,}。同一物体跨帧重复计数，片段不等于独立场景。

官方训练划分在这里作为外部评测数据；不是官方隐藏测试集，也不是BOP六维位姿榜单协议。各模型使用现有权重；尚未证明其全部预训练数据与HOT3D无重叠，不宣称严格零样本。按原始sequence/participant统计覆盖，未计算统计显著性或置信区间。

抽帧编号：{protocol['settings']['frame_indices']}。先按固定时间选帧，不按预测好坏筛选；RGB原始序列＋时间戳唯一。原始图片是JPEG，转换后的针孔图片无损保存PNG。几何覆盖只限33个建模物体，不代表场景所有物体有GT。""")
    add('参与者和原始录制覆盖',table(['参与者','采样帧数'],sorted(audit['frames_per_participant'].items()))+
        '\n\n原始录制逐项覆盖和有效目标数见sequence_coverage.csv；同一录制的相邻帧不能当作独立场景。')
    write_csv(directory/'sequence_coverage.csv',['sequence_id','frames','valid_targets'],
        [[seq,n,audit['valid_targets_per_sequence'].get(seq,0)] for seq,n in sorted(audit['frames_per_sequence'].items())])
    add('数据文件和用途',table(['文件','用途'],[
        [str(data/'protocol.json'),'冻结相机、抽帧、GT规则、数据和代码摘要'],
        [str(data/'frames.jsonl'),'帧清单、时间戳、参与者/录制编号、图片摘要、内参'],
        [str(data/'groundtruth'),'相机坐标三维框、物体ID、语义类别、ignore、投影框'],
        [str(data/'calibration'),'每帧原始鱼眼相机标定与虚拟相机变换'],
        [str(data/'rgb_fisheye')+' / rgb_pinhole','原始鱼眼图 / 模型实际使用的针孔图'],
        [str(data/'object_models/registry.json'),'GLB节点变换后的米制尺寸、局部中心、ID和语义映射'],
        ['../input_manifest.json','允许传入模型的输入；文本组不含GT位置、类别真值或数量'],
        [str(data/'dataset_audit.json'),'覆盖数量、忽略原因和几何检查结果'],
    ]))
    cam=protocol['settings']['camera']
    add('鱼眼、坐标与GT',f"""RGB流为214-1，源相机FISHEYE624。固定虚拟相机 {cam['width']}×{cam['height']}、水平视场角 {cam['horizontal_fov_degrees']}°，顺时针转正 {cam['clockwise_quarter_turns']}×90°。用完整15参数模型从目标针孔像素反查原鱼眼图片，图像与GT同时转换到右手OpenCV相机坐标（x向右、y向下、z向前），长度单位米。

GT框由物体模型局部轴向包围盒经逐帧刚体位姿变换得到；不是世界轴向包围盒，也不直接把物体原点当框中心。GLB节点包含0.001等缩放，已应用完整场景变换。T_camera_from_object=inv(T_world_from_camera)×T_world_from_object，随后应用转正相机旋转。二维提示框由最终三维框投影并裁剪到图片，未直接使用鱼眼图片上的矩形框。

有效目标要求虚拟图中amodal轮廓至少 {protocol['settings']['minimum_amodal_pixels']} 像素，原始鱼眼几何可见度≥{protocol['settings']['minimum_native_modeled_visibility']}，并有有效投影框。这里的可见度来自原鱼眼视图，不能当作重新计算的针孔遮挡比例；手臂、家具等未建模遮挡仍可能影响图像。官方SAM2估计的modal mask/visibility只作诊断，不参与主测试集筛选或模型输入。"""+
        '\n\n'+table(['排除原因','次数（同一目标可能有多个原因）'],list(audit['ignore_reasons'].items())))
    add('模型输入与公平性',table(['配置','允许输入及流程'],
        [['wilddet3d-box','统一RGB＋K＋GT投影二维框；原生geometric模式逐目标估计，无类别提示'],
         ['unipose9d-box-rgb','同一RGB＋K＋框→SAM2预测mask；MoGe从RGB预测深度→UniPose9D']]
        if track=='gt-box' else [
         ['3dmood','统一RGB＋K＋完整固定词表→原生检测/三维框'],
         ['wilddet3d','统一RGB＋K＋相同词表→原生SAM3/三维框'],
         ['unipose9d-text-rgb','相同词表→GroundingDINO二维框→SAM2→MoGe→UniPose9D']])+"""

GT三维框、GLB、GT mask和相机世界位姿仅供离线准备/评分；不交给模型。文本组不给GT框、该帧类别或数量；框提示组只给框和不含类别意义的目标编号。HOT3D本协议没有实测深度，未设置RGB-D组；GT渲染深度不作为模型输入。不同任务不混合排名。

每张图都使用全局词表，语义相同资产合并，非按GT挑选。模型内部候选/NMS规则不完全相同，比较完整流程。背景未建模同类物体可能被检测为未匹配预测并计FP，这一标注范围限制需结合图片检查；不按GT附近位置过滤预测。
""")
    if track=='gt-box':
        add('主指标：成功率',table(['配置',*primary,'失败率'],[[k,*[percent(r['conditional']['success'][n]) for n in primary],percent(r['conditional']['failure_rate'])] for k,r in results.items()])+"""

成功率=原输入目标的三维框达标数/全部输入目标数。固定关联原目标，不允许匹配到旁边物体；空mask、有效深度不足等失败也留在分母并记为不达标。输出框但IoU/中心距离不达标不计作“估计失败”。均值IoU含失败记0，中心误差中位数只统计生成有效框的目标。UniPose9D的inlier_ratio仅是补充AP的质量代理，非原生检测概率。
""")
        add('失败与误差',table(['配置','输入目标','失败数','平均IoU（含失败）','有效框中心误差中位数/米'],
            [[k,r['conditional']['targets'],r['failures'],r['conditional']['mean_iou_including_failures'],r['conditional']['median_center_error_m']] for k,r in results.items()])+
            '\n\n'+table(['配置','原因','次数'],[[k,reason,n] for k,r in results.items() for reason,n in r['conditional']['failure_reasons'].items()]))
    add('几何AP'+('（补充）' if track=='gt-box' else '（主指标）'),table(['配置',*[label(k,t) for k,t in limits if k=='iou' or t<=.1]],
        [[key,*[percent(r['geometry']['summary'][label(k,t)]) for k,t in limits if k=='iou' or t<=.1]] for key,r in results.items()]))
    add('保留的大距离阈值',table(['配置',*[label(k,t) for k,t in limits if k=='center' and t>.1]],
        [[key,*[percent(r['geometry']['summary'][label(k,t)]) for k,t in limits if k=='center' and t>.1]] for key,r in results.items()])+
        '\n\n这些阈值便于与旧实验查阅，但HOT3D多为小物体，25厘米、50厘米、1米可能过宽。不同数据集难度不同，不将分数直接合成排名。所有阈值运行前冻结。')
    if track=='text':
        add('语义AP与0.7工作点',table(['配置',*[label(k,t) for k,t in limits if k=='iou' or t<=.1]],
            [[key,*[percent(r['semantic']['summary'][label(k,t)]) for k,t in limits if k=='iou' or t<=.1]] for key,r in results.items()])+
            '\n\n'+table(['配置','条件','精确率','召回率','F1','TP','FP'],[[key,name,percent(v['precision']),percent(v['recall']),percent(v['f1']),v['tp'],v['fp']]
                for key,r in results.items() for name,v in r['workpoint']['per_class'][0]['values'].items()]))
    add('指标计算',"""IoU=完整旋转三维框相交体积/并集体积，位置、尺寸、方向都影响。中心距离为两个框中心的欧氏距离，忽略尺寸和方向。IoU阈值0.25/0.50，中心距离阈值见表。

AP按置信度降序，在同一帧一对一匹配，优先有效GT再匹配未占用ignore GT；重复/其他未匹配预测计FP。P=TP/(TP+FP)，R=TP/有效GT数，AP是101个召回位置上插值精确率的平均。几何AP不要求类别；语义AP要求类别一致，对有GT类别等权平均。框提示补充AP只匹配被提供的有效目标。

主AP使用全部导出候选；显示阈值只用于附加工作点和文本可视化，具体阈值见下表。各模型分数未统一校准。UniPose9D文本组继承前置检测器分数/类别，mask和姿态质量分数另存。
""")
    add('候选与展示阈值',table(['参数','值'],[['GroundingDINO框阈值',config['frontend']['box_threshold']],
        ['GroundingDINO文本阈值',config['frontend']['text_threshold']],['每帧最大候选数',config['frontend']['max_detections']],
        ['文本展示/附加工作点阈值',config['display_score']]]))
    def fmt(v):return '—' if v is None else f'{v:.4f}'
    add('推理速度与执行',table(['配置','模型秒/帧','模型FPS','完整流程秒/帧','完整流程FPS'],
        [[k,*[fmt(s[n]) for n in ('mean_predict_seconds_per_frame','fps_predict','mean_pipeline_seconds_per_frame','fps_pipeline')]] for k,s in speeds.items()])+
        ('\n\n'+table(['配置','模型秒/目标','完整流程秒/目标'],[[k,fmt(s['mean_predict_seconds_per_target']),fmt(s['mean_pipeline_seconds_per_target'])] for k,s in speeds.items()]) if track=='gt-box' else '')+"""

完整流程时间=模型预测＋外置定位/分割＋预测深度时间；FPS=全部帧数/该总时间，含无目标帧。设备同步后计时，包含预处理和数值导出，排除模型初始化、图片/缓存读写及CPU评分，保留首次推理。各模型边界不同：原生3D-MOOD/WildDet3D文本预测包含内部定位，UniPose9D外置步骤另计，因此跨模型优先比较完整流程。

共享深度/掩码即使复用，也按生成时记录的成本归属对应流程，不能把缓存读取速度当作推理速度。秒/目标以全部输入目标为分母，含失败并分摊整帧成本，需与成功率一起判断。本实验均为HOT3D新预测，没有使用ARKitScenes预测。中断续跑复用已经完成的帧；中断帧重算，固定每帧/目标种子。
""")
    counts=Counter(b['object_bop_id'] for bs in gt.values() for b in bs if not b['ignore'])
    add('类别和物体分布',table(['语义ID','英文提示词','有效目标数'],[[c['class_id'],c['prompt'],sum(not b['ignore'] and b['class_id']==c['class_id'] for bs in gt.values() for b in bs)] for c in classes])+
        '\n\n'+table(['物体BOP ID','有效目标数'],sorted(counts.items()))+
        '\n\n框提示组各物体成功率见模型JSON的conditional.per_object；文本组各物体0.7工作点几何召回见 *_object_recall_at_0.7.csv，不将资产ID当作通用文本检测类别。逐类AP见 *_classes.csv。')
    check_links=[]
    for f in chosen:
        check=data/'geometry_checks'/(f['clip_id']+'.png')
        first=protocol['settings']['frame_indices'][0]
        raw=data/'rgb_fisheye'/f['clip_id']/(f'{first:06d}.jpg')
        check_links.append([f['sequence_id'],f'[原鱼眼]({os.path.relpath(raw,directory)})',f'[针孔GT投影]({os.path.relpath(check,directory)})'])
    add('输出文件与检查图片',table(['文件','内容'],[
        ['comparison.csv / comparison_ap.csv','成功率或AP，CSV/JSON为0～1，报告乘100'],
        ['metrics/<模型>.json','完整指标、来源、失败原因和速度'],
        ['metrics/*_targets.csv','框提示逐目标的IoU、中心误差和达标情况'],
        ['metrics/*_classes.csv / *_object_recall_at_0.7.csv','文本逐类AP/逐物体几何召回'],
        ['runtime.csv / speed.json','秒/帧、FPS、秒/目标及原始计时'],
        ['../models/<模型>/predictions','全部帧三维预测；model.json固定权重、实现及环境'],
        ['../../aux','预测二维候选、mask和MoGe深度'],
        ['logs/ / *_inference.json','子进程日志和完成统计'],
        ['report_context.json / summary.json','机器可读报告上下文和完成标志'],
    ])+'\n\n'+table(['原始录制','源图','几何检查'],check_links)+
        '\n\n按清单前最多12段原始录制各取中间采样帧，绿框GT、红框预测；文本仅展示≥0.7。该选图不是代表性随机样本，不能仅从红框数量判断召回。\n\n'+
        table(['带框图'],[[f'[查看]({p})'] for p in images]))
    write_json(directory/'report_context.json',dict(benchmark_id=protocol['benchmark_id'],audit=audit,classes=classes,
        settings=config,models=list(results),visualizations=images,speed=speeds))
    atomic_text(directory/'report.md',f'# HOT3D：{title}\n\nbenchmark_id：`{protocol["benchmark_id"]}`。所有本组配置已完成。\n\n'+'\n\n'.join(sections)+'\n')
    write_json(directory/'summary.json',dict(complete=True,frames=len(frames),models=list(results),track=track,benchmark_id=protocol['benchmark_id']))
    links=[]
    for candidate in TRACKS:
        p=root/candidate/'full/summary.json'
        if p.exists() and read_json(p).get('complete'):links.append(f'- [{candidate}完整报告](runs/{config["data"]["name"]}/{candidate}/full/report.md)')
    atomic_text(base/'report.md','# HOT3D benchmarkv2\n\n两类任务分别评价，不混合排名。\n\n'+'\n'.join(links)+'\n')
