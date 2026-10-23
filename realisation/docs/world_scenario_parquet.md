<a id="world-scenario-parquet-file-structure"></a>

# 世界场景 Parquet 文件结构
本文详细介绍使用 `generate_control_videos.py` 脚本生成世界场景视频时，Parquet 文件的结构和要求。

<a id="overview"></a>

## 概述
世界场景生成流程将三维场景标注转换为控制视频，为 Cosmos Transfer 自动驾驶多视角视频生成模型提供条件。这些控制视频通过高精地图可视化提供空间上下文，展示以下内容：

- 车辆和行人的包围框。
- 车道线和边界。
- 交通基础设施，包括信号灯和标志。
- 道路标记和人行横道。

流程从包含结构化标注的 Parquet 文件读取场景数据，再从多个相机视角进行渲染，生成同步控制视频。

<a id="file-organization"></a>

## 文件组织
<a id="directory-structure"></a>

### 目录结构
```
scene_annotations_directory/
├── {clip_id}.obstacle.parquet              (required)
├── {clip_id}.calibration_estimate.parquet  (required)
├── {clip_id}.egomotion_estimate.parquet    (required)
├── {clip_id}.lane.parquet                  (optional)
├── {clip_id}.lane_line.parquet             (optional)
├── {clip_id}.road_boundary.parquet         (optional)
├── {clip_id}.crosswalk.parquet             (optional)
├── {clip_id}.pole.parquet                  (optional)
├── {clip_id}.road_marking.parquet          (optional)
├── {clip_id}.wait_line.parquet             (optional)
├── {clip_id}.traffic_light.parquet         (optional)
├── {clip_id}.traffic_sign.parquet          (optional)
```

所有文件必须使用相同的 `clip_id` 前缀，该前缀表示唯一的视频片段标识符。

<a id="coordinate-systems"></a>

## 坐标系
世界场景生成流程对所有数据使用一致的坐标系约定：

<a id="world-coordinate-system"></a>

### 世界坐标系
- **原点**：自车的起始位置，即第一帧位置，定义为世界原点 (0,0,0)。
- **约定**：采用右手坐标系。
  - X：向前，沿车辆前进方向。
  - Y：向左。
  - Z：向上。
- 自车运动和障碍物位置均使用世界坐标。

<a id="vehicle-reference-frame"></a>

### 车辆参考坐标系
- **车辆原点**：地面高度处的后轴中心，Z=0。
- **相机位置**：使用 FLU（前、左、上）约定，定义为*相对于*后轴的位置。
- **变换**：流程按世界 → 自车 → 相机的顺序串联变换。

<a id="temporal-alignment"></a>

### 时间对齐
- **自车运动**：通常以 10Hz 提供，插值到处理帧率，默认为 30 fps，可通过 `scripts/av_utils/render_config.py` 中的 `INPUT_POSE_FPS` 配置。
- **障碍物**：通常以 10Hz 提供，插值到与自车运动相同的处理帧率。
- **同步**：流程自动对障碍物轨迹插值，使其与自车运动时间戳对齐。

<a id="data-schema-notes"></a>

## 数据结构说明
<a id="version-field"></a>

### 版本字段
所有 Parquet 文件每行都包含 `version` 字段，类型为 uint64。该字段保存数据版本号，但**渲染流程不会使用它**。渲染代码处理所有数据，不受版本值影响。实际数据中，同一场景的所有行通常使用相同版本号。

<a id="unused-fields"></a>

### 未使用的字段
以下字段存在于 Parquet 数据结构中，但**当前渲染流程不会使用**：

- **key 字段**：key 结构中仅使用 `timestamp_micros`，忽略 `clip_id`、`label_class_id`、`map_id` 和 `map_id_version` 等字段。
- **category 字段**：交通标志和道路标记的类别会被加载，但不参与渲染区分，所有类别均按相同方式渲染。
- **lane 字段**：不使用 `map_end`、`vehicle_types` 和 `use_types`。
- **wait_line 字段**：不使用 `intersection_subtype`。

<a id="required-files"></a>

## 必需文件
<a id="1-obstacle-file-clip_idobstacleparquet"></a>

### 1. 障碍物文件（`{clip_id}.obstacle.parquet`）
包含动态物体的三维包围框标注，例如车辆、行人和骑行者。

**数据结构：**

```python
{
    "key": {
        "clip_id": str,               # Clip identifier (not used)
        "timestamp_micros": int,      # Timestamp in microseconds
        "label_class_id": str         # Annotation version (not used)
    },
    "obstacle": {
        "trackline_id": str,          # Trackline identity used to group different sensor sightings of the same obstacle together.
        "center": {                   # 3D position in world coordinates
            "x": float,               # meters
            "y": float,               # meters
            "z": float                # meters
        },
        "size": {                     # 3D dimensions
            "x": float,               # length (meters)
            "y": float,               # width (meters)
            "z": float                # height (meters)
        },
        "orientation": {              # Quaternion rotation (x,y,z,w)
            "x": float,
            "y": float,
            "z": float,
            "w": float
        },
        "category": str               # Object type: "automobile", "bus", "person", "rider", "stroller", "trailer"
    },
    "version": uint64                 # Data version number (not used by rendering pipeline)
}
```

**类别映射：**

- `automobile`、`other_vehicle`、`vehicle` → `Car`。
- `person` → `Pedestrian`。
- `rider` → `Cyclist`。
- `bus`、`heavy_truck`、`train_or_tram_car`、`trolley_bus`、`trailer` → `Truck`。
- 其他所有类别，包括 `stroller`、`protruding_object`、`animal`，均映射为 `Others`。

注意：渲染流程进行匹配时不区分大小写。

<a id="2-calibration-estimate-file-clip_idcalibration_estimateparquet"></a>

### 2. 标定估计文件（`{clip_id}.calibration_estimate.parquet`）
包含相机标定和车辆传感器配置数据。

**数据结构：**

```python
{
    "key": {
        "clip_id": str,               # Clip identifier (not used)
        "timestamp_micros": int       # Usually -1 (static calibration, not used)
    },
    "calibration_estimate": {
        "name": str,                  # Rig configuration name
        "rig_json": str               # JSON string containing full calibration
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

**Rig JSON 结构：**

`rig_json` 字段包含序列化的 JSON，其中保存完整传感器配置。下文详细说明相关部分。更多信息请参阅 [DriveWorks rig 文档](https://developer.nvidia.com/docs/drive/drive-os/6.0.8.1/public/driveworks-nvsdk/nvsdk_dw_html/rigconfiguration_usecase0.html)。

<a id="camera-intrinsic-parameters-ftheta-model"></a>

### 相机内参（FTheta 模型）
系统使用 **FTheta 相机模型**，将像素到光学中心的距离映射为光线角度。每个相机的内参保存在 `properties` 字段中：

```jsonc
{
  "Model": "ftheta",                    // Camera model type
  "cx": "1919.420",                     // Principal point X (pixels)
  "cy": "1078.106",                     // Principal point Y (pixels)
  "width": "3848",                      // Image width (pixels)
  "height": "2168",                     // Image height (pixels)
  "polynomial": "0 0.000538 2.356e-09 1.907e-12 1.204e-16",  // Polynomial coefficients
  "polynomial-type": "pixeldistance-to-angle",  // Mapping direction
  "linear-c": "1.000000",               // Linear correction factor C
  "linear-d": "0.000000",               // Linear correction factor D
  "linear-e": "0.000000"                // Linear correction factor E
}
```

**参数含义：**

- **cx、cy**：主点，即光学中心的像素坐标。
- **width、height**：图像宽高，单位为像素。
- **polynomial**：畸变模型最多 6 个系数 [k0, k1, k2, k3, k4, k5]。
  - 对于 "pixeldistance-to-angle"：θ = k0 + k1*r + k2*r² + k3*r³ + k4*r⁴ + k5*r⁵。
  - 其中 r 为像素到 (cx, cy) 的距离，θ 为光线角度，单位为弧度。
- **polynomial-type**：
  - `"pixeldistance-to-angle"`：将像素半径映射为观察角度，即反向多项式。
  - `"angle-to-pixeldistance"`：将观察角度映射为像素半径，即正向多项式。
- **linear-c、linear-d、linear-e**：附加线性校正系数。
  - 未指定时，默认 c=1.0、d=0.0、e=0.0。

<a id="camera-extrinsic-parameters"></a>

### 相机外参
各相机相对于车辆的位姿由 `nominalSensor2Rig_FLU` 字段定义：

```jsonc
{
  "nominalSensor2Rig_FLU": {
    "t": [1.697, -0.010, 1.443],        // Translation [x, y, z] in meters
    "roll-pitch-yaw": [0.130, 0.468, -0.714]  // Rotation in degrees
  }
}
```

**坐标系：**

- **FLU 约定**：前、左、上坐标系。
  - X：向前，沿车辆前进方向。
  - Y：向左。
  - Z：向上。
- **车辆原点**：地面高度处的后轴中心，Z=0。
- **平移**：相机相对于地面后轴中心的位置。
- **旋转**：按横滚、俯仰、偏航顺序应用的欧拉角。

<a id="calibration-corrections"></a>

### 标定修正
系统也可能包含在标定过程中应用的动态修正：

```jsonc
{
  "correction_sensor_R_FLU": {
    "roll-pitch-yaw": [-0.150, -0.014, 0.231]   // Additional rotation correction
  },
  "correction_rig_T": [0.080, 0.009, 0.009]     // Additional translation correction
}
```

这些修正在标称变换的基础上应用，用于补偿安装公差和进一步的标定调整。

<a id="supported-cameras"></a>

### 支持的相机
通常提供以下相机配置：

- `camera:front:wide:120fov`：前向广角相机，视场角 120°。
- `camera:front:tele:30fov`：前向长焦相机，视场角 30°。
- `camera:front:tele:sat:30fov`：第二个前向长焦相机。
- `camera:cross:left:120fov`：左侧横向交通相机，视场角 120°。
- `camera:cross:right:120fov`：右侧横向交通相机，视场角 120°。
- `camera:rear:left:70fov`：左后方相机，视场角 70°。
- `camera:rear:right:70fov`：右后方相机，视场角 70°。
- `camera:rear:tele:30fov`：后向长焦相机，视场角 30°。

**相机配置的灵活性：**虽然系统中的相机名称固定，但可以为每个相机提供自定义内参和外参。例如，名为 `camera:front:wide:120fov` 的相机并不局限于恰好 120° 的视场角；可以修改多项式系数、主点和相机位姿，以匹配实际硬件。不过，相机配置与训练数据规范偏差较小时，通常更容易获得理想效果。

<a id="modifying-calibration-data"></a>

### 修改标定数据
在世界场景生成流程中使用自有相机标定时，请按以下步骤操作：

<a id="1-extract-template-from-example-data"></a>

#### 1. 从示例数据提取模板
首先，从示例标定文件中提取 rig JSON：

```python
import pandas as pd
import json

# Load example calibration
df = pd.read_parquet('example.calibration_estimate.parquet')
rig_json = json.loads(df.iloc[0]['calibration_estimate']['rig_json'])

# Save to editable file
with open('my_rig.json', 'w') as f:
    json.dump(rig_json, f, indent=2)
```

<a id="2-locate-camera-parameters"></a>

#### 2. 定位相机参数
在 rig JSON 的 `sensors` 数组中找到目标相机。每个相机条目包含：

```jsonc
{
  "name": "camera:front:wide:120fov",
  "properties": {                    // Camera intrinsics go here
    "Model": "ftheta",
    "cx": "1919.420",
    "cy": "1078.106",
    "width": "3848",
    "height": "2168",
    "polynomial": "0 0.000538 ...",
    "polynomial-type": "pixeldistance-to-angle",
    "linear-c": "1.0",
    "linear-d": "0.0",
    "linear-e": "0.0"
  },
  "nominalSensor2Rig_FLU": {        // Camera extrinsics go here
    "t": [1.697, -0.010, 1.443],
    "roll-pitch-yaw": [0.130, 0.468, -0.714]
  },
  "correction_sensor_R_FLU": {...}, // Optional corrections (can be zeroed)
  "correction_rig_T": [...],        // Optional corrections (can be zeroed)
  // Other fields not used by rendering: car-mask, parameter, protocol
}
```

<a id="3-modify-calibration-values"></a>

#### 3. 修改标定值
**内参**位于 `properties` 中：

- 将 `cx`、`cy` 更新为实际主点坐标。
- 将 `width`、`height` 更新为实际图像尺寸。
- 将 `polynomial` 系数替换为实际畸变模型系数。
- 设置 `polynomial-type`，与所用模型一致：
  - `"pixeldistance-to-angle"`：反向多项式，像素 → 角度。
  - `"angle-to-pixeldistance"`：正向多项式，角度 → 像素。
- 将 `linear-c`、`linear-d`、`linear-e` 设置为实际线性校正系数，默认值分别为 1.0、0.0、0.0。

**外参**位于 `nominalSensor2Rig_FLU` 中：

- 将 `t` 更新为实际相机位置 [x, y, z]，单位为米。
- 将 `roll-pitch-yaw` 更新为实际相机旋转，单位为度。

**如果从头标定，可移除原有标定修正：**

```jsonc
"correction_sensor_R_FLU": {"roll-pitch-yaw": [0.0, 0.0, 0.0]},
"correction_rig_T": [0.0, 0.0, 0.0]
```

<a id="4-create-new-calibration-parquet"></a>

#### 4. 创建新的标定 Parquet 文件
```python
# Create calibration dataframe
calibration_data = {
    'key': {'clip_id': 'your_clip_id', 'timestamp_micros': -1},
    'calibration_estimate': {
        'name': 'CUSTOM_RIG',
        'rig_json': json.dumps(rig_json)
    },
    'version': 1
}

df = pd.DataFrame([calibration_data])
df.to_parquet('your_clip_id.calibration_estimate.parquet')
```

<a id="5-important-notes"></a>

#### 5. 重要说明
- **最低结构要求**：生成世界场景时，只需要：
  - `sensors` 数组中的相机条目，包含有效的 `name`、`properties` 和 `nominalSensor2Rig_FLU`。
  - 外层 rig JSON 结构。
  - 若不需要，其他传感器（IMU、GPS 等）可以省略。

<a id="3-egomotion-estimate-file-clip_idegomotion_estimateparquet"></a>

### 3. 自车运动估计文件（`{clip_id}.egomotion_estimate.parquet`）
包含车辆轨迹数据，即随时间变化的位置和朝向。

**数据结构：**

```python
{
    "key": {
        "clip_id": str,               # Clip identifier (not used)
        "timestamp_micros": int       # Timestamp in microseconds
    },
    "egomotion_estimate": {
        "name": str,                  # Estimation method (e.g., "egomotion_deepmap")
        "location": {                 # 3D position in world coordinates
            "x": float,               # meters
            "y": float,               # meters
            "z": float                # meters
        },
        "orientation": {              # Quaternion rotation (x,y,z,w)
            "x": float,
            "y": float,
            "z": float,
            "w": float
        }
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

<a id="optional-map-files"></a>

## 可选地图文件
这些文件通过静态地图元素丰富可视化效果。它们具有相似结构，但使用不同的几何类型。

<a id="lane-file-clip_idlaneparquet"></a>

### 车道文件（`{clip_id}.lane.parquet`）
表示行驶车道的边界。

**数据结构：**

```python
{
    "key": {
        "clip_id": str,               # (not used)
        "label_class_id": str,        # (not used)
        "map_id": str,                # (not used)
        "map_id_version": str         # (not used)
    },
    "lane": {
        "left_rail": [{"x": float, "y": float, "z": float}, ...],   # Polyline points
        "right_rail": [{"x": float, "y": float, "z": float}, ...],  # Polyline points
        "left_edge_styles": [str, ...],   # Style per segment: "VIRTUAL", "SOLID", "DASHED", "ROAD_BOUNDARY", etc.
        "right_edge_styles": [str, ...],
        "left_edge_colors": [str, ...],   # Color per segment: "WHITE", "YELLOW", "MIXED", etc.
        "right_edge_colors": [str, ...],
        "lane_direction": str,             # Optional: STRAIGHT, LEFT_TURN
        "speed_limit": float,              # Optional: speed limit (mph)
        "vehicle_types": [str, ...],       # Optional: allowed vehicle types
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

<a id="lane-line-file-clip_idlane_lineparquet"></a>

### 车道线文件（`{clip_id}.lane_line.parquet`）
表示独立的车道标线，例如中心线、边缘线。

**数据结构：**

```python
{
    "key": {...},  # Same structure as lane file
    "lane_line": {
        "line_rail": [{"x": float, "y": float, "z": float}, ...],   # Polyline
        "styles": [str, ...],        # Per-point style: "SOLID_SINGLE", "DASHED_SINGLE", etc.
        "colors": [str, ...],        # Per-point color: "WHITE", "YELLOW"
        "left_driving_direction": str,    # Optional
        "right_driving_direction": str    # Optional
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

**样式类型：**

- `SOLID_SINGLE`、`SOLID_DOUBLE`。
- `DASHED_SINGLE`、`LONG_DASHED_SINGLE`。
- `VIRTUAL`，不渲染。

<a id="road-boundary-file-clip_idroad_boundaryparquet"></a>

### 道路边界文件（`{clip_id}.road_boundary.parquet`）
表示物理道路边界，例如路缘、隔离设施。

**数据结构：**

```python
{
    "key": {...},  # Same structure as lane file
    "road_boundary": {
        "category": str,             # "tall_curb", "road_boundary", "barrier", "fence", "wall"
        "location": [{"x": float, "y": float, "z": float}, ...]  # Polyline
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

<a id="crosswalk-file-clip_idcrosswalkparquet"></a>

### 人行横道文件（`{clip_id}.crosswalk.parquet`）
表示供行人穿越道路的区域。

**数据结构：**

```python
{
    "key": {...},  # Same structure as lane file
    "crosswalk": {
        "category": str,             # "PEDESTRIAN"
        "location": [{"x": float, "y": float, "z": float}, ...]  # Polygon vertices
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

<a id="traffic-light-file-clip_idtraffic_lightparquet"></a>

### 交通信号灯文件（`{clip_id}.traffic_light.parquet`）
表示交通信号灯的位置和朝向。

**注意：**当前 Parquet 格式**不包含**交通信号灯状态信息，即红、黄、绿灯状态。虽然渲染代码具备根据状态显示不同颜色的能力，但由于 Parquet 文件中没有状态数据，目前所有交通信号灯均渲染为统一的灰色，表示“未知”状态。

**数据结构：**

```python
{
    "key": {...},  # Same structure as lane file
    "traffic_light": {
        "center": {"x": float, "y": float, "z": float},
        "dimensions": {"x": float, "y": float, "z": float},
        "orientation": {"x": float, "y": float, "z": float, "w": float},  # Quaternion
        "category": str              # "traffic_light"
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

<a id="traffic-sign-file-clip_idtraffic_signparquet"></a>

### 交通标志文件（`{clip_id}.traffic_sign.parquet`）
表示道路标志和信息牌。

**常见类别包括：**

- 管制标志：`TRAFFIC_SIGN_REGULATORY_R1_STOP`、`R1_YIELD`、`R2_SPEED_LIMIT`、`R3_NO_RIGHT_TURN` 等。
- 警告标志：`TRAFFIC_SIGN_WARNING_W11_PEDESTRAIN_CROSSING`、`W3_STOP_AHEAD`、`W4_MERGE` 等。
- 学校标志：`TRAFFIC_SIGN_SCHOOL_S1_SCHOOL`。
- 信息标志：`TRAFFIC_SIGN_INFORMATION`。
- 目的地／距离标志：`TRAFFIC_SIGN_DESTINATION_DISTANCE_D11_BIKE_ROUTE`。

**数据结构：**

```python
{
    "key": {...},  # Same structure as lane file
    "traffic_sign": {
        "center": {"x": float, "y": float, "z": float},
        "dimensions": {"x": float, "y": float, "z": float},
        "orientation": {"x": float, "y": float, "z": float, "w": float},  # Quaternion
        "category": str              # "TRAFFIC_SIGN_INFORMATION", "TRAFFIC_SIGN_STOP", etc. (not used for rendering)
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

<a id="additional-map-elements"></a>

### 其他地图元素
**杆体文件**（`pole.parquet`）：竖直结构。

- 类别示例：`LIGHT`、`SIGN`、`TEL`、`TREE`、`SENTRY`、`UNKNOWN`。
- 以两个端点组成的线段保存底部和顶部位置。

**数据结构：**

```python
{
    "key": {...},  # Same structure as lane file (no fields used)
    "pole": {
        "category": str,             # e.g., "LIGHT", "SIGN", "TREE" (not used for rendering)
        "location": [{"x": float, "y": float, "z": float}, ...]  # 2 points: base and top
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

**道路标记文件**（`road_marking.parquet`）：绘制在路面的标记。

- 类别示例：
  - `ROI_POLYGON_ROAD_MARKING_STOP`：停止线标记。
  - `ROI_POLYGON_ROAD_MARKING_PED_XING`：人行横道标记。
  - `ROI_POLYGON_BIKE_PAINT`：自行车道标记。
  - `ROI_POLYGON_DIRECTION_PAINT_LEFT`、`_RIGHT`、`_FORWARD`：方向箭头。
  - `ROI_POLYGON_SPEED_LIMIT_PAINT_25`：限速标记。
  - `ROI_POLYGON_TEXT_PAINT`：通用文字标记。
- 使用多边形保存。

**数据结构：**

```python
{
    "key": {...},  # Same structure as lane file (no fields used)
    "road_marking": {
        "category": str,             # e.g., "ROI_POLYGON_ROAD_MARKING_STOP" (not used for rendering)
        "location": [{"x": float, "y": float, "z": float}, ...]  # Polygon vertices
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```

**停止线文件**（`wait_line.parquet`）：交叉路口的停止线。

**数据结构：**

```python
{
    "key": {...},  # Same structure as lane file
    "wait_line": {
        "category": str,             # "STOP", "UNKNOWN"
        "location": [{"x": float, "y": float, "z": float}, ...],  # Polyline
        "is_implicit": bool,         # Whether line is implicit or explicitly marked
        "intersection_subtype": str  # Whether this wait line is entry, exit or part of buffer zone at intersections or not applicable. (not used for rendering)
    },
    "version": uint64    # Data version number (not used by rendering pipeline)
}
```
