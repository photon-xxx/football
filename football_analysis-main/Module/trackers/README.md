# Trackers 模块

## 模块功能
Trackers模块是足球视频分析系统的核心目标检测与跟踪模块，负责在视频中检测和跟踪球员、裁判和足球。该模块使用YOLO模型进行目标检测，结合ByteTrack算法进行多目标跟踪，并提供轨迹数据的缓存和可视化功能。

## 核心算法

### 1. YOLO目标检测
- **算法**: YOLO (You Only Look Once) 深度学习模型
- **功能**: 在视频帧中检测球员、裁判、守门员和足球
- **特点**: 实时检测，高精度，支持批量处理

### 2. ByteTrack多目标跟踪
- **算法**: ByteTrack跟踪算法
- **功能**: 为检测到的目标分配唯一ID，跨帧跟踪
- **特点**: 处理遮挡、重入等复杂场景

### 3. 轨迹插值
- **算法**: 基于pandas的线性插值
- **功能**: 补齐球类轨迹的缺失帧
- **特点**: 提高轨迹连续性

## 主要函数

### `__init__(model_path)`
**功能**: 初始化跟踪器
- **输入**: 
  - `model_path` (str): YOLO模型文件路径
- **输出**: 无
- **核心算法**: 加载YOLO模型，初始化ByteTrack跟踪器

### `detect_frames(frames)`
**功能**: 批量检测视频帧中的目标
- **输入**: 
  - `frames` (list): 视频帧列表
- **输出**: 
  - `detections` (list): 检测结果列表
- **核心算法**: 
  - 批量处理（每批20帧）
  - YOLO模型推理
  - 置信度阈值过滤（0.1）

### `get_object_tracks(frames, read_from_stub=False, stub_path=None)`
**功能**: 获取目标跟踪结果
- **输入**: 
  - `frames` (list): 视频帧列表
  - `read_from_stub` (bool): 是否从缓存读取
  - `stub_path` (str): 缓存文件路径
- **输出**: 
  - `tracks` (dict): 跟踪结果字典
    ```python
    {
        "players": [frame1_dict, frame2_dict, ...],
        "referees": [frame1_dict, frame2_dict, ...],
        "ball": [frame1_dict, frame2_dict, ...]
    }
    ```
- **核心算法**:
  - 目标检测 → 跟踪 → 分类存储
  - 守门员归类为球员
  - 球类不进行跟踪（每帧独立检测）

### `add_position_to_tracks(tracks)`
**功能**: 为轨迹添加位置信息
- **输入**: 
  - `tracks` (dict): 跟踪结果字典
- **输出**: 无（直接修改tracks）
- **核心算法**:
  - 球员/裁判：使用脚下位置
  - 球：使用边界框中心

### `interpolate_ball_positions(ball_positions)`
**功能**: 对球类轨迹进行插值
- **输入**: 
  - `ball_positions` (list): 球类轨迹列表
- **输出**: 
  - `ball_positions` (list): 插值后的轨迹列表
- **核心算法**:
  - pandas DataFrame插值
  - 线性插值 + 后向填充

## 可视化函数

### `draw_ellipse(frame, bbox, color, track_id=None)`
**功能**: 绘制球员/裁判的椭圆标记
- **输入**: 
  - `frame` (np.ndarray): 图像帧
  - `bbox` (list): 边界框 [x1,y1,x2,y2]
  - `color` (tuple): 颜色 (B,G,R)
  - `track_id` (int): 跟踪ID
- **输出**: 
  - `frame` (np.ndarray): 绘制后的图像帧
- **核心算法**: OpenCV椭圆绘制 + 文本标注

### `draw_traingle(frame, bbox, color)`
**功能**: 绘制球类的三角形标记
- **输入**: 
  - `frame` (np.ndarray): 图像帧
  - `bbox` (list): 边界框
  - `color` (tuple): 颜色
- **输出**: 
  - `frame` (np.ndarray): 绘制后的图像帧
- **核心算法**: OpenCV轮廓绘制

### `draw_team_ball_control(frame, frame_num, team_ball_control)`
**功能**: 绘制球队控球率信息
- **输入**: 
  - `frame` (np.ndarray): 图像帧
  - `frame_num` (int): 当前帧号
  - `team_ball_control` (np.ndarray): 控球权数组
- **输出**: 
  - `frame` (np.ndarray): 绘制后的图像帧
- **核心算法**: 实时计算控球比例并显示

### `draw_annotations(video_frames, tracks, team_ball_control)`
**功能**: 绘制完整的视频标注
- **输入**: 
  - `video_frames` (list): 视频帧列表
  - `tracks` (dict): 跟踪结果
  - `team_ball_control` (np.ndarray): 控球权数组
- **输出**: 
  - `output_video_frames` (list): 标注后的视频帧列表
- **核心算法**: 综合所有绘制功能

## 数据结构

### 轨迹数据结构
```python
tracks = {
    "players": [
        {
            track_id: {
                "bbox": [x1, y1, x2, y2],
                "position": (x, y),
                "team": 1/2,
                "team_color": (B, G, R),
                "has_ball": True/False,
                "speed": float,
                "distance": float
            }
        }
    ],
    "referees": [...],
    "ball": [...]
}
```

## 使用示例

```python
# 初始化跟踪器
tracker = Tracker('models/yolo/best.pt')

# 获取跟踪结果
tracks = tracker.get_object_tracks(
    video_frames, 
    read_from_stub=True, 
    stub_path='cache/tracks.pkl'
)

# 添加位置信息
tracker.add_position_to_tracks(tracks)

# 绘制标注
output_frames = tracker.draw_annotations(
    video_frames, tracks, team_ball_control
)
```

## 性能优化

1. **批量处理**: 每批处理20帧，减少模型加载开销
2. **缓存机制**: 支持轨迹数据的保存和读取
3. **置信度过滤**: 设置合理的检测阈值
4. **插值优化**: 使用pandas高效插值算法

## 依赖项

- ultralytics: YOLO模型
- supervision: ByteTrack跟踪
- opencv-python: 图像处理
- numpy: 数组运算
- pandas: 数据插值
