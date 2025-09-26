# Speed and Distance Estimator 模块

## 模块功能
Speed and Distance Estimator模块负责计算球员的运动速度和累计移动距离。该模块基于球员的轨迹数据，通过时间窗口内的位置变化计算瞬时速度和累计距离。

## 核心算法

### 1. 速度计算
- **算法**: 基于位置差和时间差的平均速度
- **公式**: v = Δs / Δt
- **特点**: 考虑时间窗口，平滑速度变化

### 2. 距离累计
- **算法**: 累加相邻帧间的移动距离
- **特点**: 提供累计移动距离统计

### 3. 时间窗口处理
- **算法**: 滑动窗口平均
- **功能**: 平滑速度计算，减少噪声
- **特点**: 可配置的窗口大小

## 主要函数

### `__init__()`
**功能**: 初始化速度距离估计器
- **输入**: 无
- **输出**: 无
- **核心算法**: 设置时间窗口和帧率参数

**参数设置**:
```python
self.frame_window = 5      # 时间窗口大小（帧数）
self.frame_rate = 24       # 视频帧率（fps）
```

### `add_speed_and_distance_to_tracks(tracks)`
**功能**: 为轨迹添加速度和距离信息
- **输入**: 
  - `tracks` (dict): 跟踪结果字典
- **输出**: 无（直接修改tracks）
- **核心算法**:
  1. 遍历所有对象（排除球和裁判）
  2. 按时间窗口处理轨迹
  3. 计算相邻窗口间的速度和距离
  4. 累计总距离
  5. 更新轨迹数据

**详细步骤**:
```python
# 1. 遍历所有对象
for object, object_tracks in tracks.items():
    if object == "ball" or object == "referees":
        continue  # 跳过球和裁判
    
    # 2. 按时间窗口处理
    for frame_num in range(0, number_of_frames, self.frame_window):
        last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
        
        # 3. 计算速度和距离
        for track_id, _ in object_tracks[frame_num].items():
            if track_id not in object_tracks[last_frame]:
                continue
            
            start_position = object_tracks[frame_num][track_id]['position_transformed']
            end_position = object_tracks[last_frame][track_id]['position_transformed']
            
            if start_position is None or end_position is None:
                continue
            
            # 4. 计算距离和时间
            distance_covered = measure_distance(start_position, end_position)
            time_elapsed = (last_frame - frame_num) / self.frame_rate
            
            # 5. 计算速度
            speed_meteres_per_second = distance_covered / time_elapsed
            speed_km_per_hour = speed_meteres_per_second * 3.6
            
            # 6. 累计距离
            total_distance[object][track_id] += distance_covered
            
            # 7. 更新轨迹数据
            for frame_num_batch in range(frame_num, last_frame):
                if track_id not in tracks[object][frame_num_batch]:
                    continue
                tracks[object][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                tracks[object][frame_num_batch][track_id]['distance'] = total_distance[object][track_id]
```

### `draw_speed_and_distance(frames, tracks)`
**功能**: 在视频上绘制速度和距离信息
- **输入**: 
  - `frames` (list): 视频帧列表
  - `tracks` (dict): 跟踪结果字典
- **输出**: 
  - `output_frames` (list): 绘制后的视频帧列表
- **核心算法**: OpenCV文本绘制

**详细步骤**:
```python
# 1. 遍历所有帧
for frame_num, frame in enumerate(frames):
    # 2. 遍历所有对象
    for object, object_tracks in tracks.items():
        if object == "ball" or object == "referees":
            continue
        
        # 3. 绘制每个球员的信息
        for _, track_info in object_tracks[frame_num].items():
            if "speed" in track_info:
                speed = track_info.get('speed', None)
                distance = track_info.get('distance', None)
                
                if speed is None or distance is None:
                    continue
                
                # 4. 计算绘制位置
                bbox = track_info['bbox']
                position = get_foot_position(bbox)
                position = list(position)
                position[1] += 40  # 向下偏移
                
                # 5. 绘制文本
                cv2.putText(frame, f"{speed:.2f} km/h", position, 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, f"{distance:.2f} m", (position[0], position[1] + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
```

## 核心算法详解

### 1. 速度计算公式
```python
# 距离计算（米）
distance_covered = measure_distance(start_position, end_position)

# 时间计算（秒）
time_elapsed = (last_frame - frame_num) / self.frame_rate

# 速度计算（米/秒）
speed_meteres_per_second = distance_covered / time_elapsed

# 速度转换（公里/小时）
speed_km_per_hour = speed_meteres_per_second * 3.6
```

### 2. 时间窗口处理
```python
# 滑动窗口处理
for frame_num in range(0, number_of_frames, self.frame_window):
    last_frame = min(frame_num + self.frame_window, number_of_frames - 1)
```
- **原理**: 使用固定时间窗口计算平均速度
- **优势**: 平滑速度变化，减少噪声
- **窗口大小**: 5帧（约0.2秒）

### 3. 距离累计
```python
# 累计距离
if object not in total_distance:
    total_distance[object] = {}
if track_id not in total_distance[object]:
    total_distance[object][track_id] = 0

total_distance[object][track_id] += distance_covered
```
- **原理**: 累加相邻窗口间的移动距离
- **特点**: 提供累计移动距离统计

## 参数调优

### 关键参数
- `frame_window = 5`: 时间窗口大小（帧数）
  - **调优建议**: 根据视频帧率和运动特点调整
  - **经验值**: 通常设置为0.2-0.5秒
- `frame_rate = 24`: 视频帧率（fps）
  - **调优建议**: 根据实际视频帧率设置
  - **常见值**: 24fps, 30fps, 60fps

### 性能优化
1. **窗口处理**: 使用滑动窗口减少计算量
2. **跳过无效数据**: 跳过球和裁判的计算
3. **条件检查**: 检查轨迹连续性
4. **批量更新**: 批量更新轨迹数据

## 使用示例

```python
# 初始化速度距离估计器
speed_estimator = SpeedAndDistance_Estimator()

# 添加速度和距离信息
speed_estimator.add_speed_and_distance_to_tracks(tracks)

# 绘制速度和距离信息
output_frames = speed_estimator.draw_speed_and_distance(video_frames, tracks)
```

## 数据结构

### 输入数据结构
```python
tracks = {
    "players": [
        {
            track_id: {
                "bbox": [x1, y1, x2, y2],
                "position_transformed": (x, y),  # 透视变换后的位置
                "team": 1/2,
                ...
            }
        }
    ],
    "referees": [...],
    "ball": [...]
}
```

### 输出数据结构
```python
tracks = {
    "players": [
        {
            track_id: {
                "bbox": [x1, y1, x2, y2],
                "position_transformed": (x, y),
                "speed": 15.5,        # 瞬时速度（km/h）
                "distance": 120.3,     # 累计距离（米）
                "team": 1/2,
                ...
            }
        }
    ]
}
```

## 应用场景

1. **体育分析**: 球员运动表现分析
2. **训练监控**: 运动员训练强度监控
3. **比赛统计**: 运动距离和速度统计
4. **健康监测**: 运动量监测

## 局限性

1. **透视变换依赖**: 需要准确的透视变换
2. **轨迹连续性**: 依赖轨迹的连续性
3. **时间窗口**: 固定时间窗口可能不适合所有场景
4. **噪声影响**: 位置噪声可能影响速度计算

## 改进建议

1. **自适应窗口**: 根据运动特点调整时间窗口
2. **噪声滤波**: 使用滤波算法减少位置噪声
3. **多尺度分析**: 结合不同时间尺度的速度分析
4. **机器学习**: 使用机器学习方法提高准确性

## 依赖项

- opencv-python: 图像处理和文本绘制
- numpy: 数组运算
- 自定义工具函数: measure_distance, get_foot_position
