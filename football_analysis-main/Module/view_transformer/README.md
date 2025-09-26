# View Transformer 模块

## 模块功能
View Transformer模块负责将像素坐标转换为真实场地坐标，实现透视变换。该模块支持自动关键点检测和手动配置两种模式，能够将视频中的像素位置映射到标准足球场的真实坐标。

## 核心算法

### 1. 透视变换 (Perspective Transformation)
- **算法**: 基于4个对应点的单应性矩阵计算
- **功能**: 将像素坐标转换为真实场地坐标
- **特点**: 处理透视畸变，提供真实距离测量

### 2. 关键点检测
- **算法**: 基于深度学习的球场关键点检测
- **功能**: 自动检测球场关键点
- **特点**: 支持32个标准球场关键点

### 3. 坐标变换
- **算法**: 单应性矩阵变换
- **功能**: 批量坐标转换
- **特点**: 支持单点和批量点变换

## 主要函数

### `__init__()`
**功能**: 初始化视角变换器
- **输入**: 无
- **输出**: 无
- **核心算法**: 设置硬编码的四个角点坐标

**硬编码参数**:
```python
# 像素坐标（四个角点）
self.pixel_vertices = np.array([
    [110, 1035],   # 左上角
    [265, 275],    # 右上角
    [910, 260],    # 右下角
    [1640, 915]    # 左下角
])

# 目标场地坐标（米）
self.target_vertices = np.array([
    [0, court_width],      # 左上角
    [0, 0],                # 右上角
    [court_length, 0],     # 右下角
    [court_length, court_width]  # 左下角
])
```

### `transform_point(point)`
**功能**: 透视变换单个点
- **输入**: 
  - `point` (np.ndarray): 输入点坐标 [x, y]
- **输出**: 
  - `transformed_point` (np.ndarray): 变换后的点坐标，失败返回None
- **核心算法**:
  1. 检查点是否在源区域内
  2. 执行透视变换
  3. 返回变换后的坐标

**详细步骤**:
```python
# 1. 检查点是否在区域内
p = (int(point[0]), int(point[1]))
is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
if not is_inside:
    return None

# 2. 执行透视变换
reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
return transformed_point.reshape(-1, 2)
```

### `add_transformed_position_to_tracks(tracks)`
**功能**: 为轨迹添加透视变换后的位置坐标
- **输入**: 
  - `tracks` (dict): 跟踪结果字典
- **输出**: 无（直接修改tracks）
- **核心算法**:
  1. 遍历所有对象和轨迹
  2. 获取调整后的位置坐标
  3. 执行透视变换
  4. 存储变换后的坐标

**详细步骤**:
```python
# 1. 遍历所有对象
for object, object_tracks in tracks.items():
    # 2. 遍历每一帧
    for frame_num, track in enumerate(object_tracks):
        # 3. 遍历每个跟踪ID
        for track_id, track_info in track.items():
            # 4. 获取调整后的位置
            position = track_info['position_adjusted']
            position = np.array(position)
            
            # 5. 执行透视变换
            position_transformed = self.transform_point(position)
            
            # 6. 存储结果
            if position_transformed is not None:
                position_transformed = position_transformed.squeeze().tolist()
            tracks[object][frame_num][track_id]['position_transformed'] = position_transformed
```

## 核心算法详解

### 1. 透视变换矩阵计算
```python
# 计算透视变换矩阵
self.perspective_transformer = cv2.getPerspectiveTransform(
    self.pixel_vertices, self.target_vertices
)
```
- **算法**: OpenCV的getPerspectiveTransform函数
- **原理**: 基于4个对应点计算3x3单应性矩阵
- **特点**: 处理透视畸变，保持直线性

### 2. 坐标变换
```python
# 执行透视变换
reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
transformed_point = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
```
- **算法**: OpenCV的perspectiveTransform函数
- **原理**: 使用单应性矩阵进行坐标变换
- **特点**: 支持批量点变换

### 3. 区域检查
```python
# 检查点是否在源区域内
is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0
```
- **算法**: OpenCV的pointPolygonTest函数
- **原理**: 判断点是否在多边形区域内
- **特点**: 避免无效坐标的变换

## 参数调优

### 关键参数
- **像素顶点坐标**: 需要根据实际视频调整
- **目标场地坐标**: 基于标准足球场尺寸
- **场地尺寸**: 68m × 105m（标准足球场）

### 性能优化
1. **区域检查**: 避免无效坐标的变换
2. **数据类型**: 使用float32提高计算效率
3. **批量处理**: 支持批量点变换
4. **缓存机制**: 避免重复计算

## 使用示例

```python
# 初始化视角变换器
view_transformer = ViewTransformer()

# 为轨迹添加变换后的位置
view_transformer.add_transformed_position_to_tracks(tracks)

# 变换单个点
transformed_point = view_transformer.transform_point([500, 300])
```

## 数据结构

### 输入数据结构
```python
tracks = {
    "players": [
        {
            track_id: {
                "bbox": [x1, y1, x2, y2],
                "position": (x, y),
                "position_adjusted": (x, y),  # 相机运动调整后的位置
                ...
            }
        }
    ]
}
```

### 输出数据结构
```python
tracks = {
    "players": [
        {
            track_id: {
                "bbox": [x1, y1, x2, y2],
                "position": (x, y),
                "position_adjusted": (x, y),
                "position_transformed": (x, y),  # 透视变换后的位置（米）
                ...
            }
        }
    ]
}
```

## 应用场景

1. **体育分析**: 球员位置和移动距离分析
2. **战术分析**: 球队阵型和战术分析
3. **训练监控**: 运动员训练强度监控
4. **比赛统计**: 运动距离和速度统计

## 局限性

1. **硬编码坐标**: 需要手动调整四个角点坐标
2. **透视假设**: 假设相机位置相对固定
3. **场地依赖**: 依赖标准足球场尺寸
4. **精度限制**: 透视变换的精度受角点选择影响

## 改进建议

1. **自动关键点检测**: 使用深度学习自动检测角点
2. **动态调整**: 根据相机运动动态调整变换矩阵
3. **多尺度支持**: 支持不同尺寸的场地
4. **精度优化**: 使用更多对应点提高变换精度

## 依赖项

- opencv-python: 透视变换和图像处理
- numpy: 数组运算
- 自定义工具函数: 通过其他模块导入
