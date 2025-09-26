# Camera Movement Estimator 模块

## 模块功能
Camera Movement Estimator模块负责估计视频中相机的运动，通过光流法分析连续帧之间的特征点变化，计算相机的平移运动，并据此调整目标位置以提高跟踪精度。

## 核心算法

### 1. 光流法 (Optical Flow)
- **算法**: Lucas-Kanade光流法
- **功能**: 跟踪特征点在连续帧间的运动
- **特点**: 实时性好，适合相机运动估计

### 2. 特征点检测
- **算法**: Shi-Tomasi角点检测
- **功能**: 在图像中检测稳定的特征点
- **特点**: 对光照变化鲁棒

### 3. 运动补偿
- **算法**: 基于最大位移的运动估计
- **功能**: 计算相机运动并调整目标位置
- **特点**: 提高跟踪精度

## 主要函数

### `__init__(frame)`
**功能**: 初始化相机运动估计器
- **输入**: 
  - `frame` (np.ndarray): 第一帧图像
- **输出**: 无
- **核心算法**: 
  - 设置光流参数
  - 创建特征点检测掩码
  - 配置角点检测参数

**参数配置**:
```python
# 光流参数
lk_params = {
    'winSize': (15, 15),      # 搜索窗口大小
    'maxLevel': 2,            # 金字塔层数
    'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
}

# 特征点检测参数
features = {
    'maxCorners': 100,        # 最大特征点数
    'qualityLevel': 0.3,     # 质量阈值
    'minDistance': 3,        # 最小距离
    'blockSize': 7,          # 块大小
    'mask': mask_features    # 检测区域掩码
}
```

### `get_camera_movement(frames, read_from_stub=False, stub_path=None)`
**功能**: 计算每帧的相机运动
- **输入**: 
  - `frames` (list): 视频帧列表
  - `read_from_stub` (bool): 是否从缓存读取
  - `stub_path` (str): 缓存文件路径
- **输出**: 
  - `camera_movement` (list): 每帧相机运动列表
    ```python
    [
        [0, 0],           # 第0帧（参考帧）
        [dx1, dy1],       # 第1帧运动
        [dx2, dy2],       # 第2帧运动
        ...
    ]
    ```
- **核心算法**:
  1. 检测第一帧的特征点
  2. 对每帧计算光流
  3. 找到最大位移作为相机运动
  4. 更新特征点（当运动超过阈值时）

### `add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)`
**功能**: 根据相机运动调整目标位置
- **输入**: 
  - `tracks` (dict): 跟踪结果字典
  - `camera_movement_per_frame` (list): 每帧相机运动
- **输出**: 无（直接修改tracks）
- **核心算法**: 
  ```python
  position_adjusted = (position[0] - camera_movement[0], 
                      position[1] - camera_movement[1])
  ```

### `draw_camera_movement(frames, camera_movement_per_frame)`
**功能**: 在视频上绘制相机运动信息
- **输入**: 
  - `frames` (list): 视频帧列表
  - `camera_movement_per_frame` (list): 每帧相机运动
- **输出**: 
  - `output_frames` (list): 绘制后的视频帧列表
- **核心算法**: OpenCV文本绘制

## 核心算法详解

### 1. 特征点检测策略
```python
# 创建检测区域掩码
mask_features = np.zeros_like(first_frame_grayscale)
mask_features[:, 0:20] = 1      # 左侧边缘
mask_features[:, 900:1050] = 1   # 右侧边缘
```
- **目的**: 只检测图像边缘的特征点
- **原因**: 边缘特征点对相机运动更敏感

### 2. 光流跟踪
```python
new_features, _, _ = cv2.calcOpticalFlowPyrLK(
    old_gray, frame_gray, old_features, None, **lk_params
)
```
- **算法**: Lucas-Kanade金字塔光流
- **优势**: 处理大位移，提高跟踪稳定性

### 3. 运动估计
```python
# 计算每个特征点的位移
distance = measure_distance(new_point, old_point)
if distance > max_distance:
    max_distance = distance
    camera_movement_x, camera_movement_y = measure_xy_distance(old_point, new_point)
```
- **策略**: 选择最大位移作为相机运动
- **原因**: 相机运动通常影响所有特征点

### 4. 特征点更新
```python
if max_distance > self.minimum_distance:
    old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)
```
- **条件**: 当相机运动超过阈值时
- **目的**: 重新检测特征点，保持跟踪质量

## 参数调优

### 关键参数
- `minimum_distance = 5`: 最小运动阈值
- `maxCorners = 100`: 最大特征点数
- `qualityLevel = 0.3`: 特征点质量阈值
- `winSize = (15, 15)`: 光流搜索窗口

### 性能优化
1. **区域限制**: 只在边缘检测特征点
2. **阈值过滤**: 过滤微小运动
3. **动态更新**: 根据运动强度更新特征点
4. **缓存机制**: 支持运动数据的保存和读取

## 使用示例

```python
# 初始化相机运动估计器
camera_estimator = CameraMovementEstimator(video_frames[0])

# 计算相机运动
camera_movement = camera_estimator.get_camera_movement(
    video_frames, 
    read_from_stub=True, 
    stub_path='cache/camera_movement.pkl'
)

# 调整目标位置
camera_estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

# 绘制运动信息
output_frames = camera_estimator.draw_camera_movement(video_frames, camera_movement)
```

## 应用场景

1. **体育视频分析**: 相机跟随球员移动
2. **无人机视频**: 相机姿态变化
3. **手持设备**: 相机抖动补偿
4. **监控系统**: 相机运动检测

## 局限性

1. **假设条件**: 假设相机运动是全局的
2. **特征依赖**: 依赖图像中的稳定特征点
3. **运动类型**: 主要处理平移运动，旋转运动效果有限
4. **光照敏感**: 光照变化可能影响特征点质量

## 依赖项

- opencv-python: 光流和特征点检测
- numpy: 数组运算
- pickle: 数据缓存
