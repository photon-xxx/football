# Utils 模块

## 模块功能
Utils模块提供项目中的通用工具函数，包括边界框处理、距离计算、视频读写等基础功能。该模块为其他模块提供底层支持，确保代码的复用性和可维护性。

## 核心算法

### 1. 边界框处理
- **算法**: 基于边界框坐标的几何计算
- **功能**: 计算边界框的中心点、宽度、脚下位置等
- **特点**: 简单高效，适合实时处理

### 2. 距离计算
- **算法**: 欧几里得距离
- **功能**: 计算两点间的空间距离
- **特点**: 计算简单，精度高

### 3. 视频处理
- **算法**: OpenCV视频读写
- **功能**: 读取和保存视频文件
- **特点**: 支持多种视频格式

## 主要函数

### 边界框处理函数

#### `get_center_of_bbox(bbox)`
**功能**: 计算边界框的中心点
- **输入**: 
  - `bbox` (list): 边界框坐标 [x1, y1, x2, y2]
- **输出**: 
  - `(x, y)` (tuple): 中心点坐标
- **核心算法**:
  ```python
  x = (x1 + x2) / 2
  y = (y1 + y2) / 2
  ```

#### `get_bbox_width(bbox)`
**功能**: 计算边界框的宽度
- **输入**: 
  - `bbox` (list): 边界框坐标 [x1, y1, x2, y2]
- **输出**: 
  - `width` (float): 边界框宽度
- **核心算法**:
  ```python
  width = x2 - x1
  ```

#### `get_foot_position(bbox)`
**功能**: 计算边界框的脚下位置
- **输入**: 
  - `bbox` (list): 边界框坐标 [x1, y1, x2, y2]
- **输出**: 
  - `(x, y)` (tuple): 脚下位置坐标
- **核心算法**:
  ```python
  x = (x1 + x2) / 2  # 水平中心
  y = y2             # 底部位置
  ```

### 距离计算函数

#### `measure_distance(p1, p2)`
**功能**: 计算两点间的欧几里得距离
- **输入**: 
  - `p1` (tuple): 第一个点坐标 (x1, y1)
  - `p2` (tuple): 第二个点坐标 (x2, y2)
- **输出**: 
  - `distance` (float): 两点间距离
- **核心算法**:
  ```python
  distance = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
  ```

#### `measure_xy_distance(p1, p2)`
**功能**: 计算两点间的x和y方向距离
- **输入**: 
  - `p1` (tuple): 第一个点坐标 (x1, y1)
  - `p2` (tuple): 第二个点坐标 (x2, y2)
- **输出**: 
  - `(dx, dy)` (tuple): x和y方向的距离
- **核心算法**:
  ```python
  dx = p1[0] - p2[0]
  dy = p1[1] - p2[1]
  ```

### 视频处理函数

#### `read_video(video_path)`
**功能**: 读取视频文件并返回帧列表
- **输入**: 
  - `video_path` (str): 视频文件路径
- **输出**: 
  - `frames` (list): 视频帧列表
- **核心算法**:
  ```python
  cap = cv2.VideoCapture(video_path)
  frames = []
  while True:
      ret, frame = cap.read()
      if not ret:
          break
      frames.append(frame)
  return frames
  ```

#### `save_video(output_video_frames, output_video_path)`
**功能**: 保存视频帧列表为视频文件
- **输入**: 
  - `output_video_frames` (list): 视频帧列表
  - `output_video_path` (str): 输出视频路径
- **输出**: 无
- **核心算法**:
  ```python
  fourcc = cv2.VideoWriter_fourcc(*'XVID')
  out = cv2.VideoWriter(output_video_path, fourcc, 24, (width, height))
  for frame in output_video_frames:
      out.write(frame)
  out.release()
  ```

## 核心算法详解

### 1. 边界框几何计算
```python
# 中心点计算
x_center = (x1 + x2) / 2
y_center = (y1 + y2) / 2

# 宽度计算
width = x2 - x1

# 高度计算
height = y2 - y1
```
- **原理**: 基于边界框的几何属性
- **特点**: 计算简单，精度高
- **应用**: 目标定位、尺寸计算

### 2. 欧几里得距离
```python
# 距离公式
distance = sqrt((x1-x2)² + (y1-y2)²)
```
- **原理**: 基于勾股定理的距离计算
- **特点**: 计算简单，适合实时处理
- **应用**: 目标跟踪、距离测量

### 3. 视频处理流程
```python
# 读取流程
cap = cv2.VideoCapture(video_path)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

# 保存流程
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
for frame in frames:
    out.write(frame)
```
- **原理**: 基于OpenCV的视频处理
- **特点**: 支持多种视频格式
- **应用**: 视频读写、格式转换

## 参数调优

### 关键参数
- **视频编解码器**: XVID（平衡压缩率和质量）
- **帧率**: 24fps（标准帧率）
- **数据类型**: float32（提高计算精度）

### 性能优化
1. **数据类型**: 使用合适的数据类型
2. **内存管理**: 及时释放视频资源
3. **批量处理**: 支持批量视频处理
4. **错误处理**: 添加异常处理机制

## 使用示例

```python
# 边界框处理
bbox = [100, 200, 300, 400]
center = get_center_of_bbox(bbox)
width = get_bbox_width(bbox)
foot_pos = get_foot_position(bbox)

# 距离计算
p1 = (100, 200)
p2 = (300, 400)
distance = measure_distance(p1, p2)
dx, dy = measure_xy_distance(p1, p2)

# 视频处理
frames = read_video('input.mp4')
save_video(frames, 'output.mp4')
```

## 数据结构

### 边界框格式
```python
bbox = [x1, y1, x2, y2]  # 左上角和右下角坐标
```

### 点坐标格式
```python
point = (x, y)  # 二维坐标
```

### 视频帧格式
```python
frames = [frame1, frame2, ...]  # 视频帧列表
frame = np.ndarray  # 单个视频帧（H, W, C）
```

## 应用场景

1. **目标检测**: 边界框处理和分析
2. **目标跟踪**: 距离计算和位置更新
3. **视频处理**: 视频读写和格式转换
4. **几何计算**: 空间距离和角度计算

## 局限性

1. **精度限制**: 基于像素坐标的计算精度有限
2. **格式依赖**: 依赖OpenCV支持的视频格式
3. **内存消耗**: 大视频文件可能消耗大量内存
4. **平台依赖**: 依赖OpenCV的编译版本

## 改进建议

1. **精度优化**: 使用更高精度的数据类型
2. **格式支持**: 支持更多视频格式
3. **内存优化**: 使用流式处理减少内存消耗
4. **错误处理**: 添加更完善的异常处理

## 依赖项

- opencv-python: 视频处理和图像处理
- numpy: 数组运算
- 标准库: 文件操作和数据类型
