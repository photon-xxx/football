# 足球队表现自动分析

## 介绍
该项目的目标是使用 Yolo（目前最出色的 AI 物体检测模型之一）在视频中检测和追踪球员、裁判和足球。我们还将训练模型以提高其性能。此外，我们将基于球员球衣的颜色，使用 Kmeans 进行像素分割和聚类，将球员分配到不同的队伍。利用这些信息，我们可以计算出一支球队在一场比赛中获得球权的百分比。我们还将使用光流法测量帧与帧之间的摄像机移动，从而能够准确测量球员的移动。此外，我们将实现透视变换来表示场景的深度和透视，这样就能以米为单位而非像素来测量球员的移动距离。最后，我们将计算球员的速度和移动距离。

## 使用的模型
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## 需要的包
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas

## 最新更新

### ViewTransformer 透视变换器改进 (2024-12-19)

#### 主要改进
1. **动态关键点支持**: `ViewTransformer` 现在支持使用检测到的关键点动态计算透视变换矩阵
2. **批量点变换**: 新增 `transform_points()` 方法，支持批量变换多个点坐标
3. **向后兼容**: 保持原有接口不变，支持默认的四个顶点模式

#### 使用方法

##### 基本用法
```python
from Module.view_transformer.view_transformer import ViewTransformer
import numpy as np

# 使用检测到的关键点创建变换器
transformer = ViewTransformer(
    source=detected_keypoints,  # 像素坐标
    target=target_keypoints     # 场地坐标
)

# 批量变换点
transformed_points = transformer.transform_points(points)
```

##### 与关键点检测结合使用
```python
from Module.view_transformer.KeyPoint_detection import KeyPointDetector
from Storage.field_configs.soccer import SoccerPitchConfiguration

# 检测关键点
detector = KeyPointDetector()
keypoints = detector.detect_keypoints(frame)

# 筛选有效关键点
mask = ~np.isnan(keypoints.xy[0][:, 0]) & ~np.isnan(keypoints.xy[0][:, 1])
valid_source = keypoints.xy[0][mask]
valid_target = np.array(CONFIG.vertices)[mask]

# 创建变换器
transformer = ViewTransformer(source=valid_source, target=valid_target)
```

#### 技术细节
- 支持任意数量的关键点（最少4个）
- 自动处理NaN值（低置信度点）
- 使用OpenCV的透视变换算法
- 保持原有API的完全兼容性

#### 文件结构
```
Module/view_transformer/
├── view_transformer.py      # 主变换器类
├── KeyPoint_detection.py    # 关键点检测
└── example_usage.py         # 使用示例
```
