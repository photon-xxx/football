# Football Analysis Project

## Introduction
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams using a deep learning-based approach with SigLIP vision model for feature extraction, UMAP for dimensionality reduction, and KMeans for clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](IO/output_videos/screenshot.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- SigLIP Vision Model: Deep visual feature extraction for team classification
- UMAP: Dimensionality reduction for team clustering
- KMeans: Clustering algorithm for team assignment
- Optical Flow: Measure camera movement
- Perspective Transformation: Represent scene depth and perspective
- Speed and distance calculation per player

## Trained Models
- [Trained Yolo v5](https://drive.google.com/file/d/1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK/view?usp=sharing)

## Sample video
-  [Sample input video](https://drive.google.com/file/d/1t6agoqggZKx6thamUuPAIdN_1zR9v9S_/view?usp=sharing)

## Camera Movement Compensation Module

### Camera Movement Detection and Compensation

The camera movement estimator module (`Module/camera_movement_estimator/camera_movement_estimator.py`) provides functionality to detect and compensate for camera movement in football videos:

#### `CameraMovementEstimator(frame)`
Initializes the camera movement estimator with the first frame of the video.

#### `get_camera_movement(frames, read_from_stub=False, stub_path=None)`
Detects camera movement between consecutive frames using optical flow.
- **Parameters:**
  - `frames`: List of video frames in BGR format
  - `read_from_stub`: Whether to read results from cache file
  - `stub_path`: Path to cache file for storing/loading results
- **Returns:** List of camera movement vectors for each frame `[[x, y], ...]`
- **Algorithm:** Uses Lucas-Kanade optical flow to track fixed features in video edge regions

#### `add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)`
Compensates tracked object positions by subtracting camera movement.
- **Parameters:**
  - `tracks`: Dictionary containing tracking results for all objects
  - `camera_movement_per_frame`: List of camera movement vectors per frame
- **Returns:** Updates tracks with `position_adjusted` field containing compensated positions

#### `draw_camera_movement(frames, camera_movement_per_frame)`
Visualizes camera movement information on video frames.
- **Parameters:**
  - `frames`: List of video frames
  - `camera_movement_per_frame`: List of camera movement vectors per frame
- **Returns:** Frames with semi-transparent overlay showing camera movement values

### Usage Example

```python
from Module.common.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator

# Initialize with first frame
estimator = CameraMovementEstimator(first_frame)

# Calculate camera movement for all frames
camera_movement = estimator.get_camera_movement(frames, stub_path="camera_movement.pkl")

# Compensate tracked object positions
estimator.add_adjust_positions_to_tracks(tracks, camera_movement)

# Visualize camera movement
output_frames = estimator.draw_camera_movement(frames, camera_movement)
```

## Ball Tracking Module

### Ball Path Processing Functions

The ball tracking module (`Module/ball/ball.py`) provides several functions for processing ball trajectory data:

#### `extract_ball_paths(tracks: Dict[str, Any]) -> List[np.ndarray]`
Extracts ball position data from tracking results and returns formatted path arrays.

#### `replace_outliers_based_on_distance(ball_paths: List[np.ndarray], distance_threshold: float = 500.0) -> List[np.ndarray]`
Removes outlier points from ball trajectories based on distance threshold.
- **Parameters:**
  - `ball_paths`: List of ball path arrays, each with shape (N, 2)
  - `distance_threshold`: Distance threshold for outlier detection (default: 500.0)
- **Returns:** Processed ball paths with outliers replaced by [np.nan, np.nan]
- **Algorithm:** If a point is far from both its previous and next points, but the previous and next points are close to each other, the middle point is considered an outlier.

#### `ball_filter(trajectory, window_length=5, polyorder=2)`
Applies Savitzky-Golay filtering to smooth ball trajectories.

#### `interpolate_ball_positions_transformed(ball_paths)`
Interpolates missing ball positions using linear interpolation.
- **Parameters:**
  - `ball_paths`: List of ball path arrays, may contain NaN values
- **Returns:** Interpolated ball paths with NaN values replaced by interpolated positions
- **Algorithm:** Uses scipy.interpolate.interp1d for linear interpolation of missing points

### Usage Example

```python
from Module.plugin.ball import replace_outliers_based_on_distance
import numpy as np

# Example ball path with outlier
ball_path = np.array([[0, 0], [10, 10], [1000, 1000], [20, 20], [30, 30]])
processed_path = replace_outliers_based_on_distance([ball_path], distance_threshold=50.0)
# Result: [[0, 0], [10, 10], [nan, nan], [20, 20], [30, 30]]
```

## Radar图生成测试功能

### 单帧Radar图生成测试

项目新增了单帧radar图生成测试功能，可以快速测试球场关键点检测和球员位置映射：

#### `test()` 函数
测试单帧视频的radar图生成功能，包括：
- 球员、裁判、球的检测和跟踪
- 球场关键点检测和透视变换
- 球队颜色分配和队伍分类
- Radar图生成和保存
- 视频帧标注保存

#### 使用方法
```python
# 运行测试
python test.py
```

#### 输出文件
- `IO/figures/test_radar.png`: 生成的radar图，显示球员在球场上的位置
- `IO/figures/test_annotated_frame.png`: 标注后的视频帧，显示检测到的球员、裁判和球的边界框

#### 关键特性
- **单帧处理**: 只处理视频的第10帧，避免平滑处理问题
- **自动队伍分类**: 使用K-means聚类自动分配球员队伍
- **完整标注**: 同时生成radar图和带标注的视频帧
- **调试友好**: 包含详细的调试信息和错误处理

#### 故障排除
如果遇到"变换矩阵为None"的警告，这通常是因为：
1. 球场关键点检测置信度不足
2. 单帧测试时平滑处理导致的问题（已自动修复）
3. 视频帧中球场关键点不够清晰

## BoTSORT追踪器使用方法

### 两种处理模式对比

BoTSORT追踪器支持两种主要的使用方法，各有优劣：

#### 1. 逐帧实时处理方法
- **适用场景**：实时视频流、摄像头输入、需要立即反馈
- **优势**：延迟低、内存占用小、灵活性强
- **劣势**：GPU利用率较低、检测速度较慢

#### 2. 批量预处理方法
- **适用场景**：视频文件离线处理、长视频分析
- **优势**：GPU利用率高、检测速度快（2-5倍）、结果可复用
- **劣势**：内存占用大、延迟高、不够灵活

**详细对比分析**：请参考 [`Module/common/trackers/bot_sort/BoTSORT两种使用方法对比.md`](Module/common/trackers/bot_sort/BoTSORT两种使用方法对比.md)

### 项目中的实现

本项目在 `Module/common/trackers/tracker_botsort.py` 中采用**批量预处理方法**，因为：
- 主要处理完整的视频文件
- 追求最佳处理速度和GPU利用率
- 支持结果缓存（stub机制），便于调试和参数调优

### 基本使用示例

```python
from Module.common.trackers.tracker_botsort import Tracker

# 初始化追踪器
tracker = Tracker(
    model_path="Storage/models/yolo/best.pt",
    nms_threshold=0.5,
    reid_model_path="Storage/models/reid/osnet_x0_25_msmt17.pt",
    device='cuda',
    fp16=True
)

# 处理视频帧（批量预处理模式）
frames = load_video_frames("input_video.mp4")
tracks = tracker.get_object_tracks(frames, stub_path="tracks.pkl")
```

更多详细信息请参考：
- [`Module/common/trackers/bot_sort/BoTSORT使用文档.md`](Module/common/trackers/bot_sort/BoTSORT使用文档.md) - 基本使用方法
- [`Module/common/trackers/bot_sort/BoTSORT两种使用方法对比.md`](Module/common/trackers/bot_sort/BoTSORT两种使用方法对比.md) - 两种处理模式对比
- [`Module/common/trackers/bot_sort/追踪器与检测器的关系分析.md`](Module/common/trackers/bot_sort/追踪器与检测器的关系分析.md) - 检测与跟踪的独立性分析

### 重要说明：检测与跟踪的独立性

**追踪器的更新结果不影响下一帧的检测。** 检测（YOLO）和跟踪（BoTSORT）是两个完全独立的阶段：
- 检测器每帧都进行**全图独立检测**，不依赖跟踪结果
- 跟踪器只负责**关联匹配**和**状态维护**，不反馈给检测器
- 这种设计保证了检测精度不受跟踪影响，避免误差累积

详见 [`追踪器与检测器的关系分析.md`](Module/common/trackers/bot_sort/追踪器与检测器的关系分析.md)

### 追踪算法对比

项目中实现了两种追踪算法：

1. **ByteTrack** (`tracker.py`) - 快速简单，适合基础场景
2. **BoTSORT** (`tracker_botsort.py`) - 功能强大，适合复杂场景

**详细对比分析**：请参考 [`Module/common/trackers/ByteTrack与BoTSORT原理对比.md`](Module/common/trackers/ByteTrack与BoTSORT原理对比.md)

### 相机运动的影响与补偿

在足球视频分析中，**相机运动**是影响跟踪准确性的重要因素：

- **影响**：
  - 静止目标被误判为移动
  - 运动速度计算错误
  - 轨迹匹配失败，ID切换频繁
  - 位置测量不准确

- **解决方案**：
  - **BoTSORT内置GMC**：实时补偿相机运动，提高匹配准确性
  - **后处理补偿**：修正位置数据，确保速度、距离测量准确

**详细分析**：请参考 [`Module/common/trackers/相机运动对跟踪的影响分析.md`](Module/common/trackers/相机运动对跟踪的影响分析.md)

### 球场坐标与GMC的关系

项目使用透视变换将图像坐标转换为球场坐标（米），用于速度和距离计算。**即使使用球场坐标，GMC模块在跟踪匹配阶段仍然必要**：

- **跟踪匹配阶段**：在图像空间进行，需要边界框信息（IoU），GMC补偿相机运动对匹配的影响
- **分析计算阶段**：使用球场坐标（米），不受相机运动影响，不需要额外补偿

**详细分析**：请参考 [`Module/common/trackers/球场坐标跟踪与GMC必要性分析.md`](Module/common/trackers/球场坐标跟踪与GMC必要性分析.md)

### 基于球场坐标的跟踪方案

## 球队分类算法

### 概述

项目使用**基于SigLIP深度视觉特征的分类算法**来区分两支球队。相比传统的基于颜色的方法，新算法具有更高的准确率和鲁棒性。

### 技术特点

- **特征提取**：使用预训练的SigLIP视觉模型（`google/siglip-base-patch16-224`）提取768维深度特征
- **降维处理**：使用UMAP将特征降维到3维，保留全局和局部结构
- **聚类分类**：使用KMeans对降维后的特征进行2类聚类，区分两支球队

### 优势

- ✅ **高准确率**：能捕捉纹理、形状等视觉模式，不仅限于颜色
- ✅ **鲁棒性强**：对光照变化、阴影、角度变化不敏感
- ✅ **适应性强**：能区分颜色相似但纹理不同的球衣

### 使用方法

```python
from Module.common.team_assigner import TeamAssigner

# 初始化（支持GPU加速）
team_assigner = TeamAssigner(device="cuda", batch_size=32)

# 在第一帧初始化分类器
team_assigner.assign_team_color(first_frame, player_detections)

# 为每个球员分配队伍
team_id = team_assigner.get_player_team(frame, player_bbox, player_id)
```

### 依赖要求

新算法需要以下额外依赖：
- `transformers`：用于加载SigLIP模型
- `torch`：PyTorch深度学习框架
- `umap-learn`：UMAP降维库
- `tqdm`：进度条显示

安装命令：
```bash
pip install transformers torch umap-learn tqdm
```

### 性能说明

- **处理速度**：较慢（需要GPU加速），建议使用CUDA设备
- **内存占用**：较大（需要加载预训练模型）
- **首次运行**：会自动下载SigLIP模型（约500MB）

**详细对比分析**：请参考 [`球队分类算法对比分析.md`](球队分类算法对比分析.md)

---

## 无追踪的球员检测与可视化流程

### 概述

`test_no_tracker.py` 提供了一个简化的流程，专注于球员检测和可视化，不进行追踪。这个流程适用于只需要分析球员位置分布、不需要追踪单个球员ID的场景。

### 主要功能

1. **YOLO检测球员**：使用YOLO模型检测视频中的球员（不进行追踪）
2. **视角转换**：初始化视角转换器，将像素坐标转换为场地坐标（米）
3. **队伍分配**：使用基于SigLIP深度视觉特征的分类算法自动分配队伍
4. **数据存储**：按帧/队伍/ID存储球员坐标信息，支持缓存
5. **可视化**：在球场图上可视化球员位置，区分两只球队

### 数据结构

球员数据按以下结构存储：

```python
player_data = {
    frame_num: {
        team_id: {
            player_id: {
                'bbox': [x1, y1, x2, y2],  # 检测框（像素坐标）
                'position_pixel': [x, y],  # 脚部位置（像素坐标）
                'position_transformed': [x, y]  # 场地坐标（米）
            }
        }
    }
}
```

### 使用方法

```python
# 直接运行脚本
python test_no_tracker.py
```

### 配置参数

在脚本中可以修改以下配置：

- `INPUT_VIDEO_PATH`: 输入视频路径
- `PLAYER_DETECTION_PATH`: YOLO模型路径
- `DEVICE`: 运行设备（"cpu" 或 "cuda"）
- `READ_FROM_STUB`: 是否从缓存读取（True/False）
- `DETECTION_CONF_THRESHOLD`: 检测置信度阈值（默认0.3）
- `NMS_THRESHOLD`: NMS阈值（默认0.5）

### 输出结果

1. **缓存文件**：`Storage/stubs/player_data_stub.pkl` - 存储处理后的球员数据
2. **可视化图像**：`IO/figures/players_frame_{frame_num}.png` - 每帧的球场可视化图

### 与完整流程的区别

| 特性 | 完整流程 (main.py) | 无追踪流程 (test_no_tracker.py) |
|------|-------------------|-------------------------------|
| 追踪ID | ✅ 有 | ❌ 无 |
| 检测 | ✅ YOLO | ✅ YOLO |
| 视角转换 | ✅ 有 | ✅ 有 |
| 队伍分配 | ✅ 有 | ✅ 有 |
| 速度/距离计算 | ✅ 有 | ❌ 无 |
| 球追踪 | ✅ 有 | ❌ 无 |
| 适用场景 | 完整分析 | 位置分布分析 |

### 基于球场坐标的跟踪方案

理论上，如果透视变换足够稳定，可以**在球场坐标空间进行跟踪**，从而**弃用GMC模块**：

- **方案**：YOLO检测 → 立即转换为球场坐标 → 在球场坐标空间跟踪
- **优势**：不受相机运动影响、物理意义清晰、算法简化
- **条件**：透视变换矩阵稳定、关键点检测准确

**详细分析**：请参考 [`Module/common/trackers/基于球场坐标跟踪的可行性分析.md`](Module/common/trackers/基于球场坐标跟踪的可行性分析.md)

## Requirements
To run this project, you need to have the following requirements installed:
- Python 3.x
- ultralytics
- supervision
- OpenCV
- NumPy
- Matplotlib
- Pandas
- scipy (for signal processing)
- deep-oc-sort (for BoTSORT tracker ReID features)