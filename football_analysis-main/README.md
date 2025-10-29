# Football Analysis Project

## Introduction
The goal of this project is to detect and track players, referees, and footballs in a video using YOLO, one of the best AI object detection models available. We will also train the model to improve its performance. Additionally, we will assign players to teams based on the colors of their t-shirts using Kmeans for pixel segmentation and clustering. With this information, we can measure a team's ball acquisition percentage in a match. We will also use optical flow to measure camera movement between frames, enabling us to accurately measure a player's movement. Furthermore, we will implement perspective transformation to represent the scene's depth and perspective, allowing us to measure a player's movement in meters rather than pixels. Finally, we will calculate a player's speed and the distance covered. This project covers various concepts and addresses real-world problems, making it suitable for both beginners and experienced machine learning engineers.

![Screenshot](IO/output_videos/screenshot.png)

## Modules Used
The following modules are used in this project:
- YOLO: AI object detection model
- Kmeans: Pixel segmentation and clustering to detect t-shirt color
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