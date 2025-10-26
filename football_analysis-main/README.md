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
from Module.ball.ball import replace_outliers_based_on_distance
import numpy as np

# Example ball path with outlier
ball_path = np.array([[0, 0], [10, 10], [1000, 1000], [20, 20], [30, 30]])
processed_path = replace_outliers_based_on_distance([ball_path], distance_threshold=50.0)
# Result: [[0, 0], [10, 10], [nan, nan], [20, 20], [30, 30]]
```

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