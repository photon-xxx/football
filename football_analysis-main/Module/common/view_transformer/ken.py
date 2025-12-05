"""
视野范围计算模块

该模块提供计算视频帧在场地坐标系中可见范围的功能。
通过透视变换矩阵，将视频帧的四个角点转换到场地坐标系，
从而确定当前帧在场地上的可视区域。
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional
from Storage.field_configs.soccer import SoccerPitchConfiguration
def calculate_visible_range(
    transform_matrix: np.ndarray,
    video_frame: np.ndarray
) -> Optional[List[Tuple[float, float]]]:
    """
    计算视频帧在场地坐标系中的可见范围
    
    该函数将视频帧的四个角点（左上、右上、右下、左下）通过透视变换矩阵
    转换到场地坐标系，返回可见区域的四个顶点坐标。
    
    Args:
        transform_matrix: 透视变换矩阵，形状为 (3, 3) 的 numpy 数组
            该矩阵将像素坐标转换为场地坐标（单位：厘米）
        video_frame: 视频帧，numpy 数组，形状为 (height, width, channels)
            用于获取帧的宽度和高度
    
    Returns:
        如果转换成功，返回四个顶点坐标的列表，顺序为：
        [左上角, 右上角, 右下角, 左下角]
        每个顶点为 (x, y) 元组，单位为厘米
        如果转换失败，返回 None
    
    Raises:
        ValueError: 当输入参数格式不正确时
    
    Example:
        >>> import cv2
        >>> import numpy as np
        >>> from Module.common.view_transformer.view_transformer import ViewTransformer
        >>> 
        >>> # 读取视频帧
        >>> cap = cv2.VideoCapture("video.mp4")
        >>> ret, frame = cap.read()
        >>> 
        >>> # 获取转换矩阵（假设已经初始化了 ViewTransformer）
        >>> view_transformer = ViewTransformer(video_frames)
        >>> matrix = view_transformer.perspective_transformers[0]
        >>> 
        >>> # 计算可见范围
        >>> visible_range = calculate_visible_range(matrix, frame)
        >>> if visible_range:
        >>>     print(f"左上角: {visible_range[0]}")
        >>>     print(f"右上角: {visible_range[1]}")
        >>>     print(f"右下角: {visible_range[2]}")
        >>>     print(f"左下角: {visible_range[3]}")
    """
    # 参数验证
    if transform_matrix is None:
        print("错误: 转换矩阵为 None")
        return None
    
    if not isinstance(transform_matrix, np.ndarray):
        raise ValueError("转换矩阵必须是 numpy 数组")
    
    if transform_matrix.shape != (3, 3):
        raise ValueError(f"转换矩阵形状错误，期望 (3, 3)，实际为 {transform_matrix.shape}")
    
    if video_frame is None:
        raise ValueError("视频帧不能为 None")
    
    if not isinstance(video_frame, np.ndarray):
        raise ValueError("视频帧必须是 numpy 数组")
    
    if len(video_frame.shape) < 2:
        raise ValueError("视频帧维度错误，至少需要 2 维 (height, width)")
    
    # 获取视频帧的宽度和高度
    frame_height, frame_width = video_frame.shape[:2]
    
    # 定义视频帧的四个角点（像素坐标）
    # 顺序：左上角 -> 右上角 -> 右下角 -> 左下角
    corner_points = np.array([
        [0, 0],                      # 左上角
        [frame_width, 0],            # 右上角
        [frame_width, frame_height], # 右下角
        [0, frame_height]            # 左下角
    ], dtype=np.float32)
    
    # 将角点转换为适合 cv2.perspectiveTransform 的格式
    # 需要形状为 (N, 1, 2)
    corner_points_reshaped = corner_points.reshape(-1, 1, 2)
    
    try:
        # 使用透视变换矩阵将像素坐标转换为场地坐标
        transformed_points = cv2.perspectiveTransform(
            corner_points_reshaped,
            transform_matrix
        )
        
        # 将结果转换回 (N, 2) 格式
        transformed_points = transformed_points.reshape(-1, 2)
        
        # 检查转换结果是否包含无效值
        if np.isnan(transformed_points).any() or np.isinf(transformed_points).any():
            print("警告: 转换结果包含无效值 (NaN 或 Inf)")
            return None
        
        # 转换为列表格式，每个元素为 (x, y) 元组
        # 顺序保持：左上角 -> 右上角 -> 右下角 -> 左下角
        visible_range = [
            (float(transformed_points[0][0]), float(transformed_points[0][1])),  # 左上角
            (float(transformed_points[1][0]), float(transformed_points[1][1])),  # 右上角
            (float(transformed_points[2][0]), float(transformed_points[2][1])),  # 右下角
            (float(transformed_points[3][0]), float(transformed_points[3][1]))   # 左下角
        ]
        
        return visible_range
        
    except cv2.error as e:
        print(f"OpenCV 透视变换错误: {e}")
        return None
    except Exception as e:
        print(f"计算可见范围时发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None



# -------------------- 测试入口 --------------------

