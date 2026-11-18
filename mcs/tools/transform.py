"""
坐标转换模块 (Transform Module)

提供四类坐标系统之间的转换：
1. Video Pixel (视频像素) - 相机录像中的像素坐标
2. Pitch Coordinate (球场坐标) - 真实球场物理坐标 (单位: cm)
3. Canvas Pixel (可视化像素) - 球场俯视图上的像素坐标

主要功能：
- video_pixel_to_pitch(): 视频像素 → 球场坐标 (含去畸变 + 单应性变换)
- pitch_to_video_pixel(): 球场坐标 → 视频像素 (反向单应性变换)
- pitch_to_canvas_pixel(): 球场坐标 → 可视化像素 (简单缩放)
- canvas_pixel_to_pitch(): 可视化像素 → 球场坐标 (反向缩放)
- compute_transform_matrix(): 计算单应性矩阵
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Union
from soccer import SoccerPitchConfiguration


# ============================================================================
# 1. 单应性矩阵计算
# ============================================================================

def compute_transform_matrix(
        keypoint_indices: List[int],
        image_points: List[Tuple[int, int]],
        config: SoccerPitchConfiguration = None,
        method: str = "homography",
        distortion_params: Optional[np.ndarray] = None,
        image_center: Optional[Tuple[float, float]] = None,
        enable_k3: bool = True
) -> np.ndarray:
    """
    计算从图像像素空间到球场坐标空间的单应性矩阵
    
    ⚠️ 如果图像存在畸变，应传入 distortion_params 参数自动进行去畸变。

    Args:
        keypoint_indices: 球场关键点索引列表（至少4个点）
        image_points: 图像中对应的像素坐标列表（至少4个点）
        config: 足球场配置对象，如果为None则使用默认配置
        method: 计算方法
               - "homography": 使用cv2.findHomography (推荐，鲁棒性最好)
               - "perspective": 使用cv2.getPerspectiveTransform (仅限4点)
        distortion_params: 可选，畸变参数 (来自 de_distortion.calibrate_distortion)
                          如果提供，会先对 image_points 进行去畸变
        image_center: 可选，图像中心 (配合 distortion_params 使用)
        enable_k3: 是否启用 k3 畸变参数

    Returns:
        np.ndarray: 3x3的单应性矩阵 (Video Pixel -> Pitch Coordinate)
    """
    print(f"[Transform] compute_transform_matrix called with {len(keypoint_indices)} points.")
    
    if config is None:
        config = SoccerPitchConfiguration()

    if len(keypoint_indices) < 4 or len(image_points) < 4:
        raise ValueError(f"至少需要4个对应点，当前仅有 {len(keypoint_indices)} 个")
    
    if len(keypoint_indices) != len(image_points):
        raise ValueError("关键点索引和图像点数量必须一致")

    pitch_points = []
    for idx in keypoint_indices:
        # 安全检查索引
        if idx >= len(config.vertices):
            raise ValueError(f"Vertex index {idx} out of range (max {len(config.vertices)-1})")
        pt = config.vertices[idx]
        pitch_points.append(pt)

    src_points = np.array(image_points, dtype=np.float32)
    
    # 去畸变前置处理
    if distortion_params is not None:
        try:
            from de_distortion import undistort_points
            src_points = undistort_points(
                src_points, 
                distortion_params, 
                image_center,
                enable_k3=enable_k3
            ).astype(np.float32)
            print(f"[Transform] 已对 {len(src_points)} 个关键点进行去畸变")
        except ImportError:
            print("[Transform] 警告: 无法导入 de_distortion 模块，跳过去畸变步骤")
    
    dst_points = np.array(pitch_points, dtype=np.float32)

    transform_matrix = None

    if method == "homography":
        # 优先使用所有点进行最小二乘拟合 (method=0)
        transform_matrix, mask = cv2.findHomography(
            src_points,
            dst_points,
            method=0  # 0 = 使用所有点进行最小二乘拟合
        )
        
        if transform_matrix is None:
            # 回退到 RANSAC
            print("[Transform] 警告: 最小二乘拟合失败，尝试使用 RANSAC...")
            transform_matrix, mask = cv2.findHomography(
                src_points,
                dst_points,
                method=cv2.RANSAC,
                ransacReprojThreshold=10.0
            )
            
        if transform_matrix is None:
            print("[Transform] Error: cv2.findHomography returned None!")
            raise ValueError("findHomography计算失败，请检查输入点")
        
        print(f"[Transform] Homography matrix computed successfully.")
        return transform_matrix
    
    elif method == "perspective":
        if len(keypoint_indices) != 4 or len(image_points) != 4:
            raise ValueError("使用perspective方法需要精确的4个对应点")
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return transform_matrix
        
    else:
        # 默认回退到 homography
        print(f"[Transform] 警告: 未知方法 '{method}'，回退到 'homography'")
        transform_matrix, mask = cv2.findHomography(
            src_points,
            dst_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=10.0
        )
        if transform_matrix is None:
            raise ValueError("findHomography计算失败")
        return transform_matrix


# ============================================================================
# 2. 视频像素 ↔ 球场坐标 (复杂变换：单应性 + 畸变)
# ============================================================================

def video_pixel_to_pitch(
        video_coords: np.ndarray,
        homography_matrix: np.ndarray,
        distortion_params: Optional[np.ndarray] = None,
        distortion_info: Optional[dict] = None,
        enable_k3: bool = True
) -> np.ndarray:
    """
    视频像素坐标 → 球场坐标 (完整转换流程)
    
    流程: 视频像素 → [去畸变] → [单应性变换] → 球场坐标
    
    Args:
        video_coords: 视频像素坐标
            - shape: (N, 2) 或 (N, 1, 2)
            - 格式: [[u1,v1], [u2,v2], ...]
        homography_matrix: 3x3 单应性矩阵
        distortion_params: 可选，畸变参数 [k1, k2, p1, p2, k3]
        distortion_info: 可选，畸变信息字典 {'center': (cx, cy), 'enable_k3': bool}
        enable_k3: 是否启用 k3 畸变参数
        
    Returns:
        pitch_coords: (N, 2) 球场坐标数组 [[X1,Y1], [X2,Y2], ...] (单位: cm)
        
    Example:
        >>> video_pts = np.array([[800, 600], [1200, 800]])
        >>> pitch_pts = video_pixel_to_pitch(video_pts, H, distortion_params)
        >>> print(pitch_pts)  # [[3300, 2200], [4500, 3100]]
    """
    video_coords = np.array(video_coords, dtype=np.float32)
    if video_coords.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    # 步骤1: 去畸变 (如果有畸变参数)
    if distortion_params is not None:
        try:
            from de_distortion import undistort_points
            
            # 获取图像中心
            if distortion_info is not None:
                image_center = distortion_info.get('center', None)
                if 'enable_k3' in distortion_info:
                    enable_k3 = distortion_info['enable_k3']
            else:
                image_center = None
            
            video_coords = undistort_points(
                video_coords,
                distortion_params,
                image_center,
                enable_k3=enable_k3
            )
        except ImportError:
            print("[Transform] 警告: 无法导入 de_distortion 模块，跳过去畸变")
    
    # 步骤2: 单应性变换
    # 确保输入格式为 (N, 1, 2)
    if video_coords.ndim == 2:
        video_coords = video_coords.reshape(-1, 1, 2)
    
    pitch_coords = cv2.perspectiveTransform(video_coords, homography_matrix)
    return pitch_coords.reshape(-1, 2)


def pitch_to_video_pixel(
        pitch_coords: np.ndarray,
        homography_matrix: np.ndarray,
        apply_distortion: bool = True,
        distortion_params: Optional[np.ndarray] = None,
        distortion_info: Optional[dict] = None,
        enable_k3: bool = True
) -> np.ndarray:
    """
    球场坐标 → 视频像素坐标 (反向投影)
    
    流程: 球场坐标 → [逆单应性变换] → [正向畸变] → 视频像素
    
    用途：在原始视频画面上绘制球场元素（如越位线、热力图、区域标记等）
    
    Args:
        pitch_coords: 球场坐标
            - shape: (N, 2)
            - 格式: [[X1,Y1], [X2,Y2], ...] (单位: cm)
        homography_matrix: 3x3 单应性矩阵
        apply_distortion: 是否应用正向畸变（默认 True，适用于在原始视频上绘制）
        distortion_params: 可选，畸变参数
        distortion_info: 可选，畸变信息 {'center': (cx, cy), 'enable_k3': bool}
        enable_k3: 是否启用 k3 畸变参数
        
    Returns:
        video_coords: (N, 2) 视频像素坐标数组 [[u1,v1], [u2,v2], ...]
        
    Example:
        >>> # 在原始视频上绘制中场线
        >>> pitch_pts = np.array([[3300, 0], [3300, 4400]])
        >>> video_pts = pitch_to_video_pixel(pitch_pts, H, True, params, info)
        >>> cv2.line(video_frame, tuple(video_pts[0]), tuple(video_pts[1]), (0,255,0), 2)
        
    Note:
        - apply_distortion=True: 用于在原始视频上绘制（推荐）
        - apply_distortion=False: 用于在去畸变后的视频上绘制
    """
    pitch_coords = np.array(pitch_coords, dtype=np.float32)
    if pitch_coords.size == 0:
        return np.empty((0, 2), dtype=np.float32)

    # 步骤1: 逆单应性变换 (球场坐标 → 理想像素坐标)
    H_inv = np.linalg.inv(homography_matrix)
    
    if pitch_coords.ndim == 2:
        pitch_coords = pitch_coords.reshape(-1, 1, 2)
    
    video_coords = cv2.perspectiveTransform(pitch_coords, H_inv)
    video_coords = video_coords.reshape(-1, 2)
    
    # 步骤1.5: 检查理想像素是否在合理范围内（过滤视野外的点）
    if distortion_info is not None and 'center' in distortion_info:
        cx, cy = distortion_info['center']
        # 计算相对于图像中心的归一化距离
        # 假设图像尺寸约为 2*cx 或 2*cy，合理的视野范围应该在图像尺寸的 1.5 倍内
        max_reasonable_distance = max(cx, cy) * 3.0  # 允许最大距离为中心坐标的3倍
        
        for i in range(len(video_coords)):
            dist = np.sqrt((video_coords[i, 0] - cx)**2 + (video_coords[i, 1] - cy)**2)
            if dist > max_reasonable_distance:
                video_coords[i] = [np.nan, np.nan]  # 标记为无效
    
    # 步骤2: 正向畸变 (理想像素 → 畸变像素)
    if apply_distortion and distortion_params is not None:
        try:
            from de_distortion import distort_points
            
            # 获取图像中心
            if distortion_info is not None:
                image_center = distortion_info.get('center', None)
                if 'enable_k3' in distortion_info:
                    enable_k3 = distortion_info['enable_k3']
            else:
                image_center = None
            
            video_coords = distort_points(
                video_coords,
                distortion_params,
                image_center,
                enable_k3=enable_k3
            )
        except ImportError:
            print("[Transform] 警告: 无法导入 de_distortion.distort_points，跳过加畸变")
    
    return video_coords


# ============================================================================
# 3. 球场坐标 ↔ 可视化画布像素 (简单变换：线性缩放)
# ============================================================================

def pitch_to_canvas_pixel(
        pitch_coords: np.ndarray,
        scale: float = 0.1,
        padding: int = 50
) -> np.ndarray:
    """
    球场坐标 → 可视化画布像素坐标
    
    用于在球场俯视图上绘制元素。
    
    转换公式:
        canvas_x = pitch_x * scale + padding
        canvas_y = pitch_y * scale + padding
    
    Args:
        pitch_coords: 球场坐标
            - shape: (N, 2)
            - 格式: [[X1,Y1], [X2,Y2], ...] (单位: cm)
        scale: 缩放比例 (默认 0.1，即 1cm → 0.1 像素)
        padding: 边距 (像素)
        
    Returns:
        canvas_coords: (N, 2) 画布像素坐标 [[px1,py1], [px2,py2], ...]
        
    Example:
        >>> pitch_pts = np.array([[3300, 2200], [6600, 4400]])
        >>> canvas_pts = pitch_to_canvas_pixel(pitch_pts, scale=0.1, padding=50)
        >>> print(canvas_pts)  # [[380, 270], [710, 490]]
    """
    pitch_coords = np.array(pitch_coords, dtype=np.float32)
    
    if pitch_coords.ndim == 1:
        pitch_coords = pitch_coords.reshape(1, -1)
    
    canvas_x = pitch_coords[..., 0] * scale + padding
    canvas_y = pitch_coords[..., 1] * scale + padding
    
    canvas_coords = np.stack([canvas_x, canvas_y], axis=-1).astype(np.int32)
    return canvas_coords


def canvas_pixel_to_pitch(
        canvas_coords: np.ndarray,
        scale: float = 0.1,
        padding: int = 50
) -> np.ndarray:
    """
    可视化画布像素坐标 → 球场坐标
    
    用于从俯视图点击位置获取球场坐标。
    
    转换公式:
        pitch_x = (canvas_x - padding) / scale
        pitch_y = (canvas_y - padding) / scale
    
    Args:
        canvas_coords: 画布像素坐标
            - shape: (N, 2)
            - 格式: [[px1,py1], [px2,py2], ...]
        scale: 缩放比例
        padding: 边距
        
    Returns:
        pitch_coords: (N, 2) 球场坐标数组 [[X1,Y1], [X2,Y2], ...] (单位: cm)
        
    Example:
        >>> canvas_pts = np.array([[380, 270], [710, 490]])
        >>> pitch_pts = canvas_pixel_to_pitch(canvas_pts, scale=0.1, padding=50)
        >>> print(pitch_pts)  # [[3300, 2200], [6600, 4400]]
    """
    canvas_coords = np.array(canvas_coords, dtype=np.float32)
    
    if canvas_coords.ndim == 1:
        canvas_coords = canvas_coords.reshape(1, -1)
    
    pitch_x = (canvas_coords[..., 0] - padding) / scale
    pitch_y = (canvas_coords[..., 1] - padding) / scale
    
    pitch_coords = np.stack([pitch_x, pitch_y], axis=-1)
    return pitch_coords


# ============================================================================
# 4. 工具函数 (兼容旧代码)
# ============================================================================

def apply_transform(
        transform_matrix: np.ndarray,
        points: np.ndarray
) -> np.ndarray:
    """
    应用变换矩阵将坐标转换（通用接口）
    
    ⚠️ 已弃用，建议使用 video_pixel_to_pitch() 或 pitch_to_video_pixel()

    Args:
        transform_matrix: 3x3变换矩阵
        points: 坐标点数组，形状为(N, 2)或(N, 1, 2)

    Returns:
        np.ndarray: 转换后的坐标点数组
    """
    if points.ndim == 2 and points.shape[1] == 2:
        points = points.reshape(-1, 1, 2)

    transformed = cv2.perspectiveTransform(points.astype(np.float32), transform_matrix)
    return transformed.reshape(-1, 2)


# ============================================================================
# 5. 测试代码
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("Transform 模块功能测试")
    print("=" * 70)
    
    # ===== 测试1: 球场坐标 ↔ 可视化像素 =====
    print("\n【测试1】球场坐标 ↔ 可视化画布像素")
    print("-" * 70)
    
    pitch_pts = np.array([[3300, 2200], [6600, 4400], [0, 0]])
    print(f"球场坐标 (cm):\n{pitch_pts}")
    
    canvas_pts = pitch_to_canvas_pixel(pitch_pts, scale=0.1, padding=50)
    print(f"→ 可视化像素:\n{canvas_pts}")
    
    pitch_pts_back = canvas_pixel_to_pitch(canvas_pts, scale=0.1, padding=50)
    print(f"→ 反向转换回球场坐标:\n{pitch_pts_back}")
    print(f"误差: {np.max(np.abs(pitch_pts - pitch_pts_back)):.6f} cm")
    
    # ===== 测试2: 视频像素 → 球场坐标 (不含畸变) =====
    print("\n【测试2】视频像素 → 球场坐标 (单应性变换)")
    print("-" * 70)
    
    # 构造一个简单的单应性矩阵
    H_test = np.array([
        [4.0, 1.4, -1600],
        [-0.9, -1.7, 1200],
        [-0.00002, -0.003, 1.0]
    ])
    
    video_pts = np.array([[800, 600], [1000, 700]])
    print(f"视频像素:\n{video_pts}")
    
    pitch_coords = video_pixel_to_pitch(video_pts, H_test)
    print(f"→ 球场坐标 (cm):\n{pitch_coords}")
    
    video_pts_back = pitch_to_video_pixel(pitch_coords, H_test)
    print(f"→ 反向转换回视频像素:\n{video_pts_back}")
    print(f"误差: {np.max(np.abs(video_pts - video_pts_back)):.6f} 像素")
    
    # ===== 测试3: 完整流程 =====
    print("\n【测试3】完整坐标转换链")
    print("-" * 70)
    print("视频像素 → 球场坐标 → 可视化像素")
    
    video_pt = np.array([[800, 600]])
    print(f"1. 视频像素: {video_pt[0]}")
    
    pitch_pt = video_pixel_to_pitch(video_pt, H_test)
    print(f"2. → 球场坐标: {pitch_pt[0]} cm")
    
    canvas_pt = pitch_to_canvas_pixel(pitch_pt, scale=0.1, padding=50)
    print(f"3. → 可视化像素: {canvas_pt[0]}")
    
    # ===== 测试4: 正向畸变（用于视频绘制） =====
    print("\n【测试4】正向畸变测试（球场 → 视频画面绘制）")
    print("-" * 70)
    
    # 模拟畸变参数
    distortion_params_test = np.array([0.1, -0.05, 0.001, 0.001, 0.01])
    image_center_test = (960, 540)  # 1920x1080 视频的中心
    
    print("场景：在原始视频上绘制中场线")
    midline_pitch = np.array([[3300, 0], [3300, 2200], [3300, 4400]])
    print(f"1. 球场坐标（中场线三个点）:\n{midline_pitch}")
    
    # 不加畸变（理想像素）
    midline_ideal = pitch_to_video_pixel(midline_pitch, H_test, apply_distortion=False)
    print(f"2. → 理想像素（无畸变）:\n{midline_ideal.astype(int)}")
    
    # 加畸变（实际视频像素）
    try:
        from de_distortion import distort_points
        midline_video = pitch_to_video_pixel(
            midline_pitch, H_test, 
            apply_distortion=True,
            distortion_params=distortion_params_test,
            distortion_info={'center': image_center_test}
        )
        print(f"3. → 畸变像素（可在原始视频上绘制）:\n{midline_video.astype(int)}")
        print(f"✅ 畸变偏移: {np.mean(np.abs(midline_video - midline_ideal)):.2f} 像素")
    except ImportError:
        print("⚠️ de_distortion.distort_points 未实现，跳过测试")
    
    print("\n" + "=" * 70)
    print("✅ Transform 模块测试完成")
    print("=" * 70)
