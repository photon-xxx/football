import numpy as np
import cv2
import sys
import warnings
import json
from dataclasses import dataclass, field
from typing import Tuple, Generator, Optional, Literal, List
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import dataclasses
import yaml
import check_valid
import config
import os
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_path)
from check_valid import is_valid
# 添加 BoT-SORT 到路径
if config.BOTSORT_ROOT.exists() and str(config.BOTSORT_ROOT) not in sys.path:
    sys.path.append(str(config.BOTSORT_ROOT))
NUM = 10

# 尝试导入 torch 和 YOLOX 相关模块
YOLOXPredictor = None
try:
    import torch
    import supervision as sv
    from yolox.exp import get_exp
    from tools.demo import Predictor as YOLOXPredictor
    from tracker.tracking_utils.timer import Timer
except Exception as e:
    print(f"Warning: YOLOX/Torch modules not fully available. Detail: {e}")
    print("Note: Camera calibration and basic functions will still work.")
    # 如果 torch/sv 导入失败，提供基本的替代
    try:
        import torch
        import supervision as sv
    except:
        pass

from soccer import HalfPitchConfiguration
from tools.annotation.pitch_annotation_tool import draw_pitch
from area import load_keypoints, run_annotation, imwritex
from tools import transform
from tools.transform import compute_transform_matrix

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# ==========================================
# 全局配置 (Moved to config.py)
# ==========================================

@dataclass
class CameraInfo:
    """
    相机信息数据类
    
    存储相机的完整视野信息，包括基于真实畸变边界计算的可见网格。
    """
    position: np.ndarray  # 相机位置 (球场坐标)
    left_boundary: np.ndarray  # 左边界点列表 [(x1,y1), (x2,y2), ...]，密集采样，捕捉畸变曲线
    right_boundary: np.ndarray  # 右边界点列表 [(x1,y1), (x2,y2), ...]，密集采样，捕捉畸变曲线
    orientation: float  # 相机朝向角度（度）
    angle_span: float  # 视野夹角（度）
    max_effective_dist: float = 5000.0  # 相机最大有效监看距离 (cm)
    visible_grid_cells: Optional[np.ndarray] = None  # 可见的网格单元 (N, 2) 格式: [[row, col], ...] 或 None
    grid_size: Optional[float] = None  # 网格尺寸 (cm)，如 50.0
    distortion_params: Optional[np.ndarray] = None  # 畸变参数 [k1, k2, p1, p2, k3, ...]，用于 MCS 质量评分


class CameraFOVCalculator:
    """
    相机视野计算器（畸变优化版）
    
    算法流程：
    1. 密集采样视频边界（100+点）→ 准确捕捉畸变曲线
    2. 图像纵向中心线 → 相机朝向
    3. 拟合左右边界曲线 → 准确的视野边界
    4. 三线综合 → 计算相机位置
    """
    
    def __init__(self, image_shape: Tuple[int, int], homography_matrix: np.ndarray,
                 distortion_params: Optional[np.ndarray] = None,
                 distortion_info: Optional[dict] = None,
                 pitch_bounds: Optional[Tuple[float, float, float, float]] = None,
                 num_boundary_samples: int = 100):
        """
        Args:
            image_shape: (height, width) 图像尺寸，例如 (1440, 2560)
            homography_matrix: 3x3 单应性变换矩阵 (图像坐标 -> 球场坐标)
            distortion_params: 畸变参数 (可选)
            distortion_info: 畸变信息字典，包含 center 等 (可选)
            pitch_bounds: 球场边界 (x_min, x_max, y_min, y_max)，用于边界检查
            num_boundary_samples: 边界采样点数（默认100，与可视化保持一致）
        """
        self.h, self.w = image_shape
        self.H = homography_matrix
        self.distortion_params = distortion_params
        self.distortion_info = distortion_info
        self.num_boundary_samples = num_boundary_samples
        
        # 球场边界（默认：足够大的范围）
        if pitch_bounds is None:
            self.pitch_bounds = (-1000, 20000, -1000, 20000)  # (x_min, x_max, y_min, y_max)
        else:
            self.pitch_bounds = pitch_bounds

    # TODO
    def calculate(self) -> CameraInfo:
        """
        计算相机视野信息（重构版）
        
        算法流程：
        1. 图像中心线 → 相机朝向
        2. 左右边界线 → 视野边界（考虑场地范围）
        3. 三线求交 → 相机位置（鲁棒方法）
        
        Returns:
            CameraInfo: 包含相机位置、左右边界、朝向和视野夹角的信息
        """
        # Step 1: 计算图像中心线和朝向
        center_line, orientation = self._calculate_center_line_and_orientation()
        
        # Step 2: 计算左右视野边界
        left_boundary = self._calculate_boundary_line_safe(side='left')
        right_boundary = self._calculate_boundary_line_safe(side='right')
        
        # Step 3: 计算相机位置（鲁棒方法）
        cam_pos = self._calculate_camera_position(center_line, left_boundary, right_boundary)
        
        # Step 4: 计算视野夹角
        angle_span = self._calculate_fov_angle(cam_pos, left_boundary, right_boundary)
        
        return CameraInfo(
            position=cam_pos,
            left_boundary=left_boundary,
            right_boundary=right_boundary,
            orientation=orientation,
            angle_span=angle_span,
            visible_grid_cells=None,  # 稍后计算
            grid_size=None,
            distortion_params=self.distortion_params  # 畸变参数，用于 MCS 质量评分
        )
    
    def get_pixel_points(self, pitch_points: np.ndarray) -> np.ndarray:
        """
        将球场坐标的网格中心点转换为视频像素坐标（含畸变校正）
        Args:
            pitch_points: (N, 2) 球场坐标的网格中心点
        Returns:
            (N, 2) 视频像素坐标
        """
        return transform.pitch_to_video_pixel(
            pitch_points,
            self.H,
            apply_distortion=(self.distortion_params is not None),
            distortion_params=self.distortion_params,
            distortion_info=self.distortion_info
        )

    def get_pitch_points(self, video_points: np.ndarray) -> np.ndarray:
        """
        将视频像素坐标转换为球场坐标（含去畸变）
        Args:
            video_points: (N, 2) 视频像素坐标
        Returns:
            (N, 2) 球场坐标
        """
        return self._project_points(video_points)
    
    def _project_points(self, img_points: np.ndarray) -> np.ndarray:
        """
        投影图像点到球场坐标（含去畸变）
        
        Args:
            img_points: (N, 2) 图像坐标点
            
        Returns:
            (N, 2) 球场坐标点
        """
        img_points = np.array(img_points, dtype=np.float32).reshape(-1, 1, 2)
        if img_points.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        # 去畸变
        if self.distortion_params is not None:
            from de_distortion import undistort_points
            image_center = self.distortion_info.get('center', (self.w / 2, self.h / 2))
            img_points = undistort_points(
                img_points,
                self.distortion_params,
                image_center,
                enable_k3=True
            ).reshape(-1, 1, 2)
        
        # 投影到球场坐标
        world_points = cv2.perspectiveTransform(img_points, self.H).reshape(-1, 2)
        
        # 坐标截断，防止异常值
        max_coord = 30000.0
        world_points = np.clip(world_points, -max_coord, max_coord)
        
        return world_points
    
    def _calculate_center_line_and_orientation(self) -> Tuple[np.ndarray, float]:
        """
        使用图像纵向中心线确定相机朝向
        
        Returns:
            center_line: (N, 2) 中心线上的点
            orientation: 朝向角度（度）
        """
        # 图像中心线上的多个点（从上到下）
        center_x = self.w / 2
        y_samples = np.linspace(self.h * 0.25, self.h * 0.75, 5)  # 采样5个点
        
        img_points = np.array([[center_x, y] for y in y_samples])
        center_line = self._project_points(img_points)
        
        # 计算朝向：从上到下的平均方向
        direction = center_line[-1] - center_line[0]
        orientation = np.degrees(np.arctan2(direction[1], direction[0]))
        
        return center_line, orientation
    
    # TODO
    def _calculate_boundary_line(self, side='left', output_samples: int = 100, dev = 0.0) -> np.ndarray:
        """
        计算视野边界线（优化采样 + 三次拟合）
        
        策略：
        1. 只采样视频中间区域（1/4 到 3/4 高度），避开上下边缘
        2. 转换为球场坐标并过滤异常点
        3. 使用三次多项式拟合，得到平滑曲线
        4. 从拟合曲线重新采样，返回平滑的边界点
        
        Args:
            side: 'left' 或 'right'
            output_samples: 输出的边界点数（默认100）
            dev: 边界线偏移量（像素）
        Returns:
            (N, 2) 边界线上的点（球场坐标，拟合后的平滑曲线）
        """
        from tools.transform import video_pixel_to_pitch
        
        # 1. 采样视频边界的中间区域（1/4 到 3/4 高度）
        x = dev if side == 'left' else (self.w - 1 - dev)
        y_start = self.h * 0.25
        y_end = self.h * 0.75
        y_coords = np.linspace(y_start, y_end, self.num_boundary_samples)
        
        img_points = np.column_stack([
            np.full(self.num_boundary_samples, x),
            y_coords
        ])
        
        # 2. 使用 transform 模块进行完整转换（含去畸变）
        boundary_points = video_pixel_to_pitch(
            img_points[check_valid.is_valid_img(img_points, self.get_pixel_points, self.get_pitch_points)],
            self.H,
            distortion_params=self.distortion_params,
            distortion_info=self.distortion_info,
            enable_k3=True
        )
        if boundary_points.size == 0:
            return np.array([]).reshape(0, 2)
        
        # 3. 过滤不合理的点
        x_min, x_max, y_min, y_max = self.pitch_bounds
        mask = (
            (boundary_points[:, 0] >= x_min) & 
            (boundary_points[:, 0] <= x_max) &
            (boundary_points[:, 1] >= y_min) & 
            (boundary_points[:, 1] <= y_max)
        )
        boundary_points_filtered = boundary_points[mask]
        boundary_points_filtered = boundary_points_filtered[check_valid.is_valid(
                                                                    boundary_points_filtered, 
                                                                    self.get_pixel_points, 
                                                                    self.get_pitch_points,
                                                                    self.w, self.h)]

        filtered_count = len(boundary_points) - len(boundary_points_filtered)
        if filtered_count > 0:
            print(f"[FOV] {side}: 采样中间区域[{y_start:.0f}-{y_end:.0f}], "
                  f"过滤 {filtered_count}/{len(boundary_points)} 个点, "
                  f"保留 {len(boundary_points_filtered)} 个")
        
        # 4. 如果有效点太少，返回空
        if len(boundary_points_filtered) < NUM:
            print(f"[FOV] 警告: {side} 边界有效点太少 ({len(boundary_points_filtered)}), 返回空边界")
            return np.array([]).reshape(0, 2)
        
        # 5. 三次多项式拟合（X对Y的函数）
        try:
            # 按 Y 坐标排序
            sorted_indices = np.argsort(boundary_points_filtered[:, 1])
            sorted_points = boundary_points_filtered[sorted_indices]
            
            # 拟合 X = f(Y) 的三次多项式
            poly_coeffs = np.polyfit(sorted_points[:, 1], sorted_points[:, 0], deg=3)
            
            # 6. 从拟合曲线重新采样
            y_min_fit = sorted_points[:, 1].min()
            y_max_fit = sorted_points[:, 1].max()
            y_resampled = np.linspace(y_min_fit, y_max_fit, output_samples)
            x_resampled = np.polyval(poly_coeffs, y_resampled)
            
            # 组合为边界点
            boundary_fitted = np.column_stack([x_resampled, y_resampled])
            
            print(f"[FOV] {side}: 三次拟合完成, 输出 {len(boundary_fitted)} 个点, "
                  f"Y范围 [{y_min_fit:.0f}, {y_max_fit:.0f}]")
            
            return boundary_fitted
            
        except Exception as e:
            print(f"[FOV] 警告: {side} 边界拟合失败 ({e}), 使用原始过滤点")
            return boundary_points_filtered
    
    def _calculate_boundary_line_safe(self, side='left', output_samples: int= 100) -> np.ndarray:
        """
        计算视野边界线（安全版本，增加异常点检查）
        
        在原有基础上增加了对异常点的检查和过滤，确保拟合的稳定性。
        """
        boundary_points = []
        tot = 2000
        repeat = 2000
        while repeat > 0:
            boundary_points = self._calculate_boundary_line(side, output_samples, dev = (tot - repeat))
            if len(boundary_points) > NUM:
                break
            repeat -= 1
        if len(boundary_points) <= NUM:
            print(f"[FOV] 警告: {side} 边界点过少 ({len(boundary_points)}), 返回空边界")
            return np.array([])
        return boundary_points

    def _calculate_camera_position(self, center_line: np.ndarray, 
                                   left_boundary: np.ndarray, 
                                   right_boundary: np.ndarray) -> np.ndarray:
        """
        计算相机位置（基于密集采样曲线的方法）
        
        策略（基于畸变曲线特征）：
        1. 提取左右边界的近端部分（靠近相机）
        2. 局部拟合近端切线，求交点
        3. 结合中心线延长，综合多个位置估计
        4. 取鲁棒结果（中位数）
        
        Args:
            center_line: 中心线点
            left_boundary: 左边界点（密集采样，可能包含曲线）
            right_boundary: 右边界点（密集采样，可能包含曲线）
            
        Returns:
            (2,) 相机位置
        """
        valid_positions = []
        
        # 策略1: 使用左右边界的近端点（假设相机在 Y 坐标较小处）
        left_near_idx = np.argmin(left_boundary[:, 1])
        right_near_idx = np.argmin(right_boundary[:, 1])
        
        # 取近端附近的点拟合局部切线（约10个点）
        n_fit = min(10, len(left_boundary) // 3)
        
        left_near_points = left_boundary[
            max(0, left_near_idx - n_fit//2) : min(len(left_boundary), left_near_idx + n_fit//2 + 1)
        ]
        right_near_points = right_boundary[
            max(0, right_near_idx - n_fit//2) : min(len(right_boundary), right_near_idx + n_fit//2 + 1)
        ]
        
        # 近端切线交点
        if len(left_near_points) >= 3 and len(right_near_points) >= 3:
            pos1 = self._fit_and_intersect_lines(left_near_points, right_near_points)
            if pos1 is not None and self._is_reasonable_position(pos1):
                valid_positions.append(pos1)
        
        # 策略2: 中心线与左边界近端交点
        if len(center_line) >= 2 and len(left_near_points) >= 3:
            pos2 = self._fit_and_intersect_lines(center_line, left_near_points)
            if pos2 is not None and self._is_reasonable_position(pos2):
                valid_positions.append(pos2)
        
        # 策略3: 中心线与右边界近端交点
        if len(center_line) >= 2 and len(right_near_points) >= 3:
            pos3 = self._fit_and_intersect_lines(center_line, right_near_points)
            if pos3 is not None and self._is_reasonable_position(pos3):
                valid_positions.append(pos3)
        
        # 策略4: 使用边界曲线整体拟合（多项式）
        if len(left_boundary) >= NUM and len(right_boundary) >= NUM:
            try:
                # 用多项式拟合边界曲线（捕捉畸变）
                left_poly = np.polyfit(left_boundary[:, 1], left_boundary[:, 0], deg=2)
                right_poly = np.polyfit(right_boundary[:, 1], right_boundary[:, 0], deg=2)
                
                # 在近端 Y 处计算曲线中心
                y_near = min(left_boundary[left_near_idx, 1], right_boundary[right_near_idx, 1])
                x_left = np.polyval(left_poly, y_near)
                x_right = np.polyval(right_poly, y_near)
                x_center = (x_left + x_right) / 2
                
                # 沿中心线方向延长
                if len(center_line) >= 2:
                    center_dir = center_line[0] - center_line[-1]
                    center_dir_norm = np.linalg.norm(center_dir)
                    if center_dir_norm > 1e-3:
                        center_dir = center_dir / center_dir_norm
                        # 延长距离
                        extension = 500  # cm
                        pos4 = np.array([x_center, y_near]) + center_dir * extension
                        
                        if self._is_reasonable_position(pos4):
                            valid_positions.append(pos4)
            except Exception as e:
                pass
        
        # 综合结果
        if len(valid_positions) > 0:
            # 取中位数（比均值更鲁棒）
            positions_array = np.array(valid_positions)
            cam_pos = np.median(positions_array, axis=0)
            return cam_pos
        else:
            # 后备方案：中心线延长
            if len(center_line) >= 2:
                center_dir = center_line[0] - center_line[-1]
                center_dir_norm = np.linalg.norm(center_dir)
                if center_dir_norm > 1e-3:
                    center_dir = center_dir / center_dir_norm
                    return center_line[0] + center_dir * 1000
            
            # 最后手段：边界均值
            return np.mean(np.vstack([left_boundary, right_boundary]), axis=0)
    
    def _fit_and_intersect_lines(self, line1: np.ndarray, line2: np.ndarray) -> Optional[np.ndarray]:
        """
        拟合两条线并计算交点
        
        Args:
            line1: (N, 2) 第一条线的点
            line2: (M, 2) 第二条线的点
            
        Returns:
            (2,) 交点，如果平行则返回 None
        """
        if len(line1) < 2 or len(line2) < 2:
            return None
        
        # 使用首尾点拟合直线（简化版，可以用最小二乘）
        p1, p2 = line1[0], line1[-1]
        p3, p4 = line2[0], line2[-1]
        
        return self._intersect_lines(p1, p2, p3, p4)
    
    def _is_reasonable_position(self, pos: np.ndarray, max_distance: float = 10000.0) -> bool:
        """
        检查相机位置是否合理
        
        Args:
            pos: 相机位置
            max_distance: 最大合理距离
            
        Returns:
            是否合理
        """
        # 检查是否在合理范围内
        x_min, x_max, y_min, y_max = self.pitch_bounds
        
        # 相机可以在场地外，但不能太远
        if abs(pos[0]) > max_distance or abs(pos[1]) > max_distance:
            return False
        
        return True
    
    def _calculate_fov_angle(self, cam_pos: np.ndarray,
                            left_boundary: np.ndarray,
                            right_boundary: np.ndarray) -> float:
        """
        计算视野夹角（基于曲线边界的方法）
        
        由于畸变，边界是曲线，我们取多个点计算夹角并取平均。
        
        Args:
            cam_pos: 相机位置
            left_boundary: 左边界点（密集采样）
            right_boundary: 右边界点（密集采样）
            
        Returns:
            视野夹角（度）
        """
        # 策略：在边界的不同位置采样多个点，计算夹角，取中位数
        angles = []
        
        # 在前、中、后三个位置采样
        sample_ratios = [0.25, 0.5, 0.75]
        
        for ratio in sample_ratios:
            left_idx = int(len(left_boundary) * ratio)
            right_idx = int(len(right_boundary) * ratio)
            
            left_idx = np.clip(left_idx, 0, len(left_boundary) - 1)
            right_idx = np.clip(right_idx, 0, len(right_boundary) - 1)
            
            left_point = left_boundary[left_idx]
            right_point = right_boundary[right_idx]
            
            vec_left = left_point - cam_pos
            vec_right = right_point - cam_pos
            
            angle = self._angle_between(vec_left, vec_right)
            if 0 < angle < 180:  # 合理范围
                angles.append(angle)
        
        # 取中位数（鲁棒）
        if len(angles) > 0:
            return float(np.median(angles))
        else:
            # 后备：使用中点
            left_point = left_boundary[len(left_boundary)//2]
            right_point = right_boundary[len(right_boundary)//2]
            vec_left = left_point - cam_pos
            vec_right = right_point - cam_pos
            return self._angle_between(vec_left, vec_right)

    def _intersect_lines(self, p1, p2, p3, p4):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4
        denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
        if denom == 0: return None
        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / denom
        return np.array([x1 + ua * (x2 - x1), y1 + ua * (y2 - y1)])

    def _angle_between(self, v1, v2):
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0: return 0.0
        unit_v1 = v1 / norm_v1
        unit_v2 = v2 / norm_v2
        dot_product = np.dot(unit_v1, unit_v2)
        return np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    
    def compute_visible_grids(self, camera_info: CameraInfo, 
                             pitch_length: float, pitch_width: float,
                             grid_size: float = 50.0) -> np.ndarray:
        """
        计算相机可见的网格单元（基于实际投影到画面内判断）
        
        算法：对每个球场网格中心点
        1. 将网格中心投影到视频像素坐标
        2. 检查投影点是否在视频画面内
        3. 如果在画面内，则该网格可见
        
        Args:
            camera_info: 相机信息（未使用，保留接口兼容性）
            pitch_length: 球场长度 (cm)
            pitch_width: 球场宽度 (cm)
            grid_size: 网格尺寸 (cm)，默认 50cm
            
        Returns:
            visible_cells: (N, 2) 数组，格式 [[row, col], ...]
        """
        from tools.transform import pitch_to_video_pixel
        from grid_system import PitchGridSystem
        
        print(f"[FOV] 使用实际投影算法判断可见性（忠实于画面内检测）")
        
        # 1. 创建统一的网格系统
        grid_system = PitchGridSystem(pitch_length, pitch_width, grid_size)
        
        # 2. 获取所有网格中心和索引
        grid_centers, grid_indices = grid_system.get_all_grid_centers()
        
        print(f"[FOV] 网格维度: {grid_system.num_rows}x{grid_system.num_cols}, "
              f"总计 {len(grid_indices)} 个")

        # 3. 将所有网格中心投影到视频像素坐标
        video_points = pitch_to_video_pixel(
            grid_centers,
            self.H,
            apply_distortion=(self.distortion_params is not None),
            distortion_params=self.distortion_params,
            distortion_info=self.distortion_info
        )

        import check_valid
        visible_mask = check_valid.is_valid(grid_centers, 
                                            self.get_pixel_points, 
                                            self.get_pitch_points, 
                                            self.w, self.h)
        
        # 5. 获取可见网格
        visible_cells = grid_indices[visible_mask]
        
        print(f"[FOV] 可见性判断结果: 最终可见 {len(visible_cells)}/{len(grid_indices)} 个网格")
        
        return visible_cells


class VideoManager:
    """视频管理类，负责管理视频输入"""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
    def read_video_generator(self) -> Generator[np.ndarray, None, None]:
        """
        视频帧生成器
        每次调用会重置到视频开头，然后逐帧yield
        """
        self.reset()
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def reset(self):
        """重置视频进度到开头"""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def get_fps(self) -> float:
        """
        获取视频帧率
        
        Returns:
            float: 视频的帧率（FPS, Frames Per Second）
        
        Example:
            >>> video_manager = VideoManager("path/to/video.mp4")
            >>> fps = video_manager.get_fps()
            >>> print(f"视频帧率: {fps} FPS")
        """
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps
    
    def get_resolution(self) -> Tuple[int, int]:
        """
        获取视频分辨率（高度和宽度）
        
        Returns:
            tuple[int, int]: (height, width) 视频的分辨率
                - height: 视频高度（像素）
                - width: 视频宽度（像素）
        
        Example:
            >>> video_manager = VideoManager("path/to/video.mp4")
            >>> height, width = video_manager.get_resolution()
            >>> print(f"视频分辨率: {width}x{height}")
        """
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (height, width)
        
    def release(self):
        self.cap.release()

    # 获取视频中指定帧索引的帧
    def get_frame_at_index(self, frame_index):
        cap = cv2.VideoCapture(self.video_path)
        
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {self.video_path}")
            
        # [核心代码] 设置读取位置为指定帧索引 (0-based)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            return frame
        else:
            raise ValueError(f"无法读取第 {frame_index} 帧 (可能超出范围)")

class Camera:
    """
    相机类
    
    封装了视频管理、标定、畸变校正、视场角计算等功能。
    
    核心特性：
    - 完整的相机标定流程（关键点、畸变校正、训练区域）
    - 自动畸变校正和坐标转换
    - 统一的标定文件管理
    - 多相机协同支持

    初始化参数：
        video_path: 视频文件路径
        pitch_config: 球场配置对象 (如 HalfQingYuanConfiguration)
        camera_id: 相机数值 ID (默认: 1)，用于多相机协同的网格分配
        camera_name: 相机名称 (用于标识和文件夹命名)，默认为 "camera{id}"
        mcs_name: 所属 MCS 系统名称 (可选)，用于文件路径管理
        enable_distortion_correction: 是否启用畸变校正 (默认: True)
        calibration_dir: 标定文件目录 (默认根据 mcs_name 自动确定)
        debug_mode: 调试模式 (默认: False)
            - True: 跳过读取已保存的畸变参数和 camera_info，强制重新计算，但仍然保存新结果
            - False: 优先读取已保存的标定数据
    
    标定流程（首次运行自动触发）：
        1. 关键点标定 → keypoints.json
        2. 横向线标定 → manual_horizontal.json (畸变校正)
        3. 纵向线标定 → manual_vertical.json (畸变校正)
        4. 计算畸变参数 → distortion_params.json
        5. 训练区域标定 (视频 + 俯视图) → train_area.json
    
    坐标转换（自动应用畸变校正）：
        像素坐标 → [去畸变] → 理想坐标 → [单应矩阵] → 球场坐标

    主要方法：
    1. 初始化与配置：
       - __init__: 加载视频、自动标定、计算FOV
       - set_coverage: 设置多相机协同覆盖区域
    
    2. 坐标变换：
       - pixel_to_pitch: 像素坐标 → 球场坐标 (自动去畸变)
       - physical_to_grid: 球场坐标 → 网格坐标
       - filter_points_in_coverage: 筛选覆盖区域内的点
    
    3. 检测管理：
       - init_detection: 初始化 YOLO/YOLOX 检测模块
       - detection: CameraDetection 对象，管理检测和追踪
    
    4. 训练区域：
       - train_area: TrainArea 对象，管理训练区域
       - apply_train_area: 应用训练区域限制到覆盖图
    
    5. 可视化：
       - get_fov_boundaries: 获取视野边界图
       - draw_fov_on_image: 在球场图上绘制视野
    
    6. 数据访问：
       - get_frame_generator: 视频帧生成器
       - get_homography: 获取单应矩阵
    
    文件结构（存储在 calibration_dir 中）：
        keypoints.json              - 球场关键点
        manual_horizontal.json      - 横向线（畸变校正用）
        manual_vertical.json        - 纵向线（畸变校正用）
        distortion_params.json      - 畸变参数
        train_area.json             - 训练区域
        *_annotated.png             - 可视化图片
    
    使用示例：
        # 基础用法（自动标定 + 畸变校正）
        camera = Camera(
            video_path="videos/test.mp4",
            pitch_config=HalfQingYuanConfiguration(),
            camera_id=1
        )
        
        # 坐标转换（自动去畸变）
        pitch_coords = camera.pixel_to_pitch(pixel_coords)
        
        # 使用已有标定文件
        camera = Camera(
            video_path="videos/test.mp4",
            pitch_config=config,
            camera_id=1,
            calibration_dir="path/to/calibration"
        )
    """
    
    def __init__(self, video_path: str, pitch_config: object, 
                 camera_id: int = 1,
                 camera_name: Optional[str] = None,
                 mcs_name: Optional[str] = None,
                 enable_distortion_correction: bool = True,
                 calibration_dir: Optional[str] = None,
                 debug_mode: bool = False
                 ):
        """
        Args:
            video_path: 视频文件路径
            pitch_config: 球场配置对象
            camera_id: 相机数值 ID（用于 assignment_map）
            camera_name: 相机名称（用于标识和文件夹命名）。如果为 None，使用 f"camera{camera_id}"
            mcs_name: 所属 MCS 系统名称。如果提供，优先在 store/{mcs_name}_mcs/{camera_name}/ 下查找标定文件
            enable_distortion_correction: 是否启用畸变校正（需要额外的横纵线标定）
            calibration_dir: 标定文件目录路径。如果提供，从该目录加载标定文件；
                           如果为 None，根据 mcs_name 自动确定目录：
                           - 有 mcs_name: store/{mcs_name}_mcs/{camera_name}/
                           - 无 mcs_name: store/{camera_name}/
            debug_mode: 调试模式。为 True 时，跳过读取已保存的畸变参数和 camera_info，
                       强制重新计算，但仍然保存新计算的结果供后续使用
        """
        self.numeric_id = camera_id
        self.name = camera_name if camera_name else f"camera{camera_id}"
        self.id = self.name  # 保持向后兼容（用于日志打印）
        self.mcs_name = mcs_name  # 所属 MCS 系统名称
        self.mcs = None  # MCS 对象引用（用于查询质量分数等）
        
        self.video_path = video_path
        self.config = pitch_config
        self.transform_type = 'homography'
        self.video_manager = VideoManager(video_path)
        self.debug_mode = debug_mode
        
        # 畸变校正相关
        self.enable_distortion_correction = enable_distortion_correction
        self.distortion_params: Optional[np.ndarray] = None
        self.distortion_info: Optional[dict] = None
        
        # 存储目录管理（支持多路径查找）
        if calibration_dir:
            # 手动指定的目录
            self.store_dir = Path(calibration_dir)
        elif mcs_name:
            # 优先使用 MCS 目录
            self.store_dir = Path("store") / f"{mcs_name}_mcs" / self.name
        else:
            # 默认单独目录
            self.store_dir = Path("store") / self.name
        
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置备用查找路径（用于加载标定文件）
        self._setup_search_paths()
        
        if debug_mode:
            print(f"[Camera {self.id}] 🔧 调试模式已启用：跳过读取已保存的标定数据，强制重新计算")
        
        # 1. 获取第一帧用于标定和初始化
        # gen = self.video_manager.read_video_generator()
        from distort import IDX
        try:
            self.first_frame = self.video_manager.get_frame_at_index(IDX)
        except ValueError:
            raise ValueError(f"视频为空: {self.video_path}")
            
        self.frame_shape = self.first_frame.shape[:2] # (h, w)
        
        # 2. 标定并计算坐标变换
        if self.enable_distortion_correction:
            self.homography_matrix = self._calibrate_with_distortion()
        else:
            self.homography_matrix = self._calibrate()
            print(f"[Camera {self.id}] 标定完成")
        
        if self.homography_matrix.shape != (3, 3):
            raise ValueError(f"单应性矩阵形状错误: 应为 (3, 3)，实际为 {self.homography_matrix.shape}")
        
        self.calibration_errors = self._compute_reprojection_error(
            self._calibration_keypoint_indices,
            self._calibration_image_points
        )
        px = self.calibration_errors['pixel_reprojection']
        pt = self.calibration_errors['pitch_projection']
        print(f"[Camera {self.id}] 像素重投影误差  — 平均: {px['mean']:.2f} px, "
              f"最大: {px['max']:.2f} px, 最小: {px['min']:.2f} px")
        print(f"[Camera {self.id}] 球场投影距离误差 — 平均: {pt['mean']:.2f} cm, "
              f"最大: {pt['max']:.2f} cm, 最小: {pt['min']:.2f} cm")
        
        # 3. 计算相机视野信息（支持畸变校正）
        # 定义球场合理范围（用于过滤异常点）
        margin_ratio = 0.5
        pitch_bounds = (
            -self.config.length * margin_ratio,  # x_min
            self.config.length * (1 + margin_ratio),  # x_max
            -self.config.width * margin_ratio,  # y_min
            self.config.width * (1 + margin_ratio)  # y_max
        )
        
        self.fov_calculator = CameraFOVCalculator(
            self.frame_shape, 
            self.homography_matrix,
            distortion_params=self.distortion_params,
            distortion_info=self.distortion_info,
            pitch_bounds=pitch_bounds,
            num_boundary_samples=100  # 与边界可视化保持一致
        )
        self.info = self.fov_calculator.calculate()
        
        # 4. 可视化视频边界线（调试用，需要在 self.info 创建后）
        self._visualize_video_boundaries()
        
        # 5. 计算可见网格单元（基于真实畸变边界）
        # grid_size 应该从外部传入（MCS 配置），但初始化时使用默认值
        # 实际使用的 grid_size 会在 set_coverage 时更新
        self.grid_size: float = 50.0  # 默认值，会在 set_coverage 时更新为 MCS 配置的值
        if hasattr(self.config, 'length') and hasattr(self.config, 'width'):
            visible_grids = self.fov_calculator.compute_visible_grids(
                self.info,
                pitch_length=self.config.length,
                pitch_width=self.config.width,
                grid_size=self.grid_size
            )
            # 更新 CameraInfo
            self.info.visible_grid_cells = visible_grids
            self.info.grid_size = self.grid_size
            
            # 6. 可视化可见网格（初始化后自动生成）
            self._visualize_visible_grids()
            
            # 7. 可视化可见网格到球场俯视图
            self._visualize_visible_grids_on_pitch()
            
            # 8. 可视化所有关键点投影回视频（验证标定准确性）
            self._visualize_keypoints_return()
        
        # 重置视频管理器以供后续使用
        self.video_manager.reset()

        # 4. （不再需要分配图）
        
        # 5. 初始化训练区域
        self.train_area = TrainArea(self)
        
        # 6. 如果训练区域为空，自动启动标定
        if not self.train_area.region_points:
            print(f"\n[Camera {self.id}] === 训练区域标定流程 ===")
            print(f"[Camera {self.id}] 步骤 1/2: 在视频画面中标注自定义点（可选）")
            print("提示：如果需要使用关键点以外的点（如中场线、禁区线等），请标注。")
            print("      如果只使用关键点，可以直接取消（按 's' 保存空组）。")
            self.train_area.define_from_video()
            
            print(f"\n[Camera {self.id}] 步骤 2/2: 在球场俯视图上选择训练区域边界")
            self.train_area.define_from_pitch_map()
        
        # 7. 保存相机信息到 JSON（调试模式下也保存）
        self._save_camera_info()
        
        # 8. 自动初始化检测类
        self.detection: Optional['CameraDetection'] = None
        try:
            self.init_detection()
        except Exception as e:
            print(f"[Camera {self.id}] ⚠️  检测模块初始化失败: {e}")
            print(f"[Camera {self.id}] 可以稍后手动调用 camera.init_detection() 初始化")

    def init_detection(self, 
                       yolo_model_path: Optional[str] = None, 
                       sam_checkpoint: Optional[str] = None,
                       model_type: Literal['yolox'] = 'yolox',
                       yolox_exp_file: Optional[str] = None,
                       with_reid: bool = True,
                       reid_config: Optional[str] = None,
                       reid_weights: Optional[str] = None,
                       reid_device: str = 'cuda',
                       reid_batch_size: int = 8):
        """
        初始化相机检测模块
        
        Args:
            yolo_model_path: YOLO 模型权重路径 (.pth/.tar). 
                             如果为 None，则使用默认全局配置。
            sam_checkpoint: [预留] SAM 模型权重路径
            model_type: 'yolox'
            yolox_exp_file: YOLOX 配置文件路径.
                            如果为 None，则使用默认全局配置。
            with_reid: 是否启用ReID特征提取（默认False）
            reid_config: FastReID配置文件路径（默认从config获取）
            reid_weights: FastReID权重路径（默认从config获取）
            reid_device: ReID模型运行设备（'cuda'或'cpu'）
            reid_batch_size: ReID批量处理大小（默认8）
        """
        # 设置默认路径
        if yolo_model_path is None:
            yolo_model_path = str(config.YOLOX_WEIGHTS)
        
        if yolox_exp_file is None:
            yolox_exp_file = str(config.YOLOX_EXP_FILE)

        self.detection = CameraDetection(
            self, 
            model_path=yolo_model_path, 
            model_type=model_type, 
            yolox_exp_file=yolox_exp_file,
            with_reid=with_reid,
            reid_config=reid_config,
            reid_weights=reid_weights,
            reid_device=reid_device,
            reid_batch_size=reid_batch_size
        )
        
        print(f"[Camera {self.id}] Detection initialized with {model_type} model: {yolo_model_path}")
        if with_reid:
            print(f"[Camera {self.id}] ReID feature extraction enabled")

    def _put_chinese_text(self, img, text, position, font_size=24, color=(255, 255, 255)):
        """
        在图像上绘制中文文本（使用PIL辅助，支持中文字符）
        
        Args:
            img: OpenCV图像 (BGR)
            text: 要绘制的文本（支持中文）
            position: (x, y) 文本位置
            font_size: 字体大小
            color: BGR颜色
            
        Returns:
            绘制后的图像
        """
        # OpenCV image (BGR) -> PIL image (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试加载字体
        try:
            import platform
            system = platform.system()
            if system == 'Windows':
                font = ImageFont.truetype("msyh.ttc", font_size)
            elif system == 'Darwin':
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
            else:
                font = ImageFont.truetype("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        # BGR -> RGB
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)
        
        # PIL image (RGB) -> OpenCV image (BGR)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    def _draw_pitch_boundary_safe(self, vis_frame, start, end, color=(0, 0, 255), thickness=2):
        """
        安全地绘制球场边界线（处理溢出和无效点）
        
        Args:
            vis_frame: 目标图像
            start: 起点球场坐标 [x, y]
            end: 终点球场坐标 [x, y]
            color: 线条颜色
            thickness: 线条粗细
        """
        from tools.transform import pitch_to_video_pixel
        
        # 1. 生成密集采样点
        t = np.linspace(0, 1, 50)
        line_points = np.array([
            start[0] + t * (end[0] - start[0]),
            start[1] + t * (end[1] - start[1])
        ]).T
        
        # 2. 投影到视频坐标
        video_line = pitch_to_video_pixel(
            line_points,
            self.homography_matrix,
            apply_distortion=self.enable_distortion_correction,
            distortion_params=self.distortion_params,
            distortion_info=self.distortion_info
        )
        
        h, w = vis_frame.shape[:2]
        valid_points = []

        from check_valid import is_valid
        valid_points = video_line[is_valid(line_points,
                                            self.fov_calculator.get_pixel_points,
                                            self.fov_calculator.get_pitch_points,
                                            w, h)]

        # 4. 绘制有效的线段
        if len(valid_points) >= 2:
            # 检查相邻点的距离，避免跨度太大
            filtered_points = [valid_points[0]]
            max_gap = min(w, h) * 0.3  # 最大允许间隔为图像尺寸的30%
            
            for i in range(1, len(valid_points)):
                prev = np.array(filtered_points[-1])
                curr = np.array(valid_points[i])
                dist = np.linalg.norm(curr - prev)
                
                if dist < max_gap:
                    filtered_points.append(valid_points[i])
                else:
                    # 距离太大，断开，开始新的线段
                    if len(filtered_points) >= 2:
                        pts = np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(vis_frame, [pts], isClosed=False, color=color, thickness=thickness)
                    filtered_points = [valid_points[i]]
            
            # 绘制最后一段
            if len(filtered_points) >= 2:
                pts = np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(vis_frame, [pts], isClosed=False, color=color, thickness=thickness)
    
    def _visualize_visible_grids(self):
        """
        可视化相机视野内所有可见网格（初始化后自动生成）
        
        用于验证：
        1. 可见网格的计算是否正确
        2. 标定（单应性+畸变）在整个视野范围内的准确性
        3. 网格中心点投影回视频后是否在合理位置
        """
        if self.first_frame is None or self.info.visible_grid_cells is None:
            print(f"[Camera {self.id}] 跳过可见网格可视化（无视频帧或无可见网格）")
            return
        
        vis_frame = self.first_frame.copy()
        
        # 1. 获取可见网格索引
        visible_cells = self.info.visible_grid_cells  # (N, 2) [[row, col], ...]
        
        if len(visible_cells) == 0:
            print(f"[Camera {self.id}] 警告: 没有可见网格")
            return
        
        print(f"[Camera {self.id}] 可视化 {len(visible_cells)} 个可见网格...")
        
        # 2. 转换为球场坐标（取网格中心）
        grid_size = self.grid_size
        pitch_points = np.stack([
            visible_cells[:, 1] * grid_size + grid_size/2,  # col -> x
            visible_cells[:, 0] * grid_size + grid_size/2   # row -> y
        ], axis=1)
        
        # 3. 反向投影回视频像素坐标
        from tools.transform import pitch_to_video_pixel
        
        try:
            video_points = pitch_to_video_pixel(
                pitch_points,
                self.homography_matrix,
                apply_distortion=self.enable_distortion_correction,
                distortion_params=self.distortion_params,
                distortion_info=self.distortion_info
            )
            
            # 4. 在视频帧上绘制可见网格（黄色点）
            in_bounds_count = 0
            h, w = vis_frame.shape[:2]
            
            import check_valid
            is_valid = check_valid.is_valid(pitch_points,
                                            self.fov_calculator.get_pixel_points,
                                            self.fov_calculator.get_pitch_points,
                                            w, h)

            for i, (pt, pitch_pt) in enumerate(zip(video_points, pitch_points)):
                # 检查是否为有效值（不是 NaN 或 Inf）
                if not (np.isfinite(pt[0]) and np.isfinite(pt[1])):
                    continue
                
                # 添加合理性检查：坐标不应过大
                if not (-w <= pt[0] <= 2*w and -h <= pt[1] <= 2*h):
                    continue
                
                x, y = int(pt[0]), int(pt[1])
                # 检查边界
                if 0 <= x < w and 0 <= y < h:
                    if is_valid[i]:
                        cv2.circle(vis_frame, (x, y), 4, (0, 255, 255), -1)  # 黄色点
                        in_bounds_count += 1
            
            print(f"[Camera {self.id}] 绘制了 {in_bounds_count}/{len(video_points)} 个点（其余在画面外）")
            
            # 5. 绘制场地主要边界线（红色）- 用于检查标定准确性
            pitch_w, pitch_h = self.config.length, self.config.width
            
            # 绘制四条主要边界线（使用安全绘制函数）
            self._draw_pitch_boundary_safe(vis_frame, [0, 0], [0, pitch_h], color=(0, 0, 255), thickness=3)           # 底线
            self._draw_pitch_boundary_safe(vis_frame, [0, 0], [pitch_w, 0], color=(0, 0, 255), thickness=3)           # 左边线
            self._draw_pitch_boundary_safe(vis_frame, [0, pitch_h], [pitch_w, pitch_h], color=(0, 0, 255), thickness=3)  # 右边线
            self._draw_pitch_boundary_safe(vis_frame, [pitch_w, 0], [pitch_w, pitch_h], color=(0, 0, 255), thickness=3)  # 中场线
            
            # 6. 添加图例（使用支持中文的绘制方法）
            vis_frame = self._put_chinese_text(vis_frame, f"Camera {self.id} - Visible Grids ({len(visible_cells)} grids)", 
                                                (50, 50), font_size=32, color=(0, 255, 255))
            vis_frame = self._put_chinese_text(vis_frame, f"Red Lines: Pitch Boundaries (Calibration Check)", 
                                                (50, 90), font_size=26, color=(0, 0, 255))
            vis_frame = self._put_chinese_text(vis_frame, f"Yellow Dots: Visible Grid Centers", 
                                                (50, 130), font_size=26, color=(0, 255, 255))
            
            # 7. 保存（支持中文路径）
            output_path = self.store_dir / f"{self.name}_visible_grids.jpg"
            imwritex(str(output_path), vis_frame)
            print(f"[Camera {self.id}] ✅ 可见网格可视化已保存: {output_path}")
            
        except Exception as e:
            print(f"[Camera {self.id}] ❌ 可见网格可视化失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _visualize_visible_grids_on_pitch(self, scale: float = 0.1, padding: int = 50):
        """
        将可见网格可视化到二维球场俯视图上（类似 MCS 分配图）
        
        用于验证：
        1. 可见网格在球场上的分布
        2. 单个相机的实际覆盖范围（基于画面内投影）
        
        Args:
            scale: 球场俯视图缩放比例
            padding: 球场俯视图边距
        """
        if self.info.visible_grid_cells is None or len(self.info.visible_grid_cells) == 0:
            print(f"[Camera {self.id}] 跳过球场俯视图可视化（无可见网格）")
            return
        
        print(f"[Camera {self.id}] 可视化可见网格到球场俯视图...")
        
        from tools.annotation.pitch_annotation_tool import draw_pitch
        from tools.transform import pitch_to_canvas_pixel
        import cv2
        
        # 1. 创建球场俯视图
        pitch_img = draw_pitch(config=self.config, scale=scale, padding=padding)
        
        # 2. 创建网格覆盖的 mask（与 MCS 分配图相同的逻辑）
        from grid_system import PitchGridSystem
        
        grid_system = PitchGridSystem(self.config.length, self.config.width, self.grid_size)
        rows, cols = grid_system.get_shape()
        
        # 创建只有本相机可见网格的 map
        visible_map = np.zeros((rows, cols), dtype=np.uint8)
        
        for row, col in self.info.visible_grid_cells:
            if 0 <= row < rows and 0 <= col < cols:
                visible_map[row, col] = 1
        
        # 3. 将 visible_map 转换为彩色图像并叠加到球场图上
        map_colored = np.zeros((rows, cols, 3), dtype=np.uint8)
        # 使用半透明蓝色表示可见区域
        map_colored[visible_map == 1] = (255, 100, 0)  # 蓝色
        
        # 调整大小到球场绘制区域
        target_w = int(self.config.length * scale)
        target_h = int(self.config.width * scale)
        
        map_resized = cv2.resize(map_colored, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
        
        # 创建 overlay
        overlay = np.zeros_like(pitch_img)
        overlay[padding:padding+target_h, padding:padding+target_w] = map_resized
        
        # 混合
        result = cv2.addWeighted(pitch_img, 0.7, overlay, 0.3, 0)
        
        # 4. 绘制相机位置和朝向
        if self.info is not None:
            cam_pos = self.info.position
            orientation = self.info.orientation
            
            # 相机位置
            cam_pos_canvas = pitch_to_canvas_pixel(
                cam_pos.reshape(1, -1), scale=scale, padding=padding
            )[0]
            
            if np.isfinite(cam_pos_canvas).all():
                cx, cy = int(cam_pos_canvas[0]), int(cam_pos_canvas[1])
                
                # 绘制相机位置（绿色圆点）
                cv2.circle(result, (cx, cy), 8, (0, 255, 0), -1)
                cv2.circle(result, (cx, cy), 10, (255, 255, 255), 2)
                
                # 绘制视野中线（绿色箭头）
                extend_length = min(self.config.length, self.config.width) * 0.3
                orientation_rad = np.deg2rad(orientation)
                direction = np.array([np.cos(orientation_rad), np.sin(orientation_rad)])
                center_line_end = cam_pos + extend_length * direction
                
                center_line_canvas = pitch_to_canvas_pixel(
                    center_line_end.reshape(1, -1), scale=scale, padding=padding
                )[0]
                
                if np.isfinite(center_line_canvas).all():
                    ex, ey = int(center_line_canvas[0]), int(center_line_canvas[1])
                    cv2.arrowedLine(result, (cx, cy), (ex, ey), (0, 255, 0), 3, tipLength=0.2)
        
        # 5. 添加图例和统计信息（使用支持中文的绘制方法）
        legend_x = padding + 10
        legend_y = padding + 20
        
        result = self._put_chinese_text(result, f"Camera {self.id} - Visible Grids on Pitch", 
                                         (legend_x, legend_y), font_size=26, color=(0, 0, 0))
        
        result = self._put_chinese_text(result, f"Visible Grids: {len(self.info.visible_grid_cells)}", 
                                         (legend_x, legend_y + 30), font_size=20, color=(255, 100, 0))
        
        result = self._put_chinese_text(result, f"Grid Size: {self.grid_size} cm", 
                                         (legend_x, legend_y + 60), font_size=20, color=(0, 0, 0))
        
        # 颜色图例
        cv2.rectangle(result, (legend_x, legend_y + 80), (legend_x + 30, legend_y + 100), 
                     (255, 100, 0), -1)
        result = self._put_chinese_text(result, "Visible Area", 
                                         (legend_x + 40, legend_y + 95), font_size=16, color=(0, 0, 0))
        
        cv2.circle(result, (legend_x + 15, legend_y + 125), 5, (0, 255, 0), -1)
        result = self._put_chinese_text(result, "Camera Position", 
                                         (legend_x + 40, legend_y + 130), font_size=16, color=(0, 0, 0))
        
        # 6. 保存（支持中文路径）
        output_path = self.store_dir / f"{self.name}_visible_grids_pitch.jpg"
        imwritex(str(output_path), result)
        print(f"[Camera {self.id}] ✅ 球场俯视图可视化已保存: {output_path}")
    
    def _visualize_keypoints_return(self):
        """
        可视化所有标定关键点投影回视频画面
        
        将球场配置中的所有关键点（vertices）投影回视频，
        用于验证标定的准确性——关键点应该准确对应视频中的实际标线交点。
        """
        if self.first_frame is None:
            print(f"[Camera {self.id}] 跳过关键点回投可视化（无视频帧）")
            return
        
        vis_frame = self.first_frame.copy()
        
        print(f"[Camera {self.id}] 可视化所有标定关键点回投...")
        
        # 1. 获取所有关键点的球场坐标
        pitch_vertices = np.array(self.config.vertices, dtype=np.float32)
        print(f"[Camera {self.id}] 配置中共有 {len(pitch_vertices)} 个关键点")
        
        # 2. 投影回视频像素坐标
        from tools.transform import pitch_to_video_pixel
        
        try:
            video_vertices = pitch_to_video_pixel(
                pitch_vertices,
                self.homography_matrix,
                apply_distortion=self.enable_distortion_correction,
                distortion_params=self.distortion_params,
                distortion_info=self.distortion_info
            )
            
            # 3. 在视频帧上绘制关键点
            in_bounds_count = 0
            h, w = vis_frame.shape[:2]
            
            import check_valid
            is_valid = check_valid.is_valid(pitch_vertices,
                                            self.fov_calculator.get_pixel_points,
                                            self.fov_calculator.get_pitch_points,
                                            w, h)

            for i, (pt, pitch_pt) in enumerate(zip(video_vertices, pitch_vertices)):
                # 检查是否为有效值（不是 NaN 或 Inf）
                if not (np.isfinite(pt[0]) and np.isfinite(pt[1])):
                    continue
                
                # 添加合理性检查：坐标不应过大（防止视野外的点因畸变误差出现在画面内）
                # 允许范围：[-画面尺寸, 2*画面尺寸]，超出此范围的点必然是视野外的误差
                if not (-w <= pt[0] <= 2*w and -h <= pt[1] <= 2*h):
                    continue
                
                x, y = int(pt[0]), int(pt[1])
                
                # 检查是否在画面内
                if 0 <= x < w and 0 <= y < h:

                    if is_valid[i]:
                        
                        # 绘制点（蓝色大圆点）
                        cv2.circle(vis_frame, (x, y), 8, (255, 0, 0), -1)  # 蓝色实心圆
                        cv2.circle(vis_frame, (x, y), 9, (255, 255, 255), 2)  # 白色边框
                        
                        # 标注序号（白底黑字）
                        label = str(i + 1)
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                        label_x = x + 12
                        label_y = y + 5
                        
                        # 绘制白色背景
                        cv2.rectangle(vis_frame, 
                                    (label_x - 2, label_y - label_size[1] - 2),
                                    (label_x + label_size[0] + 2, label_y + 2),
                                    (255, 255, 255), -1)
                        
                        # 绘制黑色文字
                        cv2.putText(vis_frame, label, (label_x, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        
                        in_bounds_count += 1
            
            print(f"[Camera {self.id}] 绘制了 {in_bounds_count}/{len(video_vertices)} 个关键点（其余在画面外）")
            
            # 4. 绘制场地边界线（红色，作为参考）
            pitch_w, pitch_h = self.config.length, self.config.width
            
            # 绘制四条边界线（使用安全绘制函数）
            self._draw_pitch_boundary_safe(vis_frame, [0, 0], [0, pitch_h], color=(0, 0, 255), thickness=2)
            self._draw_pitch_boundary_safe(vis_frame, [0, 0], [pitch_w, 0], color=(0, 0, 255), thickness=2)
            self._draw_pitch_boundary_safe(vis_frame, [0, pitch_h], [pitch_w, pitch_h], color=(0, 0, 255), thickness=2)
            self._draw_pitch_boundary_safe(vis_frame, [pitch_w, 0], [pitch_w, pitch_h], color=(0, 0, 255), thickness=2)
            
            # 5. 添加图例（使用支持中文的绘制方法）
            vis_frame = self._put_chinese_text(vis_frame, f"Camera {self.id} - Keypoints Return ({len(pitch_vertices)} points)", 
                                                (50, 50), font_size=32, color=(255, 0, 0))
            vis_frame = self._put_chinese_text(vis_frame, f"Blue Dots: Calibration Keypoints", 
                                                (50, 90), font_size=26, color=(255, 0, 0))
            vis_frame = self._put_chinese_text(vis_frame, f"Red Lines: Pitch Boundaries", 
                                                (50, 130), font_size=26, color=(0, 0, 255))
            
            # 6. 保存（支持中文路径）
            output_path = self.store_dir / f"{self.name}_return.jpg"
            imwritex(str(output_path), vis_frame)
            print(f"[Camera {self.id}] ✅ 关键点回投可视化已保存: {output_path}")
            
        except Exception as e:
            print(f"[Camera {self.id}] ❌ 关键点回投可视化失败: {e}")
            import traceback
            traceback.print_exc()

    def physical_to_grid(self, points: np.ndarray) -> np.ndarray:
        """
        将物理坐标转换为网格坐标
        
        Args:
            points: (N, 2) 物理坐标数组 [x, y]
            
        Returns:
            (N, 2) 网格坐标数组 [grid_x, grid_y] (int)
        """
        if self.grid_size <= 0:
            raise ValueError("Grid size not initialized")
            
        # 移除 clip，让调用者处理越界，或者在这里返回越界标记
        grid_points = np.floor(points / self.grid_size).astype(int)
        return grid_points

    def filter_points_in_coverage(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        筛选出位于场地范围内的点
        
        Args:
            points: (N, 2) 物理坐标数组 (cm)
            
        Returns:
            tuple:
                - points_in_coverage: (M, 2) 在场地范围内的点
                - mask: (N,) 布尔掩码，True 表示该点在区域内
        """
        if len(points) == 0:
            return np.empty((0, 2)), np.array([], dtype=bool)
        
        # 检查点是否在场地范围内
        pitch_length = self.config.length
        pitch_width = self.config.width
        
        x_valid = (points[:, 0] >= 0) & (points[:, 0] <= pitch_length)
        y_valid = (points[:, 1] >= 0) & (points[:, 1] <= pitch_width)
        mask = x_valid & y_valid
        
        return points[mask], mask
    
    def _visualize_video_boundaries(self, num_samples: int = 100, 
                                     scale: float = 0.1, padding: int = 50):
        """
        可视化相机视频的左右边界线到球场俯视图
        
        将视频画面的左右边缘从上到下采样，转换为球场坐标，
        绘制到球场俯视图上，保存到相机存储目录。
        
        Args:
            num_samples: 沿边界采样的点数（默认100，点越多线越平滑）
            scale: 球场俯视图缩放比例
            padding: 球场俯视图边距
        """
        print(f"\n[Camera {self.id}] === 可视化视频边界 ===")
        
        h, w = self.frame_shape  # (height, width)
        
        # 1. 定义视频画面的左右边界（像素坐标）
        # 只采样中间区域（1/4 到 3/4 高度），避开上下边缘
        y_start = h * 0.25
        y_end = h * 0.75
        y_coords = np.linspace(y_start, y_end, num_samples)
        
        left_boundary_video = np.column_stack([
            np.zeros(num_samples),  # x = 0
            y_coords
        ])
        
        right_boundary_video = np.column_stack([
            np.full(num_samples, w - 1),  # x = w - 1
            y_coords
        ])
        
        print(f"  视频分辨率: {w}x{h}")
        print(f"  采样区域: 高度 {y_start:.0f} 到 {y_end:.0f} (避开上下边缘)")
        print(f"  采样点数: {num_samples}")
        
        # 2. 转换为球场坐标
        from tools.transform import video_pixel_to_pitch
        
        left_boundary_pitch = video_pixel_to_pitch(
            left_boundary_video,
            self.homography_matrix,
            distortion_params=self.distortion_params if self.enable_distortion_correction else None,
            distortion_info=self.distortion_info if self.enable_distortion_correction else None,
            enable_k3=True
        )
        
        right_boundary_pitch = video_pixel_to_pitch(
            right_boundary_video,
            self.homography_matrix,
            distortion_params=self.distortion_params if self.enable_distortion_correction else None,
            distortion_info=self.distortion_info if self.enable_distortion_correction else None,
            enable_k3=True
        )
        
        print(f"  左边界球场坐标范围: X[{left_boundary_pitch[:, 0].min():.0f}, {left_boundary_pitch[:, 0].max():.0f}] "
              f"Y[{left_boundary_pitch[:, 1].min():.0f}, {left_boundary_pitch[:, 1].max():.0f}]")
        print(f"  右边界球场坐标范围: X[{right_boundary_pitch[:, 0].min():.0f}, {right_boundary_pitch[:, 0].max():.0f}] "
              f"Y[{right_boundary_pitch[:, 1].min():.0f}, {right_boundary_pitch[:, 1].max():.0f}]")
        
        # 2.5. 过滤不合理的点（超出场地范围太多的点）
        # 定义合理范围：场地范围 ± 余量（50%）
        margin_ratio = 0.5
        x_min = -self.config.length * margin_ratio
        x_max = self.config.length * (1 + margin_ratio)
        y_min = -self.config.width * margin_ratio
        y_max = self.config.width * (1 + margin_ratio)
        
        # 过滤左边界
        left_mask = (
            (left_boundary_pitch[:, 0] >= x_min) & 
            (left_boundary_pitch[:, 0] <= x_max) &
            (left_boundary_pitch[:, 1] >= y_min) & 
            (left_boundary_pitch[:, 1] <= y_max)
        )

        left_boundary_pitch_filtered = left_boundary_pitch[left_mask]
        
        # 过滤右边界
        right_mask = (
            (right_boundary_pitch[:, 0] >= x_min) & 
            (right_boundary_pitch[:, 0] <= x_max) &
            (right_boundary_pitch[:, 1] >= y_min) & 
            (right_boundary_pitch[:, 1] <= y_max)
        )

        right_boundary_pitch_filtered = right_boundary_pitch[right_mask]
        
        filtered_left = len(left_boundary_pitch) - len(left_boundary_pitch_filtered)
        filtered_right = len(right_boundary_pitch) - len(right_boundary_pitch_filtered)
        if filtered_left > 0 or filtered_right > 0:
            print(f"  过滤不合理点: 左边界 {filtered_left} 个, 右边界 {filtered_right} 个")
        
        # 3. 创建球场俯视图
        from tools.annotation.pitch_annotation_tool import draw_pitch
        
        pitch_img = draw_pitch(config=self.config, scale=scale, padding=padding)
        
        # 4. 转换为画布像素坐标并绘制
        from tools.transform import pitch_to_canvas_pixel
        
        left_boundary_canvas = pitch_to_canvas_pixel(
            left_boundary_pitch_filtered, scale=scale, padding=padding
        )
        
        right_boundary_canvas = pitch_to_canvas_pixel(
            right_boundary_pitch_filtered, scale=scale, padding=padding
        )
        
        # 绘制左边界（浅蓝色点 - 原始采样）
        # 过滤掉包含 NaN 或 Inf 的点
        valid_left_mask = np.isfinite(left_boundary_canvas).all(axis=1)
        for pt in left_boundary_canvas[valid_left_mask]:
            cv2.circle(pitch_img, tuple(pt.astype(int)), 2, (255, 180, 120), -1)  # 浅蓝色小点
        
        # 绘制右边界（浅红色点 - 原始采样）
        valid_right_mask = np.isfinite(right_boundary_canvas).all(axis=1)
        for pt in right_boundary_canvas[valid_right_mask]:
            cv2.circle(pitch_img, tuple(pt.astype(int)), 2, (120, 180, 255), -1)  # 浅红色小点
        
        # 4.5 三次拟合并绘制拟合曲线
        def fit_and_draw_curve(boundary_pitch_filtered, color, name):
            """对边界点进行三次拟合并绘制曲线"""
            if len(boundary_pitch_filtered) < 5:
                print(f"  警告: {name} 点太少，跳过拟合")
                return None
            
            try:
                # 按 Y 坐标排序
                sorted_indices = np.argsort(boundary_pitch_filtered[:, 1])
                sorted_points = boundary_pitch_filtered[sorted_indices]
                
                # 拟合 X = f(Y) 的三次多项式
                poly_coeffs = np.polyfit(sorted_points[:, 1], sorted_points[:, 0], deg=3)
                
                # 重新采样（100个点）
                y_min_fit = sorted_points[:, 1].min()
                y_max_fit = sorted_points[:, 1].max()
                y_resampled = np.linspace(y_min_fit, y_max_fit, 100)
                x_resampled = np.polyval(poly_coeffs, y_resampled)
                
                # 组合为边界点
                boundary_fitted = np.column_stack([x_resampled, y_resampled])
                
                # 转换为画布坐标
                boundary_fitted_canvas = pitch_to_canvas_pixel(
                    boundary_fitted, scale=scale, padding=padding
                )
                
                # 过滤掉包含 NaN 或 Inf 的点
                valid_mask = np.isfinite(boundary_fitted_canvas).all(axis=1)
                valid_canvas = boundary_fitted_canvas[valid_mask]
                
                # 绘制拟合曲线（使用 polylines）
                if len(valid_canvas) > 1:
                    pts = valid_canvas.astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(pitch_img, [pts], isClosed=False, color=color, thickness=3)
                
                print(f"  {name} 拟合: {len(sorted_points)} 点 -> 三次曲线 ({len(boundary_fitted)} 点)")
                return boundary_fitted
                
            except Exception as e:
                print(f"  警告: {name} 拟合失败 ({e})")
                return None
        
        left_fitted = fit_and_draw_curve(left_boundary_pitch_filtered, (255, 0, 0), "左边界")
        right_fitted = fit_and_draw_curve(right_boundary_pitch_filtered, (0, 0, 255), "右边界")
        
        # 4.6 绘制视野中线
        if self.info is not None:
            cam_pos = self.info.position
            orientation = self.info.orientation
            
            # 计算中线的终点（沿 orientation 方向延伸）
            # 延伸长度：使用球场对角线长度作为参考
            extend_length = np.sqrt(self.config.length**2 + self.config.width**2)
            
            # 方向向量
            orientation_rad = np.deg2rad(orientation)
            direction = np.array([np.cos(orientation_rad), np.sin(orientation_rad)])
            
            # 中线终点
            center_line_end = cam_pos + extend_length * direction
            
            # 转换为画布坐标
            center_line_pitch = np.array([cam_pos, center_line_end])
            center_line_canvas = pitch_to_canvas_pixel(
                center_line_pitch, scale=scale, padding=padding
            )
            
            # 绘制中线（绿色粗线）- 检查有效性
            if np.isfinite(center_line_canvas).all():
                cv2.line(pitch_img, 
                        tuple(center_line_canvas[0].astype(int)), 
                        tuple(center_line_canvas[1].astype(int)), 
                        (0, 255, 0), 2)
            
            # 绘制相机位置（绿色圆点）
            cam_pos_canvas = pitch_to_canvas_pixel(
                cam_pos.reshape(1, -1), scale=scale, padding=padding
            )[0]
            if np.isfinite(cam_pos_canvas).all():
                cv2.circle(pitch_img, tuple(cam_pos_canvas.astype(int)), 5, (0, 255, 0), -1)
            
            print(f"  相机位置: ({cam_pos[0]:.0f}, {cam_pos[1]:.0f}), 朝向: {orientation:.1f}°")
        
        # 添加图例（使用支持中文的绘制方法）
        legend_x = padding + 10
        legend_y = padding + 20
        pitch_img = self._put_chinese_text(pitch_img, f"Camera {self.id} - Video Boundaries (Fitted)", 
                                            (legend_x, legend_y), font_size=20, color=(0, 0, 0))
        
        # 原始采样点图例
        cv2.circle(pitch_img, (legend_x + 10, legend_y + 20), 3, (255, 180, 120), -1)
        pitch_img = self._put_chinese_text(pitch_img, "Left Samples", 
                                            (legend_x + 25, legend_y + 25), font_size=14, color=(0, 0, 0))
        
        cv2.circle(pitch_img, (legend_x + 10, legend_y + 40), 3, (120, 180, 255), -1)
        pitch_img = self._put_chinese_text(pitch_img, "Right Samples", 
                                            (legend_x + 25, legend_y + 45), font_size=14, color=(0, 0, 0))
        
        # 拟合曲线图例
        cv2.line(pitch_img, (legend_x, legend_y + 60), (legend_x + 40, legend_y + 60), 
                (255, 0, 0), 3)
        pitch_img = self._put_chinese_text(pitch_img, "Left Fitted (Cubic)", 
                                            (legend_x + 50, legend_y + 65), font_size=14, color=(255, 0, 0))
        
        cv2.line(pitch_img, (legend_x, legend_y + 80), (legend_x + 40, legend_y + 80), 
                (0, 0, 255), 3)
        pitch_img = self._put_chinese_text(pitch_img, "Right Fitted (Cubic)", 
                                            (legend_x + 50, legend_y + 85), font_size=14, color=(0, 0, 255))
        
        # 视野中线图例
        cv2.line(pitch_img, (legend_x, legend_y + 100), (legend_x + 40, legend_y + 100), 
                (0, 255, 0), 2)
        pitch_img = self._put_chinese_text(pitch_img, "Center Line (Orientation)", 
                                            (legend_x + 50, legend_y + 105), font_size=14, color=(0, 255, 0))
        
        # 相机位置图例
        cv2.circle(pitch_img, (legend_x + 20, legend_y + 120), 5, (0, 255, 0), -1)
        pitch_img = self._put_chinese_text(pitch_img, "Camera Position", 
                                            (legend_x + 30, legend_y + 125), font_size=14, color=(0, 255, 0))
        
        # 5. 保存图片（支持中文路径）
        output_path = self.store_dir / f"{self.name}_video_boundaries.png"
        imwritex(str(output_path), pitch_img)
        
        print(f"  ✅ 视频边界可视化已保存: {output_path}")
        print(f"[Camera {self.id}] === 边界可视化完成 ===\n")
    
    def _compute_reprojection_error(self, keypoint_indices, image_points):
        """
        计算所有标定关键点的重投影误差统计（像素 + 球场距离）

        像素重投影误差：球场坐标 → 逆单应性(+正向畸变) → 像素，与原始标注像素比较
        球场投影距离误差：原始标注像素 → 去畸变+单应性 → 球场坐标，与真值球场坐标比较

        Returns:
            dict: {
                'pixel_reprojection': {'mean': float, 'max': float, 'min': float},
                'pitch_projection':   {'mean': float, 'max': float, 'min': float},
            }
        """
        from tools.transform import pitch_to_video_pixel, video_pixel_to_pitch

        gt_pitch = np.array([self.config.vertices[idx] for idx in keypoint_indices],
                            dtype=np.float32)
        original_px = np.array(image_points, dtype=np.float32)

        reprojected_px = pitch_to_video_pixel(
            gt_pitch,
            self.homography_matrix,
            apply_distortion=self.enable_distortion_correction,
            distortion_params=self.distortion_params if self.enable_distortion_correction else None,
            distortion_info=self.distortion_info if self.enable_distortion_correction else None,
        )
        pixel_errors = np.linalg.norm(reprojected_px - original_px, axis=1)

        projected_pitch = video_pixel_to_pitch(
            original_px,
            self.homography_matrix,
            distortion_params=self.distortion_params if self.enable_distortion_correction else None,
            distortion_info=self.distortion_info if self.enable_distortion_correction else None,
            enable_k3=True,
        )
        pitch_errors = np.linalg.norm(projected_pitch - gt_pitch, axis=1)

        return {
            'pixel_reprojection': {
                'mean': float(np.mean(pixel_errors)),
                'max': float(np.max(pixel_errors)),
                'min': float(np.min(pixel_errors)),
            },
            'pitch_projection': {
                'mean': float(np.mean(pitch_errors)),
                'max': float(np.max(pitch_errors)),
                'min': float(np.min(pitch_errors)),
            },
        }

    def _save_camera_info(self):
        """
        保存相机信息到 JSON 文件
        
        将 CameraInfo 对象序列化并保存到 store_dir/camera_info.json
        包含相机位置、朝向、视野范围等关键信息
        """
        if self.info is None:
            print(f"[Camera {self.id}] Warning: No camera info to save.")
            return
        
        paths = self._get_calibration_paths()
        info_path = paths['camera_info']
        
        # 序列化 CameraInfo
        info_dict = {
            'camera_name': self.name,
            'camera_id': self.id,  # 字符串 ID（向后兼容）
            'numeric_id': self.numeric_id,  # 数值 ID
            'video_path': str(self.video_path),
            'frame_shape': list(self.frame_shape),
            'homography_matrix': self.homography_matrix.tolist() if self.homography_matrix is not None else None,
            'calibration_errors': self.calibration_errors if hasattr(self, 'calibration_errors') else None,
            'position': self.info.position.tolist() if self.info.position is not None else None,
            'left_boundary': self.info.left_boundary.tolist() if self.info.left_boundary is not None else None,
            'right_boundary': self.info.right_boundary.tolist() if self.info.right_boundary is not None else None,
            'orientation': float(self.info.orientation),
            'angle_span': float(self.info.angle_span),
            'max_effective_dist': float(self.info.max_effective_dist),
            'distortion_correction_enabled': self.enable_distortion_correction,
            'visible_grid_cells': self.info.visible_grid_cells.tolist() if self.info.visible_grid_cells is not None else None,
            'grid_size': float(self.info.grid_size) if self.info.grid_size is not None else None,
            'num_visible_grids': len(self.info.visible_grid_cells) if self.info.visible_grid_cells is not None else 0
        }
        
        # 保存到文件
        try:
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(info_dict, f, indent=2, ensure_ascii=False)
            print(f"[Camera {self.id}] 相机信息已保存: {info_path}")
        except Exception as e:
            print(f"[Camera {self.id}] Error saving camera info: {e}")
    
    def _setup_search_paths(self):
        """
        设置标定文件的搜索路径列表
        
        搜索优先级：
        1. 当前 store_dir（可能在 MCS 目录下）
        2. 如果有 mcs_name，尝试 store/{mcs_name}_mcs/{camera_name}/
        3. 独立目录 store/{camera_name}/
        """
        self.search_paths = [self.store_dir]
        
        # 添加备用路径
        if self.mcs_name:
            # 如果指定了 MCS，添加 MCS 目录
            mcs_path = Path("store") / f"{self.mcs_name}_mcs" / self.name
            if mcs_path != self.store_dir and mcs_path.exists():
                self.search_paths.append(mcs_path)
        
        # 添加独立目录
        standalone_path = Path("store") / self.name
        if standalone_path != self.store_dir and standalone_path.exists():
            self.search_paths.append(standalone_path)
        
        if len(self.search_paths) > 1:
            print(f"[Camera {self.id}] 搜索路径: {[str(p) for p in self.search_paths]}")
    
    def _find_calibration_file(self, filename: str) -> Optional[Path]:
        """
        在所有搜索路径中查找标定文件
        
        Args:
            filename: 文件名（如 "keypoints.json"）
        
        Returns:
            找到的文件路径，如果都不存在则返回 None
        """
        for search_path in self.search_paths:
            file_path = search_path / filename
            if file_path.exists():
                if search_path != self.store_dir:
                    print(f"[Camera {self.id}] 从备用路径加载 {filename}: {search_path}")
                return file_path
        return None
    
    def _get_calibration_paths(self) -> dict:
        """
        获取标定文件路径（支持多路径查找）
        
        Returns:
            dict: 包含所有标定文件路径的字典
        """
        filenames = {
            'keypoints': "keypoints.json",
            'manual_h': "manual_horizontal.json",
            'manual_v': "manual_vertical.json",
            'distortion': "distortion_params.json",
            'train_area': "train_area.json",
            'camera_info': "camera_info.json"
        }
        
        paths = {}
        for key, filename in filenames.items():
            # 查找文件，如果找不到则使用默认位置（store_dir）
            found_path = self._find_calibration_file(filename)
            if found_path:
                paths[key] = found_path
            else:
                # 未找到，使用主目录作为保存位置
                paths[key] = self.store_dir / filename
        
        return paths
    
    def _calibrate(self):
        """
        执行标定流程（不含畸变校正）
        
        从 store_dir 加载或创建关键点标定文件。
        
        Returns:
            根据 transform_type 返回：
            - 'homography': np.ndarray (3x3单应性矩阵)
            - 'tps': TPSTransform (TPS变换器对象)
        """
        paths = self._get_calibration_paths()
        json_path = paths['keypoints']
        
        indices = []
        points = []
        
        # 尝试加载现有标定数据
        if json_path.exists():
            print(f"[Camera {self.id}] 发现已有标定文件: {json_path}")
            try:
                indices, points = load_keypoints(str(json_path))
                print(f"[Camera {self.id}] 加载成功。")
            except Exception as e:
                print(f"[Camera {self.id}] 加载失败: {e}，将重新标定。")
        
        # 如果没有加载到有效数据，则进行手动标定
        if not indices:
            print(f"[Camera {self.id}] 开始手动标定...")
            indices, points = run_annotation(
                self.first_frame, 
                self.config, 
                str(json_path),
                frame_name=f"cam_{self.id}"
            )
        
        # TPS模式至少需要3个点，单应性矩阵至少需要4个点
        min_points = 4
        if not indices or len(indices) < min_points:
            raise ValueError(
                f"[Camera {self.id}] 关键点数量不足（{len(indices)}），"
                f"homography模式至少需要{min_points}个点。"
            )
        
        # 根据模式计算变换
        import inspect
        print(f"[Camera Debug] compute_transform_matrix function object: {compute_transform_matrix}")
        try:
            src = inspect.getsource(compute_transform_matrix)
            lines_5 = '\n'.join(src.splitlines()[:5])
            print(f"[Camera Debug] First 5 lines of source:\n{lines_5}")
        except Exception as e:
            print(f"[Camera Debug] Failed to get source: {e}")

        self._calibration_keypoint_indices = indices
        self._calibration_image_points = points
        return compute_transform_matrix(indices, points, self.config)
    
    def _calibrate_with_distortion(self):
        """
        执行带畸变校正的完整标定流程
        
        流程：
        1. 关键点标定
        2. 横纵线标定（用于畸变校正）
        3. 计算畸变参数
        4. 去畸变后计算单应矩阵
        
        Returns:
            np.ndarray: (3x3) 单应性矩阵
        """
        from de_distortion import (
            calibrate_distortion, 
            undistort_points,
            save_distortion_params,
            load_distortion_params
        )
        
        paths = self._get_calibration_paths()
        
        print(f"\n[Camera {self.id}] === 开始带畸变校正的标定流程 ===")
        print(f"[Camera {self.id}] 标定文件将保存至: {self.store_dir}")
        
        # 1. 关键点标定
        print(f"\n[Camera {self.id}] 步骤 1/3: 球场关键点标定")
        indices, points = [], []
        
        if paths['keypoints'].exists():
            print(f"[Camera {self.id}] 发现已有标定文件: {paths['keypoints']}")
            try:
                indices, points = load_keypoints(str(paths['keypoints']))
                print(f"[Camera {self.id}] 加载成功。")
            except Exception as e:
                print(f"[Camera {self.id}] 加载失败: {e}，将重新标定。")
        
        if not indices:
            print(f"[Camera {self.id}] 开始手动标定关键点...")
            indices, points = run_annotation(
                self.first_frame, 
                self.config, 
                str(paths['keypoints']),
                frame_name=f"cam_{self.id}"
            )
        
        if not indices or len(indices) < 4:
            raise ValueError(
                f"[Camera {self.id}] 关键点数量不足（{len(indices)}），至少需要4个点。"
            )
        
        # 2. 畸变校正
        print(f"\n[Camera {self.id}] 步骤 2/3: 畸变参数标定")
        
        if not self.debug_mode and paths['distortion'].exists():
            print(f"[Camera {self.id}] 发现已有畸变参数: {paths['distortion']}")
            try:
                self.distortion_params, self.distortion_info = load_distortion_params(
                    str(paths['distortion'])
                )
                print(f"[Camera {self.id}] 畸变参数加载成功。")
            except Exception as e:
                print(f"[Camera {self.id}] 加载失败: {e}，将重新标定。")
                self.distortion_params = None
        elif self.debug_mode:
            print(f"[Camera {self.id}] 🔧 调试模式：跳过加载已保存的畸变参数")
            self.distortion_params = None
        
        if self.distortion_params is None:
            print(f"[Camera {self.id}] 开始畸变标定（需要横纵线标注）...")
            self.distortion_params, self.distortion_info = calibrate_distortion(
                self.first_frame,
                keypoints_json=str(paths['keypoints']),
                manual_h_json=str(paths['manual_h']),
                manual_v_json=str(paths['manual_v']),
                pitch_config=self.config,  # 传入当前相机的球场配置
                auto_annotate=True
            )
            # 保存畸变参数（调试模式下也保存，方便后续使用）
            save_distortion_params(
                self.distortion_params, 
                self.distortion_info, 
                str(paths['distortion'])
            )
        
        # 3. 去畸变后计算单应矩阵
        print(f"\n[Camera {self.id}] 步骤 3/3: 计算单应矩阵（使用去畸变后的点）")
        
        # 去畸变关键点
        points_array = np.array(points, dtype=np.float32)
        image_center = self.distortion_info.get('center', 
                                                 (self.frame_shape[1]/2, self.frame_shape[0]/2))
        
        undistorted_points = undistort_points(
            points_array,
            self.distortion_params,
            image_center,
            enable_k3=True
        )
        
        # 使用去畸变后的点计算单应矩阵
        H = compute_transform_matrix(
            indices, 
            undistorted_points.tolist(), 
            self.config
        )
        
        self._calibration_keypoint_indices = indices
        self._calibration_image_points = points

        print(f"[Camera {self.id}] 畸变校正标定完成！")
        print(f"[Camera {self.id}] 畸变参数: k1={self.distortion_info['k1']:.4f}, "
              f"k2={self.distortion_info['k2']:.4f}, k3={self.distortion_info['k3']:.4f}")
        
        return H
    
    def get_frame_generator(self) -> Generator[np.ndarray, None, None]:
        """获取视频帧生成器"""
        gen = self.video_manager.read_video_generator()
        for frame in gen:
            yield frame
    
    def pixel_to_pitch(self, pixel_coords: np.ndarray) -> np.ndarray:
        """
        将像素坐标转换为球场坐标（统一接口）
        
        使用 transform 模块的 video_pixel_to_pitch() 方法。
        
        Args:
            pixel_coords: 像素坐标
                - (N, 2) 格式：[[u1,v1], [u2,v2], ...]
                - (N, 1, 2) 格式：OpenCV透视变换格式
        
        Returns:
            pitch_coords: (N, 2) 球场坐标数组 [[X1,Y1], [X2,Y2], ...]
        
        Example:
            >>> pixel_points = np.array([[100, 200], [300, 400]])
            >>> pitch_points = camera.pixel_to_pitch(pixel_points)
        """
        from tools.transform import video_pixel_to_pitch
        
        # 调用 transform 模块的统一接口
        pitch_coords = video_pixel_to_pitch(
            video_coords=pixel_coords,
            homography_matrix=self.homography_matrix,
            distortion_params=self.distortion_params if self.enable_distortion_correction else None,
            distortion_info=self.distortion_info if self.enable_distortion_correction else None,
            enable_k3=True
        )
        
        return pitch_coords
    
    def pitch_to_pixel(self, pitch_coords: np.ndarray, apply_distortion: bool = None) -> np.ndarray:
        """
        将球场坐标转换为视频像素坐标（反向投影）
        
        使用 transform 模块的 pitch_to_video_pixel() 方法。
        
        Args:
            pitch_coords: 球场坐标
                - (N, 2) 格式：[[X1,Y1], [X2,Y2], ...] (单位: cm)
            apply_distortion: 是否应用正向畸变
                - None (默认): 跟随相机的 enable_distortion_correction 设置
                - True: 强制应用畸变（用于在原始视频上绘制）
                - False: 不应用畸变（用于在去畸变后的视频上绘制）
        
        Returns:
            pixel_coords: (N, 2) 视频像素坐标数组 [[u1,v1], [u2,v2], ...]
        
        Example:
            >>> # 在原始视频上绘制中场线
            >>> pitch_line = np.array([[3300, 0], [3300, 4400]])
            >>> video_line = camera.pitch_to_pixel(pitch_line)  # 自动处理畸变
            >>> cv2.line(frame, tuple(video_line[0]), tuple(video_line[1]), (0,255,0), 2)
        
        Note:
            默认情况下，如果相机启用了畸变校正，会自动应用正向畸变，
            使得绘制的元素与原始视频画面对齐。
        """
        from tools.transform import pitch_to_video_pixel
        
        # 如果未指定，跟随相机设置
        if apply_distortion is None:
            apply_distortion = self.enable_distortion_correction
        
        # 调用 transform 模块的反向转换
        pixel_coords = pitch_to_video_pixel(
            pitch_coords=pitch_coords,
            homography_matrix=self.homography_matrix,
            apply_distortion=apply_distortion,
            distortion_params=self.distortion_params if apply_distortion else None,
            distortion_info=self.distortion_info if apply_distortion else None,
            enable_k3=True
        )
        
        return pixel_coords
    
    def get_homography(self) -> np.ndarray:
        """
        获取单应性矩阵
        """
        return self.homography_matrix
    
    def get_transform_type(self) -> str:
        """获取当前使用的变换类型"""
        return self.transform_type

    def draw_fov_on_image(self, pitch_img: np.ndarray, padding: int = 50, scale: float = 0.1, 
                          color: Tuple[int, int, int] = (0, 255, 255), line_thickness: int = 3) -> np.ndarray:
        """
        在给定的球场图上绘制当前相机的视野边界
        
        使用 transform 模块的 pitch_to_canvas_pixel() 进行坐标转换。
    
        Args:
            pitch_img: 球场图像 (numpy数组)
                padding: 图像边距 (需与生成pitch_img时的参数一致)
                scale: 缩放比例 (需与生成pitch_img时的参数一致)
            color: 边界线颜色 (BGR格式)
            line_thickness: 线条粗细
        
        Returns:
            np.ndarray: 绘制了边界线的球场图像
        """
        if self.info is None:
            return pitch_img

        from tools.transform import pitch_to_canvas_pixel

        # 复制图像，避免修改原图
        result_img = pitch_img.copy()

        # 绘制左边界：左中点 -> 左下角
        left_boundary_canvas = pitch_to_canvas_pixel(
            self.info.left_boundary, scale=scale, padding=padding
        )
        cv2.polylines(result_img, [left_boundary_canvas], isClosed=False,
                      color=color, thickness=line_thickness, lineType=cv2.LINE_AA)

        # 绘制右边界：右中点 -> 右下角
        right_boundary_canvas = pitch_to_canvas_pixel(
            self.info.right_boundary, scale=scale, padding=padding
        )
        cv2.polylines(result_img, [right_boundary_canvas], isClosed=False,
                      color=color, thickness=line_thickness, lineType=cv2.LINE_AA)

        # 绘制相机位置标记
        cam_canvas = pitch_to_canvas_pixel(
            self.info.position.reshape(1, -1), scale=scale, padding=padding
        )[0]
        cam_pix = tuple(cam_canvas)
        cv2.circle(result_img, cam_pix, radius=8, color=color, thickness=-1)
        cv2.circle(result_img, cam_pix, radius=8, color=(255, 255, 255), thickness=2)

        return result_img

    def get_fov_boundaries(self, padding: int = 50, scale: float = 0.1) -> np.ndarray:
        """
        获取绘制了当前相机视野边界的球场俯视图
        
        Args:
            padding: 图像边距
            scale: 缩放比例
            
        Returns:
            np.ndarray: 包含视野边界的球场图像
        """
        # 1. 生成基础球场图
        pitch_canvas = draw_pitch(config=self.config, scale=scale, padding=padding)
        
        # 2. 绘制视野边界
        return self.draw_fov_on_image(pitch_canvas, padding=padding, scale=scale)


# ==========================================
# 多模态检测数据结构
# ==========================================

def _sigmoid_confidence(confidence: float, midpoint: float = 0.5, steepness: float = 10.0) -> float:
    """
    对置信度进行 Sigmoid 处理，用于计算置信度权重
    
    使用MCS.MCSScoreModel进行计算
    
    Args:
        confidence: 原始置信度 [0, 1]
        midpoint: 半衰点，默认 0.5（此时输出为 0.5）
        steepness: 陡峭度，默认 10.0
                   - confidence=0.8 时输出约 0.95（趋近于1）
                   - confidence=0.2 时输出约 0.05（趋近于0）
    
    Returns:
        sigmoid处理后的置信度权重 [0, 1]
    """
    # 使用MCSScoreModel（如果可用）
    try:
        from MCS import MCSScoreModel
        return MCSScoreModel.sigmoid_confidence_weight(confidence, midpoint, steepness)
    except ImportError:
        # 如果无法导入MCSScoreModel，使用本地实现
        import math
        return 1.0 / (1.0 + math.exp(-steepness * (confidence - midpoint)))


@dataclass
class Detection:
    """
    多模态检测对象 - 统一数据结构
    
    用于封装单个检测的所有信息，支持：
    - 软边界过滤
    - 多模态特征（ReID、颜色）
    - 位置融合
    - 全局追踪
    """
    # ========== 必需字段（无默认值）==========
    camera_id: int                          # 来源相机ID
    bbox: np.ndarray                        # 图像坐标 [x1, y1, x2, y2] (像素)
    confidence: float                       # YOLO检测置信度 [0, 1]
    pitch_position: np.ndarray              # 球场坐标 (cm) - 在detect时计算并填入
    pixel_position: np.ndarray              # 像素坐标（脚部中心点）- 在detect时计算并填入
    
    # ========== 可选字段（有默认值）==========
    class_id: int = 0                       # 类别ID（0=person，用于多类别检测）
    grid_position: Tuple[int, int] = (0, 0) # 网格坐标 (row, col) - 在__post_init__中自动计算
    
    # ========== 质量评分信息 ==========
    quality_score: float = 0.0              # 当前网格的质量分数 (Combined Score)
    confidence_weight: float = 0.0          # Sigmoid处理后的置信度权重 [0, 1]
    final_score: float = 0.0                # 最终检测得分 = quality_score × confidence_weight
    
    # ========== 多模态特征 ==========
    crop_image: Optional[np.ndarray] = None       # crop图像 [H, W, 3]（可选保存，用于ReID提取）
    reid_feature: Optional[np.ndarray] = None     # ReID特征向量 (512,)
    color_histogram: Optional[np.ndarray] = None  # HSV颜色直方图 (436,)
    dominant_color: Optional[str] = None          # 主导颜色 'white'/'red'/'green'/'blue'/'black'
    jersey_number: Optional[int] = None           # 球衣号码 (OCR识别)
    
    # ========== 状态标记 ==========
    is_occluded: bool = False               # 是否遮挡（confidence < 0.5）
    frame_id: int = 0                       # 对应的帧号
    
    # ========== 场地配置 ==========
    grid_size: float = 50.0                # 网格尺寸 (cm)，必须与网格系统保持一致
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保 numpy 数组类型
        if not isinstance(self.bbox, np.ndarray):
            self.bbox = np.array(self.bbox)
        if not isinstance(self.pitch_position, np.ndarray):
            self.pitch_position = np.array(self.pitch_position)
        if not isinstance(self.pixel_position, np.ndarray):
            self.pixel_position = np.array(self.pixel_position)
        
        # 自动计算网格坐标
        # 严格遵守 PitchGridSystem.physical_to_index 的计算方式
        if self.grid_position == (0, 0) and isinstance(self.pitch_position, np.ndarray):
            # 使用与 PitchGridSystem.physical_to_index 相同的计算逻辑
            # physical_to_index(point) -> (row, col)
            #   col = int(point[0] / grid_size)
            #   row = int(point[1] / grid_size)
            col = int(self.pitch_position[0] / self.grid_size)
            row = int(self.pitch_position[1] / self.grid_size)
            self.grid_position = (row, col)
        
        # 自动计算置信度权重（Sigmoid处理）
        if self.confidence_weight == 0.0:
            self.confidence_weight = _sigmoid_confidence(self.confidence)
        
        # 自动计算最终分数 = 网格质量分数 × Sigmoid(置信度)
        if self.final_score == 0.0:
            self.final_score = self.quality_score * self.confidence_weight
        
        # 注意：is_occluded 通过 IoU 计算，在所有检测对象创建后由 _compute_occlusions 方法设置
    
    def to_dict(self) -> dict:
        """转换为字典格式（用于序列化或兼容旧接口）"""
        return {
            'camera_id': str(self.camera_id) if isinstance(self.camera_id, str) else int(self.camera_id),  # 支持字符串或整数
            'bbox': self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else self.bbox,
            'confidence': float(self.confidence),  # 确保转换为Python float
            'pitch_position': self.pitch_position.tolist() if isinstance(self.pitch_position, np.ndarray) else self.pitch_position,
            'pixel_position': self.pixel_position.tolist() if isinstance(self.pixel_position, np.ndarray) else None,
            'grid_position': [int(self.grid_position[0]), int(self.grid_position[1])],  # 转换为list[int]
            'quality_score': float(self.quality_score),  # 确保转换为Python float
            'confidence_weight': float(self.confidence_weight),  # 确保转换为Python float
            'final_score': float(self.final_score),  # 确保转换为Python float
            'reid_feature': self.reid_feature.tolist() if self.reid_feature is not None and isinstance(self.reid_feature, np.ndarray) else None,
            'dominant_color': self.dominant_color,
            'is_occluded': bool(self.is_occluded),  # 确保转换为Python bool
            'frame_id': int(self.frame_id),  # 确保转换为Python int
        }
    
    @classmethod
    def from_dict(cls, data: dict, grid_size: float = 50.0) -> 'Detection':
        """
        从字典格式重建Detection对象（用于反序列化）
        
        Args:
            data: 字典数据（由 to_dict() 生成）
            grid_size: 网格尺寸（如果字典中没有提供）
            
        Returns:
            Detection: 重建的Detection对象
        """
        # 转换必需的numpy数组字段
        bbox = np.array(data['bbox']) if not isinstance(data['bbox'], np.ndarray) else data['bbox']
        pitch_position = np.array(data['pitch_position']) if not isinstance(data['pitch_position'], np.ndarray) else data['pitch_position']
        pixel_position = np.array(data['pixel_position']) if data.get('pixel_position') is not None else np.array([0, 0])
        
        # 转换可选的numpy数组字段
        reid_feature = np.array(data['reid_feature']) if data.get('reid_feature') is not None else None
        color_histogram = np.array(data['color_histogram']) if data.get('color_histogram') is not None else None
        
        # 创建Detection对象
        return cls(
            camera_id=data['camera_id'],
            bbox=bbox,
            confidence=float(data['confidence']),
            pitch_position=pitch_position,
            pixel_position=pixel_position,
            class_id=int(data.get('class_id', 0)),
            grid_position=tuple(data.get('grid_position', [0, 0])),
            quality_score=float(data.get('quality_score', 0.0)),
            confidence_weight=float(data.get('confidence_weight', 0.0)),
            final_score=float(data.get('final_score', 0.0)),
            crop_image=None,  # crop_image通常不保存到JSON
            reid_feature=reid_feature,
            color_histogram=color_histogram,
            dominant_color=data.get('dominant_color'),
            jersey_number=data.get('jersey_number'),
            is_occluded=bool(data.get('is_occluded', False)),
            frame_id=int(data.get('frame_id', 0)),
            grid_size=float(data.get('grid_size', grid_size))
        )

# TODO：批处理功能似乎有问题
class ColorDetector:
    """
    基于 API 的球衣颜色检测器
    
    功能：
    - 调用多模态大模型 API 识别球衣颜色
    - 支持单帧批量检测
    - 自动管理临时文件
    
    颜色类别：white, red, green, blue, black, unknown
    """
    
    def __init__(self, api_key: str = None):
        """
        初始化颜色检测器
        
        Args:
            api_key: API密钥，如果为None则使用默认密钥
        """
        self.api_key = api_key if api_key else os.getenv('DASHSCOPE_API_KEY', '')
        self.temp_dir = Path("output/color_detection_temp")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[ColorDetector] Initialized with temp dir: {self.temp_dir}")
    
    def detect_frame_colors(self, frame: np.ndarray, detections: list, 
                           use_batch: bool = True, debug: bool = False) -> list:
        """
        对一帧中的所有检测进行颜色分类
        
        Args:
            frame: 原始图像 (BGR格式)
            detections: Detection 对象列表
            use_batch: 是否使用批量API（推荐True，速度更快）
            debug: 是否输出调试信息
        
        Returns:
            color_results: 颜色标签列表 ['red', 'white', 'blue', ...]，
                          与 detections 列表顺序对应
        """
        if not detections:
            return []
        
        if debug:
            print(f"[ColorDetector] Processing {len(detections)} detections...")
        
        # 1. 裁剪并保存检测框图像
        crop_paths = []
        crop_info = []  # 记录每个裁剪对应的检测索引
        
        for i, det in enumerate(detections):
            # 获取检测框坐标
            x1, y1, x2, y2 = det.bbox.astype(int)
            
            # 边界检查
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                crop_info.append(None)
                continue
            
            # 裁剪检测框
            bbox_crop = frame[y1:y2, x1:x2]
            
            # 保存临时文件
            crop_filename = f"det_{i:04d}.jpg"
            crop_path = self.temp_dir / crop_filename
            cv2.imwrite(str(crop_path), bbox_crop)
            
            crop_paths.append(crop_path)
            crop_info.append(i)
        
        if not crop_paths:
            return ['unknown'] * len(detections)
        
        # 2. 调用 API 检测颜色
        if use_batch:
            # 批量检测
            if debug:
                print(f"[ColorDetector] Using batch API for {len(crop_paths)} crops...")
            
            from ocr import batch_detect_jersey_colors
            color_map = batch_detect_jersey_colors(crop_paths, api_key=self.api_key)
            
            # 映射结果
            color_results = []
            for i, det in enumerate(detections):
                if crop_info[i] is None:
                    color_results.append('unknown')
                else:
                    crop_filename = f"det_{i:04d}.jpg"
                    color = color_map.get(crop_filename, 'unknown')
                    color_results.append(color)
        else:
            # 单个检测（实时调用）
            print(f"[ColorDetector] Using real-time API for {len(crop_paths)} crops...")
            print(f"[ColorDetector] ⚠️  This may take ~{len(crop_paths) * 2}s (2s per detection)")
            
            from ocr import detect_jersey_color
            color_results = []
            
            for i, det in enumerate(detections):
                if crop_info[i] is None:
                    color_results.append('unknown')
                else:
                    # 显示进度
                    print(f"[ColorDetector] Processing {i+1}/{len(crop_paths)}...", end='\r')
                    
                    crop_path = self.temp_dir / f"det_{i:04d}.jpg"
                    color = detect_jersey_color(crop_path, api_key=self.api_key)
                    color_results.append(color)
                    
                    if debug:
                        print(f"[ColorDetector] Detection {i+1}/{len(crop_paths)}: {color}        ")
        
        # 3. 清理临时文件（可选）
        if not debug:
            self._cleanup_temp_files()
        
        if debug:
            print(f"[ColorDetector] Color detection completed!")
            # 统计颜色分布
            from collections import Counter
            color_counts = Counter(color_results)
            print(f"[ColorDetector] Color distribution: {dict(color_counts)}")
        
        return color_results
    
    def _cleanup_temp_files(self):
        """清理临时文件"""
        try:
            for file in self.temp_dir.glob("det_*.jpg"):
                file.unlink()
        except Exception as e:
            print(f"[ColorDetector] Warning: Failed to cleanup temp files: {e}")
    
    def detect_single_crop(self, bbox_crop: np.ndarray, 
                          save_path: Path = None) -> str:
        """
        检测单个裁剪框的颜色（便捷方法）
        
        Args:
            bbox_crop: 裁剪后的检测框图像 (BGR格式)
            save_path: 保存路径，如果为None则使用临时路径
        
        Returns:
            color_label: 'white'/'red'/'green'/'blue'/'black'/'unknown'
        """
        from ocr import detect_jersey_color
        
        # 保存图像
        if save_path is None:
            save_path = self.temp_dir / "temp_single.jpg"
        
        cv2.imwrite(str(save_path), bbox_crop)
        
        # 检测颜色
        color = detect_jersey_color(save_path, api_key=self.api_key)
        
        # 清理
        if save_path.name == "temp_single.jpg":
            save_path.unlink()
        
        return color


class CameraDetection:
    """
    相机检测管理类
    
    封装单个相机的检测、过滤、可视化功能。
    管理 YOLO 模型和检测结果。

    给追踪器筛选所需的检测目标
    """
    
    def __init__(self, camera: Camera, model_path: str = None, 
                 model_type: Literal['yolox'] = 'yolox',
                 yolox_exp_file: Optional[str] = None,
                 with_reid: bool = True,
                 reid_config: str = None,
                 reid_weights: str = None,
                 reid_device: str = 'cuda',
                 reid_batch_size: int = 8):
        """
        初始化相机的检测管理器
        
        Args:
            camera: Camera 对象
            model_path: YOLO 模型权重路径 (对于 YOLOX 是 .pth/.tar). 默认为 config 中的配置
            model_type: 'yolox'
            yolox_exp_file: YOLOX 的 experiment 配置文件路径
            with_reid: 是否启用ReID特征提取（默认False）
            reid_config: FastReID配置文件路径（默认从config获取）
            reid_weights: FastReID权重路径（默认从config获取）
            reid_device: ReID模型运行设备（'cuda'或'cpu'）
            reid_batch_size: ReID批量处理大小（默认8）
        """
        self.camera = camera
        self.model_path = model_path if model_path else str(config.YOLOX_WEIGHTS)
        self.model_type = model_type
        
        self.yolox_predictor = None
        
        # 初始化 YOLOX 模型
        if YOLOXPredictor is None:
            raise ImportError("YOLOX modules are not available.")
        if not yolox_exp_file:
            raise ValueError("YOLOX requires 'yolox_exp_file' path.")
        
        try:
            exp = get_exp(yolox_exp_file, None)
            model = exp.get_model()
            
            # 加载权重
            ckpt = torch.load(model_path, map_location="cpu")
            model.load_state_dict(ckpt["model"])
            model.cuda().eval()
            model.half() # 关键修复：转换为 FP16 以匹配 Predictor 的设置
            
            # Fuse model (optional)
            # from yolox.utils import fuse_model
            # model = fuse_model(model)
            
            self.yolox_predictor = YOLOXPredictor(
                model=model,
                exp=exp,
                device="cuda",
                fp16=True
            )
            print(f"[Camera {self.camera.id}] YOLOX initialized: {model_path}")
        except Exception as e:
            print(f"[Camera {self.camera.id}] Error initializing YOLOX: {e}")
        
        # ========== 初始化 FastReID 模型（学习BoT-SORT的方式）==========
        self.with_reid = with_reid
        self.reid_encoder = None
        
        if with_reid:
            try:
                # 添加BoT-SORT路径到sys.path（修复路径问题）
                import sys
                from pathlib import Path
                
                # 确定BoT-SORT根目录路径
                botsort_path = Path(__file__).parent / 'BoT-SORT'
                
                # 验证路径存在
                if not botsort_path.exists():
                    raise FileNotFoundError(f"BoT-SORT directory not found at {botsort_path}")
                
                # 验证fast_reid_interfece.py存在
                fast_reid_interface_file = botsort_path / 'fast_reid' / 'fast_reid_interfece.py'
                if not fast_reid_interface_file.exists():
                    raise FileNotFoundError(f"FastReID interface not found at {fast_reid_interface_file}")
                
                # 添加到sys.path
                if str(botsort_path) not in sys.path:
                    sys.path.insert(0, str(botsort_path))
                    print(f"[Camera {self.camera.id}] Added BoT-SORT to sys.path: {botsort_path}")
                
                # 导入FastReID接口（从BoT-SORT目录导入）
                # 导入路径：BoT-SORT/fast_reid/fast_reid_interfece.py
                from fast_reid.fast_reid_interfece import FastReIDInterface
                
                # 使用默认配置或用户提供的配置
                reid_config = reid_config or str(config.FAST_REID_CONFIG)
                reid_weights = reid_weights or str(config.FAST_REID_WEIGHTS)
                
                # 验证配置文件存在
                if not Path(reid_config).exists():
                    raise FileNotFoundError(f"FastReID config not found at {reid_config}")
                if not Path(reid_weights).exists():
                    raise FileNotFoundError(f"FastReID weights not found at {reid_weights}")
                
                # 初始化FastReID编码器（与BoT-SORT相同的方式）
                self.reid_encoder = FastReIDInterface(
                    config_file=reid_config,
                    weights_path=reid_weights,
                    device=reid_device,
                    batch_size=reid_batch_size
                )
                
                print(f"[Camera {self.camera.id}] FastReID initialized:")
                print(f"  - Config: {reid_config}")
                print(f"  - Weights: {reid_weights}")
                print(f"  - Device: {reid_device}")
                print(f"  - Batch size: {reid_batch_size}")
                
            except ImportError as e:
                print(f"[Camera {self.camera.id}] FastReID import failed: {e}")
                print(f"[Camera {self.camera.id}] ReID feature extraction will be disabled")
                self.with_reid = False
                self.reid_encoder = None
            except Exception as e:
                print(f"[Camera {self.camera.id}] FastReID initialization failed: {e}")
                print(f"[Camera {self.camera.id}] ReID feature extraction will be disabled")
                self.with_reid = False
                self.reid_encoder = None
                
        self.frame = None
        self.raw_detections = None  # YOLOX 原始检测结果 [N, 7]
        self.detections_filtered = []  # 过滤后的 Detection 对象列表
        self.pitch_positions = None
        self.pixel_positions = None
        self.mask = None
        

    def detect(self, frame=None, conf=0.25, verbose=False, auto_create_detections=True):
        """
        运行YOLOX检测并更新内部状态
        
        Args:
            frame: 输入图像，如果为 None 则使用相机的第一帧
            conf: 置信度阈值
            verbose: 是否打印详细信息
            auto_create_detections: 是否自动创建Detection对象（默认True，保持向后兼容）
            
        Returns:
            如果 auto_create_detections=True: 返回 List[Detection]
            如果 auto_create_detections=False: 返回 raw_detections (np.ndarray)
        """
        # 如果没有提供 frame，使用相机的第一帧
        if frame is None:
            frame = self.camera.first_frame
        
        self.frame = frame
        self.raw_detections = None  # 清空旧结果

        if self.yolox_predictor is None: 
            return [] if auto_create_detections else None
            
        # 运行 YOLOX 检测
        self.yolox_predictor.confthre = conf
        self.yolox_predictor.nmsthre = 0.45
        
        outputs, img_info = self.yolox_predictor.inference(frame, Timer())
        
        # 存储原始检测结果
        # outputs[0]: [N, 7] (x1, y1, x2, y2, obj_conf, cls_conf, cls_id)
        if outputs[0] is not None:
            # 还原尺度（YOLOX 返回的坐标是相对于 resize 后的图像）
            h_raw, w_raw = img_info['height'], img_info['width']
            test_size = self.yolox_predictor.test_size
            scale = min(test_size[0] / h_raw, test_size[1] / w_raw)
            
            dets = outputs[0].cpu().numpy()
            dets[:, :4] /= scale  # 还原到原始图像尺度
            self.raw_detections = dets
        else:
            self.raw_detections = None
        
        # 根据参数决定是否自动创建Detection对象
        if auto_create_detections:
            # 默认行为：自动筛选、创建Detection对象（保持向后兼容）
            self.filter_detections_in_coverage()
            return self.create_detection_objects()
        else:
            # 仅返回原始检测结果
            return self.raw_detections

    def filter_detections_in_coverage(self):
        """
        筛选出场地范围内的检测（只做筛选，不创建Detection对象）
        
        计算并更新：
        - self.pixel_positions: 脚部像素坐标 [N, 2]
        - self.pitch_positions: 球场坐标 [N, 2]
        - self.mask: 布尔掩码 [N] (True表示在场地范围内)
        
        Returns:
            pitch_positions: 球场坐标 (N, 2)
            mask: 布尔掩码 (N,)
        """
        # 检查是否有检测结果
        if self.raw_detections is None or len(self.raw_detections) == 0:
            self.pixel_positions = np.empty((0, 2))
            self.pitch_positions = np.empty((0, 2))
            self.mask = np.array([], dtype=bool)
            return self.pitch_positions, self.mask
        
        # 1. 提取所有检测的 bbox（已经是原始图像尺度）
        # raw_detections: [N, 7] (x1, y1, x2, y2, obj_conf, cls_conf, cls_id)
        bboxes = self.raw_detections[:, :4]  # [N, 4]
        
        # 2. 提取脚部位置（像素坐标）
        from tools.bbox_tools import get_foot_position
        foot_positions_pixel = []
        for bbox in bboxes:
            foot_pos = get_foot_position(bbox)
            foot_positions_pixel.append(foot_pos)
        foot_positions_pixel = np.array(foot_positions_pixel, dtype=np.float32)
        self.pixel_positions = foot_positions_pixel
        
        # 3. 转换到球场坐标（自动选择单应性矩阵或TPS）
        self.pitch_positions = self.camera.pixel_to_pitch(foot_positions_pixel)
        
        # 4. 过滤检测（只保留场地范围内的点）
        _, self.mask = self.camera.filter_points_in_coverage(self.pitch_positions)
        
        return self.pitch_positions, self.mask
    
    def create_detection_objects(self, extract_reid=True):
        """
        将筛选后的检测封装为Detection对象，并可选提取ReID特征
        
        前置条件：
        - 必须先调用 filter_detections_in_coverage()
        - self.raw_detections, self.mask, self.pitch_positions, self.pixel_positions 必须已设置
        
        Args:
            extract_reid: 是否提取ReID特征（默认False，避免不必要的计算）
        
        Returns:
            detections_filtered: List[Detection]
        """
        # 检查前置条件
        if self.raw_detections is None or self.mask is None:
            print("[CameraDetection] Warning: 必须先调用 filter_detections_in_coverage()")
            self.detections_filtered = []
            return self.detections_filtered
        
        if len(self.raw_detections) == 0:
            self.detections_filtered = []
            return self.detections_filtered
        
        # 1. 收集需要创建Detection对象的检测索引
        valid_indices = np.where(self.mask)[0]
        
        if len(valid_indices) == 0:
            self.detections_filtered = []
            return self.detections_filtered
        
        # 2. 提取crop图像（如果需要ReID）
        crops_for_reid = []
        bboxes_for_reid = []
        
        # 3. 创建Detection对象
        self.detections_filtered = []
        for i in valid_indices:
            det = self.raw_detections[i]
            
            # 提取 bbox 信息
            bbox = det[:4]  # [x1, y1, x2, y2]
            obj_conf = det[4]
            cls_conf = det[5]
            confidence = obj_conf * cls_conf  # 综合置信度
            class_id = int(det[6])
            
            # 获取 grid_size（必须从配置获取，确保与 score_maps 一致）
            if hasattr(self.camera, 'mcs') and self.camera.mcs is not None:
                # 从 MCS 配置获取 grid_size（与 score_maps 生成时使用的一致）
                if hasattr(self.camera.mcs, 'config') and hasattr(self.camera.mcs.config, 'grid_size'):
                    grid_size = float(self.camera.mcs.config.grid_size)
                else:
                    raise ValueError("[CameraDetection] MCS 配置中缺少 grid_size 属性")
            else:
                # 单相机模式：从 Camera 对象获取（应该在 set_coverage 或初始化时设置）
                grid_size = float(getattr(self.camera, 'grid_size', 50.0))
            
            # 获取质量分数（从MCS查询当前相机在该网格的质量分数）
            quality_score = self._get_quality_score(self.pitch_positions[i])
            
            # 提取crop图像（如果需要ReID特征）
            crop_image = None
            if extract_reid and self.frame is not None:
                x1, y1, x2, y2 = bbox.astype(int)
                # 边界检查
                x1, y1 = max(0, x1), max(0, y1)
                x2 = min(self.frame.shape[1], x2)
                y2 = min(self.frame.shape[0], y2)
                
                # 提取crop（深拷贝，避免引用问题）
                crop_image = self.frame[y1:y2, x1:x2].copy()
            
            # 创建 Detection 对象
            # 所有属性在 __post_init__ 中完成初始化：
            # - grid_position: 由 pitch_position 自动计算
            # - confidence_weight: 由 confidence 经 Sigmoid 处理
            # - final_score: quality_score × confidence_weight
            # - is_occluded: 稍后批量计算 IoU
            detection = Detection(
                camera_id=self.camera.id,
                bbox=bbox,
                confidence=confidence,
                pitch_position=self.pitch_positions[i],
                pixel_position=self.pixel_positions[i],
                crop_image=crop_image,  # 保存crop图像（如果提取了）
                quality_score=quality_score,
                grid_size=grid_size,
                frame_id=getattr(self, 'current_frame_id', 0),
                class_id=class_id
            )
            self.detections_filtered.append(detection)
            
            # 收集用于批量ReID提取
            if extract_reid:
                crops_for_reid.append(detection)
                bboxes_for_reid.append(bbox)
        
        # 4. 批量提取ReID特征（学习BoT-SORT的方式）
        if extract_reid and len(crops_for_reid) > 0:
            if self.with_reid and self.reid_encoder is not None:
                try:
                    # 构建bbox数组用于FastReID批量推理
                    # 格式: [N, 4] (x1, y1, x2, y2) - 与BoT-SORT相同
                    bboxes_array = np.array(bboxes_for_reid)
                    
                    # 批量提取ReID特征（调用方式与BoT-SORT完全相同）
                    # self.reid_encoder.inference(image, detections)
                    # - image: 完整的帧图像
                    # - detections: bbox数组 [N, 4]
                    reid_features = self.reid_encoder.inference(self.frame, bboxes_array)
                    
                    # 验证返回的特征维度
                    if len(reid_features) != len(crops_for_reid):
                        print(f"[CameraDetection] Warning: ReID特征数量 ({len(reid_features)}) "
                              f"与检测数量 ({len(crops_for_reid)}) 不匹配")
                    
                    # 填充到Detection对象
                    for detection, reid_feat in zip(crops_for_reid, reid_features):
                        detection.reid_feature = reid_feat
                    
                    # 输出特征形状信息（首次调用时）
                    if len(reid_features) > 0:
                        feat_shape = reid_features[0].shape if hasattr(reid_features[0], 'shape') else len(reid_features[0])
                        print(f"[CameraDetection] Camera {self.camera.id}: "
                              f"成功提取 {len(reid_features)} 个ReID特征 (维度: {feat_shape})")
                    
                except Exception as e:
                    print(f"[CameraDetection] Camera {self.camera.id}: ReID特征提取失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                if extract_reid:
                    print(f"[CameraDetection] Camera {self.camera.id}: "
                          f"ReID编码器未初始化 (with_reid={self.with_reid})，跳过特征提取")
        
        # 5. 批量计算遮挡情况（基于IoU）
        if len(self.detections_filtered) >= 2:
            self._compute_occlusions(iou_threshold=0.15)
        
        return self.detections_filtered
    
    def draw_detections(self, show_all=True, show_pitch_pos=True):
        """
        绘制检测框
        
        Args:
            show_all: 是否显示所有检测（包括不在责任区的，灰色显示）
            show_pitch_pos: 是否显示球场坐标文本
            
        Returns:
            frame: 绘制后的图像
        """
        frame = self.frame.copy()
        
        if self.raw_detections is None or len(self.raw_detections) == 0:
            return frame
        
        # 遍历所有原始检测
        for i, det in enumerate(self.raw_detections):
            # 提取 bbox 和置信度
            x1, y1, x2, y2 = det[:4].astype(int)
            obj_conf = det[4]
            cls_conf = det[5]
            conf = obj_conf * cls_conf
            
            # 判断是否在责任区内
            if self.mask is not None and i < len(self.mask):
                in_coverage = self.mask[i]
            else:
                in_coverage = False
            
            # 颜色和标签
            if in_coverage:
                color = (0, 255, 0)  # 绿色：在责任区内
                label = f"Cam{self.camera.id} ID:{i}"
            else:
                if not show_all:
                    continue  # 不显示不在责任区的
                color = (128, 128, 128)  # 灰色：不在责任区
                label = f"Out ID:{i}"
            
            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # 画标签
            label_text = f"{label} {conf:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # 显示球场坐标（可选）
            if show_pitch_pos and self.pitch_positions is not None and i < len(self.pitch_positions):
                pitch_pos = self.pitch_positions[i]
                pos_text = f"({pitch_pos[0]:.0f}, {pitch_pos[1]:.0f})"
                cv2.putText(frame, pos_text, (x1, y2 + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # 添加相机信息
        total_dets = len(self.raw_detections) if self.raw_detections is not None else 0
        info_text = f"Camera {self.camera.id}: {len(self.detections_filtered)}/{total_dets} in coverage"
        cv2.putText(frame, info_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        
        return frame
    
    def get_filtered_detections(self) -> List[Detection]:
        """
        获取过滤后的检测结果（返回 Detection 对象）
        
        Returns:
            List[Detection]: Detection对象列表
        """
        # detections_filtered 已经是 Detection 对象列表
        return self.detections_filtered
    
    def _get_quality_score(self, pitch_position: np.ndarray) -> float:
        """
        获取指定位置的当前相机质量分数（从 MCS 的 score_maps 查询）
        
        Args:
            pitch_position: 球场坐标 [x, y] (cm)
        
        Returns:
            quality_score: 当前相机在该网格的质量分数 (Combined Score)
        """
        # 调试：添加详细日志（首次调用时输出）
        debug = False  # 设为 True 启用调试输出
        
        # 检查是否有 MCS 引用
        if not hasattr(self.camera, 'mcs') or self.camera.mcs is None:
            if debug:
                print(f"[Debug] _get_quality_score: camera.mcs 不存在或为 None")
            return 1.0
        
        mcs = self.camera.mcs
        
        # 检查必要的属性
        if not hasattr(mcs, 'grid_system'):
            if debug:
                print(f"[Debug] _get_quality_score: mcs 缺少 grid_system 属性")
            return 1.0
            
        if not hasattr(mcs, 'score_maps'):
            if debug:
                print(f"[Debug] _get_quality_score: mcs 缺少 score_maps 属性")
            return 1.0
        
        grid_system = mcs.grid_system
        
        try:
            # 获取当前相机的分数图
            if self.camera.numeric_id not in mcs.score_maps:
                if debug:
                    print(f"[Debug] _get_quality_score: camera {self.camera.numeric_id} 不在 score_maps 中")
                    print(f"          score_maps keys: {list(mcs.score_maps.keys())}")
                return 0.0
            
            score_map = mcs.score_maps[self.camera.numeric_id]
            
            # 转换为网格索引
            row, col = grid_system.physical_to_index(pitch_position)
            
            # 边界检查
            rows, cols = score_map.shape
            if row < 0 or row >= rows or col < 0 or col >= cols:
                if debug:
                    print(f"[Debug] _get_quality_score: 网格索引 ({row}, {col}) 超出范围 ({rows}, {cols})")
                return 0.0
            
            # 获取质量分数
            quality_score = float(score_map[row, col])
            
            # 只在第一次或前几次调用时输出调试信息
            if debug and not hasattr(self, '_debug_count'):
                self._debug_count = 0
            
            if debug and self._debug_count < 3:
                print(f"[Debug] _get_quality_score 成功:")
                print(f"        camera.numeric_id={self.camera.numeric_id}")
                print(f"        pitch_pos={pitch_position}")
                print(f"        grid=({row},{col})")
                print(f"        quality_score={quality_score:.3f}")
                self._debug_count += 1
            
            return quality_score
            
        except Exception as e:
            # 发生任何错误时返回默认值
            print(f"[CameraDetection] Warning: Failed to get quality score: {e}")
            import traceback
            traceback.print_exc()
            return 1.0
    
    def _compute_occlusions(self, iou_threshold: float = 0.15):
        """
        计算检测框之间的遮挡关系（基于IoU）
        
        Args:
            iou_threshold: IoU阈值，超过此值认为发生遮挡
        """
        if len(self.detections_filtered) < 2:
            return
        
        try:
            from torchvision.ops import box_iou
            
            # 提取所有检测框
            boxes = torch.tensor([det.bbox for det in self.detections_filtered], dtype=torch.float32)
            
            # 计算 IoU 矩阵
            iou_matrix = box_iou(boxes, boxes)
            
            # 获取上三角索引（排除对角线和重复比较）
            triu_idx = torch.triu_indices(
                iou_matrix.shape[0],
                iou_matrix.shape[1],
                offset=1,
                device=iou_matrix.device
            )
            
            # 获取上三角的所有IoU值
            upper_tri_ious = iou_matrix[triu_idx[0], triu_idx[1]]
            
            # 找到超过阈值的遮挡对
            mask = upper_tri_ious > iou_threshold
            
            if mask.any():
                idx1 = triu_idx[0][mask]
                idx2 = triu_idx[1][mask]
                
                # 标记发生遮挡的检测
                for i in idx1.tolist():
                    self.detections_filtered[i].is_occluded = True
                for i in idx2.tolist():
                    self.detections_filtered[i].is_occluded = True
                    
        except Exception as e:
            print(f"[CameraDetection] Warning: Failed to compute occlusions: {e}")

    def visualize_on_pitch(self, pitch_image=None, pitch_config=None, 
                          show_numbers=False, radius=3, 
                          color=None, scale=0.1, padding=50,
                          draw_coverage_background=True, coverage_alpha=0.3,
                          auto_save=True):
        """
        将过滤后的检测可视化到球场坐标系统上
    
        Args:
            pitch_image: 球场图像（如果为 None，则自动生成）
            pitch_config: 球场配置对象（用于生成球场图）
            show_numbers: 是否显示编号
            radius: 点的半径（实心圆）
            color: 点的颜色 (B,G,R)，None 则使用相机 ID 自动分配
            scale: 球场图缩放比例
            padding: 球场图边距
            draw_coverage_background: 是否绘制覆盖区域背景
            coverage_alpha: 覆盖区域背景透明度 (0-1)
            auto_save: 是否自动保存到 detection 路径
            
        Returns:
            pitch_image: 绘制后的球场图像
        """
        
        # 生成球场图（如果没有提供）
        if pitch_image is None:
            if pitch_config is None:
                raise ValueError("必须提供 pitch_image 或 pitch_config")
            from tools.annotation.pitch_annotation_tool import draw_pitch
            pitch_image = draw_pitch(pitch_config, scale=scale, padding=padding)
        
        pitch_image = pitch_image.copy()
        
        # 自动分配颜色（基于相机 ID）
        if color is None:
            colors_map = {
                1: (0, 0, 255),    # 红色
                2: (0, 255, 0),    # 绿色
                3: (255, 0, 0),    # 蓝色
                4: (0, 255, 255),  # 黄色
                5: (255, 0, 255),  # 品红
            }
            color = colors_map.get(self.camera.numeric_id, (255, 255, 255))
        
        # 覆盖区域背景功能已删除（不再使用分配图）
        
        # 如果没有过滤后的检测，返回背景
        if self.pitch_positions is None or len(self.detections_filtered) == 0:
            return pitch_image
        
        # 获取过滤后的位置（只绘制在责任区内的点）
        filtered_positions = self.pitch_positions[self.mask]
        
        # 使用 transform 模块转换坐标
        from tools.transform import pitch_to_canvas_pixel
        canvas_positions = pitch_to_canvas_pixel(filtered_positions, scale=scale, padding=padding)
        
        # 球员颜色：比区域颜色更深更饱和
        player_color = tuple(int(c * 0.8) for c in color)  # 稍微加深
        
        # 绘制每个球员（实心圆点）
        for i, canvas_pos in enumerate(canvas_positions):
            img_x, img_y = int(canvas_pos[0]), int(canvas_pos[1])
            
            # 画实心圆点
            cv2.circle(pitch_image, (img_x, img_y), radius, player_color, -1)
            
            # 画编号（可选）
            if show_numbers:
                label = f"{i+1}"
                cv2.putText(pitch_image, label, 
                           (img_x + radius + 2, img_y + radius),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, player_color, 1)
        
        # 自动保存到 detection 路径
        if auto_save:
            detection_dir = self.camera.store_dir / "detection"
            detection_dir.mkdir(exist_ok=True)
            save_path = detection_dir / f"{self.camera.name}_detection_pitch.jpg"
            imwritex(str(save_path), pitch_image)
            print(f"[Camera {self.camera.id}] ✅ 检测可视化已保存: {save_path}")
        
        return pitch_image
    
    def get_positions_pixel(self):
        """
        获取所有检测的脚部位置（像素坐标）
    
        Returns:
            foot_positions: (N, 2) 数组
        """
        return self.pixel_positions


class TrainArea:
    """
    训练区域管理类

    用于定义和管理球场上的训练区域（多边形）。
    支持从视频画面标定或从球场俯视图标定。
    """

    def __init__(self, camera: Camera):
        self.camera = camera
        self.region_points: List[Tuple[float, float]] = []  # 球场坐标点列表 [(x,y), ...]
        
        # 自动加载已有的训练区域文件
        self._auto_load()
    
    def _auto_load(self):
        """自动加载训练区域文件（如果存在）"""
        train_area_path = self.camera._get_calibration_paths()['train_area']
        if train_area_path.exists():
            try:
                self.load(str(train_area_path))
                print(f"[Camera {self.camera.id}] 已加载训练区域: {len(self.region_points)} 个点")
            except Exception as e:
                print(f"[Camera {self.camera.id}] 加载训练区域失败: {e}")
    
    def save(self, file_path: Optional[str] = None):
        """
        保存训练区域到 JSON 文件
        
        Args:
            file_path: 保存路径，如果为 None 则使用默认路径
        """
        import json
        
        if file_path is None:
            file_path = str(self.camera._get_calibration_paths()['train_area'])
        
        data = {
            'camera_id': self.camera.id,
            'region_points': self.region_points
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[Camera {self.camera.id}] 训练区域已保存至: {file_path}")
    
    def load(self, file_path: Optional[str] = None):
        """
        从 JSON 文件加载训练区域
        
        Args:
            file_path: 加载路径，如果为 None 则使用默认路径
        """
        import json
        
        if file_path is None:
            file_path = str(self.camera._get_calibration_paths()['train_area'])
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.region_points = data.get('region_points', [])
        print(f"[Camera {self.camera.id}] 训练区域已加载: {len(self.region_points)} 个点")

    def define_from_video(self):
        """
        从相机视频画面中交互式标定训练区域点
        """
        from area import MarkerAnnotator

        print(f"[Camera {self.camera.id}] 开始从视频画面标定训练区域...")
        annotator = MarkerAnnotator()
        # 临时获取一帧
        gen = self.camera.get_frame_generator()
        frame = next(gen)
        self.camera.video_manager.reset()

        result = annotator.annotate(frame)
        if result and "groups" in result and len(result["groups"]) > 0:
            pixel_points = result["groups"][0]  # 取第一组点
            if pixel_points:
                # 转换为球场坐标
                pitch_points = self.camera.pixel_to_pitch(pixel_points)
                self.region_points = pitch_points.tolist()
                print(f"[Camera {self.camera.id}] 训练区域已更新: {len(self.region_points)} 个点")
                # 自动保存
                self.save()
            else:
                print(f"[Camera {self.camera.id}] 未选择任何点")
        else:
            print(f"[Camera {self.camera.id}] 标定取消")

    def define_from_pitch_map(self, padding=50, scale=0.1):
        """
        从球场俯视图（带关键点）中交互式标定训练区域

        显示上一步从视频标定的点（如果有）和球场固定关键点。
        用户通过终端输入要选取的点序号序列。
        """
        from tools.annotation.pitch_annotation_tool import draw_pitch_with_keypoints

        print(f"[Camera {self.camera.id}] 开始从球场俯视图标定训练区域...")

        # 1. 生成带关键点的球场图
        pitch_img = draw_pitch_with_keypoints(
            config=self.camera.config,
            padding=padding,
            scale=scale
        )

        # 2. 如果已有视频标定点，将其绘制出来
        if self.region_points:
            from tools.transform import pitch_to_canvas_pixel
            
            # 批量转换所有区域点
            region_canvas = pitch_to_canvas_pixel(
                np.array(self.region_points), scale=scale, padding=padding
            )
            
            for idx, (px, py) in enumerate(region_canvas):
                # 绘制点
                cv2.circle(pitch_img, (int(px), int(py)), 6, (0, 0, 255), -1)  # 红色点表示自定义点

                # 绘制标签 (使用负数索引或特定前缀区分自定义点)
                label = f"V{idx + 1}"
                cv2.putText(pitch_img, label, (int(px) + 10, int(py) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # 3. 显示图像供用户参考
        print("请查看弹出的球场图窗口。")
        print("图中包含：")
        print("  - 白色点/数字: 球场固定关键点 (序号 1, 2, 3...)")
        print("  - 红色点/V数字: 从视频画面标定的点 (序号 V1, V2...)")

        sv.plot_image(pitch_img)  # 使用 supervision 显示，或 cv2.imshow

        # 4. 接收用户输入
        print("\n请输入构成训练区域的点序号，按顺序输入，用逗号或空格分隔。")
        print("示例: 1, 2, V1, V2 (将连接 关键点1 -> 关键点2 -> 视频点1 -> 视频点2)")
        user_input = input("请输入点序列: ").strip()

        if not user_input:
            print("未输入任何点，取消操作。")
            return

        # 5. 解析输入并构建区域
        selected_points = []
        tokens = user_input.replace(',', ' ').split()

        try:
            for token in tokens:
                token = token.upper()
                if token.startswith('V'):
                    # 视频标定点 V1, V2...
                    try:
                        idx = int(token[1:]) - 1
                        if 0 <= idx < len(self.region_points):
                            selected_points.append(self.region_points[idx])
                        else:
                            print(f"警告: 视频点序号 {token} 超出范围，已忽略。")
                    except ValueError:
                        print(f"警告: 视频点格式错误 {token}，已忽略。")
                else:
                    # 球场关键点 (支持数字 1, 2 或标签 T1, B1 等)
                    # 优先尝试在 config.labels 中查找
                    found = False
                    if hasattr(self.camera.config, 'labels'):
                        # 尝试匹配标签
                        try:
                            # 尝试完整匹配 (忽略前导0，例如输入1匹配01)
                            # 用户输入 '1' -> 匹配 '01' 或 '1'
                            # 用户输入 'B1' -> 匹配 'B1'
                            
                            # 1. 直接匹配
                            if token in self.camera.config.labels:
                                idx = self.camera.config.labels.index(token)
                                found = True
                            
                            # 2. 尝试补0匹配 (如果输入是纯数字)
                            if not found and token.isdigit():
                                padded = f"{int(token):02d}"
                                if padded in self.camera.config.labels:
                                    idx = self.camera.config.labels.index(padded)
                                    found = True
                                    
                            if found:
                                if 0 <= idx < len(self.camera.config.vertices):
                                    pt = self.camera.config.vertices[idx]
                                    selected_points.append([pt[0], pt[1]])
                                else:
                                    print(f"警告: 标签 {token} 对应的顶点索引 {idx} 超出范围。")
                        except Exception as e:
                            print(f"警告: 查找标签 {token} 时出错: {e}")
                    
                    if not found:
                        # 回退到直接数字索引模式 (兼容旧逻辑)
                        try:
                            idx = int(token) - 1
                            if 0 <= idx < len(self.camera.config.vertices):
                                pt = self.camera.config.vertices[idx]
                                selected_points.append([pt[0], pt[1]])
                                found = True
                        except ValueError:
                            pass
                            
                    if not found:
                        print(f"警告: 未找到点 {token}，已忽略。")

            if len(selected_points) >= 3:
                self.region_points = selected_points
                print(f"[Camera {self.camera.id}] 训练区域已更新: {len(self.region_points)} 个点")
                # 自动保存
                self.save()
            else:
                print(f"错误: 有效点数量不足 3 个，无法构成区域。")

        except ValueError as e:
            print(f"输入格式错误: {e}")

    def draw_on_pitch(self, pitch_image, color=(0, 255, 0), thickness=2, padding=50, scale=0.1):
        """在球场图上绘制训练区域"""
        if not self.region_points:
            return pitch_image

        pts = []
        for x, y in self.region_points:
            px = int(x * scale + padding)
            py = int(y * scale + padding)
            pts.append([px, py])

        pts = np.array(pts, np.int32).reshape((-1, 1, 2))
        cv2.polylines(pitch_image, [pts], True, color, thickness)

        # 绘制半透明填充
        overlay = pitch_image.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, 0.3, pitch_image, 0.7, 0, pitch_image)

        return pitch_image


if '__main__' == __name__:
    # 示例用法
    # 需要提供实际的视频文件路径
    video_file = "videos/test/test.mp4" 
    
    if Path(video_file).exists():
        CONFIG = HalfPitchConfiguration()

        try:
            # 方式1: 正常模式 - 读取已保存的标定数据（如果存在）
            print(f"Testing Camera class (Normal Mode) with {video_file}...")
            cam = Camera(video_file, CONFIG, camera_id=1)
            print(f"Camera Info: {cam.info}")

            # 方式2: 调试模式 - 强制重新计算，不读取已保存数据
            # print(f"Testing Camera class (Debug Mode) with {video_file}...")
            # cam_debug = Camera(video_file, CONFIG, camera_id=1, debug_mode=True)
            # 调试模式会跳过读取 distortion_params.json 和 camera_info.json
            # 但会在计算完成后保存新结果
            
            # 方式3: 自定义名称
            # cam_left = Camera(video_file, CONFIG, camera_id=1, camera_name="left_camera")

            # 绘制视野
            view = cam.get_fov_boundaries(padding=50, scale=0.1)
            sv.plot_image(view)

        except Exception as e:
                print(f"Error: {e}")
    else:
        print(f"Test video file not found: {video_file}")
