import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from camera import *
from soccer import *
from area import imwritex
import supervision as sv
from tools.frame_tool import create_combined_image_layout
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
matplotlib.use('Agg')  # 非交互式后端，适合服务器环境


# ==========================================
# 数学模型层：封装所有分数计算公式
# ==========================================

class MCSScoreModel:
    """
    MCS数学模型类
    
    职责：封装所有质量分数计算的数学公式
    - 距离分数：Sigmoid模型
    - 畸变分数：Brown-Conrady物理模型
    - 置信度权重：Sigmoid模型
    
    优势：
    - 便于单独测试和修改数学模型
    - 可以轻松切换不同的计算公式
    - 参数配置集中管理
    """
    
    @staticmethod
    def sigmoid_distance_score(distance: float, D_max: float, 
                               half_point_ratio: float = 0.5, 
                               steepness_factor: float = 8.0) -> float:
        """
        单点距离分数（Sigmoid模型）
        
        Args:
            distance: 点到相机的距离 (cm)
            D_max: 相机最大有效距离 (cm)
            half_point_ratio: 半衰点位置比例（相对D_max）
            steepness_factor: 陡峭度因子
            
        Returns:
            距离分数 [0, 1]
        """
        if distance > D_max:
            return 0.0
        
        D_half = half_point_ratio * D_max
        k = steepness_factor / D_max
        exponent = k * (distance - D_half)
        exponent = np.clip(exponent, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(exponent))
    
    @staticmethod
    def sigmoid_distance_score_batch(distances: np.ndarray, D_max: float,
                                     half_point_ratio: float = 0.5,
                                     steepness_factor: float = 8.0) -> np.ndarray:
        """
        批量距离分数（Sigmoid模型）
        
        Args:
            distances: (N,) 距离数组 (cm)
            D_max: 相机最大有效距离 (cm)
            half_point_ratio: 半衰点位置比例
            steepness_factor: 陡峭度因子
            
        Returns:
            (N,) 距离分数数组 [0, 1]
        """
        D_half = half_point_ratio * D_max
        k = steepness_factor / D_max
        exponents = k * (distances - D_half)
        exponents = np.clip(exponents, -20.0, 20.0)
        scores = 1.0 / (1.0 + np.exp(exponents))
        
        # 应用距离过滤
        scores[distances > D_max] = 0.0
        return scores
    
    @staticmethod
    def brown_conrady_distortion_magnitude(r_norm: np.ndarray, 
                                          k1: float, k2: float, k3: float) -> np.ndarray:
        """
        Brown-Conrady畸变量计算
        
        物理模型：Δr = r × (k₁r² + k₂r⁴ + k₃r⁶)
        
        Args:
            r_norm: 归一化半径 [0, 1]（标量或数组）
            k1, k2, k3: 畸变系数
            
        Returns:
            畸变量绝对值（标量或数组）
        """
        r2 = r_norm * r_norm
        r4 = r2 * r2
        r6 = r4 * r2
        return np.abs(r_norm * (k1 * r2 + k2 * r4 + k3 * r6))
    
    @staticmethod
    def distortion_to_score(distortion_magnitude: np.ndarray,
                           scale_factor: float = 50.0) -> np.ndarray:
        """
        将畸变量转换为分数
        
        Args:
            distortion_magnitude: 畸变量（标量或数组）
            scale_factor: 缩放系数，控制惩罚强度
            
        Returns:
            畸变分数 [0.5, 1.0]（标量或数组）
        """
        raw_score = 1.0 / (1.0 + scale_factor * distortion_magnitude)
        # 重映射到 [0.5, 1.0] 区间，确保边缘位置仍保持一定可信度
        return 0.5 + 0.5 * raw_score
    
    @staticmethod
    def sigmoid_confidence_weight(confidence: float, 
                                 midpoint: float = 0.5,
                                 steepness: float = 10.0) -> float:
        """
        Sigmoid置信度权重计算
        
        Args:
            confidence: YOLO检测置信度 [0, 1]
            midpoint: Sigmoid中点
            steepness: 陡峭度
            
        Returns:
            置信度权重 [0, 1]
        """
        x = (confidence - midpoint) * steepness
        x = np.clip(x, -20.0, 20.0)
        return 1.0 / (1.0 + np.exp(-x))


# ==========================================
# 全局追踪数据结构
# ==========================================

class TrackStatus(Enum):
    """轨迹状态枚举"""
    TENTATIVE = "tentative"    # 初始状态（帧数 < 3）
    CONFIRMED = "confirmed"    # 确认状态（稳定追踪）
    LOST = "lost"              # 丢失状态（未匹配但保留）
    DELETED = "deleted"        # 已删除


@dataclass
class Track:
    """
    全局球场坐标追踪对象
    
    用于多相机系统的全局追踪，在球场坐标系中维护轨迹。
    """
    # ========== 基础信息 ==========
    id: int                                 # 全局唯一轨迹ID
    position: np.ndarray                    # 当前球场位置 [x, y] (cm)
    velocity: np.ndarray = field(default_factory=lambda: np.array([0., 0.]))  # 速度 [vx, vy] (cm/s)
    
    # ========== 状态信息 ==========
    status: TrackStatus = TrackStatus.TENTATIVE  # 轨迹状态
    age: int = 0                            # 轨迹年龄（总帧数）
    hits: int = 0                           # 匹配次数
    time_since_update: int = 0              # 未更新帧数
    
    # ========== 相机与检测信息 ==========
    camera_id: int = 0                      # 当前主相机ID
    bbox: Optional[np.ndarray] = None       # 当前检测框 [x1, y1, x2, y2]
    confidence: float = 0.0                 # 当前检测置信度
    
    # ========== 卡尔曼滤波状态 ==========
    mean: Optional[np.ndarray] = None       # 卡尔曼滤波均值 [x, y, vx, vy]
    covariance: Optional[np.ndarray] = None # 卡尔曼滤波协方差矩阵
    
    # ========== 多模态特征 ==========
    reid_feature: Optional[np.ndarray] = None      # 平滑后的ReID特征
    reid_features_history: List[np.ndarray] = field(default_factory=list)  # ReID历史（用于平滑）
    dominant_color: Optional[str] = None           # 主颜色
    jersey_number: Optional[str] = None            # 球衣号码
    
    # ========== 历史轨迹 ==========
    positions_history: List[np.ndarray] = field(default_factory=list)  # 位置历史
    timestamps_history: List[float] = field(default_factory=list)      # 时间戳历史
    
    def __post_init__(self):
        """初始化后处理"""
        # 确保 numpy 数组类型
        if not isinstance(self.position, np.ndarray):
            self.position = np.array(self.position)
        if not isinstance(self.velocity, np.ndarray):
            self.velocity = np.array(self.velocity)
    
    def update(self, detection: 'Detection', timestamp: float):
        """
        使用新检测更新轨迹
        
        Args:
            detection: Detection对象
            timestamp: 当前时间戳（秒）
        """
        self.position = detection.pitch_position.copy()
        self.camera_id = detection.camera_id
        self.bbox = detection.bbox.copy()
        self.confidence = detection.confidence
        
        # 更新特征
        if detection.reid_feature is not None:
            self._update_reid_feature(detection.reid_feature)
        if detection.dominant_color:
            self.dominant_color = detection.dominant_color
        
        # 更新状态
        self.hits += 1
        self.time_since_update = 0
        self.age += 1
        
        # 更新状态机
        if self.hits >= 3 and self.status == TrackStatus.TENTATIVE:
            self.status = TrackStatus.CONFIRMED
        
        # 记录历史
        self.positions_history.append(self.position.copy())
        self.timestamps_history.append(timestamp)
        
        # 限制历史长度
        max_history = 100
        if len(self.positions_history) > max_history:
            self.positions_history = self.positions_history[-max_history:]
            self.timestamps_history = self.timestamps_history[-max_history:]
    
    def predict(self, dt: float = 1/30.0):
        """
        预测下一帧位置（用于卡尔曼滤波）
        
        Args:
            dt: 时间间隔（秒）
        """
        # 简单的匀速运动模型
        self.position = self.position + self.velocity * dt
        self.age += 1
        self.time_since_update += 1
        
        # 检查是否丢失
        if self.time_since_update > 30:  # 30帧未更新
            self.status = TrackStatus.LOST
    
    def _update_reid_feature(self, new_feature: np.ndarray, alpha: float = 0.9):
        """
        平滑更新 ReID 特征（指数移动平均）
        
        Args:
            new_feature: 新的ReID特征
            alpha: 平滑系数（越大越依赖历史）
        """
        self.reid_features_history.append(new_feature.copy())
        
        if self.reid_feature is None:
            self.reid_feature = new_feature.copy()
        else:
            self.reid_feature = alpha * self.reid_feature + (1 - alpha) * new_feature
            # L2归一化
            norm = np.linalg.norm(self.reid_feature)
            if norm > 0:
                self.reid_feature /= norm
        
        # 限制历史长度
        if len(self.reid_features_history) > 10:
            self.reid_features_history = self.reid_features_history[-10:]
    
    def to_dict(self) -> dict:
        """转换为字典格式（用于序列化或可视化）"""
        return {
            'id': self.id,
            'position': self.position.tolist() if isinstance(self.position, np.ndarray) else self.position,
            'velocity': self.velocity.tolist() if isinstance(self.velocity, np.ndarray) else self.velocity,
            'status': self.status.value,
            'age': self.age,
            'hits': self.hits,
            'time_since_update': self.time_since_update,
            'camera_id': self.camera_id,
            'bbox': self.bbox.tolist() if self.bbox is not None else None,
            'confidence': self.confidence,
            'color': self.dominant_color,
            'jersey_number': self.jersey_number,
        }

# TODO：现在是经验设置的硬编码参数
@dataclass
class MCSSystemConfig:
    """MCS 配置类（Brown-Conrady物理模型）"""
    pitch_config: object
    grid_size: int = 50  # 网格大小 (cm)
    
    # 距离分数参数（Sigmoid模型）
    distance_half_point_ratio: float = 0.5      # 距离半衰点比例（相对D_max）
    distance_steepness_factor: float = 8.0      # 距离陡峭度因子
    
    # 畸变分数参数（Brown-Conrady物理模型）
    distortion_scale_factor: float = 50.0       # 畸变量缩放系数（控制惩罚强度）
    
    # 相机颜色映射（用于可视化）
    # BGR 格式，适用于 OpenCV
    CAMERA_COLORS = {
        0: (50, 50, 50),      # 未覆盖区域 - 深灰色
        1: (0, 0, 255),       # 相机1 - 红色
        2: (0, 255, 0),       # 相机2 - 绿色
        3: (255, 0, 0),       # 相机3 - 蓝色
        4: (0, 255, 255),     # 相机4 - 黄色
        5: (255, 0, 255),     # 相机5 - 品红
    }

# TODO：场地分数数学模型简单，凭感觉写的，待优化
class MultiCameraScheduler:
    def __init__(self, cameras: List[object], config: MCSSystemConfig):
        """
        Args:
            cameras: Camera对象列表 (camera.py 中的 Camera 类)
            config: 系统配置
        """
        self.camera_objects = cameras
        # 提取 CameraInfo 并建立映射 {numeric_id: CameraInfo}
        # 使用 numeric_id（整数）作为键
        self.cameras = {cam.numeric_id: cam.info for cam in cameras}
        
        # 建立 numeric_id -> Camera对象 的映射（用于访问 pitch_to_pixel）
        self.camera_objects_map = {cam.numeric_id: cam for cam in cameras}
        
        self.config = config
        self.score_maps = {}        # 存储每个相机的评分图 {cam_id: (H, W) score matrix}
        self.grid_system = None     # 存储网格系统（在 compute_coverage 时创建）
        
        # 预计算每个相机的畸变参数（优化性能）
        self._precompute_distortion_params()

    def _precompute_distortion_params(self):
        """
        预计算每个相机的畸变参数
        
        缓存:
        - distortion_coeffs: [k1, k2, k3] 系数
        - image_center: (cx, cy) 图像中心
        - r_max: 归一化半径（对角线长度的一半）
        """
        self.distortion_cache = {}
        
        for cam_id, cam_info in self.cameras.items():
            # 从 Camera 对象获取图像尺寸
            camera_obj = self.camera_objects_map.get(cam_id)
            if camera_obj is None:
                print(f"[MCS] ⚠️  相机 {cam_id} 没有对应的 Camera 对象，跳过畸变预计算")
                continue
            
            h, w = camera_obj.frame_shape  # (height, width)
            distortion_params = getattr(cam_info, 'distortion_params', None)
            
            if distortion_params is not None and len(distortion_params) >= 5:
                if not isinstance(distortion_params, np.ndarray):
                    distortion_params = np.array(distortion_params)
                
                k1, k2, p1, p2, k3 = distortion_params[:5]
                
                # 缓存畸变系数（只需径向畸变k1, k2, k3）
                self.distortion_cache[cam_id] = {
                    'k1': k1,
                    'k2': k2,
                    'k3': k3,
                    'image_center': np.array([w / 2.0, h / 2.0]),
                    'r_max': np.sqrt((w / 2.0)**2 + (h / 2.0)**2),  # 对角线一半
                    'has_distortion': True
                }
            else:
                # 无畸变相机
                self.distortion_cache[cam_id] = {
                    'has_distortion': False,
                    'image_center': np.array([w / 2.0, h / 2.0]),
                }

    def _calculate_scores_batch(self, points: np.ndarray, cam: CameraInfo, cam_id: int) -> np.ndarray:
        """
        批量计算多个点的质量分数（性能优化）
        
        Args:
            points: (N, 2) 场地坐标点数组
            cam: 相机信息对象
            cam_id: 相机ID（用于访问缓存）
            
        Returns:
            scores: (N,) 分数数组
        """
        # 1. 批量计算距离分数（使用MCSScoreModel）
        distances = np.linalg.norm(points - cam.position, axis=1)
        scores_dist = MCSScoreModel.sigmoid_distance_score_batch(
            distances,
            cam.max_effective_dist,
            self.config.distance_half_point_ratio,
            self.config.distance_steepness_factor
        )
        
        # 2. 批量计算畸变分数
        scores_distortion = self._calculate_distortion_scores_batch(points, cam, cam_id)
        
        # 3. 乘法组合
        scores = scores_dist * scores_distortion
        
        return np.clip(scores, 0.0, 1.0)
    
    def _calculate_distortion_scores_batch(self, points: np.ndarray, cam: CameraInfo, cam_id: int) -> np.ndarray:
        """
        批量计算畸变分数（性能优化）
        
        Args:
            points: (N, 2) 场地坐标点数组
            cam: 相机信息对象
            cam_id: 相机ID
            
        Returns:
            scores: (N,) 畸变分数数组
        """
        N = len(points)
        
        # 检查是否有畸变
        distortion_info = self.distortion_cache.get(cam_id, {})
        if not distortion_info.get('has_distortion', False):
            return np.ones(N, dtype=np.float32)
        
        # 批量投影到像素坐标
        try:
            camera_obj = self.camera_objects_map[cam_id]
            pixel_coords = camera_obj.pitch_to_pixel(points)  # (N, 2)
        except Exception as e:
            # 投影失败，返回低分
            return np.zeros(N, dtype=np.float32)
        
        # 检查投影点是否在画面内（严格遵守 camera.py 的视野逻辑）
        h, w = camera_obj.frame_shape
        in_frame_mask = (
            (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < w) &
            (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < h) &
            np.isfinite(pixel_coords[:, 0]) & np.isfinite(pixel_coords[:, 1])
        )
        
        # 初始化分数数组（视野外的点默认为0）
        scores = np.zeros(N, dtype=np.float32)
        
        # 只对视野内的点计算畸变分数
        if not np.any(in_frame_mask):
            return scores
        
        # 批量计算到图像中心的距离（只计算视野内的点）
        image_center = distortion_info['image_center']
        pixel_coords_in_frame = pixel_coords[in_frame_mask]
        r_pixels = np.linalg.norm(pixel_coords_in_frame - image_center, axis=1)  # (M,) M为视野内点数
        
        # 批量计算归一化半径
        r_max = distortion_info['r_max']
        r_norms = r_pixels / r_max
        
        # 批量计算Brown-Conrady畸变量（使用MCSScoreModel）
        k1 = distortion_info['k1']
        k2 = distortion_info['k2']
        k3 = distortion_info['k3']
        
        distortion_magnitudes = MCSScoreModel.brown_conrady_distortion_magnitude(r_norms, k1, k2, k3)
        
        # 将畸变量转换为分数（使用MCSScoreModel）
        scores_in_frame = MCSScoreModel.distortion_to_score(
            distortion_magnitudes, 
            self.config.distortion_scale_factor
        )
        
        # 将视野内的分数填回结果数组
        scores[in_frame_mask] = scores_in_frame
        
        return scores

    def _calculate_score(self, point: np.ndarray, cam: CameraInfo) -> float:
        """
        计算单个相机对单个点的监看质量分数
        
        Score = S_dist × S_distortion
        
        设计理念：
        - 距离分数：Sigmoid平滑衰减（避免硬边界）
        - 畸变分数：基于Brown-Conrady物理模型，在像素空间计算
        - 组合方式：乘法（任一因素差则总分差）
        
        目标：让质量好的相机优先覆盖核心区域，畸变影响符合物理规律
        
        注意：此函数仅用于在可见网格内计算相对质量分数，
        不应该再次进行可见性过滤（已由 visible_grid_cells 保证）
        """
        # 1. Sigmoid 距离分数（使用MCSScoreModel）
        dist = np.linalg.norm(point - cam.position)
        score_dist = MCSScoreModel.sigmoid_distance_score(
            dist,
            cam.max_effective_dist,
            self.config.distance_half_point_ratio,
            self.config.distance_steepness_factor
        )
        
        # 2. 物理畸变分数（像素空间，基于Brown-Conrady模型）
        score_distortion = self._calculate_distortion_score(point, cam)
        
        # 3. 乘法组合（任一因素差则总分差）
        final_score = score_dist * score_distortion
        
        return max(0.0, min(1.0, final_score))  # 确保在 [0, 1] 范围
    
    def _calculate_distortion_score(self, point: np.ndarray, cam: CameraInfo) -> float:
        """
        计算畸变质量分数（基于Brown-Conrady物理模型）
        
        设计理念：
        - 在像素空间计算畸变量（符合畸变物理本质）
        - 使用Brown-Conrady模型：Δr = r × (k₁r² + k₂r⁴ + k₃r⁶)
        - 畸变量越大，分数越低（有界衰减函数）
        
        物理意义：
        - 图像中心：畸变量 ≈ 0 → 分数 ≈ 1.0
        - 图像边缘：畸变量大 → 分数低
        - k₁、k₂、k₃的作用方式与真实畸变一致
        
        Args:
            point: 场地坐标点 (x, y) cm
            cam: 相机信息对象
            
        Returns:
            畸变分数 (0-1)
            - 1.0: 无畸变或畸变极小（图像中心）
            - 0.5: 中等畸变
            - 0.0: 畸变极大（图像边缘）
        """
        cam_id = cam.id  # 假设 CameraInfo 有 id 属性
        # 如果没有，需要通过其他方式获取（如传入numeric_id）
        # 临时方案：通过遍历查找
        cam_numeric_id = None
        for nid, cinfo in self.cameras.items():
            if cinfo is cam:
                cam_numeric_id = nid
                break
        
        if cam_numeric_id is None:
            # 无法找到对应相机，返回默认分数
            return 1.0
        
        # 检查是否有畸变
        distortion_info = self.distortion_cache.get(cam_numeric_id, {})
        if not distortion_info.get('has_distortion', False):
            # 无畸变相机，所有位置分数都为1
            return 1.0
        
        # 步骤1: 将场地点投影到像素坐标
        try:
            camera_obj = self.camera_objects_map[cam_numeric_id]
            pixel_coords = camera_obj.pitch_to_pixel(np.array([point]))[0]
        except Exception as e:
            # 投影失败（可能点在视野外），返回低分
            return 0.0
        
        # 步骤1.5: 检查投影点是否在画面内（严格遵守 camera.py 的视野逻辑）
        h, w = camera_obj.frame_shape
        x, y = pixel_coords[0], pixel_coords[1]
        
        # 检查是否为有效值（不是 NaN 或 Inf）
        if not (np.isfinite(x) and np.isfinite(y)):
            return 0.0
        
        # 检查是否在画面内（与 camera.compute_visible_grids 完全一致）
        if not (0 <= x < w and 0 <= y < h):
            return 0.0
        
        # 步骤2: 计算到图像中心的距离
        image_center = distortion_info['image_center']
        r_pixel = np.linalg.norm(pixel_coords - image_center)
        
        # 步骤3: 计算归一化半径 [0, 1]
        r_max = distortion_info['r_max']
        r_norm = r_pixel / r_max
        
        # 步骤4: 使用Brown-Conrady模型计算畸变量（使用MCSScoreModel）
        k1 = distortion_info['k1']
        k2 = distortion_info['k2']
        k3 = distortion_info['k3']
        
        distortion_magnitude = MCSScoreModel.brown_conrady_distortion_magnitude(r_norm, k1, k2, k3)
        
        # 步骤5: 将畸变量转换为分数（使用MCSScoreModel）
        score = MCSScoreModel.distortion_to_score(
            distortion_magnitude,
            self.config.distortion_scale_factor
        )
        
        return score

    def get_grid_score(self, pitch_position: np.ndarray, camera_id: int) -> float:
        """
        查询指定位置在指定相机视角下的质量分数
        
        Args:
            pitch_position: 球场坐标 (x, y) cm
            camera_id: 相机ID（numeric_id）
            
        Returns:
            质量分数 [0, 1]，如果网格不在视野内则返回0
        """
        if not hasattr(self, 'grid_system') or self.grid_system is None:
            print("[MultiCameraScheduler] Warning: Grid system not initialized")
            return 0.0
        
        if camera_id not in self.score_maps:
            print(f"[MultiCameraScheduler] Warning: No score map for camera {camera_id}")
            return 0.0
        
        # 将球场坐标转换为网格索引
        row, col = self.grid_system.physical_to_index(pitch_position)
        
        # 检查索引是否在范围内
        score_map = self.score_maps[camera_id]
        rows, cols = score_map.shape
        
        if not (0 <= row < rows and 0 <= col < cols):
            return 0.0
        
        # 返回该网格的质量分数
        return float(score_map[row, col])
    
    def get_best_camera_for_position(self, pitch_position: np.ndarray) -> tuple:
        """
        查询指定位置的最佳监看相机
        
        Args:
            pitch_position: 球场坐标 (x, y) cm
            
        Returns:
            (camera_id, quality_score) 元组
        """
        if not self.score_maps:
            return (0, 0.0)
        
        best_cam_id = 0
        best_score = 0.0
        
        for cam_id in self.score_maps.keys():
            score = self.get_grid_score(pitch_position, cam_id)
            if score > best_score:
                best_score = score
                best_cam_id = cam_id
        
        return (best_cam_id, best_score)

    def compute_coverage(self):
        """
        计算各相机的网格分数图
        
        不再生成分配图，只计算每个相机对每个网格的质量分数。
        分数存储在 self.score_maps 中，供检测时查询使用。
        
        使用 Camera 初始化时计算的 visible_grid_cells，
        只对实际可见的网格计算分数，避免使用简化的锥形模型。
        """
        from grid_system import PitchGridSystem
        
        pitch = self.config.pitch_config
        w, h = pitch.length, pitch.width # 注意: soccer.py定义 length是长(x), width是宽(y)
        
        # 创建统一的网格系统（保存为实例属性供 Detection 查询使用）
        self.grid_system = PitchGridSystem(w, h, self.config.grid_size)
        rows, cols = self.grid_system.get_shape()
        
        print(f"\n{'='*70}")
        print(f"[MCS] 计算网格质量分数（Brown-Conrady物理模型）")
        print(f"{'='*70}")
        print(f"网格系统: {rows}行 × {cols}列")
        print(f"\n模型参数:")
        print(f"  距离分数 (Sigmoid):")
        print(f"    - 半衰点比例: {self.config.distance_half_point_ratio:.2f}")
        print(f"    - 陡峭度因子: {self.config.distance_steepness_factor:.1f}")
        print(f"  畸变分数 (Brown-Conrady物理模型):")
        print(f"    - 畸变量缩放系数: {self.config.distortion_scale_factor:.1f}")
        print(f"  组合方式: 乘法 (Q = S_dist × S_distortion)")
        print(f"\n相机畸变信息:")
        for cam_id, cam_info in self.cameras.items():
            distortion_info = self.distortion_cache.get(cam_id, {})
            camera_obj = self.camera_objects_map.get(cam_id)
            
            if distortion_info.get('has_distortion', False):
                print(f"  相机 {cam_id}:")
                print(f"    畸变系数: k1={distortion_info['k1']:.4f}, "
                      f"k2={distortion_info['k2']:.4f}, k3={distortion_info['k3']:.4f}")
                if camera_obj:
                    h, w = camera_obj.frame_shape
                    print(f"    图像尺寸: {int(w)}×{int(h)} px")
                print(f"    归一化半径: {distortion_info['r_max']:.1f} px")
            else:
                print(f"  相机 {cam_id}: 无畸变参数（理想相机）")
        print(f"{'='*70}\n")
        
        # 遍历每个相机计算分数图
        for cam_id, cam_info in self.cameras.items():
            print(f"Computing scores for Camera {cam_id}...")
            
            # 检查是否有预计算的可见网格
            if cam_info.visible_grid_cells is not None and len(cam_info.visible_grid_cells) > 0:
                print(f"  使用预计算的可见网格: {len(cam_info.visible_grid_cells)} 个网格")
                self._compute_score_map_from_visible_grids(
                    cam_id, cam_info, self.grid_system, rows, cols
                )
            else:
                # 后备方案：使用简化的锥形模型（旧方法）
                print(f"  警告: 没有可见网格信息，使用简化模型")
                self._compute_score_map_legacy(
                    cam_id, cam_info, self.grid_system, rows, cols
                )
        
        print(f"\n{'='*70}")
        print(f"[MCS] 网格分数计算完成")
        print(f"  - 相机数: {len(self.score_maps)}")
        print(f"  - 网格系统: {rows}行 × {cols}列")
        print(f"{'='*70}\n")
    
    def _compute_score_map_from_visible_grids(self, cam_id, cam_info, grid_system,
                                              rows, cols):
        """
        基于预计算的可见网格计算分数图（批量优化）
        
        只对 visible_grid_cells 中的网格计算分数
        使用批量计算提升性能（相比逐点计算快10-50倍）
        """
        cx, cy = cam_info.position
        
        # 只对可见网格计算分数
        visible_cells = cam_info.visible_grid_cells  # (N, 2) [[row, col], ...]
        
        if len(visible_cells) == 0:
            self.score_maps[cam_id] = np.zeros((rows, cols), dtype=np.float32)
            return
        
        # 初始化当前相机的分数图
        total_score = np.zeros((rows, cols), dtype=np.float32)
        
        # 批量计算优化
        # 1. 收集所有网格中心点
        points = np.array([grid_system.index_to_center(row, col) 
                          for row, col in visible_cells])  # (N, 2)
        
        # 2. 批量计算分数
        scores = self._calculate_scores_batch(points, cam_info, cam_id)
        
        # 3. 填充分数图
        for i, (row, col) in enumerate(visible_cells):
            total_score[row, col] = scores[i]
        
        self.score_maps[cam_id] = total_score
        
        # 统计信息
        valid_count = np.sum(total_score > 0)
        max_score = np.max(total_score) if valid_count > 0 else 0
        print(f"  > Cam {cam_id}: 可见网格={len(visible_cells)}, 有效分数={valid_count}, 最大分数={max_score:.4f}")
        
    def _compute_score_map_legacy(self, cam_id, cam_info, grid_system, rows, cols):
        """
        旧的简化锥形模型（后备方案）
        
        当相机没有预计算的可见网格时使用。
        """
        cx, cy = cam_info.position
        
        # 获取网格中心坐标的 meshgrid
        grid_x_corners, grid_y_corners = grid_system.get_meshgrid_corners()
        grid_x_center = grid_x_corners + self.config.grid_size / 2
        grid_y_center = grid_y_corners + self.config.grid_size / 2
        
        # 1. 距离矩阵
        dist_map = np.sqrt((grid_x_center - cx)**2 + (grid_y_center - cy)**2)
        
        # 2. 角度矩阵
        vec_x = grid_x_center - cx
        vec_y = grid_y_center - cy
        norms = np.sqrt(vec_x**2 + vec_y**2)
        norms[norms == 0] = 1e-6
        vec_x /= norms
        vec_y /= norms
        
        rad = np.radians(cam_info.orientation)
        cam_dir_x = np.cos(rad)
        cam_dir_y = np.sin(rad)
        cos_theta = vec_x * cam_dir_x + vec_y * cam_dir_y
        
        # 3. 计算分数
        s_dist = 1.0 - (dist_map / cam_info.max_effective_dist)
        s_dist[s_dist < 0] = 0
        s_angle = cos_theta
        
        half_fov_rad = np.radians(cam_info.angle_span / 2)
        min_cos = np.cos(half_fov_rad)
        mask = (cos_theta >= min_cos) & (dist_map <= cam_info.max_effective_dist)
        
        total_score = (self.config.weight_dist * s_dist + 
                       self.config.weight_angle * s_angle) * mask
        
        self.score_maps[cam_id] = total_score
        
        valid_count = np.sum(mask)
        max_score = np.max(total_score)
        print(f"  > Cam {cam_id}: 有效像素={valid_count}, 最大分数={max_score:.4f}")
        
    def _compute_score_grids(self, cam_id, cam_info, points, resolution):
        """
        计算单个相机的三种分数网格（可复用的核心计算逻辑）
        
        Args:
            cam_id: 相机ID
            cam_info: 相机信息对象
            points: (N, 2) 采样点数组
            resolution: 网格分辨率
            
        Returns:
            (distance_grid, distortion_grid, combined_grid, X, Y)
        """
        # 批量计算距离分数
        distances = np.linalg.norm(points - cam_info.position, axis=1)
        valid_mask = distances <= cam_info.max_effective_dist
        
        D_half = self.config.distance_half_point_ratio * cam_info.max_effective_dist
        k_dist = self.config.distance_steepness_factor / cam_info.max_effective_dist
        
        exponent_dist = k_dist * (distances - D_half)
        exponent_dist = np.clip(exponent_dist, -20.0, 20.0)
        distance_scores = 1.0 / (1.0 + np.exp(exponent_dist))
        distance_scores[~valid_mask] = 0.0
        
        # 批量计算畸变分数
        distortion_scores = self._calculate_distortion_scores_batch(points, cam_info, cam_id)
        
        # 综合分数
        combined_scores = distance_scores * distortion_scores
        
        # 重塑为2D网格
        distance_grid = distance_scores.reshape(resolution, resolution)
        distortion_grid = distortion_scores.reshape(resolution, resolution)
        combined_grid = combined_scores.reshape(resolution, resolution)
        
        return distance_grid, distortion_grid, combined_grid

    def _visualize_scores_2d(self, cam_id, cam_info, distance_grid, distortion_grid, 
                             combined_grid, x_min, x_max, y_min, y_max, save_path):
        """
        生成2D热力图可视化（3个子图）- 带球场背景
        
        Args:
            cam_id: 相机ID
            cam_info: 相机信息
            distance_grid: 距离分数网格
            distortion_grid: 畸变分数网格
            combined_grid: 综合分数网格
            x_min, x_max, y_min, y_max: 场地范围（cm）
            save_path: 保存路径
        """
        from matplotlib.colors import LinearSegmentedColormap
        
        # 获取球场配置
        pitch = self.config.pitch_config
        length_m = pitch.length / 100  # 转为米
        width_m = pitch.width / 100
        
        # 从 MCS 配置获取相机颜色（BGR格式）
        bgr = self.config.CAMERA_COLORS.get(cam_id, (255, 255, 255))
        # BGR -> RGB，归一化到 [0, 1]
        camera_color_rgb = np.array([bgr[2], bgr[1], bgr[0]]) / 255.0
        
        # 创建自定义 colormap：从黑色渐变到相机颜色
        colors_list = [(0, 0, 0), tuple(camera_color_rgb)]  # 黑色 -> 相机颜色
        n_bins = 256
        cmap_name = f'camera_{cam_id}_cmap'
        camera_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_list, N=n_bins)
        
        # 判断是否为半场
        from soccer import HalfPitchConfiguration, HalfQingYuanConfiguration, HalfWuChuanConfiguration
        is_half_pitch = isinstance(pitch, (HalfPitchConfiguration, HalfQingYuanConfiguration, HalfWuChuanConfiguration))
        
        # 从 config.edges 和 config.vertices 提取球场线条
        pitch_lines = []
        for start, end in pitch.edges:
            x1, y1 = pitch.vertices[start - 1][0] / 100, pitch.vertices[start - 1][1] / 100
            x2, y2 = pitch.vertices[end - 1][0] / 100, pitch.vertices[end - 1][1] / 100
            pitch_lines.append(([x1, x2], [y1, y2]))
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 绘制三个子图
        grids = [distance_grid, distortion_grid, combined_grid]
        titles = ['Distance Score (Sigmoid)', 'Distortion Score (Brown-Conrady)', 'Combined Score (Dist × Distortion)']
        
        for idx, (ax, grid, title) in enumerate(zip(axes, grids, titles)):
            # 1. 设置白色背景
            ax.set_facecolor('white')
            
            # 2. 叠加热力图（图像坐标系，使用相机颜色）
            im = ax.imshow(grid, origin='upper', 
                          extent=[x_min/100, x_max/100, y_max/100, y_min/100],
                          cmap=camera_cmap, vmin=0, vmax=1, aspect='equal', zorder=2)
            
            # 3. 绘制球场所有线条（深灰色）
            for line_x, line_y in pitch_lines:
                ax.plot(line_x, line_y, '#333333', linewidth=1.5, alpha=0.8, zorder=3)
            
            # 4. 绘制中圈
            circle_radius = pitch.centre_circle_radius / 100  # 转为米
            if is_half_pitch:
                # 半场：圆心在右边界中点，绘制左半圆
                theta = np.linspace(np.pi/2, 3*np.pi/2, 50)
                circle_x = length_m + circle_radius * np.cos(theta)
                circle_y = width_m/2 + circle_radius * np.sin(theta)
                ax.plot(circle_x, circle_y, '#333333', linewidth=1.5, alpha=0.8, zorder=3)
            else:
                # 全场：圆心在场地中心，绘制完整圆
                circle = plt.Circle((length_m/2, width_m/2), circle_radius, 
                                   fill=False, color='#333333', linewidth=1.5, alpha=0.8, zorder=3)
                ax.add_patch(circle)
            
            # 5. 绘制点球点
            penalty_spot_dist = pitch.penalty_spot_distance / 100
            ax.plot(penalty_spot_dist, width_m/2, 'o', color='#333333', markersize=4, alpha=0.8, zorder=3)
            if not is_half_pitch:
                ax.plot(length_m - penalty_spot_dist, width_m/2, 'o', color='#333333', markersize=4, alpha=0.8, zorder=3)
            
            # 6. 标题和标签
            ax.set_title(f'{title}\nCam {cam_id}', fontsize=14, fontweight='bold')
            ax.set_xlabel('X (m) →', fontsize=12)
            ax.set_ylabel('Y (m) ↓', fontsize=12)
            
            # 7. 相机位置（最上层）
            ax.plot(cam_info.position[0]/100, cam_info.position[1]/100, 
                   'c*', markersize=20, markeredgecolor='black', markeredgewidth=2, zorder=10)
            
            # 8. 颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            
            # 9. 设置坐标轴范围（图像坐标系：Y从上到下）
            ax.set_xlim(0, length_m)
            ax.set_ylim(width_m, 0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _visualize_scores_3d(self, cam_id, cam_info, distance_grid, distortion_grid, 
                             combined_grid, X, Y, save_path):
        """
        生成3D连续曲面图可视化（3个子图）- 带球场轮廓
        
        Args:
            cam_id: 相机ID
            cam_info: 相机信息
            distance_grid: 距离分数网格
            distortion_grid: 畸变分数网格
            combined_grid: 综合分数网格
            X, Y: 网格坐标矩阵（cm）
            save_path: 保存路径
        """
        from matplotlib.colors import LinearSegmentedColormap
        
        # 获取球场配置
        pitch = self.config.pitch_config
        length_m = pitch.length / 100
        width_m = pitch.width / 100
        
        # 从 MCS 配置获取相机颜色（BGR格式）
        bgr = self.config.CAMERA_COLORS.get(cam_id, (255, 255, 255))
        # BGR -> RGB，归一化到 [0, 1]
        camera_color_rgb = np.array([bgr[2], bgr[1], bgr[0]]) / 255.0
        
        # 创建自定义 colormap：从黑色渐变到相机颜色
        colors_list = [(0, 0, 0), tuple(camera_color_rgb)]  # 黑色 -> 相机颜色
        n_bins = 256
        cmap_name = f'camera_{cam_id}_cmap_3d'
        camera_cmap = LinearSegmentedColormap.from_list(cmap_name, colors_list, N=n_bins)
        
        # 判断是否为半场
        from soccer import HalfPitchConfiguration, HalfQingYuanConfiguration, HalfWuChuanConfiguration
        is_half_pitch = isinstance(pitch, (HalfPitchConfiguration, HalfQingYuanConfiguration, HalfWuChuanConfiguration))
        
        fig = plt.figure(figsize=(20, 6))
        
        # 转换为米
        X_m = X / 100
        Y_m = Y / 100
        cam_x_m = cam_info.position[0] / 100
        cam_y_m = cam_info.position[1] / 100
        
        # 从 config.edges 和 config.vertices 提取球场线条（底部 z=0）
        pitch_lines = []
        for start, end in pitch.edges:
            x1, y1 = pitch.vertices[start - 1][0] / 100, pitch.vertices[start - 1][1] / 100
            x2, y2 = pitch.vertices[end - 1][0] / 100, pitch.vertices[end - 1][1] / 100
            pitch_lines.append(([x1, x2], [y1, y2]))
        
        # 绘制三个子图
        grids = [distance_grid, distortion_grid, combined_grid]
        titles = ['Distance Score (Sigmoid)', 'Distortion Score (Brown-Conrady)', 'Combined Score (Dist × Distortion)']
        
        for idx, (grid, title) in enumerate(zip(grids, titles)):
            ax = fig.add_subplot(1, 3, idx+1, projection='3d')
            
            # 1. 绘制分数曲面（图像坐标系：Y向下，使用相机颜色）
            surf = ax.plot_surface(X_m, Y_m, grid, cmap=camera_cmap, 
                                   linewidth=0, antialiased=True, alpha=0.9,
                                   vmin=0, vmax=1)
            
            # 2. 绘制球场所有线条（底部 z=0）
            for line_x, line_y in pitch_lines:
                ax.plot(line_x, line_y, [0, 0], 'white', linewidth=1.5, alpha=0.6, zorder=5)
            
            # 3. 绘制中圈（底部 z=0）
            circle_radius = pitch.centre_circle_radius / 100  # 转为米
            if is_half_pitch:
                # 半场：圆心在右边界中点，绘制左半圆
                theta = np.linspace(np.pi/2, 3*np.pi/2, 50)
                circle_x = length_m + circle_radius * np.cos(theta)
                circle_y = width_m/2 + circle_radius * np.sin(theta)
            else:
                # 全场：圆心在场地中心，绘制完整圆
                theta = np.linspace(0, 2*np.pi, 100)
                circle_x = length_m/2 + circle_radius * np.cos(theta)
                circle_y = width_m/2 + circle_radius * np.sin(theta)
            circle_z = np.zeros_like(circle_x)
            ax.plot(circle_x, circle_y, circle_z, 'white', linewidth=1.5, alpha=0.6, zorder=5)
            
            # 4. 设置标题和标签（图像坐标系）
            ax.set_title(f'{title}\nCam {cam_id}', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('X (m) →', fontsize=11, labelpad=10)
            ax.set_ylabel('Y (m) ↓', fontsize=11, labelpad=10)
            ax.set_zlabel('Score', fontsize=11, labelpad=10)
            
            # 5. 设置坐标轴范围（Y轴反向：图像坐标系）
            ax.set_xlim(0, length_m)
            ax.set_ylim(width_m, 0)  # Y轴反向
            ax.set_zlim(0, 1)
            
            # 6. 设置视角（调整以适应Y轴反向）
            ax.view_init(elev=25, azim=-135)  # 调整方位角以保持相似的观察视角
            
            # 7. 标记相机位置（顶部）
            ax.scatter([cam_x_m], [cam_y_m], [1.0], c='cyan', marker='*', s=300, 
                      edgecolors='white', linewidths=2, zorder=10)
            
            # 8. 颜色条
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.1)
            
            # 9. 网格
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _visualize_all_cameras_2d(self, all_scores, X, Y, x_min, x_max, y_min, y_max, save_dir):
        """
        生成所有相机的综合2D可视化（使用RGB混合显示分数分布）
        
        使用RGB通道混合：每个相机的分数作为权重，混合各相机的颜色。
        交界处会自然显示混合色，直观展示分配情况和竞争关系。
        
        Args:
            all_scores: 所有相机的分数字典 {cam_id: {'distance': grid, 'distortion': grid, 'combined': grid}}
            X, Y: 网格坐标矩阵（cm）
            x_min, x_max, y_min, y_max: 场地范围（cm）
            save_dir: 保存目录
            
        Returns:
            生成的图像
        """
        pitch = self.config.pitch_config
        length_m = pitch.length / 100
        width_m = pitch.width / 100
        
        # 获取分辨率
        resolution = list(all_scores.values())[0]['combined'].shape[0]
        
        # 从配置获取相机颜色（BGR格式）并转为RGB
        camera_colors_rgb = {}
        for cam_id in all_scores.keys():
            bgr = self.config.CAMERA_COLORS.get(cam_id, (255, 255, 255))
            # BGR -> RGB，归一化到 [0, 1]
            camera_colors_rgb[cam_id] = np.array([bgr[2], bgr[1], bgr[0]]) / 255.0
        
        # 收集所有相机的分数
        score_maps = {}
        for cam_id, scores in all_scores.items():
            score_maps[cam_id] = scores['combined']
        
        # 创建RGB混合图像
        rgb_image = np.zeros((resolution, resolution, 3))
        total_scores = np.zeros((resolution, resolution))
        camera_count = np.zeros((resolution, resolution))  # 记录每个网格有几个相机覆盖
        
        # 计算总分数和相机数量
        for cam_id, score_map in score_maps.items():
            total_scores += score_map
            # 分数大于阈值则认为该相机覆盖此网格
            camera_count += (score_map > 1e-6).astype(float)
        
        # 计算平均分数（用于透明度）
        # average_score(x,y) = total_score(x,y) / count(x,y)
        avg_scores = np.divide(total_scores, camera_count,
                              out=np.zeros_like(total_scores),
                              where=camera_count > 0)
        
        # RGB混合：每个网格按各相机分数占比为颜色权重
        # 公式：color(x,y) = Σ[camera_i_score(x,y) / total_score(x,y) × camera_i_color]
        for cam_id, score_map in score_maps.items():
            color = camera_colors_rgb[cam_id]
            # 计算该相机在每个点的分数占比（权重）
            # weight(x,y) = score_i(x,y) / Σscore_j(x,y)
            weight = np.divide(score_map, total_scores, 
                             out=np.zeros_like(score_map), 
                             where=total_scores > 1e-6)
            # 按权重混合RGB通道
            for i in range(3):  # R, G, B
                rgb_image[:, :, i] += weight * color[i]
        
        # 计算alpha通道：使用平均分数（而非总分数）
        # alpha表示该点所有覆盖相机的平均质量
        max_avg_score = np.nanmax(avg_scores) if np.any(~np.isnan(avg_scores)) else 0.0
        if max_avg_score > 1e-6:
            # 归一化到 [0, 1]
            normalized_scores = np.clip(avg_scores / max_avg_score, 0, 1)
            
            # 应用非线性映射和阈值，使低分数区域更透明
            # 1. 设置阈值：分数低于20%的区域完全透明
            score_threshold = 0.2
            alpha = np.where(normalized_scores < score_threshold, 0, normalized_scores)
            
            # 2. 对保留区域应用幂函数映射，增强对比度
            # 这样中等分数(0.5)会变成0.35，高分数(1.0)保持1.0
            alpha = np.power(alpha, 1.5)
        else:
            # 如果所有分数都是0，完全透明
            alpha = np.zeros_like(avg_scores)
        # 清理可能的 nan/inf 值
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 提取球场线条
        from soccer import HalfPitchConfiguration, HalfQingYuanConfiguration, HalfWuChuanConfiguration
        is_half_pitch = isinstance(pitch, (HalfPitchConfiguration, HalfQingYuanConfiguration, HalfWuChuanConfiguration))
        
        pitch_lines = []
        for start, end in pitch.edges:
            x1, y1 = pitch.vertices[start - 1][0] / 100, pitch.vertices[start - 1][1] / 100
            x2, y2 = pitch.vertices[end - 1][0] / 100, pitch.vertices[end - 1][1] / 100
            pitch_lines.append(([x1, x2], [y1, y2]))
        
        # 绘制综合图（透明背景，只保留标线）
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        # 透明背景
        fig.patch.set_alpha(0.0)
        ax.set_facecolor('none')
        
        # 显示相机覆盖区域（使用RGB图像+alpha）
        ax.imshow(rgb_image, origin='upper',
                 extent=[x_min/100, x_max/100, y_max/100, y_min/100],
                 aspect='equal', alpha=alpha, zorder=2)
        
        # 绘制球场线条（白色，清晰可见）
        for line_x, line_y in pitch_lines:
            ax.plot(line_x, line_y, 'white', linewidth=2, alpha=1.0, zorder=3)
        
        # 绘制中圈
        circle_radius = pitch.centre_circle_radius / 100
        if is_half_pitch:
            theta = np.linspace(np.pi/2, 3*np.pi/2, 50)
            circle_x = length_m + circle_radius * np.cos(theta)
            circle_y = width_m/2 + circle_radius * np.sin(theta)
            ax.plot(circle_x, circle_y, 'white', linewidth=2, alpha=1.0, zorder=3)
        else:
            circle = plt.Circle((length_m/2, width_m/2), circle_radius,
                               fill=False, color='white', linewidth=2, alpha=1.0, zorder=3)
            ax.add_patch(circle)
        
        # 绘制点球点
        penalty_spot_dist = pitch.penalty_spot_distance / 100
        ax.plot(penalty_spot_dist, width_m/2, 'o', color='white', markersize=6, alpha=1.0, zorder=3)
        if not is_half_pitch:
            ax.plot(length_m - penalty_spot_dist, width_m/2, 'o', color='white', markersize=6, alpha=1.0, zorder=3)
        
        # 标记所有相机位置
        for cam_id, scores in all_scores.items():
            cam_info = scores['cam_info']
            color = camera_colors_rgb[cam_id]
            ax.plot(cam_info.position[0]/100, cam_info.position[1]/100,
                   '*', color=color, markersize=20, markeredgecolor='white',
                   markeredgewidth=2, zorder=10)
        
        # 设置坐标轴范围（不显示标签和标题）
        ax.set_xlim(0, length_m)
        ax.set_ylim(width_m, 0)
        ax.axis('off')  # 关闭坐标轴
        
        plt.tight_layout(pad=0)
        output_path = save_dir / "all_cameras_combined_2d.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   transparent=True, pad_inches=0)
        print(f"  ✓ 所有相机综合2D图: {output_path}")
        
        img = cv2.imread(str(output_path), cv2.IMREAD_UNCHANGED)  # 读取包含alpha通道的PNG
        plt.close(fig)
        return img

    def _visualize_all_cameras_3d(self, all_scores, X, Y, save_dir):
        """
        生成所有相机的综合3D可视化（显示每个位置的最优相机）
        
        Args:
            all_scores: 所有相机的分数字典
            X, Y: 网格坐标矩阵（cm）
            save_dir: 保存目录
            
        Returns:
            生成的图像
        """
        pitch = self.config.pitch_config
        length_m = pitch.length / 100
        width_m = pitch.width / 100
        
        # 为每个相机分配颜色
        camera_colors = plt.cm.tab10(np.linspace(0, 1, len(all_scores)))
        
        # 获取分辨率
        resolution = list(all_scores.values())[0]['combined'].shape[0]
        
        # 创建最优相机分配图和最大分数图
        best_cam_map = np.zeros((resolution, resolution), dtype=int)
        max_scores = np.zeros((resolution, resolution))
        
        for cam_id, scores in all_scores.items():
            combined = scores['combined']
            mask = combined > max_scores
            best_cam_map[mask] = cam_id
            max_scores[mask] = combined[mask]
        
        # 创建RGB颜色数组
        colors = np.zeros((resolution, resolution, 4))
        for idx, (cam_id, scores) in enumerate(all_scores.items()):
            mask = best_cam_map == cam_id
            colors[mask] = camera_colors[idx]
        
        # 提取球场线条
        from soccer import HalfPitchConfiguration, HalfQingYuanConfiguration, HalfWuChuanConfiguration
        is_half_pitch = isinstance(pitch, (HalfPitchConfiguration, HalfQingYuanConfiguration, HalfWuChuanConfiguration))
        
        pitch_lines = []
        for start, end in pitch.edges:
            x1, y1 = pitch.vertices[start - 1][0] / 100, pitch.vertices[start - 1][1] / 100
            x2, y2 = pitch.vertices[end - 1][0] / 100, pitch.vertices[end - 1][1] / 100
            pitch_lines.append(([x1, x2], [y1, y2]))
        
        # 绘制3D图
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # 转换为米
        X_m = X / 100
        Y_m = Y / 100
        
        # 绘制分数曲面（使用相机颜色）
        surf = ax.plot_surface(X_m, Y_m, max_scores, facecolors=colors,
                              linewidth=0, antialiased=True, alpha=0.9, shade=False)
        
        # 绘制球场线条（底部）
        for line_x, line_y in pitch_lines:
            ax.plot(line_x, line_y, [0, 0], 'white', linewidth=1.5, alpha=0.6, zorder=5)
        
        # 绘制中圈
        circle_radius = pitch.centre_circle_radius / 100
        if is_half_pitch:
            theta = np.linspace(np.pi/2, 3*np.pi/2, 50)
            circle_x = length_m + circle_radius * np.cos(theta)
            circle_y = width_m/2 + circle_radius * np.sin(theta)
        else:
            theta = np.linspace(0, 2*np.pi, 100)
            circle_x = length_m/2 + circle_radius * np.cos(theta)
            circle_y = width_m/2 + circle_radius * np.sin(theta)
        circle_z = np.zeros_like(circle_x)
        ax.plot(circle_x, circle_y, circle_z, 'white', linewidth=1.5, alpha=0.6, zorder=5)
        
        # 标记所有相机位置
        for idx, (cam_id, scores) in enumerate(all_scores.items()):
            cam_info = scores['cam_info']
            color = camera_colors[idx]
            ax.scatter([cam_info.position[0]/100], [cam_info.position[1]/100], [1.0],
                      c=[color], marker='*', s=300, edgecolors='white',
                      linewidths=2, zorder=10, label=f'Camera {cam_id}')
        
        # 设置标题和标签
        ax.set_title('All Cameras Combined - Best Coverage Area (3D)\n(Each color represents optimal camera)',
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('X (m) →', fontsize=11, labelpad=10)
        ax.set_ylabel('Y (m) ↓', fontsize=11, labelpad=10)
        ax.set_zlabel('Score', fontsize=11, labelpad=10)
        
        # 设置坐标轴范围
        ax.set_xlim(0, length_m)
        ax.set_ylim(width_m, 0)
        ax.set_zlim(0, 1)
        
        # 设置视角
        ax.view_init(elev=25, azim=-135)
        
        # 图例
        ax.legend(loc='upper right', fontsize=10)
        
        # 网格
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = save_dir / "all_cameras_combined_3d.jpg"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ 所有相机综合3D图: {output_path}")
        
        img = cv2.imread(str(output_path))
        plt.close(fig)
        return img

    def visualize_score_functions(self, resolution=100, save_dir=None, mode='both'):
        """
        可视化分数函数：为每个相机在场地坐标空间生成分数可视化
        
        生成内容：
        1. 每个相机的单独可视化（3列：距离、畸变、综合）- 用于合成大图
        2. 所有相机的综合可视化（显示每个位置的最优相机）
        
        Args:
            resolution: 采样分辨率（每个维度的采样点数）
            save_dir: 保存目录，默认为 store/{name}_mcs/score/
            mode: 可视化模式
                - '2d': 只生成2D热力图
                - '3d': 只生成3D曲面图
                - 'both': 生成2D和3D两种（默认）
            
        Returns:
            合成后的可视化图像字典 {'2d': img, '3d': img, '2d_combined': img, '3d_combined': img}
        """
        print(f"\n{'='*70}")
        print(f"[MCS] 生成分数函数可视化（分辨率: {resolution}x{resolution}, 模式: {mode}）")
        print(f"{'='*70}")
        
        # 获取场地尺寸
        pitch = self.config.pitch_config
        x_min, x_max = 0, pitch.length  # cm
        y_min, y_max = 0, pitch.width   # cm
        
        # 创建采样网格
        x_samples = np.linspace(x_min, x_max, resolution)
        y_samples = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x_samples, y_samples)
        
        # 展平为 (N, 2) 点数组
        points = np.column_stack([X.ravel(), Y.ravel()])
        
        # 准备保存目录
        if save_dir is None:
            save_dir = Path("store") / "score_visualization"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储所有相机的分数网格（用于综合可视化）
        all_scores = {}
        
        # 为每个相机生成可视化（临时图片用于合成）
        all_2d_images = []
        all_3d_images = []
        temp_files = []  # 记录临时文件以便删除
        
        for cam_id, cam_info in self.cameras.items():
            print(f"\n处理相机 {cam_id}...")
            
            # 计算分数网格（复用逻辑）
            distance_grid, distortion_grid, combined_grid = self._compute_score_grids(
                cam_id, cam_info, points, resolution
            )
            
            # 保存分数网格用于综合可视化
            all_scores[cam_id] = {
                'distance': distance_grid,
                'distortion': distortion_grid,
                'combined': combined_grid,
                'cam_info': cam_info
            }
            
            # 生成2D热力图（临时）
            if mode in ['2d', 'both']:
                output_2d = save_dir / f"_temp_camera{cam_id}_score_2d.png"
                temp_files.append(output_2d)
                self._visualize_scores_2d(
                    cam_id, cam_info, distance_grid, distortion_grid, combined_grid,
                    x_min, x_max, y_min, y_max, output_2d
                )
                img_2d = cv2.imread(str(output_2d))
                all_2d_images.append((f"Camera {cam_id}", [img_2d]))
            
            # 生成3D曲面图（临时）
            if mode in ['3d', 'both']:
                output_3d = save_dir / f"_temp_camera{cam_id}_score_3d.png"
                temp_files.append(output_3d)
                self._visualize_scores_3d(
                    cam_id, cam_info, distance_grid, distortion_grid, combined_grid,
                    X, Y, output_3d
                )
                img_3d = cv2.imread(str(output_3d))
                all_3d_images.append((f"Camera {cam_id}", [img_3d]))
        
        # 合成图片
        result = {}
        
        # 合成每个相机的单独可视化
        if mode in ['2d', 'both'] and len(all_2d_images) > 0:
            print(f"\n合成2D热力图...")
            combined_2d = create_combined_image_layout(
                images_data=all_2d_images,
                title=f"MCS Score Functions (2D Heatmap) - {len(self.cameras)} Cameras",
                layout='grid',
                label_position='left',
                target_height=600
            )
            combined_2d_path = save_dir / "all_cameras_score_2d.jpg"
            cv2.imwrite(str(combined_2d_path), combined_2d)
            print(f"  ✓ 2D合成图: {combined_2d_path}")
            result['2d'] = combined_2d
        
        if mode in ['3d', 'both'] and len(all_3d_images) > 0:
            print(f"\n合成3D曲面图...")
            combined_3d = create_combined_image_layout(
                images_data=all_3d_images,
                title=f"MCS Score Functions (3D Surface) - {len(self.cameras)} Cameras",
                layout='grid',
                label_position='left',
                target_height=600
            )
            combined_3d_path = save_dir / "all_cameras_score_3d.jpg"
            cv2.imwrite(str(combined_3d_path), combined_3d)
            print(f"  ✓ 3D合成图: {combined_3d_path}")
            result['3d'] = combined_3d
        
        # 生成所有相机的综合可视化
        print(f"\n生成所有相机的综合可视化...")
        if mode in ['2d', 'both']:
            combined_2d_img = self._visualize_all_cameras_2d(
                all_scores, X, Y, x_min, x_max, y_min, y_max, save_dir
            )
            result['2d_combined'] = combined_2d_img
        
        if mode in ['3d', 'both']:
            combined_3d_img = self._visualize_all_cameras_3d(
                all_scores, X, Y, save_dir
            )
            result['3d_combined'] = combined_3d_img
        
        # 删除临时文件
        print(f"\n清理临时文件...")
        for temp_file in temp_files:
            if temp_file.exists():
                temp_file.unlink()
        
        print(f"\n{'='*70}")
        print(f"[MCS] 分数函数可视化完成")
        print(f"{'='*70}\n")
        
        return result if result else None

class MCS:
    """
    Multi-Camera System Wrapper Class
    
    整合 MultiCameraScheduler 和 MCSSystemConfig，提供更简洁的对外接口。
    支持统一的训练区域管理。
    """
    def __init__(self, cameras: List[object], pitch_config: object, 
                 name: str = "default",
                 auto_setup_train_area: bool = True,
                 organize_files: bool = True):
        """
        Args:
            cameras: Camera对象列表
            pitch_config: 球场配置对象
            name: MCS 系统名称（用于文件夹命名）
            auto_setup_train_area: 是否自动设置训练区域（默认 True）
                                   如果为 True 且没有任何相机有训练区域，
                                   会提示用户选择一个相机进行设置
            organize_files: 是否组织文件（将 camera 文件夹移入 MCS 文件夹）
        """
        self.name = name
        self.config = MCSSystemConfig(pitch_config=pitch_config)
        self.scheduler = MultiCameraScheduler(cameras, self.config)
        self.pitch_config = pitch_config
        self.cameras = cameras  # 保存相机列表引用
        
        # 创建 MCS 存储目录
        self.store_dir = Path("store") / f"{name}_mcs"
        self.store_dir.mkdir(parents=True, exist_ok=True)
        print(f"[MCS] 存储目录: {self.store_dir}")

        # 初始化时自动计算网格分数
        self.compute_coverage()

            # 设置 MCS 引用（用于 Detection 查询质量分数）
        for camera in cameras:
            camera.mcs = self.scheduler
        
        print(f"MCS: Score maps computed for {len(cameras)} cameras")

        # 训练区域管理
        self.train_area = None  # 全局训练区域（球场坐标）
        self.train_area_source_camera_id = None  # 训练区域来源相机ID
        
        # 检查并设置训练区域
        if auto_setup_train_area:
            self._setup_train_area()
        
        # 组织文件结构
        if organize_files:
            self._organize_camera_files()
            self._create_combined_visualization()
        
        # 初始化检测器
        self.detector = MCSDetector(self)
   
    # ————————————————————检测方法（委托给detector）——————————————————————
    def detect(self, frames_dict=None, conf=0.25, verbose=False):
        """
        运行所有相机的检测（委托给 MCSDetector）
        
        默认行为：保留场地范围内的所有检测（不使用MCS分配图过滤）
        
        Args:
            frames_dict: 帧字典 {camera_id: frame}
            conf: YOLO 置信度阈值
            verbose: 是否显示详细信息
            
        Returns:
            检测统计信息字典
        """
        return self.detector.detect(frames_dict, conf, verbose)
    
    def visualize_detections(self, save_individual=True, save_combined=True):
        """
        可视化检测框（委托给 MCSDetector）
        
        Args:
            save_individual: 是否保存单个相机的检测图
            save_combined: 是否保存组合检测大图
            
        Returns:
            可视化结果字典
        """
        return self.detector.visualize_detections(save_individual, save_combined)
    
    def detect_colors(self, use_batch=False):
        """
        对所有相机的检测结果进行球衣颜色识别（独立步骤）
        
        这个方法应该在 detect() 之后、visualize_positions() 之前调用。
        它会更新所有 Detection 对象的 dominant_color 属性。
        
        Args:
            use_batch: 是否使用批处理API（默认 False，使用实时API）
                      - False: 实时API模式，适合单帧快速测试
                      - True: 批处理API模式，适合大量检测的批量处理
        
        示例用法：
            # 1. 检测
            mcs.detect(frames_dict, conf=0.5, map_type="all")
            
            # 2. 颜色识别（独立调用）
            mcs.detect_colors(use_batch=False)  # 实时API
            
            # 3. 可视化（多次调用，不同阈值）
            mcs.visualize_positions("all")
            mcs.visualize_positions("all", score_threshold=0.3)
            mcs.visualize_positions("all", score_threshold=0.5)
        """
        print(f"\n[MCS] 开始检测球衣颜色（{'批处理' if use_batch else '实时'}API模式）...")
        color_detector = ColorDetector()
        
        total_detections = 0
        for camera in self.cameras:
            if not hasattr(camera, 'detection') or camera.detection is None:
                continue
            
            # 检查是否有检测结果和帧图像
            if not hasattr(camera.detection, 'detections_filtered') or not camera.detection.detections_filtered:
                continue
            if not hasattr(camera.detection, 'frame') or camera.detection.frame is None:
                continue
            
            detections = camera.detection.detections_filtered
            frame = camera.detection.frame
            total_detections += len(detections)
            
            print(f"[MCS] 相机 {camera.id}: 检测 {len(detections)} 个球衣颜色...")
            
            # API模式检测颜色
            color_results = color_detector.detect_frame_colors(
                frame=frame,
                detections=detections,
                use_batch=use_batch,
                debug=False
            )
            
            # 更新Detection对象的颜色信息
            for det, color in zip(detections, color_results):
                det.dominant_color = color
            
            print(f"[MCS] 相机 {camera.id}: 颜色检测完成")
        
        print(f"[MCS] 所有相机颜色检测完成（共 {total_detections} 个检测）\n")
    
    def save_detections_cache(self, cache_name="detections_cache"):
        """
        保存当前detection信息到缓存（委托给 MCSDetector）
        
        Args:
            cache_name: 缓存文件名（不含扩展名）
            
        Returns:
            cache_path: 缓存文件路径
            
        示例：
            mcs.detect(frames_dict)
            mcs.save_detections_cache("frame_001")
        """
        return self.detector.save_detections_cache(cache_name)
    
    def load_detections_cache(self, cache_name="detections_cache"):
        """
        加载detection缓存（委托给 MCSDetector）
        
        Args:
            cache_name: 缓存文件名（不含扩展名）
            
        Returns:
            cache_data: 缓存数据字典
        """
        return self.detector.load_detections_cache(cache_name)
    
    def visualize_detections_per_camera(self, save=True, score_threshold=None, 
                                       cache_name=None, **kwargs):
        """
        分别可视化每个相机的detection位置，拼接为一行大图（委托给 MCSDetector）
        
        Args:
            save: 是否保存
            score_threshold: 分数阈值过滤
            cache_name: 如果提供，从缓存加载而非当前detection
            **kwargs: 其他参数（scale, padding, radius等）
            
        Returns:
            combined_image: 拼接后的大图
            
        示例用法：
            # 方式1：使用当前detection
            mcs.detect(frames_dict)
            mcs.visualize_detections_per_camera()
            
            # 方式2：保存并使用缓存
            mcs.detect(frames_dict)
            mcs.save_detections_cache("frame_001")
            mcs.visualize_detections_per_camera(cache_name="frame_001", score_threshold=0.3)
        """
        return self.detector.visualize_detections_per_camera(
            save=save, 
            score_threshold=score_threshold,
            cache_name=cache_name,
            **kwargs
        )
    
    def visualize_positions(self, save=True, score_threshold=None, **kwargs):
        """
        可视化全场球员位置（委托给 MCSDetector）
        
        这个方法只负责可视化，不进行任何检测或识别。
        如果需要颜色信息，请先调用 detect_colors() 方法。
        
        显示所有相机视野内的所有检测，透明度由检测的final_score决定。
        
        Args:
            save: 是否保存
            score_threshold: 最终分数阈值（0.0-1.0），只绘制 final_score >= threshold 的检测
                           None 表示不过滤（默认）
                           例如: 0.3 表示只显示分数 >= 0.3 的检测
            **kwargs: 其他参数（scale, padding, radius, background_color等）
            
        Returns:
            位置图像
            
        示例用法：
            # 标准流程
            mcs.detect(frames_dict, conf=0.5)         # 1. 检测
            mcs.detect_colors(use_batch=False)        # 2. 识别颜色（可选）
            mcs.visualize_positions()                 # 3. 可视化
            
            # 多阈值对比（不重复检测）
            mcs.visualize_positions(score_threshold=None)
            mcs.visualize_positions(score_threshold=0.3)
            mcs.visualize_positions(score_threshold=0.5)
            mcs.visualize_positions(score_threshold=0.7)
        """
        return self.detector.visualize_positions(save, score_threshold=score_threshold, **kwargs)
    
    def _batch_detect_colors_for_crops(self, crops_data):
        """
        批量检测 crops 的颜色并更新 Detection 对象
        
        Args:
            crops_data: 列表，每项包含 {frame_idx, camera_id, det, crop_path}
        """
        if not crops_data:
            return
        
        from ocr import batch_detect_jersey_colors
        from pathlib import Path
        
        # 提取所有crop路径
        crop_paths = [item['crop_path'] for item in crops_data]
        
        # 批量检测
        color_map = batch_detect_jersey_colors(crop_paths, api_key=None)
        
        # 更新Detection对象
        for item in crops_data:
            crop_filename = item['crop_path'].name
            color = color_map.get(crop_filename, 'unknown')
            item['det'].dominant_color = color
        
        # 清理临时文件
        try:
            for item in crops_data:
                if item['crop_path'].exists():
                    item['crop_path'].unlink()
        except Exception as e:
            print(f"  ⚠️ 清理临时文件失败: {e}")
    
    def visualize_positions_video(self, max_frames=None, fps=30.0, 
                                  conf=0.25, verbose=False, sample_interval=0.3, 
                                  score_threshold=None, color_batch_size=100, **kwargs):
        """
        对所有帧生成全场位置可视化视频（测试用）
        
        这个方法会：
        1. 遍历所有相机的视频帧（同步读取）
        2. 对每一帧执行检测（可选采样）
        3. 生成全场位置可视化图（显示所有视野内检测，透明度由final_score决定）
        4. 保存为视频文件
        
        Args:
            max_frames: 最大处理帧数（None表示处理所有帧）
            fps: 输出视频帧率（默认30.0）
            conf: 检测置信度阈值（默认0.25）
            verbose: 是否显示详细检测信息
            sample_interval: 采样间隔（秒）
                - None: 处理所有帧（默认）
                - 1.0: 每秒采样一次
                - 0.5: 每0.5秒采样一次
                - 2.0: 每2秒采样一次
            score_threshold: 分数阈值过滤
                - None: 不过滤（默认）
                - 单个值（float）: 下限阈值，只绘制 final_score >= threshold 的检测
                  例如: 0.3 表示只显示分数 >= 0.3 的检测
                - 元组/列表 (min, max): 区间阈值，只绘制 min <= final_score <= max 的检测
                  例如: (0.3, 0.7) 表示只显示分数在 [0.3, 0.7] 区间的检测
            color_batch_size: 颜色检测批次大小（默认100）
                - 每收集这么多个crops后进行一次批量颜色检测
                - 较小值(50)：更快的响应，更小的内存占用，但API调用次数更多
                - 较大值(200)：减少API调用次数，但内存占用更大
                - 推荐值：100（平衡的选择）
            **kwargs: 传递给 visualize_positions 的其他参数
                - scale: 球场缩放比例（默认0.1）
                - padding: 球场边距（默认50）
                - radius: 检测点半径（默认8）
                - background_color: 球场背景颜色
        
        Returns:
            dict: 包含统计信息
                - 'video_path': 输出视频路径
                - 'total_frames': 处理的总帧数
                - 'fps': 视频帧率
                
        示例：
            # 生成视频，处理前100帧
            stats = mcs.visualize_positions_video(max_frames=100)
            
            # 每秒采样一次
            stats = mcs.visualize_positions_video(sample_interval=1.0)
            
            # 每2秒采样一次，输出10fps
            stats = mcs.visualize_positions_video(sample_interval=2.0, fps=10)
            
            # 使用分数下限阈值（只显示高质量检测）
            stats = mcs.visualize_positions_video(
                max_frames=500, 
                score_threshold=0.5  # 只显示分数 >= 0.5 的检测
            )
            
            # 使用分数区间阈值（只显示特定范围）
            stats = mcs.visualize_positions_video(
                max_frames=500,
                score_threshold=(0.3, 0.7)  # 只显示 [0.3, 0.7] 区间
            )
            
            # 完整自定义
            stats = mcs.visualize_positions_video(
                max_frames=500, 
                fps=30,
                conf=0.3,
                score_threshold=0.6,
                scale=0.15,
                radius=10
            )
            
            # 调整颜色检测批次大小
            stats = mcs.visualize_positions_video(
                max_frames=1000,
                color_batch_size=50  # 更频繁的批量检测
            )
            
            # 大批次处理（适合服务器环境）
            stats = mcs.visualize_positions_video(
                max_frames=2000,
                color_batch_size=200  # 减少API调用次数
            )
        """
        from tools.frame_tool import VideoWriterStream
        import os
        
        # 获取第一个相机的视频fps（假设所有相机fps相同）
        source_fps = None
        for camera in self.cameras:
            if hasattr(camera, 'video_manager') and camera.video_manager:
                source_fps = camera.video_manager.cap.get(cv2.CAP_PROP_FPS)
                break
        
        if source_fps is None or source_fps == 0:
            source_fps = 30.0  # 默认值
            print(f"  ⚠️ 无法获取视频FPS，使用默认值: {source_fps}")
        
        # 计算采样参数
        if sample_interval is not None:
            # 计算跳过的帧数
            frame_skip = int(source_fps * sample_interval)
            if frame_skip < 1:
                frame_skip = 1
            sample_info = f"每{sample_interval}秒采样（每{frame_skip}帧）"
        else:
            frame_skip = 1
            sample_info = "处理所有帧"
        
        print(f"\n{'='*70}")
        print(f"[MCS] 开始生成位置可视化视频")
        print(f"  - 最大帧数: {max_frames if max_frames else '全部'}")
        print(f"  - 源视频FPS: {source_fps:.1f}")
        print(f"  - 输出视频FPS: {fps}")
        print(f"  - 采样策略: {sample_info}")
        print(f"  - 检测阈值: {conf}")
        
        # 显示分数阈值信息
        if score_threshold is not None:
            if isinstance(score_threshold, (tuple, list)) and len(score_threshold) == 2:
                print(f"  - 分数区间: [{score_threshold[0]:.2f}, {score_threshold[1]:.2f}]")
            else:
                print(f"  - 分数下限: >= {score_threshold:.2f}")
        else:
            print(f"  - 分数过滤: 无")
        
        print(f"  - 颜色批次大小: {color_batch_size} crops/batch")
        
        print(f"{'='*70}\n")
        
        # 重置所有相机的视频到开头
        for camera in self.cameras:
            if hasattr(camera, 'video_manager') and camera.video_manager:
                camera.video_manager.reset()
        
        # 输出路径
        output_dir = f"store/{self.name}_mcs/position_video"
        os.makedirs(output_dir, exist_ok=True)
        
        # 根据阈值设置文件名
        filename_suffix = ""
        if score_threshold is not None:
            if isinstance(score_threshold, (tuple, list)) and len(score_threshold) == 2:
                # 区间阈值
                min_thr, max_thr = score_threshold
                filename_suffix = f"_thr{min_thr:.2f}-{max_thr:.2f}".replace('.', 'p')
            else:
                # 单个下限阈值
                filename_suffix = f"_thr{score_threshold:.2f}".replace('.', 'p')
        
        video_path = f"{output_dir}/{self.name}_position_video{filename_suffix}.mp4"
        
        frame_count = 0  # 已处理的帧数
        total_read = 0   # 从视频读取的总帧数
        
        # 收集所有crops用于批量颜色检测
        all_crops_data = []  # [{frame_idx, camera_id, det, crop_path}, ...]
        
        with VideoWriterStream(video_path, fps=fps) as writer:
            while True:
                # 读取所有相机的当前帧
                frames_dict = {}
                all_success = True
                
                for camera in self.cameras:
                    if not hasattr(camera, 'video_manager') or not camera.video_manager:
                        print(f"  ⚠️ 相机 {camera.id}: 没有视频管理器")
                        all_success = False
                        break
                    
                    ret, frame = camera.video_manager.cap.read()
                    if not ret:
                        all_success = False
                        break
                    
                    frames_dict[camera.id] = frame
                
                total_read += 1
                
                # 如果任一相机读取失败，结束循环
                if not all_success:
                    break
                
                # 采样判断：只处理符合采样间隔的帧
                if (total_read - 1) % frame_skip != 0:
                    continue
                
                # 检查是否达到最大帧数
                if max_frames and frame_count >= max_frames:
                    break
                
                # 执行检测
                try:
                    stats = self.detector.detect(frames_dict, conf=conf, verbose=verbose)
                except Exception as e:
                    print(f"  ⚠️ 第 {frame_count} 帧检测失败: {e}")
                    frame_count += 1
                    continue
                
                # 收集当前帧的所有crops（用于后续批量颜色检测）
                for camera in self.cameras:
                    if not hasattr(camera, 'detection') or camera.detection is None:
                        continue
                    if not hasattr(camera.detection, 'detections_filtered') or not camera.detection.detections_filtered:
                        continue
                    if not hasattr(camera.detection, 'frame') or camera.detection.frame is None:
                        continue
                    
                    detections = camera.detection.detections_filtered
                    frame_img = camera.detection.frame
                    
                    # 保存crops
                    for det_idx, det in enumerate(detections):
                        x1, y1, x2, y2 = det.bbox.astype(int)
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame_img.shape[1], x2), min(frame_img.shape[0], y2)
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        crop = frame_img[y1:y2, x1:x2]
                        
                        # 保存临时文件
                        crop_dir = Path(f"output/color_detection_batch")
                        crop_dir.mkdir(parents=True, exist_ok=True)
                        crop_path = crop_dir / f"f{frame_count:04d}_c{camera.id}_d{det_idx:03d}.jpg"
                        cv2.imwrite(str(crop_path), crop)
                        
                        all_crops_data.append({
                            'frame_idx': frame_count,
                            'camera_id': camera.id,
                            'det': det,
                            'crop_path': crop_path
                        })
                
                # 检查是否需要批量检测颜色
                if len(all_crops_data) >= color_batch_size:
                    print(f"\n  [批量检测] 已收集 {len(all_crops_data)} 个crops，开始批量检测颜色...")
                    self._batch_detect_colors_for_crops(all_crops_data)
                    all_crops_data.clear()
                    print(f"  [批量检测] 完成，继续处理...")
                
                # 生成位置可视化图（不保存为文件）
                try:
                    pos_img = self.detector.visualize_positions(
                        save=False, 
                        score_threshold=score_threshold, 
                        **kwargs
                    )
                    
                    # 写入视频
                    writer.write(pos_img)
                    
                    frame_count += 1
                    
                    # 每10帧输出一次进度
                    if frame_count % 10 == 0:
                        print(f"  已处理 {frame_count} 帧 (已收集 {len(all_crops_data)} 个crops)...")
                        
                except Exception as e:
                    print(f"  ⚠️ 第 {frame_count} 帧可视化失败: {e}")
                    frame_count += 1
                    continue
        
        # 批量检测剩余的crops颜色
        if len(all_crops_data) > 0:
            print(f"\n[MCS] 处理剩余的 {len(all_crops_data)} 个crops...")
            self._batch_detect_colors_for_crops(all_crops_data)
            all_crops_data.clear()
            print(f"[MCS] 所有颜色检测完成\n")
        
        # 输出结果
        print(f"\n{'='*70}")
        print(f"[MCS] 位置可视化视频生成完成")
        print(f"  - 视频路径: {video_path}")
        print(f"  - 读取帧数: {total_read}")
        print(f"  - 处理帧数: {frame_count}")
        if sample_interval:
            print(f"  - 采样率: {frame_count}/{total_read} = {frame_count/total_read*100:.1f}%")
        print(f"  - 输出FPS: {fps}")
        print(f"  - 时长: {frame_count/fps:.2f}秒")
        print(f"{'='*70}\n")
        
        return {
            'video_path': video_path,
            'total_frames': frame_count,
            'total_read': total_read,
            'fps': fps,
            'duration': frame_count / fps,
            'sample_interval': sample_interval,
            'frame_skip': frame_skip
        }
    
    def visualize_score_functions(self, resolution=100, mode='both'):
        """
        可视化分数函数：为每个相机生成距离分数、畸变分数和综合分数的可视化
        
        生成的图片保存在 store/{name}_mcs/score/ 目录下：
        - 每个相机一张图，包含3个子图（距离分数、畸变分数、综合分数）
        - 一张合成大图，包含所有相机的可视化
        - 支持2D热力图和3D连续曲面图两种模式
        
        Args:
            resolution: 采样分辨率（每个维度的采样点数，默认100）
            mode: 可视化模式
                - '2d': 只生成2D热力图
                - '3d': 只生成3D连续曲面图
                - 'both': 生成2D和3D两种（默认）
            
        Returns:
            可视化图像字典 {'2d': img, '3d': img}
            
        示例：
            mcs = MCS(cameras, pitch_config, name="wuchuan")
            
            # 生成2D和3D两种可视化
            result = mcs.visualize_score_functions(resolution=150)
            
            # 只生成3D曲面图
            result = mcs.visualize_score_functions(resolution=100, mode='3d')
            
            # 只生成2D热力图
            result = mcs.visualize_score_functions(resolution=100, mode='2d')
        """
        score_dir = self.store_dir / "score"
        return self.scheduler.visualize_score_functions(resolution=resolution, save_dir=score_dir, mode=mode)
    
    # ————————————————————训练区域——————————————————————        
    def _setup_train_area(self):
        """
        自动设置训练区域
        
        逻辑：
        1. 检查所有相机是否已有训练区域
        2. 如果都有，使用第一个相机的训练区域作为全局区域
        3. 如果都没有，提示用户选择一个相机来设置训练区域
        4. 如果部分有，使用第一个有训练区域的相机
        """
        print("\n" + "=" * 70)
        print("MCS 训练区域设置")
        print("=" * 70)
        
        # 检查哪些相机已有训练区域
        cameras_with_area = []
        for i, camera in enumerate(self.cameras):
            if camera.train_area.region_points:
                cameras_with_area.append((i, camera))
                print(f"✅ 相机 {camera.id} 已有训练区域 ({len(camera.train_area.region_points)} 个点)")
            else:
                print(f"⚠️  相机 {camera.id} 无训练区域")
        
        if len(cameras_with_area) == len(self.cameras):
            # 所有相机都有训练区域，使用第一个
            idx, camera = cameras_with_area[0]
            self._use_camera_train_area(idx)
            print(f"\n✅ 使用相机 {camera.id} 的训练区域作为全局区域")
            
        elif len(cameras_with_area) > 0:
            # 部分相机有训练区域
            idx, camera = cameras_with_area[0]
            self._use_camera_train_area(idx)
            print(f"\n✅ 使用相机 {camera.id} 的训练区域作为全局区域")
            
            # 询问是否应用到所有相机
            response = input(f"\n是否将此训练区域应用到所有 {len(self.cameras)} 个相机? (y/n, 默认y): ").strip().lower()
            if response != 'n':
                self.apply_train_area_to_all()
            
        else:
            # 所有相机都没有训练区域
            print(f"\n⚠️  所有相机都没有训练区域！")
            print(f"请选择一个相机来设置训练区域：")
            for i, camera in enumerate(self.cameras):
                print(f"  [{i}] 相机 {camera.id}")
            
            while True:
                try:
                    choice = input(f"\n请输入相机索引 (0-{len(self.cameras)-1}, 或按回车跳过): ").strip()
                    if choice == "":
                        print("⚠️  跳过训练区域设置")
                        break
                    
                    idx = int(choice)
                    if 0 <= idx < len(self.cameras):
                        self.set_train_area_from_camera(idx)
                        break
                    else:
                        print(f"❌ 无效的索引，请输入 0-{len(self.cameras)-1}")
                except ValueError:
                    print("❌ 请输入有效的数字")
        
        print("=" * 70 + "\n")
    
    def set_train_area_from_camera(self, camera_index: int, apply_to_all: bool = True):
        """
        从指定相机设置训练区域
        
        Args:
            camera_index: 相机在列表中的索引
            apply_to_all: 是否应用到所有相机（默认 True）
        """
        if camera_index < 0 or camera_index >= len(self.cameras):
            raise ValueError(f"无效的相机索引: {camera_index}")
        
        camera = self.cameras[camera_index]
        
        print(f"\n[MCS] 使用相机 {camera.id} 设置训练区域")
        
        # 如果该相机没有训练区域，触发标定流程
        if not camera.train_area.region_points:
            print(f"[MCS] 相机 {camera.id} 没有训练区域，开始标定...")
            
            print(f"\n步骤 1/2: 在视频画面中标注自定义点（可选）")
            print("提示：如果只使用关键点，可以直接取消（按 's' 保存空组）。")
            camera.train_area.define_from_video()
            
            print(f"\n步骤 2/2: 在球场俯视图上选择训练区域边界")
            camera.train_area.define_from_pitch_map()
        
        # 设置为全局训练区域
        self._use_camera_train_area(camera_index)
        
        # 应用到所有相机
        if apply_to_all:
            self.apply_train_area_to_all()
    
    def _use_camera_train_area(self, camera_index: int):
        """
        使用指定相机的训练区域作为全局区域
        
        Args:
            camera_index: 相机索引
        """
        camera = self.cameras[camera_index]
        self.train_area = camera.train_area.region_points.copy()
        self.train_area_source_camera_id = camera.id
        print(f"[MCS] 全局训练区域已设置 (来源: 相机 {camera.id}, {len(self.train_area)} 个点)")
    
    def apply_train_area_to_all(self):
        """
        将全局训练区域应用到所有相机
        
        注意：由于不同相机的视角不同，训练区域（球场坐标）是共享的，
        但在各个相机的视频画面中的表现会不同。
        """
        if self.train_area is None or len(self.train_area) == 0:
            print("[MCS] 警告: 没有全局训练区域可应用")
            return
        
        print(f"\n[MCS] 应用训练区域到所有 {len(self.cameras)} 个相机...")
        
        for camera in self.cameras:
            # 复制全局训练区域到各个相机
            camera.train_area.region_points = self.train_area.copy()
            
            # 保存到各相机的标定目录
            camera.train_area.save()
            
            print(f"  ✅ 相机 {camera.id}: 训练区域已应用并保存")
        
        print(f"[MCS] ✅ 训练区域应用完成\n")
    
    def save_global_train_area(self, file_path: str):
        """
        保存全局训练区域到文件
        
        Args:
            file_path: 保存路径
        """
        import json
        
        if self.train_area is None:
            print("[MCS] 警告: 没有全局训练区域可保存")
            return
        
        data = {
            'source_camera_id': self.train_area_source_camera_id,
            'region_points': self.train_area,
            'num_cameras': len(self.cameras),
            'camera_ids': [cam.id for cam in self.cameras]
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"[MCS] 全局训练区域已保存至: {file_path}")
    
    def load_global_train_area(self, file_path: str, apply_to_all: bool = True):
        """
        从文件加载全局训练区域
        
        Args:
            file_path: 加载路径
            apply_to_all: 是否应用到所有相机（默认 True）
        """
        import json
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.train_area = data.get('region_points', [])
        self.train_area_source_camera_id = data.get('source_camera_id')
        
        print(f"[MCS] 全局训练区域已加载: {len(self.train_area)} 个点")
        
        if apply_to_all:
            self.apply_train_area_to_all()
    
    # ——————————————————————————————————————————————————
    
    def compute_coverage(self):
        """计算覆盖区域的分数"""
        return self.scheduler.compute_coverage()
        
    
    def _organize_camera_files(self):
        """
        将所有相机的文件夹移入 MCS 存储目录
        
        store/camera1/ -> store/{name}_mcs/camera1/
        store/camera2/ -> store/{name}_mcs/camera2/
        """
        import shutil
        
        print(f"\n[MCS] === 组织相机文件 ===")
        
        for camera in self.cameras:
            old_dir = camera.store_dir
            new_dir = self.store_dir / camera.name
            
            if old_dir.exists() and old_dir != new_dir:
                try:
                    # 如果目标已存在，先删除
                    if new_dir.exists():
                        shutil.rmtree(new_dir)
                    
                    # 移动文件夹
                    shutil.move(str(old_dir), str(new_dir))
                    
                    # 更新 camera 的属性
                    camera.store_dir = new_dir
                    camera.mcs_name = self.name
                    camera._setup_search_paths()  # 重新设置搜索路径
                    
                    print(f"  ✅ {camera.name}: {old_dir} -> {new_dir}")
                except Exception as e:
                    print(f"  ❌ {camera.name}: 移动失败 - {e}")
            elif old_dir == new_dir:
                print(f"  ℹ️  {camera.name}: 已在正确位置")
            else:
                print(f"  ⚠️  {camera.name}: 源目录不存在 ({old_dir})")
        
        print(f"[MCS] === 文件组织完成 ===\n")
    
    def _create_combined_visualization(self):
        """
        创建组合可视化图：将每个 camera 的 4 张图拼接成大图
        
        布局：每个 camera 占一行，包含 4 列：
        - detect (检测框可视化)
        - return (关键点回投)
        - video_boundaries (视频边界)
        - visible_grids (可见网格)
        """
        print(f"\n[MCS] === 创建组合可视化 ===")
        
        # 图片类型和文件名模式
        image_types = [
            ('detect', '../detect/{name}_detect.jpg'),  # 检测图在detect文件夹中
            ('return', '{name}_return.jpg'),
            ('video_boundaries', '{name}_video_boundaries.png'),
            ('visible_grids', '{name}_visible_grids.jpg')
        ]
        
        # 加载所有图片
        images_data = []
        for camera in self.cameras:
            cam_images = []
            for img_type, filename_pattern in image_types:
                img_path = self.store_dir / camera.name / filename_pattern.format(name=camera.name)
                
                if img_path.exists():
                    img = cv2.imread(str(img_path))
                    if img is not None:
                        cam_images.append(img)
                    else:
                        print(f"  ⚠️  {camera.name}: 无法读取 {img_type}")
                        cam_images.append(None)
                else:
                    print(f"  ℹ️  {camera.name}: {img_type} 不存在")
                    cam_images.append(None)
            
            if any(img is not None for img in cam_images):
                images_data.append((camera.name, cam_images))
        
        if not images_data:
            print(f"  ❌ 没有找到任何相机图片")
            return None
        
        # 使用通用工具创建组合图
        final_image = create_combined_image_layout(
            images_data=images_data,
            title=f"MCS: {self.name} - {len(self.cameras)} Cameras",
            layout='grid',
            target_height=800,
            label_position='left',
            label_width=150,
            header_height=100,
            add_column_headers=True,
            column_headers=['Detection', 'Keypoints', 'Boundaries', 'Grids']
        )
        
        # 保存（支持中文路径）
        output_path = self.store_dir / f"{self.name}_combined_view.jpg"
        imwritex(str(output_path), final_image)
        print(f"  ✅ 组合可视化已保存: {output_path}")
        print(f"     尺寸: {final_image.shape[1]}x{final_image.shape[0]}")
        print(f"[MCS] === 可视化完成 ===\n")
        
        return final_image
    
    # ————————————————————测试方法：多阈值可视化——————————————————————
    
    def test_positions_sequential_frames(self,
                                         threshold_ranges=None,
                                         single_thresholds=None,
                                         num_frames=30,
                                         fps=1.0,
                                         conf=0.25,
                                         save_dir=None,
                                         use_batch_api=False,
                                         **kwargs):
        """
        在多个时间点采样帧，为每帧生成阈值对比图
        
        Args:
            threshold_ranges: 阈值区间列表
            single_thresholds: 单阈值列表
            num_frames: 采样帧数
            fps: 采样帧率（每秒采样多少帧）
            conf: 检测置信度
            save_dir: 保存目录名（相对于 position/），默认为 "sequential"
            use_batch_api: 是否使用批处理API检测颜色
            **kwargs: 传递给 visualize_positions 的参数
        
        Returns:
            list of saved paths
        """
        from pathlib import Path
        import time
        
        if save_dir is None:
            save_dir = "sequential"
        
        output_dir = self.store_dir / "position" / save_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*70}")
        print(f"[MCS] 生成序列帧阈值对比图")
        print(f"{'='*70}")
        print(f"  采样帧数: {num_frames}")
        print(f"  采样间隔: {1.0/fps:.2f}秒")
        print(f"  使用批处理API: {use_batch_api}")
        print(f"  保存目录: {output_dir}")
        print(f"{'='*70}\n")
        
        # 获取视频帧间隔
        sample_interval = 1.0 / fps  # 秒
        saved_paths = []
        
        for frame_idx in range(num_frames):
            timestamp = frame_idx * sample_interval
            print(f"\n{'─'*70}")
            print(f"[{frame_idx+1}/{num_frames}] 处理时间点: {timestamp:.2f}秒")
            print(f"{'─'*70}")
            
            # 读取所有相机的帧
            frames_dict = {}
            all_success = True
            
            for camera in self.cameras:
                # 设置视频到指定时间点
                camera.video_manager.cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
                ret, frame = camera.video_manager.cap.read()
                
                if not ret:
                    print(f"  ⚠️  相机 {camera.id} 无法读取帧（时间 {timestamp:.2f}s）")
                    all_success = False
                    break
                
                frames_dict[camera.id] = frame
                print(f"  ✓ 相机 {camera.id}: 读取帧成功")
            
            if not all_success:
                print(f"  ✗ 跳过此帧")
                continue
            
            # 执行检测（通过 MCS 的 detect 方法统一处理）
            print(f"\n  检测中...")
            start_time = time.time()
            
            # 使用 MCS 的 detect 方法来汇总所有相机的检测结果
            stats = self.detect(frames_dict, conf=conf, verbose=False)
            
            # 如果需要检测颜色
            if not use_batch_api:
                # 不使用批处理API：逐个相机、逐个检测地调用
                from camera import ColorDetector
                color_detector = ColorDetector(api_key=None)
                
                for camera in self.cameras:
                    if not hasattr(camera.detection, 'detections_filtered'):
                        continue
                    
                    detections = camera.detection.detections_filtered
                    if len(detections) == 0:
                        continue
                    
                    frame = frames_dict.get(camera.id)
                    if frame is None:
                        continue
                    
                    print(f"    相机 {camera.id}: 检测 {len(detections)} 个球衣颜色...")
                    
                    # 单独检测颜色
                    colors = color_detector.detect_frame_colors(
                        frame, 
                        detections, 
                        use_batch=False,
                        debug=False
                    )
                    
                    # 更新 Detection 对象的颜色属性
                    for det, color in zip(detections, colors):
                        det.dominant_color = color
            else:
                # 使用批处理API
                self.detect_colors(use_batch=True)
            
            elapsed = time.time() - start_time
            print(f"  ✓ 检测完成 ({elapsed:.2f}秒)")
            print(f"    总检测数: {stats.get('total_detections', 0)}")
            
            # 生成阈值对比图
            print(f"\n  生成阈值对比图...")
            combined_img = self.test_positions_thresholds(
                threshold_ranges=threshold_ranges,
                single_thresholds=single_thresholds,
                save=False,
                **kwargs
            )
            
            if combined_img is not None:
                # 保存到指定目录，文件名包含时间信息
                filename = f"frame_{frame_idx:03d}_t{timestamp:.2f}s.png"
                save_path = output_dir / filename
                cv2.imwrite(str(save_path), combined_img)
                saved_paths.append(save_path)
                print(f"  ✓ 保存: {filename}")
            else:
                print(f"  ✗ 生成失败")
        
        print(f"\n{'='*70}")
        print(f"[MCS] 完成！共生成 {len(saved_paths)}/{num_frames} 张图片")
        print(f"[MCS] 保存目录: {output_dir}")
        print(f"{'='*70}\n")
        
        return saved_paths
    
    def test_positions_thresholds(self, 
                                   threshold_ranges=None, 
                                   single_thresholds=None,
                                   save=True,
                                   **kwargs):
        """
        测试方法：生成多个不同阈值的位置图并拼接为大图
        
        这个方法会：
        1. 遍历所有指定的阈值（区间或下限）
        2. 为每个阈值生成位置可视化图
        3. 将所有图片拼接成一张大图
        
        Args:
            threshold_ranges: 阈值区间列表，例如 [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
                            None 表示使用默认区间
            single_thresholds: 单阈值（下限）列表，例如 [0.2, 0.4, 0.6, 0.8]
                             None 表示使用默认值
            save: 是否保存拼接后的大图
            **kwargs: 传递给 visualize_positions 的其他参数（scale, radius等）
            
        Returns:
            combined_image: 拼接后的大图
            
        示例:
            # 使用默认阈值
            mcs.detect(frames_dict)
            combined = mcs.test_positions_thresholds()
            
            # 自定义阈值
            combined = mcs.test_positions_thresholds(
                threshold_ranges=[(0.0, 0.4), (0.4, 0.8), (0.8, 1.0)],
                single_thresholds=[0.3, 0.5, 0.7],
                radius=10
            )
        """
        from tools.frame_tool import create_combined_image_layout
        
        print(f"\n{'='*70}")
        print(f"[MCS] 测试：多阈值位置可视化拼接")
        print(f"{'='*70}\n")
        
        # 默认阈值
        if threshold_ranges is None:
            threshold_ranges = [
                (0.0, 0.3),   # 低分数区间
                (0.3, 0.5),   # 中低分数区间
                (0.5, 0.7),   # 中等分数区间
                (0.7, 1.0),   # 高分数区间
            ]
        
        if single_thresholds is None:
            single_thresholds = [0.3, 0.5, 0.7]
        
        # 分别收集区间阈值和单阈值配置
        range_thresholds = []
        single_threshold_configs = []
        
        # 添加区间阈值
        for min_thr, max_thr in threshold_ranges:
            range_thresholds.append({
                'type': 'range',
                'value': (min_thr, max_thr),
                'label': f'[{min_thr:.1f}, {max_thr:.1f}]'
            })
        
        # 添加单阈值
        if single_thresholds:
            for thr in single_thresholds:
                single_threshold_configs.append({
                    'type': 'single',
                    'value': thr,
                    'label': f'>= {thr:.1f}'
                })
        
        # 生成区间阈值的可视化图（第一行）
        range_images = []
        if range_thresholds:
            print(f"\n{'─'*60}")
            print("生成区间阈值可视化（第一行）")
            print(f"{'─'*60}")
            for config in range_thresholds:
                print(f"\n生成阈值 {config['label']} 的可视化...")
                
                pos_img = self.detector.visualize_positions(
                    save=False,
                    score_threshold=config['value'],
                    **kwargs
                )
                
                if pos_img is not None:
                    # 确保是RGBA格式（保持透明背景）
                    if pos_img.shape[2] == 3:  # BGR格式，转换为BGRA
                        pos_img_rgba = cv2.cvtColor(pos_img, cv2.COLOR_BGR2BGRA)
                    else:
                        pos_img_rgba = pos_img
                    
                    range_images.append((config['label'], pos_img_rgba))
                    print(f"  ✓ 完成")
                else:
                    print(f"  ✗ 失败")
        
        # 生成单阈值的可视化图（第二行）
        single_images = []
        if single_threshold_configs:
            print(f"\n{'─'*60}")
            print("生成单阈值可视化（第二行）")
            print(f"{'─'*60}")
            for config in single_threshold_configs:
                print(f"\n生成阈值 {config['label']} 的可视化...")
                
                pos_img = self.detector.visualize_positions(
                    save=False,
                    score_threshold=config['value'],
                    **kwargs
                )
                
                if pos_img is not None:
                    # 确保是RGBA格式（保持透明背景）
                    if pos_img.shape[2] == 3:  # BGR格式，转换为BGRA
                        pos_img_rgba = cv2.cvtColor(pos_img, cv2.COLOR_BGR2BGRA)
                    else:
                        pos_img_rgba = pos_img
                    
                    single_images.append((config['label'], pos_img_rgba))
                    print(f"  ✓ 完成")
                else:
                    print(f"  ✗ 失败")
        
        if not range_images and not single_images:
            print(f"\n❌ 没有成功生成任何图片")
            return None
        
        # 拼接成大图：区间阈值一行，单阈值一行
        print(f"\n{'─'*60}")
        print(f"拼接图片：区间阈值 {len(range_images)} 张 + 单阈值 {len(single_images)} 张")
        print(f"{'─'*60}")
        
        # 辅助函数：调整图片大小并添加标签
        def resize_and_label_images(images_list, target_height=400, label_width=120):
            """将图片调整为统一高度并添加左侧标签（支持RGBA透明背景）"""
            processed = []
            for label, img in images_list:
                # 调整图片高度
                h, w = img.shape[:2]
                scale = target_height / h
                new_w = int(w * scale)
                resized = cv2.resize(img, (new_w, target_height))
                
                # 创建标签区域（RGBA格式，白色不透明）
                label_img = np.ones((target_height, label_width, 4), dtype=np.uint8) * 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_x = (label_width - text_size[0]) // 2
                text_y = (target_height + text_size[1]) // 2
                cv2.putText(label_img, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
                
                # 拼接标签和图片
                combined = np.hstack([label_img, resized])
                processed.append(combined)
            
            return processed
        
        # 处理两组图片
        combined_rows = []
        
        if range_images:
            range_processed = resize_and_label_images(range_images, target_height=400, label_width=120)
            range_row = np.hstack(range_processed)
            combined_rows.append(range_row)
            print(f"✓ 第一行（区间阈值）拼接完成: {range_row.shape[1]}x{range_row.shape[0]}")
        
        if single_images:
            single_processed = resize_and_label_images(single_images, target_height=400, label_width=120)
            single_row = np.hstack(single_processed)
            combined_rows.append(single_row)
            print(f"✓ 第二行（单阈值）拼接完成: {single_row.shape[1]}x{single_row.shape[0]}")
        
        # 如果两行宽度不同，需要对齐
        if len(combined_rows) == 2:
            max_width = max(row.shape[1] for row in combined_rows)
            aligned_rows = []
            for row in combined_rows:
                if row.shape[1] < max_width:
                    # 右侧填充白色不透明
                    padding = np.ones((row.shape[0], max_width - row.shape[1], 4), dtype=np.uint8) * 255
                    row = np.hstack([row, padding])
                aligned_rows.append(row)
            combined_rows = aligned_rows
        
        # 垂直拼接两行
        combined_image = np.vstack(combined_rows)
        print(f"✓ 最终拼接完成: {combined_image.shape[1]}x{combined_image.shape[0]}")
        
        # 保存为PNG格式（支持透明通道）
        if save:
            output_path = self.store_dir / "position" / f"{self.name}_position_thresholds_combined.png"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), combined_image)
            print(f"\n✅ 拼接大图已保存: {output_path}")
            print(f"   尺寸: {combined_image.shape[1]}x{combined_image.shape[0]}")
            print(f"   格式: PNG (RGBA, 透明背景)")
        
        print(f"\n{'='*70}")
        print(f"[MCS] 测试完成")
        print(f"{'='*70}\n")
        
        return combined_image
    
    def test_positions_video_thresholds(self,
                                       threshold_ranges=None,
                                       single_thresholds=None,
                                       max_frames=None,
                                       fps=30.0,
                                       conf=0.25,
                                       sample_interval=0.3,
                                       layout='horizontal',
                                       color_batch_size=100,
                                       **kwargs):
        """
        测试方法：生成多个不同阈值的位置视频并拼接为一个视频
        
        优化逻辑：
        1. 第一步：遍历所有视频帧，进行一次全局检测
        2. 第二步：收集所有 crops，进行一次全局批量颜色识别
        3. 第三步：对于每个阈值区间，使用相同的检测结果生成不同的可视化帧
        4. 第四步：同步拼接所有阈值的视频帧
        
        Args:
            threshold_ranges: 阈值区间列表，例如 [(0.0, 0.3), (0.3, 0.7), (0.7, 1.0)]
                            None 表示使用默认区间
            single_thresholds: 单阈值（下限）列表，例如 [0.2, 0.4, 0.6, 0.8]
                             None 表示使用默认值
            max_frames: 最大处理帧数
            fps: 输出视频帧率
            conf: 检测置信度阈值
            sample_interval: 采样间隔（秒）
            layout: 拼接布局
                - 'horizontal': 水平拼接（所有视频并排）
                - 'vertical': 垂直拼接（所有视频上下堆叠）
                - 'grid': 网格拼接（自动计算行列）
            color_batch_size: 颜色检测批次大小（默认100）
            **kwargs: 传递给 visualize_positions 的其他参数
            
        Returns:
            dict: 包含统计信息
                - 'video_path': 输出视频路径
                - 'total_frames': 处理的总帧数
                - 'fps': 视频帧率
                
        示例：
            stats = mcs.test_positions_video_thresholds(
                threshold_ranges=[(0.0, 0.4), (0.4, 0.6), (0.6, 1.0)],
                max_frames=300,
                layout='horizontal'
            )
        """
        from tools.frame_tool import VideoWriterStream
        from pathlib import Path
        import tempfile
        
        print(f"\n{'='*70}")
        print(f"[MCS] 测试：多阈值位置视频拼接（优化版：一次检测）")
        print(f"{'='*70}\n")
        
        # 默认阈值
        if threshold_ranges is None:
            threshold_ranges = [
                (0.0, 0.5),   # 低分数区间
                (0.5, 1.0),   # 高分数区间
            ]
        
        if single_thresholds is None:
            single_thresholds = []
        
        # 收集所有阈值配置
        all_thresholds = []
        
        # 添加区间阈值
        for min_thr, max_thr in threshold_ranges:
            all_thresholds.append({
                'type': 'range',
                'value': (min_thr, max_thr),
                'label': f'[{min_thr:.1f}, {max_thr:.1f}]'
            })
        
        # 添加单阈值
        for thr in single_thresholds:
            all_thresholds.append({
                'type': 'single',
                'value': thr,
                'label': f'>= {thr:.1f}'
            })
        
        print(f"将生成 {len(all_thresholds)} 个阈值的视频\n")
        
        # ============================================================
        # 第一阶段：全局检测所有帧
        # ============================================================
        print(f"\n{'─'*70}")
        print("[阶段 1/3] 全局检测所有帧")
        print(f"{'─'*70}\n")
        
        # 获取第一个相机的视频fps
        source_fps = None
        for camera in self.cameras:
            if hasattr(camera, 'video_manager') and camera.video_manager:
                source_fps = camera.video_manager.cap.get(cv2.CAP_PROP_FPS)
                break
        
        if source_fps is None or source_fps == 0:
            source_fps = 30.0
            print(f"[警告] 无法获取源视频fps，使用默认值: {source_fps}")
        
        # 计算采样策略
        if sample_interval is not None:
            sample_frames = max(1, int(source_fps * sample_interval))
            print(f"采样策略: 每{sample_interval}秒采样（每{sample_frames}帧）")
        else:
            sample_frames = 1
            print(f"采样策略: 处理所有帧")
        
        # 重置所有相机的视频读取位置
        for camera in self.cameras:
            if hasattr(camera, 'video_manager') and camera.video_manager:
                camera.video_manager.reset()
        
        # 收集所有帧的检测结果
        all_frames_detections = []  # [(frame_idx, {camera_id: detections}), ...]
        all_crops_data = []  # 收集所有 crops
        
        frame_idx = 0
        processed_count = 0
        
        print(f"开始遍历视频帧并检测...")
        
        while True:
            if max_frames and processed_count >= max_frames:
                break
            
            # 读取所有相机的当前帧（同步）
            frames_dict = {}
            all_cameras_success = True
            should_process = (frame_idx % sample_frames == 0)  # 判断是否采样此帧
            
            for camera in self.cameras:
                if hasattr(camera, 'video_manager') and camera.video_manager:
                    ret, frame = camera.video_manager.cap.read()
                    if not ret:
                        all_cameras_success = False
                        break
                    # 只有采样帧才保存到 frames_dict
                    if should_process:
                        frames_dict[camera.id] = frame
            
            if not all_cameras_success:
                break
            
            frame_idx += 1
            
            # 如果不需要处理此帧，直接跳过
            if not should_process:
                continue
            
            # 对当前帧进行检测
            self.detector.detect(frames_dict, conf=conf, verbose=False)
            
            # 收集当前帧的所有检测结果和 crops
            frame_detections = {}
            
            for camera in self.cameras:
                if not hasattr(camera, 'detection') or camera.detection is None:
                    continue
                
                if not hasattr(camera.detection, 'detections_filtered') or not camera.detection.detections_filtered:
                    continue
                
                detections = camera.detection.detections_filtered
                frame = camera.detection.frame
                
                # 保存检测结果（深拷贝，避免后续修改影响）
                frame_detections[camera.id] = detections
                
                # 收集 crops
                temp_dir = Path(tempfile.gettempdir()) / "mcs_crops_temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                
                for det_idx, det in enumerate(detections):
                    x1, y1, x2, y2 = det.bbox
                    crop = frame[int(y1):int(y2), int(x1):int(x2)]
                    
                    if crop.size > 0:
                        crop_filename = f"frame{processed_count:06d}_cam{camera.id}_det{det_idx:03d}.jpg"
                        crop_path = temp_dir / crop_filename
                        cv2.imwrite(str(crop_path), crop)
                        
                        all_crops_data.append({
                            'frame_idx': processed_count,
                            'camera_id': camera.id,
                            'det': det,
                            'crop_path': crop_path
                        })
            
            all_frames_detections.append((processed_count, frame_detections))
            processed_count += 1
            frame_idx += 1
            
            if processed_count % 10 == 0:
                print(f"  已处理 {processed_count} 帧...")
        
        print(f"\n✓ 检测完成：共处理 {processed_count} 帧，收集 {len(all_crops_data)} 个 crops\n")
        
        if processed_count == 0:
            print("❌ 没有处理任何帧")
            return None
        
        # ============================================================
        # 第二阶段：全局批量颜色识别（分批处理）
        # ============================================================
        print(f"\n{'─'*70}")
        print("[阶段 2/3] 全局批量颜色识别（分批处理）")
        print(f"{'─'*70}\n")
        
        if all_crops_data:
            from ocr import batch_detect_jersey_colors
            
            total_crops = len(all_crops_data)
            print(f"总共 {total_crops} 个 crops，每批 {color_batch_size} 个")
            
            # 分批处理
            num_batches = (total_crops + color_batch_size - 1) // color_batch_size
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * color_batch_size
                end_idx = min(start_idx + color_batch_size, total_crops)
                batch_crops = all_crops_data[start_idx:end_idx]
                
                print(f"\n[批次 {batch_idx+1}/{num_batches}] 处理 {len(batch_crops)} 个 crops...")
                
                # 提取当前批次的 crop 路径
                crop_paths = [item['crop_path'] for item in batch_crops]
                
                # 批量检测颜色
                color_map = batch_detect_jersey_colors(crop_paths, api_key=None)
                
                # 更新 Detection 对象的颜色
                for item in batch_crops:
                    crop_filename = item['crop_path'].name
                    color = color_map.get(crop_filename, 'unknown')
                    item['det'].dominant_color = color
                
                print(f"  ✓ 批次 {batch_idx+1}/{num_batches} 完成")
            
            print(f"\n✓ 所有颜色识别完成（共 {num_batches} 批）\n")
            
            # 清理临时 crop 文件
            try:
                for item in all_crops_data:
                    if item['crop_path'].exists():
                        item['crop_path'].unlink()
                temp_crops_dir = Path(tempfile.gettempdir()) / "mcs_crops_temp"
                if temp_crops_dir.exists():
                    import shutil
                    shutil.rmtree(temp_crops_dir)
                print(f"✓ 清理临时文件完成\n")
            except Exception as e:
                print(f"⚠️ 清理临时文件失败: {e}\n")
        else:
            print("⚠️ 没有 crops 需要颜色识别\n")
        
        # ============================================================
        # 第三阶段：生成多个阈值的视频并拼接
        # ============================================================
        print(f"\n{'─'*70}")
        print("[阶段 3/3] 生成多个阈值的视频并拼接")
        print(f"{'─'*70}\n")
        
        # 输出路径
        output_dir = Path(f"store/{self.name}_mcs/position_video")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_video_path = output_dir / f"{self.name}_position_video_thresholds_combined.mp4"
        
        frame_count = 0
        
        with VideoWriterStream(str(output_video_path), fps=fps) as writer:
            for frame_idx, frame_detections in all_frames_detections:
                # 为每个阈值生成可视化帧
                threshold_frames = []
                
                for threshold_config in all_thresholds:
                    # 恢复当前帧的检测结果到相机
                    for camera in self.cameras:
                        if camera.id in frame_detections:
                            if not hasattr(camera, 'detection') or camera.detection is None:
                                camera.detection = type('obj', (object,), {})()
                            camera.detection.detections_filtered = frame_detections[camera.id]
                    
                    # 生成可视化图（不保存）
                    pos_img = self.detector._draw_all_detections_in_fov(
                        score_threshold=threshold_config['value'],
                        **kwargs
                    )
                    
                    if pos_img is not None:
                        # 转换为BGR格式（如果是RGBA）
                        if pos_img.shape[2] == 4:
                            # 将RGBA转换为BGR（白色背景）
                            bgr = pos_img[:, :, :3]
                            alpha = pos_img[:, :, 3:4] / 255.0
                            white_bg = np.ones_like(bgr) * 255
                            pos_img_bgr = (bgr * alpha + white_bg * (1 - alpha)).astype(np.uint8)
                        else:
                            pos_img_bgr = pos_img
                        
                        threshold_frames.append(pos_img_bgr)
                    else:
                        # 如果生成失败，创建一个空白图
                        empty_frame = np.ones((400, 600, 3), dtype=np.uint8) * 255
                        threshold_frames.append(empty_frame)
                
                # 拼接所有阈值的帧
                if layout == 'horizontal':
                    combined_frame = np.hstack(threshold_frames)
                elif layout == 'vertical':
                    combined_frame = np.vstack(threshold_frames)
                elif layout == 'grid':
                    # 网格拼接
                    n = len(threshold_frames)
                    cols = int(np.ceil(np.sqrt(n)))
                    rows = int(np.ceil(n / cols))
                    
                    # 填充到完整网格
                    while len(threshold_frames) < rows * cols:
                        threshold_frames.append(np.ones_like(threshold_frames[0]) * 255)
                    
                    # 按行拼接
                    grid_rows = []
                    for r in range(rows):
                        row_frames = threshold_frames[r*cols:(r+1)*cols]
                        grid_rows.append(np.hstack(row_frames))
                    combined_frame = np.vstack(grid_rows)
                else:
                    raise ValueError(f"不支持的布局方式: {layout}")
                
                # 写入拼接后的帧
                writer.write(combined_frame)
                frame_count += 1
                
                if frame_count % 10 == 0:
                    print(f"  已生成 {frame_count}/{processed_count} 帧...")
        
        print(f"\n✅ 拼接视频已保存: {output_video_path}")
        print(f"   总帧数: {frame_count}")
        print(f"   FPS: {fps}")
        print(f"   时长: {frame_count/fps:.2f}秒")
        
        print(f"\n{'='*70}")
        print(f"[MCS] 测试完成")
        print(f"{'='*70}\n")
        
        return {
            'video_path': str(output_video_path),
            'total_frames': frame_count,
            'fps': fps
        }

class MCSDetector:
    """
    MCS 检测器类
    
    职责：
    1. 运行所有相机的 YOLO 检测
    2. 过滤检测结果（基于 MCS 覆盖区域）
    3. 可视化检测框（detect/ 目录）
    4. 可视化球员位置（position/ 目录）
    
    目录结构：
        store/{name}_mcs/
            detect/              # 检测框可视化
                {name}_detect_combined.jpg   # 所有相机拼接图
                camera1_detect.jpg
                camera2_detect.jpg
                ...
            position/            # 球场位置可视化
                {name}_position_best.jpg
                {name}_position_second.jpg
    
    使用方法：
        mcs = MCS(cameras, config, name="wuchuan")
        
        # 方法1：分步调用
        mcs.detect()                      # 检测
        mcs.visualize_detections()        # 绘制检测框
        mcs.visualize_positions("best")   # 绘制位置图
        
        # 方法2：直接访问detector
        mcs.detector.detect()
        mcs.detector.visualize_detections()
        mcs.detector.visualize_positions()
    """
    
    def __init__(self, mcs: MCS):
        """
        初始化 MCS 检测器
        
        Args:
            mcs: MCS 系统实例
        """
        self.mcs = mcs
        self.name = mcs.name
        
        # 创建检测相关目录
        self.detect_dir = mcs.store_dir / "detect"
        self.position_dir = mcs.store_dir / "position"
        
        self.detect_dir.mkdir(parents=True, exist_ok=True)
        self.position_dir.mkdir(parents=True, exist_ok=True)
        
        # 检测统计信息
        self.last_detection_stats = None
        
        print(f"[MCSDetector] 初始化完成")
        print(f"  - 检测目录: {self.detect_dir}")
        print(f"  - 位置目录: {self.position_dir}")
    
    def detect(self, frames_dict=None, conf=0.25, verbose=False):
        """
        运行所有相机的 YOLO 检测
        
        默认行为：保留场地范围内的所有检测（不使用MCS分配图过滤）
        
        Args:
            frames_dict: 字典 {camera_id: frame}，如果为 None 则自动从视频读取
            conf: YOLO 置信度阈值
            verbose: 是否显示详细信息
            
        Returns:
            stats: 检测统计信息字典
                {
                    'cameras_detected': int,  # 有检测结果的相机数
                    'total_detections': int,  # 总检测数
                    'frames': {camera_id: frame},  # 保存的帧
                }
        """
        print(f"\n{'='*70}")
        print(f"[MCSDetector] 运行检测")
        print(f"{'='*70}")
        
        stats = {
            'cameras_detected': 0,
            'total_detections': 0,
            'frames': {}
        }
        
        # 步骤: 运行 YOLO 检测并过滤
        print(f"\n[步骤] 运行 YOLO 检测并过滤...")
        for camera in self.mcs.cameras:
            # 获取帧
            if frames_dict is not None:
                if camera.id not in frames_dict:
                    print(f"  ⚠️ 相机 {camera.id}: frames_dict 中没有对应的帧")
                    continue
                frame = frames_dict[camera.id].copy()
            else:
                # 自动读取帧 (从 VideoManager 中读取)
                if not hasattr(camera, 'video_manager') or camera.video_manager is None:
                    print(f"  ⚠️ 相机 {camera.id}: 没有视频管理器")
                    continue
                if not hasattr(camera.video_manager, 'cap') or camera.video_manager.cap is None:
                    print(f"  ⚠️ 相机 {camera.id}: 视频捕获对象未初始化")
                    continue
                ret, frame = camera.video_manager.cap.read()
                if not ret:
                    print(f"  ⚠️ 相机 {camera.id}: 无法读取帧")
                    continue
            
            # 检查检测器是否初始化
            if not hasattr(camera, 'detection') or camera.detection is None:
                print(f"  ⚠️ 相机 {camera.id}: 检测器未初始化")
                continue
            
            # 运行检测（只保留场地范围内的检测）
            # 新API: detect() 返回 Detection 对象列表
            detections_filtered = camera.detection.detect(
                frame, conf=conf, verbose=verbose
            )
            
            # 检查检测结果
            if hasattr(camera.detection, 'raw_detections') and camera.detection.raw_detections is not None:
                num_detections = len(camera.detection.raw_detections)
                # 过滤后的检测数量
                num_in_range = len(detections_filtered) if detections_filtered else 0
                
                if num_detections > 0:
                    print(f"  ✅ 相机 {camera.id}: {num_detections} 个检测 -> {num_in_range} 个在场地范围内")
                    stats['frames'][camera.id] = frame
                    stats['cameras_detected'] += 1
                    stats['total_detections'] += num_in_range
                else:
                    print(f"  ℹ️ 相机 {camera.id}: 未检测到目标")
            else:
                print(f"  ❌ 相机 {camera.id}: 检测失败")
        
        # 保存统计信息
        self.last_detection_stats = stats
        
        # 输出汇总
        print(f"\n{'='*70}")
        print(f"[MCSDetector] 检测完成")
        print(f"  - 相机数: {stats['cameras_detected']}/{len(self.mcs.cameras)}")
        print(f"  - 总检测数: {stats['total_detections']}")
        print(f"{'='*70}\n")
        
        return stats
    
    def visualize_detections(self, save_individual=True, save_combined=True):
        """
        绘制检测框可视化
        
        Args:
            save_individual: 是否保存单个相机的检测图
            save_combined: 是否保存组合检测大图
            
        Returns:
            results: 字典
                {
                    'individual_images': {camera_id: annotated_frame},
                    'combined_image': combined_frame
                }
        
        示例用法：
            mcs.detect()                  # 先检测
            mcs.visualize_detections()    # 再可视化
        """
        if self.last_detection_stats is None:
            print("[MCSDetector] ⚠️ 请先运行 detect() 方法")
            return None
        
        print(f"\n[MCSDetector] 绘制检测框...")
        
        results = {
            'individual_images': {},
            'combined_image': None
        }
        
        frames = self.last_detection_stats['frames']
        
        # 绘制每个相机的检测框
        for camera in self.mcs.cameras:
            if camera.id not in frames:
                continue
            
            # 检查检测器是否存在
            if not hasattr(camera, 'detection') or camera.detection is None:
                continue
            
            # 保存帧供 draw_detections 使用
            camera.detection.frame = frames[camera.id]
            
            # 绘制检测框
            annotated_frame = camera.detection.draw_detections(show_all=True, show_pitch_pos=False)
            
            if annotated_frame is not None:
                results['individual_images'][camera.id] = annotated_frame
                
                # 保存单个相机的检测图
                if save_individual:
                    save_path = self.detect_dir / f"{camera.name}_detect.jpg"
                    imwritex(str(save_path), annotated_frame)
                    print(f"  ✅ {camera.name}_detect.jpg")
        
        # 生成组合检测大图
        if save_combined and len(results['individual_images']) > 0:
            combined_image = self._create_combined_detection_view(results['individual_images'])
            
            if combined_image is not None:
                results['combined_image'] = combined_image
                combined_path = self.detect_dir / f"{self.name}_detect_combined.jpg"
                imwritex(str(combined_path), combined_image)
                print(f"  ✅ {self.name}_detect_combined.jpg")
        
        print(f"[MCSDetector] 检测框可视化完成\n")
        
        return results
    
    def _draw_all_detections_in_fov(self, scale=0.1, padding=50, radius=3, background_color=None, score_threshold=None):
        """
        内部方法：绘制所有相机视野内的所有检测（不使用MCS过滤）
        
        在这个模式下：
        - 不绘制分配区域背景
        - 绘制所有相机视野内的所有检测
        - 根据检测的最终得分设置透明度（得分越低越透明）
        - 不同相机使用不同颜色
        - 背景默认全透明
        
        Args:
            scale: 球场缩放比例
            padding: 球场边距
            radius: 球员圆点半径
            background_color: 球场背景颜色，默认为 None（全透明）
                            可选: 黑色 sv.Color(0, 0, 0)，白色 sv.Color(255, 255, 255)，
                                 深灰色 sv.Color(40, 40, 40)，绿色草地 sv.Color(34, 139, 34)
            score_threshold: 最终分数阈值（0.0-1.0），只绘制 final_score >= threshold 的检测
                           None 表示不过滤（默认）
            
        Returns:
            pitch_image: 绘制后的球场图像（RGBA格式，带透明通道）
        """
        from tools.annotation.pitch_annotation_tool import draw_pitch
        from tools.transform import pitch_to_canvas_pixel
        
        # 默认背景全透明
        if background_color is None:
            # 创建透明背景
            use_transparent = True
            # 先用黑色生成球场线（后续会替换背景为透明）
            background_color = sv.Color(0, 0, 0)
        else:
            use_transparent = False
        
        # 生成球场图（RGB格式）
        pitch_image_rgb = draw_pitch(
            self.mcs.pitch_config, 
            scale=scale, 
            padding=padding,
            background_color=background_color
        )
        
        # 转换为 RGBA 格式
        pitch_image = cv2.cvtColor(pitch_image_rgb, cv2.COLOR_BGR2BGRA)
        
        # 如果需要透明背景，将背景色设为透明
        if use_transparent:
            # 创建背景掩码（黑色部分 = 背景）
            gray = cv2.cvtColor(pitch_image_rgb, cv2.COLOR_BGR2GRAY)
            # 背景是黑色(0)，球场线是白色(255)
            background_mask = (gray < 10)  # 接近黑色的部分视为背景
            # 设置 alpha 通道：背景=0（透明），其他=255（不透明）
            pitch_image[:, :, 3] = np.where(background_mask, 0, 255)
        
        # 使用统一的颜色映射
        colors_map = self.mcs.config.CAMERA_COLORS
        
        # 收集所有检测及其信息
        all_detections = []
        
        print(f"\n[MCSDetector] === 开始收集检测（all模式）===")
        
        for camera in self.mcs.cameras:
            # 检查检测器是否存在
            if not hasattr(camera, 'detection') or camera.detection is None:
                print(f"[MCSDetector] ⚠️ 相机 {camera.id}: 检测器未初始化")
                continue
            
            # 检查是否有检测结果
            if not hasattr(camera.detection, 'detections_filtered') or not camera.detection.detections_filtered:
                print(f"[MCSDetector] ⚠️ 相机 {camera.id}: 缺少 detections_filtered")
                continue
            
            # 获取所有检测（Detection对象列表）
            detections = camera.detection.detections_filtered
            
            if len(detections) == 0:
                print(f"[MCSDetector] ℹ️ 相机 {camera.id}: 视野内没有检测")
                continue
            
            print(f"[MCSDetector] ✅ 相机 {camera.id}: 收集 {len(detections)} 个检测点")
            
            # 收集该相机的所有检测（直接使用Detection对象）
            for det in detections:
                all_detections.append({
                    'detection': det,  # 保存整个Detection对象
                    'camera_id': camera.numeric_id,
                    'color': colors_map.get(camera.numeric_id, (255, 255, 255))
                })
        
        if len(all_detections) == 0:
            print(f"[MCSDetector] ⚠️ 所有相机都没有检测点")
            return pitch_image
        
        print(f"[MCSDetector] 总计收集 {len(all_detections)} 个检测点")
        
        # 应用分数阈值过滤
        if score_threshold is not None:
            original_count = len(all_detections)
            
            # 判断是单阈值还是区间阈值
            if isinstance(score_threshold, (tuple, list)) and len(score_threshold) == 2:
                # 区间阈值 [min, max]
                min_thr, max_thr = score_threshold
                all_detections = [
                    det_info for det_info in all_detections 
                    if min_thr <= det_info['detection'].final_score <= max_thr
                ]
                filtered_count = len(all_detections)
                print(f"[MCSDetector] 应用分数区间 [{min_thr:.2f}, {max_thr:.2f}]: {original_count} → {filtered_count} 个检测点")
            else:
                # 单个下限阈值
                all_detections = [
                    det_info for det_info in all_detections 
                    if det_info['detection'].final_score >= score_threshold
                ]
                filtered_count = len(all_detections)
                print(f"[MCSDetector] 应用分数下限 >= {score_threshold:.2f}: {original_count} → {filtered_count} 个检测点")
            
            if len(all_detections) == 0:
                print(f"[MCSDetector] ⚠️ 过滤后没有检测点")
                return pitch_image
        
        print()
        
        # 转换所有位置到画布坐标
        positions = np.array([det['detection'].pitch_position for det in all_detections])
        canvas_positions = pitch_to_canvas_pixel(positions, scale=scale, padding=padding)
        
        # 统计最终得分的分布（用于调试）
        final_scores = [det['detection'].final_score for det in all_detections]
        print(f"[Debug] 最终得分统计: min={min(final_scores):.3f}, max={max(final_scores):.3f}, mean={np.mean(final_scores):.3f}")
        
        # 绘制所有检测点（每个点独立进行alpha混合）
        for i, (det_info, canvas_pos) in enumerate(zip(all_detections, canvas_positions)):
            x, y = int(canvas_pos[0]), int(canvas_pos[1])
            
            # 检查坐标是否在图像范围内
            if not (0 <= x < pitch_image.shape[1] and 0 <= y < pitch_image.shape[0]):
                continue
            
            # 从Detection对象获取所有预计算的信息
            det = det_info['detection']
            
            # 使用最终得分计算透明度
            # final_score = quality_score × sigmoid(confidence)
            # 已经在Detection.__post_init__中计算完成
            final_score = det.final_score
            
            # 将最终得分线性映射到透明度 [0.0, 1.0]
            # 分数为 1.0 → 完全不透明，分数为 0.0 → 完全透明
            alpha = np.clip(final_score, 0.0, 1.0)
            
            # 打印前几个点的详细信息（调试用）
            if i < 10:
                print(f"  [Debug] 检测点 {i}: "
                      f"camera_id={det_info['camera_id']}, "
                      f"grid=({det.grid_position[0]},{det.grid_position[1]}), "
                      f"grid_size={det.grid_size:.0f}cm, "
                      f"conf={det.confidence:.3f}, "
                      f"conf_weight={det.confidence_weight:.3f}, "
                      f"Q={det.quality_score:.3f}, "
                      f"final_score={final_score:.3f}, "
                      f"alpha={alpha:.3f}, "
                      f"pos=({det.pitch_position[0]:.0f},{det.pitch_position[1]:.0f})cm")
            
            # 相机颜色（外圈）
            camera_color = det_info['color']
            
            # 球衣颜色（内圈）
            jersey_color_map = {
                'white': (255, 255, 255),
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'blue': (255, 0, 0),
                'black': (50, 50, 50),
                'unknown': (128, 128, 128)
            }
            jersey_color = jersey_color_map.get(
                getattr(det, 'dominant_color', 'unknown'),
                (128, 128, 128)
            )
            
            # 为当前检测点创建独立的overlay（BGRA格式）
            overlay = pitch_image.copy()
            
            # 在overlay上绘制圆点（全不透明）
            # 先画球衣颜色实心圆（填充）- 注意BGRA格式需要4个值
            jersey_color_rgba = tuple(list(jersey_color) + [255])
            camera_color_rgba = tuple(list(camera_color) + [255])
            cv2.circle(overlay, (x, y), radius, jersey_color_rgba, -1)
            # 再画相机颜色边框（外圈）
            cv2.circle(overlay, (x, y), radius, camera_color_rgba, 2)
            
            # 创建圆形掩码
            mask = np.zeros(pitch_image.shape[:2], dtype=np.uint8)
            cv2.circle(mask, (x, y), radius + 2, 255, -1)
            
            # 在圆形区域内进行alpha混合（包括透明度）
            roi_indices = np.where(mask > 0)
            if len(roi_indices[0]) > 0:
                # 对 BGR 通道进行混合
                pitch_image[roi_indices[0], roi_indices[1], :3] = (
                    alpha * overlay[roi_indices[0], roi_indices[1], :3] + 
                    (1 - alpha) * pitch_image[roi_indices[0], roi_indices[1], :3]
                ).astype(np.uint8)
                # 对 Alpha 通道进行混合（确保检测点区域不透明）
                pitch_image[roi_indices[0], roi_indices[1], 3] = np.maximum(
                    pitch_image[roi_indices[0], roi_indices[1], 3],
                    (alpha * 255).astype(np.uint8)
                )
        
        print(f"[MCSDetector] ✅ 总计绘制了 {len(all_detections)} 个检测点（视野模式，透明度由最终得分决定）\n")
        
        return pitch_image
    
    def save_detections_cache(self, cache_name="detections_cache"):
        """
        保存当前所有相机的detection信息到缓存文件
        
        Args:
            cache_name: 缓存文件名（不含扩展名）
            
        Returns:
            cache_path: 缓存文件路径
        """
        import json
        
        if self.last_detection_stats is None:
            print("[MCSDetector] ⚠️ 请先运行 detect() 方法")
            return None
        
        # 创建缓存目录
        cache_dir = self.position_dir / "detections"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # 收集所有相机的detection信息
        cache_data = {
            'name': self.name,
            'timestamp': str(np.datetime64('now')),
            'cameras': {}
        }
        
        print(f"\n[MCSDetector] 缓存detection信息...")
        
        for camera in self.mcs.cameras:
            if not hasattr(camera, 'detection') or camera.detection is None:
                continue
            
            if not hasattr(camera.detection, 'detections_filtered') or not camera.detection.detections_filtered:
                continue
            
            # 转换Detection对象为可序列化的字典
            detections_list = []
            for det in camera.detection.detections_filtered:
                # 使用Detection类自带的to_dict()方法，确保所有字段都被正确序列化（包括ReID特征）
                det_dict = det.to_dict()
                # 添加grid_size字段（to_dict中未包含）
                det_dict['grid_size'] = float(det.grid_size)
                detections_list.append(det_dict)
            
            cache_data['cameras'][camera.numeric_id] = {
                'camera_id': camera.numeric_id,
                'num_detections': len(detections_list),
                'detections': detections_list
            }
            
            print(f"  ✅ 相机 {camera.numeric_id}: {len(detections_list)} 个检测")
        
        # 保存为JSON文件
        cache_path = cache_dir / f"{cache_name}.json"
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2)
        
        print(f"\n✅ Detection缓存已保存: {cache_path}")
        print(f"   相机数: {len(cache_data['cameras'])}")
        print(f"   总检测数: {sum(cam['num_detections'] for cam in cache_data['cameras'].values())}")
        
        return cache_path
    
    def load_detections_cache(self, cache_name="detections_cache", as_objects=False):
        """
        从缓存文件加载detection信息
        
        Args:
            cache_name: 缓存文件名（不含扩展名）
            as_objects: 是否将字典转换为Detection对象（默认False，返回原始字典）
            
        Returns:
            cache_data: 缓存数据字典，如果as_objects=True，则字典中的detections会被转换为Detection对象列表
            
        示例:
            # 加载为字典（默认）
            cache = mcs.detector.load_detections_cache("with_reid")
            det_dict = cache['cameras']['1']['detections'][0]  # dict
            
            # 加载为Detection对象
            cache = mcs.detector.load_detections_cache("with_reid", as_objects=True)
            det_obj = cache['cameras']['1']['detections'][0]  # Detection对象
            print(det_obj.reid_feature.shape)  # numpy数组
        """
        import json
        from camera import Detection
        
        cache_dir = self.position_dir / "detections"
        cache_path = cache_dir / f"{cache_name}.json"
        
        if not cache_path.exists():
            print(f"[MCSDetector] ❌ 缓存文件不存在: {cache_path}")
            return None
        
        print(f"\n[MCSDetector] 加载detection缓存: {cache_path}")
        
        with open(cache_path, 'r', encoding='utf-8') as f:
            cache_data = json.load(f)
        
        print(f"  ✅ 加载成功")
        print(f"     缓存时间: {cache_data['timestamp']}")
        print(f"     相机数: {len(cache_data['cameras'])}")
        print(f"     总检测数: {sum(cam['num_detections'] for cam in cache_data['cameras'].values())}")
        
        # 如果需要转换为Detection对象
        if as_objects:
            print(f"  🔄 转换为Detection对象...")
            for cam_id, cam_data in cache_data['cameras'].items():
                detections_list = []
                for det_dict in cam_data['detections']:
                    det_obj = Detection.from_dict(det_dict)
                    detections_list.append(det_obj)
                cam_data['detections'] = detections_list
            print(f"  ✅ 转换完成")
        
        return cache_data
    
    # TODO：可视化方法抽象出基类，继承以统一样式，减少冗余
    def visualize_detections_per_camera(self, save=True, scale=0.1, 
                                       padding=50, radius=8, 
                                       background_color=None,
                                       score_threshold=None,
                                       cache_name=None,
                                       target_height=500):
        """
        分别可视化每个相机的detection在球场上的位置，拼接为一行大图
        
        Args:
            save: 是否保存图片
            scale: 球场缩放比例
            padding: 边距
            radius: 球员圆点半径
            background_color: 背景颜色（None=透明）
            score_threshold: 分数阈值过滤
            cache_name: 如果提供，从缓存加载数据而非当前detection
            target_height: 拼接时统一的目标高度（像素）
            
        Returns:
            combined_image: 拼接后的大图（RGBA格式）
        """
        from tools.annotation.pitch_annotation_tool import draw_pitch
        from tools.transform import pitch_to_canvas_pixel
        import supervision as sv
        
        print(f"\n{'='*70}")
        print(f"[MCSDetector] 分相机可视化检测位置")
        print(f"{'='*70}")
        
        # 准备数据源
        if cache_name is not None:
            # 从缓存加载
            cache_data = self.load_detections_cache(cache_name)
            if cache_data is None:
                return None
            use_cache = True
        else:
            # 使用当前detection
            if self.last_detection_stats is None:
                print("[MCSDetector] ⚠️ 请先运行 detect() 方法，或使用 cache_name 参数加载缓存")
                return None
            use_cache = False
        
        # 相机颜色映射
        colors_map = {
            1: (0, 0, 255),      # 红色
            2: (0, 255, 0),      # 绿色
            3: (255, 0, 0),      # 蓝色
            4: (0, 255, 255),    # 黄色
            5: (255, 0, 255),    # 品红
        }
        
        camera_images = []  # [(camera_id, image), ...]
        
        # 为每个相机生成单独的位置图
        if use_cache:
            camera_ids = sorted(cache_data['cameras'].keys(), key=int)
        else:
            camera_ids = [cam.numeric_id for cam in self.mcs.cameras 
                         if hasattr(cam, 'detection') and cam.detection is not None]
        
        for camera_id in camera_ids:
            print(f"\n[相机 {camera_id}] 绘制位置图...")
            
            # 创建球场背景
            if background_color is None:
                # 使用黑色背景生成球场（后续会设为透明）
                use_transparent = True
                import supervision as sv
                temp_bg_color = sv.Color(0, 0, 0)
            else:
                use_transparent = False
                temp_bg_color = background_color
            
            pitch_canvas = draw_pitch(
                config=self.mcs.scheduler.config.pitch_config,
                scale=scale,
                padding=padding,
                background_color=temp_bg_color
            )
            
            # 转换为RGBA格式
            if pitch_canvas.shape[2] == 3:
                pitch_image = cv2.cvtColor(pitch_canvas, cv2.COLOR_BGR2BGRA)
            else:
                pitch_image = pitch_canvas.copy()
            
            # 如果需要透明背景，将背景色（黑色部分）设为透明，保留球场线条
            if use_transparent:
                # 创建背景掩码（黑色部分 = 背景）
                gray = cv2.cvtColor(pitch_canvas, cv2.COLOR_BGR2GRAY)
                # 背景是黑色(0)，球场线是白色(255)
                background_mask = (gray < 10)  # 接近黑色的部分视为背景
                # 设置 alpha 通道：背景=0（透明），球场线=255（不透明）
                pitch_image[:, :, 3] = np.where(background_mask, 0, 255)
            
            # 收集该相机的检测
            detections = []
            
            if use_cache:
                # 从缓存加载
                if str(camera_id) not in cache_data['cameras']:
                    print(f"  ⚠️ 缓存中没有相机 {camera_id} 的数据")
                    continue
                
                cam_data = cache_data['cameras'][str(camera_id)]
                for det_dict in cam_data['detections']:
                    # 应用分数阈值过滤
                    if score_threshold is not None:
                        if isinstance(score_threshold, (tuple, list)) and len(score_threshold) == 2:
                            min_thr, max_thr = score_threshold
                            if not (min_thr <= det_dict['final_score'] <= max_thr):
                                continue
                        else:
                            if det_dict['final_score'] < score_threshold:
                                continue
                    
                    detections.append({
                        'pitch_position': np.array(det_dict['pitch_position']),
                        'final_score': det_dict['final_score'],
                        'dominant_color': det_dict['dominant_color'],
                        'confidence': det_dict['confidence'],
                        'quality_score': det_dict['quality_score']
                    })
            else:
                # 从当前detection获取
                camera = next((cam for cam in self.mcs.cameras if cam.numeric_id == camera_id), None)
                if camera is None:
                    continue
                
                if not hasattr(camera.detection, 'detections_filtered') or not camera.detection.detections_filtered:
                    print(f"  ⚠️ 没有检测结果")
                    continue
                
                for det in camera.detection.detections_filtered:
                    # 应用分数阈值过滤
                    if score_threshold is not None:
                        if isinstance(score_threshold, (tuple, list)) and len(score_threshold) == 2:
                            min_thr, max_thr = score_threshold
                            if not (min_thr <= det.final_score <= max_thr):
                                continue
                        else:
                            if det.final_score < score_threshold:
                                continue
                    
                    detections.append({
                        'pitch_position': det.pitch_position,
                        'final_score': det.final_score,
                        'dominant_color': det.dominant_color,
                        'confidence': det.confidence,
                        'quality_score': det.quality_score
                    })
            
            if len(detections) == 0:
                print(f"  ⚠️ 过滤后没有检测点")
                # 仍然保存空图
                camera_images.append((f"Camera {camera_id}", pitch_image))
                continue
            
            print(f"  绘制 {len(detections)} 个检测点...")
            
            # 转换位置到画布坐标
            positions = np.array([det['pitch_position'] for det in detections])
            canvas_positions = pitch_to_canvas_pixel(positions, scale=scale, padding=padding)
            
            # 获取相机颜色
            camera_color = colors_map.get(camera_id, (255, 255, 255))
            
            # 绘制每个检测点
            for det, canvas_pos in zip(detections, canvas_positions):
                x, y = int(canvas_pos[0]), int(canvas_pos[1])
                
                if not (0 <= x < pitch_image.shape[1] and 0 <= y < pitch_image.shape[0]):
                    continue
                
                # 透明度由final_score决定
                alpha = np.clip(det['final_score'], 0.0, 1.0)
                
                # 球衣颜色（内圈）
                jersey_color_map = {
                    'white': (255, 255, 255),
                    'red': (0, 0, 255),
                    'green': (0, 255, 0),
                    'blue': (255, 0, 0),
                    'black': (50, 50, 50),
                    'unknown': (128, 128, 128)
                }
                jersey_color = jersey_color_map.get(det['dominant_color'], (128, 128, 128))
                
                # 为当前检测点创建独立的overlay（BGRA格式）
                overlay = pitch_image.copy()
                
                # 在overlay上绘制圆点（全不透明）- BGRA格式需要4个值
                jersey_color_rgba = tuple(list(jersey_color) + [255])
                camera_color_rgba = tuple(list(camera_color) + [255])
                
                # 先画球衣颜色实心圆（填充）
                cv2.circle(overlay, (x, y), radius, jersey_color_rgba, -1)
                # 再画相机颜色边框（外圈）
                cv2.circle(overlay, (x, y), radius, camera_color_rgba, 2)
                
                # 创建圆形掩码
                mask = np.zeros(pitch_image.shape[:2], dtype=np.uint8)
                cv2.circle(mask, (x, y), radius + 2, 255, -1)
                
                # 在圆形区域内进行alpha混合（包括透明度）
                roi_indices = np.where(mask > 0)
                if len(roi_indices[0]) > 0:
                    # 对 BGR 通道进行混合
                    pitch_image[roi_indices[0], roi_indices[1], :3] = (
                        alpha * overlay[roi_indices[0], roi_indices[1], :3] + 
                        (1 - alpha) * pitch_image[roi_indices[0], roi_indices[1], :3]
                    ).astype(np.uint8)
                    # 对 Alpha 通道进行混合（确保检测点区域不透明）
                    pitch_image[roi_indices[0], roi_indices[1], 3] = np.maximum(
                        pitch_image[roi_indices[0], roi_indices[1], 3],
                        (alpha * 255).astype(np.uint8)
                    )
            
            print(f"  ✅ 完成")
            camera_images.append((f"Camera {camera_id}", pitch_image))
        
        if len(camera_images) == 0:
            print(f"\n❌ 没有成功生成任何相机的位置图")
            return None
        
        # 拼接为一行大图
        print(f"\n{'─'*60}")
        print(f"拼接图片：{len(camera_images)} 个相机")
        print(f"{'─'*60}")
        
        def resize_and_label_camera_images(images_list, target_height=500, label_width=120):
            """调整图片大小并添加左侧标签（类似test_positions_thresholds的样式）"""
            processed = []
            
            for label, img in images_list:
                # 调整图片高度
                h, w = img.shape[:2]
                scale_factor = target_height / h
                new_w = int(w * scale_factor)
                resized = cv2.resize(img, (new_w, target_height))
                
                # 创建标签区域（左侧，RGBA格式，白色不透明）
                label_img = np.ones((target_height, label_width, 4), dtype=np.uint8) * 255
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                thickness = 2
                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
                text_x = (label_width - text_size[0]) // 2
                text_y = (target_height + text_size[1]) // 2
                cv2.putText(label_img, label, (text_x, text_y), font, font_scale, (0, 0, 0, 255), thickness)
                
                # 水平拼接标签和图片
                combined = np.hstack([label_img, resized])
                processed.append(combined)
            
            return processed
        
        # 处理并拼接
        processed_images = resize_and_label_camera_images(camera_images, target_height=target_height, label_width=120)
        combined_image = np.hstack(processed_images)
        
        print(f"✓ 拼接完成: {combined_image.shape[1]}x{combined_image.shape[0]}")
        
        # 保存
        if save:
            cache_dir = self.position_dir / "detections"
            cache_dir.mkdir(parents=True, exist_ok=True)
            
            if cache_name:
                output_name = f"{cache_name}_per_camera.png"
            else:
                output_name = f"{self.name}_detections_per_camera.png"
            
            output_path = cache_dir / output_name
            cv2.imwrite(str(output_path), combined_image)
            print(f"\n✅ 已保存: {output_path}")
            print(f"   尺寸: {combined_image.shape[1]}x{combined_image.shape[0]}")
            print(f"   格式: PNG (RGBA)")
        
        print(f"\n{'='*70}")
        print(f"[MCSDetector] 完成")
        print(f"{'='*70}\n")
        
        return combined_image
    
    def visualize_positions(self, save=True, scale=0.1, 
                           padding=50, radius=8, background_color=None,
                           score_threshold=None):
        """
        绘制全场球员位置图
        
        显示所有相机视野内的所有检测，透明度由检测的final_score决定。
        
        Args:
            save: 是否保存
            scale: 球场缩放比例
            padding: 球场边距
            radius: 球员圆点半径
            background_color: 球场背景颜色
                            默认为 None（透明背景）
                            可选: 黑色 sv.Color(0, 0, 0)，白色 sv.Color(255, 255, 255)，
                                 深灰色 sv.Color(40, 40, 40)，绿色草地 sv.Color(34, 139, 34)
            score_threshold: 分数阈值过滤
                           - None: 不过滤（默认）
                           - 单个值（float）: 下限阈值，只绘制 final_score >= threshold 的检测
                             例如: 0.3 表示只显示分数 >= 0.3 的检测
                           - 元组/列表 (min, max): 区间阈值，只绘制 min <= final_score <= max 的检测
                             例如: (0.3, 0.5) 表示只显示分数在 [0.3, 0.5] 区间的检测
            
        Returns:
            position_image: 位置图像
        
        示例用法：
            mcs.detect()                          # 先检测
            mcs.visualize_positions()             # 绘制所有视野内检测
            mcs.visualize_positions(score_threshold=0.3)        # 分数 >= 0.3
            mcs.visualize_positions(score_threshold=(0.3, 0.5)) # 分数在 [0.3, 0.5]
        """
        if self.last_detection_stats is None:
            print("[MCSDetector] ⚠️ 请先运行 detect() 方法")
            return None
        
        print(f"\n[MCSDetector] 绘制位置图...")
        
        # 调用自己的内部方法绘制
        position_image = self._draw_all_detections_in_fov(
            scale=scale,
            padding=padding,
            radius=radius,
            background_color=background_color,
            score_threshold=score_threshold
        )
        
        if position_image is None:
            print(f"[MCSDetector] ❌ 位置图生成失败\n")
            return None
        
        # 保存
        if save:
            # 文件名处理
            filename_suffix = "fov"
            
            # 添加分数阈值信息到文件名
            if score_threshold is not None:
                if isinstance(score_threshold, (tuple, list)) and len(score_threshold) == 2:
                    # 区间阈值
                    min_thr, max_thr = score_threshold
                    threshold_str = f"_thr{min_thr:.2f}-{max_thr:.2f}".replace('.', 'p')
                else:
                    # 单个下限阈值
                    threshold_str = f"_thr{score_threshold:.2f}".replace('.', 'p')
                filename_suffix = f"{filename_suffix}{threshold_str}"
            
            # 检查是否有 alpha 通道（透明背景）
            if position_image.shape[2] == 4:  # RGBA格式
                # 使用 PNG 格式保存透明图像
                save_path = self.position_dir / f"{self.name}_position_{filename_suffix}.png"
                cv2.imwrite(str(save_path), position_image)
                print(f"  ✅ {self.name}_position_{filename_suffix}.png (透明背景)")
            else:  # BGR格式
                # 使用 JPG 格式
                save_path = self.position_dir / f"{self.name}_position_{filename_suffix}.jpg"
                imwritex(str(save_path), position_image)
                print(f"  ✅ {self.name}_position_{filename_suffix}.jpg")
        
        print(f"[MCSDetector] 位置图完成\n")
        
        return position_image
    
    def _create_combined_detection_view(self, individual_images):
        """
        创建组合检测视图（将所有相机的检测图拼接成大图）
        
        Args:
            individual_images: 字典 {camera_id: annotated_frame}
            
        Returns:
            combined_image: 拼接后的大图
        """
        if not individual_images:
            return None
        
        # 按相机 ID 排序，准备数据
        sorted_items = sorted(individual_images.items(), key=lambda x: x[0])
        
        images_data = []
        for cam_id, img in sorted_items:
            # 查找相机名称
            camera_name = f"Camera {cam_id}"
            for camera in self.mcs.cameras:
                if camera.id == cam_id:
                    camera_name = camera.name
                    break
            
            images_data.append((camera_name, img))
        
        # 使用通用工具创建组合图
        final = create_combined_image_layout(
            images_data=images_data,
            title=f"MCS Detection: {self.name} - {len(images_data)} Cameras",
            layout='horizontal',
            target_height=800,
            label_position='top',
            label_height=60,
            header_height=80
        )
        
        return final
    
    def detect_from_video(self, frame_indices=None):
        """
        从视频中提取指定帧并运行检测（不含可视化）
        
        Args:
            frame_indices: 帧索引列表，如 [100, 200, 300]
                          如果为 None，使用当前帧
            
        Returns:
            all_stats: 列表，每个元素对应一帧的检测统计
            
        示例用法：
            # 检测当前帧
            stats = mcs.detector.detect_from_video()
            
            # 检测指定帧
            all_stats = mcs.detector.detect_from_video([100, 200, 300])
        """
        if frame_indices is None:
            # 检测当前帧
            return [self.detect()]
        
        all_stats = []
        
        for frame_idx in frame_indices:
            print(f"\n[MCSDetector] 处理帧 {frame_idx}")
            
            # 为每个相机设置帧位置并读取
            frames_dict = {}
            for camera in self.mcs.cameras:
                if hasattr(camera, 'video_manager') and camera.video_manager is not None:
                    if hasattr(camera.video_manager, 'cap') and camera.video_manager.cap is not None:
                        camera.video_manager.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = camera.video_manager.cap.read()
                        if ret:
                            frames_dict[camera.id] = frame
            
            # 运行检测
            stats = self.detect(frames_dict=frames_dict)
            stats['frame_index'] = frame_idx
            all_stats.append(stats)
        
        return all_stats

if __name__ == "__main__":
    from soccer import *
    from camera import Camera
    import cv2

    # %%
    config = HalfWuChuanConfiguration()
    # %%
    video_path_1 = "D:/Edge/4v4+3-30s.mp4"
    # video_path_2 = "D:/视频/wuchuan/wuchuan2_low_1m.mp4"
    # video_path_3 = "C:/Users/x1919/Videos/视角1_tracking.mp4"

    camera_1 = Camera(video_path_1, config, camera_id=1, camera_name="wuchuan0430", enable_distortion_correction=False)
    # camera_2 = Camera(video_path_2, config, camera_id=2, camera_name="camera2", mcs_name="wuchuan")
    # camera_3 = Camera(video_path_3, config, camera_id=3, camera_name="camera3")
    # %%
    # cameras = [camera_1, camera_2]
    # mcs = MCS(cameras, config, name="wuchuan")

    # mcs.detect()
    # mcs.detect_colors()
    # mcs.save_detections_cache(cache_name="with_reid")
    # 步骤 3: 可视化 - 不同阈值区间对比
    # print("\n" + "="*60)
    # print("开始生成不同阈值区间的可视化对比图")
    # print("="*60 + "\n")
    
    # 定义阈值区间
    # threshold_ranges = [
    #     (0.0, 0.4),   
    #     (0.4, 0.6),  
    #     (0.6, 1.0),   
    # ]
    
    # 生成30张序列帧的阈值对比图
    # 每秒采样一帧，共30秒，不使用批处理API
    # print(f"\n{'='*60}")
    # print("生成30张序列帧的阈值对比图（单独API调用）...")
    # print(f"{'='*60}\n")
    # saved_paths = mcs.test_positions_sequential_frames(
    #     threshold_ranges=threshold_ranges,
    #     single_thresholds=None,  # 不使用单阈值
    #     num_frames=30,  # 生成30张图
    #     fps=1.0,  # 每秒采样一帧
    #     conf=0.25,
    #     save_dir="30s",  # 保存到 position/30s/ 目录
    #     use_batch_api=False,  # 不使用批处理API
    #     radius=6
    # )
    # print(f"\n✅ 完成！共生成 {len(saved_paths)} 张图片")
    
    # mcs.visualize_detections_per_camera(
    #     save=True,
    #     score_threshold=None,
    #     radius=6,
    #     # cache_name="test_frame_0126"
    # )

    # # 也可以生成单阈值（下限）的可视化
    # single_thresholds = [0.2, 0.4,0.5, 0.6, 0.8]
    
    # # 使用 test_positions_video_thresholds 生成区间可视化拼接视频
    # print(f"\n{'='*60}")
    # print("生成多个阈值区间的可视化拼接视频...")
    # print(f"{'='*60}")
    
    # stats = mcs.test_positions_video_thresholds(
    #     threshold_ranges=threshold_ranges,
    #     single_thresholds=None,  # 不使用单阈值
    #     max_frames=300,  # 处理300帧
    #     fps=5.0,
    #     conf=0.25,
    #     sample_interval=0.5,  # 每0.3秒采样一次
    #     layout='horizontal',  # 水平拼接三个视频
    #     color_batch_size=500,
    #     background_color=sv.Color(200, 200, 200),  # 浅灰色背景
    #     radius=6
    # )
    
    # print("\n" + "="*60)
    # print("✅ 阈值区间可视化拼接视频生成完成！")
    # print("="*60)
    # if stats:
    #     print(f"输出文件: {stats.get('video_path', 'N/A')}")
    #     print(f"总帧数: {stats.get('total_frames', 'N/A')}")
    #     print(f"帧率: {stats.get('fps', 'N/A')} fps")
    # print()
    
