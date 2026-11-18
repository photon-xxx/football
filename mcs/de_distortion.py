"""
相机畸变校正模块 (Camera Distortion Calibration)

核心概念：
1. 相机畸变参数 (k_camera)：描述相机镜头的物理畸变特性
   - k < 0: 桶形畸变 (Barrel) - 广角镜头，直线向外凸
   - k > 0: 枕形畸变 (Pincushion) - 长焦镜头，直线向内凹

2. 校正参数 (k_correction)：用于去除畸变的参数
   - 关系：k_correction = -k_camera（径向参数符号相反）

3. 标定过程：
   - 输入：原始图像上的直线/关键点（有相机畸变）
   - 优化：找到校正参数，使直线在去畸变后变直
   - 输出：相机畸变参数（符号已转换）= -校正参数

使用方法：
- distort_point(ideal, camera_params): 模拟相机畸变（理想 → 畸变）
- undistort_point(distorted, camera_params): 去除相机畸变（畸变 → 理想）
- 两个函数内部自动处理符号转换

使用配置 (LM Full):
- Brown-Conrady 5参数模型 (k1, k2, k3, p1, p2)
- 几何约束 (平行线 + 交点约束 + 投影误差)
- Huber 鲁棒损失
- Levenberg-Marquardt 求解器

作为单应性变换的前置步骤，先去畸变再计算单应矩阵。
"""

import numpy as np
import cv2
import json
import os
import time
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union
from scipy.optimize import least_squares
from area import load_keypoints, run_marker_annotation, run_annotation


# ==========================================
# 参数类型定义
# ==========================================

@dataclass
class DistortionParams:
    """
    畸变参数（Brown-Conrady 模型）
    
    描述相机的畸变特性：
    - k1, k2, k3: 径向畸变系数
      - k < 0: 桶形畸变 (Barrel, 广角镜头常见)
      - k > 0: 枕形畸变 (Pincushion)
    - p1, p2: 切向畸变系数
    """
    k1: float
    k2: float
    p1: float
    p2: float
    k3: float = 0.0
    
    def to_array(self) -> np.ndarray:
        return np.array([self.k1, self.k2, self.p1, self.p2, self.k3])


# ==========================================
# 核心配置 (使用 LM Full)
# ==========================================

# TODO：经验设置的
@dataclass
class DistortionConfig:
    """畸变校正配置 (LM Full 标准配置)"""
    enable_k3: bool = True                    # 启用 k3 高阶径向畸变
    weight_straightness: float = 1.0          # 直线度约束
    weight_parallelism: float = 0.0           # 平行线约束
    weight_intersection: float = 0.0          # 交点约束
    weight_projection: float = 1.0            # [新增] 投影误差约束 (关键点 -> 单应性一致性)
    use_robust_loss: str = 'huber'           # 标准最小二乘（关闭鲁棒损失） 'huber' 开启   linear关闭
    robust_scale: float = 1.0
    weight_regularization: float = 0.0


# ==========================================
# 畸变模型 (Brown-Conrady)
# ==========================================

class BrownConradyModel:
    """Brown-Conrady 畸变模型 (5参数: k1, k2, k3, p1, p2)"""
    
    def __init__(self, image_center: Tuple[float, float], enable_k3: bool = True):
        self.center = np.array(image_center)
        self.scale = image_center[0]  # 归一化尺度
        self.enable_k3 = enable_k3
        
    def get_initial_params(self):
        """初始参数: k1, k2, p1, p2, [k3]"""
        params = [
            -0.2,
            0.0,
            0.0,
            0.0
        ]
        if self.enable_k3:
            params.append(0.0)
        return np.array(params)
    
    def unpack_params(self, params):
        """解包参数"""
        k1, k2, p1, p2 = params[0:4]
        k3 = params[4] if self.enable_k3 and len(params) > 4 else 0.0
        return k1, k2, p1, p2, k3
    
    def undistort_point(self, point: np.ndarray, params: Union[np.ndarray, DistortionParams]) -> np.ndarray:
        """
        去畸变：将畸变图像点转换为理想点（直接应用畸变模型的逆向形式）
        
        Args:
            point: 畸变坐标 x_d（相机观测到的点）
            params: 畸变参数 k（唯一一份，描述相机畸变）
            
        公式：x_u = x_d / (1 + k*r_d²) （一阶近似，直接计算）
        """
        # 统一处理参数
        if isinstance(params, DistortionParams):
            k1, k2, p1, p2, k3 = params.k1, params.k2, params.p1, params.p2, params.k3
        else:
            k1, k2, p1, p2, k3 = self.unpack_params(params)
        
        x_d, y_d = point
        cx, cy = self.center
        
        # 归一化
        x_d_n = (x_d - cx) / self.scale
        y_d_n = (y_d - cy) / self.scale
        
        from distort import my_undistort_point
        
        # 去畸变
        x_u_n, y_u_n = my_undistort_point(x_d_n, y_d_n, k1, k2, k3, p1, p2)
        
        # 反归一化
        ux = x_u_n * self.scale + cx
        uy = y_u_n * self.scale + cy
        
        return np.array([ux, uy])
    
    def undistort_points(self, points: np.ndarray, params: Union[np.ndarray, DistortionParams]) -> np.ndarray:
        """批量去畸变"""
        return np.array([self.undistort_point(p, params) for p in points])
    
    def distort_point(self, point: np.ndarray, params: Union[np.ndarray, DistortionParams]) -> np.ndarray:
        """
        加畸变：将理想点转换为畸变点（通过迭代求解）
        
        Args:
            point: 理想坐标 x_u
            params: 畸变参数 k
        
        迭代求解：找到 x_d 使得 undistort(x_d) = x_u
        """
        # 统一处理参数
        if isinstance(params, DistortionParams):
            k1, k2, p1, p2, k3 = params.k1, params.k2, params.p1, params.p2, params.k3
        else:
            k1, k2, p1, p2, k3 = self.unpack_params(params)
        
        x_u, y_u = point
        cx, cy = self.center
        
        # 归一化
        x_u_n = (x_u - cx) / self.scale
        y_u_n = (y_u - cy) / self.scale
        
        from distort import my_distort_point
        # 加畸变
        x_d_n, y_d_n = my_distort_point(x_u_n, y_u_n, k1, k2, k3, p1, p2)
        
        # 反归一化
        x_d = x_d_n * self.scale + cx
        y_d = y_d_n * self.scale + cy
        
        return np.array([x_d, y_d])
    
    def distort_points(self, points: np.ndarray, params: Union[np.ndarray, DistortionParams]) -> np.ndarray:
        """批量加畸变"""
        return np.array([self.distort_point(p, params) for p in points])


# ==========================================
# 损失函数 (几何约束)
# ==========================================

# TODO：优化这里的损失设计，当前对于多个约束叠加的损失没有比较仔细地计算合理性
class DistortionLoss:
    """多约束损失函数 (含关键点投影损失)"""
    
    def __init__(self, model: BrownConradyModel, config: DistortionConfig,
                 groups: List[List], names: List[str],
                 keypoints_data: Optional[Dict] = None,
                 pitch_config: Optional[object] = None):
        self.model = model
        self.config = config
        self.groups = groups
        self.names = names
        
        # 新增：用于投影损失计算的关键点数据
        # keypoints_data: {idx: img_coord}
        self.keypoints_data = keypoints_data
        self.pitch_config = pitch_config
    
    def _fit_line(self, points):
        """拟合直线 ax + by + c = 0"""
        if len(points) < 2:
            return None
        
        # 检查输入点
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            print(f"\n❌ [ERROR] _fit_line 输入包含 NaN/Inf:")
            print(f"    points shape: {points.shape}")
            print(f"    NaN count: {np.sum(np.isnan(points))}")
            print(f"    Inf count: {np.sum(np.isinf(points))}")
            print(f"    points: {points}")
            return None
        
        mean = np.mean(points, axis=0)
        centered = points - mean
        
        # 检查 centered 数据
        if np.any(np.isnan(centered)) or np.any(np.isinf(centered)):
            print(f"\n❌ [ERROR] centered 包含 NaN/Inf:")
            print(f"    mean: {mean}")
            print(f"    centered: {centered}")
            return None
        
        try:
            u, s, vh = np.linalg.svd(centered)
        except np.linalg.LinAlgError as e:
            print(f"\n❌ [ERROR] SVD 失败: {e}")
            print(f"    points: {points}")
            print(f"    mean: {mean}")
            print(f"    centered: {centered}")
            print(f"    centered range: [{np.min(centered)}, {np.max(centered)}]")
            return None
        
        normal = vh[-1]
        a, b = normal
        c = -np.dot(normal, mean)
        return np.array([a, b, c])
    
    def _line_intersection(self, line1, line2):
        """计算两条直线的交点"""
        a1, b1, c1 = line1
        a2, b2, c2 = line2
        det = a1*b2 - a2*b1
        if abs(det) < 1e-6:
            return None
        x = (b1*c2 - b2*c1) / det
        y = (c1*a2 - c2*a1) / det
        return np.array([x, y])
    
    def calculate_residuals(self, params):
        """计算所有约束的残差向量"""
        # 只在出现错误时输出调试信息
        if np.any(np.isnan(params)) or np.any(np.isinf(params)):
            print(f"\n❌ [ERROR] calculate_residuals: params 包含 NaN/Inf!")
            print(f"    params: {params}")
            return np.array([1e10])  # 返回大残差
        
        residuals = []
        ud_lines = {}
        ud_pts_dict = {}
        ud_keypoints = {}
        ud_lines_h = []
        ud_lines_v = []
        
        # 1. 线条约束 (直线度 + 平行性 + 交点)
        for i, grp in enumerate(self.groups):
            name = self.names[i]
            if len(grp) < 2:
                continue
            
            grp_array = np.array(grp)
            
            try:
                ud_pts = self.model.undistort_points(grp_array, params)
            except Exception as e:
                print(f"\n❌ [ERROR] undistort_points 失败: {e}")
                print(f"    线组 {i}: {name}, 点数: {len(grp)}")
                print(f"    原始点范围: x=[{grp_array[:, 0].min():.1f}, {grp_array[:, 0].max():.1f}], "
                      f"y=[{grp_array[:, 1].min():.1f}, {grp_array[:, 1].max():.1f}]")
                print(f"    grp: {grp_array}")
                return np.array([1e10])
            
            # 检查去畸变后的点
            if np.any(np.isnan(ud_pts)) or np.any(np.isinf(ud_pts)):
                print(f"\n❌ [ERROR] 去畸变后包含 NaN/Inf!")
                print(f"    线组 {i}: {name}, 点数: {len(grp)}")
                print(f"    原始点范围: x=[{grp_array[:, 0].min():.1f}, {grp_array[:, 0].max():.1f}], "
                      f"y=[{grp_array[:, 1].min():.1f}, {grp_array[:, 1].max():.1f}]")
                print(f"    去畸变点范围: x=[{ud_pts[:, 0].min():.1f}, {ud_pts[:, 0].max():.1f}], "
                      f"y=[{ud_pts[:, 1].min():.1f}, {ud_pts[:, 1].max():.1f}]")
                print(f"    ud_pts: {ud_pts}")
                return np.array([1e10])
            
            ud_pts_dict[name] = ud_pts
            
            # 记录关键点 (仅用于交点约束)
            if "上边线" in name:
                ud_keypoints["01"] = ud_pts[0]
                ud_keypoints["14"] = ud_pts[-1]
            elif "下边线" in name:
                ud_keypoints["06"] = ud_pts[0]
                ud_keypoints["17"] = ud_pts[-1]
            
            # 拟合直线
            line_params = self._fit_line(ud_pts)
            if line_params is not None:
                ud_lines[name] = line_params
                
                # 收集平行线组
                if name.startswith("H_"):
                    ud_lines_h.append(line_params)
                elif name.startswith("V_"):
                    ud_lines_v.append(line_params)
                
                # 直线度残差
                if len(ud_pts) >= 3 and self.config.weight_straightness > 0:
                    a, b, c = line_params
                    dists = np.abs(np.dot(ud_pts, np.array([a, b])) + c)
                    residuals.extend(dists * self.config.weight_straightness)
        
        # 2. 平行约束
        if self.config.weight_parallelism > 0:
            for g_lines in [ud_lines_h, ud_lines_v]:
                if len(g_lines) < 2:
                    continue
                
                valid_intersections = []
                pair_residuals = []
                
                for i in range(len(g_lines)):
                    for j in range(i + 1, len(g_lines)):
                        l1, l2 = g_lines[i], g_lines[j]
                        det = l1[0]*l2[1] - l1[1]*l2[0]
                        
                        if abs(det) < 1e-3:
                            pair_residuals.append(None)
                        else:
                            pt = self._line_intersection(l1, l2)
                            if pt is not None:
                                valid_intersections.append(pt)
                                pair_residuals.append(pt)
                            else:
                                pair_residuals.append(None)
                
                if len(valid_intersections) < 2:
                    residuals.extend([0.0] * len(pair_residuals))
                else:
                    center = np.median(valid_intersections, axis=0)
                    for item in pair_residuals:
                        if item is None:
                            residuals.append(0.0)
                        else:
                            dist = np.linalg.norm(item - center)
                            residuals.append(dist * self.config.weight_parallelism * 1e-3)
        
        # 3. 交点约束
        if self.config.weight_intersection > 0:
            constraints = [
                ("H_自动_上边线", "V_自动_左底线", "01"),
                ("V_自动_左底线", "H_自动_下边线", "06"),
                ("H_自动_下边线", "V_自动_右中线", "17"),
                ("V_自动_右中线", "H_自动_上边线", "14")
            ]
            
            for k1, k2, pid in constraints:
                l1 = next((v for k, v in ud_lines.items() if k1 in k), None)
                l2 = next((v for k, v in ud_lines.items() if k2 in k), None)
                pt = ud_keypoints.get(pid, None)
                
                if l1 is not None and l2 is not None and pt is not None:
                    int_pt = self._line_intersection(l1, l2)
                    if int_pt is not None:
                        dist = np.linalg.norm(int_pt - pt)
                        if dist > 100:
                            dist = 100 + np.log(max(1.0, dist - 99))
                        residuals.append(dist * self.config.weight_intersection * 0.1)
        
        # 4. [新增] 投影误差约束 (Projection Error)
        if self.config.weight_projection > 0 and self.keypoints_data and self.pitch_config:
            sorted_indices = sorted(list(self.keypoints_data.keys()))
            if len(sorted_indices) >= 4:
                # 4.1 收集原始标注点（畸变像素）和对应的球场坐标
                raw_img_pts = np.array([self.keypoints_data[idx] for idx in sorted_indices])
                
                world_pts = []
                valid_mask = []
                for idx in sorted_indices:
                    if idx < len(self.pitch_config.vertices):
                        world_pts.append(self.pitch_config.vertices[idx])
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                
                raw_img_pts = raw_img_pts[valid_mask]
                world_pts = np.array(world_pts)
                
                if len(raw_img_pts) >= 4:
                    try:
                        # 4.2 去畸变：原始标注点 → 理想像素
                        undistorted_img_pts = self.model.undistort_points(raw_img_pts, params)
                        
                        # 4.3 计算单应矩阵 H（球场坐标 → 理想像素）
                        H, _ = cv2.findHomography(world_pts, undistorted_img_pts, method=0)

                        if H is not None:
                            # 4.4 正向投影：球场坐标 → 理想像素
                            reprojected_world_pts = cv2.perspectiveTransform(
                                world_pts.reshape(-1, 1, 2), H
                            ).reshape(-1, 2)
                            
                            # 4.5 计算损失：真实球场坐标 vs 重投影球场坐标（单位：cm）
                            proj_diff = undistorted_img_pts - reprojected_world_pts
                            proj_errors = np.linalg.norm(proj_diff, axis=1)
                            
                            # [Debug] 首次调用时打印一次
                            if not hasattr(self, '_debug_printed'):
                                print(f"[投影误差] 计算 {len(proj_errors)} 个关键点的投影误差")
                                print(f"[投影误差] 平均误差: {np.mean(proj_errors):.2f} cm, 最大误差: {np.max(proj_errors):.2f} cm")
                                print(f"[投影误差] 权重: {self.config.weight_projection}")
                                self._debug_printed = True
                            
                            # 使用配置中的权重
                            residuals.extend(proj_errors * self.config.weight_projection)
                            
                    except Exception as e:
                        # SVD 不收敛等情况忽略
                        print(f"[投影误差] 警告: 计算失败 - {e}")
                        import traceback
                        traceback.print_exc()

        # 5. 正则化
        if self.config.weight_regularization > 0:
            k1, k2, p1, p2, k3 = self.model.unpack_params(params)
            w = self.config.weight_regularization
            
            reg_list = [
                k1 * w,
                k2 * w,
                p1 * w * 100.0,
                p2 * w * 100.0
            ]
            if self.config.enable_k3:
                reg_list.append(k3 * w * 10.0)
            
            residuals.extend(reg_list)
        
        return np.array(residuals)

    def evaluate_residual_stats(self, params):
        """
        计算并打印各类残差的详细统计信息（模方和、均方根误差等）。
        仅用于调试和评估，不要直接用于 least_squares 优化。
        """

        # 定义分类容器
        res_categories = {
            "straightness": [],  # 直线度
            "parallelism": [],   # 平行度
            "intersection": [],  # 交点约束
            "projection": [],    # 投影误差
            "regularization": [] # 正则化
        }

        ud_lines = {}
        ud_pts_dict = {}
        ud_keypoints = {}
        ud_lines_h = []
        ud_lines_v = []

        # --- 1. 线条约束 (直线度) ---
        for i, grp in enumerate(self.groups):
            name = self.names[i]
            if len(grp) < 2: continue

            grp_array = np.array(grp)
            try:
                ud_pts = self.model.undistort_points(grp_array, params)
            except Exception:
                continue # 忽略错误，仅做统计

            ud_pts_dict[name] = ud_pts
            
            # 记录关键点
            if "上边线" in name:
                ud_keypoints["01"], ud_keypoints["14"] = ud_pts[0], ud_pts[-1]
            elif "下边线" in name:
                ud_keypoints["06"], ud_keypoints["17"] = ud_pts[0], ud_pts[-1]

            # 拟合直线
            line_params = self._fit_line(ud_pts)
            if line_params is not None:
                ud_lines[name] = line_params
                if name.startswith("H_"): ud_lines_h.append(line_params)
                elif name.startswith("V_"): ud_lines_v.append(line_params)

                # [收集] 直线度残差
                if len(ud_pts) >= 3 and self.config.weight_straightness > 0:
                    a, b, c = line_params
                    dists = np.abs(np.dot(ud_pts, np.array([a, b])) + c)
                    res_categories["straightness"].extend(dists * self.config.weight_straightness)

        # --- 2. 平行约束 ---
        if self.config.weight_parallelism > 0:
            for g_lines in [ud_lines_h, ud_lines_v]:
                if len(g_lines) < 2: continue
                
                valid_intersections = []
                pair_residuals = [] # 临时存储
                
                for i in range(len(g_lines)):
                    for j in range(i + 1, len(g_lines)):
                        l1, l2 = g_lines[i], g_lines[j]
                        det = l1[0]*l2[1] - l1[1]*l2[0]
                        if abs(det) >= 1e-3:
                            pt = self._line_intersection(l1, l2)
                            if pt is not None:
                                valid_intersections.append(pt)
                                pair_residuals.append(pt)
                                continue
                        pair_residuals.append(None)

                if len(valid_intersections) >= 2:
                    center = np.median(valid_intersections, axis=0)
                    for item in pair_residuals:
                        if item is not None:
                            dist = np.linalg.norm(item - center)
                            # [收集] 平行度残差
                            res_categories["parallelism"].append(dist * self.config.weight_parallelism * 1e-3)
                        else:
                            res_categories["parallelism"].append(0.0) # 或者根据你的逻辑填补

        # --- 3. 交点约束 ---
        if self.config.weight_intersection > 0:
            constraints = [
                ("H_自动_上边线", "V_自动_左底线", "01"),
                ("V_自动_左底线", "H_自动_下边线", "06"),
                ("H_自动_下边线", "V_自动_右中线", "17"),
                ("V_自动_右中线", "H_自动_上边线", "14")
            ]
            for k1, k2, pid in constraints:
                l1 = next((v for k, v in ud_lines.items() if k1 in k), None)
                l2 = next((v for k, v in ud_lines.items() if k2 in k), None)
                pt = ud_keypoints.get(pid, None)
                
                if l1 is not None and l2 is not None and pt is not None:
                    int_pt = self._line_intersection(l1, l2)
                    if int_pt is not None:
                        dist = np.linalg.norm(int_pt - pt)
                        if dist > 100: dist = 100 + np.log(max(1.0, dist - 99))
                        # [收集] 交点残差
                        res_categories["intersection"].append(dist * self.config.weight_intersection * 0.1)

        # --- 4. 投影误差 ---
        # 4. [新增] 投影误差约束 (Projection Error) - 在球场坐标空间计算
        # 算法流程：
        # 原始标注点（畸变） → 去畸变 → 理想像素 → 计算H → 球场坐标
        # 损失 = 真实球场坐标 vs 重投影球场坐标
        if self.config.weight_projection > 0 and self.keypoints_data and self.pitch_config:
            sorted_indices = sorted(list(self.keypoints_data.keys()))
            if len(sorted_indices) >= 4:
                # 4.1 收集原始标注点（畸变像素）和对应的球场坐标
                raw_img_pts = np.array([self.keypoints_data[idx] for idx in sorted_indices])
                
                world_pts = []
                valid_mask = []
                for idx in sorted_indices:
                    if idx < len(self.pitch_config.vertices):
                        world_pts.append(self.pitch_config.vertices[idx])
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                
                raw_img_pts = raw_img_pts[valid_mask]
                world_pts = np.array(world_pts)
                
                if len(raw_img_pts) >= 4:
                    try:
                        # 4.2 去畸变：原始标注点 → 理想像素
                        undistorted_img_pts = self.model.undistort_points(raw_img_pts, params)
                        
                        # 4.3 计算单应矩阵 H（理想像素 → 球场坐标）
                        H, _ = cv2.findHomography(undistorted_img_pts, world_pts, method=0)
                        
                        if H is not None:
                            # 4.4 正向投影：理想像素 → 球场坐标
                            reprojected_world_pts = cv2.perspectiveTransform(
                                undistorted_img_pts.reshape(-1, 1, 2), H
                            ).reshape(-1, 2)
                            
                            # 4.5 计算损失：真实球场坐标 vs 重投影球场坐标（单位：cm）
                            proj_diff = world_pts - reprojected_world_pts
                            proj_errors = np.linalg.norm(proj_diff, axis=1)
                            

                            print(f"[球场投影误差] 计算 {len(proj_errors)} 个关键点的投影误差")
                            print(f"[球场投影误差] 平均误差: {np.mean(proj_errors):.2f} cm, 最大误差: {np.max(proj_errors):.2f} cm")
                            print(f"[球场投影误差] 权重: {self.config.weight_projection}")
                            self._debug_printed = True
                            
                    except Exception as e:
                        # SVD 不收敛等情况忽略
                        print(f"[球场投影误差] 警告: 计算失败 - {e}")
                        import traceback
                        traceback.print_exc()

        if self.config.weight_projection > 0 and self.keypoints_data and self.pitch_config:
            sorted_indices = sorted(list(self.keypoints_data.keys()))
            if len(sorted_indices) >= 4:
                # ... (数据准备代码同原函数) ...
                raw_img_pts = np.array([self.keypoints_data[idx] for idx in sorted_indices])
                world_pts = []
                valid_mask = []
                for idx in sorted_indices:
                    if idx < len(self.pitch_config.vertices):
                        world_pts.append(self.pitch_config.vertices[idx])
                        valid_mask.append(True)
                    else:
                        valid_mask.append(False)
                
                raw_img_pts = raw_img_pts[valid_mask]
                world_pts = np.array(world_pts)

                if len(raw_img_pts) >= 4:
                    try:
                        undistorted_img_pts = self.model.undistort_points(raw_img_pts, params)
                        H, _ = cv2.findHomography(world_pts, undistorted_img_pts, method=0)
                        if H is not None:
                            # 注意：这里保留你的原始逻辑，但请检查 perspectiveTransform 的方向是否正确
                            # 通常比较误差应该在同一坐标系下进行 (比如都转到 World 或都转到 Pixel)
                            reprojected_world_pts = cv2.perspectiveTransform(
                                world_pts.reshape(-1, 1, 2), H
                            ).reshape(-1, 2)
                            
                            proj_diff = undistorted_img_pts - reprojected_world_pts
                            proj_errors = np.linalg.norm(proj_diff, axis=1)
                            
                            # [收集] 投影残差
                            res_categories["projection"].extend(proj_errors * self.config.weight_projection)
                    except Exception:
                        pass

        # --- 5. 正则化 ---
        if self.config.weight_regularization > 0:
            k1, k2, p1, p2, k3 = self.model.unpack_params(params)
            w = self.config.weight_regularization
            reg_list = [k1*w, k2*w, p1*w*100.0, p2*w*100.0]
            if self.config.enable_k3: reg_list.append(k3*w*10.0)
            
            # [收集] 正则化残差
            res_categories["regularization"].extend(reg_list)

        # --- 6. 统计与输出 ---
        print("\n" + "="*50)
        print(f"{'Category':<20} | {'Count':<6} | {'RSS (Cost)':<12} | {'RMSE':<10}")
        print("-" * 50)
        
        total_rss = 0.0
        total_count = 0
        
        stats = {}

        for cat, values in res_categories.items():
            arr = np.array(values)
            count = len(arr)
            if count > 0:
                rss = np.sum(arr**2)     # 残差平方和 (Residual Sum of Squares)
                rmse = np.sqrt(np.mean(arr**2)) # 均方根误差
            else:
                rss = 0.0
                rmse = 0.0
            
            total_rss += rss
            total_count += count
            stats[cat] = rss
            
            print(f"{cat:<20} | {count:<6} | {rss:12.4f} | {rmse:10.4f}")

        print("-" * 50)
        print(f"{'TOTAL':<20} | {total_count:<6} | {total_rss:12.4f} |")
        print("="*50 + "\n")
        return stats

# ==========================================
# 数据加载
# ==========================================
def get_auto_edge_groups(keypoints_dict):
    """自动提取场地边缘线"""
    groups, names = [], []
    definitions = [
        ("H_自动_上边线", [0] + [22 + 2*i for i in range(7)] + [13]),
        ("H_自动_下边线", [5] + [23 + 2*i for i in range(7)] + [16]),
        ("V_自动_左底线", [0, 1, 2, 3, 4, 5]),
        ("V_自动_右中线", [13, 14, 18, 15, 16])
    ]
    for name, indices in definitions:
        current = [keypoints_dict[idx] for idx in indices if idx in keypoints_dict]
        if len(current) >= 3:
            groups.append(current)
            names.append(name)
    return groups, names


def get_manual_groups(json_path, prefix=""):
    """加载手动标注的线组"""
    groups, names = [], []
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                raw = data.get("groups", data if isinstance(data, list) else [])
                for i, g in enumerate(raw):
                    pts = []
                    if isinstance(g, list):
                        pts = g
                    elif isinstance(g, dict) and "points" in g:
                        pts = [p.get("img_coord", p) for p in g["points"]]
                    if len(pts) >= 2:
                        groups.append(pts)
                        names.append(f"{prefix}手动_{i+1}")
        except:
            pass
    return groups, names


# ==========================================
# 主接口函数
# ==========================================

def calibrate_distortion(
    image: np.ndarray,
    keypoints_json: Optional[str] = None,
    manual_h_json: Optional[str] = None,
    manual_v_json: Optional[str] = None,
    config: Optional[DistortionConfig] = None,
    pitch_config: Optional[object] = None,
    auto_annotate: bool = True
) -> Tuple[np.ndarray, Dict]:
    """
    相机畸变标定 (使用 LM Full 配置)
    
    Args:
        image: 输入图像
        keypoints_json: 关键点标注文件路径
        manual_h_json: 手动横向线标注路径
        manual_v_json: 手动纵向线标注路径
        config: 畸变配置 (默认使用 LM Full)
        pitch_config: 球场配置对象 (用于投影误差计算，默认 HalfQingYuanConfiguration)
        auto_annotate: 是否在文件缺失时自动启动标注
    
    Returns:
        params: 畸变参数 (k1, k2, p1, p2, k3)
        info: 标定信息字典 (cost, time, center, etc.)
    """
    if config is None:
        config = DistortionConfig()
    
    if pitch_config is None:
        pitch_config = HalfQingYuanConfiguration()
    
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    
    # 1. 加载关键点
    indices, points = [], []
    if keypoints_json and os.path.exists(keypoints_json):
        try:
            indices, points = load_keypoints(keypoints_json)
        except:
            print(f"读取关键点失败: {keypoints_json}")
    
    if not indices and auto_annotate:
        print("\n[提示] 未找到关键点文件，启动标注...")
        indices, points = run_annotation(
            image, 
            pitch_config,  # 使用传入的 pitch_config
            keypoints_json or "temp_keypoints.json",
            "distortion_calibration"
        )
        if not indices:
            raise RuntimeError("关键点标注未完成")
    
    kpts_dict = {idx: pt for idx, pt in zip(indices, points)}
    
    # 2. 收集线组
    all_groups, all_names = [], []
    
    # 自动边缘线
    ag, an = get_auto_edge_groups(kpts_dict)
    all_groups.extend(ag)
    all_names.extend(an)
    
    # 手动横向线
    if manual_h_json:
        if not os.path.exists(manual_h_json) and auto_annotate:
            print("\n[提示] 启动横向线标注...")
            run_marker_annotation(image, manual_h_json, "manual_horizontal")
        mg_h, mn_h = get_manual_groups(manual_h_json, prefix="H_")
        all_groups.extend(mg_h)
        all_names.extend(mn_h)
    
    # 手动纵向线
    if manual_v_json:
        if not os.path.exists(manual_v_json) and auto_annotate:
            print("\n[提示] 启动纵向线标注...")
            run_marker_annotation(image, manual_v_json, "manual_vertical")
        mg_v, mn_v = get_manual_groups(manual_v_json, prefix="V_")
        all_groups.extend(mg_v)
        all_names.extend(mn_v)
    
    print(f"共加载 {len(all_groups)} 组约束线")
    
    # 3. 优化
    model = BrownConradyModel(center, enable_k3=config.enable_k3)
    loss_fn = DistortionLoss(model, config, all_groups, all_names, 
                             keypoints_data=kpts_dict, 
                             pitch_config=pitch_config)  # 使用参数传入的 pitch_config
    x0 = model.get_initial_params()
    
    if config.enable_k3:
        # 包含 k3 的边界
        lower_bounds = [-0.5, -0.3, -0.1, -0.1, -0.1]  # k1, k2, p1, p2, k3
        upper_bounds = [0.5, 0.3, 0.1, 0.1, 0.1]
        
    else:
        # 不包含 k3 的边界
        lower_bounds = [-0.5, -0.3, -0.1, -0.1]  # k1, k2, p1, p2
        upper_bounds = [0.5, 0.3, 0.1, 0.1]
    
    print(f"参数边界: lower={lower_bounds}, upper={upper_bounds}")
    
    dict = loss_fn.evaluate_residual_stats(x0)
    print(dict)

    import yaml
    root_path = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(root_path, "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f)

    start_time = time.time()
    res = least_squares(
        loss_fn.calculate_residuals,
        x0,
        method='trf',
        bounds=(lower_bounds, upper_bounds),  # 🆕 添加参数边界
        loss=config.use_robust_loss,
        f_scale=config.robust_scale,
        jac = config_dict["jac"],
        verbose=0,
        max_nfev=100000
    )
    # -------------------------------------------------------------------

    duration = time.time() - start_time

    # 标定优化得到的参数直接用于应用畸变
    params = res.x
    loss_fn.evaluate_residual_stats(params)
    k1, k2, p1, p2, k3 = model.unpack_params(params)
    print(f"\n[优化结果] success={res.success}, status={res.status}, message='{res.message}'")
    print(f"\n[标定完成]")
    print(f"耗时: {duration:.3f}s | Cost: {res.cost:.4f}")
    print(f"畸变参数: k1={k1:.4f}, k2={k2:.4f}, k3={k3:.4f}, p1={p1:.4f}, p2={p2:.4f}")
    print(f"  (k1 > 0 时在此实现中产生桶形畸变效果)")
    
    info = {
        'params': params,
        'cost': res.cost,
        'time': duration,
        'center': center,
        'image_size': (w, h),
        'k1': k1, 'k2': k2, 'k3': k3, 'p1': p1, 'p2': p2
    }
    
    return params, info


def undistort_points(
    points: np.ndarray,
    params: Union[np.ndarray, DistortionParams],
    image_center: Tuple[float, float],
    enable_k3: bool = True
) -> np.ndarray:
    """
    对点进行去畸变
    
    Args:
        points: 输入点坐标 (N, 2) 或 (N, 1, 2)
        params: 校正参数或相机参数或数组
        image_center: 图像中心 (cx, cy)
        enable_k3: 是否使用 k3 参数
    
    Returns:
        去畸变后的点坐标 (N, 2)
    """
    # 处理输入形状
    original_shape = points.shape
    if points.ndim == 3:
        points = points.reshape(-1, 2)
    
    model = BrownConradyModel(image_center, enable_k3=enable_k3)
    undistorted = model.undistort_points(points, params)
    
    # 恢复形状
    if len(original_shape) == 3:
        undistorted = undistorted.reshape(original_shape)
    
    return undistorted


def distort_points(
    points: np.ndarray,
    params: Union[np.ndarray, DistortionParams],
    image_center: Tuple[float, float],
    enable_k3: bool = True
) -> np.ndarray:
    """
    对点进行正向畸变
    
    Args:
        points: 理想点坐标 (N, 2) 或 (N, 1, 2)
        params: 校正参数或相机参数或数组
        image_center: 图像中心 (cx, cy)
        enable_k3: 是否使用 k3 参数
    
    Returns:
        加畸变后的点坐标 (N, 2)
    """
    # 处理输入形状
    original_shape = points.shape
    if points.ndim == 3:
        points = points.reshape(-1, 2)
    
    model = BrownConradyModel(image_center, enable_k3=enable_k3)
    distorted = model.distort_points(points, params)
    
    # 恢复形状
    if len(original_shape) == 3:
        distorted = distorted.reshape(original_shape)
    
    return distorted


def undistort_image(
    image: np.ndarray,
    params: Union[np.ndarray, DistortionParams],
    image_center: Optional[Tuple[float, float]] = None,
    enable_k3: bool = True
) -> np.ndarray:
    """
    对整张图像进行去畸变
    
    Args:
        image: 输入图像
        params: 畸变参数（校正参数或相机参数或数组）
        image_center: 图像中心 (默认使用图像几何中心)
        enable_k3: 是否使用 k3 参数
    
    Returns:
        去畸变后的图像
    """
    h, w = image.shape[:2]
    if image_center is None:
        image_center = (w / 2.0, h / 2.0)
    
    model = BrownConradyModel(image_center, enable_k3=enable_k3)
    
    # 创建映射表
    map_x = np.zeros((h, w), dtype=np.float32)
    map_y = np.zeros((h, w), dtype=np.float32)
    
    for y in range(h):
        for x in range(w):
            # 对每个输出像素,找到对应的输入像素位置
            ud_pt = model.undistort_point(np.array([x, y]), params)
            map_x[y, x] = ud_pt[0]
            map_y[y, x] = ud_pt[1]
    
    # 应用重映射
    undistorted = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
    
    return undistorted


# ==========================================
# 保存/加载畸变参数
# ==========================================

def save_distortion_params(params: np.ndarray, info: Dict, save_path: str):
    """保存畸变参数到文件"""
    data = {
        'params': params.tolist(),
        'info': {k: (v.tolist() if isinstance(v, np.ndarray) else v) 
                 for k, v in info.items()}
    }
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"畸变参数已保存: {save_path}")


def load_distortion_params(load_path: str) -> Tuple[np.ndarray, Dict]:
    """从文件加载畸变参数"""
    with open(load_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    params = np.array(data['params'])
    info = data.get('info', {})
    print(f"畸变参数已加载: {load_path}")
    return params, info


# ==========================================
# 示例使用
# ==========================================

if __name__ == "__main__":
    import supervision as sv
    from soccer import *
    from camera import Camera
    from tools.frame_tool import save_frame
    from tools.annotation.pitch_annotation_tool import draw_pitch
    from MCS import MCS
    import numpy as np

    #
    config = HalfWuChuanConfiguration()
    #
    video_path_1 = "videos/视角1.mp4"
    video_path_2 = "videos/视角2.mp4"
    video_path_3 = "videos/视角3.mp4"

    camera_1 = Camera(video_path_1, config, camera_id=1,camera_name="camera1", enable_distortion_correction=True, debug_mode=True, mcs_name="wuchuan")
    camera_2 = Camera(video_path_2, config, camera_id=2,camera_name="camera2", enable_distortion_correction=True, debug_mode=True, mcs_name="wuchuan")
    camera_3 = Camera(video_path_3, config, camera_id=3,camera_name="camera3", enable_distortion_correction=True, debug_mode=True, mcs_name="wuchuan")
    #
    cameras = [camera_1, camera_2, camera_3]

    mcs = MCS(cameras, config, name="wuchuan")
    #
    mcs_vis = mcs.visualize()
