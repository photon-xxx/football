"""
统一的网格坐标系统

提供全局唯一的网格定义和坐标转换功能，确保 Camera 和 MCS 使用完全一致的网格系统。
"""

import numpy as np
from typing import Tuple


class PitchGridSystem:
    """
    球场网格坐标系统
    
    定义球场上的统一网格划分，提供索引和坐标之间的转换。
    
    坐标系统：
    - 物理坐标：(x, y) 单位 cm，x 沿球场长度方向，y 沿球场宽度方向
    - 网格索引：[row, col]，row 对应 y 方向，col 对应 x 方向
    - 网格编号从 (0, 0) 开始，对应球场左下角
    
    网格定义：
    - 每个网格的左下角坐标：(col * grid_size, row * grid_size)
    - 每个网格的中心坐标：(col * grid_size + grid_size/2, row * grid_size + grid_size/2)
    """
    
    def __init__(self, pitch_length: float, pitch_width: float, grid_size: float):
        """
        Args:
            pitch_length: 球场长度 (cm)，x 方向
            pitch_width: 球场宽度 (cm)，y 方向
            grid_size: 网格尺寸 (cm)
        """
        self.pitch_length = pitch_length
        self.pitch_width = pitch_width
        self.grid_size = grid_size
        
        # 生成网格边界（左下角坐标）
        self.gx = np.arange(0, pitch_length, grid_size, dtype=np.float32)
        self.gy = np.arange(0, pitch_width, grid_size, dtype=np.float32)
        
        # 网格维度
        self.num_cols = len(self.gx)  # x 方向（列数）
        self.num_rows = len(self.gy)  # y 方向（行数）
        
        # 生成网格中心坐标
        self.grid_x_centers = self.gx + grid_size / 2
        self.grid_y_centers = self.gy + grid_size / 2
        
        # 生成 meshgrid（左下角）用于 MCS 兼容
        self.grid_x_corners, self.grid_y_corners = np.meshgrid(self.gx, self.gy)
        
        print(f"[GridSystem] 初始化: {self.num_rows}行 × {self.num_cols}列 = {self.num_rows * self.num_cols} 个网格")
        print(f"[GridSystem] 球场: {pitch_length}cm × {pitch_width}cm, 网格: {grid_size}cm")
    
    def get_all_grid_centers(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取所有网格中心坐标和对应的索引
        
        Returns:
            grid_centers: (N, 2) 数组，每行是 [x, y] 坐标
            grid_indices: (N, 2) 数组，每行是 [row, col] 索引
        """
        grid_centers = []
        grid_indices = []
        
        for row, y in enumerate(self.grid_y_centers):
            for col, x in enumerate(self.grid_x_centers):
                grid_centers.append([x, y])
                grid_indices.append([row, col])
        
        return np.array(grid_centers), np.array(grid_indices)
    
    def index_to_center(self, row: int, col: int) -> np.ndarray:
        """
        将网格索引转换为中心坐标
        
        Args:
            row: 行索引（y 方向）
            col: 列索引（x 方向）
            
        Returns:
            center: [x, y] 坐标
        """
        x = self.grid_x_centers[col]
        y = self.grid_y_centers[row]
        return np.array([x, y])
    
    def indices_to_centers(self, indices: np.ndarray) -> np.ndarray:
        """
        批量将网格索引转换为中心坐标
        
        Args:
            indices: (N, 2) 数组，每行是 [row, col]
            
        Returns:
            centers: (N, 2) 数组，每行是 [x, y]
        """
        centers = []
        for row, col in indices:
            centers.append(self.index_to_center(row, col))
        return np.array(centers)
    
    def index_to_corner(self, row: int, col: int) -> np.ndarray:
        """
        将网格索引转换为左下角坐标
        
        Args:
            row: 行索引（y 方向）
            col: 列索引（x 方向）
            
        Returns:
            corner: [x, y] 坐标
        """
        x = self.gx[col]
        y = self.gy[row]
        return np.array([x, y])
    
    def physical_to_index(self, point: np.ndarray) -> Tuple[int, int]:
        """
        将物理坐标转换为网格索引
        
        Args:
            point: [x, y] 物理坐标
            
        Returns:
            (row, col): 网格索引
        """
        col = int(point[0] / self.grid_size)
        row = int(point[1] / self.grid_size)
        return row, col
    
    def get_meshgrid_corners(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取左下角坐标的 meshgrid（兼容 MCS 现有代码）
        
        Returns:
            grid_x: (rows, cols) 数组，每个元素是该网格的 x 坐标（左下角）
            grid_y: (rows, cols) 数组，每个元素是该网格的 y 坐标（左下角）
        """
        return self.grid_x_corners, self.grid_y_corners
    
    def validate_index(self, row: int, col: int) -> bool:
        """
        验证索引是否有效
        
        Args:
            row: 行索引
            col: 列索引
            
        Returns:
            bool: 索引是否在有效范围内
        """
        return 0 <= row < self.num_rows and 0 <= col < self.num_cols
    
    def get_shape(self) -> Tuple[int, int]:
        """
        获取网格维度
        
        Returns:
            (num_rows, num_cols): 行数和列数
        """
        return self.num_rows, self.num_cols

