import numpy as np  # 导入 numpy，处理数组与数值运算
import cv2  # 导入 OpenCV，用于图像处理与透视变换
import numpy.typing as npt
from typing import Tuple

class ViewTransformer():  # 定义视角（透视）变换器类
    def __init__(self, 
                 source: npt.NDArray[np.float32],
                 target: npt.NDArray[np.float32]):  # 构造函数：初始化参数与透视变换矩阵
        """
        初始化透视变换器
        
        Args:
            source (npt.NDArray[np.float32]): 源关键点坐标 (N, 2)，像素坐标
            target (npt.NDArray[np.float32]): 目标关键点坐标 (N, 2)，场地坐标
        """
        if source.shape != target.shape:
            raise ValueError("源点和目标点必须具有相同的形状")
        if source.shape[1] != 2:
            raise ValueError("源点和目标点必须是2D坐标")

        # 使用动态关键点计算透视变换矩阵
        self.source_points = source.astype(np.float32)
        self.target_points = target.astype(np.float32)
        
        # 计算透视变换矩阵
        self.trans_matrix = cv2.findHomography(self.source_points, self.target_points)
        if self.trans_matrix is None:
            raise ValueError("无法计算单应性矩阵")

    def transform_points(
            self,
            points: npt.NDArray[np.float32]
    ) -> npt.NDArray[np.float32]:
        """
        使用单应性矩阵变换给定的点坐标

        Args:
            points (npt.NDArray[np.float32]): 待变换的点坐标

        Returns:
            npt.NDArray[np.float32]: 变换后的点坐标

        Raises:
            ValueError: 如果点不是2D坐标
        """
        if points.size == 0:
            return points

        if points.shape[1] != 2:
            raise ValueError("点必须是2D坐标")

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.trans_matrix)
        return transformed_points.reshape(-1, 2).astype(np.float32)

    # TODO：透视变换
    # TODO ：mask检测
    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():  # 遍历每个对象及其轨迹
            for frame_num, track in enumerate(object_tracks):  # 遍历每一帧的轨迹
                for track_id, track_info in track.items():  # 遍历轨迹中的每个跟踪 ID
                    position = track_info['position_adjusted']  # 获取调整后的像素位置
                    position = np.array(position)  # 转为 numpy 数组
                    position_transformed = self.transform_point(position)  # 调用透视变换
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()  # 压缩维度并转为列表
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed  # 存储结果
