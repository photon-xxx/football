"""
相机移动补偿模块 (Camera Movement Estimator)

该模块用于检测和补偿足球视频中相机的移动，确保球员和球的跟踪位置准确。
通过光流法检测相机移动，并调整跟踪对象的位置坐标，消除相机移动对位置测量的影响。

主要功能：
1. 使用光流法检测相机移动
2. 补偿跟踪对象的位置坐标
3. 可视化相机移动信息
4. 支持缓存计算结果以提高性能

"""

import pickle
import cv2
import numpy as np
import os
import sys 
sys.path.append('../../../')
from Module.common.utils import measure_distance,measure_xy_distance

class CameraMovementEstimator():
    """
    相机移动估计器类
    
    使用光流法检测视频帧之间的相机移动，并补偿跟踪对象的位置坐标。
    通过检测视频边缘区域的固定特征点来估算相机移动量。
    """
    
    def __init__(self, frame):
        """
        初始化相机移动估计器
        
        参数:
            frame: 第一帧图像，用于初始化特征检测参数
        """
        # 相机移动的最小距离阈值，低于此值认为相机未移动
        self.minimum_distance = 5

        # Lucas-Kanade光流法参数配置
        self.lk_params = dict(
            winSize = (15, 15),        # 搜索窗口大小
            maxLevel = 2,              # 金字塔层数
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)  # 迭代终止条件
        )

        # 将第一帧转换为灰度图像
        first_frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 创建特征点检测掩码，只在视频边缘区域检测特征点
        # 这样可以避免检测到移动的球员等对象，只检测背景中的固定特征
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:, 0:20] = 1      # 左边缘区域
        mask_features[:, 900:1050] = 1  # 右边缘区域

        # 特征点检测参数配置
        self.features = dict(
            maxCorners = 100,      # 最大特征点数量
            qualityLevel = 0.3,    # 特征点质量阈值
            minDistance = 3,       # 特征点间最小距离
            blockSize = 7,         # 计算特征点的邻域大小
            mask = mask_features   # 特征点检测掩码
        )

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        """
        为跟踪对象添加相机移动补偿后的位置坐标
        
        该方法遍历所有跟踪对象，根据每帧的相机移动量调整对象的位置坐标，
        消除相机移动对位置测量的影响，使位置数据更加准确。
        
        参数:
            tracks: 跟踪结果字典，包含所有对象的跟踪信息
            camera_movement_per_frame: 每帧的相机移动量列表，格式为[[x, y], ...]
        """
        # 遍历所有跟踪对象类型（如球员、球等）
        for object, object_tracks in tracks.items():
            # 遍历每一帧的跟踪结果
            for frame_num, track in enumerate(object_tracks):
                # 遍历该帧中的所有跟踪ID
                for track_id, track_info in track.items():
                    position = track_info['position']
                    
                    # 检查位置数据是否有效
                    if position is None or (isinstance(position, (list, np.ndarray)) and np.isnan(position).any()):
                        # 如果位置无效，设置为NaN
                        position_adjusted = [np.nan, np.nan]
                    else:
                        # 获取当前帧的相机移动量
                        camera_movement = camera_movement_per_frame[frame_num]
                        # 计算补偿后的位置：原始位置减去相机移动量
                        position_adjusted = (position[0] - camera_movement[0], position[1] - camera_movement[1])
                    
                    # 将补偿后的位置添加到跟踪信息中
                    tracks[object][frame_num][track_id]['position_adjusted'] = position_adjusted
                    


    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        """
        计算视频中每帧的相机移动量
        
        使用光流法检测相邻帧之间的相机移动，通过跟踪视频边缘区域的固定特征点
        来估算相机的移动量。支持从缓存文件读取结果以提高性能。
        
        参数:
            frames: 视频帧列表，每帧为BGR格式的图像
            read_from_stub: 是否从缓存文件读取结果
            stub_path: 缓存文件路径
            
        返回:
            camera_movement: 每帧的相机移动量列表，格式为[[x, y], ...]
                           第一帧默认为[0, 0]
        """
        # 尝试从缓存文件读取相机移动数据
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                return pickle.load(f)

        # 初始化相机移动数组，第一帧移动量为0
        camera_movement = [[0, 0]] * len(frames)

        # 处理第一帧，转换为灰度图像并检测特征点
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.features)

        # 遍历后续帧，计算相机移动
        for frame_num in range(1, len(frames)):
            # 将当前帧转换为灰度图像
            frame_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            
            # 使用光流法跟踪特征点
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, frame_gray, old_features, None, **self.lk_params
            )

            # 初始化最大移动距离和相机移动量
            max_distance = 0
            camera_movement_x, camera_movement_y = 0, 0

            # 计算所有特征点的移动距离，选择最大移动距离作为相机移动量
            for i, (new, old) in enumerate(zip(new_features, old_features)):
                new_features_point = new.ravel()
                old_features_point = old.ravel()

                # 计算特征点移动距离
                distance = measure_distance(new_features_point, old_features_point)
                
                # 更新最大移动距离和对应的相机移动量
                if distance > max_distance:
                    max_distance = distance
                    camera_movement_x, camera_movement_y = measure_xy_distance(
                        old_features_point, new_features_point
                    ) 
            
            # 只有当移动距离超过阈值时才认为相机发生了移动
            if max_distance > self.minimum_distance:
                camera_movement[frame_num] = [camera_movement_x, camera_movement_y]
                # 重新检测特征点，为下一帧做准备
                old_features = cv2.goodFeaturesToTrack(frame_gray, **self.features)

            # 更新参考帧
            old_gray = frame_gray.copy()
        
        # 保存计算结果到缓存文件
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(camera_movement, f)

        return camera_movement
    
    def draw_camera_movement(self, frames, camera_movement_per_frame):
        """
        在视频帧上绘制相机移动信息
        
        为每帧添加半透明的信息面板，显示当前帧的相机移动量。
        这有助于可视化相机移动检测的效果和调试。
        
        参数:
            frames: 视频帧列表，每帧为BGR格式的图像
            camera_movement_per_frame: 每帧的相机移动量列表
            
        返回:
            output_frames: 添加了相机移动信息的帧列表
        """
        output_frames = []

        # 遍历每一帧，添加相机移动信息
        for frame_num, frame in enumerate(frames):
            # 复制帧以避免修改原始数据
            frame = frame.copy()

            # 创建半透明的信息面板
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
            alpha = 0.6
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

            # 获取当前帧的相机移动量
            x_movement, y_movement = camera_movement_per_frame[frame_num]
            
            # 在帧上绘制相机移动信息
            frame = cv2.putText(
                frame, f"Camera Movement X: {x_movement:.2f}", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3
            )
            frame = cv2.putText(
                frame, f"Camera Movement Y: {y_movement:.2f}", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3
            )

            output_frames.append(frame) 

        return output_frames