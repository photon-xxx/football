"""
基于场地坐标的追踪器

利用项目的独特优势：可以将像素坐标转换为场地坐标（米），在场地坐标空间进行追踪匹配。

核心特点：
- 使用物理距离（米）进行匹配，不受相机运动影响
- 使用4维卡尔曼滤波（x, y, vx, vy）预测位置
- 当距离范围内有多个候选时，可结合ReID特征辅助匹配
- 不使用GMC或仅作为辅助验证

使用方法：
    from Module.common.trackers.tracker_pitch import Tracker
    from Module.common.view_transformer import ViewTransformer
    
    # 初始化ViewTransformer
    view_transformer = ViewTransformer(video_frames, device="cuda")
    
    # 初始化追踪器（必须提供view_transformer）
    tracker = Tracker(
        model_path="path/to/yolo/model.pt",
        view_transformer=view_transformer,
        distance_threshold_primary=2.0,  # 主要匹配阈值（米）
        distance_threshold_secondary=3.0,  # 次要匹配阈值（米）
        max_velocity=0.4,  # 最大速度（米/帧）
        track_buffer=30  # 轨迹保留时间（帧）
    )
    
    # 获取追踪结果（接口与tracker.py完全兼容）
    tracks = tracker.get_object_tracks(video_frames)
"""

from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import os
import sys
import cv2
from scipy.optimize import linear_sum_assignment
from typing import Optional

sys.path.append('../../../')
from Module.common.utils import get_foot_position
from Storage.field_configs.soccer import SoccerPitchConfiguration
from Module.visualizer.pitch_annotation_tool import draw_pitch


class PitchKalmanFilter:
    """
    场地坐标空间的卡尔曼滤波器（4维状态空间）
    
    状态向量: [x, y, vx, vy]
    - x, y: 场地坐标（米）
    - vx, vy: 速度（米/采样间隔）
    """
    
    def __init__(self, process_noise=0.1, measurement_noise=0.5, dt=1.0):
        """
        初始化卡尔曼滤波器
        
        Args:
            process_noise: 过程噪声（位置不确定性）
            measurement_noise: 测量噪声（检测误差）
            dt: 时间步长（采样间隔，单位：帧数）
        """
        self.dt = dt
        # 状态转移矩阵（恒定速度模型）
        # [x, y, vx, vy] -> [x+dt*vx, y+dt*vy, vx, vy]
        self.F = np.array([
            [1, 0, dt, 0],  # x = x + dt*vx
            [0, 1, 0, dt],  # y = y + dt*vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # 观测矩阵（只观测位置，不观测速度）
        self.H = np.array([
            [1, 0, 0, 0],  # 观测x
            [0, 1, 0, 0]   # 观测y
        ], dtype=np.float32)
        
        # 过程噪声协方差矩阵
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.Q[2, 2] = process_noise * 0.1  # 速度噪声较小
        self.Q[3, 3] = process_noise * 0.1
        
        # 测量噪声协方差矩阵
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        
    def init(self, position):
        """
        初始化状态
        
        Args:
            position: [x, y] 场地坐标（米）
        
        Returns:
            mean: 4维状态向量 [x, y, vx, vy]
            covariance: 4x4协方差矩阵
        """
        mean = np.array([position[0], position[1], 0, 0], dtype=np.float32)
        covariance = np.eye(4, dtype=np.float32) * 1.0
        return mean, covariance
    
    def predict(self, mean, covariance):
        """
        预测下一帧状态
        
        Args:
            mean: 当前状态均值 [x, y, vx, vy]
            covariance: 当前协方差矩阵
        
        Returns:
            mean: 预测状态均值
            covariance: 预测协方差矩阵
        """
        mean = self.F @ mean
        covariance = self.F @ covariance @ self.F.T + self.Q
        return mean, covariance
    
    def update(self, mean, covariance, measurement):
        """
        使用观测值更新状态
        
        Args:
            mean: 预测状态均值 [x, y, vx, vy]
            covariance: 预测协方差矩阵
            measurement: 观测值 [x, y] 场地坐标（米）
        
        Returns:
            mean: 更新后的状态均值
            covariance: 更新后的协方差矩阵
        """
        # 计算创新（观测残差）
        y = measurement - self.H @ mean
        
        # 计算创新协方差
        S = self.H @ covariance @ self.H.T + self.R
        
        # 计算卡尔曼增益
        K = covariance @ self.H.T @ np.linalg.inv(S)
        
        # 更新状态
        mean = mean + K @ y
        covariance = (np.eye(4) - K @ self.H) @ covariance
        
        return mean, covariance


class PitchTrack:
    """
    场地坐标空间的轨迹对象
    """
    
    def __init__(self, position_pitch, class_id, frame_num, track_id=None, dt=1.0):
        """
        初始化轨迹
        
        Args:
            position_pitch: [x, y] 场地坐标（米）
            class_id: 类别ID
            frame_num: 帧号
            track_id: 轨迹ID（如果为None则自动分配）
            dt: 时间步长（采样间隔，单位：帧数）
        """
        self.track_id = track_id
        self.class_id = class_id
        self.frame_num = frame_num
        self.last_seen_frame = frame_num
        
        # 场地坐标
        self.position_pitch = np.array(position_pitch, dtype=np.float32)
        
        # 卡尔曼滤波器（使用dt）
        self.kalman_filter = PitchKalmanFilter(dt=dt)
        self.mean, self.covariance = self.kalman_filter.init(position_pitch)
        
        # 轨迹状态
        self.lost_count = 0
        self.is_confirmed = False
        self.hit_streak = 1  # 连续匹配次数
        
        # 存储信息（用于输出）
        self.bbox = None
        self.position_pixel = None
        self.team = None  # 球队ID (1 或 2)
        self.team_color = None  # 球队颜色
        
        # 轨迹历史缓存（重要输出结果）
        # 格式: [{frame_num, position_pitch, velocity, bbox, position_pixel}, ...]
        self.trajectory_history = []
        
    def predict(self):
        """预测下一帧位置"""
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        # 更新预测位置
        self.position_pitch = self.mean[:2]
    
    def update(self, position_pitch, bbox=None, position_pixel=None, frame_num=None, team=None, team_color=None):
        """
        使用新检测更新轨迹
        
        Args:
            position_pitch: [x, y] 场地坐标（米）
            bbox: 边界框（可选）
            position_pixel: 像素坐标（可选）
            frame_num: 帧号（可选）
            team: 球队ID（可选）
            team_color: 球队颜色（可选）
        """
        # 更新卡尔曼滤波器
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, position_pitch
        )
        
        # 更新位置
        self.position_pitch = np.array(position_pitch, dtype=np.float32)
        
        # 更新其他信息
        if bbox is not None:
            self.bbox = bbox
        if position_pixel is not None:
            self.position_pixel = position_pixel
        if frame_num is not None:
            self.last_seen_frame = frame_num
        if team is not None:
            self.team = team
        if team_color is not None:
            self.team_color = team_color
        
        # 缓存轨迹信息（坐标、速度）
        velocity = np.array([self.mean[2], self.mean[3]], dtype=np.float32)  # [vx, vy]
        self.trajectory_history.append({
            'frame_num': frame_num if frame_num is not None else self.frame_num,
            'position_pitch': self.position_pitch.copy(),  # [x, y] 米
            'velocity': velocity.copy(),  # [vx, vy] 米/采样间隔
            'speed': np.sqrt(velocity[0]**2 + velocity[1]**2),  # 速度大小
            'bbox': self.bbox.copy() if self.bbox is not None else None,
            'position_pixel': self.position_pixel.copy() if self.position_pixel is not None else None
        })
        
        # 重置丢失计数
        self.lost_count = 0
        self.hit_streak += 1
        
        # 如果连续匹配超过1帧，确认为有效轨迹（降低确认阈值，提高追踪连续性）
        if self.hit_streak >= 1:
            self.is_confirmed = True
    
    def mark_missed(self):
        """标记为丢失"""
        self.lost_count += 1
    
    def draw_on_pitch(
        self,
        pitch_image: np.ndarray,
        config: SoccerPitchConfiguration,
        draw_trajectory: bool = True,
        draw_current_position: bool = True,
        draw_track_id: bool = True,
        point_radius: int = 10,
        trajectory_thickness: int = 2,
        trajectory_color: Optional[sv.Color] = None,
        point_color: Optional[sv.Color] = None,
        padding: int = 50,
        scale: float = 0.1,
        max_frame_num: Optional[int] = None
    ) -> np.ndarray:
        """
        在球场图像上绘制轨迹
        
        Args:
            pitch_image: 球场图像（BGR格式）
            config: 球场配置对象
            draw_trajectory: 是否绘制轨迹历史路径，默认True
            draw_current_position: 是否绘制当前位置点，默认True
            draw_track_id: 是否显示轨迹ID文本，默认True
            point_radius: 当前位置点的半径（像素），默认10
            trajectory_thickness: 轨迹线条粗细（像素），默认2
            trajectory_color: 轨迹线条颜色，如果为None则根据team自动选择
            point_color: 当前位置点颜色，如果为None则根据team自动选择
            padding: 球场周围的填充像素，默认50
            scale: 球场尺寸缩放比例，默认0.1
            max_frame_num: 最大帧号，如果指定则只绘制该帧及之前的轨迹历史，用于按时间逐步显示
        
        Returns:
            np.ndarray: 绘制后的球场图像
        """
        # 根据team选择颜色（如果没有指定）
        if trajectory_color is None or point_color is None:
            if self.team == 1:
                default_color = sv.Color.RED
            elif self.team == 2:
                default_color = sv.Color.BLUE
            else:
                default_color = sv.Color.WHITE
            
            if trajectory_color is None:
                trajectory_color = default_color
            if point_color is None:
                point_color = default_color
        
        # 绘制轨迹历史路径
        if draw_trajectory and len(self.trajectory_history) > 1:
            # 提取轨迹历史中的所有位置点（如果指定了max_frame_num，只提取该帧及之前的）
            trajectory_points = []
            for hist in self.trajectory_history:
                # 如果指定了max_frame_num，只绘制该帧及之前的轨迹
                if max_frame_num is not None:
                    hist_frame = hist.get('frame_num')
                    if hist_frame is not None and hist_frame > max_frame_num:
                        continue
                
                pos = hist.get('position_pitch')
                if pos is not None:
                    if isinstance(pos, (list, np.ndarray)):
                        pos_array = np.array(pos)
                        if len(pos_array) == 2 and not np.isnan(pos_array).any():
                            trajectory_points.append(pos_array)
            
            if len(trajectory_points) > 1:
                trajectory_points = np.array(trajectory_points)
                # 将路径中的每个点按比例缩放并添加填充偏移
                scaled_path = [
                    (
                        int(point[0] * scale) + padding,
                        int(point[1] * scale) + padding
                    )
                    for point in trajectory_points
                ]
                
                # 连接相邻点绘制路径
                for i in range(len(scaled_path) - 1):
                    cv2.line(
                        img=pitch_image,
                        pt1=scaled_path[i],
                        pt2=scaled_path[i + 1],
                        color=trajectory_color.as_bgr(),
                        thickness=trajectory_thickness
                    )
        
        # 绘制当前位置点
        if draw_current_position:
            if self.position_pitch is not None:
                pos_array = np.array(self.position_pitch)
                if len(pos_array) == 2 and not np.isnan(pos_array).any():
                    scaled_point = (
                        int(pos_array[0] * scale) + padding,
                        int(pos_array[1] * scale) + padding
                    )
                    
                    # 绘制填充的圆（点的内部）
                    cv2.circle(
                        img=pitch_image,
                        center=scaled_point,
                        radius=point_radius,
                        color=point_color.as_bgr(),
                        thickness=-1  # -1表示填充整个圆
                    )
                    
                    # 绘制边框圆（点的边缘）
                    cv2.circle(
                        img=pitch_image,
                        center=scaled_point,
                        radius=point_radius,
                        color=sv.Color.BLACK.as_bgr(),
                        thickness=2
                    )
                    
                    # 显示轨迹ID（透明背景，使用描边增强可读性）
                    if draw_track_id and self.track_id is not None:
                        text = str(self.track_id)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        text_thickness = 2
                        
                        # 计算文本大小以居中显示
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, font, font_scale, text_thickness
                        )
                        text_x = scaled_point[0] - text_width // 2
                        text_y = scaled_point[1] + text_height // 2
                        
                        # 先绘制白色描边（增强可读性，模拟透明背景效果）
                        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                            cv2.putText(
                                img=pitch_image,
                                text=text,
                                org=(text_x + dx, text_y + dy),
                                fontFace=font,
                                fontScale=font_scale,
                                color=(255, 255, 255),  # 白色描边
                                thickness=text_thickness + 1,
                                lineType=cv2.LINE_AA
                            )
                        
                        # 再绘制黑色文本（在描边之上）
                        cv2.putText(
                            img=pitch_image,
                            text=text,
                            org=(text_x, text_y),
                            fontFace=font,
                            fontScale=font_scale,
                            color=(0, 0, 0),  # 黑色文本
                            thickness=text_thickness,
                            lineType=cv2.LINE_AA
                        )
        
        return pitch_image
    
    @staticmethod
    def draw_tracks_as_video(
        tracks_list: list,  # 每帧的轨迹列表 [[track1, track2, ...], [track1, track2, ...], ...]
        output_path: str,
        config: SoccerPitchConfiguration,
        fps: int = 30,
        draw_trajectory: bool = True,
        draw_current_position: bool = True,
        draw_track_id: bool = True,
        point_radius: int = 10,
        trajectory_thickness: int = 2,
        padding: int = 50,
        scale: float = 0.1,
        frame_nums: Optional[list] = None  # 每帧对应的帧号列表，如果为None则使用索引
    ) -> None:
        """
        将轨迹按帧绘制为视频
        
        Args:
            tracks_list: 每帧的轨迹列表，格式为 [[frame0_tracks], [frame1_tracks], ...]
                        每个元素是一个PitchTrack对象的列表
            output_path: 输出视频文件路径
            config: 球场配置对象
            fps: 视频帧率，默认30
            draw_trajectory: 是否绘制轨迹历史路径，默认True
            draw_current_position: 是否绘制当前位置点，默认True
            draw_track_id: 是否显示轨迹ID文本，默认True
            point_radius: 当前位置点的半径（像素），默认10
            trajectory_thickness: 轨迹线条粗细（像素），默认2
            padding: 球场周围的填充像素，默认50
            scale: 球场尺寸缩放比例，默认0.1
            frame_nums: 每帧对应的帧号列表，如果为None则使用索引作为帧号
        
        Returns:
            None，视频保存到output_path
        """
        if len(tracks_list) == 0:
            print("⚠ 警告: 没有轨迹数据，无法生成视频")
            return
        
        # 创建球场背景图像（用于确定视频尺寸）
        pitch_image = draw_pitch(config=config, padding=padding, scale=scale)
        height, width = pitch_image.shape[:2]
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            print(f"⚠ 错误: 无法创建视频文件 {output_path}")
            return
        
        print(f"正在生成轨迹视频: {len(tracks_list)} 帧...")
        
        # 逐帧绘制
        for frame_idx, frame_tracks in enumerate(tracks_list):
            # 创建新的球场图像
            frame_image = draw_pitch(config=config, padding=padding, scale=scale)
            
            # 获取当前帧号
            if frame_nums is not None and frame_idx < len(frame_nums):
                current_frame_num = frame_nums[frame_idx]
            else:
                # 尝试从轨迹历史中获取当前帧号
                current_frame_num = None
                if len(frame_tracks) > 0 and frame_tracks[0].trajectory_history:
                    # 从轨迹历史中获取最新的帧号
                    for hist in frame_tracks[0].trajectory_history:
                        if hist.get('frame_num') is not None:
                            current_frame_num = max(current_frame_num or 0, hist.get('frame_num'))
                
                # 如果没有找到帧号，使用frame_idx作为备用
                if current_frame_num is None:
                    current_frame_num = frame_idx
            
            # 按每帧读取轨迹信息：处理所有轨迹（包括丢失的）
            for track in frame_tracks:
                track_last_seen = getattr(track, 'last_seen_frame', None)
                
                # 获取当前帧的位置（用于绘制圆点和轨迹ID）
                current_position = None
                if track.position_pitch is not None:
                    current_position = np.array(track.position_pitch)
                
                # 如果当前位置无效，跳过
                if current_position is None or len(current_position) != 2 or np.isnan(current_position).any():
                    continue
                
                # 判断轨迹在该帧是否仍然活跃
                is_track_active = (track_last_seen is None or current_frame_num <= track_last_seen)
                
                # 获取轨迹历史中到当前帧为止的所有位置点（用于绘制轨迹线）
                # 对于丢失的轨迹，只绘制到 last_seen_frame 为止的轨迹线
                trajectory_points = []
                for hist in track.trajectory_history:
                    hist_frame = hist.get('frame_num')
                    # 只包含当前帧及之前帧的位置
                    if hist_frame is not None and hist_frame <= current_frame_num:
                        # 如果轨迹已丢失，只绘制到 last_seen_frame 为止的轨迹线
                        if not is_track_active and track_last_seen is not None and hist_frame > track_last_seen:
                            continue
                        pos = hist.get('position_pitch')
                        if pos is not None:
                            if isinstance(pos, (list, np.ndarray)):
                                pos_array = np.array(pos)
                                if len(pos_array) == 2 and not np.isnan(pos_array).any():
                                    trajectory_points.append((hist_frame, pos_array))
                
                # 按帧号排序
                trajectory_points.sort(key=lambda x: x[0])
                
                # 根据team选择颜色
                if track.team == 1:
                    track_color = sv.Color.RED
                elif track.team == 2:
                    track_color = sv.Color.BLUE
                else:
                    track_color = sv.Color.WHITE
                
                # 绘制轨迹历史路径（之前帧的位置连续标注为轨迹线）
                if draw_trajectory and len(trajectory_points) > 1:
                    # 提取位置点
                    path_points = [point[1] for point in trajectory_points]
                    path_points = np.array(path_points)
                    
                    # 将路径中的每个点按比例缩放并添加填充偏移
                    scaled_path = [
                        (
                            int(point[0] * scale) + padding,
                            int(point[1] * scale) + padding
                        )
                        for point in path_points
                    ]
                    
                    # 连接相邻点绘制路径
                    for i in range(len(scaled_path) - 1):
                        cv2.line(
                            img=frame_image,
                            pt1=scaled_path[i],
                            pt2=scaled_path[i + 1],
                            color=track_color.as_bgr(),
                            thickness=trajectory_thickness
                        )
                
                # 绘制当前帧的位置为圆点
                # 对于丢失的轨迹，仍然绘制最后可见位置的点
                if draw_current_position:
                    scaled_point = (
                        int(current_position[0] * scale) + padding,
                        int(current_position[1] * scale) + padding
                    )
                    
                    # 绘制填充的圆（点的内部）
                    # 如果轨迹已丢失，使用半透明或不同的样式
                    if not is_track_active:
                        # 丢失的轨迹：使用半透明或虚线边框
                        overlay = frame_image.copy()
                        cv2.circle(
                            img=overlay,
                            center=scaled_point,
                            radius=point_radius,
                            color=track_color.as_bgr(),
                            thickness=-1
                        )
                        cv2.addWeighted(overlay, 0.5, frame_image, 0.5, 0, frame_image)
                    else:
                        # 活跃的轨迹：正常绘制
                        cv2.circle(
                            img=frame_image,
                            center=scaled_point,
                            radius=point_radius,
                            color=track_color.as_bgr(),
                            thickness=-1  # -1表示填充整个圆
                        )
                    
                    # 绘制边框圆（点的边缘）
                    cv2.circle(
                        img=frame_image,
                        center=scaled_point,
                        radius=point_radius,
                        color=sv.Color.BLACK.as_bgr(),
                        thickness=2
                    )
                    
                    # 标注轨迹号（在当前帧位置）
                    # 对于丢失的轨迹，仍然显示轨迹ID，方便识别
                    if draw_track_id and track.track_id is not None:
                        text = str(track.track_id)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.6
                        text_thickness = 2
                        
                        # 计算文本大小以居中显示
                        (text_width, text_height), baseline = cv2.getTextSize(
                            text, font, font_scale, text_thickness
                        )
                        text_x = scaled_point[0] - text_width // 2
                        text_y = scaled_point[1] + text_height // 2
                        
                        # 先绘制白色描边（增强可读性，透明背景效果）
                        for dx, dy in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
                            cv2.putText(
                                img=frame_image,
                                text=text,
                                org=(text_x + dx, text_y + dy),
                                fontFace=font,
                                fontScale=font_scale,
                                color=(255, 255, 255),  # 白色描边
                                thickness=text_thickness + 1,
                                lineType=cv2.LINE_AA
                            )
                        
                        # 再绘制黑色文本（在描边之上）
                        cv2.putText(
                            img=frame_image,
                            text=text,
                            org=(text_x, text_y),
                            fontFace=font,
                            fontScale=font_scale,
                            color=(0, 0, 0),  # 黑色文本
                            thickness=text_thickness,
                            lineType=cv2.LINE_AA
                        )
            
            # 添加帧号文本（使用真实帧号）
            frame_text = f"Frame: {current_frame_num}"
            cv2.putText(
                img=frame_image,
                text=frame_text,
                org=(10, 30),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.0,
                color=(255, 255, 255),
                thickness=2,
                lineType=cv2.LINE_AA
            )
            
            # 写入视频帧
            video_writer.write(frame_image)
            
            # 显示进度
            if (frame_idx + 1) % 30 == 0:
                print(f"  已处理 {frame_idx + 1}/{len(tracks_list)} 帧")
        
        # 释放视频写入器
        video_writer.release()
        print(f"✓ 轨迹视频已保存到: {output_path}")


class Tracker:
    """
    基于场地坐标的追踪器
    
    兼容现有接口，但需要提供view_transformer参数才能使用场地坐标追踪。
    如果不提供view_transformer，会回退到简单的距离匹配。
    """
    
    def __init__(self, model_path, nms_threshold=0.5, 
                 view_transformer=None,
                 distance_threshold_primary=2.0,
                 distance_threshold_secondary=3.0,
                 max_velocity=0.4,
                 track_buffer=30,
                 min_track_length=3,
                 sampling_rate=5.0,
                 video_fps=30.0):
        """
        初始化追踪器
        
        Args:
            model_path: YOLO模型路径
            nms_threshold: NMS阈值
            view_transformer: ViewTransformer对象（必需，用于坐标转换）
            distance_threshold_primary: 主要匹配距离阈值（米）
            distance_threshold_secondary: 次要匹配距离阈值（米）
            max_velocity: 最大速度限制（米/秒）
            track_buffer: 轨迹保留时间（丢失后仍保留的采样次数）
            min_track_length: 最小轨迹长度（确认轨迹）
            sampling_rate: 采样频率（Hz），默认5Hz
            video_fps: 视频帧率（FPS），默认30fps
        """
        self.model = YOLO(model_path)
        self.nms_threshold = nms_threshold
        self.view_transformer = view_transformer
        
        # 采样参数
        self.sampling_rate = sampling_rate  # Hz
        self.video_fps = video_fps  # FPS
        # 如果采样频率大于等于视频帧率，则禁用采样（处理所有帧）
        if sampling_rate >= video_fps:
            self.use_sampling = False
            self.sample_interval = 1
            self.dt = 1.0  # 卡尔曼滤波时间步长（帧数）
        else:
            self.use_sampling = True
        self.sample_interval = int(video_fps / sampling_rate)  # 采样间隔（帧数）
        self.dt = self.sample_interval  # 卡尔曼滤波时间步长（帧数）
        
        # 追踪参数
        self.distance_threshold_primary = distance_threshold_primary
        self.distance_threshold_secondary = distance_threshold_secondary
        # 将最大速度从米/秒转换为米/帧（或米/采样间隔）
        if self.use_sampling:
            self.max_velocity = max_velocity * (1.0 / sampling_rate)  # 米/采样间隔
        else:
            self.max_velocity = max_velocity / video_fps  # 米/帧
        self.track_buffer = track_buffer
        self.min_track_length = min_track_length
        
        # 轨迹管理
        self.tracks = []  # 所有轨迹
        self.next_id = 1  # 下一个轨迹ID
        
        if self.use_sampling:
            print(f"追踪器配置: 采样频率={sampling_rate}Hz, 视频帧率={video_fps}fps, 采样间隔={self.sample_interval}帧")
        else:
            print(f"追踪器配置: 采样已禁用，处理所有帧，视频帧率={video_fps}fps")
        
        # 检查view_transformer
        if view_transformer is None:
            print("警告: 未提供view_transformer，将使用简化的追踪模式")
    
    def track_detections(self, tracking_inputs, frames=None):
        """
        基于检测结果进行追踪（独立方法）
        
        新的追踪逻辑：
        1. 使用第一次有球员检测的帧初始化轨迹
        2. 循环匹配：对比前后两帧进行智能匹配
        3. 删除小于5帧的轨迹
        
        Args:
            tracking_inputs: 追踪输入数据（由save_tracking_inputs保存的格式）
            frames: 视频帧列表（可选，用于处理球检测）
        
        Returns:
            tracks: 追踪结果字典
                {
                    "players": [{track_id: {"bbox": [x1,y1,x2,y2], "team": 1, "team_color": [...]}}, ...],
                    "referees": [{track_id: {"bbox": [x1,y1,x2,y2]}}, ...],
                    "ball": [{1: {"bbox": [x1,y1,x2,y2]}}, ...]
                }
        """
        # 初始化输出结构
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
        # 处理球检测（如果有frames）
        if frames is not None:
            detections = self.detect_frames(frames, conf_threshold=0.3)
            for frame_num, detection in enumerate(detections):
                cls_names = detection.names
                cls_names_inv = {v: k for k, v in cls_names.items()}
                detection_supervision = sv.Detections.from_ultralytics(detection)
                
                # 处理守门员类别
                for object_ind, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[object_ind] = cls_names_inv["player"]
                
                # 分离球
                BALL_ID = cls_names_inv['ball']
                ball_detections = detection_supervision[detection_supervision.class_id == BALL_ID]
                
                # 对球进行边界框扩展
                if len(ball_detections) > 0:
                    ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
                    for frame_detection in ball_detections:
                        bbox = frame_detection[0].tolist()
                        tracks["ball"].append({1: {"bbox": bbox}})
                        break
                else:
                    tracks["ball"].append({1: {"bbox": [np.nan, np.nan, np.nan, np.nan]}})
        else:
            # 如果没有frames，初始化空的球检测
            for _ in tracking_inputs:
                tracks["ball"].append({1: {"bbox": [np.nan, np.nan, np.nan, np.nan]}})
    
        # 步骤1：找到第一次有球员检测的帧，初始化轨迹
        first_frame_with_players = None
        for idx, tracking_input in enumerate(tracking_inputs):
            detections_pitch = tracking_input['detections_pitch']
            detections_info = tracking_input['detections_info']
            
            # 检查是否有球员检测
            player_detections = []
            for i, det_info in enumerate(detections_info):
                if det_info.get('class_id') == tracking_input['cls_names_inv'].get('player', 0) - 1:
                    player_detections.append((i, detections_pitch[i], det_info))
            
            if len(player_detections) > 0:
                first_frame_with_players = idx
                frame_num = tracking_input['frame_num']
                
                # 初始化轨迹：有几个球员检测，初始化几条轨迹
                for det_idx, det_pitch, det_info in player_detections:
                    new_track = PitchTrack(
                        det_pitch,
                        det_info['class_id'],
                        frame_num,
                        track_id=self.next_id,
                        dt=self.dt
                    )
                    new_track.bbox = det_info.get('bbox')
                    new_track.position_pixel = det_info.get('position_pixel')
                    if 'team' in det_info:
                        new_track.team = det_info['team']
                    if 'team_color' in det_info:
                        new_track.team_color = det_info['team_color']
                    # 初始化轨迹历史
                    new_track.update(
                        det_pitch,
                        bbox=det_info.get('bbox'),
                        position_pixel=det_info.get('position_pixel'),
                        frame_num=frame_num,
                        team=det_info.get('team'),
                        team_color=det_info.get('team_color')
                    )
                    self.tracks.append(new_track)
                    self.next_id += 1
                
                print(f"  初始化: 帧{frame_num}，创建{len(player_detections)}条轨迹")
                break
        
        if first_frame_with_players is None:
            print("  警告: 没有找到有球员检测的帧")
            return tracks
        
        # 步骤2：处理每一帧的追踪（从初始化帧之后开始）
        for frame_idx, tracking_input in enumerate(tracking_inputs):
            frame_num = tracking_input['frame_num']
            detections_pitch = tracking_input['detections_pitch']
            detections_info = tracking_input['detections_info']
            cls_names = tracking_input['cls_names']
            cls_names_inv = tracking_input['cls_names_inv']
            
            # 初始化该帧的字典
            tracks["players"].append({})
            tracks["referees"].append({})
            
            # 跳过初始化帧（已经在步骤1中处理）
            if frame_idx == first_frame_with_players:
                # 初始化帧已经处理，只需要存储结果
                # 存储初始化帧的轨迹
                for track in self.tracks:
                    if track.lost_count == 0:
                        bbox = track.bbox
                        if bbox is not None:
                            cls_id = track.class_id
                            track_id = track.track_id
                            if cls_id == cls_names_inv['player'] - 1:
                                track_data = {
                                    "bbox": bbox,
                                    "position_pitch": track.position_pitch.tolist() if isinstance(track.position_pitch, np.ndarray) else track.position_pitch
                                }
                                if track.position_pixel is not None:
                                    track_data['position_pixel'] = track.position_pixel
                                if track.team is not None:
                                    track_data['team'] = track.team
                                if track.team_color is not None:
                                    track_data['team_color'] = track.team_color
                                tracks["players"][frame_num][track_id] = track_data
                continue
            
            # 如果使用采样，只处理采样间隔的帧进行追踪；否则处理所有帧
            if self.use_sampling and frame_num % self.sample_interval != 0:
                # 对于非采样帧，仍然需要预测轨迹位置并存储结果（用于插值）
                for track in self.tracks:
                    if track.is_confirmed:
                        track.predict()  # 预测位置（在场地坐标空间）
                # 存储预测的追踪结果（用于非采样帧的插值）
                # 核心：必须包含场地坐标（追踪器在场地坐标空间工作）
                for track in self.tracks:
                    if track.lost_count == 0 and track.is_confirmed:
                        bbox = track.bbox
                        if bbox is not None:
                            cls_id = track.class_id
                            track_id = track.track_id
                            if cls_id == cls_names_inv['player'] - 1:
                                # 核心：输出场地坐标（预测位置）
                                track_data = {
                                    "bbox": bbox,
                                    "position_pitch": track.position_pitch.tolist() if isinstance(track.position_pitch, np.ndarray) else track.position_pitch
                                }
                                if track.position_pixel is not None:
                                    track_data['position_pixel'] = track.position_pixel
                                if track.team is not None:
                                    track_data['team'] = track.team
                                if track.team_color is not None:
                                    track_data['team_color'] = track.team_color
                                tracks["players"][frame_num][track_id] = track_data
                            elif cls_id == cls_names_inv['referee'] - 1:
                                tracks["referees"][frame_num][track_id] = {"bbox": bbox}
                continue
            
            # 处理球员和裁判（使用场地坐标追踪）
            if len(detections_pitch) > 0:
                # 1. 预测所有轨迹的位置
                for track in self.tracks:
                    track.predict()
                
                # 2. 匹配轨迹和检测（使用新的匹配算法，threshold=3.0米）
                matches, unmatched_tracks, unmatched_dets = self._match_tracks_detections(
                    self.tracks, detections_pitch, detections_info, threshold=3.0
                )
                
                # 调试信息（每30帧输出一次）
                if frame_num % 30 == 0:
                    print(f"  调试[帧{frame_num}]: 检测数={len(detections_pitch)}, 轨迹数={len(self.tracks)}, "
                          f"匹配数={len(matches)}, 未匹配轨迹={len(unmatched_tracks)}, 未匹配检测={len(unmatched_dets)}")
                
                # 3. 更新匹配的轨迹
                for track_idx, det_idx in matches:
                    track = self.tracks[track_idx]
                    det_info = detections_info[det_idx]
                    track.update(
                        detections_pitch[det_idx],
                        bbox=det_info['bbox'],
                        position_pixel=det_info['position_pixel'],
                        frame_num=frame_num,
                        team=det_info.get('team'),
                        team_color=det_info.get('team_color')
                    )
                
                # 4. 标记未匹配的轨迹为丢失
                for track_idx in unmatched_tracks:
                    self.tracks[track_idx].mark_missed()
                
                # 5. 为未匹配的检测创建新轨迹
                for det_idx in unmatched_dets:
                    det_info = detections_info[det_idx]
                    new_track = PitchTrack(
                        detections_pitch[det_idx],
                        det_info['class_id'],
                        frame_num,
                        track_id=self.next_id,
                        dt=self.dt
                    )
                    new_track.bbox = det_info['bbox']
                    new_track.position_pixel = det_info['position_pixel']
                    if 'team' in det_info:
                        new_track.team = det_info['team']
                    if 'team_color' in det_info:
                        new_track.team_color = det_info['team_color']
                    # 初始化轨迹历史
                    new_track.update(
                        detections_pitch[det_idx],
                        bbox=det_info.get('bbox'),
                        position_pixel=det_info.get('position_pixel'),
                        frame_num=frame_num,
                        team=det_info.get('team'),
                        team_color=det_info.get('team_color')
                    )
                    self.tracks.append(new_track)
                    self.next_id += 1
            else:
                # 没有检测，所有轨迹标记为丢失
                for track in self.tracks:
                    track.mark_missed()
                
            # 6. 存储追踪结果（对所有帧，包括非采样帧）
            # 核心：基于场地坐标空间追踪，输出时必须包含场地坐标
            # 修改：存储所有未丢失的轨迹，即使未确认（提高追踪连续性）
            for track in self.tracks:
                if track.lost_count == 0:  # 只要求未丢失，不要求确认
                    bbox = track.bbox
                    if bbox is not None:
                        cls_id = track.class_id
                        track_id = track.track_id
                        
                        # 根据类别ID判断对象类型
                        if cls_id == cls_names_inv['player'] - 1:
                            # 核心：输出场地坐标（追踪器在场地坐标空间工作）
                            track_data = {
                                "bbox": bbox,
                                "position_pitch": track.position_pitch.tolist() if isinstance(track.position_pitch, np.ndarray) else track.position_pitch
                            }
                            if track.position_pixel is not None:
                                track_data['position_pixel'] = track.position_pixel
                            if track.team is not None:
                                track_data['team'] = track.team
                            if track.team_color is not None:
                                track_data['team_color'] = track.team_color
                            tracks["players"][frame_num][track_id] = track_data
                        elif cls_id == cls_names_inv['referee'] - 1:
                            tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            
        # 删除不符合条件的轨迹（在最终统计前）
        # 计算总帧数
        if len(tracking_inputs) > 0:
            first_frame = tracking_inputs[0]['frame_num']
            last_frame = tracking_inputs[-1]['frame_num']
            total_frames = last_frame - first_frame + 1
            min_track_duration = total_frames * 0.2  # 最小存在时间：总帧数的20%
        else:
            total_frames = 0
            min_track_duration = 0
        
        tracks_to_remove = []
        for track_idx, track in enumerate(self.tracks):
            # 计算轨迹存在时间
            track_duration = track.last_seen_frame - track.frame_num + 1
            
            # 条件1：删除存在时间小于总帧数20%的轨迹
            if track_duration < min_track_duration:
                tracks_to_remove.append(track_idx)
                continue
            
            # 条件2：删除一直不动的轨迹
            if len(track.trajectory_history) > 1:
                # 获取轨迹历史中的所有位置
                positions = [hist['position_pitch'] for hist in track.trajectory_history if hist.get('position_pitch') is not None]
                if len(positions) > 1:
                    positions_array = np.array(positions)
                    # 计算位置的范围（最小值和最大值）
                    min_pos = np.min(positions_array, axis=0)
                    max_pos = np.max(positions_array, axis=0)
                    # 计算范围的对角线距离（更高效的方法）
                    # 注意：position_pitch的单位是米
                    range_distance = np.linalg.norm(max_pos - min_pos)
                    
                    # 如果位置范围小于10cm，认为轨迹一直不动
                    if range_distance < 10:  
                        tracks_to_remove.append(track_idx)
                        continue
            
            # 条件3：删除存在帧数小于30帧的轨迹
            if track_duration < 30:
                tracks_to_remove.append(track_idx)
                continue
        
        # 从后往前删除，避免索引变化
        for track_idx in reversed(tracks_to_remove):
            removed_track = self.tracks.pop(track_idx)
            if len(tracks_to_remove) <= 10:  # 只打印少量删除信息，避免输出过多
                print(f"  删除轨迹 {removed_track.track_id}: 存在时间={removed_track.last_seen_frame - removed_track.frame_num + 1}帧")
        
        if len(tracks_to_remove) > 0:
            print(f"  已删除 {len(tracks_to_remove)} 条不符合条件的轨迹")
            
        # 最终统计
        total_stored = sum(len(tracks["players"][i]) for i in range(len(tracks["players"])))
        print(f"  调试：追踪完成，共存储 {total_stored} 个球员追踪结果")
        print(f"  调试：最终轨迹数: {len(self.tracks)}, 已确认轨迹数: {sum(1 for t in self.tracks if t.is_confirmed)}")
        
        return tracks
    
    def _calculate_distance_matrix(self, tracks, detections_pitch):
        """
        计算距离矩阵（场地坐标空间）
        
        Args:
            tracks: 轨迹列表
            detections_pitch: 检测的场地坐标列表 [[x, y], ...]
        
        Returns:
            distance_matrix: (N_tracks, N_detections) 距离矩阵（米）
        """
        if len(tracks) == 0 or len(detections_pitch) == 0:
            return np.array([])
        
        # 获取轨迹的预测位置
        track_positions = np.array([track.position_pitch for track in tracks])
        
        # 获取检测位置
        det_positions = np.array(detections_pitch)
        
        # 计算欧氏距离矩阵
        # track_positions: (N_tracks, 2)
        # det_positions: (N_detections, 2)
        # distance_matrix: (N_tracks, N_detections)
        diff = track_positions[:, np.newaxis, :] - det_positions[np.newaxis, :, :]
        distance_matrix = np.sqrt(np.sum(diff ** 2, axis=2))
        
        return distance_matrix
    

    def _match_tracks_detections(self, tracks, detections_pitch, detections_info, threshold=3.0):
        """
        新的匹配算法：基于距离和球队信息的智能匹配
        
        匹配规则：
        1. 若球员坐标周围threshold米内没有除自身以外的其他球员，并入最近的轨迹
        2. 若threshold范围内有其他球员，忽略对方球员，本方球员根据轨迹历史并入最近轨迹
        
        Args:
            tracks: 轨迹列表
            detections_pitch: 检测的场地坐标列表（单位：cm）
            detections_info: 检测信息列表 [{bbox, position_pixel, class_id, team, ...}, ...]
            threshold: 匹配距离阈值（米），默认3.0米
        
        Returns:
            matches: [(track_idx, det_idx), ...] 匹配对
            unmatched_tracks: [track_idx, ...] 未匹配的轨迹索引
            unmatched_dets: [det_idx, ...] 未匹配的检测索引
        
        注意：position_pitch的单位是cm，需要将阈值从米转换为cm
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections_pitch)))
        if len(detections_pitch) == 0:
            return [], list(range(len(tracks))), []
        
        # 将阈值从米转换为cm（position_pitch的单位是cm）
        threshold_cm = threshold * 100.0
        
        matches = []
        matched_track_indices = set()
        matched_det_indices = set()
        
        # 为每个检测找到匹配的轨迹
        for det_idx, det_pitch in enumerate(detections_pitch):
            det_info = detections_info[det_idx]
            det_team = det_info.get('team')
            
            # 如果检测没有球队信息，跳过
            if det_team is None:
                continue
            
            # 找到该检测周围threshold米内的所有其他检测
            nearby_detections = []
            for other_det_idx, other_det_pitch in enumerate(detections_pitch):
                if other_det_idx == det_idx:
                    continue
                distance = np.linalg.norm(np.array(det_pitch) - np.array(other_det_pitch))
                if distance <= threshold_cm:
                    nearby_detections.append((other_det_idx, other_det_pitch, detections_info[other_det_idx]))
            
            # 找到该检测周围threshold米内的所有轨迹
            candidate_tracks = []
            for track_idx, track in enumerate(tracks):
                if track_idx in matched_track_indices:
                    continue
                
                # 获取轨迹的最新位置（从轨迹历史缓存）
                if len(track.trajectory_history) > 0:
                    # 使用轨迹历史中的最新位置
                    latest_position = track.trajectory_history[-1]['position_pitch']
                else:
                    # 如果没有历史，使用当前位置
                    latest_position = track.position_pitch
                
                distance = np.linalg.norm(np.array(det_pitch) - np.array(latest_position))
                
                if distance <= threshold_cm:
                    candidate_tracks.append((track_idx, track, distance, latest_position))
            
            # 匹配逻辑
            if len(nearby_detections) == 0:
                # 情况1：周围3米内没有其他球员，并入最近的轨迹
                if len(candidate_tracks) > 0:
                    # 按距离排序，选择最近的
                    candidate_tracks.sort(key=lambda x: x[2])
                    best_track_idx, best_track, best_distance, _ = candidate_tracks[0]
                    
                    matches.append((best_track_idx, det_idx))
                    matched_track_indices.add(best_track_idx)
                    matched_det_indices.add(det_idx)
            else:
                # 情况2：周围3米内有其他球员
                # 检查是否有本方球员
                same_team_detections = [d for d in nearby_detections if d[2].get('team') == det_team]
                
                if len(same_team_detections) > 0:
                    # 有本方球员，需要根据轨迹历史匹配
                    # 只考虑同队的轨迹
                    same_team_tracks = [t for t in candidate_tracks if t[1].team == det_team]
                    
                    if len(same_team_tracks) > 0:
                        # 按距离排序，选择最近的同队轨迹
                        same_team_tracks.sort(key=lambda x: x[2])
                        best_track_idx, best_track, best_distance, _ = same_team_tracks[0]
                        
                        matches.append((best_track_idx, det_idx))
                        matched_track_indices.add(best_track_idx)
                        matched_det_indices.add(det_idx)
                else:
                    # 周围只有对方球员，忽略对方球员，直接匹配最近的轨迹
                    if len(candidate_tracks) > 0:
                        candidate_tracks.sort(key=lambda x: x[2])
                        best_track_idx, best_track, best_distance, _ = candidate_tracks[0]
                        
                        matches.append((best_track_idx, det_idx))
                        matched_track_indices.add(best_track_idx)
                        matched_det_indices.add(det_idx)
        
        # 找出未匹配的轨迹和检测
        unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_track_indices]
        unmatched_dets = [i for i in range(len(detections_pitch)) if i not in matched_det_indices]
        
        return matches, unmatched_tracks, unmatched_dets
    

    # TODO:去除这个函数，只保留track_detections函数
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, view_transformer=None):
        """
        获取目标跟踪结果（兼容接口，内部调用检测+追踪）
        
        Args:
            frames: 视频帧列表
            read_from_stub: 是否从缓存读取
            stub_path: 缓存文件路径
            view_transformer: ViewTransformer对象（如果初始化时未提供，可以在这里提供）
            team_assigner: TeamAssigner对象（可选，如果提供则在检测后立即分配球队）
        
        Returns:
            tracks: 追踪结果字典，格式与tracker.py完全兼容
                {
                    "players": [{track_id: {"bbox": [x1,y1,x2,y2], "team": 1, "team_color": [...]}}, ...],
                    "referees": [{track_id: {"bbox": [x1,y1,x2,y2]}}, ...],
                    "ball": [{1: {"bbox": [x1,y1,x2,y2]}}, ...]
                }
        """
        # 使用参数中的view_transformer或初始化时的
        view_transformer = view_transformer or self.view_transformer
        
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks
        
        if view_transformer is None:
            raise ValueError("必须提供view_transformer参数才能使用场地坐标追踪")
        
        # 步骤1：YOLO检测
        print(f"正在使用YOLO检测 {len(frames)} 帧...")
        detections = self.detect_frames(frames)
        
        # 统计检测结果
        total_raw_detections = 0
        player_detections_count = 0
        for det in detections[:10]:  # 检查前10帧
            if hasattr(det, 'boxes') and det.boxes is not None:
                total_raw_detections += len(det.boxes)
                cls_names = det.names
                cls_names_inv = {v: k for k, v in cls_names.items()}
                if 'player' in cls_names_inv:
                    player_id = cls_names_inv['player']
                    player_boxes = det.boxes[det.boxes.cls == player_id]
                    player_detections_count += len(player_boxes)
        print(f"  调试：前10帧原始检测数: {total_raw_detections}, 球员检测数: {player_detections_count}")
        
        # 步骤2：将检测结果转换为追踪输入（不包含球队信息，追踪器只负责追踪）
        tracking_inputs = self._process_detections_to_tracking_input(
            detections, view_transformer, detections_with_teams=None
        )
        
        # 统计追踪输入
        total_tracking_detections = sum(len(ti['detections_pitch']) for ti in tracking_inputs[:10])
        print(f"  调试：前10帧追踪输入检测数: {total_tracking_detections}")
        
        # 步骤3：基于检测结果进行追踪（不包含球队信息）
        tracks = self.track_detections(tracking_inputs, frames)
            
        # 保存缓存
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        # 返回tracks（兼容现有接口）
        return tracks

