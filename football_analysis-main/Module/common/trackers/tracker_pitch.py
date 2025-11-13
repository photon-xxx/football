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
import pandas as pd
import os
import sys
from collections import deque
from scipy.optimize import linear_sum_assignment

sys.path.append('../../../')
from Module.common.utils import get_center_of_bbox, get_bbox_width, get_foot_position


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
        
        # 轨迹历史缓存（重要输出结果）
        # 格式: [{frame_num, position_pitch, velocity, bbox, position_pixel}, ...]
        self.trajectory_history = []
        
    def predict(self):
        """预测下一帧位置"""
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        # 更新预测位置
        self.position_pitch = self.mean[:2]
    
    def update(self, position_pitch, bbox=None, position_pixel=None, frame_num=None):
        """
        使用新检测更新轨迹
        
        Args:
            position_pitch: [x, y] 场地坐标（米）
            bbox: 边界框（可选）
            position_pixel: 像素坐标（可选）
            frame_num: 帧号（可选）
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
        
        # 如果连续匹配超过3帧，确认为有效轨迹
        if self.hit_streak >= 3:
            self.is_confirmed = True
    
    def mark_missed(self):
        """标记为丢失"""
        self.lost_count += 1


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
        self.sample_interval = int(video_fps / sampling_rate)  # 采样间隔（帧数）
        self.dt = self.sample_interval  # 卡尔曼滤波时间步长（帧数）
        
        # 追踪参数
        self.distance_threshold_primary = distance_threshold_primary
        self.distance_threshold_secondary = distance_threshold_secondary
        # 将最大速度从米/秒转换为米/采样间隔
        self.max_velocity = max_velocity * (1.0 / sampling_rate)  # 米/采样间隔
        self.track_buffer = track_buffer
        self.min_track_length = min_track_length
        
        # 轨迹管理
        self.tracks = []  # 所有轨迹
        self.next_id = 1  # 下一个轨迹ID
        
        print(f"追踪器配置: 采样频率={sampling_rate}Hz, 视频帧率={video_fps}fps, 采样间隔={self.sample_interval}帧")
        
        # 检查view_transformer
        if view_transformer is None:
            print("警告: 未提供view_transformer，将使用简化的追踪模式")
    
    def add_position_to_tracks(self, tracks):
        """给轨迹增加位置点信息（兼容接口）"""
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    
                    if np.isnan(bbox).any():
                        position = (np.nan, np.nan)
                    else:
                        if object == 'ball':
                            position = get_center_of_bbox(bbox)
                        else:
                            position = get_foot_position(bbox)
                    
                    tracks[object][frame_num][track_id]['position'] = position
    
    def interpolate_ball_positions(self, ball_positions):
        """处理将球位置置为NAN的帧（兼容接口）"""
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])
        
        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()
        
        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions
    
    def detect_frames(self, frames, batch_size=12):
        """处理批次控制"""
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            results = self.model.predict(batch_frames, conf=0.3, verbose=False)
            detections += results
        return detections
    
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
    
    def _match_tracks_detections(self, tracks, detections_pitch, detections_info):
        """
        匹配轨迹和检测
        
        Args:
            tracks: 轨迹列表
            detections_pitch: 检测的场地坐标列表
            detections_info: 检测信息列表 [{bbox, position_pixel, class_id, confidence}, ...]
        
        Returns:
            matches: [(track_idx, det_idx), ...] 匹配对
            unmatched_tracks: [track_idx, ...] 未匹配的轨迹索引
            unmatched_dets: [det_idx, ...] 未匹配的检测索引
        """
        if len(tracks) == 0:
            return [], [], list(range(len(detections_pitch)))
        if len(detections_pitch) == 0:
            return [], list(range(len(tracks))), []
        
        # 计算距离矩阵
        distance_matrix = self._calculate_distance_matrix(tracks, detections_pitch)
        
        # 第一轮匹配：主要阈值
        matches_primary = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections_pitch)))
        
        # 找到距离小于主要阈值的匹配
        for track_idx in unmatched_tracks[:]:
            candidates = []
            for det_idx in unmatched_dets:
                if distance_matrix[track_idx, det_idx] < self.distance_threshold_primary:
                    candidates.append((det_idx, distance_matrix[track_idx, det_idx]))
            
            if len(candidates) == 1:
                # 只有一个候选，直接匹配
                det_idx = candidates[0][0]
                matches_primary.append((track_idx, det_idx))
                unmatched_tracks.remove(track_idx)
                unmatched_dets.remove(det_idx)
            elif len(candidates) > 1:
                # 多个候选，选择距离最近的
                candidates.sort(key=lambda x: x[1])
                det_idx = candidates[0][0]
                matches_primary.append((track_idx, det_idx))
                unmatched_tracks.remove(track_idx)
                unmatched_dets.remove(det_idx)
        
        # 第二轮匹配：次要阈值（仅对未匹配的轨迹）
        matches_secondary = []
        for track_idx in unmatched_tracks[:]:
            candidates = []
            for det_idx in unmatched_dets:
                if distance_matrix[track_idx, det_idx] < self.distance_threshold_secondary:
                    candidates.append((det_idx, distance_matrix[track_idx, det_idx]))
            
            if len(candidates) == 1:
                det_idx = candidates[0][0]
                matches_secondary.append((track_idx, det_idx))
                unmatched_tracks.remove(track_idx)
                unmatched_dets.remove(det_idx)
            elif len(candidates) > 1:
                # 多个候选，选择距离最近的
                candidates.sort(key=lambda x: x[1])
                det_idx = candidates[0][0]
                matches_secondary.append((track_idx, det_idx))
                unmatched_tracks.remove(track_idx)
                unmatched_dets.remove(det_idx)
        
        # 合并匹配结果
        matches = matches_primary + matches_secondary
        
        return matches, unmatched_tracks, unmatched_dets
    
    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None, view_transformer=None):
        """
        获取目标跟踪结果（兼容接口）
        
        Args:
            frames: 视频帧列表
            read_from_stub: 是否从缓存读取
            stub_path: 缓存文件路径
            view_transformer: ViewTransformer对象（如果初始化时未提供，可以在这里提供）
        
        Returns:
            tracks: 追踪结果字典，格式与tracker.py完全兼容
                {
                    "players": [{track_id: {"bbox": [x1,y1,x2,y2]}}, ...],
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
        
        # 检测
        detections = self.detect_frames(frames)
        
        # 初始化输出结构
        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }
        
        # 处理每一帧（按采样频率）
        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # 处理守门员类别
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]
            
            # 分离球和其他对象
            BALL_ID = cls_names_inv['ball']
            ball_detections = detection_supervision[detection_supervision.class_id == BALL_ID]
            all_detections = detection_supervision[detection_supervision.class_id != BALL_ID]
            
            # 对球进行边界框扩展
            if len(ball_detections) > 0:
                ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)
            
            # 初始化该帧的字典
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})
            
            # 处理球（每帧都处理，不追踪）
            if len(ball_detections) > 0:
                for frame_detection in ball_detections:
                    bbox = frame_detection[0].tolist()
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
                    break
            else:
                tracks["ball"][frame_num][1] = {"bbox": [np.nan, np.nan, np.nan, np.nan]}
            
            # 5Hz采样：只处理采样间隔的帧进行追踪
            if frame_num % self.sample_interval != 0:
                # 非采样帧：跳过追踪处理，但已处理球
                continue
            
            # 应用NMS（只在采样帧处理）
            all_detections = all_detections.with_nms(threshold=self.nms_threshold, class_agnostic=True)
            
            # 处理球员和裁判（使用场地坐标追踪）
            if len(all_detections) > 0:
                # 1. 预测所有轨迹的位置
                for track in self.tracks:
                    track.predict()
                
                # 2. 转换检测到场地坐标
                detections_pitch = []
                detections_info = []
                
                for i in range(len(all_detections)):
                    bbox = all_detections.xyxy[i]
                    bbox_list = bbox.tolist()
                    
                    # 提取位置点
                    if cls_names[all_detections.class_id[i]] == "ball":
                        position_pixel = get_center_of_bbox(bbox_list)
                    else:
                        position_pixel = get_foot_position(bbox_list)
                    
                    # 转换为场地坐标
                    position_pitch = view_transformer.transform_point(position_pixel, frame_num)
                    
                    if position_pitch is not None and not np.isnan(position_pitch).any():
                        position_pitch = position_pitch.squeeze().tolist()
                        if isinstance(position_pitch, list) and len(position_pitch) == 2:
                            detections_pitch.append(position_pitch)
                            detections_info.append({
                                'bbox': bbox_list,
                                'position_pixel': list(position_pixel),
                                'class_id': all_detections.class_id[i] - 1,  # Roboflow类别ID调整
                                'confidence': all_detections.confidence[i] if hasattr(all_detections, 'confidence') else 1.0
                            })
                
                # 3. 匹配轨迹和检测
                matches, unmatched_tracks, unmatched_dets = self._match_tracks_detections(
                    self.tracks, detections_pitch, detections_info
                )
                
                # 4. 更新匹配的轨迹
                for track_idx, det_idx in matches:
                    track = self.tracks[track_idx]
                    det_info = detections_info[det_idx]
                    track.update(
                        detections_pitch[det_idx],
                        bbox=det_info['bbox'],
                        position_pixel=det_info['position_pixel'],
                        frame_num=frame_num
                    )
                
                # 5. 标记未匹配的轨迹为丢失
                for track_idx in unmatched_tracks:
                    self.tracks[track_idx].mark_missed()
                
                # 6. 为未匹配的检测创建新轨迹
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
                    self.tracks.append(new_track)
                    self.next_id += 1
                
                # 7. 删除丢失时间过长的轨迹
                self.tracks = [t for t in self.tracks if t.lost_count <= self.track_buffer]
                
                # 8. 存储追踪结果
                for track in self.tracks:
                    if track.lost_count == 0 and track.is_confirmed:
                        # 只存储已确认且未丢失的轨迹
                        bbox = track.bbox
                        if bbox is not None:
                            cls_id = track.class_id
                            track_id = track.track_id
                            
                            # 根据类别ID判断对象类型
                            if cls_id == cls_names_inv['player'] - 1:
                                tracks["players"][frame_num][track_id] = {"bbox": bbox}
                            elif cls_id == cls_names_inv['referee'] - 1:
                                tracks["referees"][frame_num][track_id] = {"bbox": bbox}
            
        # 保存缓存
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)
        
        # 自动保存轨迹缓存（两个版本）
        try:
            self.save_trajectory_cache()
        except Exception as e:
            print(f"警告: 保存轨迹缓存时出错: {e}")
        
        # 返回tracks（兼容现有接口）
        # 轨迹详细信息可通过get_trajectory_info()方法获取
        return tracks
    
    def get_trajectory_info(self):
        """
        获取所有轨迹的详细信息（坐标、速度等）- 以track_id组织
        
        Returns:
            dict: 轨迹信息字典（按track_id组织）
                {
                    track_id: {
                        'class_id': int,
                        'trajectory_history': [
                            {
                                'frame_num': int,
                                'position_pitch': [x, y],  # 米
                                'velocity': [vx, vy],  # 米/采样间隔
                                'speed': float,  # 速度大小
                                'bbox': [x1, y1, x2, y2],
                                'position_pixel': [x, y]
                            },
                            ...
                        ],
                        'is_confirmed': bool,
                        'last_seen_frame': int
                    },
                    ...
                }
        """
        trajectory_info = {}
        for track in self.tracks:
            if len(track.trajectory_history) > 0:
                trajectory_info[track.track_id] = {
                    'class_id': track.class_id,
                    'trajectory_history': track.trajectory_history,
                    'is_confirmed': track.is_confirmed,
                    'last_seen_frame': track.last_seen_frame
                }
        return trajectory_info
    
    def get_trajectory_info_by_frame(self):
        """
        获取所有轨迹的详细信息（坐标、速度等）- 以帧组织
        
        Returns:
            dict: 轨迹信息字典（按帧组织）
                {
                    frame_num: {
                        track_id: {
                            'class_id': int,
                            'position_pitch': [x, y],  # 米
                            'velocity': [vx, vy],  # 米/采样间隔
                            'speed': float,  # 速度大小
                            'bbox': [x1, y1, x2, y2],
                            'position_pixel': [x, y]
                        },
                        ...
                    },
                    ...
                }
        """
        trajectory_by_frame = {}
        for track in self.tracks:
            for point in track.trajectory_history:
                frame_num = point['frame_num']
                if frame_num not in trajectory_by_frame:
                    trajectory_by_frame[frame_num] = {}
                
                trajectory_by_frame[frame_num][track.track_id] = {
                    'class_id': track.class_id,
                    'position_pitch': point['position_pitch'],
                    'velocity': point['velocity'],
                    'speed': point['speed'],
                    'bbox': point['bbox'],
                    'position_pixel': point['position_pixel']
                }
        return trajectory_by_frame
    
    def save_trajectory_cache(self, base_path=None):
        """
        保存轨迹结果缓存（两个版本）
        
        Args:
            base_path: 基础路径（如果不提供，使用默认路径Storage/stubs）
        
        Returns:
            tuple: (track_id版本路径, 帧版本路径)
        """
        import os
        
        if base_path is None:
            # 获取项目根目录
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir)))
            base_path = os.path.join(project_root, 'Storage', 'stubs')
        
        # 确保目录存在
        os.makedirs(base_path, exist_ok=True)
        
        # 版本1：以track_id组织
        trajectory_by_track_id = self.get_trajectory_info()
        track_id_path = os.path.join(base_path, 'trajectory_by_track_id.pkl')
        with open(track_id_path, 'wb') as f:
            pickle.dump(trajectory_by_track_id, f)
        print(f"轨迹缓存（按track_id）已保存: {track_id_path}")
        
        # 版本2：以帧组织
        trajectory_by_frame = self.get_trajectory_info_by_frame()
        frame_path = os.path.join(base_path, 'trajectory_by_frame.pkl')
        with open(frame_path, 'wb') as f:
            pickle.dump(trajectory_by_frame, f)
        print(f"轨迹缓存（按帧）已保存: {frame_path}")
        
        return track_id_path, frame_path
    
    @staticmethod
    def load_trajectory_cache(base_path=None, version='track_id'):
        """
        加载轨迹结果缓存
        
        Args:
            base_path: 基础路径（如果不提供，使用默认路径Storage/stubs）
            version: 版本类型，'track_id' 或 'frame'
        
        Returns:
            dict: 轨迹信息字典
        """
        import os
        
        if base_path is None:
            # 获取项目根目录
            current_file_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file_dir)))
            base_path = os.path.join(project_root, 'Storage', 'stubs')
        
        if version == 'track_id':
            cache_path = os.path.join(base_path, 'trajectory_by_track_id.pkl')
        elif version == 'frame':
            cache_path = os.path.join(base_path, 'trajectory_by_frame.pkl')
        else:
            raise ValueError(f"不支持的版本类型: {version}，请使用 'track_id' 或 'frame'")
        
        if not os.path.exists(cache_path):
            raise FileNotFoundError(f"轨迹缓存文件不存在: {cache_path}")
        
        with open(cache_path, 'rb') as f:
            trajectory_data = pickle.load(f)
        
        print(f"轨迹缓存已加载: {cache_path}")
        return trajectory_data

