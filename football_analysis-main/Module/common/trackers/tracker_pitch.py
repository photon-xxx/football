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
from scipy.optimize import linear_sum_assignment

sys.path.append('../../../')
from Module.common.utils import get_foot_position


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
    
    
    def detect_frames(self, frames, batch_size=12, conf_threshold=0.3):
        """
        使用YOLO进行检测（独立方法）
        
        Args:
            frames: 视频帧列表
            batch_size: 批次大小
            conf_threshold: 检测置信度阈值
        
        Returns:
            list: 每帧的检测结果列表，每个元素是ultralytics Results对象
        """
        detections = []
        total_detections = 0
        player_detections = 0
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            results = self.model.predict(batch_frames, conf=conf_threshold, verbose=False)
            detections.extend(results)
            # 统计检测数量
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    total_detections += len(result.boxes)
                    # 统计球员检测
                    if hasattr(result, 'names'):
                        cls_names = result.names
                        cls_names_inv = {v: k for k, v in cls_names.items()}
                        if 'player' in cls_names_inv:
                            player_id = cls_names_inv['player']
                            player_boxes = result.boxes[result.boxes.cls == player_id]
                            player_detections += len(player_boxes)
        return detections
    
    def _process_detections_to_tracking_input(self, detections, view_transformer, detections_with_teams=None):
        """
        将YOLO检测结果转换为追踪器输入格式
        
        Args:
            detections: YOLO检测结果列表
            view_transformer: ViewTransformer对象
            detections_with_teams: 球队分配结果（可选）{frame_num: {detection_idx: (team, team_color)}}
        
        Returns:
            list: 每帧的追踪输入数据
                [
                    {
                        'frame_num': int,
                        'detections_pitch': [[x, y], ...],  # 场地坐标
                        'detections_info': [{bbox, position_pixel, class_id, confidence, team, team_color}, ...]
                    },
                    ...
                ]
        """
        tracking_inputs = []
        
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
            all_detections = detection_supervision[detection_supervision.class_id != BALL_ID]
            
            # 应用NMS（如果使用采样，只在采样帧处理；否则每帧都处理）
            if not self.use_sampling or frame_num % self.sample_interval == 0:
                all_detections = all_detections.with_nms(threshold=self.nms_threshold, class_agnostic=True)
            
            # 转换检测到场地坐标
            detections_pitch = []
            detections_info = []
            
            # 获取当前帧的球队分配结果
            frame_teams = detections_with_teams.get(frame_num, {}) if detections_with_teams else {}
            
            for i in range(len(all_detections)):
                bbox = all_detections.xyxy[i]
                bbox_list = bbox.tolist()
                
                # 提取位置点
                position_pixel = get_foot_position(bbox_list)
                
                # 转换为场地坐标
                position_pitch = view_transformer.transform_point(position_pixel, frame_num)
                
                if position_pitch is not None and not np.isnan(position_pitch).any():
                    position_pitch = position_pitch.squeeze().tolist()
                    if isinstance(position_pitch, list) and len(position_pitch) == 2:
                        det_info = {
                            'bbox': bbox_list,
                            'position_pixel': list(position_pixel),
                            'class_id': all_detections.class_id[i] - 1,  # Roboflow类别ID调整
                            'confidence': all_detections.confidence[i] if hasattr(all_detections, 'confidence') else 1.0
                        }
                        
                        # 如果提供了球队信息，添加到检测信息中
                        if i in frame_teams:
                            team, team_color = frame_teams[i]
                            det_info['team'] = team
                            det_info['team_color'] = team_color
                        
                        detections_pitch.append(position_pitch)
                        detections_info.append(det_info)
            
            tracking_inputs.append({
                'frame_num': frame_num,
                'detections_pitch': detections_pitch,
                'detections_info': detections_info,
                'cls_names': cls_names,
                'cls_names_inv': cls_names_inv
            })
        
        return tracking_inputs
    
    def track_detections(self, tracking_inputs, frames=None):
        """
        基于检测结果进行追踪（独立方法）
        
        Args:
            tracking_inputs: 追踪输入数据（由_process_detections_to_tracking_input生成）
            frames: 视频帧列表（可选，用于处理球检测）
        
        Returns:
            dict: 追踪结果，格式与get_object_tracks兼容
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
        
        # 处理每一帧的追踪（按采样频率）
        for tracking_input in tracking_inputs:
            frame_num = tracking_input['frame_num']
            detections_pitch = tracking_input['detections_pitch']
            detections_info = tracking_input['detections_info']
            cls_names = tracking_input['cls_names']
            cls_names_inv = tracking_input['cls_names_inv']
            
            # 初始化该帧的字典
            tracks["players"].append({})
            tracks["referees"].append({})
            
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
                
                # 2. 匹配轨迹和检测
                matches, unmatched_tracks, unmatched_dets = self._match_tracks_detections(
                    self.tracks, detections_pitch, detections_info
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
                    self.tracks.append(new_track)
                    self.next_id += 1
                
                # 6. 删除丢失时间过长的轨迹
                self.tracks = [t for t in self.tracks if t.lost_count <= self.track_buffer]
            
            # 7. 存储追踪结果（对所有帧，包括非采样帧）
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
        
        # 使用匈牙利算法进行最优匹配（更稳定的匹配方式）
        if len(tracks) > 0 and len(detections_pitch) > 0:
            # 创建成本矩阵（距离），对于超过阈值的距离设置为无穷大
            cost_matrix = distance_matrix.copy()
            cost_matrix[cost_matrix > self.distance_threshold_primary] = np.inf
            
            # 使用匈牙利算法找到最优匹配
            track_indices, det_indices = linear_sum_assignment(cost_matrix)
            
            # 筛选出有效的匹配（距离小于阈值）
            matches_primary = []
            for track_idx, det_idx in zip(track_indices, det_indices):
                if cost_matrix[track_idx, det_idx] < np.inf:
                    matches_primary.append((track_idx, det_idx))
            
            # 找出未匹配的轨迹和检测
            matched_tracks = set(track_idx for track_idx, _ in matches_primary)
            matched_dets = set(det_idx for _, det_idx in matches_primary)
            unmatched_tracks = [i for i in range(len(tracks)) if i not in matched_tracks]
            unmatched_dets = [i for i in range(len(detections_pitch)) if i not in matched_dets]
        else:
            matches_primary = []
            unmatched_tracks = list(range(len(tracks)))
            unmatched_dets = list(range(len(detections_pitch)))
        
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

