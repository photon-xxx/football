import numpy as np
from typing import List, Optional, Tuple, Dict, Any
from scipy.optimize import linear_sum_assignment

# 场地轨迹管理，后续处理追踪器返回的轨迹

class PitchKalmanFilter:
    """
    场地坐标空间的卡尔曼滤波器（4维状态空间）

    状态向量: [x, y, vx, vy]
    - x, y: 场地坐标（cm）
    - vx, vy: 速度（cm/采样间隔）
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
            [0, 0, 0, 1]  # vy = vy
        ], dtype=np.float32)

        # 观测矩阵（只观测位置，不观测速度）
        self.H = np.array([
            [1, 0, 0, 0],  # 观测x
            [0, 1, 0, 0]  # 观测y
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
            position: [x, y] 场地坐标（cm）

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
            measurement: 观测值 [x, y] 场地坐标（cm）

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
            position_pitch: [x, y] 场地坐标（cm）
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

        # reID信息
        self.reid_info = None

        # 号码信息
        self.number_info = None
        # # 轨迹状态
        # self.lost_count = 0
        # self.is_confirmed = False
        # self.hit_streak = 1  # 连续匹配次数

        # # 存储信息（用于输出）
        # self.bbox = None  # 像素坐标 bbox [x1, y1, x2, y2]
        # self.position_pixel = None  # 像素坐标 [px, py]
        # self.team = None  # 球队ID (1 或 2)
        # self.team_color = None  # 球队颜色

        # # 轨迹历史缓存（重要输出结果）
        # # 格式: [{frame_num, position_pitch, velocity, bbox, position_pixel}, ...]
        # self.trajectory_history = []

    def predict(self):
        """预测下一帧位置"""
        self.mean, self.covariance = self.kalman_filter.predict(self.mean, self.covariance)
        # 更新预测位置
        self.position_pitch = self.mean[:2]

    def update(self, position_pitch, bbox=None, position_pixel=None, frame_num=None, team=None, team_color=None):
        """
        使用新检测更新轨迹

        Args:
            position_pitch: [x, y] 场地坐标（cm）
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
            'position_pitch': self.position_pitch.copy(),  # [x, y] cm
            'velocity': velocity.copy(),  # [vx, vy] cm/采样间隔
            'speed': np.sqrt(velocity[0] ** 2 + velocity[1] ** 2),  # 速度大小
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
        self.hit_streak = 0


class Tracker:
    """
    多目标追踪器

    负责管理 PitchTrack 列表，处理检测结果的关联和更新。
    支持为后续 SAM2/SAM3 提供追踪信息。
    """

    def __init__(self,
                 max_lost_frames: int = 30,
                 match_distance_threshold: float = 200.0,  # cm, 假设 2米作为匹配阈值
                 dt: float = 1.0):
        """
        Args:
            max_lost_frames: 轨迹最长丢失帧数，超过后删除
            match_distance_threshold: 匹配阈值 (cm)
            dt: 时间步长
        """
        self.tracks: List[PitchTrack] = []
        self.max_lost_frames = max_lost_frames
        self.match_distance_threshold = match_distance_threshold
        self.next_track_id = 0
        self.dt = dt

    def update(self,
               detections_pitch: np.ndarray,
               detections_bbox: Optional[np.ndarray] = None,
               detections_cls: Optional[np.ndarray] = None,
               frame_num: int = 0,
               camera_id: Optional[int] = None):
        """
        更新追踪器状态

        Args:
            detections_pitch: (N, 2) 检测到的物理坐标 [x, y]
            detections_bbox: (N, 4) 检测到的像素坐标 bbox [x1, y1, x2, y2]
            detections_cls: (N,) 类别ID
            frame_num: 当前帧号
            camera_id: 来源相机ID (可选，用于记录)
        """

        # 1. 预测所有现有轨迹
        for track in self.tracks:
            track.predict()

        # 2. 关联检测与轨迹
        if len(detections_pitch) == 0:
            # 如果没有检测，所有轨迹标记为丢失
            for track in self.tracks:
                track.mark_missed()
        else:
            # 计算距离矩阵
            track_indices = []
            track_positions = []

            for i, track in enumerate(self.tracks):
                track_indices.append(i)
                track_positions.append(track.position_pitch)  # use predicted position

            if len(self.tracks) > 0:
                track_positions = np.array(track_positions)

                # Cost matrix: (num_tracks, num_detections)
                # Euclidean distance
                dists = np.linalg.norm(track_positions[:, None, :] - detections_pitch[None, :, :], axis=2)

                # 使用匈牙利算法进行匹配
                row_inds, col_inds = linear_sum_assignment(dists)

                matched_track_indices = set()
                matched_detection_indices = set()

                # 处理匹配结果
                for r, c in zip(row_inds, col_inds):
                    dist = dists[r, c]
                    if dist < self.match_distance_threshold:
                        track_idx = track_indices[r]

                        # 准备更新数据
                        bbox = detections_bbox[c] if detections_bbox is not None else None

                        # 更新匹配的轨迹
                        self.tracks[track_idx].update(
                            position_pitch=detections_pitch[c],
                            bbox=bbox,
                            frame_num=frame_num
                        )
                        matched_track_indices.add(track_idx)
                        matched_detection_indices.add(c)

                # 标记未匹配的轨迹
                for i in range(len(self.tracks)):
                    if i not in matched_track_indices:
                        self.tracks[i].mark_missed()

                # 为未匹配的检测创建新轨迹
                for i in range(len(detections_pitch)):
                    if i not in matched_detection_indices:
                        self._create_track(
                            position_pitch=detections_pitch[i],
                            bbox=detections_bbox[i] if detections_bbox is not None else None,
                            cls_id=detections_cls[i] if detections_cls is not None else 0,
                            frame_num=frame_num
                        )
            else:
                # 如果没有现有轨迹，所有检测都创建新轨迹
                for i in range(len(detections_pitch)):
                    self._create_track(
                        position_pitch=detections_pitch[i],
                        bbox=detections_bbox[i] if detections_bbox is not None else None,
                        cls_id=detections_cls[i] if detections_cls is not None else 0,
                        frame_num=frame_num
                    )

        # 3. 清理丢失过久的轨迹
        self.tracks = [t for t in self.tracks if t.lost_count <= self.max_lost_frames]

    def _create_track(self, position_pitch, bbox, cls_id, frame_num):
        """创建新轨迹"""
        new_track = PitchTrack(
            position_pitch=position_pitch,
            class_id=cls_id,
            frame_num=frame_num,
            track_id=self.next_track_id,
            dt=self.dt
        )
        if bbox is not None:
            new_track.bbox = bbox

        self.tracks.append(new_track)
        self.next_track_id += 1

    def get_active_tracks(self) -> List[PitchTrack]:
        """获取当前活跃（确认且未丢失）的轨迹"""
        return [t for t in self.tracks if t.is_confirmed and t.lost_count == 0]

    def get_tracks_for_sam(self) -> List[Dict[str, Any]]:
        """
        获取用于 SAM (Segment Anything Model) 的提示信息

        Returns:
            List of dicts, each containing:
            - track_id: int
            - bbox: [x1, y1, x2, y2] (pixel)
            - point: [x, y] (pixel center of bbox)
        """
        sam_prompts = []
        for track in self.get_active_tracks():
            if track.bbox is not None:
                # 计算中心点作为 point prompt
                x1, y1, x2, y2 = track.bbox
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                sam_prompts.append({
                    "track_id": track.track_id,
                    "bbox": track.bbox,
                    "point": np.array([center_x, center_y]),
                    "label": 1  # Foreground
                })
        return sam_prompts
