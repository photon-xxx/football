"""
BoTSORT (Boosting Online and Real-time Simple Online and Realtime Tracking) 跟踪器

这是一个基于深度学习的多目标跟踪算法，结合了：
1. 卡尔曼滤波进行运动预测
2. 深度特征进行外观匹配
3. 全局运动补偿处理相机运动
4. 匈牙利算法进行最优分配

主要特点：
- 支持ReID特征提取和匹配
- 处理相机运动补偿
- 多级关联策略
- 实时性能优化

作者: 基于Deep OC-SORT改进
"""

import numpy as np
from collections import deque

from . import matching
from .gmc import GMC
from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter

# from fast_reid.fast_reid_interfece import FastReIDInterface

from deep_oc_sort.reid_multibackend import ReIDDetectMultiBackend
from ultralytics.yolo.utils.ops import xyxy2xywh, xywh2xyxy


class STrack(BaseTrack):
    """
    单个目标跟踪轨迹类
    
    继承自BaseTrack，实现了单个目标的完整生命周期管理，包括：
    - 卡尔曼滤波状态预测和更新
    - 深度特征提取和匹配
    - 类别历史记录
    - 轨迹状态管理
    
    Attributes:
        shared_kalman: 共享的卡尔曼滤波器实例
        _tlwh: 边界框坐标 (top-left-width-height)
        kalman_filter: 卡尔曼滤波器
        mean: 状态均值向量
        covariance: 状态协方差矩阵
        is_activated: 是否已激活
        cls: 当前类别
        cls_hist: 类别历史记录 [(类别ID, 频率)]
        score: 置信度分数
        tracklet_len: 轨迹长度
        smooth_feat: 平滑后的特征向量
        curr_feat: 当前帧特征向量
        features: 特征历史队列
        alpha: 特征平滑系数
        tracklab_id: 轨迹标签ID
    """
    shared_kalman = KalmanFilter()

    def __init__(self, tlwh, score, cls, feat=None, feat_history=50, tracklab_id=None):
        """
        初始化单个目标轨迹
        
        Args:
            tlwh: 边界框坐标 [x, y, w, h]
            score: 检测置信度分数
            cls: 目标类别
            feat: 深度特征向量 (可选)
            feat_history: 特征历史队列最大长度
            tracklab_id: 轨迹标签ID (可选)
        """
        # 等待激活状态
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        # 类别管理
        self.cls = -1
        self.cls_hist = []  # 存储 (类别ID, 频率) 的列表
        self.update_cls(cls, score)

        # 轨迹信息
        self.score = score
        self.tracklet_len = 0

        # 特征管理
        self.smooth_feat = None
        self.curr_feat = None
        if feat is not None:
            self.update_features(feat)
        self.features = deque([], maxlen=feat_history)  # 特征历史队列
        self.alpha = 0.9  # 特征平滑系数

        self.tracklab_id = tracklab_id

    def update_features(self, feat):
        """
        更新目标的外观特征
        
        使用指数移动平均对特征进行平滑处理，提高特征稳定性
        
        Args:
            feat: 新的特征向量
        """
        # L2归一化特征向量
        feat /= np.linalg.norm(feat)
        self.curr_feat = feat
        
        # 指数移动平均更新平滑特征
        if self.smooth_feat is None:
            self.smooth_feat = feat
        else:
            self.smooth_feat = self.alpha * self.smooth_feat + (1 - self.alpha) * feat
        
        # 添加到特征历史队列
        self.features.append(feat)
        
        # 重新归一化平滑特征
        self.smooth_feat /= np.linalg.norm(self.smooth_feat)

    def update_cls(self, cls, score):
        """
        更新目标类别信息
        
        维护类别历史记录，选择出现频率最高的类别作为当前类别
        
        Args:
            cls: 新的类别ID
            score: 类别置信度分数
        """
        if len(self.cls_hist) > 0:
            max_freq = 0
            found = False
            
            # 查找是否已存在该类别
            for c in self.cls_hist:
                if cls == c[0]:
                    c[1] += score  # 累加分数
                    found = True

                # 更新最频繁的类别
                if c[1] > max_freq:
                    max_freq = c[1]
                    self.cls = c[0]
            
            # 如果未找到，添加新类别
            if not found:
                self.cls_hist.append([cls, score])
                self.cls = cls
        else:
            # 首次添加类别
            self.cls_hist.append([cls, score])
            self.cls = cls

    def predict(self):
        """
        使用卡尔曼滤波预测下一帧的状态
        
        对于非跟踪状态的目标，将速度分量置零
        """
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0  # 速度x置零
            mean_state[7] = 0  # 速度y置零

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        """
        批量预测多个轨迹的状态
        
        向量化的卡尔曼滤波预测，提高计算效率
        
        Args:
            stracks: 轨迹列表
        """
        if len(stracks) > 0:
            # 提取所有轨迹的状态
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            
            # 对非跟踪状态的目标置零速度
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0  # 速度x置零
                    multi_mean[i][7] = 0  # 速度y置零
            
            # 批量预测
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            
            # 更新轨迹状态
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    @staticmethod
    def multi_gmc(stracks, H=np.eye(2, 3)):
        """
        批量应用全局运动补偿 (Global Motion Compensation)
        
        使用仿射变换矩阵补偿相机运动对轨迹的影响
        
        Args:
            stracks: 轨迹列表
            H: 2x3仿射变换矩阵，用于补偿相机运动
        """
        if len(stracks) > 0:
            # 提取所有轨迹的状态
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])

            # 提取旋转和平移分量
            R = H[:2, :2]  # 旋转矩阵
            R8x8 = np.kron(np.eye(4, dtype=float), R)  # 扩展到8维状态空间
            t = H[:2, 2]  # 平移向量

            # 对每个轨迹应用变换
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                # 变换状态均值
                mean = R8x8.dot(mean)
                mean[:2] += t  # 添加平移
                
                # 变换协方差矩阵
                cov = R8x8.dot(cov).dot(R8x8.transpose())

                # 更新轨迹状态
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id):
        """
        激活新的轨迹
        
        初始化卡尔曼滤波器状态，分配轨迹ID，开始跟踪
        
        Args:
            kalman_filter: 卡尔曼滤波器实例
            frame_id: 当前帧ID
        """
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()

        # 初始化卡尔曼滤波器状态
        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        # 重置轨迹信息
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False):
        """
        重新激活丢失的轨迹
        
        当丢失的轨迹重新匹配到检测时调用
        
        Args:
            new_track: 匹配的新检测
            frame_id: 当前帧ID
            new_id: 是否分配新的轨迹ID
        """
        # 更新卡尔曼滤波器状态
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        
        # 更新特征
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)
        
        # 重置轨迹信息
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        
        # 可选分配新ID
        if new_id:
            self.track_id = self.next_id()
        
        # 更新分数和类别
        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)
        self.tracklab_id = new_track.tracklab_id

    def update(self, new_track, frame_id):
        """
        更新匹配的轨迹
        
        当轨迹成功匹配到检测时调用，更新所有状态信息
        
        Args:
            new_track: 匹配的新检测
            frame_id: 当前帧ID
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        # 更新卡尔曼滤波器状态
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        # 更新特征
        if new_track.curr_feat is not None:
            self.update_features(new_track.curr_feat)

        # 更新轨迹状态
        self.state = TrackState.Tracked
        self.is_activated = True

        # 更新分数和类别
        self.score = new_track.score
        self.update_cls(new_track.cls, new_track.score)
        self.tracklab_id = new_track.tracklab_id

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        ret = np.asarray(tlbr).copy()
        ret[2:] -= ret[:2]
        return ret

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    """
    BoTSORT多目标跟踪器主类
    
    实现了基于深度学习的多目标跟踪算法，主要特点：
    1. 使用ReID特征进行外观匹配
    2. 卡尔曼滤波进行运动预测
    3. 全局运动补偿处理相机运动
    4. 多级关联策略提高跟踪稳定性
    
    Attributes:
        tracked_stracks: 当前跟踪的轨迹列表
        lost_stracks: 丢失的轨迹列表
        removed_stracks: 已移除的轨迹列表
        frame_id: 当前帧ID
        kalman_filter: 卡尔曼滤波器
        model: ReID特征提取模型
        gmc: 全局运动补偿模块
    """
    
    def __init__(self, 
                model_weights,
                device,
                fp16,
                track_high_thresh:float = 0.45,
                new_track_thresh:float = 0.6,
                track_buffer:int = 30,
                match_thresh:float = 0.8,
                proximity_thresh:float = 0.5,
                appearance_thresh:float = 0.25,
                cmc_method:str = 'sparseOptFlow',
                frame_rate=30,
                lambda_=0.985
                ):
        """
        初始化BoTSORT跟踪器
        
        Args:
            model_weights: ReID模型权重路径
            device: 计算设备 ('cpu' 或 'cuda')
            fp16: 是否使用半精度推理
            track_high_thresh: 高置信度检测阈值
            new_track_thresh: 新轨迹创建阈值
            track_buffer: 轨迹缓冲帧数
            match_thresh: 匹配阈值
            proximity_thresh: 邻近阈值
            appearance_thresh: 外观阈值
            cmc_method: 相机运动补偿方法
            frame_rate: 视频帧率
            lambda_: 运动-外观融合权重
        """
        # 轨迹管理
        self.tracked_stracks = []  # 当前跟踪的轨迹列表
        self.lost_stracks = []     # 丢失的轨迹列表
        self.removed_stracks = []  # 已移除的轨迹列表
        BaseTrack.clear_count()    # 重置轨迹ID计数器

        # 帧管理
        self.frame_id = 0

        # 阈值参数
        self.lambda_ = lambda_                    # 运动-外观融合权重
        self.track_high_thresh = track_high_thresh  # 高置信度阈值
        self.new_track_thresh = new_track_thresh    # 新轨迹阈值

        # 缓冲管理
        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        
        # 卡尔曼滤波器
        self.kalman_filter = KalmanFilter()

        # ReID模块参数
        self.proximity_thresh = proximity_thresh    # 邻近阈值
        self.appearance_thresh = appearance_thresh  # 外观阈值
        self.match_thresh = match_thresh            # 匹配阈值

        # 初始化ReID模型
        self.model = ReIDDetectMultiBackend(weights=model_weights, device=device, fp16=fp16)

        # 全局运动补偿
        self.gmc = GMC(method=cmc_method, verbose=[None,False])

    def update(self, output_results, img):
        """
        更新跟踪器状态，处理新一帧的检测结果
        
        这是BoTSORT的核心方法，实现了完整的多目标跟踪流程：
        1. 检测结果预处理和特征提取
        2. 轨迹预测和全局运动补偿
        3. 多级关联策略
        4. 轨迹状态更新
        
        Args:
            output_results: 检测结果数组 [N, 7] (x1,y1,x2,y2,conf,cls,tracklab_id)
            img: 当前帧图像
            
        Returns:
            outputs: 跟踪结果列表，每个元素包含 [x1,y1,x2,y2,track_id,cls,score,tracklab_id]
        """
        self.frame_id += 1
        activated_starcks = []  # 新激活的轨迹
        refind_stracks = []     # 重新找到的轨迹
        lost_stracks = []       # 丢失的轨迹
        removed_stracks = []    # 移除的轨迹
        
        # 解析检测结果
        xyxys = output_results[:, 0:4]      # 边界框坐标
        xywh = xyxy2xywh(xyxys.numpy())     # 转换为中心点格式
        confs = output_results[:, 4]         # 置信度
        clss = output_results[:, 5]          # 类别
        tracklab_ids = output_results[:, 6]  # 轨迹标签ID
        
        # 转换为numpy数组
        classes = clss.numpy()
        xyxys = xyxys.numpy()
        confs = confs.numpy()
        tracklab_ids = tracklab_ids.numpy()

        # 根据置信度分离检测结果
        remain_inds = confs > self.track_high_thresh  # 高置信度检测
        inds_low = confs > 0.1                        # 低置信度检测
        inds_high = confs < self.track_high_thresh    # 中等置信度检测

        inds_second = np.logical_and(inds_low, inds_high)  # 中等置信度索引
        
        # 分离不同置信度的检测
        dets_second = xywh[inds_second]      # 中等置信度检测
        dets = xywh[remain_inds]             # 高置信度检测
        
        scores_keep = confs[remain_inds]     # 高置信度分数
        scores_second = confs[inds_second]   # 中等置信度分数
        
        classes_keep = classes[remain_inds]  # 高置信度类别
        clss_second = classes[inds_second]   # 中等置信度类别

        tracklab_ids_keep = tracklab_ids[remain_inds]    # 高置信度轨迹ID
        tracklab_ids_second = tracklab_ids[inds_second]  # 中等置信度轨迹ID

        self.height, self.width = img.shape[:2]

        # 提取ReID特征
        features_keep = self._get_features(dets, img)


        if len(dets) > 0:
            '''Detections'''
            
            detections = [STrack(xyxy, s, c, f.cpu().numpy(), tracklab_id=id) for
                              (xyxy, s, c, f, id) in zip(dets, scores_keep, classes_keep, features_keep, tracklab_ids_keep)]
        else:
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        warp = self.gmc.apply(img, dets)
        STrack.multi_gmc(strack_pool, warp)
        STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes
        raw_emb_dists = matching.embedding_distance(strack_pool, detections)
        dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections, only_position=False, lambda_=self.lambda_)

        # ious_dists = matching.iou_distance(strack_pool, detections)
        # ious_dists_mask = (ious_dists > self.proximity_thresh)

        # ious_dists = matching.fuse_score(ious_dists, detections)

        # emb_dists = matching.embedding_distance(strack_pool, detections) / 2.0
        # raw_emb_dists = emb_dists.copy()
        # emb_dists[emb_dists > self.appearance_thresh] = 1.0
        # emb_dists[ious_dists_mask] = 1.0
        # dists = np.minimum(ious_dists, emb_dists)

            # Popular ReID method (JDE / FairMOT)
            # raw_emb_dists = matching.embedding_distance(strack_pool, detections)
            # dists = matching.fuse_motion(self.kalman_filter, raw_emb_dists, strack_pool, detections)
            # emb_dists = dists

            # IoU making ReID
            # dists = matching.embedding_distance(strack_pool, detections)
            # dists[ious_dists_mask] = 1.0
    
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        # if len(scores):
        #     inds_high = scores < self.track_high_thresh
        #     inds_low = scores > self.track_low_thresh
        #     inds_second = np.logical_and(inds_low, inds_high)
        #     dets_second = bboxes[inds_second]
        #     scores_second = scores[inds_second]
        #     classes_second = classes[inds_second]
        # else:
        #     dets_second = []
        #     scores_second = []
        #     classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(STrack.tlbr_to_tlwh(tlbr), s, c, tracklab_id=id) for
                (tlbr, s, c, id) in zip(dets_second, scores_second, clss_second, tracklab_ids_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        ious_dists = matching.iou_distance(unconfirmed, detections)
        ious_dists_mask = (ious_dists > self.proximity_thresh)
        
        ious_dists = matching.fuse_score(ious_dists, detections)
    
        emb_dists = matching.embedding_distance(unconfirmed, detections) / 2.0
        raw_emb_dists = emb_dists.copy()
        emb_dists[emb_dists > self.appearance_thresh] = 1.0
        emb_dists[ious_dists_mask] = 1.0
        dists = np.minimum(ious_dists, emb_dists)
    
        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue

            track.activate(self.kalman_filter, self.frame_id)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)

        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        outputs = []
        for t in output_stracks:
            output= []
            tlwh = t.tlwh
            tid = t.track_id
            tlwh = np.expand_dims(tlwh, axis=0)
            xyxy = xywh2xyxy(tlwh)
            xyxy = np.squeeze(xyxy, axis=0)
            output.extend(xyxy)
            output.append(tid)
            output.append(t.cls)
            output.append(t.score)
            output.append(t.tracklab_id)
            outputs.append(output)

        return outputs

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.model(im_crops)
        else:
            features = np.array([])
        return features

def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb
