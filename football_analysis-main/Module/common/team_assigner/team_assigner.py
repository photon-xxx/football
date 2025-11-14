"""
球队分配器 - 使用基于SigLIP深度视觉特征的分类算法

这个类使用预训练的SigLIP视觉模型来提取球员的深度特征，然后通过UMAP降维和KMeans聚类
来区分两支球队。相比基于颜色的方法，这种方法更鲁棒，能捕捉纹理、形状等视觉模式。
"""

import numpy as np
from .team_classifier import TeamClassifier


class TeamAssigner:
    """
    球队分配器
    
    使用基于SigLIP深度视觉特征的分类算法来区分两支球队。
    保持与旧版本相同的接口以兼容现有代码。
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32, local_model_path: str = None, verbose: bool = False):
        """
        初始化球队分配器
        
        Args:
            device: 运行设备 ('cpu' 或 'cuda')，建议使用'cuda'以加速处理
            batch_size: 批量处理图像的大小，默认32
            local_model_path: 本地模型路径（如果模型已手动下载），如果为None则使用默认配置
            verbose: 是否显示详细输出（默认False）
        """
        self.device = device
        self.batch_size = batch_size
        self.classifier = TeamClassifier(device=device, batch_size=batch_size, local_model_path=local_model_path, verbose=verbose)
        self.team_colors = {}  # 保持兼容性，存储队伍的代表颜色（用于可视化）
        self.player_team_dict = {}  # 缓存球员ID到队伍的映射
        self.is_fitted = False

    def _extract_player_crop(self, frame, bbox):
        """
        从帧中提取球员裁剪图像
        
        Args:
            frame: 视频帧 (BGR格式)
            bbox: 边界框 [x1, y1, x2, y2]
        
        Returns:
            球员裁剪图像 (BGR格式)
        """
        x1, y1, x2, y2 = map(int, bbox)
        # 提取完整球员区域（不再只取上半部分，因为深度模型能理解完整图像）
        crop = frame[y1:y2, x1:x2]
        return crop

    def assign_team_color(self, frame, player_detections):
        """
        在第一帧上初始化队伍分类器
        
        Args:
            frame: 第一帧视频帧
            player_detections: 第一帧的球员检测结果字典 {player_id: {'bbox': [x1,y1,x2,y2]}, ...}
        """
        if len(player_detections) == 0:
            print("警告: 第一帧没有检测到球员，无法初始化队伍分类器")
            return
        
        # 提取所有球员的裁剪图像
        player_crops = []
        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            crop = self._extract_player_crop(frame, bbox)
            if crop.size > 0:  # 确保裁剪不为空
                player_crops.append(crop)
        
        if len(player_crops) < 2:
            print(f"警告: 检测到的球员数量不足 ({len(player_crops)})，无法进行队伍分类")
            return
        
        # 训练分类器
        print(f"正在使用 {len(player_crops)} 个球员初始化队伍分类器...")
        self.classifier.fit(player_crops)
        self.is_fitted = True
        
        # 获取预测结果来确定队伍
        predictions = self.classifier.predict(player_crops)
        
        # 为了保持兼容性，计算每支队伍的平均颜色（用于可视化）
        # 这里我们计算每支队伍的平均RGB颜色作为代表颜色
        team_1_colors = []
        team_2_colors = []
        
        for i, (_, player_detection) in enumerate(player_detections.items()):
            if i < len(predictions):
                bbox = player_detection["bbox"]
                crop = self._extract_player_crop(frame, bbox)
                if crop.size > 0:
                    # 计算裁剪图像的平均颜色
                    avg_color = np.mean(crop.reshape(-1, 3), axis=0)
                    if predictions[i] == 0:
                        team_1_colors.append(avg_color)
                    else:
                        team_2_colors.append(avg_color)
        
        # 存储队伍代表颜色（用于可视化）
        if team_1_colors:
            self.team_colors[1] = np.mean(team_1_colors, axis=0).tolist()
        else:
            self.team_colors[1] = [255, 0, 0]  # 默认红色
        
        if team_2_colors:
            self.team_colors[2] = np.mean(team_2_colors, axis=0).tolist()
        else:
            self.team_colors[2] = [0, 0, 255]  # 默认蓝色
        
        print(f"队伍分类器初始化完成")
        print(f"队伍1代表颜色 (RGB): {self.team_colors[1]}")
        print(f"队伍2代表颜色 (RGB): {self.team_colors[2]}")

    def get_player_team(self, frame, player_bbox, player_id):
        """
        获取球员所属队伍
        
        Args:
            frame: 视频帧
            player_bbox: 球员边界框 [x1, y1, x2, y2]
            player_id: 球员ID
        
        Returns:
            队伍ID (1 或 2)
        """
        if not self.is_fitted:
            raise ValueError("队伍分类器尚未初始化，请先调用 assign_team_color() 方法")
        
        # 如果已经缓存过，直接返回
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        
        # 提取球员裁剪图像
        crop = self._extract_player_crop(frame, player_bbox)
        
        if crop.size == 0:
            # 如果裁剪为空，返回默认队伍1
            self.player_team_dict[player_id] = 1
            return 1
        
        # 使用分类器预测队伍
        prediction = self.classifier.predict([crop])[0]
        
        # 将预测结果转换为队伍ID (0 -> 1, 1 -> 2)
        team_id = int(prediction) + 1
        
        # 特殊处理（保持兼容性）
        if player_id == 91:
            team_id = 1
        
        # 缓存结果
        self.player_team_dict[player_id] = team_id
        
        return team_id
    
    def assign_teams_from_tracks(self, video_frames, tracks, training_frames=20):
        """
        从追踪结果中分配球队（用于main.py）
        
        算法流程：
        1. 从多个采样帧收集球员crops（用于训练）
        2. 统一训练分类器
        3. 计算队伍代表颜色
        4. 对每帧进行批量预测并分配队伍
        
        Args:
            video_frames: 视频帧列表
            tracks: 追踪结果字典 {"players": [{track_id: {"bbox": [...]}}, ...], ...}
            training_frames: 用于训练分类器的采样帧数（默认20帧）
        
        Returns:
            None (直接修改tracks字典，添加'team'和'team_color'字段)
        """
        # 第一步：从多个帧收集球员crops（用于训练分类器）
        total_frames = len(tracks['players'])
        
        # 先找到所有有球员的帧（因为tracker_pitch使用采样，很多帧可能是空的）
        frames_with_players = []
        for frame_idx in range(total_frames):
            player_track = tracks['players'][frame_idx]
            if len(player_track) > 0:  # 该帧有球员
                frames_with_players.append(frame_idx)
        
        if len(frames_with_players) == 0:
            raise ValueError("未找到任何有球员的帧，无法进行队伍分类")
        
        print(f"找到 {len(frames_with_players)} 个有球员的帧（总帧数: {total_frames}）")
        
        # 从有球员的帧中采样（最多使用training_frames帧）
        if len(frames_with_players) <= training_frames:
            sampled_frames = frames_with_players
            print(f"第一步：收集所有 {len(sampled_frames)} 个有球员的帧的球员裁剪图像（用于训练分类器）...")
        else:
            stride = max(1, len(frames_with_players) // training_frames)
            sampled_frames = frames_with_players[::stride][:training_frames]
            print(f"第一步：从 {len(frames_with_players)} 个有球员的帧中采样 {len(sampled_frames)} 帧（stride={stride}）收集球员裁剪图像（用于训练分类器）...")
        
        all_player_crops = []
        
        for frame_idx in sampled_frames:
            player_track = tracks['players'][frame_idx]
            for player_id, track in player_track.items():
                bbox = track['bbox']
                if not np.isnan(bbox).any():
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = video_frames[frame_idx][y1:y2, x1:x2]
                    if crop.size > 0:
                        all_player_crops.append(crop)
        
        if len(all_player_crops) < 2:
            raise ValueError(f"收集到的球员数量不足 ({len(all_player_crops)})，无法进行队伍分类。请检查追踪结果是否正确。")
        
        print(f"共收集到 {len(all_player_crops)} 个球员裁剪图像")
        
        # 第二步：统一训练分类器（使用所有收集到的crops）
        print("第二步：使用所有球员裁剪图像训练分类器...")
        self.classifier.fit(all_player_crops)
        self.is_fitted = True
        print("分类器训练完成")
        
        # 第三步：计算队伍代表颜色
        print("第三步：计算队伍代表颜色...")
        all_predictions = self.classifier.predict(all_player_crops)
        team_1_colors = []
        team_2_colors = []
        
        for crop, prediction in zip(all_player_crops, all_predictions):
            avg_color = np.mean(crop.reshape(-1, 3), axis=0)
            if prediction == 0:
                team_1_colors.append(avg_color)
            else:
                team_2_colors.append(avg_color)
        
        if team_1_colors:
            self.team_colors[1] = np.mean(team_1_colors, axis=0).tolist()
        else:
            self.team_colors[1] = [255, 0, 0]  # 默认红色
        
        if team_2_colors:
            self.team_colors[2] = np.mean(team_2_colors, axis=0).tolist()
        else:
            self.team_colors[2] = [0, 0, 255]  # 默认蓝色
        
        print(f"队伍1代表颜色 (RGB): {self.team_colors[1]}")
        print(f"队伍2代表颜色 (RGB): {self.team_colors[2]}")
        
        # 第四步：对每帧进行队伍分配
        print("第四步：对每帧进行队伍分配...")
        for frame_num, player_track in enumerate(tracks['players']):
            # 批量提取当前帧的所有crops
            current_frame_crops = []
            player_ids = []
            
            for player_id, track in player_track.items():
                bbox = track['bbox']
                if not np.isnan(bbox).any():
                    x1, y1, x2, y2 = map(int, bbox)
                    crop = video_frames[frame_num][y1:y2, x1:x2]
                    if crop.size > 0:
                        current_frame_crops.append(crop)
                        player_ids.append(player_id)
            
            # 批量预测当前帧的球员队伍
            if current_frame_crops:
                current_frame_predictions = self.classifier.predict(current_frame_crops)
                for crop_idx, player_id in enumerate(player_ids):
                    team_id = int(current_frame_predictions[crop_idx]) + 1  # 0 -> 1, 1 -> 2
                    tracks['players'][frame_num][player_id]['team'] = team_id
                    tracks['players'][frame_num][player_id]['team_color'] = self.team_colors[team_id]
    
    def assign_teams_from_detections(self, detections, video_frames, nms_threshold=0.5, training_frames=100):
        """
        从检测结果中分配球队（用于test_no_tracker.py）
        
        算法流程：
        1. 从多个采样帧收集球员crops（用于训练）
        2. 统一训练分类器
        3. 计算队伍代表颜色
        4. 对每帧进行批量预测并分配队伍
        
        Args:
            detections: YOLO检测结果列表
            video_frames: 视频帧列表
            nms_threshold: NMS阈值
            training_frames: 用于训练分类器的采样帧数（默认100帧）
        
        Returns:
            dict: 按帧/队伍/id存储的球员数据
                {
                    frame_num: {
                        team_id: {
                            player_id: {
                                'bbox': [x1, y1, x2, y2],
                                'position_pixel': [x, y],
                                'position_transformed': [x, y]  # 需要view_transformer处理
                            }
                        }
                    }
                }
        """
        import supervision as sv
        from Module.common.utils import get_foot_position
        
        # 获取类别映射
        cls_names = detections[0].names if detections else {}
        cls_names_inv = {v: k for k, v in cls_names.items()}
        
        # 计算采样stride（用于训练分类器）
        total_frames = len(detections)
        if total_frames <= training_frames:
            stride = 1
            sampled_frames = list(range(total_frames))
            print(f"第一步：收集所有 {total_frames} 帧的球员裁剪图像（用于训练分类器）...")
        else:
            stride = max(1, total_frames // training_frames)
            sampled_frames = list(range(0, total_frames, stride))
            print(f"第一步：从 {total_frames} 帧中采样 {len(sampled_frames)} 帧（stride={stride}）收集球员裁剪图像（用于训练分类器）...")
        
        # 第一步：收集采样帧的球员crops（用于统一训练）
        all_player_crops = []
        
        for frame_num in sampled_frames:
            detection = detections[frame_num]
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # 处理守门员类别：将守门员归为player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names.get(class_id) == "goalkeeper":
                    PLAYER_CLASS_ID = cls_names_inv.get("player", -1)
                    if PLAYER_CLASS_ID != -1:
                        detection_supervision.class_id[object_ind] = PLAYER_CLASS_ID
            
            # 只处理球员类别
            PLAYER_CLASS_ID = cls_names_inv.get('player', -1)
            if PLAYER_CLASS_ID == -1:
                continue
            
            player_detections = detection_supervision[detection_supervision.class_id == PLAYER_CLASS_ID]
            player_detections = player_detections.with_nms(threshold=nms_threshold, class_agnostic=True)
            
            # 提取每个球员的crop
            for bbox in player_detections.xyxy:
                bbox_list = bbox.tolist()
                x1, y1, x2, y2 = map(int, bbox_list)
                crop = video_frames[frame_num][y1:y2, x1:x2]
                
                if crop.size > 0:
                    all_player_crops.append(crop)
        
        if len(all_player_crops) < 2:
            raise ValueError(f"收集到的球员数量不足 ({len(all_player_crops)})，无法进行队伍分类")
        
        print(f"共收集到 {len(all_player_crops)} 个球员裁剪图像（来自 {len(sampled_frames)} 个采样帧）")
        
        # 第二步：统一训练分类器（使用所有收集到的crops）
        print("第二步：使用所有球员裁剪图像训练分类器...")
        self.classifier.fit(all_player_crops)
        self.is_fitted = True
        print("分类器训练完成")
        
        # 第三步：计算队伍代表颜色
        print("正在计算队伍代表颜色...")
        all_predictions = self.classifier.predict(all_player_crops)
        
        team_1_colors = []
        team_2_colors = []
        
        for crop, prediction in zip(all_player_crops, all_predictions):
            avg_color = np.mean(crop.reshape(-1, 3), axis=0)
            if prediction == 0:
                team_1_colors.append(avg_color)
            else:
                team_2_colors.append(avg_color)
        
        if team_1_colors:
            self.team_colors[1] = np.mean(team_1_colors, axis=0).tolist()
        else:
            self.team_colors[1] = [255, 0, 0]  # 默认红色
        
        if team_2_colors:
            self.team_colors[2] = np.mean(team_2_colors, axis=0).tolist()
        else:
            self.team_colors[2] = [0, 0, 255]  # 默认蓝色
        
        print(f"队伍1代表颜色 (RGB): {self.team_colors[1]}")
        print(f"队伍2代表颜色 (RGB): {self.team_colors[2]}")
        
        # 第四步：对每帧进行预测并存储数据
        print("第三步：对每帧进行队伍分配...")
        player_data = {}
        
        for frame_num, detection in enumerate(detections):
            player_data[frame_num] = {1: {}, 2: {}}  # 初始化两个队伍的字典
            
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # 处理守门员类别：将守门员归为player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names.get(class_id) == "goalkeeper":
                    PLAYER_CLASS_ID = cls_names_inv.get("player", -1)
                    if PLAYER_CLASS_ID != -1:
                        detection_supervision.class_id[object_ind] = PLAYER_CLASS_ID
            
            # 只处理球员类别
            PLAYER_CLASS_ID = cls_names_inv.get('player', -1)
            if PLAYER_CLASS_ID == -1:
                continue
            
            player_detections = detection_supervision[detection_supervision.class_id == PLAYER_CLASS_ID]
            player_detections = player_detections.with_nms(threshold=nms_threshold, class_agnostic=True)
            
            # 处理每个检测到的球员
            # 先提取当前帧的所有crops，然后批量预测
            current_frame_crops = []
            
            for bbox in player_detections.xyxy:
                bbox_list = bbox.tolist()
                x1, y1, x2, y2 = map(int, bbox_list)
                crop = video_frames[frame_num][y1:y2, x1:x2]
                
                if crop.size > 0:
                    current_frame_crops.append(crop)
            
            # 对当前帧的crops进行预测（批量预测，提高效率）
            if current_frame_crops:
                current_frame_predictions = self.classifier.predict(current_frame_crops)
            else:
                current_frame_predictions = []
            
            # 处理每个检测到的球员
            crop_idx = 0  # crops的索引
            for i, bbox in enumerate(player_detections.xyxy):
                bbox_list = bbox.tolist()
                
                # 获取预测结果（crops和detections的顺序一致）
                if crop_idx < len(current_frame_predictions):
                    team_id = int(current_frame_predictions[crop_idx]) + 1  # 0 -> 1, 1 -> 2
                    crop_idx += 1
                else:
                    # 如果crops数量不足，跳过这个检测
                    continue
                
                # 获取脚部位置（像素坐标）
                position_pixel = get_foot_position(bbox_list)
                
                # 存储数据（使用检测索引作为player_id）
                player_id = i
                player_data[frame_num][team_id][player_id] = {
                    'bbox': bbox_list,
                    'position_pixel': list(position_pixel),
                    'position_transformed': None  # 需要view_transformer处理
                }
        
        print("处理完成！")
        return player_data
    
    def assign_teams_to_detections_raw(self, detections, video_frames, nms_threshold=0.5, training_frames=20):
        """
        从检测结果中分配球队，返回检测索引到球队的映射（用于追踪器）
        
        算法流程：
        1. 从多个采样帧收集球员crops（用于训练）
        2. 统一训练分类器
        3. 计算队伍代表颜色
        4. 对每帧的检测结果分配球队，返回检测索引映射
        
        Args:
            detections: YOLO检测结果列表
            video_frames: 视频帧列表
            nms_threshold: NMS阈值
            training_frames: 用于训练分类器的采样帧数（默认20帧）
        
        Returns:
            dict: 每帧的球队分配结果 {frame_num: {detection_idx: (team, team_color)}}
        """
        import supervision as sv
        
        # 获取类别映射
        cls_names = detections[0].names if detections else {}
        cls_names_inv = {v: k for k, v in cls_names.items()}
        
        # 计算采样stride（用于训练分类器）
        total_frames = len(detections)
        training_frames = min(training_frames, total_frames)
        stride = max(1, total_frames // training_frames)
        sampled_frames = list(range(0, total_frames, stride))[:training_frames]
        
        # 第一步：收集训练数据
        all_player_crops = []
        for frame_idx in sampled_frames:
            detection = detections[frame_idx]
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # 处理守门员类别：将守门员归为player
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names.get(class_id) == "goalkeeper":
                    PLAYER_CLASS_ID = cls_names_inv.get("player", -1)
                    if PLAYER_CLASS_ID != -1:
                        detection_supervision.class_id[object_ind] = PLAYER_CLASS_ID
            
            # 只处理球员类别
            PLAYER_CLASS_ID = cls_names_inv.get('player', -1)
            if PLAYER_CLASS_ID == -1:
                continue
            
            player_detections = detection_supervision[detection_supervision.class_id == PLAYER_CLASS_ID]
            player_detections = player_detections.with_nms(threshold=nms_threshold, class_agnostic=True)
            
            # 提取每个球员的crop
            for bbox in player_detections.xyxy:
                bbox_list = bbox.tolist()
                x1, y1, x2, y2 = map(int, bbox_list)
                crop = video_frames[frame_idx][y1:y2, x1:x2]
                if crop.size > 0:
                    all_player_crops.append(crop)
        
        if len(all_player_crops) < 2:
            raise ValueError(f"收集到的球员数量不足 ({len(all_player_crops)})，无法进行队伍分类")
        
        # 第二步：训练分类器
        self.classifier.fit(all_player_crops)
        self.is_fitted = True
        
        # 第三步：计算队伍代表颜色
        all_predictions = self.classifier.predict(all_player_crops)
        team_1_colors = []
        team_2_colors = []
        for crop, prediction in zip(all_player_crops, all_predictions):
            avg_color = np.mean(crop.reshape(-1, 3), axis=0)
            if prediction == 0:
                team_1_colors.append(avg_color)
            else:
                team_2_colors.append(avg_color)
        
        if team_1_colors:
            self.team_colors[1] = np.mean(team_1_colors, axis=0).tolist()
        else:
            self.team_colors[1] = [255, 0, 0]
        
        if team_2_colors:
            self.team_colors[2] = np.mean(team_2_colors, axis=0).tolist()
        else:
            self.team_colors[2] = [0, 0, 255]
        
        # 第四步：对每帧的检测结果分配球队
        detections_with_teams = {}
        
        for frame_num, detection in enumerate(detections):
            detection_supervision = sv.Detections.from_ultralytics(detection)
            
            # 处理守门员类别
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names.get(class_id) == "goalkeeper":
                    PLAYER_CLASS_ID = cls_names_inv.get("player", -1)
                    if PLAYER_CLASS_ID != -1:
                        detection_supervision.class_id[object_ind] = PLAYER_CLASS_ID
            
            # 只处理球员类别
            PLAYER_CLASS_ID = cls_names_inv.get('player', -1)
            if PLAYER_CLASS_ID == -1:
                detections_with_teams[frame_num] = {}
                continue
            
            player_detections = detection_supervision[detection_supervision.class_id == PLAYER_CLASS_ID]
            player_detections = player_detections.with_nms(threshold=nms_threshold, class_agnostic=True)
            
            # 批量提取当前帧的所有crops
            current_frame_crops = []
            detection_indices = []
            
            for i, bbox in enumerate(player_detections.xyxy):
                bbox_list = bbox.tolist()
                x1, y1, x2, y2 = map(int, bbox_list)
                crop = video_frames[frame_num][y1:y2, x1:x2]
                if crop.size > 0:
                    current_frame_crops.append(crop)
                    detection_indices.append(i)
            
            # 批量预测当前帧的球员队伍
            detections_with_teams[frame_num] = {}
            if current_frame_crops:
                current_frame_predictions = self.classifier.predict(current_frame_crops)
                for crop_idx, det_idx in enumerate(detection_indices):
                    team_id = int(current_frame_predictions[crop_idx]) + 1  # 0 -> 1, 1 -> 2
                    detections_with_teams[frame_num][det_idx] = (team_id, self.team_colors[team_id])
        
        return detections_with_teams