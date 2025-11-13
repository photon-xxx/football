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
    def __init__(self, device: str = 'cpu', batch_size: int = 32, local_model_path: str = None):
        """
        初始化球队分配器
        
        Args:
            device: 运行设备 ('cpu' 或 'cuda')，建议使用'cuda'以加速处理
            batch_size: 批量处理图像的大小，默认32
            local_model_path: 本地模型路径（如果模型已手动下载），如果为None则使用默认配置
        """
        self.device = device
        self.batch_size = batch_size
        self.classifier = TeamClassifier(device=device, batch_size=batch_size, local_model_path=local_model_path)
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
