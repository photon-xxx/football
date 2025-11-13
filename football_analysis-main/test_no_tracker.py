"""
无追踪的球员检测与可视化流程

这个脚本实现了以下功能：
1. 使用YOLO检测球员（不进行追踪）
2. 初始化视角转换器
3. 将球员检测框转化为真实坐标（场地坐标）
4. 按帧/队伍/id存储球员坐标信息，并支持缓存
5. 在球场图上可视化所有帧，生成视频，区分两只球队

数据结构：
player_data = {
    frame_num: {
        team_id: {
            player_id: {
                'bbox': [x1, y1, x2, y2],
                'position_pixel': [x, y],  # 像素坐标（脚部位置）
                'position_transformed': [x, y]  # 场地坐标（米）
            }
        }
    }
}
"""

from Module.common.utils import read_video, get_foot_position, save_video
from Module.common.team_assigner import TeamAssigner
from Module.common.view_transformer import ViewTransformer
from ultralytics import YOLO
import supervision as sv
import numpy as np
import pickle
import os
import cv2
from Storage.field_configs.soccer import SoccerPitchConfiguration
from Module.visualizer.pitch_annotation_tool import draw_pitch, draw_points_on_pitch

# =====================================================================================================================
# 全局配置变量
# =====================================================================================================================

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 输入视频路径
INPUT_VIDEO_PATH = os.path.join(PROJECT_ROOT, 'IO', 'input_videos', 'test.mp4')

# 模型路径
PLAYER_DETECTION_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'Player_detection', 'football-player-detection.pt')

# SigLIP模型本地路径（如果已手动下载，设置此路径）
# 如果为None，则从HuggingFace下载或使用缓存
SIGLIP_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'siglip', 'siglip-base-patch16-224')
# 如果模型不在上述路径，可以取消注释并修改下面的路径：
# SIGLIP_MODEL_PATH = None  # 使用HuggingFace下载或缓存

# 球场配置
PITCH_CONFIG = SoccerPitchConfiguration()

# 缓存文件路径
PLAYER_DATA_STUB_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'stubs', 'player_data_stub.pkl')

# 输出路径
FIGURES_SAVE_DIR = os.path.join(PROJECT_ROOT, 'IO', 'figures')
OUTPUT_VIDEO_PATH = os.path.join(PROJECT_ROOT, 'IO', 'output_videos', 'players_on_pitch.avi')
OUTPUT_MERGED_VIDEO_PATH = os.path.join(PROJECT_ROOT, 'IO', 'output_videos', 'merged_video.avi')

# 设备配置
DEVICE = "cuda"  # 可选: "cpu" 或 "cuda"

# 缓存读取设置
READ_FROM_STUB = False  # 是否从缓存文件读取结果

# 队伍颜色配置（用于可视化）
TEAM_COLORS = {
    1: sv.Color.from_hex('#FF1493'),  # 队伍1：深粉色
    2: sv.Color.from_hex('#00BFFF'),  # 队伍2：深天蓝色
}

# YOLO检测配置
DETECTION_CONF_THRESHOLD = 0.3  # 检测置信度阈值
NMS_THRESHOLD = 0.5  # NMS阈值

# 队伍分类器训练配置
TRAINING_FRAMES = 20  # 用于训练分类器的采样帧数（默认100帧，如果视频帧数少于100则使用所有帧）


def detect_players_without_tracking(frames, model_path, conf_threshold=0.3, nms_threshold=0.5):
    """
    使用YOLO检测球员，不进行追踪
    
    Args:
        frames: 视频帧列表
        model_path: YOLO模型路径
        conf_threshold: 检测置信度阈值
        nms_threshold: NMS阈值
    
    Returns:
        list: 每帧的检测结果列表，每个元素是ultralytics Results对象
    """
    print("正在初始化YOLO模型...")
    model = YOLO(model_path)
    
    print(f"正在检测 {len(frames)} 帧...")
    detections = []
    batch_size = 12
    
    for i in range(0, len(frames), batch_size):
        batch_frames = frames[i:i + batch_size]
        results = model.predict(batch_frames, conf=conf_threshold, verbose=False)
        detections.extend(results)
    
    print(f"检测完成，共处理 {len(detections)} 帧")
    return detections


def process_detections_to_player_data(detections, video_frames, view_transformer, team_assigner, nms_threshold=0.5, training_frames=100):
    """
    处理检测结果，转换为按帧/队伍/id存储的球员数据
    
    参考代码的处理方式：
    1. 先收集采样帧的球员crops（用于训练）
    2. 统一训练分类器（fit）
    3. 然后对每帧进行预测（predict）
    
    Args:
        detections: YOLO检测结果列表
        video_frames: 视频帧列表
        view_transformer: 视角转换器
        team_assigner: 队伍分配器
        nms_threshold: NMS阈值
        training_frames: 用于训练分类器的采样帧数（默认100帧）
    
    Returns:
        dict: 按帧/队伍/id存储的球员数据
    """
    print("正在处理检测结果并分配队伍...")
    
    # 获取类别映射
    cls_names = detections[0].names if detections else {}
    cls_names_inv = {v: k for k, v in cls_names.items()}
    
    # 计算采样stride（用于训练分类器）
    total_frames = len(detections)
    if total_frames <= training_frames:
        # 如果总帧数少于目标帧数，使用所有帧
        stride = 1
        sampled_frames = list(range(total_frames))
        print(f"第一步：收集所有 {total_frames} 帧的球员裁剪图像（用于训练分类器）...")
    else:
        # 计算stride，确保采样到大约training_frames帧
        stride = max(1, total_frames // training_frames)
        sampled_frames = list(range(0, total_frames, stride))
        print(f"第一步：从 {total_frames} 帧中采样 {len(sampled_frames)} 帧（stride={stride}）收集球员裁剪图像（用于训练分类器）...")
    
    # 第一步：收集采样帧的球员crops（用于统一训练）
    all_player_crops = []
    
    for frame_num in sampled_frames:
        detection = detections[frame_num]
        # 转换为supervision格式
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
        
        # 应用NMS
        player_detections = player_detections.with_nms(threshold=nms_threshold, class_agnostic=True)
        
        # 提取每个球员的crop
        for bbox in player_detections.xyxy:
            bbox_list = bbox.tolist()
            x1, y1, x2, y2 = map(int, bbox_list)
            crop = video_frames[frame_num][y1:y2, x1:x2]
            
            if crop.size > 0:  # 确保裁剪不为空
                all_player_crops.append(crop)
    
    if len(all_player_crops) < 2:
        raise ValueError(f"收集到的球员数量不足 ({len(all_player_crops)})，无法进行队伍分类")
    
    print(f"共收集到 {len(all_player_crops)} 个球员裁剪图像（来自 {len(sampled_frames)} 个采样帧）")
    
    # 第二步：统一训练分类器（使用所有收集到的crops）
    print("第二步：使用所有球员裁剪图像训练分类器...")
    team_assigner.classifier.fit(all_player_crops)
    team_assigner.is_fitted = True
    print("分类器训练完成")
    
    # 第三步：对每帧进行预测并存储数据
    print("第三步：对每帧进行队伍分配...")
    player_data = {}
    
    # 计算队伍代表颜色（用于可视化）
    # 对所有训练用的crops进行预测，用于计算队伍代表颜色
    print("正在计算队伍代表颜色...")
    all_predictions = team_assigner.classifier.predict(all_player_crops)
    
    team_1_colors = []
    team_2_colors = []
    
    for i, (crop, prediction) in enumerate(zip(all_player_crops, all_predictions)):
        avg_color = np.mean(crop.reshape(-1, 3), axis=0)
        if prediction == 0:
            team_1_colors.append(avg_color)
        else:
            team_2_colors.append(avg_color)
    
    if team_1_colors:
        team_assigner.team_colors[1] = np.mean(team_1_colors, axis=0).tolist()
    else:
        team_assigner.team_colors[1] = [255, 0, 0]  # 默认红色
    
    if team_2_colors:
        team_assigner.team_colors[2] = np.mean(team_2_colors, axis=0).tolist()
    else:
        team_assigner.team_colors[2] = [0, 0, 255]  # 默认蓝色
    
    print(f"队伍1代表颜色 (RGB): {team_assigner.team_colors[1]}")
    print(f"队伍2代表颜色 (RGB): {team_assigner.team_colors[2]}")
    
    # 按帧组织数据
    for frame_num, detection in enumerate(detections):
        player_data[frame_num] = {1: {}, 2: {}}  # 初始化两个队伍的字典
        
        # 转换为supervision格式
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
        
        # 应用NMS
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
            current_frame_predictions = team_assigner.classifier.predict(current_frame_crops)
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
            
            # 转换为场地坐标
            position_transformed = None
            if view_transformer.perspective_transformers[frame_num] is not None:
                position_transformed = view_transformer.transform_point(position_pixel, frame_num)
                if position_transformed is not None:
                    position_transformed = position_transformed.squeeze().tolist()
                    # 检查是否为NaN
                    if isinstance(position_transformed, (list, np.ndarray)) and np.isnan(position_transformed).any():
                        position_transformed = None
            
            # 存储数据（使用检测索引作为player_id）
            player_id = i
            player_data[frame_num][team_id][player_id] = {
                'bbox': bbox_list,
                'position_pixel': list(position_pixel),
                'position_transformed': position_transformed
            }
        
        if frame_num % 30 == 0:
            team1_count = len(player_data[frame_num][1])
            team2_count = len(player_data[frame_num][2])
            print(f"帧 {frame_num}: 队伍1有 {team1_count} 个球员, 队伍2有 {team2_count} 个球员")
    
    print("处理完成！")
    return player_data


def visualize_players_on_pitch(player_data, frame_num, config=PITCH_CONFIG):
    """
    在球场图上可视化指定帧的球员位置
    
    Args:
        player_data: 球员数据字典
        frame_num: 要可视化的帧号
        config: 球场配置
    
    Returns:
        np.ndarray: 可视化后的球场图像
    """
    if frame_num not in player_data:
        print(f"警告: 帧 {frame_num} 不存在于数据中")
        return draw_pitch(config=config)
    
    # 绘制球场
    pitch_image = draw_pitch(config=config)
    
    frame_data = player_data[frame_num]
    
    # 为每个队伍绘制球员位置
    for team_id in [1, 2]:
        if team_id not in frame_data:
            continue
        
        team_players = frame_data[team_id]
        if not team_players:
            continue
        
        # 收集该队伍所有球员的场地坐标
        team_positions = []
        for player_id, player_info in team_players.items():
            pos_transformed = player_info.get('position_transformed')
            if pos_transformed is not None and not (isinstance(pos_transformed, (list, np.ndarray)) and np.isnan(pos_transformed).any()):
                team_positions.append(pos_transformed)
        
        if team_positions:
            team_positions = np.array(team_positions)
            color = TEAM_COLORS.get(team_id, sv.Color.RED)
            
            # 在球场上绘制该队伍的球员位置
            pitch_image = draw_points_on_pitch(
                config=config,
                xy=team_positions,
                face_color=color,
                radius=20,  # 球员点的大小
                pitch=pitch_image
            )
    
    return pitch_image


def save_player_data(player_data, stub_path):
    """
    保存球员数据到缓存文件
    
    Args:
        player_data: 球员数据字典
        stub_path: 缓存文件路径
    """
    print(f"正在保存球员数据到: {stub_path}")
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    with open(stub_path, 'wb') as f:
        pickle.dump(player_data, f)
    print("保存完成！")


def load_player_data(stub_path):
    """
    从缓存文件加载球员数据
    
    Args:
        stub_path: 缓存文件路径
    
    Returns:
        dict: 球员数据字典，如果文件不存在则返回None
    """
    if os.path.exists(stub_path):
        print(f"正在从缓存加载数据: {stub_path}")
        with open(stub_path, 'rb') as f:
            player_data = pickle.load(f)
        print(f"加载完成，共 {len(player_data)} 帧")
        return player_data
    return None


def draw_teams_on_frame(frame, player_data, frame_num):
    """
    在原视频帧上绘制检测结果和球队信息（不追踪，只显示队伍）
    
    Args:
        frame: 原始视频帧
        player_data: 球员数据字典（包含队伍信息）
        frame_num: 当前帧号
    
    Returns:
        np.ndarray: 标注后的视频帧
    """
    import cv2
    
    annotated_frame = frame.copy()
    
    # 获取当前帧的球员数据
    if frame_num not in player_data:
        return annotated_frame
    
    frame_player_data = player_data[frame_num]
    
    # 直接遍历player_data中已分配好队伍的球员
    for team_id in [1, 2]:
        if team_id not in frame_player_data:
            continue
        
        team_players = frame_player_data[team_id]
        
        # 获取队伍颜色
        if team_id in TEAM_COLORS:
            team_color = TEAM_COLORS[team_id]
            # 转换为BGR格式（OpenCV使用BGR）
            if hasattr(team_color, 'as_bgr'):
                team_color_bgr = team_color.as_bgr()
            else:
                # 如果是RGB元组，转换为BGR
                team_color_bgr = (team_color[2], team_color[1], team_color[0]) if len(team_color) >= 3 else (0, 255, 0)
        else:
            # 默认颜色：队伍1绿色，队伍2蓝色
            team_color_bgr = (0, 255, 0) if team_id == 1 else (255, 0, 0)
        
        # 为每个球员绘制标注
        for player_id, player_info in team_players.items():
            bbox = player_info.get('bbox', [])
            if len(bbox) == 4 and not np.isnan(bbox).any():
                x1, y1, x2, y2 = map(int, bbox)
                
                # 绘制检测框（使用队伍颜色）
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), team_color_bgr, 2)
                
                # 绘制标签（只显示队伍）
                label = f"Team {team_id}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # 绘制半透明背景
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0] + 10, y1), team_color_bgr, -1)
                cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                
                # 绘制文字
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame


def merge_frames(original_frame, pitch_frame, layout='horizontal'):
    """
    将原视频帧和可视化帧合并
    
    Args:
        original_frame: 原视频帧 (BGR格式)
        pitch_frame: 球场可视化帧 (BGR格式)
        layout: 布局方式，'horizontal'（左右并排）或 'vertical'（上下堆叠）
    
    Returns:
        np.ndarray: 合并后的帧
    """
    h1, w1 = original_frame.shape[:2]
    h2, w2 = pitch_frame.shape[:2]
    
    if layout == 'horizontal':
        # 左右并排：调整高度使其一致
        target_height = max(h1, h2)
        target_width = w1 + w2
        
        # 调整原视频帧高度
        if h1 != target_height:
            original_resized = cv2.resize(original_frame, (w1, target_height))
        else:
            original_resized = original_frame
        
        # 调整可视化帧高度
        if h2 != target_height:
            pitch_resized = cv2.resize(pitch_frame, (w2, target_height))
        else:
            pitch_resized = pitch_frame
        
        # 水平合并
        merged_frame = np.hstack([original_resized, pitch_resized])
    
    else:  # vertical
        # 上下堆叠：调整宽度使其一致
        target_width = max(w1, w2)
        target_height = h1 + h2
        
        # 调整原视频帧宽度
        if w1 != target_width:
            original_resized = cv2.resize(original_frame, (target_width, h1))
        else:
            original_resized = original_frame
        
        # 调整可视化帧宽度
        if w2 != target_width:
            pitch_resized = cv2.resize(pitch_frame, (target_width, h2))
        else:
            pitch_resized = pitch_frame
        
        # 垂直合并
        merged_frame = np.vstack([original_resized, pitch_resized])
    
    return merged_frame


def main():
    """
    主函数：执行完整的检测、转换、存储和可视化流程
    """
    print("=" * 80)
    print("开始无追踪的球员检测与可视化流程")
    print("=" * 80)
    
    # 1. 读取视频
    print("\n[步骤 1/5] 读取视频...")
    video_frames = read_video(INPUT_VIDEO_PATH)
    print(f"成功读取 {len(video_frames)} 帧")
    
    # 2. 检测球员（不追踪）
    print("\n[步骤 2/5] 使用YOLO检测球员（不进行追踪）...")
    detections = detect_players_without_tracking(
        video_frames,
        PLAYER_DETECTION_PATH,
        conf_threshold=DETECTION_CONF_THRESHOLD,
        nms_threshold=NMS_THRESHOLD
    )
    
    # 3. 初始化视角转换器
    print("\n[步骤 3/5] 初始化视角转换器...")
    view_transformer = ViewTransformer(video_frames, device=DEVICE)
    print("视角转换器初始化完成")
    
    # 4. 处理检测结果并分配队伍
    print("\n[步骤 4/5] 处理检测结果并分配队伍...")
    
    # 检查是否从缓存读取
    player_data = None
    if READ_FROM_STUB:
        player_data = load_player_data(PLAYER_DATA_STUB_PATH)
    
    if player_data is None:
        # 初始化队伍分配器（使用新算法，支持GPU加速）
        # 只使用本地模型路径
        if not os.path.exists(SIGLIP_MODEL_PATH):
            raise FileNotFoundError(
                f"未找到本地SigLIP模型！\n"
                f"模型路径: {SIGLIP_MODEL_PATH}\n"
                f"请确保模型已下载到该路径，包含以下文件：\n"
                f"  - config.json\n"
                f"  - preprocessor_config.json\n"
                f"  - model.safetensors (或 pytorch_model.bin)\n"
                f"  - tokenizer_config.json\n"
                f"  - spiece.model\n"
                f"  - special_tokens_map.json"
            )
        
        print(f"使用本地模型路径: {SIGLIP_MODEL_PATH}")
        team_assigner = TeamAssigner(device=DEVICE, batch_size=32, local_model_path=SIGLIP_MODEL_PATH)
        
        # 处理检测结果
        player_data = process_detections_to_player_data(
            detections,
            video_frames,
            view_transformer,
            team_assigner,
            nms_threshold=NMS_THRESHOLD,
            training_frames=TRAINING_FRAMES
        )
        
        # 保存到缓存
        save_player_data(player_data, PLAYER_DATA_STUB_PATH)
    else:
        print("使用缓存数据，跳过检测和队伍分配")
    
    # 5. 可视化（生成视频）
    print("\n[步骤 5/5] 生成球员位置可视化视频...")
    
    # 为每一帧生成可视化图像
    pitch_frames_list = []
    total_frames = len(player_data)
    
    print(f"正在生成 {total_frames} 帧的可视化图像...")
    for frame_num in range(total_frames):
        pitch_image = visualize_players_on_pitch(player_data, frame_num)
        pitch_frames_list.append(pitch_image)
        
        # 每处理30帧打印一次进度
        if (frame_num + 1) % 30 == 0:
            print(f"已处理 {frame_num + 1}/{total_frames} 帧 ({(frame_num + 1) / total_frames * 100:.1f}%)")
    
    print(f"可视化图像生成完成，共 {len(pitch_frames_list)} 帧")
    
    # 保存单独的可视化视频（可选）
    print(f"正在保存可视化视频到: {OUTPUT_VIDEO_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    save_video(pitch_frames_list, OUTPUT_VIDEO_PATH)
    print(f"可视化视频已保存到: {OUTPUT_VIDEO_PATH}")
    
    # 合并原视频（带队伍标注）和可视化视频
    print(f"\n正在合并原视频（带队伍标注）和可视化结果...")
    print(f"原视频帧数: {len(video_frames)}, 可视化帧数: {len(pitch_frames_list)}")
    
    merged_frames = []
    min_frames = min(len(video_frames), len(pitch_frames_list))
    
    print(f"正在合并 {min_frames} 帧...")
    for frame_num in range(min_frames):
        # 对原视频帧进行队伍标注（不追踪）
        annotated_frame = draw_teams_on_frame(
            video_frames[frame_num], 
            player_data, 
            frame_num
        )
        pitch_frame = pitch_frames_list[frame_num]
        
        # 合并帧（左右并排）
        merged_frame = merge_frames(annotated_frame, pitch_frame, layout='horizontal')
        merged_frames.append(merged_frame)
        
        # 每处理30帧打印一次进度
        if (frame_num + 1) % 30 == 0:
            print(f"已合并 {frame_num + 1}/{min_frames} 帧 ({(frame_num + 1) / min_frames * 100:.1f}%)")
    
    print(f"合并完成，共 {len(merged_frames)} 帧")
    
    # 保存合并后的视频
    print(f"正在保存合并视频到: {OUTPUT_MERGED_VIDEO_PATH}")
    os.makedirs(os.path.dirname(OUTPUT_MERGED_VIDEO_PATH), exist_ok=True)
    save_video(merged_frames, OUTPUT_MERGED_VIDEO_PATH)
    print(f"合并视频已保存到: {OUTPUT_MERGED_VIDEO_PATH}")
    
    # 可选：保存第一帧和最后一帧作为静态图像示例
    if pitch_frames_list:
        os.makedirs(FIGURES_SAVE_DIR, exist_ok=True)
        cv2.imwrite(os.path.join(FIGURES_SAVE_DIR, 'players_frame_0.png'), pitch_frames_list[0])
        cv2.imwrite(os.path.join(FIGURES_SAVE_DIR, f'players_frame_{len(pitch_frames_list)-1}.png'), pitch_frames_list[-1])
        
        # 保存合并后的第一帧作为示例
        if merged_frames:
            cv2.imwrite(os.path.join(FIGURES_SAVE_DIR, 'merged_frame_0.png'), merged_frames[0])
        print(f"已保存示例图像: 第0帧和第{len(pitch_frames_list)-1}帧")
    
    print("\n" + "=" * 80)
    print("流程完成！")
    print("=" * 80)
    
    # 打印统计信息
    print("\n统计信息:")
    total_frames = len(player_data)
    frames_with_players = sum(1 for f in player_data.values() if any(len(team) > 0 for team in f.values()))
    print(f"总帧数: {total_frames}")
    print(f"有球员的帧数: {frames_with_players}")
    
    # 计算平均每帧的球员数
    total_players = sum(
        len(team) 
        for frame_data in player_data.values() 
        for team in frame_data.values()
    )
    avg_players_per_frame = total_players / total_frames if total_frames > 0 else 0
    print(f"平均每帧球员数: {avg_players_per_frame:.2f}")


if __name__ == '__main__':
    main()

