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
TRACKING_INPUTS_STUB_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'stubs', 'tracking_inputs_stub.pkl')

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
DETECTION_CONF_THRESHOLD = 0.7  # 检测置信度阈值
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


def save_tracking_inputs(tracking_inputs, stub_path):
    """
    保存追踪输入数据到缓存文件
    
    Args:
        tracking_inputs: 追踪输入数据列表
        stub_path: 缓存文件路径
    """
    print(f"正在保存追踪输入数据到: {stub_path}")
    os.makedirs(os.path.dirname(stub_path), exist_ok=True)
    with open(stub_path, 'wb') as f:
        pickle.dump(tracking_inputs, f)
    print("保存完成！")


def load_tracking_inputs(stub_path):
    """
    从缓存文件加载追踪输入数据
    
    Args:
        stub_path: 缓存文件路径
    
    Returns:
        list: 追踪输入数据列表，如果文件不存在则返回None
    """
    if os.path.exists(stub_path):
        print(f"正在从缓存加载追踪输入数据: {stub_path}")
        with open(stub_path, 'rb') as f:
            tracking_inputs = pickle.load(f)
        print(f"加载完成，共 {len(tracking_inputs)} 帧")
        return tracking_inputs
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
    
    # 4. 处理检测结果并分配队伍，生成tracking_inputs格式
    print("\n[步骤 4/5] 处理检测结果并分配队伍，生成tracking_inputs格式...")
    
    # 检查是否从缓存读取
    tracking_inputs = None
    if READ_FROM_STUB:
        tracking_inputs = load_tracking_inputs(TRACKING_INPUTS_STUB_PATH)
    
    if tracking_inputs is None:
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
        
        # 使用封装好的方法分配球队
        player_data_raw = team_assigner.assign_teams_from_detections(
            detections,
            video_frames,
            nms_threshold=NMS_THRESHOLD,
            training_frames=TRAINING_FRAMES
        )
        
        # 直接生成tracking_inputs格式
        tracking_inputs = []
        cls_names = {0: 'player', 1: 'referee', 2: 'ball'}
        cls_names_inv = {v: k for k, v in cls_names.items()}
        
        for frame_num, detection in enumerate(detections):
            detections_pitch = []
            detections_info = []
            
            if frame_num in player_data_raw:
                # 遍历每个队伍的球员
                for team_id in [1, 2]:
                    if team_id not in player_data_raw[frame_num]:
                        continue
                    
                    team_players = player_data_raw[frame_num][team_id]
                    if not team_players:
                        continue
                    
                    # 获取队伍颜色
                    team_color = TEAM_COLORS.get(team_id, sv.Color.RED)
                    if hasattr(team_color, 'as_bgr'):
                        team_color_bgr = team_color.as_bgr()
                    else:
                        team_color_bgr = team_color
                    
                    # 遍历该队伍的每个球员
                    for player_id, player_info in team_players.items():
                        bbox = player_info.get('bbox')
                        position_pixel = player_info.get('position_pixel')
                        
                        # 检查数据有效性
                        if bbox is None or len(bbox) != 4 or np.isnan(bbox).any():
                            continue
                        
                        # 转换为场地坐标
                        position_transformed = None
                        if view_transformer.perspective_transformers[frame_num] is not None:
                            position_transformed = view_transformer.transform_point(position_pixel, frame_num)
                            if position_transformed is not None:
                                position_transformed = position_transformed.squeeze().tolist()
                                if isinstance(position_transformed, (list, np.ndarray)) and np.isnan(position_transformed).any():
                                    position_transformed = None
                        
                        if position_transformed is None:
                            continue
                        
                        # 确保position_transformed是有效的
                        if isinstance(position_transformed, (list, np.ndarray)):
                            pos_array = np.array(position_transformed)
                            if len(pos_array) != 2 or np.isnan(pos_array).any():
                                continue
                        else:
                            continue
                        
                        # 添加到追踪输入
                        detections_pitch.append(position_transformed)
                        detections_info.append({
                            'bbox': bbox,
                            'position_pixel': position_pixel,
                            'class_id': cls_names_inv['player'],  # 类别ID（player=0, referee=1, ball=2）
                            'confidence': 1.0,  # player_data中没有置信度，使用默认值
                            'team': team_id,
                            'team_color': team_color_bgr
                        })
            
            tracking_inputs.append({
                'frame_num': frame_num,
                'detections_pitch': detections_pitch,
                'detections_info': detections_info,
                'cls_names': cls_names,
                'cls_names_inv': cls_names_inv
            })
        
        # 保存到缓存
        save_tracking_inputs(tracking_inputs, TRACKING_INPUTS_STUB_PATH)
    else:
        print("使用缓存数据，跳过检测和队伍分配")
    
    # 为了兼容可视化部分，从tracking_inputs生成player_data（如果需要）
    # 注意：如果后续步骤不需要player_data，可以删除这部分
    player_data = {}
    for tracking_input in tracking_inputs:
        frame_num = tracking_input['frame_num']
        detections_pitch = tracking_input['detections_pitch']
        detections_info = tracking_input['detections_info']
        
        player_data[frame_num] = {1: {}, 2: {}}
        
        for i, det_info in enumerate(detections_info):
            if det_info.get('class_id') != 0:  # 只处理球员
                continue
            
            team_id = det_info.get('team')
            if team_id not in [1, 2]:
                continue
            
            position_pitch = detections_pitch[i]
            player_id = len(player_data[frame_num][team_id])  # 简单的ID分配
            
            player_data[frame_num][team_id][player_id] = {
                'bbox': det_info['bbox'],
                'position_pixel': det_info['position_pixel'],
                'position_transformed': position_pitch
            }
    
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

