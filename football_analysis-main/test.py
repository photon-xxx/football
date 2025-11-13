from Module.common.utils import read_video  # 导入工具函数：读取视频和保存视频
from Module.common.trackers import Tracker  # 导入跟踪器类
import numpy as np  # numpy 数组运算
from Module.common.team_assigner import TeamAssigner  # 导入球队分配器
from Module.common.view_transformer import ViewTransformer  # 导入视角变换器
import supervision as sv
from Storage.field_configs.soccer import SoccerPitchConfiguration
from Module.visualizer.pitch_annotation_tool import draw_pitch, draw_points_on_pitch
import cv2
# =====================================================================================================================
# 全局配置变量 - 所有路径和参数设置
# =====================================================================================================================

# 项目根目录
import os
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# 输入视频路径
INPUT_VIDEO_PATH = os.path.join(PROJECT_ROOT, 'IO', 'input_videos', 'test.mp4')

# 模型路径
YOLO_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'yolo', 'best.pt')
PLAYER_DETECTION_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'Player_detection', 'football-player-detection.pt')
PITCH_DETECTION_PATH = "E:/Soccer/football/football_analysis-main/Storage/models/KeyPoint_detection/football-pitch-detection.pt"
# 球场配置路径
CONFIG = SoccerPitchConfiguration()

# 缓存文件路径
TRACK_STUB_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'stubs', 'track_stubs.pkl')
CAMERA_MOVEMENT_STUB_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'stubs', 'camera_movement_stub.pkl')

# 输出路径
OUTPUT_VIDEO_PATH = os.path.join(PROJECT_ROOT, 'IO', 'output_videos', 'output_video.avi')
FIGURES_SAVE_DIR = os.path.join(PROJECT_ROOT, 'IO', 'figures')

# 设备配置
DEVICE = "cuda"  # 可选: "cpu" 或 "cuda"

# 缓存读取设置
READ_FROM_STUB = True  # 是否从缓存文件读取结果

STRIDE = 60

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3


COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)
def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar

# 外部导入的run_radar函数已注释，使用项目内部的实现替代
# def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
#     # 此函数包含未定义的外部依赖，已注释
#     pass

def generate_radar_from_tracks(tracks, frame, view_transformer, team_assigner):
    """
    基于tracks数据结构生成radar图
    
    Args:
        tracks: 跟踪结果字典
        frame: 输入帧
        view_transformer: 视角变换器
        team_assigner: 球队分配器
    
    Returns:
        np.ndarray: radar图
    """
    # 获取球场关键点检测结果
    keypoint_detector = view_transformer.keypoint_detector
    keypoints = keypoint_detector.detect_keypoints(frame)
    
    # 过滤有效关键点
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if not mask.any():
        print("警告: 未检测到有效的球场关键点")
        return draw_pitch(config=CONFIG)
    
    # 使用现有的view_transformer进行坐标变换
    # 不需要重新创建ViewTransformer，直接使用已有的变换矩阵
    
    # 绘制球场
    radar = draw_pitch(config=CONFIG)
    
    # 处理球员
    player_positions = []
    player_teams = []
    
    for player_id, track in tracks['players'][0].items():
        if 'position_transformed' in track and track['position_transformed'] is not None:
            pos = track['position_transformed']
            if not (isinstance(pos, (list, np.ndarray)) and np.isnan(pos).any()):
                player_positions.append(pos)
                player_teams.append(track.get('team', 0))
    
    if player_positions:
        player_positions = np.array(player_positions)
        player_teams = np.array(player_teams)
        
        # 为每个队伍绘制不同颜色的点
        for team_id in [1, 2]:
            team_mask = player_teams == team_id
            if team_mask.any():
                team_positions = player_positions[team_mask]
                radar = draw_points_on_pitch(
                    config=CONFIG, 
                    xy=team_positions,
                    face_color=sv.Color.from_hex(COLORS[team_id-1]), 
                    radius=20, 
                    pitch=radar
                )
    
    # 处理球
    for ball_id, track in tracks['ball'][0].items():
        if 'position_transformed' in track and track['position_transformed'] is not None:
            ball_pos = track['position_transformed']
            if not (isinstance(ball_pos, (list, np.ndarray)) and np.isnan(ball_pos).any()):
                radar = draw_points_on_pitch(
                    config=CONFIG,
                    xy=[ball_pos],
                    face_color=sv.Color.from_hex(COLORS[2]),  # 球用第3种颜色
                    radius=15,
                    pitch=radar
                )
    
    # 处理裁判
    for ref_id, track in tracks['referees'][0].items():
        if 'position_transformed' in track and track['position_transformed'] is not None:
            ref_pos = track['position_transformed']
            if not (isinstance(ref_pos, (list, np.ndarray)) and np.isnan(ref_pos).any()):
                radar = draw_points_on_pitch(
                    config=CONFIG,
                    xy=[ref_pos],
                    face_color=sv.Color.from_hex(COLORS[3]),  # 裁判用第4种颜色
                    radius=15,
                    pitch=radar
                )
    
    return radar

def draw_annotations_on_frame(frame, tracks, team_assigner):
    """
    在视频帧上绘制检测结果的标注
    
    Args:
        frame: 输入视频帧
        tracks: 跟踪结果字典
        team_assigner: 球队分配器
    
    Returns:
        np.ndarray: 标注后的视频帧
    """
    annotated_frame = frame.copy()
    
    # 绘制球员
    for player_id, track in tracks['players'][0].items():
        bbox = track['bbox']
        team = track.get('team', 0)
        team_color = track.get('team_color', [0, 0, 0])
        
        # 转换颜色格式 (RGB -> BGR for OpenCV)
        if isinstance(team_color, (list, np.ndarray)) and len(team_color) >= 3:
            color = (int(team_color[2]), int(team_color[1]), int(team_color[0]))  # RGB -> BGR
        else:
            color = (0, 255, 0) if team == 1 else (255, 0, 0)  # 默认颜色
        
        # 绘制边界框
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"Player {player_id} (Team {team})"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 绘制裁判
    for ref_id, track in tracks['referees'][0].items():
        bbox = track['bbox']
        
        # 裁判用黄色
        color = (0, 255, 255)
        
        # 绘制边界框
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"Referee {ref_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # 绘制球
    for ball_id, track in tracks['ball'][0].items():
        bbox = track['bbox']
        
        # 球用白色
        color = (255, 255, 255)
        
        # 绘制边界框
        x1, y1, x2, y2 = map(int, bbox)
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
        
        # 绘制标签
        label = f"Ball {ball_id}"
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        cv2.rectangle(annotated_frame, (x1, y1 - label_size[1] - 10), 
                     (x1 + label_size[0], y1), color, -1)
        cv2.putText(annotated_frame, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    return annotated_frame

        
def test():
    """
    测试radar图生成功能，只处理视频的第一帧
    """
    print("开始测试radar图生成功能...")
    
    # 读取视频第一帧
    video_frames = read_video(INPUT_VIDEO_PATH)
    first_frame = video_frames[10]  # 使用第10帧，通常球场关键点更清晰
    print(f"成功读取视频帧，尺寸: {first_frame.shape}")

    # 初始化Tracker
    tracker = Tracker(PLAYER_DETECTION_PATH)
    
    # 只对第一帧进行检测和跟踪
    tracks = tracker.get_object_tracks([first_frame],
                                       read_from_stub=READ_FROM_STUB,
                                       stub_path=TRACK_STUB_PATH)
    
    # 添加位置信息
    tracker.add_position_to_tracks(tracks)
    print(f"检测结果 - 球员: {len(tracks['players'][0])}, 裁判: {len(tracks['referees'][0])}, 球: {len(tracks['ball'][0])}")

    # 初始化ViewTransformer（禁用平滑处理用于单帧测试）
    print("初始化ViewTransformer...")
    import Module.common.view_transformer.view_transformer as vt_module
    original_USE_SMOOTHING = vt_module.USE_SMOOTHING
    vt_module.USE_SMOOTHING = False  # 禁用平滑处理
    
    view_transformer = ViewTransformer([first_frame], device=DEVICE)
    view_transformer.add_transformed_position_to_tracks(tracks)
    
    # 恢复原始设置
    vt_module.USE_SMOOTHING = original_USE_SMOOTHING
    print("完成视角变换")

    # 初始化球队分配器
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(first_frame, tracks['players'][0])
    print("完成球队颜色分配")

    # 为球员分配队伍
    for player_id, track in tracks['players'][0].items():
        team = team_assigner.get_player_team(first_frame, track['bbox'], player_id)
        tracks['players'][0][player_id]['team'] = team
        tracks['players'][0][player_id]['team_color'] = team_assigner.team_colors[team]
    print("完成球员队伍分配")

    # 生成radar图
    radar_image = generate_radar_from_tracks(tracks, first_frame, view_transformer, team_assigner)
    
    # 保存radar图
    radar_output_path = os.path.join(FIGURES_SAVE_DIR, 'test_radar.png')
    cv2.imwrite(radar_output_path, radar_image)
    print(f"Radar图已保存到: {radar_output_path}")
    
    # 生成带标注的视频帧
    annotated_frame = draw_annotations_on_frame(first_frame, tracks, team_assigner)
    
    # 保存标注后的视频帧
    frame_output_path = os.path.join(FIGURES_SAVE_DIR, 'test_annotated_frame.png')
    cv2.imwrite(frame_output_path, annotated_frame)
    print(f"标注后的视频帧已保存到: {frame_output_path}")
    
    # 显示radar图
    sv.plot_image(radar_image)
    print("测试完成！")

if __name__ == "__main__":
    test()

