from Module.utils import read_video, save_video  # 导入工具函数：读取视频和保存视频
from Module.trackers import Tracker  # 导入跟踪器类
import numpy as np  # numpy 数组运算
from Module.team_assigner import TeamAssigner  # 导入球队分配器
from Module.player_ball_assigner import PlayerBallAssigner  # 导入球员与球分配器
from Module.camera_movement_estimator import CameraMovementEstimator  # 导入相机运动估计器
from Module.view_transformer import ViewTransformer  # 导入视角变换器
from Module.speed_and_distance_estimator import SpeedAndDistance_Estimator  # 导入速度与距离估计器
from Module.visualizer import plot_team_ball_control ,plot_players_speed_distance, plot_players_speed_distance_by_team
#from Module.test import TestRunner  # 导入测试运行器
import supervision as sv
from Storage.field_configs.soccer import SoccerPitchConfiguration
from Module.ball import extract_ball_paths, replace_outliers_based_on_distance
from Module.visualizer.pitch_annotation_tool import draw_pitch, draw_paths_on_pitch

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
ROBOFLOW_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'Player_detection', 'football-player-detection.pt')

# 球场配置路径
PITCH_CONFIG = SoccerPitchConfiguration()

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

def test():
#  ----------------------------------------------------------------------------------------------------------------------#

    # Read Video
    video_frames = read_video(INPUT_VIDEO_PATH)  # 读取输入视频，得到逐帧图像

#  ----------------------------------------------------------------------------------------------------------------------#

    # Initialize Tracker
    tracker = Tracker(ROBOFLOW_MODEL_PATH)  # 初始化目标检测与跟踪器，加载训练好的模型权重

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=READ_FROM_STUB,
                                       stub_path=TRACK_STUB_PATH)  # 获取视频中物体的跟踪结果，可从缓存文件读取
    # Get object positions
    tracker.add_position_to_tracks(tracks)  # 在轨迹中添加物体的位置坐标信息

# ----------------------------------------------------------------------------------------------------------------------#

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])  # 用第一帧初始化相机运动估计器
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=READ_FROM_STUB,
                                                                              stub_path=CAMERA_MOVEMENT_STUB_PATH)  # 获取相机在每一帧的运动（可从缓存读取）
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)  # 根据相机运动修正物体轨迹坐标

# ----------------------------------------------------------------------------------------------------------------------#

    # View Trasnformer
    view_transformer = ViewTransformer(video_frames, device=DEVICE)  # 初始化视角变换器（像素坐标 -> 场地坐标）
    view_transformer.add_transformed_position_to_tracks(tracks)  # 为轨迹添加透视变换后的坐标

# ----------------------------------------------------------------------------------------------------------------------#

    # 运行测试
    # test_runner = TestRunner(FIGURES_SAVE_DIR)
    # test_runner.run_full_test(tracks, video_frames, sample_rate=1)

    # ball_frame0 = tracks["ball"][0]
    # ball_id = list(ball_frame0.keys())[0]
    # print(ball_frame0[ball_id]["transformed_position"])


    ball_paths = extract_ball_paths(tracks)

    # ball_paths = replace_outliers_based_on_distance(ball_paths)

    BALL_on_PITCH = draw_pitch(PITCH_CONFIG)

    BALL_on_PITCH = draw_paths_on_pitch(
    config=PITCH_CONFIG,
    paths=ball_paths,
    color=sv.Color.WHITE,
    pitch=BALL_on_PITCH)

    sv.plot_image(BALL_on_PITCH)
