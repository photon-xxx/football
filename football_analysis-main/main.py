from utils import read_video, save_video  # 导入工具函数：读取视频和保存视频
from trackers import Tracker  # 导入跟踪器类
import cv2  # OpenCV 库
import numpy as np  # numpy 数组运算
from team_assigner import TeamAssigner  # 导入球队分配器
from player_ball_assigner import PlayerBallAssigner  # 导入球员与球分配器
from camera_movement_estimator import CameraMovementEstimator  # 导入相机运动估计器
from view_transformer import ViewTransformer  # 导入视角变换器
from speed_and_distance_estimator import SpeedAndDistance_Estimator  # 导入速度与距离估计器
from visualizer import plot_team_ball_control ,plot_players_speed_distance

def main():
    # Read Video
    video_frames = read_video('input_videos/08fd33_4.mp4')  # 读取输入视频，得到逐帧图像

    # Initialize Tracker
    tracker = Tracker('models/best.pt')  # 初始化目标检测与跟踪器，加载训练好的模型权重

    tracks = tracker.get_object_tracks(video_frames,
                                       read_from_stub=True,
                                       stub_path='stubs/track_stubs.pkl')  # 获取视频中物体的跟踪结果，可从缓存文件读取
    # Get object positions
    tracker.add_position_to_tracks(tracks)  # 在轨迹中添加物体的位置坐标信息

    # camera movement estimator
    camera_movement_estimator = CameraMovementEstimator(video_frames[0])  # 用第一帧初始化相机运动估计器
    camera_movement_per_frame = camera_movement_estimator.get_camera_movement(video_frames,
                                                                              read_from_stub=True,
                                                                              stub_path='stubs/camera_movement_stub.pkl')  # 获取相机在每一帧的运动（可从缓存读取）
    camera_movement_estimator.add_adjust_positions_to_tracks(tracks, camera_movement_per_frame)  # 根据相机运动修正物体轨迹坐标

    # View Trasnformer
    view_transformer = ViewTransformer()  # 初始化视角变换器（像素坐标 -> 场地坐标）
    view_transformer.add_transformed_position_to_tracks(tracks)  # 为轨迹添加透视变换后的坐标

    # Interpolate Ball Positions
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])  # 对球的轨迹插值，补齐丢失的帧位置

    # Speed and distance estimator
    speed_and_distance_estimator = SpeedAndDistance_Estimator()  # 初始化速度与距离估计器
    speed_and_distance_estimator.add_speed_and_distance_to_tracks(tracks)  # 在轨迹中加入速度和移动距离

    # Assign Player Teams
    team_assigner = TeamAssigner()  # 初始化球队分配器
    team_assigner.assign_team_color(video_frames[0],
                                    tracks['players'][0])  # 在第一帧上为球员分配队伍颜色（两队）

    for frame_num, player_track in enumerate(tracks['players']):  # 遍历每一帧的球员轨迹
        for player_id, track in player_track.items():  # 遍历该帧中每个球员
            team = team_assigner.get_player_team(video_frames[frame_num],
                                                 track['bbox'],
                                                 player_id)  # 根据球员的外观（颜色）分配队伍
            tracks['players'][frame_num][player_id]['team'] = team  # 记录队伍编号
            tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]  # 记录队伍颜色

    # Assign Ball Aquisition
    player_assigner = PlayerBallAssigner()  # 初始化球员-球分配器
    team_ball_control = []  # 记录每一帧的球队控球权
    for frame_num, player_track in enumerate(tracks['players']):  # 遍历每一帧
        ball_bbox = tracks['ball'][frame_num][1]['bbox']  # 获取该帧球的包围框
        assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)  # 判断哪位球员持球

        if assigned_player != -1:  # 如果找到了持球队员
            tracks['players'][frame_num][assigned_player]['has_ball'] = True  # 标记该球员持球
            team_ball_control.append(tracks['players'][frame_num][assigned_player]['team'])  # 记录该球员所在队伍
        else:  # 如果无人持球
            team_ball_control.append(team_ball_control[-1])  # 默认沿用上一帧的控球队伍
    team_ball_control = np.array(team_ball_control)  # 转为 numpy 数组，方便后续处理

    # Draw output
    ## Draw object Tracks
    output_video_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)  # 在视频帧上绘制检测与跟踪结果

    ## Draw Camera movement
    output_video_frames = camera_movement_estimator.draw_camera_movement(output_video_frames,
                                                                         camera_movement_per_frame)  # 绘制相机运动轨迹

    ## Draw Speed and Distance
    speed_and_distance_estimator.draw_speed_and_distance(output_video_frames, tracks)  # 在视频上绘制速度和移动距离信息

    # Save visalizer png
    plot_team_ball_control(team_ball_control, save_dir="figures")
    plot_players_speed_distance(tracks, save_dir="figures")

    # Save video
    save_video(output_video_frames, 'output_videos/output_video.avi')  # 保存处理后的视频


if __name__ == '__main__':  # 程序入口
    main()  # 调用主函数
