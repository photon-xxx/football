"""
轨迹可视化模块

提供将追踪器处理后的轨迹信息可视化为视频的功能。

主要功能：
- 将轨迹数据按帧组织
- 生成轨迹可视化视频（在球场上绘制轨迹路径、当前位置和轨迹ID）
- 支持活跃轨迹和丢失轨迹的不同显示样式

使用方法：
    from Module.visualizer.track_visualizer import visualize_tracks_as_video
    from Module.common.trackers.tracker_pitch import Tracker
    
    # 执行追踪
    tracker = Tracker(...)
    tracks = tracker.track_detections(tracking_inputs)
    
    # 生成可视化视频
    frame_nums = [inp['frame_num'] for inp in tracking_inputs]
    visualize_tracks_as_video(
        tracks=tracker.tracks,
        frame_nums=frame_nums,
        output_path="output/tracks_visualization.mp4",
        fps=30
    )
"""

import os
from typing import List, Optional
from Module.common.trackers.tracker_pitch import PitchTrack
from Storage.field_configs.soccer import SoccerPitchConfiguration

def organize_tracks_by_frame(
    tracks: List[PitchTrack],
    frame_nums: List[int]
) -> tuple:
    """
    将轨迹数据按帧组织
    
    Args:
        tracks: 追踪器返回的轨迹列表
        frame_nums: 帧号列表，表示需要可视化的帧范围
    
    Returns:
        tuple: (tracks_by_frame, frame_nums_list)
            - tracks_by_frame: 每帧的轨迹列表 [[track1, track2, ...], ...]
            - frame_nums_list: 每帧对应的真实帧号列表
    """
    if not frame_nums:
        return [], []
    
    min_frame = min(frame_nums)
    max_frame = max(frame_nums)
    
    tracks_by_frame = []
    frame_nums_list = []
    
    for frame_num in range(min_frame, max_frame + 1):
        frame_tracks = []
        
        # 遍历所有轨迹，包含所有轨迹（包括丢失的）
        for track in tracks:
            track_start_frame = track.frame_num
            track_end_frame = track.last_seen_frame
            
            # 如果当前帧在轨迹的起始帧之前，跳过
            if frame_num < track_start_frame:
                continue
            
            # 确定该帧的位置：
            # - 如果轨迹在该帧存在（frame_num <= last_seen_frame），使用该帧的位置
            # - 如果轨迹在该帧已丢失（frame_num > last_seen_frame），使用最后可见帧的位置
            frame_position = None
            
            if frame_num <= track_end_frame:
                # 轨迹在该帧仍然活跃，尝试从历史中获取该帧的位置
                for hist in track.trajectory_history:
                    if hist.get('frame_num') == frame_num:
                        frame_position = hist.get('position_pitch')
                        break
                # 如果没找到，使用轨迹的当前位置
                if frame_position is None:
                    frame_position = track.position_pitch.copy() if track.position_pitch is not None else [0, 0]
            else:
                # 轨迹在该帧已丢失，使用最后可见帧的位置
                # 从历史中找到最后可见帧的位置
                last_position = None
                for hist in reversed(track.trajectory_history):
                    if hist.get('frame_num') == track_end_frame:
                        last_position = hist.get('position_pitch')
                        break
                if last_position is not None:
                    frame_position = last_position
                else:
                    # 如果历史中没有，使用轨迹的当前位置
                    frame_position = track.position_pitch.copy() if track.position_pitch is not None else [0, 0]
            
            # 创建轨迹副本用于绘制
            frame_track = PitchTrack(
                position_pitch=frame_position,
                class_id=track.class_id,
                frame_num=frame_num,
                track_id=track.track_id,
                dt=track.kalman_filter.dt
            )
            frame_track.team = track.team
            frame_track.team_color = track.team_color
            frame_track.last_seen_frame = track.last_seen_frame  # 保留最后可见帧信息
            frame_track.trajectory_history = track.trajectory_history.copy()  # 保留完整历史
            frame_tracks.append(frame_track)
        
        tracks_by_frame.append(frame_tracks)
        frame_nums_list.append(frame_num)
    
    return tracks_by_frame, frame_nums_list


def visualize_tracks_as_video(
    tracks: List[PitchTrack],
    frame_nums: List[int],
    output_path: str,
    config: Optional[SoccerPitchConfiguration] = None,
    fps: int = 30,
    draw_trajectory: bool = True,
    draw_current_position: bool = True,
    draw_track_id: bool = True,
    point_radius: int = 10,
    trajectory_thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    verbose: bool = True
) -> None:
    """
    将追踪器处理后的轨迹信息可视化为视频
    
    Args:
        tracks: 追踪器返回的轨迹列表（tracker.tracks）
        frame_nums: 帧号列表，表示需要可视化的帧范围
        output_path: 输出视频路径
        config: 球场配置，如果为None则使用默认配置
        fps: 视频帧率
        draw_trajectory: 是否绘制轨迹线
        draw_current_position: 是否绘制当前位置点
        draw_track_id: 是否绘制轨迹ID
        point_radius: 位置点半径（像素）
        trajectory_thickness: 轨迹线粗细（像素）
        padding: 球场图像边距（像素）
        scale: 坐标缩放比例（像素/厘米）
        verbose: 是否输出进度信息
    
    Returns:
        None
    """
    if config is None:
        config = SoccerPitchConfiguration()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    if verbose:
        print(f"正在生成轨迹可视化视频...")
        print(f"  轨迹数: {len(tracks)}")
        print(f"  帧范围: {min(frame_nums)} - {max(frame_nums)}")
        print(f"  输出路径: {output_path}")
    
    # 组织轨迹数据
    tracks_by_frame, frame_nums_list = organize_tracks_by_frame(tracks, frame_nums)
    
    if verbose:
        print(f"  已组织 {len(tracks_by_frame)} 帧数据")
    
    # 调用轨迹绘制方法生成视频
    PitchTrack.draw_tracks_as_video(
        tracks_list=tracks_by_frame,
        output_path=output_path,
        config=config,
        fps=fps,
        draw_trajectory=draw_trajectory,
        draw_current_position=draw_current_position,
        draw_track_id=draw_track_id,
        point_radius=point_radius,
        trajectory_thickness=trajectory_thickness,
        padding=padding,
        scale=scale,
        frame_nums=frame_nums_list
    )
    
    if verbose:
        print(f"✓ 轨迹可视化视频已保存到: {output_path}")

