"""
追踪器调试测试脚本

用于详细测试YOLO检测和追踪器效果，诊断问题
"""

import os
import sys
import numpy as np
import cv2
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Module.common.utils import read_video
from Module.common.trackers.tracker_pitch import Tracker
from Module.common.view_transformer import ViewTransformer
from Module.common.team_assigner import TeamAssigner

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
INPUT_VIDEO_PATH = os.path.join(PROJECT_ROOT, 'IO', 'input_videos', 'test.mp4')
ROBOFLOW_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'Player_detection', 'football-player-detection.pt')
SIGLIP_MODEL_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'siglip', 'siglip-base-patch16-224')
TRACK_STUB_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'stubs', 'track_stubs.pkl')
DEVICE = "cuda"

def analyze_detections(detections, frames):
    """分析YOLO检测结果"""
    print("\n" + "="*80)
    print("YOLO检测分析")
    print("="*80)
    
    total_detections = 0
    player_detections = 0
    referee_detections = 0
    ball_detections = 0
    
    frame_stats = []
    
    for frame_idx, det in enumerate(detections[:30]):  # 分析前30帧
        if not hasattr(det, 'boxes') or det.boxes is None:
            frame_stats.append({
                'frame': frame_idx,
                'total': 0,
                'players': 0,
                'referees': 0,
                'ball': 0
            })
            continue
        
        num_detections = len(det.boxes)
        total_detections += num_detections
        
        cls_names = det.names
        cls_names_inv = {v: k for k, v in cls_names.items()}
        
        players = 0
        referees = 0
        ball = 0
        
        if 'player' in cls_names_inv:
            player_id = cls_names_inv['player']
            player_boxes = det.boxes[det.boxes.cls == player_id]
            players = len(player_boxes)
            player_detections += players
        
        if 'referee' in cls_names_inv:
            referee_id = cls_names_inv['referee']
            referee_boxes = det.boxes[det.boxes.cls == referee_id]
            referees = len(referee_boxes)
            referee_detections += referees
        
        if 'ball' in cls_names_inv:
            ball_id = cls_names_inv['ball']
            ball_boxes = det.boxes[det.boxes.cls == ball_id]
            ball = len(ball_boxes)
            ball_detections += ball
        
        frame_stats.append({
            'frame': frame_idx,
            'total': num_detections,
            'players': players,
            'referees': referees,
            'ball': ball
        })
    
    print(f"前30帧统计:")
    print(f"  总检测数: {total_detections}")
    print(f"  球员检测数: {player_detections}")
    print(f"  裁判检测数: {referee_detections}")
    print(f"  球检测数: {ball_detections}")
    print(f"  平均每帧检测数: {total_detections/30:.2f}")
    print(f"  平均每帧球员数: {player_detections/30:.2f}")
    
    print(f"\n前10帧详细统计:")
    for stat in frame_stats[:10]:
        print(f"  帧{stat['frame']:3d}: 总={stat['total']:2d}, 球员={stat['players']:2d}, "
              f"裁判={stat['referees']:2d}, 球={stat['ball']:2d}")
    
    return frame_stats

def analyze_tracks(tracks, frames):
    """分析追踪结果"""
    print("\n" + "="*80)
    print("追踪结果分析")
    print("="*80)
    
    total_tracked = 0
    frames_with_players = 0
    unique_track_ids = set()
    
    frame_track_stats = []
    
    for frame_idx in range(min(30, len(tracks['players']))):
        frame_players = tracks['players'][frame_idx]
        num_players = len(frame_players)
        
        if num_players > 0:
            frames_with_players += 1
            total_tracked += num_players
            unique_track_ids.update(frame_players.keys())
        
        frame_track_stats.append({
            'frame': frame_idx,
            'players': num_players,
            'track_ids': list(frame_players.keys())
        })
    
    print(f"前30帧追踪统计:")
    print(f"  有球员的帧数: {frames_with_players}/30")
    print(f"  总追踪数: {total_tracked}")
    print(f"  唯一轨迹ID数: {len(unique_track_ids)}")
    print(f"  平均每帧追踪数: {total_tracked/30:.2f}")
    
    print(f"\n前10帧详细追踪统计:")
    for stat in frame_track_stats[:10]:
        track_ids_str = ','.join(map(str, stat['track_ids'][:5]))  # 只显示前5个ID
        if len(stat['track_ids']) > 5:
            track_ids_str += f",...({len(stat['track_ids'])}个)"
        print(f"  帧{stat['frame']:3d}: {stat['players']:2d}个球员, ID=[{track_ids_str}]")
    
    # 分析轨迹连续性
    print(f"\n轨迹连续性分析:")
    track_id_frames = {}  # {track_id: [frame1, frame2, ...]}
    for frame_idx in range(min(100, len(tracks['players']))):
        for track_id in tracks['players'][frame_idx].keys():
            if track_id not in track_id_frames:
                track_id_frames[track_id] = []
            track_id_frames[track_id].append(frame_idx)
    
    if track_id_frames:
        track_lengths = [len(frames) for frames in track_id_frames.values()]
        print(f"  轨迹数量: {len(track_id_frames)}")
        print(f"  平均轨迹长度: {np.mean(track_lengths):.1f}帧")
        print(f"  最长轨迹: {max(track_lengths)}帧")
        print(f"  最短轨迹: {min(track_lengths)}帧")
        
        # 显示前10个轨迹的连续性
        print(f"\n  前10个轨迹的连续性:")
        for i, (track_id, frames) in enumerate(list(track_id_frames.items())[:10]):
            if len(frames) > 1:
                gaps = [frames[j+1] - frames[j] for j in range(len(frames)-1)]
                max_gap = max(gaps) if gaps else 0
                print(f"    ID {track_id}: {len(frames)}帧, 帧范围=[{min(frames)}-{max(frames)}], "
                      f"最大间隔={max_gap}帧")
            else:
                print(f"    ID {track_id}: {len(frames)}帧, 帧={frames[0]}")
    
    return frame_track_stats, track_id_frames

def compare_detection_tracking(detections, tracks, frames):
    """对比检测和追踪结果"""
    print("\n" + "="*80)
    print("检测 vs 追踪对比")
    print("="*80)
    
    for frame_idx in range(min(10, len(detections), len(tracks['players']))):
        det = detections[frame_idx]
        tracked_players = tracks['players'][frame_idx]
        
        # 统计检测中的球员数
        det_players = 0
        if hasattr(det, 'boxes') and det.boxes is not None:
            cls_names = det.names
            cls_names_inv = {v: k for k, v in cls_names.items()}
            if 'player' in cls_names_inv:
                player_id = cls_names_inv['player']
                player_boxes = det.boxes[det.boxes.cls == player_id]
                det_players = len(player_boxes)
        
        tracked_count = len(tracked_players)
        
        match_rate = (tracked_count / det_players * 100) if det_players > 0 else 0
        
        print(f"  帧{frame_idx:3d}: 检测={det_players:2d}个球员, 追踪={tracked_count:2d}个球员, "
              f"匹配率={match_rate:5.1f}%")

def main():
    print("="*80)
    print("追踪器调试测试")
    print("="*80)
    
    # 1. 读取视频
    print("\n[步骤 1] 读取视频...")
    video_frames = read_video(INPUT_VIDEO_PATH)
    print(f"✓ 成功读取 {len(video_frames)} 帧")
    print(f"  视频分辨率: {video_frames[0].shape if video_frames else 'N/A'}")
    
    # 2. 初始化视角变换器
    print("\n[步骤 2] 初始化视角变换器...")
    view_transformer = ViewTransformer(video_frames, device=DEVICE)
    print("✓ 视角变换器初始化完成")
    valid_transforms = sum(1 for t in view_transformer.perspective_transformers if t is not None)
    print(f"  有效变换矩阵: {valid_transforms}/{len(video_frames)} 帧")
    
    # 3. 初始化追踪器
    print("\n[步骤 3] 初始化追踪器...")
    tracker = Tracker(
        model_path=ROBOFLOW_MODEL_PATH,
        nms_threshold=0.5,
        view_transformer=view_transformer,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        max_velocity=0.4,
        track_buffer=30,
        sampling_rate=30.0,  # 禁用采样
        video_fps=30.0
    )
    print("✓ 追踪器初始化完成")
    
    # 4. 单独测试YOLO检测
    print("\n[步骤 4] 测试YOLO检测...")
    detections = tracker.detect_frames(video_frames[:30], batch_size=12, conf_threshold=0.3)
    frame_stats = analyze_detections(detections, video_frames[:30])
    
    # 5. 初始化球队分配器
    print("\n[步骤 5] 初始化球队分配器...")
    if not os.path.exists(SIGLIP_MODEL_PATH):
        print(f"⚠ 警告: SigLIP模型路径不存在: {SIGLIP_MODEL_PATH}")
        print("  将跳过球队分配测试")
        team_assigner = None
    else:
        team_assigner = TeamAssigner(device=DEVICE, batch_size=32, local_model_path=SIGLIP_MODEL_PATH, verbose=False)
        print("✓ 球队分配器初始化完成")
    
    # 6. 测试完整追踪流程
    print("\n[步骤 6] 测试完整追踪流程...")
    tracks = tracker.get_object_tracks(
        video_frames[:100],  # 只测试前100帧
        read_from_stub=False,
        stub_path=None,  # 不使用缓存
        team_assigner=team_assigner
    )
    print("✓ 追踪完成")
    
    # 7. 分析追踪结果
    frame_track_stats, track_id_frames = analyze_tracks(tracks, video_frames[:100])
    
    # 8. 对比检测和追踪
    compare_detection_tracking(detections[:10], tracks, video_frames[:10])
    
    print("\n" + "="*80)
    print("测试完成")
    print("="*80)

if __name__ == '__main__':
    main()




