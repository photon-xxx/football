"""
详细诊断匹配问题的测试模块
分析为什么轨迹没有被匹配到检测
"""

import sys
import os
import numpy as np
import pickle

# 添加项目根目录到路径
# 测试文件在 Module/common/trackers/test/，需要向上4层到达项目根目录
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from Module.common.trackers.tracker_pitch import Tracker, PitchTrack
from get_trackerinput import load_tracking_inputs, TRACKING_INPUTS_STUB_PATH


def diagnose_matching_process(tracker, tracking_inputs, start_frame=0, num_frames=5):
    """
    详细诊断匹配过程
    
    Args:
        tracker: Tracker实例
        tracking_inputs: 追踪输入数据
        start_frame: 开始帧索引
        num_frames: 要诊断的帧数
    """
    print("=" * 80)
    print("匹配过程详细诊断")
    print("=" * 80)
    
    # 找到第一帧有球员检测的帧
    first_frame_idx = None
    for idx, inp in enumerate(tracking_inputs):
        player_count = sum(1 for det_info in inp['detections_info'] 
                          if det_info.get('class_id') == inp['cls_names_inv'].get('player', 0) - 1)
        if player_count > 0:
            first_frame_idx = idx
            break
    
    if first_frame_idx is None:
        print("⚠ 没有找到有球员检测的帧")
        return
    
    print(f"\n第一帧有球员检测: 帧索引{first_frame_idx}, 帧号{tracking_inputs[first_frame_idx]['frame_num']}")
    
    # 先运行追踪到初始化帧
    test_inputs = tracking_inputs[:first_frame_idx + 1]
    tracker.track_detections(test_inputs, frames=None)
    
    print(f"\n初始化后轨迹数: {len(tracker.tracks)}")
    for track in tracker.tracks[:5]:  # 只显示前5条
        print(f"  轨迹{track.track_id}: team={track.team}, position={track.position_pitch}, "
              f"历史长度={len(track.trajectory_history)}")
    
    # 诊断后续帧的匹配过程
    for frame_offset in range(1, min(num_frames + 1, len(tracking_inputs) - first_frame_idx)):
        frame_idx = first_frame_idx + frame_offset
        if frame_idx >= len(tracking_inputs):
            break
            
        inp = tracking_inputs[frame_idx]
        frame_num = inp['frame_num']
        detections_pitch = inp['detections_pitch']
        detections_info = inp['detections_info']
        cls_names_inv = inp['cls_names_inv']
        
        # 筛选球员检测
        player_detections = []
        player_detections_pitch = []
        for det_idx, det_info in enumerate(detections_info):
            if det_info.get('class_id') == cls_names_inv.get('player', 0) - 1:
                player_detections.append((det_idx, det_info))
                player_detections_pitch.append(detections_pitch[det_idx])
        
        print(f"\n{'=' * 80}")
        print(f"帧{frame_num} (索引{frame_idx}) 匹配诊断")
        print(f"{'=' * 80}")
        print(f"检测数: {len(player_detections)}")
        print(f"当前轨迹数: {len(tracker.tracks)}")
        
        # 预测所有轨迹
        for track in tracker.tracks:
            track.predict()
        
        # 详细分析每个检测的匹配情况
        print(f"\n【检测详情】")
        for det_idx, det_info in player_detections:
            det_pitch = detections_pitch[det_idx]
            det_team = det_info.get('team')
            
            print(f"\n  检测{det_idx}:")
            print(f"    位置: {det_pitch}")
            print(f"    球队: {det_team}")
            print(f"    class_id: {det_info.get('class_id')}")
            
            # 检查是否有team信息
            if det_team is None:
                print(f"    ⚠ 警告: 检测没有team信息，将被跳过匹配")
                continue
            
            # 找到周围3米内的其他检测（注意：position_pitch单位是cm，3米=300cm）
            nearby_detections = []
            threshold_cm = 3.0 * 100.0  # 3米 = 300cm
            for other_det_idx, other_det_pitch in enumerate(detections_pitch):
                if other_det_idx == det_idx:
                    continue
                distance = np.linalg.norm(np.array(det_pitch) - np.array(other_det_pitch))
                if distance <= threshold_cm:
                    other_det_info = detections_info[other_det_idx]
                    nearby_detections.append({
                        'idx': other_det_idx,
                        'position': other_det_pitch,
                        'team': other_det_info.get('team'),
                        'distance': distance
                    })
            
            print(f"    周围3米内的其他检测数: {len(nearby_detections)}")
            if nearby_detections:
                for nearby in nearby_detections[:3]:  # 只显示前3个
                    print(f"      检测{nearby['idx']}: 距离={nearby['distance']:.2f}cm ({nearby['distance']/100:.2f}m), team={nearby['team']}")
            
            # 找到周围3米内的轨迹（注意：position_pitch单位是cm，3米=300cm）
            candidate_tracks = []
            threshold_cm = 3.0 * 100.0  # 3米 = 300cm
            for track_idx, track in enumerate(tracker.tracks):
                # 获取轨迹的最新位置
                if len(track.trajectory_history) > 0:
                    latest_position = track.trajectory_history[-1]['position_pitch']
                else:
                    latest_position = track.position_pitch
                
                distance = np.linalg.norm(np.array(det_pitch) - np.array(latest_position))
                
                if distance <= threshold_cm:
                    candidate_tracks.append({
                        'track_idx': track_idx,
                        'track_id': track.track_id,
                        'team': track.team,
                        'position': latest_position,
                        'distance': distance,
                        'history_len': len(track.trajectory_history),
                        'lost_count': track.lost_count
                    })
            
            print(f"    周围3米内的轨迹数: {len(candidate_tracks)}")
            if candidate_tracks:
                # 按距离排序
                candidate_tracks.sort(key=lambda x: x['distance'])
                for cand in candidate_tracks[:5]:  # 只显示前5个
                    print(f"      轨迹{cand['track_id']}: 距离={cand['distance']:.2f}cm ({cand['distance']/100:.2f}m), team={cand['team']}, "
                          f"历史长度={cand['history_len']}, lost_count={cand['lost_count']}")
            
            # 分析匹配逻辑
            if len(nearby_detections) == 0:
                print(f"    → 情况1: 周围3米内没有其他球员")
                if len(candidate_tracks) > 0:
                    best = candidate_tracks[0]
                    print(f"    → 应该匹配到轨迹{best['track_id']} (距离={best['distance']:.2f}cm = {best['distance']/100:.2f}m)")
                else:
                    print(f"    → ⚠ 没有候选轨迹，将创建新轨迹")
            else:
                same_team_detections = [d for d in nearby_detections if d['team'] == det_team]
                print(f"    → 情况2: 周围3米内有其他球员")
                print(f"      同队检测数: {len(same_team_detections)}")
                
                if len(same_team_detections) > 0:
                    same_team_tracks = [t for t in candidate_tracks if t['team'] == det_team]
                    print(f"      同队轨迹数: {len(same_team_tracks)}")
                    if len(same_team_tracks) > 0:
                        best = same_team_tracks[0]
                        print(f"    → 应该匹配到同队轨迹{best['track_id']} (距离={best['distance']:.2f}cm = {best['distance']/100:.2f}m)")
                    else:
                        print(f"    → ⚠ 没有同队轨迹，将创建新轨迹")
                else:
                    print(f"    → 周围只有对方球员，忽略对方球员")
                    if len(candidate_tracks) > 0:
                        best = candidate_tracks[0]
                        print(f"    → 应该匹配到轨迹{best['track_id']} (距离={best['distance']:.2f}cm = {best['distance']/100:.2f}m)")
                    else:
                        print(f"    → ⚠ 没有候选轨迹，将创建新轨迹")
        
        # 执行实际匹配
        print(f"\n【执行匹配】")
        matches, unmatched_tracks, unmatched_dets = tracker._match_tracks_detections(
            tracker.tracks, detections_pitch, detections_info, threshold=3.0
        )
        
        print(f"匹配结果:")
        print(f"  匹配数: {len(matches)}")
        print(f"  未匹配轨迹数: {len(unmatched_tracks)}")
        print(f"  未匹配检测数: {len(unmatched_dets)}")
        
        if matches:
            print(f"\n  匹配详情:")
            for track_idx, det_idx in matches[:10]:  # 只显示前10个
                track = tracker.tracks[track_idx]
                det_info = detections_info[det_idx]
                det_pitch = detections_pitch[det_idx]
                
                if len(track.trajectory_history) > 0:
                    track_pos = track.trajectory_history[-1]['position_pitch']
                else:
                    track_pos = track.position_pitch
                
                distance = np.linalg.norm(np.array(det_pitch) - np.array(track_pos))
                print(f"    轨迹{track.track_id} ← 检测{det_idx}: 距离={distance:.2f}cm ({distance/100:.2f}m), "
                      f"track_team={track.team}, det_team={det_info.get('team')}")
        
        if unmatched_dets:
            print(f"\n  未匹配的检测:")
            for det_idx in unmatched_dets[:10]:  # 只显示前10个
                det_info = detections_info[det_idx]
                det_pitch = detections_pitch[det_idx]
                print(f"    检测{det_idx}: position={det_pitch}, team={det_info.get('team')}, "
                      f"class_id={det_info.get('class_id')}")
        
        if unmatched_tracks:
            print(f"\n  未匹配的轨迹:")
            for track_idx in unmatched_tracks[:10]:  # 只显示前10个
                track = tracker.tracks[track_idx]
                if len(track.trajectory_history) > 0:
                    track_pos = track.trajectory_history[-1]['position_pitch']
                else:
                    track_pos = track.position_pitch
                print(f"    轨迹{track.track_id}: position={track_pos}, team={track.team}, "
                      f"历史长度={len(track.trajectory_history)}, lost_count={track.lost_count}")
        
        # 更新轨迹
        for track_idx, det_idx in matches:
            track = tracker.tracks[track_idx]
            det_info = detections_info[det_idx]
            track.update(
                detections_pitch[det_idx],
                bbox=det_info['bbox'],
                position_pixel=det_info['position_pixel'],
                frame_num=frame_num,
                team=det_info.get('team'),
                team_color=det_info.get('team_color')
            )
        
        for track_idx in unmatched_tracks:
            tracker.tracks[track_idx].mark_missed()
        
        # 统计新创建的轨迹
        tracks_before = len(tracker.tracks)
        for det_idx in unmatched_dets:
            det_info = detections_info[det_idx]
            new_track = PitchTrack(
                detections_pitch[det_idx],
                det_info['class_id'],
                frame_num,
                track_id=tracker.next_id,
                dt=tracker.dt
            )
            new_track.bbox = det_info['bbox']
            new_track.position_pixel = det_info['position_pixel']
            if 'team' in det_info:
                new_track.team = det_info['team']
            if 'team_color' in det_info:
                new_track.team_color = det_info['team_color']
            new_track.update(
                detections_pitch[det_idx],
                bbox=det_info.get('bbox'),
                position_pixel=det_info.get('position_pixel'),
                frame_num=frame_num,
                team=det_info.get('team'),
                team_color=det_info.get('team_color')
            )
            tracker.tracks.append(new_track)
            tracker.next_id += 1
        
        tracks_after = len(tracker.tracks)
        new_tracks_count = tracks_after - tracks_before
        
        print(f"\n【帧后统计】")
        print(f"  匹配前轨迹数: {tracks_before}")
        print(f"  匹配后轨迹数: {tracks_after}")
        print(f"  新创建轨迹数: {new_tracks_count}")
        print(f"  匹配率: {len(matches)}/{len(player_detections)} = {len(matches)/len(player_detections)*100:.1f}%" if len(player_detections) > 0 else "  匹配率: N/A")


def main():
    """主函数"""
    # 加载追踪输入数据
    tracking_inputs = load_tracking_inputs(TRACKING_INPUTS_STUB_PATH)
    if tracking_inputs is None:
        print("⚠ 无法加载追踪输入数据")
        return
    
    print(f"加载追踪输入数据: {len(tracking_inputs)} 帧")
    
    # 创建追踪器
    # 注意：设置 sampling_rate >= video_fps 会自动禁用采样
    # 配置路径
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
    MODEL_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'Player_detection', 'football-player-detection.pt')
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,  # 等于 video_fps，自动禁用采样
        video_fps=30.0
    )
    
    # 诊断匹配过程（只诊断从有球员检测开始的前3帧）
    diagnose_matching_process(tracker, tracking_inputs, start_frame=0, num_frames=3)


if __name__ == '__main__':
    main()

