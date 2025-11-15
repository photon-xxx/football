"""
追踪问题详细分析测试

针对以下问题进行分析：
1. 匹配率低（20-35%）
2. 轨迹不稳定（13775个唯一ID）
3. 轨迹累积（626个轨迹对象）
4. 轨迹生命周期短
"""

import pickle
import numpy as np
import os
import sys
from collections import defaultdict, Counter

sys.path.append('../../../')

from Module.common.trackers.tracker_pitch import Tracker, PitchTrack

# 配置路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRACKING_INPUTS_PATH = os.path.join(PROJECT_ROOT, 'Module', 'common', 'trackers', 'tracking_inputs_stub.pkl')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'Player_detection', 'football-player-detection.pt')


def analyze_matching_rate():
    """分析匹配率问题"""
    print("=" * 80)
    print("问题1: 匹配率分析")
    print("=" * 80)
    
    with open(TRACKING_INPUTS_PATH, 'rb') as f:
        tracking_inputs = pickle.load(f)
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    matching_stats = {
        'total_frames': 0,
        'frames_with_detections': 0,
        'total_detections': 0,
        'total_matches': 0,
        'total_unmatched_tracks': 0,
        'total_unmatched_dets': 0,
        'match_rates': [],
        'distance_stats': []
    }
    
    # 分析前100帧
    for frame_idx, tracking_input in enumerate(tracking_inputs[:100]):
        frame_num = tracking_input['frame_num']
        detections_pitch = tracking_input['detections_pitch']
        detections_info = tracking_input['detections_info']
        
        matching_stats['total_frames'] += 1
        
        if len(detections_pitch) == 0:
            continue
        
        matching_stats['frames_with_detections'] += 1
        matching_stats['total_detections'] += len(detections_pitch)
        
        # 预测所有轨迹
        for track in tracker.tracks:
            track.predict()
        
        # 匹配
        if len(tracker.tracks) > 0:
            matches, unmatched_tracks, unmatched_dets = tracker._match_tracks_detections(
                tracker.tracks, detections_pitch, detections_info
            )
            
            matching_stats['total_matches'] += len(matches)
            matching_stats['total_unmatched_tracks'] += len(unmatched_tracks)
            matching_stats['total_unmatched_dets'] += len(unmatched_dets)
            
            match_rate = len(matches) / len(detections_pitch) if len(detections_pitch) > 0 else 0
            matching_stats['match_rates'].append(match_rate)
            
            # 计算匹配距离
            for track_idx, det_idx in matches:
                track = tracker.tracks[track_idx]
                det_pitch = detections_pitch[det_idx]
                distance = np.linalg.norm(np.array(track.position_pitch) - np.array(det_pitch))
                matching_stats['distance_stats'].append(distance)
            
            # 更新匹配的轨迹
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
            
            # 标记未匹配的轨迹
            for track_idx in unmatched_tracks:
                tracker.tracks[track_idx].mark_missed()
            
            # 创建新轨迹
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
                tracker.tracks.append(new_track)
                tracker.next_id += 1
        
        else:
            # 没有轨迹，创建新轨迹
            for det_idx, det_pitch in enumerate(detections_pitch):
                det_info = detections_info[det_idx]
                new_track = PitchTrack(
                    det_pitch,
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
                tracker.tracks.append(new_track)
                tracker.next_id += 1
        
        # 清理丢失的轨迹
        tracker.tracks = [t for t in tracker.tracks if t.lost_count <= tracker.track_buffer]
    
    # 输出统计
    print(f"\n总帧数: {matching_stats['total_frames']}")
    print(f"有检测的帧数: {matching_stats['frames_with_detections']}")
    print(f"总检测数: {matching_stats['total_detections']}")
    print(f"总匹配数: {matching_stats['total_matches']}")
    print(f"总未匹配轨迹数: {matching_stats['total_unmatched_tracks']}")
    print(f"总未匹配检测数: {matching_stats['total_unmatched_dets']}")
    
    if matching_stats['match_rates']:
        avg_match_rate = np.mean(matching_stats['match_rates'])
        print(f"\n平均匹配率: {avg_match_rate:.2%}")
        print(f"匹配率范围: {np.min(matching_stats['match_rates']):.2%} - {np.max(matching_stats['match_rates']):.2%}")
        print(f"匹配率中位数: {np.median(matching_stats['match_rates']):.2%}")
    
    if matching_stats['distance_stats']:
        distances = np.array(matching_stats['distance_stats'])
        print(f"\n匹配距离统计:")
        print(f"  平均距离: {np.mean(distances):.2f}米")
        print(f"  最小距离: {np.min(distances):.2f}米")
        print(f"  最大距离: {np.max(distances):.2f}米")
        print(f"  中位数距离: {np.median(distances):.2f}米")
        print(f"  距离阈值: {tracker.distance_threshold_primary}米")
        print(f"  超过阈值的匹配数: {np.sum(distances > tracker.distance_threshold_primary)}")
    
    return matching_stats


def analyze_track_lifecycle():
    """分析轨迹生命周期"""
    print("\n" + "=" * 80)
    print("问题2: 轨迹生命周期分析")
    print("=" * 80)
    
    with open(TRACKING_INPUTS_PATH, 'rb') as f:
        tracking_inputs = pickle.load(f)
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    track_lifecycle = {}  # {track_id: [first_frame, last_frame, hit_count, lost_count]}
    track_frames = defaultdict(list)  # {track_id: [frame_nums]}
    
    # 追踪前100帧
    for frame_idx, tracking_input in enumerate(tracking_inputs[:100]):
        frame_num = tracking_input['frame_num']
        detections_pitch = tracking_input['detections_pitch']
        detections_info = tracking_input['detections_info']
        
        if len(detections_pitch) == 0:
            # 所有轨迹标记为丢失
            for track in tracker.tracks:
                track.mark_missed()
                if track.track_id not in track_lifecycle:
                    track_lifecycle[track.track_id] = {
                        'first_frame': frame_num,
                        'last_frame': frame_num,
                        'hit_count': 0,
                        'lost_count': track.lost_count
                    }
                else:
                    track_lifecycle[track.track_id]['last_frame'] = frame_num
                    track_lifecycle[track.track_id]['lost_count'] = track.lost_count
            continue
        
        # 预测
        for track in tracker.tracks:
            track.predict()
        
        # 匹配
        if len(tracker.tracks) > 0:
            matches, unmatched_tracks, unmatched_dets = tracker._match_tracks_detections(
                tracker.tracks, detections_pitch, detections_info
            )
            
            # 更新匹配的轨迹
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
                
                track_id = track.track_id
                track_frames[track_id].append(frame_num)
                
                if track_id not in track_lifecycle:
                    track_lifecycle[track_id] = {
                        'first_frame': frame_num,
                        'last_frame': frame_num,
                        'hit_count': 1,
                        'lost_count': 0
                    }
                else:
                    track_lifecycle[track_id]['last_frame'] = frame_num
                    track_lifecycle[track_id]['hit_count'] += 1
                    track_lifecycle[track_id]['lost_count'] = track.lost_count
            
            # 标记未匹配的轨迹
            for track_idx in unmatched_tracks:
                track = tracker.tracks[track_idx]
                track.mark_missed()
                track_id = track.track_id
                if track_id not in track_lifecycle:
                    track_lifecycle[track_id] = {
                        'first_frame': frame_num,
                        'last_frame': frame_num,
                        'hit_count': 0,
                        'lost_count': track.lost_count
                    }
                else:
                    track_lifecycle[track_id]['last_frame'] = frame_num
                    track_lifecycle[track_id]['lost_count'] = track.lost_count
            
            # 创建新轨迹
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
                tracker.tracks.append(new_track)
                track_frames[tracker.next_id].append(frame_num)
                track_lifecycle[tracker.next_id] = {
                    'first_frame': frame_num,
                    'last_frame': frame_num,
                    'hit_count': 1,
                    'lost_count': 0
                }
                tracker.next_id += 1
        else:
            # 创建新轨迹
            for det_idx, det_pitch in enumerate(detections_pitch):
                det_info = detections_info[det_idx]
                new_track = PitchTrack(
                    det_pitch,
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
                tracker.tracks.append(new_track)
                track_frames[tracker.next_id].append(frame_num)
                track_lifecycle[tracker.next_id] = {
                    'first_frame': frame_num,
                    'last_frame': frame_num,
                    'hit_count': 1,
                    'lost_count': 0
                }
                tracker.next_id += 1
        
        # 清理丢失的轨迹
        removed_tracks = [t for t in tracker.tracks if t.lost_count > tracker.track_buffer]
        tracker.tracks = [t for t in tracker.tracks if t.lost_count <= tracker.track_buffer]
    
    # 计算生命周期统计
    lifetimes = []
    hit_counts = []
    lost_counts = []
    
    for track_id, lifecycle in track_lifecycle.items():
        lifetime = lifecycle['last_frame'] - lifecycle['first_frame'] + 1
        lifetimes.append(lifetime)
        hit_counts.append(lifecycle['hit_count'])
        lost_counts.append(lifecycle['lost_count'])
    
    lifetimes = np.array(lifetimes)
    hit_counts = np.array(hit_counts)
    lost_counts = np.array(lost_counts)
    
    print(f"\n总轨迹数: {len(track_lifecycle)}")
    print(f"\n生命周期统计:")
    print(f"  平均生命周期: {np.mean(lifetimes):.1f}帧")
    print(f"  中位数生命周期: {np.median(lifetimes):.1f}帧")
    print(f"  最小生命周期: {np.min(lifetimes)}帧")
    print(f"  最大生命周期: {np.max(lifetimes)}帧")
    print(f"  生命周期分布:")
    print(f"    1帧: {np.sum(lifetimes == 1)} ({np.sum(lifetimes == 1)/len(lifetimes)*100:.1f}%)")
    print(f"    2-5帧: {np.sum((lifetimes >= 2) & (lifetimes <= 5))} ({np.sum((lifetimes >= 2) & (lifetimes <= 5))/len(lifetimes)*100:.1f}%)")
    print(f"    6-10帧: {np.sum((lifetimes >= 6) & (lifetimes <= 10))} ({np.sum((lifetimes >= 6) & (lifetimes <= 10))/len(lifetimes)*100:.1f}%)")
    print(f"    11-20帧: {np.sum((lifetimes >= 11) & (lifetimes <= 20))} ({np.sum((lifetimes >= 11) & (lifetimes <= 20))/len(lifetimes)*100:.1f}%)")
    print(f"    >20帧: {np.sum(lifetimes > 20)} ({np.sum(lifetimes > 20)/len(lifetimes)*100:.1f}%)")
    
    print(f"\n命中次数统计:")
    print(f"  平均命中次数: {np.mean(hit_counts):.1f}")
    print(f"  中位数命中次数: {np.median(hit_counts):.1f}")
    print(f"  只命中1次的轨迹数: {np.sum(hit_counts == 1)} ({np.sum(hit_counts == 1)/len(hit_counts)*100:.1f}%)")
    
    print(f"\n丢失次数统计:")
    print(f"  平均丢失次数: {np.mean(lost_counts):.1f}")
    print(f"  中位数丢失次数: {np.median(lost_counts):.1f}")
    print(f"  从未丢失的轨迹数: {np.sum(lost_counts == 0)} ({np.sum(lost_counts == 0)/len(lost_counts)*100:.1f}%)")
    
    return track_lifecycle


def analyze_track_accumulation():
    """分析轨迹累积问题"""
    print("\n" + "=" * 80)
    print("问题3: 轨迹累积分析")
    print("=" * 80)
    
    with open(TRACKING_INPUTS_PATH, 'rb') as f:
        tracking_inputs = pickle.load(f)
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    accumulation_stats = {
        'frame_nums': [],
        'total_tracks': [],
        'active_tracks': [],
        'lost_tracks': [],
        'confirmed_tracks': [],
        'new_tracks_per_frame': []
    }
    
    # 追踪前100帧
    for frame_idx, tracking_input in enumerate(tracking_inputs[:100]):
        frame_num = tracking_input['frame_num']
        detections_pitch = tracking_input['detections_pitch']
        detections_info = tracking_input['detections_info']
        
        accumulation_stats['frame_nums'].append(frame_num)
        accumulation_stats['total_tracks'].append(len(tracker.tracks))
        
        active = sum(1 for t in tracker.tracks if t.lost_count == 0)
        lost = sum(1 for t in tracker.tracks if t.lost_count > 0)
        confirmed = sum(1 for t in tracker.tracks if t.is_confirmed)
        
        accumulation_stats['active_tracks'].append(active)
        accumulation_stats['lost_tracks'].append(lost)
        accumulation_stats['confirmed_tracks'].append(confirmed)
        
        tracks_before = len(tracker.tracks)
        
        if len(detections_pitch) > 0:
            # 预测
            for track in tracker.tracks:
                track.predict()
            
            # 匹配
            if len(tracker.tracks) > 0:
                matches, unmatched_tracks, unmatched_dets = tracker._match_tracks_detections(
                    tracker.tracks, detections_pitch, detections_info
                )
                
                # 更新匹配的轨迹
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
                
                # 标记未匹配的轨迹
                for track_idx in unmatched_tracks:
                    tracker.tracks[track_idx].mark_missed()
                
                # 创建新轨迹
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
                    tracker.tracks.append(new_track)
                    tracker.next_id += 1
            else:
                # 创建新轨迹
                for det_idx, det_pitch in enumerate(detections_pitch):
                    det_info = detections_info[det_idx]
                    new_track = PitchTrack(
                        det_pitch,
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
                    tracker.tracks.append(new_track)
                    tracker.next_id += 1
        
        tracks_after = len(tracker.tracks)
        new_tracks = tracks_after - tracks_before + len([t for t in tracker.tracks if t.lost_count > tracker.track_buffer])
        accumulation_stats['new_tracks_per_frame'].append(new_tracks)
        
        # 清理丢失的轨迹
        tracker.tracks = [t for t in tracker.tracks if t.lost_count <= tracker.track_buffer]
    
    # 输出统计
    print(f"\n轨迹数量变化:")
    print(f"  初始轨迹数: {accumulation_stats['total_tracks'][0]}")
    print(f"  最终轨迹数: {accumulation_stats['total_tracks'][-1]}")
    print(f"  最大轨迹数: {max(accumulation_stats['total_tracks'])}")
    print(f"  平均轨迹数: {np.mean(accumulation_stats['total_tracks']):.1f}")
    
    print(f"\n活跃轨迹数:")
    print(f"  平均活跃轨迹数: {np.mean(accumulation_stats['active_tracks']):.1f}")
    print(f"  最大活跃轨迹数: {max(accumulation_stats['active_tracks'])}")
    
    print(f"\n丢失轨迹数:")
    print(f"  平均丢失轨迹数: {np.mean(accumulation_stats['lost_tracks']):.1f}")
    print(f"  最大丢失轨迹数: {max(accumulation_stats['lost_tracks'])}")
    
    print(f"\n每帧新建轨迹数:")
    print(f"  平均新建轨迹数: {np.mean(accumulation_stats['new_tracks_per_frame']):.1f}")
    print(f"  最大新建轨迹数: {max(accumulation_stats['new_tracks_per_frame'])}")
    print(f"  总新建轨迹数: {sum(accumulation_stats['new_tracks_per_frame'])}")
    
    return accumulation_stats


def analyze_unmatched_reasons():
    """分析未匹配的原因"""
    print("\n" + "=" * 80)
    print("问题4: 未匹配原因分析")
    print("=" * 80)
    
    with open(TRACKING_INPUTS_PATH, 'rb') as f:
        tracking_inputs = pickle.load(f)
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    unmatched_stats = {
        'unmatched_track_distances': [],
        'unmatched_det_distances': [],
        'unmatched_track_lost_counts': [],
        'unmatched_track_ages': []
    }
    
    # 分析前50帧
    for frame_idx, tracking_input in enumerate(tracking_inputs[:50]):
        frame_num = tracking_input['frame_num']
        detections_pitch = tracking_input['detections_pitch']
        detections_info = tracking_input['detections_info']
        
        if len(detections_pitch) == 0 or len(tracker.tracks) == 0:
            continue
        
        # 预测
        for track in tracker.tracks:
            track.predict()
        
        # 计算距离矩阵
        distance_matrix = tracker._calculate_distance_matrix(tracker.tracks, detections_pitch)
        
        # 匹配
        matches, unmatched_tracks, unmatched_dets = tracker._match_tracks_detections(
            tracker.tracks, detections_pitch, detections_info
        )
        
        # 分析未匹配的轨迹
        for track_idx in unmatched_tracks:
            track = tracker.tracks[track_idx]
            unmatched_stats['unmatched_track_lost_counts'].append(track.lost_count)
            unmatched_stats['unmatched_track_ages'].append(frame_num - track.frame_num)
            
            # 找到最近的检测距离
            min_distance = np.min(distance_matrix[track_idx, :])
            unmatched_stats['unmatched_track_distances'].append(min_distance)
        
        # 分析未匹配的检测
        for det_idx in unmatched_dets:
            # 找到最近的轨迹距离
            min_distance = np.min(distance_matrix[:, det_idx])
            unmatched_stats['unmatched_det_distances'].append(min_distance)
        
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
            tracker.tracks.append(new_track)
            tracker.next_id += 1
        
        tracker.tracks = [t for t in tracker.tracks if t.lost_count <= tracker.track_buffer]
    
    # 输出统计
    if unmatched_stats['unmatched_track_distances']:
        distances = np.array(unmatched_stats['unmatched_track_distances'])
        print(f"\n未匹配轨迹到最近检测的距离:")
        print(f"  平均距离: {np.mean(distances):.2f}米")
        print(f"  中位数距离: {np.median(distances):.2f}米")
        print(f"  最小距离: {np.min(distances):.2f}米")
        print(f"  最大距离: {np.max(distances):.2f}米")
        print(f"  距离阈值: {tracker.distance_threshold_primary}米")
        print(f"  超过阈值的数量: {np.sum(distances > tracker.distance_threshold_primary)} ({np.sum(distances > tracker.distance_threshold_primary)/len(distances)*100:.1f}%)")
        print(f"  在阈值内的数量: {np.sum(distances <= tracker.distance_threshold_primary)} ({np.sum(distances <= tracker.distance_threshold_primary)/len(distances)*100:.1f}%)")
    
    if unmatched_stats['unmatched_det_distances']:
        distances = np.array(unmatched_stats['unmatched_det_distances'])
        print(f"\n未匹配检测到最近轨迹的距离:")
        print(f"  平均距离: {np.mean(distances):.2f}米")
        print(f"  中位数距离: {np.median(distances):.2f}米")
        print(f"  最小距离: {np.min(distances):.2f}米")
        print(f"  最大距离: {np.max(distances):.2f}米")
        print(f"  距离阈值: {tracker.distance_threshold_primary}米")
        print(f"  超过阈值的数量: {np.sum(distances > tracker.distance_threshold_primary)} ({np.sum(distances > tracker.distance_threshold_primary)/len(distances)*100:.1f}%)")
        print(f"  在阈值内的数量: {np.sum(distances <= tracker.distance_threshold_primary)} ({np.sum(distances <= tracker.distance_threshold_primary)/len(distances)*100:.1f}%)")
    
    if unmatched_stats['unmatched_track_lost_counts']:
        lost_counts = np.array(unmatched_stats['unmatched_track_lost_counts'])
        print(f"\n未匹配轨迹的丢失计数:")
        print(f"  平均丢失计数: {np.mean(lost_counts):.1f}")
        print(f"  中位数丢失计数: {np.median(lost_counts):.1f}")
        print(f"  从未丢失的轨迹数: {np.sum(lost_counts == 0)} ({np.sum(lost_counts == 0)/len(lost_counts)*100:.1f}%)")
        print(f"  丢失1-5次的轨迹数: {np.sum((lost_counts >= 1) & (lost_counts <= 5))} ({np.sum((lost_counts >= 1) & (lost_counts <= 5))/len(lost_counts)*100:.1f}%)")
        print(f"  丢失>5次的轨迹数: {np.sum(lost_counts > 5)} ({np.sum(lost_counts > 5)/len(lost_counts)*100:.1f}%)")
    
    if unmatched_stats['unmatched_track_ages']:
        ages = np.array(unmatched_stats['unmatched_track_ages'])
        print(f"\n未匹配轨迹的年龄:")
        print(f"  平均年龄: {np.mean(ages):.1f}帧")
        print(f"  中位数年龄: {np.median(ages):.1f}帧")
        print(f"  新轨迹(0-5帧): {np.sum(ages <= 5)} ({np.sum(ages <= 5)/len(ages)*100:.1f}%)")
        print(f"  中等轨迹(6-20帧): {np.sum((ages >= 6) & (ages <= 20))} ({np.sum((ages >= 6) & (ages <= 20))/len(ages)*100:.1f}%)")
        print(f"  老轨迹(>20帧): {np.sum(ages > 20)} ({np.sum(ages > 20)/len(ages)*100:.1f}%)")
    
    return unmatched_stats


def main():
    """运行所有分析"""
    print("=" * 80)
    print("追踪问题详细分析")
    print("=" * 80)
    
    # 检查文件是否存在
    if not os.path.exists(TRACKING_INPUTS_PATH):
        print(f"错误: 未找到输入文件 {TRACKING_INPUTS_PATH}")
        return
    
    # 运行各项分析
    matching_stats = analyze_matching_rate()
    track_lifecycle = analyze_track_lifecycle()
    accumulation_stats = analyze_track_accumulation()
    unmatched_stats = analyze_unmatched_reasons()
    
    # 总结
    print("\n" + "=" * 80)
    print("问题总结")
    print("=" * 80)
    
    if matching_stats['match_rates']:
        avg_match_rate = np.mean(matching_stats['match_rates'])
        print(f"\n1. 匹配率问题:")
        print(f"   平均匹配率: {avg_match_rate:.2%}")
        if avg_match_rate < 0.5:
            print(f"   ⚠ 匹配率过低，需要优化匹配算法或调整距离阈值")
    
    if track_lifecycle:
        lifetimes = [v['last_frame'] - v['first_frame'] + 1 for v in track_lifecycle.values()]
        avg_lifetime = np.mean(lifetimes)
        print(f"\n2. 轨迹生命周期问题:")
        print(f"   平均生命周期: {avg_lifetime:.1f}帧")
        if avg_lifetime < 5:
            print(f"   ⚠ 轨迹生命周期过短，说明轨迹不稳定")
    
    if accumulation_stats['total_tracks']:
        max_tracks = max(accumulation_stats['total_tracks'])
        print(f"\n3. 轨迹累积问题:")
        print(f"   最大轨迹数: {max_tracks}")
        if max_tracks > 100:
            print(f"   ⚠ 轨迹累积过多，需要优化清理策略")
    
    if unmatched_stats['unmatched_track_distances']:
        distances = np.array(unmatched_stats['unmatched_track_distances'])
        within_threshold = np.sum(distances <= 2.0)
        print(f"\n4. 未匹配原因:")
        print(f"   未匹配轨迹中，距离在阈值内的比例: {within_threshold/len(distances)*100:.1f}%")
        if within_threshold / len(distances) > 0.3:
            print(f"   ⚠ 大量轨迹距离在阈值内但仍未匹配，可能是匹配算法问题")


if __name__ == '__main__':
    main()

