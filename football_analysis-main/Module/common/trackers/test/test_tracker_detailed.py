"""
tracker_pitch 详细调试测试

详细展示追踪过程中每帧的数据处理信息
"""

import pickle
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append('../../../../')

from Module.common.trackers.tracker_pitch import Tracker, PitchTrack

# 配置路径
# 测试文件在 Module/common/trackers/test/，需要向上5层到达项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
TRACKING_INPUTS_PATH = os.path.join(PROJECT_ROOT, 'Module', 'common', 'trackers', 'tracking_inputs_stub.pkl')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'Player_detection', 'football-player-detection.pt')


def test_tracking_detailed():
    """详细测试追踪过程，展示每帧的数据处理"""
    print("=" * 80)
    print("tracker_pitch 详细追踪调试")
    print("=" * 80)
    
    # 加载追踪输入数据
    if not os.path.exists(TRACKING_INPUTS_PATH):
        print(f"✗ 未找到输入文件: {TRACKING_INPUTS_PATH}")
        return
    
    with open(TRACKING_INPUTS_PATH, 'rb') as f:
        tracking_inputs = pickle.load(f)
    
    if len(tracking_inputs) == 0:
        print("✗ 输入数据为空")
        return
    
    print(f"\n加载完整缓存数据: {len(tracking_inputs)} 帧\n")
    
    # 创建追踪器
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    # 初始化输出结构
    tracks = {
        "players": [],
        "referees": [],
        "ball": []
    }
    
    # 初始化球检测（空）
    for _ in tracking_inputs:
        tracks["ball"].append({1: {"bbox": [np.nan, np.nan, np.nan, np.nan]}})
    
    # 详细追踪前50帧（或找到有检测的帧）
    test_frames = 50
    frames_with_detections = []
    
    for i, inp in enumerate(tracking_inputs[:test_frames]):
        if len(inp['detections_pitch']) > 0:
            frames_with_detections.append(i)
    
    print(f"前{test_frames}帧中有检测的帧: {len(frames_with_detections)} 帧")
    print(f"帧索引: {frames_with_detections[:10]}...\n")
    
    # 选择前几个有检测的帧进行详细测试
    detailed_frames = frames_with_detections[:5] if len(frames_with_detections) >= 5 else frames_with_detections
    
    print("=" * 80)
    print("开始详细追踪测试")
    print("=" * 80)
    
    for frame_idx, tracking_input in enumerate(tracking_inputs):
        frame_num = tracking_input['frame_num']
        detections_pitch = tracking_input['detections_pitch']
        detections_info = tracking_input['detections_info']
        cls_names = tracking_input.get('cls_names', {0: 'player', 1: 'referee', 2: 'ball'})
        cls_names_inv = tracking_input.get('cls_names_inv', {v: k for k, v in cls_names.items()})
        
        # 初始化该帧的字典
        tracks["players"].append({})
        tracks["referees"].append({})
        
        # 详细输出（仅对有检测的帧或前几帧）
        is_detailed = frame_idx in detailed_frames or frame_idx < 5
        
        if is_detailed:
            print(f"\n{'='*80}")
            print(f"帧 {frame_num} (索引 {frame_idx})")
            print(f"{'='*80}")
            print(f"输入检测数: {len(detections_pitch)}")
            
            if len(detections_pitch) > 0:
                print(f"\n检测详情:")
                for i, (det_pitch, det_info) in enumerate(zip(detections_pitch[:5], detections_info[:5])):
                    class_id = det_info.get('class_id', 'N/A')
                    team = det_info.get('team', 'N/A')
                    print(f"  检测{i}: class_id={class_id}, team={team}, position_pitch={det_pitch}")
                    print(f"    cls_names_inv['player']={cls_names_inv.get('player', 'N/A')}")
                    print(f"    匹配条件: class_id == cls_names_inv['player'] - 1 ? {class_id} == {cls_names_inv.get('player', 'N/A')} - 1")
        
        # 处理球员和裁判（使用场地坐标追踪）
        if len(detections_pitch) > 0:
            # 1. 预测所有轨迹的位置
            for track in tracker.tracks:
                track.predict()
            
            if is_detailed:
                print(f"\n当前轨迹数: {len(tracker.tracks)}")
                if len(tracker.tracks) > 0:
                    sample_track = tracker.tracks[0]
                    print(f"  示例轨迹: track_id={sample_track.track_id}, class_id={sample_track.class_id}, "
                          f"lost_count={sample_track.lost_count}, is_confirmed={sample_track.is_confirmed}")
                    print(f"    position_pitch={sample_track.position_pitch}, bbox={sample_track.bbox}")
            
            # 2. 匹配轨迹和检测
            matches, unmatched_tracks, unmatched_dets = tracker._match_tracks_detections(
                tracker.tracks, detections_pitch, detections_info
            )
            
            if is_detailed:
                print(f"\n匹配结果:")
                print(f"  匹配数: {len(matches)}")
                print(f"  未匹配轨迹数: {len(unmatched_tracks)}")
                print(f"  未匹配检测数: {len(unmatched_dets)}")
                
                if len(matches) > 0:
                    print(f"  匹配详情 (前3个):")
                    for match_idx, (track_idx, det_idx) in enumerate(matches[:3]):
                        track = tracker.tracks[track_idx]
                        det_info = detections_info[det_idx]
                        distance = np.linalg.norm(np.array(track.position_pitch) - np.array(detections_pitch[det_idx]))
                        print(f"    匹配{match_idx}: track_id={track.track_id} <-> 检测{det_idx}, "
                              f"距离={distance:.2f}米, class_id={det_info.get('class_id')}")
            
            # 3. 更新匹配的轨迹
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
            
            # 4. 标记未匹配的轨迹为丢失
            for track_idx in unmatched_tracks:
                tracker.tracks[track_idx].mark_missed()
            
            # 5. 为未匹配的检测创建新轨迹
            new_tracks_count = 0
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
                new_tracks_count += 1
            
            if is_detailed and new_tracks_count > 0:
                print(f"\n新建轨迹数: {new_tracks_count}")
                if new_tracks_count > 0:
                    last_track = tracker.tracks[-1]
                    print(f"  最后新建轨迹: track_id={last_track.track_id}, class_id={last_track.class_id}, "
                          f"is_confirmed={last_track.is_confirmed}, hit_streak={last_track.hit_streak}")
            
            # 6. 删除丢失时间过长的轨迹
            before_remove = len(tracker.tracks)
            tracker.tracks = [t for t in tracker.tracks if t.lost_count <= tracker.track_buffer]
            removed_count = before_remove - len(tracker.tracks)
            
            if is_detailed and removed_count > 0:
                print(f"\n删除丢失轨迹数: {removed_count}")
        
        # 7. 存储追踪结果
        stored_count = 0
        for track in tracker.tracks:
            if track.lost_count == 0:
                bbox = track.bbox
                if bbox is not None:
                    cls_id = track.class_id
                    track_id = track.track_id
                    
                    # 检查匹配条件
                    player_class_id_expected = cls_names_inv.get('player', 0) - 1
                    is_player_match = (cls_id == player_class_id_expected)
                    
                    if is_detailed and stored_count < 3:
                        print(f"\n存储检查 (track_id={track_id}):")
                        print(f"  cls_id={cls_id}, cls_names_inv['player']={cls_names_inv.get('player')}")
                        print(f"  期望值: cls_names_inv['player'] - 1 = {player_class_id_expected}")
                        print(f"  匹配: {is_player_match}")
                        print(f"  lost_count={track.lost_count}, bbox={bbox is not None}")
                    
                    # 根据类别ID判断对象类型
                    if cls_id == cls_names_inv.get('player', 0) - 1:
                        track_data = {
                            "bbox": bbox,
                            "position_pitch": track.position_pitch.tolist() if isinstance(track.position_pitch, np.ndarray) else track.position_pitch
                        }
                        if track.position_pixel is not None:
                            track_data['position_pixel'] = track.position_pixel
                        if track.team is not None:
                            track_data['team'] = track.team
                        if track.team_color is not None:
                            track_data['team_color'] = track.team_color
                        tracks["players"][frame_num][track_id] = track_data
                        stored_count += 1
                    elif cls_id == cls_names_inv.get('referee', 1) - 1:
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}
        
        if is_detailed:
            print(f"\n存储结果: {stored_count} 个球员追踪")
            print(f"当前帧输出: {len(tracks['players'][frame_num])} 个追踪")
        
        # 每30帧输出一次统计
        if frame_num % 30 == 0 and frame_num > 0:
            active_tracks = sum(1 for t in tracker.tracks if t.lost_count == 0)
            confirmed_tracks = sum(1 for t in tracker.tracks if t.is_confirmed)
            print(f"\n[统计 帧{frame_num}]: 总轨迹={len(tracker.tracks)}, "
                  f"活跃={active_tracks}, 已确认={confirmed_tracks}, "
                  f"存储数={sum(len(f) for f in tracks['players'][:frame_num+1])}")
    
    # 最终统计
    print(f"\n{'='*80}")
    print("最终统计")
    print(f"{'='*80}")
    total_tracks = sum(len(frame_tracks) for frame_tracks in tracks['players'])
    track_ids_all = set()
    for frame_tracks in tracks['players']:
        track_ids_all.update(frame_tracks.keys())
    
    active_tracks = sum(1 for t in tracker.tracks if t.lost_count == 0)
    confirmed_tracks = sum(1 for t in tracker.tracks if t.is_confirmed)
    
    print(f"总帧数: {len(tracks['players'])}")
    print(f"总追踪结果数: {total_tracks}")
    print(f"唯一追踪ID数: {len(track_ids_all)}")
    print(f"最终轨迹数: {len(tracker.tracks)}")
    print(f"活跃轨迹数: {active_tracks}")
    print(f"已确认轨迹数: {confirmed_tracks}")
    
    # 分析问题
    print(f"\n{'='*80}")
    print("问题分析")
    print(f"{'='*80}")
    
    if total_tracks == 0:
        print("⚠ 问题: 没有存储任何追踪结果")
        print("\n可能原因:")
        print("1. class_id 匹配失败")
        print("2. bbox 为 None")
        print("3. lost_count > 0")
        
        # 检查轨迹状态
        if len(tracker.tracks) > 0:
            sample_tracks = tracker.tracks[:5]
            print(f"\n前5个轨迹状态:")
            for t in sample_tracks:
                player_class_id_expected = tracking_inputs[0]['cls_names_inv'].get('player', 0) - 1
                print(f"  track_id={t.track_id}, class_id={t.class_id}, "
                      f"期望={player_class_id_expected}, 匹配={t.class_id == player_class_id_expected}, "
                      f"lost_count={t.lost_count}, bbox={t.bbox is not None}, "
                      f"is_confirmed={t.is_confirmed}")


if __name__ == '__main__':
    test_tracking_detailed()

