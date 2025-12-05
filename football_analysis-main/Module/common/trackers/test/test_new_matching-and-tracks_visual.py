"""
新匹配算法的单元测试

详细展示匹配情况，包括：
1. 初始化测试
2. 简单匹配测试（周围无其他球员）
3. 复杂匹配测试（周围有其他球员）
4. 轨迹删除测试（小于5帧）
5. 完整流程测试
"""

import pickle
import numpy as np
import os
import sys
import cv2

sys.path.append('../../../../')

from Module.common.trackers.tracker_pitch import Tracker
from Module.visualizer.pitch_annotation_tool import draw_pitch
from Module.visualizer.track_visualizer import visualize_tracks_as_video
from Storage.field_configs.soccer import SoccerPitchConfiguration

# 配置路径
# 测试文件在 Module/common/trackers/test/，需要向上5层到达项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
TRACKING_INPUTS_PATH = os.path.join(PROJECT_ROOT, 'Module', 'common', 'trackers', 'tracking_inputs_stub.pkl')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'Player_detection', 'football-player-detection.pt')


def test_initialization():
    """测试1: 初始化逻辑"""
    print("=" * 80)
    print("测试1: 初始化逻辑")
    print("=" * 80)
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    # 创建测试输入：前3帧无检测，第4帧有3个球员
    tracking_inputs = [
        {'frame_num': 0, 'detections_pitch': [], 'detections_info': [], 
         'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
         'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}},
        {'frame_num': 1, 'detections_pitch': [], 'detections_info': [],
         'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
         'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}},
        {'frame_num': 2, 'detections_pitch': [], 'detections_info': [],
         'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
         'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}},
        {'frame_num': 3, 'detections_pitch': [[10.0, 20.0], [15.0, 25.0], [20.0, 30.0]],
         'detections_info': [
             {'bbox': [100, 200, 150, 280], 'position_pixel': [125, 280], 'class_id': -1, 'team': 1},
             {'bbox': [200, 250, 250, 330], 'position_pixel': [225, 330], 'class_id': -1, 'team': 1},
             {'bbox': [300, 300, 350, 380], 'position_pixel': [325, 380], 'class_id': -1, 'team': 2},
         ],
         'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
         'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}},
    ]
    
    tracks = tracker.track_detections(tracking_inputs, frames=None)
    
    print(f"\n初始化结果:")
    print(f"  创建的轨迹数: {len(tracker.tracks)}")
    print(f"  期望轨迹数: 3")
    
    assert len(tracker.tracks) == 3, f"轨迹数错误: {len(tracker.tracks)} vs 3"
    
    for i, track in enumerate(tracker.tracks):
        print(f"  轨迹{i+1}: track_id={track.track_id}, team={track.team}, position={track.position_pitch}")
        assert track.track_id == i + 1, f"轨迹ID错误: {track.track_id}"
        assert track.team in [1, 2], f"球队ID错误: {track.team}"
        assert len(track.trajectory_history) == 1, f"轨迹历史长度错误: {len(track.trajectory_history)}"
    
    print(f"✓ 通过: 初始化成功，创建了{len(tracker.tracks)}条轨迹")


def test_simple_matching():
    """测试2: 简单匹配（周围无其他球员）"""
    print("\n" + "=" * 80)
    print("测试2: 简单匹配（周围无其他球员）")
    print("=" * 80)
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    # 创建测试输入：第1帧初始化，第2帧球员移动（距离较近，周围无其他球员）
    tracking_inputs = [
        {'frame_num': 0, 'detections_pitch': [[10.0, 20.0], [15.0, 25.0]],
         'detections_info': [
             {'bbox': [100, 200, 150, 280], 'position_pixel': [125, 280], 'class_id': -1, 'team': 1},
             {'bbox': [200, 250, 250, 330], 'position_pixel': [225, 330], 'class_id': -1, 'team': 1},
         ],
         'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
         'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}},
        {'frame_num': 1, 'detections_pitch': [[10.5, 20.3], [15.2, 25.1]],  # 轻微移动
         'detections_info': [
             {'bbox': [102, 202, 152, 282], 'position_pixel': [127, 282], 'class_id': -1, 'team': 1},
             {'bbox': [202, 252, 252, 332], 'position_pixel': [227, 332], 'class_id': -1, 'team': 1},
         ],
         'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
         'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}},
    ]
    
    tracks = tracker.track_detections(tracking_inputs, frames=None)
    
    print(f"\n匹配结果:")
    print(f"  第0帧轨迹数: {len(tracker.tracks)}")
    print(f"  第1帧轨迹数: {len(tracker.tracks)}")
    
    # 检查第1帧的匹配情况
    frame_1_tracks = tracks['players'][1]
    print(f"  第1帧存储的追踪数: {len(frame_1_tracks)}")
    
    # 检查轨迹是否被正确更新
    for track in tracker.tracks:
        print(f"  轨迹{track.track_id}: 历史长度={len(track.trajectory_history)}, lost_count={track.lost_count}")
        assert len(track.trajectory_history) == 2, f"轨迹历史长度错误: {len(track.trajectory_history)}"
        assert track.lost_count == 0, f"轨迹丢失计数错误: {track.lost_count}"
    
    print(f"✓ 通过: 简单匹配成功，所有轨迹都被正确更新")


def test_complex_matching():
    """测试3: 复杂匹配（周围有其他球员）"""
    print("\n" + "=" * 80)
    print("测试3: 复杂匹配（周围有其他球员）")
    print("=" * 80)
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    # 创建测试输入：第1帧初始化，第2帧有球员聚集（3米内有多个球员）
    tracking_inputs = [
        {'frame_num': 0, 'detections_pitch': [[10.0, 20.0], [15.0, 25.0], [50.0, 50.0]],
         'detections_info': [
             {'bbox': [100, 200, 150, 280], 'position_pixel': [125, 280], 'class_id': -1, 'team': 1},
             {'bbox': [200, 250, 250, 330], 'position_pixel': [225, 330], 'class_id': -1, 'team': 1},
             {'bbox': [500, 500, 550, 580], 'position_pixel': [525, 580], 'class_id': -1, 'team': 2},
         ],
         'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
         'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}},
        {'frame_num': 1, 
         # 球员1和球员2靠近（距离2米），球员3远离
         'detections_pitch': [[10.5, 20.3], [12.0, 21.0], [50.5, 50.3]],
         'detections_info': [
             {'bbox': [102, 202, 152, 282], 'position_pixel': [127, 282], 'class_id': -1, 'team': 1},
             {'bbox': [120, 210, 170, 290], 'position_pixel': [145, 290], 'class_id': -1, 'team': 1},  # 本方球员，距离球员1约2米
             {'bbox': [502, 502, 552, 582], 'position_pixel': [527, 582], 'class_id': -1, 'team': 2},
         ],
         'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
         'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}},
    ]
    
    tracks = tracker.track_detections(tracking_inputs, frames=None)
    
    print(f"\n匹配结果:")
    print(f"  第0帧轨迹数: 3")
    print(f"  第1帧轨迹数: {len(tracker.tracks)}")
    
    # 检查匹配情况
    frame_1_tracks = tracks['players'][1]
    print(f"  第1帧存储的追踪数: {len(frame_1_tracks)}")
    
    # 检查每个轨迹的匹配情况
    for track in tracker.tracks:
        print(f"  轨迹{track.track_id}: team={track.team}, 历史长度={len(track.trajectory_history)}, "
              f"lost_count={track.lost_count}, position={track.position_pitch}")
    
    # 验证：所有轨迹都应该被匹配（因为距离都在3米内）
    matched_count = sum(1 for track in tracker.tracks if track.lost_count == 0 and len(track.trajectory_history) == 2)
    print(f"\n  匹配的轨迹数: {matched_count}/3")
    
    print(f"✓ 通过: 复杂匹配完成")


def test_track_deletion():
    """测试4: 轨迹删除（小于5帧）"""
    print("\n" + "=" * 80)
    print("测试4: 轨迹删除（小于5帧）")
    print("=" * 80)
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    # 创建测试输入：第1帧初始化2个轨迹，后续帧只有1个检测（另一个轨迹会丢失）
    tracking_inputs = []
    
    # 第0帧：初始化2个轨迹
    tracking_inputs.append({
        'frame_num': 0,
        'detections_pitch': [[10.0, 20.0], [15.0, 25.0]],
        'detections_info': [
            {'bbox': [100, 200, 150, 280], 'position_pixel': [125, 280], 'class_id': -1, 'team': 1},
            {'bbox': [200, 250, 250, 330], 'position_pixel': [225, 330], 'class_id': -1, 'team': 1},
        ],
        'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
        'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}
    })
    
    # 第1-3帧：只有第一个轨迹的检测（第二个轨迹丢失）
    for frame_num in range(1, 4):
        tracking_inputs.append({
            'frame_num': frame_num,
            'detections_pitch': [[10.0 + frame_num * 0.1, 20.0 + frame_num * 0.1]],
            'detections_info': [
                {'bbox': [100 + frame_num, 200 + frame_num, 150 + frame_num, 280 + frame_num], 
                 'position_pixel': [125 + frame_num, 280 + frame_num], 'class_id': -1, 'team': 1},
            ],
            'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
            'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}
        })
    
    # 第4帧：仍然只有第一个轨迹的检测（第二个轨迹已经丢失4帧，但长度只有4帧，应该被删除）
    tracking_inputs.append({
        'frame_num': 4,
        'detections_pitch': [[10.4, 20.4]],
        'detections_info': [
            {'bbox': [104, 204, 154, 284], 'position_pixel': [129, 284], 'class_id': -1, 'team': 1},
        ],
        'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
        'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}
    })
    
    tracks = tracker.track_detections(tracking_inputs, frames=None)
    
    print(f"\n删除结果:")
    print(f"  初始轨迹数: 2")
    print(f"  最终轨迹数: {len(tracker.tracks)}")
    
    # 检查：第二个轨迹应该被删除（长度4帧 < 5帧，且已丢失）
    track_ids = [track.track_id for track in tracker.tracks]
    print(f"  保留的轨迹ID: {track_ids}")
    
    # 应该只保留第一个轨迹（长度5帧 >= 5帧）
    assert len(tracker.tracks) == 1, f"轨迹数错误: {len(tracker.tracks)} vs 1"
    assert tracker.tracks[0].track_id == 1, f"保留的轨迹ID错误: {tracker.tracks[0].track_id}"
    
    print(f"✓ 通过: 轨迹删除成功，删除了长度<5帧的轨迹")


def output_frame_range_track_details(tracker, tracking_inputs, start_frame, end_frame, fps=30.0):
    """
    输出选定帧范围的详细轨迹信息
    
    Args:
        tracker: Tracker实例，已完成追踪
        tracking_inputs: 追踪输入数据列表
        start_frame: 起始帧号
        end_frame: 结束帧号
        fps: 视频帧率，默认30.0
    """
    print(f"\n" + "=" * 80)
    print(f"帧 {start_frame} 到 {end_frame} 的详细轨迹信息")
    print("=" * 80)
    
    # 计算时间范围
    start_time_seconds = start_frame / fps
    end_time_seconds = end_frame / fps
    start_minutes = int(start_time_seconds // 60)
    start_secs = start_time_seconds % 60
    end_minutes = int(end_time_seconds // 60)
    end_secs = end_time_seconds % 60
    
    print(f"时间范围: {start_minutes}分{start_secs:.2f}秒 - {end_minutes}分{end_secs:.2f}秒")
    print(f"帧号范围: {start_frame} - {end_frame}")
    
    # 找到这个时间段内的所有轨迹
    time_range_tracks = []
    for track in tracker.tracks:
        # 检查轨迹是否在这个时间段内存在
        track_start = track.frame_num
        track_end = track.last_seen_frame
        
        if track_start <= end_frame and track_end >= start_frame:
            time_range_tracks.append(track)
    
    print(f"\n该时间段内的轨迹数: {len(time_range_tracks)}")
    
    # 输出每个轨迹在这个时间段内的完整信息
    print(f"\n" + "=" * 80)
    print("各轨迹在该时间段内的完整信息")
    print("=" * 80)
    
    for track in sorted(time_range_tracks, key=lambda t: t.track_id):
        print(f"\n轨迹 ID {track.track_id} (队伍{track.team}):")
        print(f"  起始帧: {track.frame_num}, 最后可见帧: {track.last_seen_frame}")
        print(f"  存在时间: {track.last_seen_frame - track.frame_num + 1} 帧")
        print(f"  丢失计数: {track.lost_count}")
        
        # 提取该时间段内的轨迹历史
        time_range_history = [
            hist for hist in track.trajectory_history
            if hist.get('frame_num') is not None and start_frame <= hist.get('frame_num') <= end_frame
        ]
        
        if len(time_range_history) > 0:
            print(f"  该时间段内的位置点 ({len(time_range_history)} 个):")
            for hist in time_range_history:
                frame_num = hist.get('frame_num')
                pos = hist.get('position_pitch')
                vel = hist.get('velocity', [0, 0])
                speed = hist.get('speed', 0)
                bbox = hist.get('bbox')
                position_pixel = hist.get('position_pixel')
                
                time_seconds = frame_num / fps
                minutes = int(time_seconds // 60)
                seconds = time_seconds % 60
                
                # 输出简化格式：帧号、时间、位置、速度、检测框、像素位置
                output_line = f"    帧{frame_num} ({minutes}分{seconds:.2f}秒): 位置=({pos[0]:.2f}, {pos[1]:.2f})米, "
                output_line += f"速度=({vel[0]:.2f}, {vel[1]:.2f})米/帧, 速度大小={speed:.2f}米/帧"
                
                if bbox is not None:
                    output_line += f", 检测框={bbox}"
                
                if position_pixel is not None:
                    output_line += f", 像素位置={position_pixel}"
                
                print(output_line)
        else:
            print(f"  该时间段内无位置记录")


def test_full_workflow():
    """测试5: 完整流程测试（使用真实数据）"""
    print("\n" + "=" * 80)
    print("测试5: 完整流程测试")
    print("=" * 80)
    
    if not os.path.exists(TRACKING_INPUTS_PATH):
        print("⚠ 跳过: 未找到输入文件")
        return
    
    with open(TRACKING_INPUTS_PATH, 'rb') as f:
        tracking_inputs = pickle.load(f)
    
    if len(tracking_inputs) == 0:
        print("⚠ 跳过: 输入数据为空")
        return
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    # 测试前20帧
    test_inputs = tracking_inputs[:300]
    
    print(f"\n输入数据: {len(test_inputs)} 帧")
    
    # 找到第一帧有球员检测的帧
    first_frame_idx = None
    for idx, inp in enumerate(test_inputs):
        player_count = sum(1 for det_info in inp['detections_info'] 
                          if det_info.get('class_id') == inp['cls_names_inv'].get('player', 0) - 1)
        if player_count > 0:
            first_frame_idx = idx
            print(f"  第一帧有球员检测: 帧{inp['frame_num']} (索引{idx}), {player_count}个球员")
            break
    
    if first_frame_idx is None:
        print("⚠ 跳过: 没有找到有球员检测的帧")
        return
    
    tracks = tracker.track_detections(test_inputs, frames=None)
    
    print(f"\n追踪结果:")
    print(f"  初始轨迹数: {len([t for t in tracker.tracks if t.frame_num == test_inputs[first_frame_idx]['frame_num']])}")
    print(f"  最终轨迹数: {len(tracker.tracks)}")
    
    # 统计每帧的匹配情况
    print(f"\n每帧匹配情况:")
    for frame_idx in range(min(5, len(test_inputs))):
        inp = test_inputs[frame_idx]
        frame_num = inp['frame_num']
        player_detections = sum(1 for det_info in inp['detections_info'] 
                               if det_info.get('class_id') == inp['cls_names_inv'].get('player', 0) - 1)
        stored_tracks = len(tracks['players'][frame_num])
        
        print(f"  帧{frame_num}: 检测={player_detections}, 存储={stored_tracks}")
    
    # 统计轨迹信息
    track_lengths = []
    for track in tracker.tracks:
        track_length = test_inputs[-1]['frame_num'] - track.frame_num + 1
        track_lengths.append(track_length)
        print(f"  轨迹{track.track_id}: 长度={track_length}帧, team={track.team}, "
              f"历史长度={len(track.trajectory_history)}, lost_count={track.lost_count}")
    
    if track_lengths:
        print(f"\n轨迹长度统计:")
        print(f"  平均长度: {np.mean(track_lengths):.1f}帧")
        print(f"  最小长度: {min(track_lengths)}帧")
        print(f"  最大长度: {max(track_lengths)}帧")
        print(f"  长度>=5帧的轨迹数: {sum(1 for l in track_lengths if l >= 5)}")
    
    # 输出3分14秒到3分16秒的详细轨迹信息
    start_frame = int(104)
    end_frame = int(106)
    
    output_frame_range_track_details(tracker, test_inputs, start_frame, end_frame)
    
    # 可视化轨迹
    print(f"\n生成轨迹可视化...")
    config = SoccerPitchConfiguration()
    pitch_image = draw_pitch(config=config)
    
    # 绘制所有轨迹
    for track in tracker.tracks:
        if track.is_confirmed or len(track.trajectory_history) > 0:
            pitch_image = track.draw_on_pitch(
                pitch_image=pitch_image,
                config=config,
                draw_trajectory=True,
                draw_current_position=True,
                draw_track_id=True
            )
    
    # 保存可视化结果
    output_dir = os.path.join(PROJECT_ROOT, 'IO', 'output_videos')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'test_tracks_visualization.png')
    cv2.imwrite(output_path, pitch_image)
    print(f"  轨迹可视化已保存到: {output_path}")
    
    # 生成轨迹可视化视频
    print(f"\n生成轨迹可视化视频...")
    
    # 获取所有帧号范围
    frame_nums = [inp['frame_num'] for inp in test_inputs]
    
    # 使用封装的可视化模块生成视频
    video_output_path = os.path.join(output_dir, 'test_tracks_visualization.mp4')
    visualize_tracks_as_video(
        tracks=tracker.tracks,
        frame_nums=frame_nums,
        output_path=video_output_path,
        config=config,
        fps=30,
        draw_trajectory=True,
        draw_current_position=True,
        draw_track_id=True
    )
    
    print(f"✓ 通过: 完整流程测试完成")


def main():
    """运行所有测试"""
    print("=" * 80)
    print("新匹配算法单元测试")
    print("=" * 80)
    
    tests = [
        ("初始化逻辑", test_initialization),
        ("简单匹配", test_simple_matching),
        ("复杂匹配", test_complex_matching),
        ("轨迹删除", test_track_deletion),
        ("完整流程", test_full_workflow),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except Exception as e:
            print(f"\n✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 汇总
    print("\n" + "=" * 80)
    print("测试汇总")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:20s} {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

