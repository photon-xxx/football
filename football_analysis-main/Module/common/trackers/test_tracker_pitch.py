"""
tracker_pitch 单元测试模块

对每个函数和模块进行独立的单元测试
"""

import pickle
import numpy as np
import os
import sys

# 添加项目路径
sys.path.append('../../../')

from Module.common.trackers.tracker_pitch import PitchKalmanFilter, PitchTrack, Tracker

# 配置路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
TRACKING_INPUTS_PATH = os.path.join(PROJECT_ROOT, 'Module', 'common', 'trackers', 'tracking_inputs_stub.pkl')
MODEL_PATH = os.path.join(PROJECT_ROOT, 'Storage', 'models', 'Player_detection', 'football-player-detection.pt')


# ==================== PitchKalmanFilter 单元测试 ====================

def test_kalman_filter_init():
    """测试卡尔曼滤波器初始化"""
    print("\n[单元测试] PitchKalmanFilter.__init__")
    print("-" * 50)
    
    kf = PitchKalmanFilter(process_noise=0.1, measurement_noise=0.5, dt=1.0)
    
    assert kf.dt == 1.0, "dt设置错误"
    assert kf.F.shape == (4, 4), f"状态转移矩阵维度错误: {kf.F.shape}"
    assert kf.H.shape == (2, 4), f"观测矩阵维度错误: {kf.H.shape}"
    assert kf.Q.shape == (4, 4), f"过程噪声矩阵维度错误: {kf.Q.shape}"
    assert kf.R.shape == (2, 2), f"测量噪声矩阵维度错误: {kf.R.shape}"
    
    print("✓ 通过: 初始化参数正确")


def test_kalman_filter_init_method():
    """测试卡尔曼滤波器init方法"""
    print("\n[单元测试] PitchKalmanFilter.init")
    print("-" * 50)
    
    kf = PitchKalmanFilter(process_noise=0.1, measurement_noise=0.5, dt=1.0)
    position = [10.0, 20.0]
    
    mean, covariance = kf.init(position)
    
    assert mean.shape == (4,), f"状态向量维度错误: {mean.shape}"
    assert covariance.shape == (4, 4), f"协方差矩阵维度错误: {covariance.shape}"
    assert mean[0] == 10.0, f"x坐标错误: {mean[0]}"
    assert mean[1] == 20.0, f"y坐标错误: {mean[1]}"
    assert mean[2] == 0.0, f"vx初始值错误: {mean[2]}"
    assert mean[3] == 0.0, f"vy初始值错误: {mean[3]}"
    
    print(f"✓ 通过: mean={mean}, covariance.shape={covariance.shape}")


def test_kalman_filter_predict():
    """测试卡尔曼滤波器predict方法"""
    print("\n[单元测试] PitchKalmanFilter.predict")
    print("-" * 50)
    
    kf = PitchKalmanFilter(process_noise=0.1, measurement_noise=0.5, dt=1.0)
    mean = np.array([10.0, 20.0, 0.5, 0.3], dtype=np.float32)  # 有速度
    covariance = np.eye(4, dtype=np.float32) * 1.0
    
    mean_pred, covariance_pred = kf.predict(mean, covariance)
    
    assert mean_pred.shape == (4,), "预测状态向量维度错误"
    assert covariance_pred.shape == (4, 4), "预测协方差矩阵维度错误"
    # 验证位置预测: x = x + dt*vx, y = y + dt*vy
    assert abs(mean_pred[0] - (10.0 + 1.0 * 0.5)) < 0.001, f"x预测错误: {mean_pred[0]}"
    assert abs(mean_pred[1] - (20.0 + 1.0 * 0.3)) < 0.001, f"y预测错误: {mean_pred[1]}"
    
    print(f"✓ 通过: 预测位置从[{mean[0]:.1f}, {mean[1]:.1f}]到[{mean_pred[0]:.1f}, {mean_pred[1]:.1f}]")


def test_kalman_filter_update():
    """测试卡尔曼滤波器update方法"""
    print("\n[单元测试] PitchKalmanFilter.update")
    print("-" * 50)
    
    kf = PitchKalmanFilter(process_noise=0.1, measurement_noise=0.5, dt=1.0)
    mean = np.array([10.0, 20.0, 0.0, 0.0], dtype=np.float32)
    covariance = np.eye(4, dtype=np.float32) * 1.0
    measurement = [10.5, 20.3]
    
    mean_upd, covariance_upd = kf.update(mean, covariance, measurement)
    
    assert mean_upd.shape == (4,), "更新状态向量维度错误"
    assert covariance_upd.shape == (4, 4), "更新协方差矩阵维度错误"
    # 更新后的位置应该更接近测量值
    assert abs(mean_upd[0] - 10.5) < abs(mean[0] - 10.5), "x更新未接近测量值"
    assert abs(mean_upd[1] - 20.3) < abs(mean[1] - 20.3), "y更新未接近测量值"
    
    print(f"✓ 通过: 更新位置从[{mean[0]:.1f}, {mean[1]:.1f}]到[{mean_upd[0]:.2f}, {mean_upd[1]:.2f}] (测量值[{measurement[0]}, {measurement[1]}])")


# ==================== PitchTrack 单元测试 ====================

def test_pitch_track_init():
    """测试轨迹对象初始化"""
    print("\n[单元测试] PitchTrack.__init__")
    print("-" * 50)
    
    position = [10.0, 20.0]
    track = PitchTrack(position, class_id=0, frame_num=0, track_id=1, dt=1.0)
    
    assert track.position_pitch is not None, "位置未初始化"
    assert np.allclose(track.position_pitch, [10.0, 20.0]), f"位置错误: {track.position_pitch}"
    assert track.track_id == 1, f"轨迹ID错误: {track.track_id}"
    assert track.class_id == 0, f"类别ID错误: {track.class_id}"
    assert track.frame_num == 0, f"帧号错误: {track.frame_num}"
    assert track.lost_count == 0, f"丢失计数错误: {track.lost_count}"
    assert track.is_confirmed == False, "初始状态不应为confirmed"
    
    print(f"✓ 通过: track_id={track.track_id}, position={track.position_pitch}")


def test_pitch_track_predict():
    """测试轨迹预测"""
    print("\n[单元测试] PitchTrack.predict")
    print("-" * 50)
    
    position = [10.0, 20.0]
    track = PitchTrack(position, class_id=0, frame_num=0, track_id=1, dt=1.0)
    
    # 设置初始速度
    track.mean[2] = 0.5  # vx
    track.mean[3] = 0.3  # vy
    
    old_position = track.position_pitch.copy()
    track.predict()
    new_position = track.position_pitch
    
    assert new_position is not None, "预测后位置为空"
    # 位置应该根据速度更新
    assert not np.allclose(old_position, new_position), "位置未更新"
    
    print(f"✓ 通过: 位置从{old_position}预测到{new_position}")


def test_pitch_track_update():
    """测试轨迹更新"""
    print("\n[单元测试] PitchTrack.update")
    print("-" * 50)
    
    position = [10.0, 20.0]
    track = PitchTrack(position, class_id=0, frame_num=0, track_id=1, dt=1.0)
    
    new_position = [10.5, 20.3]
    bbox = [100, 200, 150, 280]
    position_pixel = [125, 280]
    team = 1
    
    track.update(new_position, bbox=bbox, position_pixel=position_pixel, frame_num=1, team=team)
    
    assert np.allclose(track.position_pitch, new_position), f"位置更新错误: {track.position_pitch}"
    assert track.bbox == bbox, f"bbox更新错误: {track.bbox}"
    assert track.position_pixel == position_pixel, f"像素坐标更新错误: {track.position_pixel}"
    assert track.team == team, f"队伍信息更新错误: {track.team}"
    assert track.lost_count == 0, "丢失计数应重置为0"
    assert track.hit_streak == 2, f"连续匹配次数错误: {track.hit_streak}"
    assert track.is_confirmed == True, "应确认为有效轨迹"
    
    print(f"✓ 通过: 位置={track.position_pitch}, team={track.team}, hit_streak={track.hit_streak}")


def test_pitch_track_mark_missed():
    """测试轨迹标记丢失"""
    print("\n[单元测试] PitchTrack.mark_missed")
    print("-" * 50)
    
    position = [10.0, 20.0]
    track = PitchTrack(position, class_id=0, frame_num=0, track_id=1, dt=1.0)
    
    initial_lost_count = track.lost_count
    track.mark_missed()
    
    assert track.lost_count == initial_lost_count + 1, f"丢失计数未增加: {track.lost_count}"
    assert track.lost_count > 0, "丢失计数应大于0"
    
    print(f"✓ 通过: lost_count={track.lost_count}")


# ==================== Tracker 单元测试 ====================

def test_tracker_init():
    """测试追踪器初始化"""
    print("\n[单元测试] Tracker.__init__")
    print("-" * 50)
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        max_velocity=0.4,
        track_buffer=30,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    assert tracker.model is not None, "模型未加载"
    assert tracker.distance_threshold_primary == 2.0, "主要阈值设置错误"
    assert tracker.distance_threshold_secondary == 3.0, "次要阈值设置错误"
    assert tracker.use_sampling == False, "采样设置错误"
    assert tracker.tracks == [], "轨迹列表应为空"
    assert tracker.next_id == 1, "下一个ID应为1"
    
    print(f"✓ 通过: 主要阈值={tracker.distance_threshold_primary}米, 采样={'启用' if tracker.use_sampling else '禁用'}")


def test_calculate_distance_matrix():
    """测试距离矩阵计算"""
    print("\n[单元测试] Tracker._calculate_distance_matrix")
    print("-" * 50)
    
    tracker = Tracker(MODEL_PATH, view_transformer=None)
    
    # 创建测试轨迹
    tracks = [
        PitchTrack([10.0, 20.0], class_id=0, frame_num=0, track_id=1),
        PitchTrack([15.0, 25.0], class_id=0, frame_num=0, track_id=2),
    ]
    
    # 创建测试检测
    detections_pitch = [
        [10.5, 20.3],  # 接近轨迹1
        [15.2, 25.1],  # 接近轨迹2
        [50.0, 50.0]   # 远离
    ]
    
    distance_matrix = tracker._calculate_distance_matrix(tracks, detections_pitch)
    
    assert distance_matrix.shape == (2, 3), f"距离矩阵维度错误: {distance_matrix.shape}"
    assert np.all(distance_matrix >= 0), "距离不能为负"
    
    # 验证距离计算
    dist_00 = distance_matrix[0, 0]  # 轨迹1到检测1
    expected_dist_00 = np.linalg.norm(np.array([10.0, 20.0]) - np.array([10.5, 20.3]))
    assert abs(dist_00 - expected_dist_00) < 0.01, f"距离计算错误: {dist_00} vs {expected_dist_00}"
    
    print(f"✓ 通过: shape={distance_matrix.shape}, 最小距离={np.min(distance_matrix):.2f}米, 最大距离={np.max(distance_matrix):.2f}米")


def test_match_tracks_detections():
    """测试轨迹和检测匹配"""
    print("\n[单元测试] Tracker._match_tracks_detections")
    print("-" * 50)
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0
    )
    
    # 创建测试轨迹
    tracks = [
        PitchTrack([10.0, 20.0], class_id=0, frame_num=0, track_id=1),
        PitchTrack([15.0, 25.0], class_id=0, frame_num=0, track_id=2),
    ]
    
    # 创建测试检测（与轨迹位置接近）
    detections_pitch = [
        [10.5, 20.3],  # 应该匹配轨迹1 (距离约0.58米)
        [15.2, 25.1],  # 应该匹配轨迹2 (距离约0.22米)
        [50.0, 50.0]   # 距离太远，不匹配
    ]
    
    detections_info = [
        {'bbox': [100, 200, 150, 280], 'position_pixel': [125, 280], 'class_id': 0, 'confidence': 0.9},
        {'bbox': [200, 250, 250, 330], 'position_pixel': [225, 330], 'class_id': 0, 'confidence': 0.85},
        {'bbox': [500, 500, 550, 580], 'position_pixel': [525, 580], 'class_id': 0, 'confidence': 0.8},
    ]
    
    matches, unmatched_tracks, unmatched_dets = tracker._match_tracks_detections(
        tracks, detections_pitch, detections_info
    )
    
    assert len(matches) == 2, f"匹配数错误: {len(matches)}, 期望2"
    assert len(unmatched_tracks) == 0, f"未匹配轨迹数错误: {len(unmatched_tracks)}"
    assert len(unmatched_dets) == 1, f"未匹配检测数错误: {len(unmatched_dets)}"
    
    # 验证匹配正确性
    matched_track_ids = [tracks[m[0]].track_id for m in matches]
    assert 1 in matched_track_ids, "轨迹1未匹配"
    assert 2 in matched_track_ids, "轨迹2未匹配"
    
    print(f"✓ 通过: {len(matches)}个匹配, {len(unmatched_tracks)}个未匹配轨迹, {len(unmatched_dets)}个未匹配检测")


def test_match_tracks_detections_empty():
    """测试空轨迹或空检测的匹配"""
    print("\n[单元测试] Tracker._match_tracks_detections (空输入)")
    print("-" * 50)
    
    tracker = Tracker(MODEL_PATH, view_transformer=None)
    
    # 测试空轨迹
    matches1, unmatched_tracks1, unmatched_dets1 = tracker._match_tracks_detections(
        [], [[10.0, 20.0]], [{'class_id': 0}]
    )
    assert len(matches1) == 0, "空轨迹应无匹配"
    assert len(unmatched_tracks1) == 0, "空轨迹应无未匹配轨迹"
    assert len(unmatched_dets1) == 1, "应有1个未匹配检测"
    
    # 测试空检测
    tracks = [PitchTrack([10.0, 20.0], class_id=0, frame_num=0, track_id=1)]
    matches2, unmatched_tracks2, unmatched_dets2 = tracker._match_tracks_detections(
        tracks, [], []
    )
    assert len(matches2) == 0, "空检测应无匹配"
    assert len(unmatched_tracks2) == 1, "应有1个未匹配轨迹"
    assert len(unmatched_dets2) == 0, "空检测应无未匹配检测"
    
    print("✓ 通过: 空输入处理正确")


def test_track_detections_single_frame():
    """测试单帧追踪"""
    print("\n[单元测试] Tracker.track_detections (单帧)")
    print("-" * 50)
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    # 创建单帧输入
    # 注意：tracker_pitch 期望 class_id 是经过减1处理的（Roboflow类别ID调整）
    # 如果 cls_names_inv['player'] = 0，那么 class_id 应该是 -1
    tracking_input = {
        'frame_num': 0,
        'detections_pitch': [[10.0, 20.0], [15.0, 25.0]],
        'detections_info': [
            {'bbox': [100, 200, 150, 280], 'position_pixel': [125, 280], 'class_id': -1, 'confidence': 0.9, 'team': 1},  # -1 = player (0-1)
            {'bbox': [200, 250, 250, 330], 'position_pixel': [225, 330], 'class_id': -1, 'confidence': 0.85, 'team': 2},  # -1 = player (0-1)
        ],
        'cls_names': {0: 'player', 1: 'referee', 2: 'ball'},
        'cls_names_inv': {'player': 0, 'referee': 1, 'ball': 2}
    }
    
    tracks = tracker.track_detections([tracking_input], frames=None)
    
    assert 'players' in tracks, "缺少players键"
    assert len(tracks['players']) == 1, f"输出帧数错误: {len(tracks['players'])}"
    assert len(tracks['players'][0]) == 2, f"追踪数错误: {len(tracks['players'][0])}"
    
    print(f"✓ 通过: 1帧, {len(tracks['players'][0])}个追踪")


def test_track_detections_multi_frame():
    """测试多帧追踪（使用完整缓存数据）"""
    print("\n[单元测试] Tracker.track_detections (完整数据)")
    print("-" * 50)
    
    if not os.path.exists(TRACKING_INPUTS_PATH):
        print("⚠ 跳过: 未找到输入文件")
        return True
    
    with open(TRACKING_INPUTS_PATH, 'rb') as f:
        tracking_inputs = pickle.load(f)
    
    if len(tracking_inputs) == 0:
        print("⚠ 跳过: 输入数据为空")
        return True
    
    print(f"  加载完整缓存数据: {len(tracking_inputs)} 帧")
    
    tracker = Tracker(
        MODEL_PATH,
        view_transformer=None,
        distance_threshold_primary=2.0,
        distance_threshold_secondary=3.0,
        sampling_rate=30.0,
        video_fps=30.0
    )
    
    # 使用全部数据
    print(f"  开始追踪: {len(tracking_inputs)} 帧")
    
    # 展示前几帧输入信息
    print(f"  输入数据示例 (前3帧):")
    for i, inp in enumerate(tracking_inputs[:3]):
        print(f"    帧{inp['frame_num']}: {len(inp['detections_pitch'])} 个检测")
        if len(inp['detections_pitch']) > 0:
            print(f"      检测位置(前2个): {inp['detections_pitch'][:2]}")
    
    tracks = tracker.track_detections(tracking_inputs, frames=None)
    
    assert len(tracks['players']) == len(tracking_inputs), f"输出帧数错误: {len(tracks['players'])} vs {len(tracking_inputs)}"
    
    # 统计输出信息
    total_tracks = sum(len(frame_tracks) for frame_tracks in tracks['players'])
    track_ids_all = set()
    for frame_tracks in tracks['players']:
        track_ids_all.update(frame_tracks.keys())
    
    print(f"  输出统计:")
    print(f"    总帧数: {len(tracks['players'])}")
    print(f"    总追踪结果数: {total_tracks}")
    print(f"    唯一追踪ID数: {len(track_ids_all)}")
    print(f"    平均每帧追踪数: {total_tracks / len(tracking_inputs):.1f}")
    
    # 展示前几帧输出信息
    print(f"  输出数据示例 (前3帧):")
    for i in range(min(3, len(tracks['players']))):
        frame_tracks = tracks['players'][i]
        if len(frame_tracks) > 0:
            track_ids = list(frame_tracks.keys())[:3]
            sample_track = frame_tracks[track_ids[0]]
            pos_pitch = sample_track.get('position_pitch', 'N/A')
            print(f"    帧{i}: {len(frame_tracks)} 个追踪, ID={track_ids[:3]}, 示例位置={pos_pitch}")
        else:
            print(f"    帧{i}: 0 个追踪")
    
    print(f"✓ 通过: {len(tracking_inputs)}帧完整追踪完成")


# ==================== 运行所有测试 ====================

def run_all_tests():
    """运行所有单元测试"""
    print("=" * 60)
    print("tracker_pitch 单元测试")
    print("=" * 60)
    
    tests = [
        # PitchKalmanFilter
        ("PitchKalmanFilter.__init__", test_kalman_filter_init),
        ("PitchKalmanFilter.init", test_kalman_filter_init_method),
        ("PitchKalmanFilter.predict", test_kalman_filter_predict),
        ("PitchKalmanFilter.update", test_kalman_filter_update),
        
        # PitchTrack
        ("PitchTrack.__init__", test_pitch_track_init),
        ("PitchTrack.predict", test_pitch_track_predict),
        ("PitchTrack.update", test_pitch_track_update),
        ("PitchTrack.mark_missed", test_pitch_track_mark_missed),
        
        # Tracker
        ("Tracker.__init__", test_tracker_init),
        ("Tracker._calculate_distance_matrix", test_calculate_distance_matrix),
        ("Tracker._match_tracks_detections", test_match_tracks_detections),
        ("Tracker._match_tracks_detections (空输入)", test_match_tracks_detections_empty),
        ("Tracker.track_detections (单帧)", test_track_detections_single_frame),
        ("Tracker.track_detections (多帧)", test_track_detections_multi_frame),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            test_func()
            results.append((name, True))
        except Exception as e:
            print(f"✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试汇总")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{name:50s} {status}")
    
    print(f"\n总计: {passed}/{total} 通过")
    
    if passed == total:
        print("\n✓ 所有测试通过！")
    else:
        print(f"\n✗ {total - passed} 个测试失败")
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
