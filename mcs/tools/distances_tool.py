import numpy as np
from typing import List, Tuple, Union, Dict
from soccer import HalfPitchConfiguration
import cv2
import supervision as sv

def calculate_pairwise_distances(points: List[Tuple[int, int]], is_closed: bool = False) -> Tuple[List[float], float]:
    """
    计算点列表中相邻对象之间的欧几里得距离
    
    Args:
        points: 点列表 [(x, y), ...] (单位: cm)
        is_closed: 是否为闭合环形（如果是True，会计算最后一个点到第一个点的距离）
        
    Returns:
        distances: 相邻点间距列表 [d1, d2, ...] (单位: m, 保留1位小数)
        average_distance: 平均间距 (单位: m, 保留1位小数)
    """
    pts = np.array(points)
    if len(pts) < 2:
        return [], 0.0
    
    # 计算线性相邻点间距 (i 到 i+1)
    diffs = pts[1:] - pts[:-1]
    distances = np.linalg.norm(diffs, axis=1).tolist()
    
    # 如果是环形，添加首尾距离
    if is_closed:
        last_to_first = np.linalg.norm(pts[-1] - pts[0])
        distances.append(last_to_first)
    
    average_distance = np.mean(distances)
    
    # 转换为米并保留1位小数
    distances_m = [round(d / 100.0, 1) for d in distances]
    avg_dist_m = round(average_distance / 100.0, 1)
    
    return distances_m, avg_dist_m


def calculate_average_distance_to_line(
    points: List[Tuple[int, int]], 
    line_x: int
) -> float:
    """
    计算点列表到指定垂直线 (X=line_x) 的平均距离
    
    Args:
        points: 点列表 [(x, y), ...] (单位: cm)
        line_x: 垂直线的 X 坐标 (单位: cm)，如禁区线 X=2015
        
    Returns:
        average_distance: 平均距离 (单位: cm)
    """
    # 兼容 List 和 NumPy array 的空检查
    if len(points) == 0:
        return 0.0
        
    pts = np.array(points)
    
    # 计算每个点的 x 坐标与 line_x 的绝对差值
    dists = np.abs(pts[:, 0] - line_x)
    
    return float(np.mean(dists))


def calculate_average_x_distance_between_groups(
    group1: List[Tuple[int, int]], 
    group2: List[Tuple[int, int]]
) -> float:
    """
    计算两组点之间在 X 方向上的平均距离
    
    Args:
        group1: 第一组点列表 (如起始点)
        group2: 第二组点列表 (如障碍杆)
        
    Returns:
        average_x_dist: 平均 X 距离 (单位: cm)
        
    Note:
        - 如果两组点数量相同，计算成对点的 X 距离均值
        - 如果数量不同，计算两组点 X 均值之差的绝对值
    """
    if len(group1) == 0 or len(group2) == 0:
        return 0.0
        
    pts1 = np.array(group1)
    pts2 = np.array(group2)
    
    if len(pts1) == len(pts2):
        # 数量相同，假设一一对应，计算每一对的差值
        x_diffs = np.abs(pts1[:, 0] - pts2[:, 0])
        return float(np.mean(x_diffs))
    else:
        # 数量不同，计算整体重心(均值)的差值
        avg_x1 = np.mean(pts1[:, 0])
        avg_x2 = np.mean(pts2[:, 0])
        return float(np.abs(avg_x1 - avg_x2))


def print_shot_train_metrics(
    start_points: List[Tuple[int, int]], 
    obstacles: List[Tuple[int, int]],
    pitch_config: HalfPitchConfiguration = None
):
    """
    打印射门训练相关的综合距离指标报告
    
    Args:
        start_points: 起始点列表 (cm)
        obstacles: 障碍杆列表 (cm)
        pitch_config: 球场配置，默认为 HalfPitchConfiguration
    """
    if pitch_config is None:
        pitch_config = HalfPitchConfiguration()
        
    print("=== 射门训练距离分析报告 (单位: 米) ===")
    
    # 1. 起始点间距
    dists_start, avg_start = calculate_pairwise_distances(start_points)
    print(f"\n1. 起始点间距:")
    if dists_start:
        for i, d in enumerate(dists_start):
            print(f"   S{i+1} -> S{i+2}: {d:.1f} m")
        print(f"   平均间距: {avg_start:.1f} m")
    else:
        print("   (数据不足)")
        
    # 2. 障碍杆间距
    dists_obs, avg_obs = calculate_pairwise_distances(obstacles)
    print(f"\n2. 障碍杆间距:")
    if dists_obs:
        for i, d in enumerate(dists_obs):
            print(f"   Ob{i+1} -> Ob{i+2}: {d:.1f} m")
        print(f"   平均间距: {avg_obs:.1f} m")
    else:
        print("   (数据不足)")
        
    # 禁区线位置
    penalty_x = pitch_config.penalty_box_length
    
    # 3. 到禁区线距离
    avg_start_line = calculate_average_distance_to_line(start_points, penalty_x)
    print(f"\n3. 起始点到大禁区线平均距离: {avg_start_line/100:.2f} m")
    
    avg_obs_line = calculate_average_distance_to_line(obstacles, penalty_x)
    print(f"4. 障碍杆到大禁区线平均距离: {avg_obs_line/100:.2f} m")
    
    # 4. 起始点与障碍杆 X 距离
    avg_x_diff = calculate_average_x_distance_between_groups(start_points, obstacles)
    match_type = "(成对)" if len(start_points) == len(obstacles) else "(整体均值)"
    print(f"\n5. 起始点与障碍杆 X 方向平均距离 {match_type}: {avg_x_diff/100:.2f} m")

# ==========================================================
# 可视化功能
# ==========================================================

def _draw_dashed_line(img, p1, p2, color, thickness=2, gap=10):
    """绘制虚线"""
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((p1[0] * (1 - r) + p2[0] * r))
        y = int((p1[1] * (1 - r) + p2[1] * r))
        pts.append((x, y))
    
    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, pts[i], pts[i+1], color, thickness)

def _to_pix(pt, padding=50, scale=0.1):
    """球场坐标转像素坐标"""
    px = int(pt[0] * scale + padding)
    py = int(pt[1] * scale + padding)
    return (px, py)

def draw_pairwise_distances(pitch_img, points, distances, 
                          color=(200, 200, 200), 
                          padding=50, scale=0.1,
                          is_closed: bool = False):
    """
    可视化相邻点间距：虚线连接 + 距离标注
    
    Args:
        is_closed: 是否绘制闭合环形（首尾连接）
    """
    result_img = pitch_img.copy()
    pts = np.array(points)
    num_points = len(pts)
    
    # 兼容处理：如果传入的是 calculate_pairwise_distances 的完整返回值 (list, avg)
    if isinstance(distances, tuple):
        distances = distances[0]
    
    # 确定循环次数：线性为 N-1，环形为 N
    num_segments = num_points if is_closed else num_points - 1
    
    for i in range(num_segments):
        if i >= len(distances): break
        
        p1_idx = i
        p2_idx = (i + 1) % num_points  # 环形时最后一个点连回第一个点
        
        p1_pix = _to_pix(pts[p1_idx], padding, scale)
        p2_pix = _to_pix(pts[p2_idx], padding, scale)
        
        # 绘制虚线
        _draw_dashed_line(result_img, p1_pix, p2_pix, color)
        
        # 绘制距离文字 (带描边以保证透明背景下的可见性)
        mid_pix = ((p1_pix[0] + p2_pix[0]) // 2, (p1_pix[1] + p2_pix[1]) // 2)
        # distances已经是m单位 (由 calculate_pairwise_distances 返回)
        dist_m = distances[i]
        text = f"{dist_m:.1f}m"
        
        # 描边 (黑色)
        cv2.putText(result_img, text, (mid_pix[0], mid_pix[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3)
        # 文字 (白色)
        cv2.putText(result_img, text, (mid_pix[0], mid_pix[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
    return result_img

def draw_group_to_line_distance(pitch_img, points, line_x, avg_distance,
                              color=(0, 255, 255),
                              padding=50, scale=0.1):
    """
    可视化点到垂直线的距离：重心垂线 + 距离标注
    """
    result_img = pitch_img.copy()
    if len(points) == 0:
        return result_img
        
    pts = np.array(points)
    
    # 计算重心
    center = np.mean(pts, axis=0)
    center_pix = _to_pix(center, padding, scale)
    
    # 计算垂足
    # line_x 是物理坐标 X
    foot_pix = _to_pix((line_x, center[1]), padding, scale)
    
    # 绘制虚线
    _draw_dashed_line(result_img, center_pix, foot_pix, color)
    
    # 绘制目标线 (局部)
    line_x_pix = _to_pix((line_x, 0), padding, scale)[0]
    cv2.line(result_img, (line_x_pix, center_pix[1] - 30), 
             (line_x_pix, center_pix[1] + 30), color, 2)
    
    # 标注距离 (带描边)
    mid_pix = ((center_pix[0] + foot_pix[0]) // 2, center_pix[1])
    text = f"Dist: {avg_distance/100:.1f}m"
    
    # 描边 (黑色)
    cv2.putText(result_img, text, (mid_pix[0], mid_pix[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    # 文字 (目标色)
    cv2.putText(result_img, text, (mid_pix[0], mid_pix[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
    return result_img

def draw_distance_between_groups(pitch_img, group1, group2, 
                               color=(255, 0, 255),
                               padding=50, scale=0.1):
    """
    可视化两组点之间的距离：重心连线 + 距离标注
    """
    result_img = pitch_img.copy()
    if len(group1) == 0 or len(group2) == 0:
        return result_img
        
    # 计算重心
    c1 = np.mean(np.array(group1), axis=0)
    c2 = np.mean(np.array(group2), axis=0)
    
    # 计算重心距离
    avg_distance = np.linalg.norm(c1 - c2)
    
    p1_pix = _to_pix(c1, padding, scale)
    p2_pix = _to_pix(c2, padding, scale)
    
    # 绘制虚线
    _draw_dashed_line(result_img, p1_pix, p2_pix, color, thickness=2)
    
    # 绘制重心点
    cv2.circle(result_img, p1_pix, 4, color, -1)
    cv2.circle(result_img, p2_pix, 4, color, -1)
    
    # 标注距离 (带描边)
    mid_pix = ((p1_pix[0] + p2_pix[0]) // 2, (p1_pix[1] + p2_pix[1]) // 2)
    text = f" {avg_distance/100:.1f}m"
    
    # 描边 (黑色)
    # cv2.putText(result_img, text, (mid_pix[0], mid_pix[1] - 5),
    #             cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 3)
    # 文字 (目标色)
    cv2.putText(result_img, text, (mid_pix[0], mid_pix[1] - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
                
    return result_img

def draw_nearest_points_distance(pitch_img, group1, group2,
                                color=(255, 128, 0),
                                padding=50, scale=0.1):
    """
    可视化两组点之间最近的两个点及其距离
    
    Args:
        pitch_img: 球场图像
        group1: 第一组点 [(x,y),...] (单位: cm)
        group2: 第二组点 [(x,y),...] (单位: cm)
        color: 线条颜色
        padding: 图像边距
        scale: 缩放比例
    
    Returns:
        result_img: 绘制结果
    """
    result_img = pitch_img.copy()
    if len(group1) == 0 or len(group2) == 0:
        return result_img
    
    # 找到最近的两个点
    g1 = np.array(group1)
    g2 = np.array(group2)
    
    min_dist = float('inf')
    nearest_p1 = None
    nearest_p2 = None
    
    for p1 in g1:
        for p2 in g2:
            dist = np.linalg.norm(p1 - p2)
            if dist < min_dist:
                min_dist = dist
                nearest_p1 = p1
                nearest_p2 = p2
    
    if nearest_p1 is None or nearest_p2 is None:
        return result_img
    
    # 转换为像素坐标
    p1_pix = _to_pix(nearest_p1, padding, scale)
    p2_pix = _to_pix(nearest_p2, padding, scale)
    
    # 绘制实线（表示这是实际最近距离）
    cv2.line(result_img, p1_pix, p2_pix, color, 2)
    
    # 高亮最近的两个点
    cv2.circle(result_img, p1_pix, 6, color, -1)
    cv2.circle(result_img, p2_pix, 6, color, -1)
    cv2.circle(result_img, p1_pix, 8, (255, 255, 255), 2)
    cv2.circle(result_img, p2_pix, 8, (255, 255, 255), 2)
    
    # 标注距离
    mid_pix = ((p1_pix[0] + p2_pix[0]) // 2, (p1_pix[1] + p2_pix[1]) // 2)
    dist_m = min_dist / 100.0  # cm -> m
    text = f"Min: {dist_m:.2f}m"
    
    # 描边（黑色）
    cv2.putText(result_img, text, (mid_pix[0], mid_pix[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    # 文字（目标色）
    cv2.putText(result_img, text, (mid_pix[0], mid_pix[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    return result_img

def calculate_all_pairwise_group_distances(groups: List[List[Tuple[int, int]]]) -> Dict:
    """
    计算所有组之间的两两距离（重心距离和最近点距离）
    
    Args:
        groups: 多组点 [[(x,y),...], [(x,y),...], ...]
    
    Returns:
        Dict: {
            'center_distances': [[d_01, d_02, ...], [d_12, ...], ...],
            'nearest_distances': [[d_01, d_02, ...], [d_12, ...], ...]
        }
    """
    n_groups = len(groups)
    center_distances = []
    nearest_distances = []
    
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            g1 = np.array(groups[i])
            g2 = np.array(groups[j])
            
            if len(g1) == 0 or len(g2) == 0:
                continue
            
            # 重心距离
            c1 = np.mean(g1, axis=0)
            c2 = np.mean(g2, axis=0)
            center_dist = np.linalg.norm(c1 - c2) / 100.0  # cm -> m
            
            # 最近点距离
            min_dist = float('inf')
            for p1 in g1:
                for p2 in g2:
                    dist = np.linalg.norm(p1 - p2)
                    if dist < min_dist:
                        min_dist = dist
            nearest_dist = min_dist / 100.0  # cm -> m
            
            center_distances.append({
                'group1': i,
                'group2': j,
                'distance': round(center_dist, 2)
            })
            
            nearest_distances.append({
                'group1': i,
                'group2': j,
                'distance': round(nearest_dist, 2)
            })
    
    return {
        'center_distances': center_distances,
        'nearest_distances': nearest_distances
    }
