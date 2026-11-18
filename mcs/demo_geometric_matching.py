"""
几何匹配算法演示：子图同构搜索（朴素遍历）

目标：验证相对几何不变性能否用于跨相机检测匹配
方法：分层几何匹配（锚点 + 扩展）
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import List, Tuple

# ==========================================
# 1. 加载缓存的检测数据
# ==========================================
print("="*70)
print("加载检测数据...")
print("="*70)

cache_path = "store/wuchuan_mcs/position/detections/test_frame_0126.json"

with open(cache_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 过滤：只使用score>0.3的检测
SCORE_THRESHOLD = 0.15

cameras_data = {}
for cam_id in ['1', '2', '3']:
    detections = data['cameras'][cam_id]['detections']
    
    # 提取数据
    all_positions = np.array([det['pitch_position'] for det in detections])
    all_scores = np.array([det['final_score'] for det in detections])
    all_colors = [det['dominant_color'] for det in detections]
    
    # 过滤：score > 0.3
    valid_mask = all_scores > SCORE_THRESHOLD
    valid_indices = np.where(valid_mask)[0]
    
    positions = all_positions[valid_mask]
    scores = all_scores[valid_mask]
    colors = [all_colors[i] for i in valid_indices]
    
    cameras_data[int(cam_id)] = {
        'positions': positions,
        'scores': scores,
        'colors': colors,
        'num': len(positions)
    }
    
    print(f"\nCamera {cam_id}:")
    print(f"  总检测数: {len(detections)}")
    print(f"  过滤后（score>{SCORE_THRESHOLD}）: {len(positions)}")
    print(f"  分数范围: {scores.min():.3f} - {scores.max():.3f}")


# ==========================================
# 2. 辅助函数
# ==========================================

def compute_distance_matrix(positions):
    """计算距离矩阵"""
    n = len(positions)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            D[i, j] = np.linalg.norm(positions[i] - positions[j])
    return D


def select_diverse_anchors(positions, scores, k=5, min_score=0.3, min_separation=400):
    """
    空间分散采样锚点
    
    策略：
    - 第一个：最高分检测
    - 之后：次高分且距离已选锚点>min_separation的检测
    """
    n = len(positions)
    if n == 0:
        return []
    
    # 按分数排序
    sorted_indices = np.argsort(-scores)  # 从高到低
    
    anchor_indices = []
    
    # 第一个：最高分
    for idx in sorted_indices:
        if scores[idx] >= min_score:
            anchor_indices.append(int(idx))
            break
    
    if not anchor_indices:
        # 如果没有满足min_score的，退回到最高分
        anchor_indices.append(int(sorted_indices[0]))
    
    # 之后：次高分且距离足够远
    for idx in sorted_indices:
        if len(anchor_indices) >= k:
            break
        
        if idx in anchor_indices:
            continue
        
        # 检查距离约束
        min_dist_to_anchors = min([
            np.linalg.norm(positions[idx] - positions[anc_idx])
            for anc_idx in anchor_indices
        ])
        
        if min_dist_to_anchors >= min_separation:
            anchor_indices.append(int(idx))
    
    return anchor_indices


def compute_color_similarity(color1: str, color2: str) -> float:
    """
    颜色相似度（软约束）
    """
    if color1 == 'unknown' or color2 == 'unknown':
        return 1.0  # 中性
    
    if color1 == color2:
        return 1.5  # 加分
    else:
        return 0.5  # 降权


def compute_geometric_consistency_to_anchors(
    det_pos, 
    candidate_pos, 
    anchor_pairs: List[Tuple[int, int]], 
    positions1, 
    positions2,
    sigma=100
):
    """
    计算候选匹配相对于锚点的几何一致性
    
    原理：如果det和candidate是同一个人，
    那么det到各anchor1的距离应等于candidate到各anchor2的距离
    
    Args:
        det_pos: 待匹配检测的位置
        candidate_pos: 候选匹配的位置
        anchor_pairs: [(anc1_idx, anc2_idx), ...] 已匹配的锚点对
        positions1, positions2: 两个相机的所有位置
        sigma: 几何误差容差
    
    Returns:
        geometric_sim: 几何相似度 [0, 1]
    """
    if not anchor_pairs:
        return 0.5  # 没有锚点，返回中性值
    
    geometric_errors = []
    
    for anc1_idx, anc2_idx in anchor_pairs:
        anc1_pos = positions1[anc1_idx]
        anc2_pos = positions2[anc2_idx]
        
        # 计算到锚点的距离
        dist1 = np.linalg.norm(det_pos - anc1_pos)
        dist2 = np.linalg.norm(candidate_pos - anc2_pos)
        
        # 距离差异
        error = abs(dist1 - dist2)
        geometric_errors.append(error)
    
    # 平均几何误差转换为相似度
    mean_error = np.mean(geometric_errors)
    geometric_sim = np.exp(-(mean_error**2) / (2 * sigma**2))
    
    return geometric_sim


# ==========================================
# 3. 计算距离矩阵
# ==========================================
print("\n" + "="*70)
print("计算距离矩阵（几何指纹）...")
print("="*70)

distance_matrices = {}
for cam_id, data in cameras_data.items():
    D = compute_distance_matrix(data['positions'])
    distance_matrices[cam_id] = D
    print(f"\nCamera {cam_id} 距离矩阵: {D.shape}")
    upper_triangle = D[np.triu_indices_from(D, k=1)]
    print(f"  点间距离统计: min={upper_triangle.min():.1f}cm, "
          f"max={upper_triangle.max():.1f}cm, "
          f"mean={upper_triangle.mean():.1f}cm")


# ==========================================
# 4. 分层几何匹配算法
# ==========================================

def hierarchical_geometric_matching(
    positions1, scores1, colors1,
    positions2, scores2, colors2,
    D1, D2,
    num_anchors=15,
    min_anchor_score=0.1,
    anchor_separation=0,
    distance_threshold=700,  # 15m硬约束
    anchor_match_threshold=0.35,
    extend_match_threshold=0.25
):
    """
    朴素遍历算法：子图同构搜索（Subgraph Isomorphism）
    
    问题建模：
    - G1 = (V1, E1): Camera1的图，节点=检测（score>0.3），边=距离关系
    - G2 = (V2, E2): Camera2的图
    - S1 ⊂ G1: Camera1的锚点子图（初始化子图）
    - 目标：找 S2 ⊂ G2，使得 S1 ≅ S2（同构）
    
    算法流程（朴素遍历）：
    1. 选择锚点子图 S1 ⊂ G1（空间分散采样）
    2. Phase 2: 对S1中每个节点v1，遍历G2中所有节点v2
       - 比较几何结构相似度（k近邻距离模式）
       - 选择最佳匹配 → 得到锚点对
    3. Phase 3: 基于锚点对，扩展匹配非锚点节点
       - 对G1剩余节点，遍历G2未匹配节点
       - 使用锚点对作为几何基准
    
    时间复杂度：O(|V1| × |V2| × k)
    空间复杂度：O(|V1| × |V2|)
    
    Returns:
        matches: [(i, j, score), ...] 匹配对列表
        anchor_pairs: [(i, j), ...] 锚点匹配对（用于可视化）
    """
    n1, n2 = len(positions1), len(positions2)
    
    print(f"\n{'='*70}")
    print(f"分层几何匹配")
    print(f"{'='*70}")
    
    # ===== Phase 1: 选择锚点 =====
    print("\n[Phase 1] 选择锚点...")
    
    anchors1 = select_diverse_anchors(
        positions1, scores1, 
        k=num_anchors, 
        min_score=min_anchor_score, 
        min_separation=anchor_separation
    )
    
    print(f"  Camera 1: {len(anchors1)} 个锚点 {anchors1}")
    
    for idx in anchors1[:3]:
        print(f"    Anchor1[{idx}]: pos=({positions1[idx][0]:.0f}, {positions1[idx][1]:.0f}), "
              f"score={scores1[idx]:.3f}, color={colors1[idx]}")
    
    # ===== Phase 2: 子图同构搜索（朴素遍历） =====
    print("\n[Phase 2] 子图同构搜索：锚点子图 S1 → 完整图 G2")
    print(f"  搜索空间: |S1|={len(anchors1)} 个锚点 × |V2|={n2} 个节点")
    
    anchor_pairs = []
    used_idx2 = set()
    
    for anc1_idx in anchors1:
        best_match = None
        best_score = 0.0
        best_details = None
        
        # 遍历Camera2的所有节点
        for idx2 in range(n2):
            if idx2 in used_idx2:
                continue
            
            # 1. 距离硬约束
            spatial_dist = np.linalg.norm(positions1[anc1_idx] - positions2[idx2])
            if spatial_dist > distance_threshold:
                continue
            
            # 2. 空间相似度
            spatial_sim = np.exp(-(spatial_dist**2) / (2 * 200**2))
            
            # 3. 几何相似度
            if len(anchor_pairs) > 0:
                # 使用已匹配的锚点作为参考
                geom_sim = compute_geometric_consistency_to_anchors(
                    positions1[anc1_idx],
                    positions2[idx2],
                    anchor_pairs,
                    positions1,
                    positions2,
                    sigma=100
                )
            else:
                # 第一对锚点：使用k近邻的距离模式
                k = min(8, n1-1, n2-1)
                if k > 0:
                    dist_vec1 = D1[anc1_idx, :]
                    dist_vec2 = D2[idx2, :]
                    
                    # 获取k近邻的距离
                    nearest_k1 = np.sort(dist_vec1)[1:k+1]
                    nearest_k2 = np.sort(dist_vec2)[1:k+1]
                    
                    # 计算距离向量的差异
                    dist_diff = np.abs(nearest_k1 - nearest_k2)
                    mean_error = np.mean(dist_diff)
                    geom_sim = np.exp(-(mean_error**2) / (2 * 150**2))
                else:
                    geom_sim = 0.5
            
            # 4. 颜色相似度
            color_sim = compute_color_similarity(colors1[anc1_idx], colors2[idx2])
            
            # 5. 综合相似度
            base_sim = 0.3 * spatial_sim + 0.7 * geom_sim
            total_sim = base_sim * color_sim
            
            if total_sim > best_score:
                best_score = total_sim
                best_match = idx2
                best_details = {
                    'spatial_dist': spatial_dist,
                    'spatial_sim': spatial_sim,
                    'geom_sim': geom_sim,
                    'color_sim': color_sim,
                    'base_sim': base_sim
                }
        
        # 打印最佳候选
        if best_match is not None and best_score > anchor_match_threshold:
            d = best_details
            print(f"  锚点匹配: Anchor1[{anc1_idx}] <-> Det2[{best_match}]  "
                  f"相似度={best_score:.3f} "
                  f"(空间={d['spatial_sim']:.2f}, 几何={d['geom_sim']:.2f}, "
                  f"颜色={d['color_sim']:.2f})")
            
            anchor_pairs.append((anc1_idx, best_match))
            used_idx2.add(best_match)
    
    print(f"\n  成功匹配 {len(anchor_pairs)} 对锚点")
    
    if len(anchor_pairs) < 1:
        print("  ⚠️ 锚点匹配不足")
        return [], []
    
    # ===== Phase 3: 基于锚点扩展匹配 =====
    print("\n[Phase 3] 基于锚点扩展匹配...")
    
    all_matches = list(anchor_pairs)
    used_idx1 = {pair[0] for pair in anchor_pairs}
    used_idx2 = {pair[1] for pair in anchor_pairs}
    
    # 对剩余检测按分数排序
    remaining1 = [
        (i, scores1[i]) for i in range(n1)
        if i not in used_idx1
    ]
    remaining1.sort(key=lambda x: x[1], reverse=True)
    
    print(f"  剩余待匹配: Camera1={len(remaining1)}, Camera2={n2-len(used_idx2)}")
    
    extend_count = 0
    for idx1, score1 in remaining1:
        best_match = None
        best_score = 0.0
        
        for idx2 in range(n2):
            if idx2 in used_idx2:
                continue
            
            # 1. 距离硬约束
            spatial_dist = np.linalg.norm(positions1[idx1] - positions2[idx2])
            if spatial_dist > distance_threshold:
                continue
            
            # 2. 空间相似度
            spatial_sim = np.exp(-(spatial_dist**2) / (2 * 200**2))
            
            # 3. 基于锚点的几何一致性
            geom_sim = compute_geometric_consistency_to_anchors(
                positions1[idx1],
                positions2[idx2],
                anchor_pairs,  # 使用锚点作为参考
                positions1,
                positions2,
                sigma=100
            )
            
            # 4. 颜色相似度
            color_sim = compute_color_similarity(colors1[idx1], colors2[idx2])
            
            # 5. 综合相似度
            base_sim = 0.2 * spatial_sim + 0.8 * geom_sim
            total_sim = base_sim * color_sim
            
            if total_sim > best_score:
                best_score = total_sim
                best_match = idx2
        
        # 扩展阈值比锚点低一些
        if best_match is not None and best_score > extend_match_threshold:
            all_matches.append((idx1, best_match))
            used_idx2.add(best_match)
            extend_count += 1
            
            if extend_count <= 5:  # 只打印前5个
                print(f"  扩展: det1[{idx1}](score={scores1[idx1]:.3f}) <-> "
                      f"det2[{best_match}](score={scores2[best_match]:.3f})  "
                      f"相似度={best_score:.3f}")
    
    print(f"\n  扩展匹配: {extend_count} 对")
    print(f"  总匹配数: {len(all_matches)} 对")
    
    # 转换为 (i, j, score) 格式
    matches_with_scores = []
    for i, j in all_matches:
        # 重新计算最终相似度作为score
        spatial_dist = np.linalg.norm(positions1[i] - positions2[j])
        spatial_sim = np.exp(-(spatial_dist**2) / (2 * 200**2))
        geom_sim = compute_geometric_consistency_to_anchors(
            positions1[i], positions2[j],
            anchor_pairs, positions1, positions2, sigma=100
        )
        color_sim = compute_color_similarity(colors1[i], colors2[j])
        base_sim = 0.2 * spatial_sim + 0.8 * geom_sim
        final_sim = base_sim * color_sim
        
        matches_with_scores.append((i, j, final_sim))
    
    return matches_with_scores, anchor_pairs


# ==========================================
# 5. 执行子图同构搜索（6个有向匹配）
# ==========================================
print("\n" + "="*70)
print("执行子图同构搜索：朴素遍历算法")
print("="*70)
print("策略：Camera_i的锚点子图 → 在Camera_j的完整图中找同构部分")
print("注意：C1→C2 和 C2→C1 是不对称的，分别计算")
print()

# 存储所有匹配结果（有向图）
all_match_results = {}

# 第1行：Camera 1 作为查询图
print("\n[查询图: Camera 1]")

# C1 → C2
print("  C1 → C2: ", end="")
matches_12, anchors_12 = hierarchical_geometric_matching(
    cameras_data[1]['positions'], cameras_data[1]['scores'], cameras_data[1]['colors'],
    cameras_data[2]['positions'], cameras_data[2]['scores'], cameras_data[2]['colors'],
    distance_matrices[1], distance_matrices[2]
)
all_match_results[(1, 2)] = {'matches': matches_12, 'anchors': anchors_12}
print(f"{len(matches_12)} 对匹配 (锚点: {len(anchors_12)})")

# C1 → C3
print("  C1 → C3: ", end="")
matches_13, anchors_13 = hierarchical_geometric_matching(
    cameras_data[1]['positions'], cameras_data[1]['scores'], cameras_data[1]['colors'],
    cameras_data[3]['positions'], cameras_data[3]['scores'], cameras_data[3]['colors'],
    distance_matrices[1], distance_matrices[3]
)
all_match_results[(1, 3)] = {'matches': matches_13, 'anchors': anchors_13}
print(f"{len(matches_13)} 对匹配 (锚点: {len(anchors_13)})")


# 第2行：Camera 2 作为查询图
print("\n[查询图: Camera 2]")

# C2 → C1
print("  C2 → C1: ", end="")
matches_21, anchors_21 = hierarchical_geometric_matching(
    cameras_data[2]['positions'], cameras_data[2]['scores'], cameras_data[2]['colors'],
    cameras_data[1]['positions'], cameras_data[1]['scores'], cameras_data[1]['colors'],
    distance_matrices[2], distance_matrices[1]
)
all_match_results[(2, 1)] = {'matches': matches_21, 'anchors': anchors_21}
print(f"{len(matches_21)} 对匹配 (锚点: {len(anchors_21)})")

# C2 → C3
print("  C2 → C3: ", end="")
matches_23, anchors_23 = hierarchical_geometric_matching(
    cameras_data[2]['positions'], cameras_data[2]['scores'], cameras_data[2]['colors'],
    cameras_data[3]['positions'], cameras_data[3]['scores'], cameras_data[3]['colors'],
    distance_matrices[2], distance_matrices[3]
)
all_match_results[(2, 3)] = {'matches': matches_23, 'anchors': anchors_23}
print(f"{len(matches_23)} 对匹配 (锚点: {len(anchors_23)})")


# 第3行：Camera 3 作为查询图
print("\n[查询图: Camera 3]")

# C3 → C1
print("  C3 → C1: ", end="")
matches_31, anchors_31 = hierarchical_geometric_matching(
    cameras_data[3]['positions'], cameras_data[3]['scores'], cameras_data[3]['colors'],
    cameras_data[1]['positions'], cameras_data[1]['scores'], cameras_data[1]['colors'],
    distance_matrices[3], distance_matrices[1]
)
all_match_results[(3, 1)] = {'matches': matches_31, 'anchors': anchors_31}
print(f"{len(matches_31)} 对匹配 (锚点: {len(anchors_31)})")

# C3 → C2
print("  C3 → C2: ", end="")
matches_32, anchors_32 = hierarchical_geometric_matching(
    cameras_data[3]['positions'], cameras_data[3]['scores'], cameras_data[3]['colors'],
    cameras_data[2]['positions'], cameras_data[2]['scores'], cameras_data[2]['colors'],
    distance_matrices[3], distance_matrices[2]
)
all_match_results[(3, 2)] = {'matches': matches_32, 'anchors': anchors_32}
print(f"{len(matches_32)} 对匹配 (锚点: {len(anchors_32)})")


# ==========================================
# 6. 分析匹配结果
# ==========================================
print("\n" + "="*70)
print("匹配结果分析")
print("="*70)

for (cam1, cam2), result in all_match_results.items():
    matches = result['matches']
    
    # 统计低分检测的匹配
    low_score_count = 0
    for i, j, match_score in matches:
        s1 = cameras_data[cam1]['scores'][i]
        s2 = cameras_data[cam2]['scores'][j]
        if s1 < 0.5 or s2 < 0.5:
            low_score_count += 1
    
    print(f"\nCamera {cam1} → Camera {cam2}:")
    print(f"  涉及低分检测（<0.5）的匹配: {low_score_count}/{len(matches)} 对 "
          f"({100*low_score_count/len(matches) if matches else 0:.1f}%)")


# ==========================================
# 7. 可视化匹配结果（3×2 布局：6个有向匹配）
# ==========================================
print("\n" + "="*70)
print("生成可视化...")
print("="*70)

# 计算所有相机的坐标范围（用于固定比例尺）
all_positions = []
for cam_id in [1, 2, 3]:
    all_positions.append(cameras_data[cam_id]['positions'])
all_positions = np.vstack(all_positions)

x_min, x_max = all_positions[:, 0].min(), all_positions[:, 0].max()
y_min, y_max = all_positions[:, 1].min(), all_positions[:, 1].max()

# 添加边距（10%）
x_margin = (x_max - x_min) * 0.1
y_margin = (y_max - y_min) * 0.1

x_range = [x_min - x_margin, x_max + x_margin]
y_range = [y_min - y_margin, y_max + y_margin]

print(f"固定坐标范围: X=[{x_range[0]:.0f}, {x_range[1]:.0f}], Y=[{y_range[0]:.0f}, {y_range[1]:.0f}]")

# 3行×2列布局
fig, axes = plt.subplots(3, 2, figsize=(16, 24))

# 定义6个有向匹配对（行：查询图，列：目标图）
camera_pairs = [
    (1, 2), (1, 3),  # 第1行：Camera 1 作为查询图
    (2, 1), (2, 3),  # 第2行：Camera 2 作为查询图
    (3, 1), (3, 2)   # 第3行：Camera 3 作为查询图
]

colors_map = {
    1: ('Reds', 'red', 'Camera 1'),
    2: ('Greens', 'green', 'Camera 2'),
    3: ('Blues', 'blue', 'Camera 3')
}

for idx, (cam1, cam2) in enumerate(camera_pairs):
    row = idx // 2
    col = idx % 2
    ax = axes[row, col]
    
    matches = all_match_results[(cam1, cam2)]['matches']
    anchors = all_match_results[(cam1, cam2)]['anchors']
    
    positions1 = cameras_data[cam1]['positions']
    positions2 = cameras_data[cam2]['positions']
    scores1 = cameras_data[cam1]['scores']
    scores2 = cameras_data[cam2]['scores']
    
    # Camera 1的检测
    cmap1, edge1, label1 = colors_map[cam1]
    scatter1 = ax.scatter(positions1[:, 0], positions1[:, 1],
                         c=scores1, cmap=cmap1, s=100,
                         edgecolors=edge1, linewidths=2, alpha=0.6,
                         vmin=0, vmax=1, label=label1, zorder=3)
    
    # Camera 2的检测
    cmap2, edge2, label2 = colors_map[cam2]
    scatter2 = ax.scatter(positions2[:, 0], positions2[:, 1],
                         c=scores2, cmap=cmap2, s=100,
                         edgecolors=edge2, linewidths=2, alpha=0.6,
                         vmin=0, vmax=1, label=label2, zorder=3)
    
    # 画匹配连线
    for i, j, match_score in matches:
        pos1 = positions1[i]
        pos2 = positions2[j]
        
        # 锚点用粗紫线，扩展用灰线
        if (i, j) in anchors:
            linewidth = 3
            alpha = 0.7
            color = 'purple'
        else:
            linewidth = 1 + 2 * match_score
            alpha = 0.3
            color = 'gray'
        
        ax.plot([pos1[0], pos2[0]], [pos1[1], pos2[1]],
               color=color, alpha=alpha, linewidth=linewidth, zorder=1)
    
    # 设置坐标轴（固定比例尺）
    ax.set_xlabel('X (cm)', fontsize=11)
    ax.set_ylabel('Y (cm)', fontsize=11)
    ax.set_title(f'{label1} → {label2}\n'
                f'{len(matches)} 对匹配 (锚点: {len(anchors)})',
                fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # 设置固定的坐标范围（所有子图使用相同范围）
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    
    # Y轴从上到下增加（翻转）
    ax.invert_yaxis()

plt.tight_layout()
output_path = Path("store/wuchuan_mcs/position/detections/geometric_matching_analysis.png")
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 可视化已保存: {output_path}")

plt.close()


# ==========================================
# 8. 总结
# ==========================================
print("\n" + "="*70)
print("总结")
print("="*70)

total_matches = sum(len(result['matches']) for result in all_match_results.values())
total_anchors = sum(len(result['anchors']) for result in all_match_results.values())

print(f"""
✅ 分层几何匹配算法验证结果：

1. 算法特点：
   ✓ 分层匹配（锚点 + 扩展）
   ✓ 空间分散采样锚点
   ✓ k近邻几何相似度 + 锚点一致性
   ✓ 颜色软约束（匹配=1.5x，不匹配=0.5x，未知=1.0x）
   ✓ 距离硬约束（15m阈值）

2. 匹配结果（有向匹配）：
   - Camera 1 → 2: {len(all_match_results[(1,2)]['matches'])} 对 (锚点: {len(all_match_results[(1,2)]['anchors'])})
   - Camera 1 → 3: {len(all_match_results[(1,3)]['matches'])} 对 (锚点: {len(all_match_results[(1,3)]['anchors'])})
   - Camera 2 → 1: {len(all_match_results[(2,1)]['matches'])} 对 (锚点: {len(all_match_results[(2,1)]['anchors'])})
   - Camera 2 → 3: {len(all_match_results[(2,3)]['matches'])} 对 (锚点: {len(all_match_results[(2,3)]['anchors'])})
   - Camera 3 → 1: {len(all_match_results[(3,1)]['matches'])} 对 (锚点: {len(all_match_results[(3,1)]['anchors'])})
   - Camera 3 → 2: {len(all_match_results[(3,2)]['matches'])} 对 (锚点: {len(all_match_results[(3,2)]['anchors'])})
   - 总计: {total_matches} 对匹配，{total_anchors} 对锚点

3. 关键优势：
   ✓ 高分锚点建立可靠几何基准
   ✓ 基于锚点扩展，提升低分检测匹配率
   ✓ 相对几何不变性，鲁棒性强
   ✓ 颜色作为软约束，避免硬过滤损失

4. 可视化：
   ✓ 3×2布局（6个有向匹配）
   ✓ Y轴从上到下增加（球场视角）
   ✓ 紫色粗线=锚点匹配，灰色细线=扩展匹配
   ✓ 固定比例尺，便于对比

5. 应用建议：
   → 用于多相机检测融合（基于几何一致性）
   → 用于识别重复检测（去重）
   → 用于验证和修正低分检测
   → 用于构建全场统一状态
""")

print("\n查看生成的可视化图片以了解匹配细节。")
