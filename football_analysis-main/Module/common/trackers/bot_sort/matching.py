"""
匹配算法模块

实现了多目标跟踪中的各种匹配算法，包括：
1. 匈牙利算法进行最优分配
2. IoU距离计算
3. 外观特征距离计算
4. 运动-外观融合
5. 门控距离计算

主要功能：
- 线性分配问题求解
- 多种距离度量方法
- 特征融合策略
- 门控机制优化
"""

import numpy as np
import scipy
import lap
from scipy.spatial.distance import cdist

from . import kalman_filter


def merge_matches(m1, m2, shape):
    """
    合并两个匹配结果
    
    将两个中间匹配结果合并成最终的匹配结果。这用于级联匹配（cascade matching）
    等场景，其中需要先进行一次匹配，然后再进行二次匹配。
    
    Args:
        m1: 第一组匹配结果，形状为 (K1, 2)，每行为 [idx_O, idx_P]
            表示O集合和P集合之间的匹配
        m2: 第二组匹配结果，形状为 (K2, 2)，每行为 [idx_P, idx_Q]
            表示P集合和Q集合之间的匹配
        shape: 三个集合的大小，元组 (O, P, Q)
            - O: 第一个集合的大小
            - P: 中间集合的大小（连接O和Q）
            - Q: 第三个集合的大小
    
    Returns:
        match: 合并后的匹配结果，列表，每个元素为 (idx_O, idx_Q)
               表示通过中间集合P连接的O和Q之间的匹配
        unmatched_O: 未匹配的O集合索引，元组
        unmatched_Q: 未匹配的Q集合索引，元组
    
    算法原理：
        使用稀疏矩阵的乘法来找到通过中间集合P连接的匹配路径。
        如果 m1[i] = (o, p) 且 m2[j] = (p, q)，则最终匹配为 (o, q)。
    """
    O, P, Q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)

    # 创建稀疏矩阵表示第一组匹配：M1[O, P]
    # 如果 (i, j) 在m1中，则M1[i, j] = 1
    M1 = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(O, P))
    
    # 创建稀疏矩阵表示第二组匹配：M2[P, Q]
    # 如果 (j, k) 在m2中，则M2[j, k] = 1
    M2 = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(P, Q))

    # 矩阵乘法：mask = M1 * M2，得到最终的匹配矩阵
    # mask[i, k] = 1 当且仅当存在 j 使得 M1[i, j] = 1 且 M2[j, k] = 1
    # 这表示通过中间集合P连接的匹配路径
    mask = M1 * M2
    
    # 提取非零元素的位置，即所有匹配对
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))  # 转换为列表 [(o_idx, q_idx), ...]
    
    # 找出未匹配的索引
    unmatched_O = tuple(set(range(O)) - set([i for i, j in match]))  # O中未匹配的
    unmatched_Q = tuple(set(range(Q)) - set([j for i, j in match]))   # Q中未匹配的

    return match, unmatched_O, unmatched_Q


def _indices_to_matches(cost_matrix, indices, thresh):
    """
    将索引对转换为匹配结果（内部函数）
    
    根据成本矩阵和阈值，将索引对转换为最终的匹配结果。
    只有成本小于等于阈值的匹配才会被接受。
    
    Args:
        cost_matrix: 成本矩阵，形状为 (M, N)
                    表示M个轨迹和N个检测之间的匹配成本
        indices: 候选匹配的索引对，形状为 (K, 2)
                 每行为 [row_idx, col_idx]，表示成本矩阵中的位置
        thresh: 成本阈值，只有成本 <= thresh 的匹配才会被接受
    
    Returns:
        matches: 最终的匹配结果，形状为 (K', 2)，每行为 [a_idx, b_idx]
                其中 K' <= K（经过阈值过滤后）
        unmatched_a: 未匹配的a集合（行）索引，元组
        unmatched_b: 未匹配的b集合（列）索引，元组
    """
    # 提取所有候选匹配的成本值
    matched_cost = cost_matrix[tuple(zip(*indices))]
    
    # 根据阈值过滤：只保留成本 <= thresh 的匹配
    matched_mask = (matched_cost <= thresh)

    # 提取符合阈值的匹配
    matches = indices[matched_mask]
    
    # 找出未匹配的索引
    unmatched_a = tuple(set(range(cost_matrix.shape[0])) - set(matches[:, 0]))  # 行索引未匹配的
    unmatched_b = tuple(set(range(cost_matrix.shape[1])) - set(matches[:, 1]))  # 列索引未匹配的

    return matches, unmatched_a, unmatched_b


def linear_assignment(cost_matrix, thresh):
    """
    使用匈牙利算法求解线性分配问题
    
    在给定成本矩阵和阈值下，找到最优的分配方案
    
    Args:
        cost_matrix: 成本矩阵，形状为 (M, N)
        thresh: 成本阈值，超过此值的分配将被拒绝
        
    Returns:
        matches: 匹配对列表，形状为 (K, 2)，每行为 [track_idx, detection_idx]
        unmatched_a: 未匹配的轨迹索引
        unmatched_b: 未匹配的检测索引
    """
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    matches, unmatched_a, unmatched_b = [], [], []
    
    # 使用Jonker-Volgenant算法求解分配问题
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    
    # 提取匹配对
    for ix, mx in enumerate(x):
        if mx >= 0:  # 有效的匹配
            matches.append([ix, mx])
    
    # 提取未匹配的索引
    unmatched_a = np.where(x < 0)[0]  # 未匹配的轨迹
    unmatched_b = np.where(y < 0)[0]  # 未匹配的检测
    
    matches = np.asarray(matches)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    计算两组边界框之间的IoU（Intersection over Union，交并比）矩阵
    
    IoU是目标跟踪中常用的相似度度量，表示两个边界框的重叠程度。
    IoU值范围在[0, 1]之间，值越大表示重叠越多。
    
    Args:
        atlbrs: 第一组边界框，可以是列表或numpy数组
                格式为 [x1, y1, x2, y2]（左上角和右下角坐标）
                如果是列表，每个元素是一个边界框；如果是数组，形状为 (N, 4)
        btlbrs: 第二组边界框，格式同上
                如果是列表，每个元素是一个边界框；如果是数组，形状为 (M, 4)

    Returns:
        ious: IoU矩阵，形状为 (len(atlbrs), len(btlbrs))
              元素 ious[i, j] 表示第i个atlbrs边界框与第j个btlbrs边界框的IoU值
              IoU值范围：[0, 1]，1表示完全重叠，0表示不重叠
    
    使用场景：
        - 计算轨迹与检测之间的IoU，用于匹配
        - 计算预测边界框与真实边界框的IoU，用于评估
    """
    # 初始化IoU矩阵
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=np.float32)
    
    # 处理空输入的情况
    if ious.size == 0:
        return ious

    # 调用底层函数计算IoU
    # 使用连续数组以提高计算效率
    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=np.float32),
        np.ascontiguousarray(btlbrs, dtype=np.float32)
    )

    return ious


def tlbr_expand(tlbr, scale=1.2):
    """
    扩展边界框（扩大边界框范围）
    
    将给定的边界框按照指定比例进行扩展。常用于：
    - 增加边界框的容错范围
    - 在进行IoU计算前预处理边界框
    - 为跟踪提供更大的搜索范围
    
    Args:
        tlbr: 边界框坐标，格式为 [x1, y1, x2, y2]
              - (x1, y1): 左上角坐标
              - (x2, y2): 右下角坐标
              注意：此函数会直接修改输入数组
        scale: 扩展比例，默认1.2
               - scale > 1.0: 扩大边界框
               - scale = 1.0: 不改变
               - scale < 1.0: 缩小边界框

    Returns:
        tlbr: 扩展后的边界框（修改后的原数组）
              新的边界框在保持中心不变的情况下，宽高扩展为原来的scale倍
    
    算法原理：
        扩展是在中心点不变的情况下进行的：
        - 宽度增加: (scale - 1) * w
        - 高度增加: (scale - 1) * h
        - 每个方向各扩展一半，保持中心点不变
    """
    # 计算原始边界框的宽度和高度
    w = tlbr[2] - tlbr[0]
    h = tlbr[3] - tlbr[1]

    # 计算每个方向需要扩展的量（一半）
    # 例如 scale=1.2，则每个方向扩展 0.2 * w / 2
    half_scale = 0.5 * scale

    # 扩展左上角坐标：向左上方向移动
    tlbr[0] -= half_scale * w  # x1 向左移动
    tlbr[1] -= half_scale * h  # y1 向上移动
    
    # 扩展右下角坐标：向右下方向移动
    tlbr[2] += half_scale * w  # x2 向右移动
    tlbr[3] += half_scale * h  # y3 向下移动

    return tlbr


def iou_distance(atracks, btracks):
    """
    基于IoU计算成本矩阵
    
    计算轨迹和检测之间的IoU距离，用于匹配
    
    Args:
        atracks: 轨迹列表 (STrack对象或边界框数组)
        btracks: 检测列表 (STrack对象或边界框数组)

    Returns:
        cost_matrix: 成本矩阵，形状为 (len(atracks), len(btracks))
                    值越小表示匹配度越高
    """
    # 处理输入类型：支持STrack对象或直接的边界框数组
    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks  # 直接使用数组
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]  # 提取边界框
        btlbrs = [track.tlbr for track in btracks]
    
    # 计算IoU矩阵
    _ious = ious(atlbrs, btlbrs)
    
    # 转换为距离：距离 = 1 - IoU
    cost_matrix = 1 - _ious

    return cost_matrix


def v_iou_distance(atracks, btracks):
    """
    基于预测边界框的IoU计算成本矩阵（预测版本）
    
    与 iou_distance 类似，但使用的是轨迹的预测边界框（pred_bbox）
    而不是当前边界框。预测边界框是通过卡尔曼滤波预测得到的。
    
    Args:
        atracks: 轨迹列表，每个元素是STrack对象或边界框数组
                如果是STrack对象，使用其预测边界框
        btracks: 检测列表，每个元素是STrack对象或边界框数组
                如果是STrack对象，使用其预测边界框

    Returns:
        cost_matrix: 成本矩阵，形状为 (len(atracks), len(btracks))
                    值范围：[0, 1]
                    值越小表示匹配度越高（IoU越大）
    
    使用场景：
        - 在第二阶段的匹配中使用预测边界框
        - 处理未激活的轨迹（使用预测位置进行匹配）
        - 当需要基于运动预测进行匹配时
    
    与 iou_distance 的区别：
        - iou_distance: 使用当前边界框（tlbr）
        - v_iou_distance: 使用预测边界框（pred_bbox）
    """
    # 处理输入类型
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or \
       (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        # 如果输入已经是数组，直接使用
        atlbrs = atracks
        btlbrs = btracks
    else:
        # 如果是STrack对象，提取预测边界框并转换为tlbr格式
        # pred_bbox 是 tlwh 格式（top-left-width-height），需要转换为 tlbr
        atlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in atracks]
        btlbrs = [track.tlwh_to_tlbr(track.pred_bbox) for track in btracks]
    
    # 计算IoU矩阵
    _ious = ious(atlbrs, btlbrs)
    
    # 转换为成本：成本 = 1 - IoU
    # IoU越大（重叠越多），成本越小
    cost_matrix = 1 - _ious

    return cost_matrix


def embedding_distance(tracks, detections, metric='cosine'):
    """
    基于外观特征（embedding）计算成本矩阵
    
    通过比较轨迹和检测的外观特征来计算匹配成本。
    外观特征通常是通过深度神经网络（如ReID网络）提取的。
    
    Args:
        tracks: 轨迹列表，每个元素是STrack对象
                使用每个轨迹的平滑特征（smooth_feat）
        detections: 检测列表，每个元素是BaseTrack对象
                   使用每个检测的当前特征（curr_feat）
        metric: 距离度量方式，默认'cosine'
                - 'cosine': 余弦距离（推荐，适合归一化特征）
                - 'euclidean': 欧氏距离
                - 其他scipy.spatial.distance支持的距离度量
    
    Returns:
        cost_matrix: 成本矩阵，形状为 (len(tracks), len(detections))
                    值范围：[0, inf)
                    值越小表示外观越相似，匹配度越高
    
    使用场景：
        - 在数据关联中考虑外观相似性
        - 当目标被遮挡后重新出现时，通过外观匹配
        - 处理ID切换（ID switch）问题
    
    特征说明：
        - smooth_feat: 轨迹的平滑特征（历史特征的加权平均）
        - curr_feat: 检测的当前特征（最新帧的特征）
        - 特征通常已经归一化，所以适合使用余弦距离
    """
    # 初始化成本矩阵
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float32)
    
    # 处理空输入
    if cost_matrix.size == 0:
        return cost_matrix
    
    # 提取检测的当前特征（当前帧的外观特征）
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float32)
    
    # 提取轨迹的平滑特征（历史特征的加权平均，更稳定）
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float32)

    # 计算特征之间的距离矩阵
    # cdist计算两个特征集之间的成对距离
    # 使用最大值0.0确保距离非负（某些度量可能返回负值）
    cost_matrix = np.maximum(0.0, cdist(track_features, det_features, metric))
    # 注意：如果特征已归一化，余弦距离范围是[0, 2]，可以除以2.0归一化到[0, 1]
    # 当前代码注释掉了归一化： # / 2.0  # Nomalized features
    
    return cost_matrix


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    """
    对成本矩阵应用门控（gating）机制
    
    使用卡尔曼滤波器的门控距离来过滤不可能的匹配。
    如果轨迹与检测之间的门控距离超过阈值，则将成本设为无穷大，
    这样这些匹配就不会在后续的匈牙利算法中被选中。
    
    Args:
        kf: 卡尔曼滤波器对象，用于计算门控距离
        cost_matrix: 原始成本矩阵，形状为 (len(tracks), len(detections))
                     会被直接修改
        tracks: 轨迹列表，每个元素是STrack对象
                需要包含mean和covariance属性（卡尔曼滤波状态）
        detections: 检测列表，每个元素是BaseTrack对象
        only_position: 是否只使用位置信息计算门控距离
                       - False: 使用4维距离（位置+尺寸），阈值使用chi2inv95[4]
                       - True: 使用2维距离（仅位置），阈值使用chi2inv95[2]
    
    Returns:
        cost_matrix: 经过门控处理后的成本矩阵（修改后的原矩阵）
                    不可能匹配的位置被设为np.inf
    
    算法原理：
        对每个轨迹，计算它与所有检测的马哈拉诺比斯距离。
        如果距离超过卡方分布的0.95分位数（显著性水平5%），
        则认为该匹配不太可能，将成本设为无穷大。
    
    使用场景：
        - 在数据关联前预处理成本矩阵
        - 减少匈牙利算法需要处理的匹配数量
        - 提高跟踪的准确性和效率
    """
    # 处理空输入
    if cost_matrix.size == 0:
        return cost_matrix
    
    # 确定门控维度并获取对应的卡方分布阈值
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    
    # 将检测转换为测量格式（xywh格式：中心点+宽高）
    # 注意：注释显示可能使用xyah格式，但实际代码使用xywh
    measurements = np.asarray([det.to_xywh() for det in detections])
    
    # 对每个轨迹计算门控距离并过滤
    for row, track in enumerate(tracks):
        # 计算该轨迹与所有检测的马哈拉诺比斯距离
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position)
        
        # 将距离超过阈值的匹配成本设为无穷大
        # 这样在匈牙利算法中就不会被选中
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    
    return cost_matrix


def fuse_motion(kf, cost_matrix, tracks, detections, only_position=False, lambda_=0.98):
    """
    融合运动信息到成本矩阵中
    
    将运动距离（马哈拉诺比斯距离）与原始成本矩阵进行加权融合。
    同时应用门控机制，过滤不可能的匹配。
    这种方法结合了外观成本（原始cost_matrix）和运动成本（门控距离）。
    
    Args:
        kf: 卡尔曼滤波器对象，用于计算门控距离
        cost_matrix: 原始成本矩阵（通常是外观特征距离），形状为 (len(tracks), len(detections))
                     会被修改为融合后的成本矩阵
        tracks: 轨迹列表，每个元素是STrack对象
                需要包含mean和covariance属性
        detections: 检测列表，每个元素是BaseTrack对象
        only_position: 是否只使用位置信息计算门控距离
                       - False: 使用4维距离（位置+尺寸）
                       - True: 使用2维距离（仅位置）
        lambda_: 融合权重，范围 [0, 1]，默认0.98
                 - 较大的lambda_: 更重视原始成本矩阵（外观）
                 - 较小的lambda_: 更重视运动距离
                 - lambda_ = 0.98 意味着98%权重给外观，2%给运动
    
    Returns:
        cost_matrix: 融合后的成本矩阵（修改后的原矩阵）
                    融合公式：cost = lambda_ * original_cost + (1 - lambda_) * motion_distance
    
    算法步骤：
        1. 对每个轨迹计算与所有检测的马哈拉诺比斯距离
        2. 应用门控：距离超过阈值的设为无穷大
        3. 加权融合：cost = lambda * original_cost + (1 - lambda) * motion_distance
    
    使用场景：
        - 在深度跟踪中结合外观和运动信息
        - 当外观特征不稳定时，运动信息可以作为补充
        - BoTSORT等先进跟踪算法的核心方法
    """
    # 处理空输入
    if cost_matrix.size == 0:
        return cost_matrix
    
    # 确定门控维度和阈值
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    
    # 将检测转换为测量格式
    measurements = np.asarray([det.to_xywh() for det in detections])
    
    # 对每个轨迹进行融合
    for row, track in enumerate(tracks):
        # 计算马哈拉诺比斯距离（运动距离）
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position, metric='maha')
        
        # 应用门控：过滤不可能的匹配
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
        
        # 加权融合运动距离和原始成本
        # 公式：cost = lambda * original_cost + (1 - lambda) * motion_distance
        cost_matrix[row] = lambda_ * cost_matrix[row] + (1 - lambda_) * gating_distance
    
    return cost_matrix


def fuse_iou(cost_matrix, tracks, detections):
    """
    融合IoU信息到成本矩阵中
    
    将外观特征成本（ReID成本）与IoU成本进行融合。
    IoU表示边界框的重叠程度，ReID表示外观相似性。
    融合两者可以得到更准确的匹配成本。
    
    Args:
        cost_matrix: 原始成本矩阵（通常是ReID特征距离），形状为 (len(tracks), len(detections))
                     值范围通常在[0, 1]（如果使用归一化的余弦距离）
        tracks: 轨迹列表，每个元素是STrack对象
        detections: 检测列表，每个元素是BaseTrack对象
    
    Returns:
        fuse_cost: 融合后的成本矩阵，形状与原cost_matrix相同
                   值范围：[0, 1]，值越小表示匹配度越高
    
    融合公式：
        1. 将成本转换为相似度：sim = 1 - cost
        2. ReID相似度：reid_sim = 1 - cost_matrix
        3. IoU相似度：iou_sim = 1 - iou_distance
        4. 融合相似度：fuse_sim = reid_sim * (1 + iou_sim) / 2
        5. 转回成本：fuse_cost = 1 - fuse_sim
    
    算法特点：
        - 当ReID相似度高且IoU相似度也高时，融合相似度会很高
        - 当其中一个相似度低时，融合相似度会降低
        - 公式中的(1 + iou_sim)/2 会将IoU相似度归一化并加权
    
    使用场景：
        - 在数据关联中同时考虑外观和空间重叠
        - 当目标外观相似但位置相距较远时，IoU可以提供区分度
        - 提高匹配的准确性，减少误匹配
    """
    # 处理空输入
    if cost_matrix.size == 0:
        return cost_matrix
    
    # 将ReID成本转换为相似度（相似度 = 1 - 成本）
    # 假设cost_matrix是归一化的距离，范围[0, 1]
    reid_sim = 1 - cost_matrix
    
    # 计算IoU距离矩阵
    iou_dist = iou_distance(tracks, detections)
    
    # 将IoU距离转换为IoU相似度
    iou_sim = 1 - iou_dist
    
    # 融合ReID相似度和IoU相似度
    # 公式：fuse_sim = reid_sim * (1 + iou_sim) / 2
    # (1 + iou_sim)/2 将IoU相似度[0,1]映射到[0.5, 1]，然后与ReID相似度相乘
    fuse_sim = reid_sim * (1 + iou_sim) / 2
    
    # 提取检测的置信度分数（可选，当前代码已注释）
    # 理论上可以用检测置信度进一步加权融合相似度
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    # 如果使用检测分数加权（当前已注释）：
    # fuse_sim = fuse_sim * (1 + det_scores) / 2
    
    # 将融合后的相似度转回成本
    fuse_cost = 1 - fuse_sim
    
    return fuse_cost


def fuse_score(cost_matrix, detections):
    """
    融合检测置信度分数到成本矩阵中
    
    将IoU成本与检测置信度分数进行融合。
    检测置信度分数表示检测的可靠性，置信度越高的检测应该优先匹配。
    
    Args:
        cost_matrix: 原始成本矩阵（通常是IoU距离），形状为 (len(tracks), len(detections))
                     值范围：[0, 1]
        detections: 检测列表，每个元素需要包含score属性（检测置信度）
    
    Returns:
        fuse_cost: 融合后的成本矩阵，形状与原cost_matrix相同
                   值范围：[0, 1]
                   融合了检测置信度后的匹配成本
    
    融合公式：
        1. 将成本转换为相似度：iou_sim = 1 - cost_matrix
        2. 融合相似度：fuse_sim = iou_sim * det_scores
        3. 转回成本：fuse_cost = 1 - fuse_sim
    
    算法特点：
        - 检测置信度高时，即使IoU稍低，融合相似度也可能较高
        - 检测置信度低时，融合相似度会被降低
        - 这种方法优先匹配高置信度的检测，提高跟踪质量
    
    使用场景：
        - 当检测器输出的置信度差异较大时
        - 需要优先匹配高置信度检测的场景
        - 过滤低质量检测，提高跟踪稳定性
    """
    # 处理空输入
    if cost_matrix.size == 0:
        return cost_matrix
    
    # 将成本转换为相似度（假设是IoU距离）
    iou_sim = 1 - cost_matrix
    
    # 提取所有检测的置信度分数
    det_scores = np.array([det.score for det in detections])
    
    # 扩展维度以匹配成本矩阵的形状
    # 将一维数组扩展为矩阵，每行都是相同的检测分数
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    
    # 融合IoU相似度和检测置信度
    # 公式：fuse_sim = iou_sim * det_scores
    # 高置信度检测的相似度会被增强
    fuse_sim = iou_sim * det_scores
    
    # 将融合后的相似度转回成本
    fuse_cost = 1 - fuse_sim
    
    return fuse_cost

def bbox_ious(boxes, query_boxes):
    """
    计算两组边界框之间的IoU矩阵（底层实现函数）
    
    这是IoU计算的核心实现，使用循环方式逐个计算每对边界框的IoU值。
    虽然速度较慢，但实现简单清晰，适合小规模计算或作为参考实现。
    
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
        第一组边界框，形状为(N, 4)
        每行格式：[x1, y1, x2, y2]，即左上角和右下角坐标
    query_boxes: (K, 4) ndarray of float
        第二组边界框，形状为(K, 4)
        每行格式：[x1, y1, x2, y2]
    
    Returns
    -------
    overlaps: (N, K) ndarray of float
        IoU矩阵，形状为(N, K)
        元素overlaps[n, k]表示boxes[n]与query_boxes[k]之间的IoU值
        IoU值范围：[0, 1]
        - 1.0: 完全重叠
        - 0.0: 不重叠
    
    IoU计算公式：
        IoU = 交集面积 / 并集面积
        交集面积 = (重叠宽度) * (重叠高度)
        并集面积 = 框1面积 + 框2面积 - 交集面积
    
    算法步骤：
        1. 对每个query_box，计算其面积
        2. 对每个box，计算与query_box的交集区域
        3. 如果交集存在（宽度和高度都>0），计算IoU
        4. 否则IoU为0（默认初始值）
    
    性能说明：
        - 时间复杂度：O(N * K)
        - 适合小规模计算（N, K较小）
        - 对于大规模计算，建议使用向量化实现以提高速度
    """
    N = boxes.shape[0]  # 第一组边界框的数量
    K = query_boxes.shape[0]  # 第二组边界框的数量
    
    # 初始化IoU矩阵
    overlaps = np.zeros((N, K), dtype=np.float32)
    
    # 遍历每个查询框
    for k in range(K):
        # 计算查询框的面积
        # 注意：+1是因为像素坐标是从1开始计数的（或者包含边界像素）
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *  # 宽度
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)    # 高度
        )
        
        # 遍历每个边界框，计算与当前查询框的IoU
        for n in range(N):
            # 计算交集区域的宽度
            # 交集宽度 = min(右边界) - max(左边界) + 1
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -  # 两个框的右边界的最小值
                max(boxes[n, 0], query_boxes[k, 0]) + 1  # 两个框的左边界的最大值
            )
            
            # 只有当交集宽度>0时，才可能有重叠
            if iw > 0:
                # 计算交集区域的高度
                # 交集高度 = min(下边界) - max(上边界) + 1
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -  # 两个框的下边界的最小值
                    max(boxes[n, 1], query_boxes[k, 1]) + 1  # 两个框的上边界的最大值
                )
                
                # 只有当交集高度>0时，才真正有重叠
                if ih > 0:
                    # 计算交集面积
                    intersection = iw * ih
                    
                    # 计算并集面积 = 框1面积 + 框2面积 - 交集面积
                    box1_area = (boxes[n, 2] - boxes[n, 0] + 1) * (boxes[n, 3] - boxes[n, 1] + 1)
                    ua = float(box1_area + box_area - intersection)
                    
                    # 计算IoU：交集 / 并集
                    overlaps[n, k] = intersection / ua
    
    return overlaps