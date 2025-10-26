# #######################################
# 将tracker得到的球坐标信息进行处理
# #######################################
import numpy as np
from typing import Dict, List, Any, Union
from scipy.signal import savgol_filter

def extract_ball_paths(tracks: Dict[str, Any]) -> List[np.ndarray]:
    """
    从 tracks 数据中提取球的 'transformed_position' 信息，
    并返回可直接用于绘制的路径列表。

    Args:
        tracks (dict): 包含球、球员、裁判等追踪信息的字典结构

    Returns:
        List[np.ndarray]: 球路径列表，每个路径为形状 (N, 2) 的 NumPy 数组
    """
    ball_path = []

    # 遍历每一帧的球数据
    for frame_data in tracks.get("ball", []):
        for _, ball_info in frame_data.items():
            if "position_transformed" in ball_info:
                pos = ball_info["position_transformed"]
                
                # 检查位置数据是否有效
                if pos is not None and not (isinstance(pos, (list, np.ndarray)) and np.isnan(pos).any()):
                    ball_path.append(pos)
                else:
                    # 对于无效数据，可以选择跳过或使用NaN
                    ball_path.append([np.nan, np.nan])

    # 转换为 numpy 数组，并包装为列表（以兼容 draw_paths_on_pitch）
    if ball_path:
        try:
            # 尝试转换为numpy数组
            path_array = np.array(ball_path, dtype=np.float32)
            return [path_array]
        except ValueError as e:
            print(f"extract_ball_paths转换失败: {e}")
            # 如果转换失败，返回空列表
            return []
    else:
        return []

def interpolate_ball_positions_transformed(ball_paths): 
    """
    对球路径中的NaN值进行线性插值处理
    
    Args:
        ball_paths: 球路径列表，可能包含NaN值
        
    Returns:
        插值后的球路径列表，NaN值被替换为插值结果
    """
    from scipy.interpolate import interp1d
    
    ball_paths_interpolated = []
    
    for path in ball_paths:
        if len(path) == 0:
            ball_paths_interpolated.append(path)
            continue
            
        # 创建路径副本
        interpolated_path = path.copy().astype(np.float64)
        
        # 检查是否包含NaN值
        if not np.isnan(interpolated_path).any():
            # 没有NaN值，直接返回
            ball_paths_interpolated.append(interpolated_path)
            continue
        
        # 分离x和y坐标
        x_coords = interpolated_path[:, 0]
        y_coords = interpolated_path[:, 1]
        
        # 找到有效点的索引
        valid_x_mask = ~np.isnan(x_coords)
        valid_y_mask = ~np.isnan(y_coords)
        
        # 如果x坐标有有效值，进行插值
        if np.any(valid_x_mask):
            valid_x_indices = np.where(valid_x_mask)[0]
            if len(valid_x_indices) > 1:  # 需要至少2个点才能插值
                try:
                    # 对x坐标进行线性插值
                    f_x = interp1d(valid_x_indices, x_coords[valid_x_indices], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
                    x_coords = f_x(np.arange(len(x_coords)))
                except Exception as e:
                    print(f"X坐标插值失败: {e}")
        
        # 如果y坐标有有效值，进行插值
        if np.any(valid_y_mask):
            valid_y_indices = np.where(valid_y_mask)[0]
            if len(valid_y_indices) > 1:  # 需要至少2个点才能插值
                try:
                    # 对y坐标进行线性插值
                    f_y = interp1d(valid_y_indices, y_coords[valid_y_indices], 
                                 kind='linear', bounds_error=False, fill_value='extrapolate')
                    y_coords = f_y(np.arange(len(y_coords)))
                except Exception as e:
                    print(f"Y坐标插值失败: {e}")
        
        # 重新组合坐标
        interpolated_path[:, 0] = x_coords
        interpolated_path[:, 1] = y_coords
        
        ball_paths_interpolated.append(interpolated_path)
    
    return ball_paths_interpolated

def replace_outliers_based_on_distance(
    ball_paths: List[np.ndarray],
    distance_threshold: float =200.0   # 需要根据实际情况调整阈值
) -> List[np.ndarray]:
    """
    根据与上一个有效点的距离，剔除异常点。
    若当前点与上个点距离超过阈值，则认为该点异常，用空数组替代。
    
    Args:
        ball_paths: 球路径列表，每个路径为形状 (N, 2) 的 NumPy 数组
        distance_threshold: 距离阈值，超过此距离的点将被视为异常点
        
    Returns:
        处理后的球路径列表，异常点被替换为 [np.nan, np.nan]
    """
    ball_paths_processed = []
    
    for path in ball_paths:
        if len(path) == 0:
            ball_paths_processed.append(path)
            continue
            
        # 创建路径副本以避免修改原始数据，确保数据类型为float
        processed_path = path.copy().astype(np.float64)
        
        # 遍历轨迹中的坐标（从第二个点开始，到倒数第二个点结束）
        for i in range(1, len(processed_path) - 1):
            current_point = processed_path[i]
            prev_point = processed_path[i-1]
            next_point = processed_path[i+1]
            
            # 跳过已经是NaN的点
            if np.isnan(current_point).any() or np.isnan(prev_point).any() or np.isnan(next_point).any():
                continue
            
            # 计算当前点与前一个点、后一个点之间的距离
            dist_to_prev = np.linalg.norm(current_point - prev_point)
            dist_to_next = np.linalg.norm(current_point - next_point)
            dist_prev_to_next = np.linalg.norm(next_point - prev_point)
            
            # 如果当前点与前一个点、后一个点之间的距离都超过阈值，
            # 且前一个点和后一个点之间的距离小于阈值，则认为该点异常
            if (dist_to_prev > distance_threshold and 
                dist_to_next > distance_threshold and 
                dist_prev_to_next < distance_threshold):
                # 用[np.nan,np.nan]替代异常点
                processed_path[i] = [np.nan, np.nan]
        
        ball_paths_processed.append(processed_path)
    
    return ball_paths_processed

def ball_filter(trajectory, window_length=5, polyorder=2):
    """
    Savitzky-Golay滤波
    
    Args:
        trajectory: 可以是单个np.ndarray或List[np.ndarray]
        window_length: 滤波窗口长度
        polyorder: 多项式阶数
    
    Returns:
        滤波后的轨迹，格式与输入相同
    """
    from scipy.signal import savgol_filter
    
    # 如果输入是列表，处理每个路径
    if isinstance(trajectory, list):
        filtered_paths = []
        for path in trajectory:
            if len(path) == 0:
                # 空路径直接返回
                filtered_paths.append(path)
            elif len(path) < window_length:
                # 路径太短，直接返回
                filtered_paths.append(path)
            else:
                # 应用Savitzky-Golay滤波
                try:
                    x_filtered = savgol_filter(path[:, 0], window_length, polyorder)
                    y_filtered = savgol_filter(path[:, 1], window_length, polyorder)
                    filtered_path = np.column_stack([x_filtered, y_filtered])
                    filtered_paths.append(filtered_path)
                except Exception as e:
                    print(f"滤波失败: {e}")
                    filtered_paths.append(path)  # 失败时返回原路径
        
        return filtered_paths
    
    # 如果输入是单个numpy数组
    elif isinstance(trajectory, np.ndarray):
        if len(trajectory) == 0:
            return trajectory
        elif len(trajectory) < window_length:
            return trajectory
        else:
            try:
                x_filtered = savgol_filter(trajectory[:, 0], window_length, polyorder)
                y_filtered = savgol_filter(trajectory[:, 1], window_length, polyorder)
                return np.column_stack([x_filtered, y_filtered])
            except Exception as e:
                print(f"滤波失败: {e}")
                return trajectory
    
    else:
        raise ValueError(f"不支持的输入类型: {type(trajectory)}")


# function TODO：5hz采样    