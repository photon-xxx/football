# #######################################
# 将tracker得到的球坐标信息进行处理
# #######################################
import numpy as np
from typing import Dict, List, Any, Union

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
            # print("#########ball_info_test#########")
            # print(ball_info)
            # print("#########ball_info_test#########")
            if "position_transformed" in ball_info:

                #测试
                # print("#########ball_info_test#########")
                # print(ball_info["position_transformed"])
                # print("#########ball_info_test#########")

                ball_path.append(ball_info["position_transformed"])

    # 转换为 numpy 数组，并包装为列表（以兼容 draw_paths_on_pitch）
    #测试
    # print("#########ball_path_test#########")
    # print(ball_path)
    # print("#########ball_path_test#########")

    if ball_path:
        return [np.array(ball_path, dtype=np.float32)]
    else:
        return []

def replace_outliers_based_on_distance(
    positions: List[np.ndarray],
    distance_threshold: float =500.0
) -> List[np.ndarray]:
    """
    根据与上一个有效点的距离，剔除异常点。
    若当前点与上个点距离超过阈值，则认为该点异常，用空数组替代。
    """

    last_valid_position: Union[np.ndarray, None] = None  # 上一个有效位置
    cleaned_positions: List[np.ndarray] = []             # 清洗后的坐标列表

    print("#########positions_test#########")
    print("positions type:", type(positions))
    print("positions[0] type:", type(positions[0]))
    print("positions[0] shape:", positions[0].shape if hasattr(positions[0], 'shape') else 'no shape')
    print("#########positions_test#########")

    positions_count = 0
    for position in positions[0]:
        positions_count += 1
        if len(position) == 0:
            # 若当前帧没有检测到球，则直接保留空数组
            cleaned_positions.append(position)

            print("########当前帧没有检测到球##########")
        else:
            if last_valid_position is None:
                # 第一个有效位置直接保留
                print("#########第一个有效位置直接保留##########")
                cleaned_positions.append(position)
                last_valid_position = position
            else:
                # 计算与上一个有效点的距离
                distance = np.linalg.norm(position - last_valid_position)

                # print("#########distance_test#########")
                # print("position:", position)
                # print("last_valid_position:", last_valid_position)
                # print("distance calculation:", np.linalg.norm(position - last_valid_position))
                # print("#########distance_test#########")

                if distance > distance_threshold:
                    
                    print("position_count:", positions_count)
                    print("position:", position) 
                    print("last_valid_position:", last_valid_position) 
                    print("distance:", distance)
                    print("########################################################") 

                    # 若距离过大（跳跃异常），置为空
                    cleaned_positions.append(np.array([], dtype=np.float64))
                else:
                    # 否则认为是正常点，保留并更新last_valid_position
                    cleaned_positions.append(position)
                    last_valid_position = position

    # print("#########positions_count_test#########")
    # print(positions_count)
    # print("#########positions_count_test#########")
    # 修改返回部分
    if cleaned_positions:
        # 过滤掉空数组，只保留有效坐标
        valid_positions = [pos for pos in cleaned_positions if len(pos) > 0]
        if valid_positions:
            return [np.array(valid_positions, dtype=np.float32)]
        else:
            return []
    else:
        return []

    #TODO:　空数组插值


# function TODO：5hz采样    