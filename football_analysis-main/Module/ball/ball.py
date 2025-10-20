# #######################################
# 将tracker得到的球坐标信息进行处理
# #######################################
import numpy as np
from typing import Dict, List, Any

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
                print("#########ball_info_test#########")
                print(ball_info["position_transformed"])
                print("#########ball_info_test#########")

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
