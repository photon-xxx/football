"""
足球场可视化工具模块

这个模块提供了绘制足球场、在球场上绘制点、路径和Voronoi图的功能。
主要用于足球分析中的场地可视化，包括球员位置、运动轨迹和区域控制分析。

"""

from typing import Optional, List

import cv2
import supervision as sv
import numpy as np

from Storage.field_configs.soccer import SoccerPitchConfiguration


def draw_pitch(
    config: SoccerPitchConfiguration,
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    绘制标准足球场图像
    
    根据配置参数绘制一个完整的足球场，包括边界线、中心圆、罚球点等所有标准元素。
    这个函数是其他可视化功能的基础，用于创建球场背景。
    
    Args:
        config (SoccerPitchConfiguration): 足球场配置对象，包含球场的尺寸和布局信息
        background_color (sv.Color, optional): 球场背景颜色，默认为绿色(34, 139, 34)
        line_color (sv.Color, optional): 球场线条颜色，默认为白色
        padding (int, optional): 球场周围的填充像素，默认为50
        line_thickness (int, optional): 线条粗细（像素），默认为4
        point_radius (int, optional): 罚球点半径（像素），默认为8
        scale (float, optional): 球场尺寸缩放比例，默认为0.1

    Returns:
        np.ndarray: 足球场的图像数组，形状为(height, width, 3)
    """
    # 根据缩放比例计算球场的实际尺寸
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    # 创建球场背景图像，使用指定的背景颜色
    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # 绘制球场的所有边界线
    for start, end in config.edges:
        # 计算线条的起点和终点坐标，并应用缩放和填充
        point1 = (int(config.vertices[start - 1][0] * scale) + padding,
                  int(config.vertices[start - 1][1] * scale) + padding)
        point2 = (int(config.vertices[end - 1][0] * scale) + padding,
                  int(config.vertices[end - 1][1] * scale) + padding)
        cv2.line(
            img=pitch_image,
            pt1=point1,
            pt2=point2,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )

    # 绘制中心圆
    centre_circle_center = (
        scaled_length // 2 + padding,
        scaled_width // 2 + padding
    )
    cv2.circle(
        img=pitch_image,
        center=centre_circle_center,
        radius=scaled_circle_radius,
        color=line_color.as_bgr(),
        thickness=line_thickness
    )

    # 绘制两个罚球点
    penalty_spots = [
        (
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        ),
        (
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        )
    ]
    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1  # -1表示填充整个圆
        )

    return pitch_image


def draw_points_on_pitch(
    config: SoccerPitchConfiguration,
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 10,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    在足球场上绘制点（如球员位置、球的位置等）
    
    这个函数用于在球场上标记特定的位置点，比如球员位置、球的位置等。
    每个点都有填充色和边框色，可以自定义大小和颜色。
    
    Args:
        config (SoccerPitchConfiguration): 足球场配置对象，包含球场的尺寸和布局信息
        xy (np.ndarray): 要绘制的点坐标数组，每个点包含(x, y)坐标
        face_color (sv.Color, optional): 点的填充颜色，默认为红色
        edge_color (sv.Color, optional): 点的边框颜色，默认为黑色
        radius (int, optional): 点的半径（像素），默认为10
        thickness (int, optional): 点边框的粗细（像素），默认为2
        padding (int, optional): 球场周围的填充像素，默认为50
        scale (float, optional): 球场尺寸缩放比例，默认为0.1
        pitch (Optional[np.ndarray], optional): 现有的球场图像，如果为None则创建新的球场

    Returns:
        np.ndarray: 带有标记点的足球场图像
    """
    # 如果没有提供现有的球场图像，则创建一个新的
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    # 遍历所有要绘制的点
    for point in xy:
        # 将点坐标按比例缩放并添加填充偏移
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        # 绘制填充的圆（点的内部）
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=face_color.as_bgr(),
            thickness=-1  # -1表示填充整个圆
        )
        # 绘制边框圆（点的边缘）
        cv2.circle(
            img=pitch,
            center=scaled_point,
            radius=radius,
            color=edge_color.as_bgr(),
            thickness=thickness
        )

    return pitch


def draw_paths_on_pitch(
    config: SoccerPitchConfiguration,
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    在足球场上绘制路径（如球员运动轨迹、球的运动路径等）
    
    这个函数用于在球场上绘制运动轨迹，比如球员的跑动路径、球的运动轨迹等。
    可以同时绘制多条路径，每条路径用连续的线条表示。
    
    Args:
        config (SoccerPitchConfiguration): 足球场配置对象，包含球场的尺寸和布局信息
        paths (List[np.ndarray]): 路径列表，每个路径是一个包含(x, y)坐标的数组
        color (sv.Color, optional): 路径线条颜色，默认为白色
        thickness (int, optional): 路径线条粗细（像素），默认为2
        padding (int, optional): 球场周围的填充像素，默认为50
        scale (float, optional): 球场尺寸缩放比例，默认为0.1
        pitch (Optional[np.ndarray], optional): 现有的球场图像，如果为None则创建新的球场

    Returns:
        np.ndarray: 带有路径的足球场图像
    """
    # 如果没有提供现有的球场图像，则创建一个新的
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    # 遍历所有要绘制的路径
    for path in paths:
        # 将路径中的每个点按比例缩放并添加填充偏移
        scaled_path = [
            (
                int(point[0] * scale) + padding,
                int(point[1] * scale) + padding
            )
            for point in path if point.size > 0  # 过滤掉空点
        ]

        # 如果路径中的点少于2个，则跳过这条路径
        if len(scaled_path) < 2:
            continue

        # 将路径中的相邻点用线条连接起来
        for i in range(len(scaled_path) - 1):
            cv2.line(
                img=pitch,
                pt1=scaled_path[i],
                pt2=scaled_path[i + 1],
                color=color.as_bgr(),
                thickness=thickness
            )

    return pitch


def draw_pitch_voronoi_diagram(
    config: SoccerPitchConfiguration,
    team_1_xy: np.ndarray,
    team_2_xy: np.ndarray,
    team_1_color: sv.Color = sv.Color.RED,
    team_2_color: sv.Color = sv.Color.WHITE,
    opacity: float = 0.5,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    在足球场上绘制Voronoi图，显示两队球员的控制区域
    
    Voronoi图是一种空间分割方法，将球场分割成不同的区域，每个区域由最近的球员控制。
    这个函数用于分析两队球员在球场上的控制区域分布，是战术分析的重要工具。
    
    Args:
        config (SoccerPitchConfiguration): 足球场配置对象，包含球场的尺寸和布局信息
        team_1_xy (np.ndarray): 第一队球员的位置坐标数组，每个元素包含(x, y)坐标
        team_2_xy (np.ndarray): 第二队球员的位置坐标数组，每个元素包含(x, y)坐标
        team_1_color (sv.Color, optional): 第一队控制区域的颜色，默认为红色
        team_2_color (sv.Color, optional): 第二队控制区域的颜色，默认为白色
        opacity (float, optional): Voronoi图的透明度，范围0-1，默认为0.5
        padding (int, optional): 球场周围的填充像素，默认为50
        scale (float, optional): 球场尺寸缩放比例，默认为0.1
        pitch (Optional[np.ndarray], optional): 现有的球场图像，如果为None则创建新的球场

    Returns:
        np.ndarray: 带有Voronoi图的足球场图像
    """
    # 如果没有提供现有的球场图像，则创建一个新的
    if pitch is None:
        pitch = draw_pitch(
            config=config,
            padding=padding,
            scale=scale
        )

    # 计算缩放后的球场尺寸
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    # 创建Voronoi图的画布
    voronoi = np.zeros_like(pitch, dtype=np.uint8)

    # 将颜色转换为BGR格式的numpy数组
    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    # 创建球场坐标网格
    y_coordinates, x_coordinates = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    # 调整坐标，使其相对于球场中心
    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        """
        计算每个球员到球场每个像素点的距离
        
        Args:
            xy: 球员位置坐标数组
            x_coordinates: 球场x坐标网格
            y_coordinates: 球场y坐标网格
            
        Returns:
            np.ndarray: 距离矩阵，形状为(球员数量, 球场高度, 球场宽度)
        """
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)

    # 计算每个队球员到球场每个像素点的距离
    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    # 找到每个像素点到最近球员的距离
    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    # 确定每个像素点属于哪个队的控制区域
    control_mask = min_distances_team_1 < min_distances_team_2

    # 根据控制区域分配颜色
    voronoi[control_mask] = team_1_color_bgr
    voronoi[~control_mask] = team_2_color_bgr

    # 将Voronoi图与球场背景混合，应用透明度
    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay


if __name__ == "__main__":
    CONFIG = SoccerPitchConfiguration()
    annotated_frame = draw_pitch(CONFIG)
    sv.plot_image(annotated_frame)