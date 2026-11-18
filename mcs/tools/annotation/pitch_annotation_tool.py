from dataclasses import dataclass, field
from typing import Optional, List, Union, Tuple
import cv2
import supervision as sv
import numpy as np
from soccer import *

def draw_pitch(
    config: Union[SoccerPitchConfiguration, HalfPitchConfiguration, QingYuanConfiguration, HalfQingYuanConfiguration, WuChuanConfiguration, HalfWuChuanConfiguration],
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1
) -> np.ndarray:
    """
    绘制足球场图像（自适应全场或半场）
    """
    # 1. 基础尺寸计算
    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)
    scaled_circle_radius = int(config.centre_circle_radius * scale)
    scaled_penalty_spot_distance = int(config.penalty_spot_distance * scale)

    # 判断是否为半场配置
    is_half_pitch = isinstance(config, (HalfPitchConfiguration, HalfQingYuanConfiguration, HalfWuChuanConfiguration))

    # 2. 创建画布
    pitch_image = np.ones(
        (scaled_width + 2 * padding,
         scaled_length + 2 * padding, 3),
        dtype=np.uint8
    ) * np.array(background_color.as_bgr(), dtype=np.uint8)

    # 3. 绘制线段 (基于 config.edges 和 config.vertices)
    for start, end in config.edges:
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

    # 4. 绘制中心圆 (根据类型区分)
    if is_half_pitch:
        # 半场模式：
        # 圆心 X 坐标 = 整个画布长度 (即中线位置)
        centre_circle_center = (
            scaled_length + padding,
            scaled_width // 2 + padding
        )
        # 绘制椭圆弧：从 90度 (下) 到 270度 (上)，即左半圆
        cv2.ellipse(
            img=pitch_image,
            center=centre_circle_center,
            axes=(scaled_circle_radius, scaled_circle_radius),
            angle=0,
            startAngle=90,
            endAngle=270,
            color=line_color.as_bgr(),
            thickness=line_thickness
        )
    else:
        # 全场模式：
        # 圆心 X 坐标 = 画布长度的一半
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

    # 5. 绘制点球点
    # 总是绘制左侧点球点
    penalty_spots = [
        (
            scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        )
    ]
    
    # 如果是全场，添加右侧点球点
    if not is_half_pitch:
        penalty_spots.append((
            scaled_length - scaled_penalty_spot_distance + padding,
            scaled_width // 2 + padding
        ))

    for spot in penalty_spots:
        cv2.circle(
            img=pitch_image,
            center=spot,
            radius=point_radius,
            color=line_color.as_bgr(),
            thickness=-1
        )

    return pitch_image


def draw_points_on_pitch(
    config: Union[SoccerPitchConfiguration, HalfPitchConfiguration, QingYuanConfiguration, HalfQingYuanConfiguration, WuChuanConfiguration, HalfWuChuanConfiguration],
    xy: np.ndarray,
    face_color: sv.Color = sv.Color.RED,
    edge_color: sv.Color = sv.Color.BLACK,
    radius: int = 5,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None,
    show_numbers: bool = True,
    font_scale: float = 0.5,
    number_color: Optional[sv.Color] = None
) -> np.ndarray:
    """
    在足球场上绘制点
    """
    if pitch is None:
        pitch = draw_pitch(config=config)

    if number_color is None:
        number_color = sv.Color.WHITE

    for idx, point in enumerate(xy):
        scaled_point = (
            int(point[0] * scale) + padding,
            int(point[1] * scale) + padding
        )
        cv2.circle(img=pitch, center=scaled_point, radius=radius,
                   color=face_color.as_bgr(), thickness=-1)
        cv2.circle(img=pitch, center=scaled_point, radius=radius,
                   color=edge_color.as_bgr(), thickness=thickness)

        if show_numbers:
            number_text = str(idx + 1)
            (text_width, text_height), baseline = cv2.getTextSize(
                number_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
            )
            text_x = scaled_point[0] + radius + 3
            text_y = scaled_point[1] - radius - 3

            cv2.rectangle(
                img=pitch,
                pt1=(text_x - 2, text_y - text_height - 2),
                pt2=(text_x + text_width + 2, text_y + baseline + 2),
                color=(0, 0, 0),
                thickness=-1
            )
            cv2.putText(
                img=pitch, text=number_text, org=(text_x, text_y),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale,
                color=number_color.as_bgr(), thickness=1, lineType=cv2.LINE_AA
            )

    return pitch


def draw_paths_on_pitch(
    config: Union[SoccerPitchConfiguration, HalfPitchConfiguration, QingYuanConfiguration, HalfQingYuanConfiguration, WuChuanConfiguration, HalfWuChuanConfiguration],
    paths: List[np.ndarray],
    color: sv.Color = sv.Color.WHITE,
    thickness: int = 2,
    padding: int = 50,
    scale: float = 0.1,
    pitch: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    在足球场上绘制路径
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    for path in paths:
        scaled_path = [
            (int(point[0] * scale) + padding, int(point[1] * scale) + padding)
            for point in path if point.size > 0
        ]

        if len(scaled_path) < 2:
            continue

        for i in range(len(scaled_path) - 1):
            cv2.line(img=pitch, pt1=scaled_path[i], pt2=scaled_path[i + 1],
                     color=color.as_bgr(), thickness=thickness)

    return pitch


def draw_pitch_voronoi_diagram(
    config: Union[SoccerPitchConfiguration, HalfPitchConfiguration, QingYuanConfiguration, HalfQingYuanConfiguration, WuChuanConfiguration, HalfWuChuanConfiguration],
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
    绘制 Voronoi 图
    """
    if pitch is None:
        pitch = draw_pitch(config=config, padding=padding, scale=scale)

    scaled_width = int(config.width * scale)
    scaled_length = int(config.length * scale)

    voronoi = np.zeros_like(pitch, dtype=np.uint8)
    team_1_color_bgr = np.array(team_1_color.as_bgr(), dtype=np.uint8)
    team_2_color_bgr = np.array(team_2_color.as_bgr(), dtype=np.uint8)

    y_coordinates, x_coordinates = np.indices((
        scaled_width + 2 * padding,
        scaled_length + 2 * padding
    ))

    y_coordinates -= padding
    x_coordinates -= padding

    def calculate_distances(xy, x_coordinates, y_coordinates):
        return np.sqrt((xy[:, 0][:, None, None] * scale - x_coordinates) ** 2 +
                       (xy[:, 1][:, None, None] * scale - y_coordinates) ** 2)

    distances_team_1 = calculate_distances(team_1_xy, x_coordinates, y_coordinates)
    distances_team_2 = calculate_distances(team_2_xy, x_coordinates, y_coordinates)

    min_distances_team_1 = np.min(distances_team_1, axis=0)
    min_distances_team_2 = np.min(distances_team_2, axis=0)

    control_mask = min_distances_team_1 < min_distances_team_2

    voronoi[control_mask] = team_1_color_bgr
    voronoi[~control_mask] = team_2_color_bgr

    overlay = cv2.addWeighted(voronoi, opacity, pitch, 1 - opacity, 0)

    return overlay


def draw_pitch_with_keypoints(
    config: Union[SoccerPitchConfiguration, HalfPitchConfiguration, QingYuanConfiguration, HalfQingYuanConfiguration, WuChuanConfiguration, HalfWuChuanConfiguration],
    background_color: sv.Color = sv.Color(34, 139, 34),
    line_color: sv.Color = sv.Color.WHITE,
    padding: int = 50,
    line_thickness: int = 4,
    point_radius: int = 8,
    scale: float = 0.1,
    keypoint_radius: int = 12,
    keypoint_thickness: int = 3,
    show_labels: bool = True,
    label_font_scale: float = 0.6,
    label_thickness: int = 2,
    label_offset: int = 20
) -> np.ndarray:
    """
    绘制带关键点标注的足球场图像
    """
    pitch_image = draw_pitch(
        config=config,
        background_color=background_color,
        line_color=line_color,
        padding=padding,
        line_thickness=line_thickness,
        point_radius=point_radius,
        scale=scale
    )
    
    vertices = config.vertices
    labels = config.labels
    colors = config.colors
    
    for idx, (vertex, label, color_hex) in enumerate(zip(vertices, labels, colors)):
        scaled_point = (
            int(vertex[0] * scale) + padding,
            int(vertex[1] * scale) + padding
        )
        
        color_rgb = tuple(int(color_hex[i:i+2], 16) for i in (1, 3, 5))
        color_bgr = (color_rgb[2], color_rgb[1], color_rgb[0])
        
        # 判断是否为边线分点（标签以 T 或 B 开头）
        is_sideline_point = label.startswith('T') or label.startswith('B')
        
        # 根据点的类型调整绘制参数
        if is_sideline_point:
            current_radius = max(4, keypoint_radius // 2)  # 边线分点使用更小的半径
            current_thickness = max(1, keypoint_thickness // 2)
            current_font_scale = max(0.3, label_font_scale * 0.5)  # 字体缩小到一半
            current_label_thickness = max(1, label_thickness // 2)
            current_label_offset = max(8, label_offset // 2)
        else:
            current_radius = keypoint_radius
            current_thickness = keypoint_thickness
            current_font_scale = label_font_scale
            current_label_thickness = label_thickness
            current_label_offset = label_offset
        
        cv2.circle(img=pitch_image, center=scaled_point, radius=current_radius,
                   color=color_bgr, thickness=-1)
        cv2.circle(img=pitch_image, center=scaled_point, radius=current_radius,
                   color=(255, 255, 255), thickness=current_thickness)
        
        if show_labels:
            label_position = (scaled_point[0] + current_label_offset, scaled_point[1] + current_label_offset)
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, current_font_scale, current_label_thickness
            )
            
            overlay = pitch_image.copy()
            cv2.rectangle(
                img=overlay,
                pt1=(label_position[0] - 5, label_position[1] - text_height - 5),
                pt2=(label_position[0] + text_width + 5, label_position[1] + baseline + 5),
                color=(0, 0, 0), thickness=-1
            )
            cv2.addWeighted(overlay, 0.6, pitch_image, 0.4, 0, pitch_image)
            
            cv2.putText(
                img=pitch_image, text=label, org=label_position,
                fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=current_font_scale,
                color=(255, 255, 255), thickness=current_label_thickness, lineType=cv2.LINE_AA
            )
    
    return pitch_image

if __name__ == "__main__":
    # 测试代码
    points = np.array([
        [521.49, 3835.4],
        [2700.9, 4994.7],
        [3212.8, 5533.3],
        [3015.7, 4780.8],
        [1911.9, 3554.7],
        [4041.8, 4152.8],
        [3113.5, 5922.6],
        [3093.4, 3425.7],
        [4131.8, 5817],
        [4072.4, 2341.6],
        [1873.2, 3670.8],
        [3664.6, 2019.8],
        [3729.2, 2157.7],
        [3871.6, 3237.6],
        [4131.1, 4816.9],
        [4396.8, 4689],
        [4087.1, 4762.4],
        [4117.7, 4745.7]
    ])

    config = CUMTBPitchConfiguration()
    annotated_frame = draw_pitch_with_keypoints(config)
    
    sv.plot_image(annotated_frame)