import numpy as np
from typing import List, Tuple
import cv2
def get_all_foot_positions(detection):
    """
    从YOLO检测结果中提取所有球员的脚部位置

    Args:
        detection: ultralytics.engine.results.Results对象，包含YOLO模型推理结果

    Returns:
        List[Tuple[int, int]]: 包含所有球员脚部像素坐标的列表，每个元素为(x, y)
    """
    foot_positions = []
    for bbox in detection[0].boxes.xyxy.cpu().numpy():
        # 从Results对象中提取bbox（xyxy格式）
        foot_pos = get_foot_position(bbox)
        foot_positions.append(foot_pos)

    return foot_positions


def get_all_foot_positions_yolo(results_boxes_xyxy):
    """
    从YOLO检测结果中提取所有球员的脚部位置

    Args:
        results_boxes_xyxy: bbox列表，每个元素为[x1, y1, x2, y2]

    Returns:
        List[Tuple[int, int]]: 包含所有球员脚部像素坐标的列表，每个元素为(x, y)
    """
    foot_positions = []
    for result in results_boxes_xyxy:
        # 从Results对象中提取bbox（xyxy格式）
        foot_pos = get_foot_position(result)
        foot_positions.append(foot_pos)

    return foot_positions

def get_all_foot_positions_xywh(results_boxes_xywh):
    """
    从xywh格式的bbox列表提取所有球员的脚部位置

    Args:
        results_boxes_xywh: bbox列表，每个元素为[x_center, y_center, width, height]
                            或者是 [x_top_left, y_top_left, width, height]，取决于具体定义。
                            通常yolo输出的xywh是中心点坐标和宽高。
                            如果是左上角xywh，则计算方式不同。
                            此处假设输入为 [x1, y1, w, h] (左上角坐标 + 宽高)，
                            或者 [cx, cy, w, h] (中心坐标 + 宽高)。
                            
                            如果是 [x1, y1, w, h] (top-left):
                              foot_x = x1 + w/2
                              foot_y = y1 + h
                            
                            如果是 [cx, cy, w, h] (center):
                              foot_x = cx
                              foot_y = cy + h/2

    Returns:
        List[Tuple[int, int]]: 包含所有球员脚部像素坐标的列表，每个元素为(x, y)
    """
    foot_positions = []
    for result in results_boxes_xywh:
        if len(result) >= 4:
            x, y, w, h = result[:4]
            # 这里假设是 [x1, y1, w, h] (左上角格式)，因为这是比较通用的 xywh 理解
            # 如果是中心点格式，可以根据需要修改
            foot_x = int(x + w / 2)
            foot_y = int(y + h)
            foot_positions.append((foot_x, foot_y))
            
    return foot_positions

def draw_foot_positions(
        image: np.ndarray,
        foot_positions: List[Tuple[int, int]],
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 8,
        thickness: int = 2,
        show_numbers: bool = True
) -> np.ndarray:
    """
    在图像上标注脚部位置

    Args:
        image: 原始图像帧（numpy数组）
        foot_positions: 脚部位置列表，每个元素为(x, y)坐标
        color: 点的颜色，BGR格式，默认为绿色(0, 255, 0)
        radius: 点的半径（像素），默认为8
        thickness: 点边框的粗细（像素），默认为2
        show_numbers: 是否显示点编号，默认为True

    Returns:
        np.ndarray: 标注了脚部位置的图像
    """
    annotated_image = image.copy()

    for i, (x, y) in enumerate(foot_positions):
        # 绘制填充圆
        cv2.circle(annotated_image, (int(x), int(y)), radius, color, -1)

        # 绘制边框圆
        cv2.circle(annotated_image, (int(x), int(y)), radius, (255, 255, 255), thickness)

        # 显示点编号
        if show_numbers:
            cv2.putText(
                annotated_image,
                str(i + 1),
                (int(x) + radius + 5, int(y) - radius - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

    return annotated_image


def draw_player_detection(detections, frame):
    frame = frame.copy()

    for r in detections:
        boxes = r.boxes  # YOLO 的 Boxes 对象
        for box in boxes:
            # 坐标
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            # 置信度和类别
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = f"{r.names[cls]} {conf:.2f}"

            # 画框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # 画文字
            cv2.putText(frame, label, (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 显示结果
    # sv.plot_image(frame)
    return frame

def get_center_of_bbox(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int((y1+y2)/2)

def get_bbox_width(bbox):
    return bbox[2]-bbox[0]

def measure_distance(p1,p2):
    return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)**0.5

def measure_xy_distance(p1,p2):
    return p1[0]-p2[0],p1[1]-p2[1]

def get_foot_position(bbox):
    x1,y1,x2,y2 = bbox
    return int((x1+x2)/2),int(y2)