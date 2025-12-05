import cv2
import numpy as np
from pathlib import Path
from typing import Union

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

def save_video(ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()


def load_frame(image_path: Union[str, Path]) -> np.ndarray:
    """
    读取图片文件为frame（numpy数组）

    Args:
        image_path: 图片文件路径（支持str或Path对象）

    Returns:
        np.ndarray: 图像数组，形状为(H, W, 3)，BGR格式
                   如果读取失败返回None

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 无法读取图像
    """
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")

    frame = cv2.imread(str(image_path))

    if frame is None:
        raise ValueError(f"无法读取图像文件: {image_path}")

    return frame


def save_frame(frame: np.ndarray, output_path: Union[str, Path]) -> bool:
    """
    保存frame为图片文件

    Args:
        frame: 图像数组，形状为(H, W, 3)，BGR格式
        output_path: 输出图片文件路径（支持str或Path对象）
                    支持格式：.jpg, .png, .bmp等

    Returns:
        bool: 保存成功返回True，失败返回False

    Raises:
        ValueError: frame为空或格式不正确
    """
    if frame is None or frame.size == 0:
        raise ValueError("frame为空，无法保存")

    output_path = Path(output_path)

    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存图片
    success = cv2.imwrite(str(output_path), frame)

    if not success:
        raise ValueError(f"保存图像失败: {output_path}")

    return success