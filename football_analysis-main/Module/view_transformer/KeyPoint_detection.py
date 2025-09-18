import os
import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO

# 绝对导入
from Storage.field_configs.soccer import SoccerPitchConfiguration

# 模型路径
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR,
                                          "E:/Soccer/football/football_analysis-main/Storage/models/KeyPoint_detection/football-pitch-detection.pt")

# 全局配置
CONFIG = SoccerPitchConfiguration()

class KeyPointDetector:
    """
    球场关键点检测器
    使用 YOLO 模型检测球场上的关键点
    """

    def __init__(self, device: str = "cpu"):
        """
        Args:
            device (str): 推理设备, 如 'cpu' 或 'cuda'
        """
        self.model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
        self.device = device

    def detect_keypoints(self, frame: np.ndarray) -> sv.KeyPoints:
        """
        在单帧图像上检测球场关键点

        Args:
            frame (np.ndarray): 输入图像 (H, W, C)

        Returns:
            sv.KeyPoints: 检测到的关键点
        """
        result = self.model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        # 调试
        print(keypoints)
        return keypoints

    def detect_from_video(self, video_path: str, stride: int = 1, conf_threshold: float = 0.7):  # TODO: 关键点的平滑处理
        """
        在视频中逐帧检测关键点 (生成器)

        Args:
            video_path (str): 输入视频路径
            stride (int): 帧间隔，默认每帧都检测
            conf_threshold (float): 关键点置信度阈值，低于该值的点会被过滤
                                   // 置信度取得较高，矩阵的计算只需要最少四个点就够

        Yields:
            (frame, keypoints, indices):
                frame: 原始帧
                keypoints: 过滤后的关键点对象
                indices: 过滤后关键点对应的原始序号
        """
        frame_gen = sv.get_video_frames_generator(source_path=video_path, stride=stride)
        for frame in frame_gen:
            keypoints = self.detect_keypoints(frame)

            filtered_idx = None  # 默认没有过滤
            # 根据置信度过滤关键点
            if keypoints.confidence is not None:
                mask = keypoints.confidence[0] > conf_threshold
                filtered_xy = keypoints.xy[0][mask]
                filtered_idx = np.where(mask)[0]

                keypoints = sv.KeyPoints(
                    xy=filtered_xy[np.newaxis, ...],
                    confidence=keypoints.confidence[:, mask] if keypoints.confidence is not None else None
                )

            yield frame, keypoints, filtered_idx

    def visualize_keypoints(
            self,
            frame: np.ndarray,
            keypoints: sv.KeyPoints,
            indices: np.ndarray = None
    ) -> np.ndarray:
        """
        在图像上绘制检测到的关键点（红点 + 序号）

        Args:
            frame (np.ndarray): 输入图像
            keypoints (sv.KeyPoints): 已经过滤的关键点
            indices (np.ndarray): 关键点对应的原始序号（来自 detect_from_video 的 filtered_idx）

        Returns:
            np.ndarray: 带标注的图像
        """
        if keypoints is None or len(keypoints.xy[0]) == 0:
            return frame  # 没有点，直接返回

        annotated = frame.copy()

        # ✅ 先画红点
        vertex_annotator = sv.VertexAnnotator(
            color=sv.Color.from_hex("#FF0000"),
            radius=6
        )
        annotated = vertex_annotator.annotate(annotated, keypoints)

        # ✅ 再标注编号
        if indices is not None:
            for (x, y), idx in zip(keypoints.xy[0], indices):
                cv2.putText(
                    annotated,
                    str(idx),  # 原始序号
                    (int(x) + 5, int(y) - 5),  # 点的右上方
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,  # 字号
                    (0, 255, 0),  # 绿色文字
                    1,  # 线宽
                    cv2.LINE_AA
                )

        return annotated


# -------------------- 测试入口 --------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="KeyPoint Detection for Football Pitch")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备: cpu/cuda")
    args = parser.parse_args()

    detector = KeyPointDetector(device=args.device)

    for frame, keypoints, indices in detector.detect_from_video(args.video, stride=1):  # TODO：stride 选取适当的数值
        annotated = detector.visualize_keypoints(frame, keypoints, indices)
        cv2.imshow("KeyPoints", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()
