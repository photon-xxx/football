import os
import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
from collections import deque

# 绝对导入
from Storage.field_configs.soccer import SoccerPitchConfiguration

# 模型路径
PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PITCH_DETECTION_MODEL_PATH = "E:/Soccer/football/football_analysis-main/Storage/models/KeyPoint_detection/football-pitch-detection.pt"
# 全局配置
CONFIG = SoccerPitchConfiguration()

class KeyPointDetector:
    """
    球场关键点检测器
    使用 YOLO 模型检测球场上的关键点
    """

    def __init__(self, device: str = "cpu", window_size: int = 5, ema_alpha: float =0.7):   #TODO: 确定平滑参数
        """
        Args:
            device (str): 推理设备, 如 'cpu' 或 'cuda'
            window_size (int): 滑动平均的窗口大小
            ema_alpha (float): EMA的平滑系数，0<alpha<=1，越大越依赖最新值
        """
        self.model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
        self.device = device

        self.window_size = window_size
        self.ema_alpha = ema_alpha

        # 用于存储历史关键点
        self.history = deque(maxlen=window_size)
        self.ema_prev = None



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
        # print(keypoints)
        return keypoints

    def smooth_keypoints(self, xy: np.ndarray, method: str = "none") -> np.ndarray:
        """
        对关键点进行平滑处理，支持 NaN 占位的情况

        Args:
            xy (np.ndarray): 当前帧关键点，shape (N, 2)，可能包含 NaN
            method (str): 平滑方式 ["none", "moving", "ema"]

        Returns:
            np.ndarray: 平滑后的关键点 (N, 2)
        """
        if method == "none" or xy is None or len(xy) == 0:
            return xy

        if method == "moving":
            self.history.append(xy)
            stacked = np.stack(self.history, axis=0)  # (T, N, 2)

            # 对 NaN 做 nanmean，避免传染
            return np.nanmean(stacked, axis=0)

        elif method == "ema":
            if self.ema_prev is None:
                self.ema_prev = xy.copy()

            # 逐点处理：当前帧是 NaN 的点，用上一帧的值
            updated = np.where(
                np.isnan(xy),
                self.ema_prev,  # 保留上一次的
                self.ema_alpha * xy + (1 - self.ema_alpha) * self.ema_prev
            )
            self.ema_prev = updated
            return updated

        else:
            raise ValueError(f"Unsupported smoothing method: {method}")

    def detect_from_video(self, video_frames, stride: int = 1, conf_threshold: float = 0.6, smoothing: str = "moving"):  # 平滑可选
        """
        在视频帧中逐帧检测关键点 (生成器)

        Args:
            video_frames: 输入视频帧列表
            stride (int): 帧间隔，默认每帧都检测
            conf_threshold (float): 关键点置信度阈值，低于该值的点会被过滤,置为NaN
                                   // 置信度取得较高，矩阵的计算只需要最少四个点就够
            smoothing (str): 平滑方法 ["none", "moving", "ema"]

        Yields:
            (frame, keypoints, indices):
                frame: 原始帧
                keypoints: 过滤后的关键点对象，只是把低置信度的点置为NaN
                indices: 过滤后关键点对应的原始序号
        """
        for frame_idx, frame in enumerate(video_frames):
            if frame_idx % stride != 0:
                continue
            keypoints = self.detect_keypoints(frame)

            filtered_idx = None  # 默认没有过滤
            # 根据置信度过滤关键点
            if keypoints.confidence is not None:

                all_xy = np.full(keypoints.xy[0].shape, np.nan, dtype=np.float32)
                mask = keypoints.confidence[0] > conf_threshold
                all_xy[mask] = keypoints.xy[0][mask]

                filtered_idx = np.where(mask)[0]

                #平滑调用
                all_conf = np.full(keypoints.confidence.shape, np.nan, dtype=np.float32)
                all_conf[:, mask] = keypoints.confidence[:, mask]

                keypoints = sv.KeyPoints(
                    xy=all_xy[np.newaxis, ...],   # 低置信度点被置为NaN
                    confidence=all_conf[0:1]  # 取第一行，保持2D形状 (1, 32)
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
            keypoints (sv.KeyPoints): 已经过滤并保留 NaN 占位的关键点
            indices (np.ndarray): 关键点对应的原始序号

        Returns:
            np.ndarray: 带标注的图像
        """
        if keypoints is None or len(keypoints.xy[0]) == 0:
            return frame  # 没有点，直接返回

        annotated = frame.copy()

        # ✅ 逐点绘制
        for idx, (x, y) in enumerate(keypoints.xy[0]):
            if np.isnan(x) or np.isnan(y):
                continue  # 跳过 NaN

            # 红点
            cv2.circle(annotated, (int(x), int(y)), 6, (0, 0, 255), -1)

            # 标注编号
            cv2.putText(
                annotated,
                str(idx),  # 原始序号
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,  # 字号
                (0, 255, 0),  # 绿色文字
                1,
                cv2.LINE_AA
            )

        return annotated


# -------------------- 测试入口 --------------------
if __name__ == "__main__":
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="KeyPoint Detection for Football Pitch")
    parser.add_argument("--video", type=str, required=True, help="输入视频路径")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备: cpu/cuda")
    parser.add_argument("--save_video", action="store_true", help="是否保存测试结果视频")
    parser.add_argument("--output_dir", type=str, default="IO/output_videos/test", help="输出视频目录")
    args = parser.parse_args()

    detector = KeyPointDetector(device=args.device)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 获取输入视频文件名（不含扩展名）
    input_video_path = Path(args.video)
    video_name = input_video_path.stem
    output_video_path = output_dir / f"{video_name}_test.mp4"

    # 初始化视频写入器
    video_writer = None
    frame_count = 0

    print(f"开始处理视频: {args.video}")
    print(f"输出视频将保存到: {output_video_path}")

    

    try:
        for frame, keypoints, indices in detector.detect_from_video(args.video, stride=1):  # TODO： 修改为读取视频帧
            annotated = detector.visualize_keypoints(frame, keypoints, indices)
            
            # 初始化视频写入器（使用第一帧的尺寸）
            if video_writer is None and args.save_video:
                height, width = annotated.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(str(output_video_path), fourcc, 24.0, (width, height))
                print(f"视频写入器已初始化: {width}x{height}")
            
            # 保存帧到视频文件
            if video_writer is not None:
                video_writer.write(annotated)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"已处理 {frame_count} 帧")
            
            # 显示实时预览
            cv2.imshow("KeyPoints", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
                
    except KeyboardInterrupt:
        print("\n用户中断处理")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
    finally:
        # 清理资源
        if video_writer is not None:
            video_writer.release()
            print(f"视频已保存到: {output_video_path}")
            print(f"总共处理了 {frame_count} 帧")
        
        cv2.destroyAllWindows()
        print("处理完成")
