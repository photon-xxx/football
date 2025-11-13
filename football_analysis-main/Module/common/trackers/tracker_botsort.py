from ultralytics import YOLO
import supervision as sv
import pickle
import numpy as np
import pandas as pd
import os
import sys
import cv2
import torch

# 修复numpy版本兼容性问题 (numpy >= 1.24移除了np.object)
if not hasattr(np, "object"):
    np.object = object

sys.path.append('../../../')
from Module.common.utils import get_center_of_bbox, get_bbox_width, get_foot_position
from Module.common.trackers.bot_sort.bot_sort import BoTSORT

default_reid_path = "E:/Soccer/football/football_analysis-main/Storage/models/reid/osnet_x0_25_msmt17.pt"

# TODO：数据流优化，批次处理多帧，缓存文件销毁，减少资源占用

class Tracker:
    """使用BoTSORT算法的跟踪器类，接口与tracker.py中的Tracker类完全兼容"""
    
    def __init__(self, model_path, nms_threshold=0.5, reid_model_path=default_reid_path, device='cpu', fp16=False):
        """
        初始化跟踪器
        
        Args:
            model_path: YOLO模型路径
            nms_threshold: NMS阈值
            reid_model_path: ReID模型路径（可选，如果为None则不使用ReID特征）
            device: 计算设备 'cpu' 或 'cuda'
            fp16: 是否使用半精度推理
        """
        self.model = YOLO(model_path)
        self.nms_threshold = nms_threshold
        
        # 初始化BoTSORT跟踪器      
        try:
            self.tracker = BoTSORT(
                model_weights=reid_model_path if reid_model_path else "",
                device=device,
                fp16=fp16,
                track_high_thresh=0.3,  # 与YOLO的conf=0.3对应
                new_track_thresh=0.4,
                track_buffer=30,
                match_thresh=0.8,
                proximity_thresh=0.5,
                appearance_thresh=0.25,
                cmc_method='sparseOptFlow',
                frame_rate=30
            )
        except Exception as e:
            # 如果BoTSORT初始化失败（可能是ReID模型问题），尝试不使用ReID
            raise RuntimeError(f"警告: BoTSORT初始化时出错: {e}")


    def add_position_to_tracks(self, tracks):
        """给轨迹增加位置点信息"""
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info['bbox']
                    
                    if np.isnan(bbox).any():
                        position = (np.nan, np.nan)
                    else:
                        if object == 'ball':
                            position = get_center_of_bbox(bbox)
                        else:
                            position = get_foot_position(bbox)
        
                    tracks[object][frame_num][track_id]['position'] = position

    def interpolate_ball_positions(self, ball_positions):
        """处理将球位置置为NAN的帧"""
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]
        return ball_positions

    def detect_frames(self, frames, batch_size=12):
        """处理批次控制"""
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            results = self.model.predict(batch_frames, conf=0.3)
            detections += results
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """获取目标跟踪结果"""
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames)

        tracks = {
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)

            # 处理守门员类别
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]

            # 分离球和其他对象
            BALL_ID = cls_names_inv['ball']
            ball_detections = detection_supervision[detection_supervision.class_id == BALL_ID]
            all_detections = detection_supervision[detection_supervision.class_id != BALL_ID]

            # 对球进行边界框扩展
            if len(ball_detections) > 0:
                ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            # 对非球对象应用NMS
            all_detections = all_detections.with_nms(threshold=self.nms_threshold, class_agnostic=True)
            
            # 转换检测结果为BoTSORT格式 [x1, y1, x2, y2, conf, cls, tracklab_id]
            frame = frames[frame_num]
            if len(all_detections) > 0:
                # 准备BoTSORT输入格式
                output_results = []
                for i in range(len(all_detections)):
                    bbox = all_detections.xyxy[i]
                    conf = all_detections.confidence[i]
                    cls = all_detections.class_id[i] - 1  # Roboflow类别ID调整
                    tracklab_id = 0  # 默认值
                    output_results.append([
                        bbox[0], bbox[1], bbox[2], bbox[3],  # x1, y1, x2, y2
                        conf, cls, tracklab_id
                    ])
                output_results = np.array(output_results)
                
                # 转换为torch tensor（BoTSORT需要）
                if isinstance(output_results, np.ndarray):
                    output_results = torch.from_numpy(output_results).float()
                
                # 使用BoTSORT更新跟踪
                tracking_results = self.tracker.update(output_results, frame)
            else:
                tracking_results = []

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            # 存储追踪结果（球员和裁判）
            for result in tracking_results:
                # BoTSORT返回格式: [x1, y1, x2, y2, track_id, cls, score, tracklab_id]
                x1, y1, x2, y2 = float(result[0]), float(result[1]), float(result[2]), float(result[3])
                track_id = int(result[4])
                cls_id = int(result[5])
                
                bbox = [x1, y1, x2, y2]

                # 根据调整后的类别ID判断对象类型
                player_id = cls_names_inv['player'] - 1
                referee_id = cls_names_inv['referee'] - 1

                if cls_id == player_id:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == referee_id:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # 存储球（不追踪，使用扩展后的边界框）
            if len(ball_detections) > 0:
                for frame_detection in ball_detections:
                    bbox = frame_detection[0].tolist()
                    tracks["ball"][frame_num][1] = {"bbox": bbox}
                    break  # 只取第一个球的检测
            else:
                tracks["ball"][frame_num][1] = {"bbox": [np.nan, np.nan, np.nan, np.nan]}

        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

# ---------------------------------------------------------------------------------------------------------
# 主函数测试
if __name__ == "__main__":
    import argparse
    import time
    import cv2
    
    parser = argparse.ArgumentParser(description="实时追踪测试")
    parser.add_argument("--video", type=str, default="IO/input_videos/clip_30s.mp4", 
                       help="输入视频路径")
    parser.add_argument("--model", type=str, default="Storage/models/Player_detection/football-player-detection.pt", 
                       help="YOLO模型路径")
    parser.add_argument("--fps", action="store_true", 
                       help="显示FPS信息")
    parser.add_argument("--save", default=True, action="store_true", 
                       help="保存追踪结果视频")
    parser.add_argument("--output", type=str, default="E:\Soccer\\football\\football_analysis-main\IO\output_videos\\track_test", 
                       help="输出目录")
    parser.add_argument("--nms", type=float, default=0.5, 
                       help="NMS阈值，用于减少重复检测 (0.1-0.9)")
    parser.add_argument("--reid", type=str, default=default_reid_path,
                       help="ReID模型路径")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cpu", "cuda"],
                       help="计算设备 (cpu 或 cuda)")
    parser.add_argument("--fp16", default=True, action="store_true",
                       help="使用半精度推理 (仅GPU)")
    
    args = parser.parse_args()
    
    # 创建追踪器实例（使用BoTSORT）
    try:
        tracker = Tracker(
            model_path=args.model,
            nms_threshold=args.nms,
            reid_model_path=args.reid if args.reid else None,
            device=args.device,
            fp16=args.fp16
        )
    except Exception as e:
        print(f"错误: 无法初始化追踪器: {e}")
        exit(1)
    
    print("=" * 80)
    print("实时追踪测试")
    print("=" * 80)
    print(f"视频路径: {args.video}")
    print(f"模型路径: {args.model}")
    print(f"保存视频: {'是' if args.save else '否'}")
    if args.save:
        print(f"输出目录: {args.output}")
    print("按 'q' 键退出，按 'p' 键暂停/继续")
    print("=" * 80)

    # 初始化视频捕获
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"错误: 无法打开视频文件 {args.video}")
        exit(1)

    # 获取视频信息
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频信息: {width}x{height}, {fps}FPS, 总帧数: {total_frames}")

    # 显示详细的视频参数
    print(f"详细参数:")
    print(f"  宽度: {width} 像素")
    print(f"  高度: {height} 像素")
    print(f"  帧率: {fps} FPS")
    print(f"  总帧数: {total_frames}")
    print(f"  时长: {total_frames/fps:.2f} 秒")
    print(f"  NMS阈值: {args.nms}")
    print(f"  ReID模型: {args.reid}")
    print(f"  计算设备: {args.device}")
    print(f"  半精度推理: {'是' if args.fp16 else '否'}")

    # 初始化视频写入器（如果需要保存）
    video_writer = None
    if args.save:
        import os
        os.makedirs(args.output, exist_ok=True)
        
        # 生成输出文件名
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        output_path = os.path.join(args.output, f"{video_name}_tracked.avi")
        
        # 尝试多种编码格式，确保兼容性
        encodings = [
            ('XVID', cv2.VideoWriter_fourcc(*'XVID')),
            ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
            ('MP4V', cv2.VideoWriter_fourcc(*'MP4V')),
            ('H264', cv2.VideoWriter_fourcc(*'H264'))
        ]
        
        video_writer = None
        for encoding_name, fourcc in encodings:
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if video_writer.isOpened():
                print(f"视频将保存到: {output_path} ({encoding_name}编码)")
                print(f"输出参数: {width}x{height}, {fps}FPS")
                break
            else:
                video_writer.release()
                video_writer = None
        
        if video_writer is None:
            print(f"错误: 无法创建视频文件 {output_path}")
            print("尝试的所有编码格式都失败")

    # 使用YOLO原生绘制函数的绘制方法（只显示检测框和跟踪ID）
    def draw_tracks_on_frame(frame, detection_result, tracks, frame_num):
        """
        使用YOLO原生绘制函数绘制检测框（不显示类别），然后叠加跟踪ID
        
        Args:
            frame: 原始视频帧
            detection_result: YOLO检测结果（ultralytics Results对象）
            tracks: 跟踪结果字典
            frame_num: 当前帧号
        """
        # 使用YOLO的plot()方法，但设置labels=False和conf=False来隐藏类别和置信度
        annotated_frame = detection_result.plot(labels=False, conf=False)
        
        # 在YOLO绘制的检测框上叠加跟踪ID信息
        if frame_num < len(tracks["players"]):
            for track_id, player in tracks["players"][frame_num].items():
                bbox = player.get("bbox", [])
                if len(bbox) == 4 and not np.isnan(bbox).any():
                    x1, y1, x2, y2 = map(int, bbox)
                    # 在检测框顶部绘制跟踪ID
                    label = f"{track_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    # 绘制半透明背景（绿色，表示球员）
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0] + 10, y1), (0, 255, 0), -1)
                    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                    # 绘制文字
                    cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 绘制裁判的跟踪ID
        if frame_num < len(tracks["referees"]):
            for track_id, referee in tracks["referees"][frame_num].items():
                bbox = referee.get("bbox", [])
                if len(bbox) == 4 and not np.isnan(bbox).any():
                    x1, y1, x2, y2 = map(int, bbox)
                    label = f"{track_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0] + 10, y1), (0, 255, 255), -1)
                    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                    cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # 球的绘制（只显示ID，如果有的话）
        if frame_num < len(tracks["ball"]):
            for ball_id, ball in tracks["ball"][frame_num].items():
                bbox = ball.get("bbox", [])
                if len(bbox) == 4 and not np.isnan(bbox).any():
                    x1, y1, x2, y2 = map(int, bbox)
                    label = f"{ball_id}"
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                    overlay = annotated_frame.copy()
                    cv2.rectangle(overlay, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0] + 10, y1), (255, 255, 255), -1)
                    cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
                    cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        return annotated_frame

    frame_count = 0
    start_time = time.time()
    paused = False
    annotated_frame = None  # 初始化，避免首次显示错误

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("视频播放完毕")
                    break
                
                frame_count += 1
                
                # 获取YOLO检测结果（用于原生绘制）
                detection_result = tracker.model.predict(frame, conf=0.3, verbose=False)[0]
                
                # 使用get_object_tracks方法进行检测和追踪
                tracks = tracker.get_object_tracks([frame])
                
                # 调试信息：打印检测结果
                if frame_count % 30 == 0:  # 每30帧打印一次
                    players_count = len(tracks['players'][0]) if tracks['players'] else 0
                    referees_count = len(tracks['referees'][0]) if tracks['referees'] else 0
                    ball_count = len(tracks['ball'][0]) if tracks['ball'] else 0
                    print(f"帧 {frame_count}: 检测到 {players_count} 个球员, {referees_count} 个裁判, {ball_count} 个球")

                # 使用YOLO原生绘制函数绘制检测框，然后叠加跟踪ID
                annotated_frame = draw_tracks_on_frame(frame, detection_result, tracks, 0)
                
                # 添加信息文本
                current_time = time.time()
                elapsed_time = current_time - start_time
                current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                
                # 绘制状态信息
                cv2.putText(annotated_frame, f"Frame: {frame_count}/{total_frames}", 
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"FPS: {current_fps:.1f}", 
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_frame, f"Time: {elapsed_time:.1f}s", 
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if paused:
                    cv2.putText(annotated_frame, "PAUSED", 
                                (width//2 - 50, height//2), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                
                # 保存帧到视频（如果启用）
                if args.save and video_writer is not None:
                    # 确保帧尺寸与输入视频一致
                    if annotated_frame.shape[:2] != (height, width):
                        annotated_frame = cv2.resize(annotated_frame, (width, height))
                    video_writer.write(annotated_frame)
            
            # 显示帧（如果已处理过至少一帧）
            if annotated_frame is not None:
                cv2.imshow('Real-time Tracking Test', annotated_frame)
            
            # 处理按键
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("用户退出")
                break
            elif key == ord('p'):
                paused = not paused
                print(f"{'暂停' if paused else '继续'}播放")
            elif key == ord('r'):
                # 重置到开始
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_count = 0
                start_time = time.time()
                paused = False
                print("重置到开始")

    except KeyboardInterrupt:
        print("\n用户中断")

    finally:
        # 清理资源
        cap.release()
        if video_writer is not None:
            video_writer.release()
            print(f"✓ 视频已保存到: {output_path}")
            
            # 验证输出视频参数
            try:
                test_cap = cv2.VideoCapture(output_path)
                if test_cap.isOpened():
                    out_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    out_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    out_fps = int(test_cap.get(cv2.CAP_PROP_FPS))
                    out_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    print(f"输出视频验证:")
                    print(f"  尺寸: {out_width}x{out_height}")
                    print(f"  帧率: {out_fps} FPS")
                    print(f"  帧数: {out_frames}")
                    
                    # 检查参数是否匹配
                    if out_width == width and out_height == height and out_fps == fps:
                        print(f"✓ 输出视频参数与输入视频完全匹配")
                    else:
                        print(f"⚠ 警告: 输出视频参数与输入视频不匹配")
                        print(f"  输入: {width}x{height}, {fps}FPS")
                        print(f"  输出: {out_width}x{out_height}, {out_fps}FPS")
                    
                    test_cap.release()
                else:
                    print(f"⚠ 无法验证输出视频")
            except Exception as e:
                print(f"⚠ 验证输出视频时出错: {e}")
        
        cv2.destroyAllWindows()
        
        # 显示统计信息
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"\n统计信息:")
        print(f"总帧数: {frame_count}")
        print(f"总时间: {total_time:.2f}s")
        print(f"平均FPS: {avg_fps:.2f}")
        if args.save:
            print(f"输出视频: {output_path}")
        print("=" * 80)

