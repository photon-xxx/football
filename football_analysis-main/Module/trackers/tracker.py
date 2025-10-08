from ultralytics import YOLO  # 导入 YOLO 检测模型
import supervision as sv  # 导入 supervision 库，用于跟踪和可视化
import pickle  # 用于保存/读取缓存数据
import os  # 文件路径操作
import numpy as np  # 数组运算
import pandas as pd  # 表格数据处理（插值）
import cv2  # OpenCV 绘图与图像处理
import sys

sys.path.append('../../')  # 将上级目录加入搜索路径，方便导入 utils
from Module.utils import get_center_of_bbox, get_bbox_width, get_foot_position  # 导入工具函数


# TODO: 追踪器
class Tracker:  # 定义跟踪器类
    def __init__(self, model_path, nms_threshold=0.5):
        self.model = YOLO(model_path)  # 加载 YOLO 模型
        self.tracker = sv.ByteTrack()  # 初始化 ByteTrack 跟踪器
        self.tracker.reset()
        self.nms_threshold = nms_threshold  # NMS阈值，用于减少重复检测

    def add_position_to_tracks(self, tracks):  # 给轨迹增加位置点信息
        for object, object_tracks in tracks.items():  # 遍历 players/referees/ball
            for frame_num, track in enumerate(object_tracks):  # 遍历每一帧
                for track_id, track_info in track.items():  # 遍历该帧中的每个对象
                    bbox = track_info['bbox']  # 获取目标的边界框
                    if object == 'ball':
                        position = get_center_of_bbox(bbox)  # 球用 bbox 中心
                    else:
                        position = get_foot_position(bbox)  # 球员/裁判用脚下点
                    tracks[object][frame_num][track_id]['position'] = position  # 写入位置

    def interpolate_ball_positions(self, ball_positions):
        ball_positions = [x.get(1, {}).get('bbox', []) for x in ball_positions]  # 提取每帧球的 bbox
        df_ball_positions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])  # 转为 DataFrame

        df_ball_positions = df_ball_positions.interpolate()  # 插值补齐缺失值
        df_ball_positions = df_ball_positions.bfill()  # 向后填充剩余空值

        ball_positions = [{1: {"bbox": x}} for x in df_ball_positions.to_numpy().tolist()]  # 转回字典列表格式
        return ball_positions

    def detect_frames(self, frames, batch_size = 20):
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i + batch_size]
            results = self.model.predict(batch_frames, conf=0.3)
            detections += results  # 直接返回YOLO结果
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):  # 若存在缓存文件
            with open(stub_path, 'rb') as f:
                tracks = pickle.load(f)  # 直接读取缓存
            return tracks

        detections = self.detect_frames(frames)  # 运行检测

        tracks = {  # 初始化存储结构
            "players": [],
            "referees": [],
            "ball": []
        }

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names  # 类别名字典 id->name
            cls_names_inv = {v: k for k, v in cls_names.items()}  # 反向映射 name->id

            detection_supervision = sv.Detections.from_ultralytics(detection)  # 转为 supervision 格式

            # 处理守门员类别
            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]  # 把守门员归为 player

            # 按照Roboflow逻辑分离球和其他对象
            BALL_ID = cls_names_inv['ball']
            ball_detections = detection_supervision[detection_supervision.class_id == BALL_ID]
            all_detections = detection_supervision[detection_supervision.class_id != BALL_ID]

            # 对球进行边界框扩展（Roboflow优化）
            if len(ball_detections) > 0:
                ball_detections.xyxy = sv.pad_boxes(xyxy=ball_detections.xyxy, px=10)

            # 只对非球对象应用NMS和追踪
            all_detections = all_detections.with_nms(threshold=self.nms_threshold, class_agnostic=True)
            all_detections.class_id -= 1  # Roboflow的类别ID调整
            detection_with_tracks = self.tracker.update_with_detections(detections=all_detections)

            tracks["players"].append({})  # 初始化该帧字典
            tracks["referees"].append({})
            tracks["ball"].append({})

            # 存储追踪结果（球员和裁判）
            for frame_detection in detection_with_tracks:
                bbox = frame_detection[0].tolist()  # 获取 bbox
                cls_id = frame_detection[3]  # 获取类别 ID
                track_id = frame_detection[4]  # 获取跟踪 ID

                # 根据调整后的类别ID判断对象类型
                if cls_id == cls_names_inv['player'] - 1:  # 调整后的球员ID
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}

                if cls_id == cls_names_inv['referee'] - 1:  # 调整后的裁判ID
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # 存储球（不追踪，使用扩展后的边界框）
            for frame_detection in ball_detections:
                bbox = frame_detection[0].tolist()
                tracks["ball"][frame_num][1] = {"bbox": bbox}  # 存球，默认 ID=1

        if stub_path is not None:  # 可选缓存保存
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

# ---------------------------------------------------------------------------------------------------------
# 绘制
    def draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])  # bbox 底部 y
        x_center, _ = get_center_of_bbox(bbox)  # 中心 x
        width = get_bbox_width(bbox)  # bbox 宽度

        cv2.ellipse(  # 在脚下画椭圆
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40  # 标注 ID 的矩形宽
        rectangle_height = 20  # 高
        x1_rect = x_center - rectangle_width // 2
        x2_rect = x_center + rectangle_width // 2
        y1_rect = (y2 - rectangle_height // 2) + 15
        y2_rect = (y2 + rectangle_height // 2) + 15

        if track_id is not None:  # 如果有 ID，就画矩形框 + 文本
            cv2.rectangle(frame,
                          (int(x1_rect), int(y1_rect)),
                          (int(x2_rect), int(y2_rect)),
                          color,
                          cv2.FILLED)

            x1_text = x1_rect + 12
            if track_id > 99:  # ID 为三位数时左移一点
                x1_text -= 10

            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text), int(y1_rect + 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        return frame

    def draw_traingle(self, frame, bbox, color):
        y = int(bbox[1])  # bbox 顶部 y
        x, _ = get_center_of_bbox(bbox)  # bbox 中心 x

        triangle_points = np.array([  # 定义三角形
            [x, y],
            [x - 10, y - 20],
            [x + 10, y - 20],
        ])
        cv2.drawContours(frame, [triangle_points], 0, color, cv2.FILLED)  # 填充三角形
        cv2.drawContours(frame, [triangle_points], 0, (0, 0, 0), 2)  # 画黑边

        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), -1)  # 画透明矩形
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # 合成透明效果

        team_ball_control_till_frame = team_ball_control[:frame_num + 1]  # 截取到当前帧的控球信息
        team_1_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 1].shape[0]  # 队 1 控球帧数
        team_2_num_frames = team_ball_control_till_frame[team_ball_control_till_frame == 2].shape[0]  # 队 2 控球帧数
        team_1 = team_1_num_frames / (team_1_num_frames + team_2_num_frames)  # 控球比例
        team_2 = team_2_num_frames / (team_1_num_frames + team_2_num_frames)

        cv2.putText(frame, f"Team 1 Ball Control: {team_1 * 100:.2f}%", (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)  # 写比例
        cv2.putText(frame, f"Team 2 Ball Control: {team_2 * 100:.2f}%", (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 3)

        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control=None):
        output_video_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict = tracks["players"][frame_num]  # 当前帧球员
            ball_dict = tracks["ball"][frame_num]  # 当前帧球
            referee_dict = tracks["referees"][frame_num]  # 当前帧裁判

            for track_id, player in player_dict.items():  # 绘制球员
                color = player.get("team_color", (0, 0, 255))
                frame = self.draw_ellipse(frame, player["bbox"], color, track_id)

                if player.get('has_ball', False):
                    frame = self.draw_traingle(frame, player["bbox"], (0, 0, 255))

            for _, referee in referee_dict.items():  # 绘制裁判
                frame = self.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

            for track_id, ball in ball_dict.items():  # 绘制球
                frame = self.draw_traingle(frame, ball["bbox"], (0, 255, 0))

            # 只有当team_ball_control不为None时才绘制控球信息
            if team_ball_control is not None:
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

            output_video_frames.append(frame)

        return output_video_frames
       
# 主函数测试
if __name__ == "__main__":
    import argparse
    import time
    import cv2
    
    parser = argparse.ArgumentParser(description="实时追踪测试")
    parser.add_argument("--video", type=str, default="IO/input_videos/clip_30s.mp4", 
                       help="输入视频路径")
    parser.add_argument("--model", type=str, default="Storage/models/yolo/best.pt", 
                       help="YOLO模型路径")
    parser.add_argument("--fps", action="store_true", 
                       help="显示FPS信息")
    parser.add_argument("--save", action="store_true", 
                       help="保存追踪结果视频")
    parser.add_argument("--output", type=str, default="IO/output_videos/test/tracker", 
                       help="输出目录")
    parser.add_argument("--nms", type=float, default=0.5, 
                       help="NMS阈值，用于减少重复检测 (0.1-0.9)")
    
    args = parser.parse_args()
    
    # 创建追踪器实例
    tracker = Tracker(args.model, nms_threshold=args.nms)
    
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

    # 初始化追踪器
    tracker.model = YOLO(args.model)
    tracker.tracker = sv.ByteTrack()
    tracker.nms_threshold = args.nms  # 设置NMS阈值

    frame_count = 0
    start_time = time.time()
    paused = False

    try:
        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    print("视频播放完毕")
                    break
                
                frame_count += 1
                
                # 使用get_object_tracks方法进行检测和追踪
                tracks = tracker.get_object_tracks([frame])
                
                # 调试信息：打印检测结果
                if frame_count % 30 == 0:  # 每30帧打印一次
                    print(f"帧 {frame_count}: 检测到 {len(tracks['players'][0])} 个球员, {len(tracks['referees'][0])} 个裁判, {len(tracks['ball'][0])} 个球")
                    # 打印详细的tracks信息
                    print("=== Tracks详细信息 ===")
                    print(f"Players: {tracks['players']}")
                    print(f"Referees: {tracks['referees']}")
                    print(f"Ball: {tracks['ball']}")
                    print("========================")

                # 使用现有的draw_annotations方法绘制检测结果
                annotated_frames = tracker.draw_annotations([frame], tracks)
                annotated_frame = annotated_frames[0]
                
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
            
            # 显示帧
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

