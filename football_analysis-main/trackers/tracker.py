from ultralytics import YOLO  # 导入 YOLO 检测模型
import supervision as sv  # 导入 supervision 库，用于跟踪和可视化
import pickle  # 用于保存/读取缓存数据
import os  # 文件路径操作
import numpy as np  # 数组运算
import pandas as pd  # 表格数据处理（插值）
import cv2  # OpenCV 绘图与图像处理
import sys

sys.path.append('../')# 将上级目录加入搜索路径，方便导入 utils
from utils import get_center_of_bbox, get_bbox_width, get_foot_position  # 导入工具函数


class Tracker:  # 定义跟踪器类
    def __init__(self, model_path):
        self.model = YOLO(model_path)  # 加载 YOLO 模型
        self.tracker = sv.ByteTrack()  # 初始化 ByteTrack 跟踪器

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

    def detect_frames(self, frames):
        batch_size = 20  # 每次处理的帧数
        detections = []
        for i in range(0, len(frames), batch_size):
            detections_batch = self.model.predict(frames[i:i + batch_size], conf=0.1)  # 批量预测
            detections += detections_batch
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

            for object_ind, class_id in enumerate(detection_supervision.class_id):
                if cls_names[class_id] == "goalkeeper":
                    detection_supervision.class_id[object_ind] = cls_names_inv["player"]  # 把守门员归为 player

            detection_with_tracks = self.tracker.update_with_detections(detection_supervision)  # 运行 ByteTrack

            tracks["players"].append({})  # 初始化该帧字典
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_detection in detection_with_tracks:  # 遍历跟踪结果
                bbox = frame_detection[0].tolist()  # 获取 bbox
                cls_id = frame_detection[3]  # 获取类别 ID
                track_id = frame_detection[4]  # 获取跟踪 ID

                if cls_id == cls_names_inv['player']:
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}  # 存球员

                if cls_id == cls_names_inv['referee']:
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}  # 存裁判

            for frame_detection in detection_supervision:  # 遍历检测结果（不跟踪）
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]

                if cls_id == cls_names_inv['ball']:
                    tracks["ball"][frame_num][1] = {"bbox": bbox}  # 存球，默认 ID=1

        if stub_path is not None:  # 可选缓存保存
            with open(stub_path, 'wb') as f:
                pickle.dump(tracks, f)

        return tracks

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

    def draw_annotations(self, video_frames, tracks, team_ball_control):
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

            frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)  # 绘制控球信息

            output_video_frames.append(frame)

        return output_video_frames
