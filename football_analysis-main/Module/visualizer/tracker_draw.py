from ultralytics import YOLO  # 导入 YOLO 检测模型
import supervision as sv  # 导入 supervision 库，用于跟踪和可视化
import numpy as np  # 数组运算
import cv2  # 导入 OpenCV 库
import sys

sys.path.append('../../../')  # 将上级目录加入搜索路径，方便导入 utils
from Module.common.utils import get_center_of_bbox, get_bbox_width, get_foot_position  # 导入工具函数

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