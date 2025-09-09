import numpy as np  # 导入 numpy，处理数组与数值运算
import cv2  # 导入 OpenCV，用于图像处理与透视变换


class ViewTransformer():  # 定义视角（透视）变换器类
    def __init__(self):  # 构造函数：初始化参数与透视变换矩阵
        court_width = 68  # 场地宽度（假设单位米）
        court_length = 23.32  # 场地长度（假设单位米）

        self.pixel_vertices = np.array([[110, 1035],  # 原图（像素）四个顶点坐标
                                        [265, 275],
                                        [910, 260],
                                        [1640, 915]])

        self.target_vertices = np.array([  # 目标场地坐标系对应的四个顶点
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)  # 转为 float32，OpenCV 要求
        self.target_vertices = self.target_vertices.astype(np.float32)  # 转为 float32，OpenCV 要求

        self.persepctive_trasnformer = cv2.getPerspectiveTransform(self.pixel_vertices,
                                                                   self.target_vertices)  # 计算透视变换矩阵

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))  # 将点转换为整数像素坐标元组
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, p, False) >= 0  # 判断点是否在多边形区域内
        if not is_inside:
            return None  # 如果点不在区域内，返回 None

        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)  # 调整点的形状以适配透视变换函数
        tranform_point = cv2.perspectiveTransform(reshaped_point, self.persepctive_trasnformer)  # 执行透视变换
        return tranform_point.reshape(-1, 2)  # 返回二维坐标

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():  # 遍历每个对象及其轨迹
            for frame_num, track in enumerate(object_tracks):  # 遍历每一帧的轨迹
                for track_id, track_info in track.items():  # 遍历轨迹中的每个跟踪 ID
                    position = track_info['position_adjusted']  # 获取调整后的像素位置
                    position = np.array(position)  # 转为 numpy 数组
                    position_trasnformed = self.transform_point(position)  # 调用透视变换
                    if position_trasnformed is not None:
                        position_trasnformed = position_trasnformed.squeeze().tolist()  # 压缩维度并转为列表
                    tracks[object][frame_num][track_id]['position_transformed'] = position_trasnformed  # 存储结果
