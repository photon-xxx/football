import numpy as np  # 导入 numpy，处理数组与数值运算
import cv2  # 导入 OpenCV，用于图像处理与透视变换
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from Storage.field_configs.soccer import SoccerPitchConfiguration

# 处理相对导入问题
try:
    from .KeyPoint_detection import KeyPointDetector
except ImportError:
    # 如果相对导入失败，使用绝对导入
    from Module.view_transformer.KeyPoint_detection import KeyPointDetector

#TODO: 注意soccer设置变换为初始的cm度量，转换需要注意

class ViewTransformer():  # 定义视角（透视）变换器类
    def __init__(self, video_frames, device="cpu"):  # 构造函数：初始化参数与透视变换矩阵
        # 从配置文件导入场地设置
        self.config = SoccerPitchConfiguration()
        self.device = device
        
        # 初始化关键点检测器
        self.keypoint_detector = KeyPointDetector(device=device)
        
        # 存储所有帧的变换矩阵
        self.perspective_transformers = []
        
        # 调用关键点检测，为每一帧计算变换矩阵
        for frame, keypoints, indices in self.keypoint_detector.detect_from_video(video_frames, stride=1):
            # 为每一帧计算透视变换矩阵
            matrix = self._calculate_homography(keypoints, indices)
            self.perspective_transformers.append(matrix)

    def _calculate_homography(self, keypoints, indices):
        """
        根据检测到的关键点计算透视变换矩阵
        
        Args:
            keypoints: 检测到的关键点
            indices: 关键点索引
            
        Returns:
            np.ndarray: 透视变换矩阵，如果计算失败返回None
        """
        if keypoints is None or indices is None:
            print("关键点检测失败")
            return None
            
        # 获取检测到的关键点坐标
        detected_points = keypoints.xy[0]  # shape: (N, 2)
        
        # 直接使用传入的indices，它们已经包含了通过置信度过滤的高质量关键点索引
        if len(indices) < 4:
            print(f"有效关键点数量不足: {len(indices)} < 4")
            return None
            
        # 获取标准场地关键点
        standard_vertices = self.config.vertices  # 32个标准关键点
        
        # 根据indices选择对应的检测关键点和标准关键点
        source_points = []  # 检测到的关键点（像素坐标）
        target_points = []  # 对应的标准场地关键点（场地坐标）
        
        for idx in indices:
            if 0 <= idx < len(detected_points) and 0 <= idx < len(standard_vertices):
                x, y = detected_points[idx]
                if not np.isnan(x) and not np.isnan(y):
                    source_points.append([x, y])
                    target_points.append(standard_vertices[idx])
        
        if len(source_points) < 4:
            print(f"匹配的关键点数量不足: {len(source_points)} < 4")
            return None
            
        # 转换为numpy数组
        source_points = np.array(source_points, dtype=np.float32)
        target_points = np.array(target_points, dtype=np.float32)
        
        # 使用cv2.findHomography计算透视变换矩阵
        perspective_transformer, mask = cv2.findHomography(source_points, target_points)
        
        if perspective_transformer is not None:
           # print(f"成功计算透视变换矩阵，使用了 {len(source_points)} 个关键点")
            return perspective_transformer
        else:
            print("透视变换矩阵计算失败")
            return None

        
    def transform_point(self, point, frame_idx=0):
        """
        透视变换单个点
        
        Args:
            point: 输入点坐标 [x, y]
            frame_idx: 帧索引，用于选择对应的变换矩阵
            
        Returns:
            变换后的点坐标，如果变换失败返回None
        """
        if frame_idx >= len(self.perspective_transformers):
            return None
            
        matrix = self.perspective_transformers[frame_idx]
        
        # 如果当前帧矩阵为None，寻找最邻近的有效帧矩阵
        if matrix is None:
            print(f"警告: 帧 {frame_idx} 的变换矩阵为None，正在寻找最邻近的有效矩阵...")
            
            # 向前搜索
            for i in range(frame_idx - 1, -1, -1):
                if self.perspective_transformers[i] is not None:
                    matrix = self.perspective_transformers[i]
                    print(f"使用帧 {i} 的变换矩阵替代帧 {frame_idx}")
                    break
            
            # 如果向前搜索失败，向后搜索
            if matrix is None:
                for i in range(frame_idx + 1, len(self.perspective_transformers)):
                    if self.perspective_transformers[i] is not None:
                        matrix = self.perspective_transformers[i]
                        print(f"使用帧 {i} 的变换矩阵替代帧 {frame_idx}")
                        break
            
            # 如果仍然找不到有效矩阵
            if matrix is None:
                print(f"错误: 无法找到帧 {frame_idx} 附近的有效变换矩阵")
                return None
            
        point = np.array(point, dtype=np.float32)
        
        # 执行透视变换
        reshaped_point = point.reshape(-1, 1, 2).astype(np.float32)
        transformed_point = cv2.perspectiveTransform(reshaped_point, matrix)
        return transformed_point.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for object, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):

                # print("#########frame_num_test#########")
                # print(frame_num if frame_num % 96 == 0 else "")
                # print("#########frame_num_test#########")
                # 测试

                for track_id, track_info in track.items():
                    position = track_info['position_adjusted']
                    position = np.array(position)
                    
                    # 使用对应帧的变换矩阵
                    position_transformed = self.transform_point(position, frame_num)
                    
                    if position_transformed is not None:
                        position_transformed = position_transformed.squeeze().tolist()
                    tracks[object][frame_num][track_id]['position_transformed'] = position_transformed


# -------------------- 测试入口 --------------------
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ViewTransformer 测试")
    parser.add_argument("--video", type=str, default="E:\Soccer\\football\\football_analysis-main\IO\input_videos\\test.mp4", help="输入视频路径")
    parser.add_argument("--device", type=str, default="cpu", help="运行设备: cpu/cuda")
    args = parser.parse_args()
    
    print("=" * 60)
    print("ViewTransformer 测试")
    print("=" * 60)
    print(f"视频路径: {args.video}")
    print(f"运行设备: {args.device}")
    print()
    
    try:
        # 读取视频帧
        print("正在读取视频帧...")
        import cv2
        cap = cv2.VideoCapture(args.video)
        video_frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            video_frames.append(frame)
        
        cap.release()
        print(f"成功读取 {len(video_frames)} 帧")
        
        # 初始化 ViewTransformer
        print("正在初始化 ViewTransformer...")
        view_transformer = ViewTransformer(video_frames, device=args.device)
        
        # 检查透视变换矩阵
        if len(view_transformer.perspective_transformers) > 0:
            print("✓ ViewTransformer 初始化成功")
            print(f"总共计算了 {len(view_transformer.perspective_transformers)} 个变换矩阵")
            
            # 显示第一个矩阵作为示例
            first_matrix = view_transformer.perspective_transformers[0]
            if first_matrix is not None:
                print(f"第一帧变换矩阵形状: {first_matrix.shape}")
                print(f"第一帧变换矩阵:\n{first_matrix}")
            print()
        else:
            print("✗ ViewTransformer 初始化失败，未计算到任何变换矩阵")
            exit(1)
        
        # 每24帧打印一次变换矩阵信息
        print("开始显示变换矩阵信息，每24帧打印一次...")
        print("-" * 60)
        
        for frame_idx in range(0, len(view_transformer.perspective_transformers), 24):
            matrix = view_transformer.perspective_transformers[frame_idx]
            
            if frame_idx % 24 == 0:
                print(f"帧 {frame_idx + 1}:")
                
                if matrix is not None:
                    print(f"  变换矩阵形状: {matrix.shape}")
                    print(f"  变换矩阵:\n{matrix}")
                else:
                    print("  该帧变换矩阵计算失败")
                
                print()
        
        print(f"测试完成，总共处理了 {len(view_transformer.perspective_transformers)} 帧")
        
        # 测试 transform_point 方法
        print("\n" + "=" * 60)
        print("测试 transform_point 方法")
        print("=" * 60)
        
        # 测试几个示例点
        test_points = [
            [500, 100],   # 左上角区域
            [1000, 600],   # 中心区域
            [1600, 1000],   # 右下角区域
            [800, 300],   # 中间区域
        ]
        
        print("测试点变换: ->x   ↓ y")
        for i, point in enumerate(test_points):
            # 使用第一帧的变换矩阵进行测试
            if len(view_transformer.perspective_transformers) > 0:
                # 临时设置透视变换矩阵
                view_transformer.perspective_transformer = view_transformer.perspective_transformers[0]
                
                if view_transformer.perspective_transformer is not None:
                    transformed = view_transformer.transform_point(point)
                    if transformed is not None:
                        print(f"  测试点 {i+1}: {point} -> {transformed}")
                    else:
                        print(f"  测试点 {i+1}: {point} -> 变换失败")
                else:
                    print(f"  测试点 {i+1}: {point} -> 矩阵为None")
            else:
                print(f"  测试点 {i+1}: {point} -> 无可用变换矩阵")
        
        print("\ntransform_point 测试完成")
        
        # 获取第一帧并标注测试点
        print("\n" + "=" * 60)
        print("标注测试点到第一帧并保存")
        print("=" * 60)
        
        # 创建test目录
        import os
        test_dir = "E:\Soccer\\football\\football_analysis-main\IO\output_videos"
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)
        
        # 获取第一帧
        cap = cv2.VideoCapture(args.video)
        ret, first_frame = cap.read()
        cap.release()
        
        if ret:
            # 在帧上标注测试点
            annotated_frame = first_frame.copy()
            
            # 定义颜色
            colors = [
                (0, 255, 0),    # 绿色
                (255, 0, 0),    # 蓝色
                (0, 0, 255),    # 红色
                (255, 255, 0),  # 青色
            ]
            
            for i, point in enumerate(test_points):
                x, y = int(point[0]), int(point[1])
                color = colors[i % len(colors)]
                
                # 绘制圆点
                cv2.circle(annotated_frame, (x, y), 10, color, -1)
                
                # 绘制标签
                label = f"P{i+1}"
                cv2.putText(annotated_frame, label, (x + 15, y - 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # 显示坐标信息
                coord_text = f"({x},{y})"
                cv2.putText(annotated_frame, coord_text, (x + 15, y + 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # 保存标注后的帧
            output_path = os.path.join(test_dir, "test_points_annotated.jpg")
            cv2.imwrite(output_path, annotated_frame)
            print(f"✓ 测试点标注完成，保存到: {output_path}")
            
            # 显示测试点信息
            print("标注的测试点:")
            for i, point in enumerate(test_points):
                print(f"  P{i+1}: {point} (颜色: {colors[i % len(colors)]})")
        else:
            print("✗ 无法读取第一帧")
        
        print("\n测试点标注完成")
        
    except KeyboardInterrupt:
        print("\n用户中断测试")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
