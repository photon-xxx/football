"""
ViewTransformer 使用示例
展示如何使用修改后的 ViewTransformer 类进行透视变换
"""

import numpy as np
import cv2
from view_transformer import ViewTransformer
from Storage.field_configs.soccer import SoccerPitchConfiguration

def example_usage():
    """
    使用示例：展示如何使用 ViewTransformer 进行透视变换
    """
    # 模拟检测到的关键点（像素坐标）
    # 在实际使用中，这些点来自 KeyPointDetector.detect_keypoints()
    detected_keypoints = np.array([
        [100, 200],   # 关键点1
        [300, 150],   # 关键点2
        [500, 180],   # 关键点3
        [700, 250],   # 关键点4
        [900, 300],   # 关键点5
        [1100, 350],  # 关键点6
        [1300, 400],  # 关键点7
        [1500, 450],  # 关键点8
    ], dtype=np.float32)
    
    # 获取球场配置
    config = SoccerPitchConfiguration()
    
    # 对应的球场关键点坐标（场地坐标系）
    target_keypoints = np.array([
        [0, 0],                    # 对应关键点1
        [1000, 0],                 # 对应关键点2
        [2000, 0],                 # 对应关键点3
        [3000, 0],                 # 对应关键点4
        [4000, 0],                 # 对应关键点5
        [5000, 0],                 # 对应关键点6
        [6000, 0],                 # 对应关键点7
        [7000, 0],                 # 对应关键点8
    ], dtype=np.float32)
    
    # 创建 ViewTransformer 实例
    transformer = ViewTransformer(
        source=detected_keypoints,
        target=target_keypoints
    )
    
    # 示例：变换一些检测点
    test_points = np.array([
        [200, 300],
        [400, 250],
        [600, 280],
        [800, 320]
    ], dtype=np.float32)
    
    # 批量变换点
    transformed_points = transformer.transform_points(test_points)
    print("原始点坐标:")
    print(test_points)
    print("\n变换后坐标:")
    print(transformed_points)
    
    # 单个点变换
    single_point = np.array([500, 300], dtype=np.float32)
    transformed_single = transformer.transform_point(single_point)
    print(f"\n单个点变换:")
    print(f"原始: {single_point}")
    print(f"变换后: {transformed_single}")

def example_with_keypoint_detection():
    """
    与 KeyPointDetector 结合使用的示例
    """
    from KeyPoint_detection import KeyPointDetector
    
    # 创建关键点检测器
    detector = KeyPointDetector(device="cpu")
    
    # 模拟视频帧
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    
    # 检测关键点
    keypoints = detector.detect_keypoints(frame)
    
    # 筛选出有效的关键点（非NaN）
    if keypoints.xy is not None and len(keypoints.xy[0]) > 0:
        # 创建mask筛选非NaN点
        mask = ~np.isnan(keypoints.xy[0][:, 0]) & ~np.isnan(keypoints.xy[0][:, 1])
        
        if np.sum(mask) >= 4:  # 至少需要4个点来计算透视变换
            # 获取有效关键点
            valid_source_points = keypoints.xy[0][mask].astype(np.float32)
            
            # 获取对应的球场配置点
            config = SoccerPitchConfiguration()
            valid_target_points = np.array(config.vertices)[mask].astype(np.float32)
            
            # 创建透视变换器
            transformer = ViewTransformer(
                source=valid_source_points,
                target=valid_target_points
            )
            
            print(f"成功创建透视变换器，使用了 {len(valid_source_points)} 个关键点")
            
            # 示例：变换检测到的目标位置
            detection_points = np.array([
                [640, 360],  # 图像中心
                [320, 180],  # 左上区域
                [960, 540],  # 右下区域
            ], dtype=np.float32)
            
            transformed_detections = transformer.transform_points(detection_points)
            print("检测点变换结果:")
            for i, (orig, trans) in enumerate(zip(detection_points, transformed_detections)):
                print(f"点 {i+1}: {orig} -> {trans}")
        else:
            print("有效关键点数量不足，无法创建透视变换器")
    else:
        print("未检测到关键点")

if __name__ == "__main__":
    print("=== ViewTransformer 使用示例 ===")
    example_usage()
    
    print("\n=== 与关键点检测结合使用 ===")
    try:
        example_with_keypoint_detection()
    except ImportError as e:
        print(f"导入错误: {e}")
        print("请确保 KeyPoint_detection.py 文件存在且路径正确")


