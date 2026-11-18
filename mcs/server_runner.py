import argparse
import json
import os
import sys
import time
from pathlib import Path

# 添加项目路径
current_dir = Path(__file__).parent.absolute()
sys.path.append(str(current_dir))

import config
from camera import Camera
from soccer import HalfPitchConfiguration
from tools.single_camera_tracker import tracker_botsort, BoTSORTArgs

def main():
    parser = argparse.ArgumentParser(description="服务器端计算任务运行器")
    parser.add_argument("--config", default="configs/cam1_config.json", type=str, help="标定JSON文件路径")
    parser.add_argument("--video-override", type=str, help="强制覆盖JSON中的视频路径（可选）")
    
    args = parser.parse_args()
    
    start_time = time.time()
    print(f"\n[{time.strftime('%H:%M:%S')}] === 开始执行服务器计算任务 ===")
    
    # 1. 读取配置文件
    if not os.path.exists(args.config):
        print(f"错误: 找不到配置文件 {args.config}")
        sys.exit(1)

    print(f"读取配置: {args.config}")
    with open(args.config, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    camera_id = data.get('camera_id', 1)
    video_path = data.get('video_path')
    homography_matrix = data.get('homography_matrix')
    train_area_points = data.get('train_area') # 读取训练区域
    
    # 允许命令行覆盖视频路径
    if args.video_override:
        video_path = args.video_override
        
    print(f"相机 ID: {camera_id}")
    print(f"视频路径: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在! 请检查路径是否正确: {video_path}")
        sys.exit(1)

    try:
        # 2. 初始化配置
        pitch_config = HalfPitchConfiguration()
        
        # 3. 初始化相机 (跳过标定，直接使用矩阵)
        print("正在初始化相机和模型...")
        camera = Camera(
            video_path=video_path,
            pitch_config=pitch_config,
            camera_id=camera_id,
            homography_matrix=homography_matrix
        )
        
        # 应用训练区域
        if train_area_points:
            print(f"应用训练区域: {len(train_area_points)} points")
            camera.train_area.region_points = train_area_points
            camera.apply_train_area()
        
        # 4. 初始化检测
        # 使用 config.py 中的默认路径
        camera.init_detection(model_type='yolox') 
        
        # 5. 初始化追踪器
        tracker_args = BoTSORTArgs()
        tracker = tracker_botsort(camera, args=tracker_args)
        
        # 6. 设置输出路径
        video_name = Path(video_path).stem
        output_dir = config.OUTPUT_DIR
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = str(output_dir / f"{video_name}_tracked.mp4")
        
        print(f"输出路径: {output_file}")
        print("开始追踪计算 (这可能需要一些时间)...")
        
        # 7. 运行追踪
        tracker.save_tracking_video(output_path=output_file, refine=True)
        
        elapsed = time.time() - start_time
        print(f"\n[{time.strftime('%H:%M:%S')}] === 任务完成! 耗时: {elapsed/60:.1f} 分钟 ===")
        print(f"结果已保存至: {output_file}")
        
    except Exception as e:
        print(f"\n发生严重错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
