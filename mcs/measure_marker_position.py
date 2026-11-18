"""
训练场地标志物测量工具

功能：
1. 标注视频中的障碍桶/标志物（支持多组）
2. 将像素坐标转换为球场物理坐标
3. 计算多种距离指标（组间、组内、最近距离等）
4. 可视化测量结果
5. 保存标定和测量数据

用法示例：
    python measure_marker_position.py
"""

import numpy as np
import cv2
import json
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import supervision as sv
import yaml

from area import run_marker_annotation, load_marker_keypoints, imwritex
from camera import Camera
import soccer
from tools.annotation.pitch_annotation_tool import draw_pitch
from tools.distances_tool import (
    calculate_pairwise_distances, 
    draw_pairwise_distances,
    draw_distance_between_groups,
    _to_pix
)
import os

class MarkerMeasurement:
    """场地标志物测量类"""
    
    def __init__(self, camera: Camera, output_dir: str = None):
        """
        Args:
            camera: Camera 对象
            output_dir: 输出目录，默认为 store/manual/{camera_name}
        """
        self.camera = camera
        self.config = camera.config
        
        # 设置输出目录
        if output_dir is None:
            self.output_dir = Path("store") / "manual" / camera.name
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 数据存储
        self.marker_groups_pixel = []  # 像素坐标 [[(x,y),...], ...]
        self.marker_groups_pitch = []  # 球场坐标 [[(x,y),...], ...]
        
        print(f"[MarkerMeasurement] 初始化完成")
        print(f"[MarkerMeasurement] 输出目录: {self.output_dir}")
    
    def annotate_markers(self, frame: np.ndarray = None) -> bool:
        """
        标注标志物
        
        Args:
            frame: 视频帧，如果为None则使用camera的第一帧
        
        Returns:
            成功返回True，失败或取消返回False
        """
        if frame is None:
            frame = self.camera.first_frame
        
        print("\n" + "="*60)
        print("开始标注标志物")
        print("说明：")
        print("  - 每组代表一类标志物（如一排障碍桶）")
        print("  - 按 'n' 切换到下一组")
        print("  - 按 's' 保存所有标注")
        print("="*60)
        
        # 调用标注工具
        output_json = self.output_dir / "markers.json"
        results = run_marker_annotation(frame, str(output_json), 
                                       frame_name=f"{self.camera.name}_markers")
        
        if results is None or not results.get("groups"):
            print("⚠ 标注取消或无有效数据")
            return False
        
        # 存储像素坐标
        self.marker_groups_pixel = results["groups"]
        
        print(f"✅ 标注完成：共 {len(self.marker_groups_pixel)} 组")
        for i, group in enumerate(self.marker_groups_pixel):
            print(f"   组 {i+1}: {len(group)} 个点")
        
        return True
    
    def load_markers(self, json_path: str = None) -> bool:
        """
        从文件加载已标注的标志物
        
        Args:
            json_path: JSON文件路径，默认为 output_dir/markers.json
        """
        if json_path is None:
            json_path = self.output_dir / "markers.json"
        
        try:
            results = load_marker_keypoints(str(json_path))
            self.marker_groups_pixel = results["groups"]
            print(f"✅ 从 {json_path} 加载了 {len(self.marker_groups_pixel)} 组标志物")
            return True
        except Exception as e:
            print(f"❌ 加载失败: {e}")
            return False
    
    def convert_to_pitch_coordinates(self):
        """将像素坐标转换为球场坐标"""
        if not self.marker_groups_pixel:
            print("⚠ 没有标志物数据")
            return
        
        self.marker_groups_pitch = []
        
        for group_idx, pixel_group in enumerate(self.marker_groups_pixel):
            if not pixel_group:
                self.marker_groups_pitch.append([])
                continue
            
            # 转换为numpy数组
            pixel_array = np.array(pixel_group, dtype=np.float32)
            
            # 使用camera的坐标转换
            pitch_array = self.camera.pixel_to_pitch(pixel_array)
            
            # 转换回列表
            pitch_group = [(float(p[0]), float(p[1])) for p in pitch_array]
            self.marker_groups_pitch.append(pitch_group)
            
            print(f"✅ 组 {group_idx+1}: 转换了 {len(pitch_group)} 个点")
    
    def calculate_group_distances(self, group_idx: int, is_closed: bool = False) -> Dict:
        """
        计算组内相邻点距离
        
        Args:
            group_idx: 组索引（从0开始）
            is_closed: 是否为环形（首尾相连）
        
        Returns:
            Dict: {
                'distances': [d1, d2, ...],  # 单位：米
                'average': avg,              # 单位：米
                'is_closed': bool
            }
        """
        if group_idx >= len(self.marker_groups_pitch):
            return None
        
        points = self.marker_groups_pitch[group_idx]
        distances, avg_dist = calculate_pairwise_distances(points, is_closed)
        
        return {
            'distances': distances,
            'average': avg_dist,
            'is_closed': is_closed
        }
    
    def calculate_center_to_center_distance(self, group1_idx: int, group2_idx: int) -> float:
        """
        计算两组的重心距离
        
        Returns:
            距离（米）
        """
        if group1_idx >= len(self.marker_groups_pitch) or group2_idx >= len(self.marker_groups_pitch):
            return None
        
        g1 = np.array(self.marker_groups_pitch[group1_idx])
        g2 = np.array(self.marker_groups_pitch[group2_idx])
        
        if len(g1) == 0 or len(g2) == 0:
            return None
        
        center1 = np.mean(g1, axis=0)
        center2 = np.mean(g2, axis=0)
        
        distance_cm = np.linalg.norm(center1 - center2)
        return round(distance_cm / 100.0, 2)  # 转换为米
    
    def calculate_nearest_points_distance(self, group1_idx: int, group2_idx: int) -> Tuple[float, Tuple, Tuple]:
        """
        计算两组之间最近的两个点的距离
        
        Returns:
            (距离(米), 组1中的点, 组2中的点)
        """
        if group1_idx >= len(self.marker_groups_pitch) or group2_idx >= len(self.marker_groups_pitch):
            return None, None, None
        
        g1 = np.array(self.marker_groups_pitch[group1_idx])
        g2 = np.array(self.marker_groups_pitch[group2_idx])
        
        if len(g1) == 0 or len(g2) == 0:
            return None, None, None
        
        # 计算所有点对之间的距离
        min_dist = float('inf')
        nearest_p1 = None
        nearest_p2 = None
        
        for p1 in g1:
            for p2 in g2:
                dist = np.linalg.norm(p1 - p2)
                if dist < min_dist:
                    min_dist = dist
                    nearest_p1 = tuple(p1)
                    nearest_p2 = tuple(p2)
        
        return round(min_dist / 100.0, 2), nearest_p1, nearest_p2
    
    def generate_measurement_report(self, 
                                   group_configs: List[Dict] = None,
                                   pair_configs: List[Dict] = None) -> str:
        """
        生成测量报告
        
        Args:
            group_configs: 组内测量配置，如 [{'group': 0, 'is_closed': False}, ...]
            pair_configs: 组间测量配置，如 [{'group1': 0, 'group2': 1, 'type': 'center'}, ...]
                         type可选: 'center'(重心距离), 'nearest'(最近点距离), 'both'(两者都算)
        
        Returns:
            报告文本
        """
        report = []
        report.append("=" * 70)
        report.append(f"训练场地标志物测量报告")
        report.append(f"相机: {self.camera.name}")
        report.append(f"组数: {len(self.marker_groups_pitch)}")
        report.append("=" * 70)
        
        # 组概览
        report.append("\n【标志物组概览】")
        for i, group in enumerate(self.marker_groups_pitch):
            report.append(f"  组 {i+1}: {len(group)} 个标志物")
        
        # 组内距离测量
        if group_configs:
            report.append("\n【组内距离测量】")
            for config in group_configs:
                group_idx = config['group']
                is_closed = config.get('is_closed', False)
                
                if group_idx >= len(self.marker_groups_pitch):
                    continue
                
                result = self.calculate_group_distances(group_idx, is_closed)
                if result:
                    mode = "环形" if is_closed else "线性"
                    report.append(f"\n  组 {group_idx+1} ({mode}):")
                    for i, d in enumerate(result['distances']):
                        if is_closed and i == len(result['distances']) - 1:
                            report.append(f"    点 {i+1} → 点 1: {d:.2f} m")
                        else:
                            report.append(f"    点 {i+1} → 点 {i+2}: {d:.2f} m")
                    report.append(f"    平均间距: {result['average']:.2f} m")
        
        # 组间距离测量
        if pair_configs:
            report.append("\n【组间距离测量】")
            for config in pair_configs:
                g1_idx = config['group1']
                g2_idx = config['group2']
                measure_type = config.get('type', 'center')
                
                report.append(f"\n  组 {g1_idx+1} ↔ 组 {g2_idx+1}:")
                
                if measure_type in ['center', 'both']:
                    dist = self.calculate_center_to_center_distance(g1_idx, g2_idx)
                    if dist is not None:
                        report.append(f"    重心距离: {dist:.2f} m")
                
                if measure_type in ['nearest', 'both']:
                    dist, p1, p2 = self.calculate_nearest_points_distance(g1_idx, g2_idx)
                    if dist is not None:
                        report.append(f"    最近点距离: {dist:.2f} m")
        
        report.append("\n" + "=" * 70)
        return "\n".join(report)
    
    def visualize_measurements(self, 
                              group_configs: List[Dict] = None,
                              pair_configs: List[Dict] = None,
                              save_path: str = None) -> np.ndarray:
        """
        可视化测量结果
        
        Args:
            group_configs: 组内测量配置
            pair_configs: 组间测量配置
            save_path: 保存路径，如果为None则不保存
        
        Returns:
            可视化图像
        """
        # 创建球场底图
        pitch_img = draw_pitch(self.config, padding=50, scale=0.1)
        
        # 定义颜色
        colors = [
            (0, 255, 0),    # 绿
            (0, 0, 255),    # 红
            (255, 0, 0),    # 蓝
            (0, 255, 255),  # 黄
            (255, 0, 255),  # 紫
            (255, 255, 0),  # 青
        ]
        
        # 绘制所有组的点
        for group_idx, pitch_group in enumerate(self.marker_groups_pitch):
            if not pitch_group:
                continue
            
            color = colors[group_idx % len(colors)]
            
            for pt_idx, pt in enumerate(pitch_group):
                pt_pix = _to_pix(pt, padding=50, scale=0.1)
                # 减小点的大小：内圆从8->5，外圆从10->7
                cv2.circle(pitch_img, pt_pix, 5, color, -1)
                cv2.circle(pitch_img, pt_pix, 7, (255, 255, 255), 2)
        
        # 绘制组内距离
        if group_configs:
            for config in group_configs:
                group_idx = config['group']
                is_closed = config.get('is_closed', False)
                
                if group_idx >= len(self.marker_groups_pitch):
                    continue
                
                result = self.calculate_group_distances(group_idx, is_closed)
                if result:
                    color = colors[group_idx % len(colors)]
                    pitch_img = draw_pairwise_distances(
                        pitch_img, 
                        self.marker_groups_pitch[group_idx],
                        result['distances'],
                        color=color,
                        is_closed=is_closed
                    )
        
        # 绘制组间距离
        if pair_configs:
            for config in pair_configs:
                g1_idx = config['group1']
                g2_idx = config['group2']
                measure_type = config.get('type', 'center')
                
                if measure_type in ['center', 'both']:
                    pitch_img = draw_distance_between_groups(
                        pitch_img,
                        self.marker_groups_pitch[g1_idx],
                        self.marker_groups_pitch[g2_idx],
                        color=(255, 0, 255),
                        padding=50,
                        scale=0.1
                    )
                
                if measure_type == 'nearest':
                    # 绘制最近点连线
                    dist, p1, p2 = self.calculate_nearest_points_distance(g1_idx, g2_idx)
                    if dist is not None and p1 is not None and p2 is not None:
                        p1_pix = _to_pix(p1, padding=50, scale=0.1)
                        p2_pix = _to_pix(p2, padding=50, scale=0.1)
                        cv2.line(pitch_img, p1_pix, p2_pix, (255, 128, 0), 2)
                        
                        # 标注距离
                        mid_pix = ((p1_pix[0] + p2_pix[0]) // 2, (p1_pix[1] + p2_pix[1]) // 2)
                        text = f"{dist:.2f}m"
                        cv2.putText(pitch_img, text, mid_pix,
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # 保存（使用支持中文路径的函数）
        if save_path:
            imwritex(str(save_path), pitch_img)
            print(f"✅ 可视化图已保存: {save_path}")
        
        return pitch_img
    
    def save_results(self, 
                    group_configs: List[Dict] = None,
                    pair_configs: List[Dict] = None):
        """
        保存所有结果（报告、可视化、原始数据）
        """
        # 1. 保存原始坐标数据
        data = {
            'camera': self.camera.name,
            'num_groups': len(self.marker_groups_pitch),
            'groups': []
        }
        
        for i, (pixel_group, pitch_group) in enumerate(zip(self.marker_groups_pixel, self.marker_groups_pitch)):
            group_data = {
                'group_id': i,
                'num_points': len(pitch_group),
                'pixel_coords': [list(p) for p in pixel_group],
                'pitch_coords': [list(p) for p in pitch_group]
            }
            data['groups'].append(group_data)
        
        coord_path = self.output_dir / "coordinates.json"
        with open(coord_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ 坐标数据已保存: {coord_path}")
        
        # 2. 生成并保存报告
        report = self.generate_measurement_report(group_configs, pair_configs)
        print("\n" + report)
        
        report_path = self.output_dir / "measurement_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"✅ 测量报告已保存: {report_path}")
        
        # 3. 保存可视化
        vis_path = self.output_dir / "visualization.jpg"
        self.visualize_measurements(group_configs, pair_configs, str(vis_path))


def load_configure():

    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    video_path = os.path.join(cfg['dir_path'], cfg['video_name'])
    return video_path, cfg['CAMERA_NAME'], cfg['config'], cfg['GROUP_CONFIGS'], cfg['PAIR_CONFIGS']

def main():
    """主函数示例"""
    
    # 配置测量参数
    VIDEO_PATH, CAMERA_NAME, config, GROUP_CONFIGS, PAIR_CONFIGS = load_configure()
    config = getattr(soccer, config)()  # 从 soccer 模块加载配置对象
    # ========== 执行流程 ==========
    
    print("="*70)
    print("训练场地标志物测量工具")
    print("="*70)
    
    # 1. 初始化相机
    print("\n[1/5] 初始化相机...")

    '''
    camera = Camera(
        video_path=VIDEO_PATH,
        pitch_config=config,
        camera_name=CAMERA_NAME,
        # mcs_name=MCS_NAME,
        enable_distortion_correction=True
    )
    '''

    # 1. 初始化相机（使用本地视频路径和配置）
    store_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "store", CAMERA_NAME)
    camera = Camera(
        video_path=VIDEO_PATH,
        pitch_config=config,
        camera_name=CAMERA_NAME,
        calibration_dir=store_path,
        enable_distortion_correction=False
    )

    # 2. 创建测量对象
    print("\n[2/5] 创建测量对象...")
    measurement = MarkerMeasurement(camera)
    
    # 3. 标注标志物
    print("\n[3/5] 标注标志物...")
    
    # 检查是否有缓存文件
    markers_cache = measurement.output_dir / "markers.json"
    if markers_cache.exists():
        print(f"✨ 发现缓存文件: {markers_cache}")
        use_cache = input("是否使用缓存？(y/n，默认y): ").strip().lower()
        
        if use_cache != 'n':
            # 选项1: 加载已有标注
            success = measurement.load_markers()
        else:
            # 选项2: 进行新标注（会覆盖缓存）
            success = measurement.annotate_markers()
    else:
        # 首次标注
        print("⚠ 未发现缓存，开始新标注...")
        success = measurement.annotate_markers()
    
    if not success:
        print("❌ 标注失败或取消，退出")
        return
    
    # 4. 转换坐标
    print("\n[4/5] 转换坐标到球场系统...")
    measurement.convert_to_pitch_coordinates()
    
    # 5. 计算和保存结果
    print("\n[5/5] 计算距离并生成报告...")
    measurement.save_results(GROUP_CONFIGS, PAIR_CONFIGS)
    
    print("\n✅ 测量完成！")
    print(f"📁 所有结果已保存至: {measurement.output_dir}")

if __name__ == "__main__":
    main()

