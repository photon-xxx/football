import json
import cv2
import numpy as np
import os
from typing import List, Tuple, Optional, Any, Dict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def imwritex(filename, img, params=None):
    """
    支持中文路径的图片保存函数
    """
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='wb') as f:
                n.tofile(f)
            return True
        return False
    except Exception as e:
        print(f"Error saving image {filename}: {e}")
        return False


# TODO：设计标注工具抽象基类
class KeypointAnnotator:
    """
    手动关键点标定器
    
    依赖外部传入的球场配置对象(config)，该对象需包含:
    - labels: List[str]
    - vertices: List[Tuple[int, int]]
    """
    
    def __init__(self, config: Any):
        """
        Args:
            config: 球场配置对象 (如 HalfPitchConfiguration)
        """
        self.config = config
        self.image = None
        self.display_image = None
        self.clicked_points = []      # [(x,y), ...]
        self.keypoint_indices = []    # [index, ...]
        self.window_name = "Keypoint Annotation Tool"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        if not hasattr(config, 'labels') or not hasattr(config, 'vertices'):
            raise ValueError("Config对象缺少 'labels' 或 'vertices' 属性")

    def put_chinese_text(self, img, text, position, font_size=24, color=(255, 255, 255)):
        """在图像上绘制中文 (使用PIL辅助)"""
        # OpenCV image (BGR) -> PIL image (RGB)
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # 尝试加载字体
        try:
            import platform
            system = platform.system()
            if system == 'Windows':
                font = ImageFont.truetype("msyh.ttc", font_size)
            elif system == 'Darwin':
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
            else:
                font = ImageFont.truetype("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_points.append((x, y))
            # 立即反馈
            cv2.circle(self.display_image, (x, y), 6, (0, 255, 0), -1)
            self.update_display()

    def update_display(self):
        """刷新显示画面"""
        if self.image is None: return
        
        display = self.image.copy()
        
        # 1. 绘制已确认的点
        for i, (px, py) in enumerate(self.clicked_points):
            if i < len(self.keypoint_indices):
                # 已关联
                kp_idx = self.keypoint_indices[i]
                label_text = f"P{i+1}:{self.config.labels[kp_idx]}"
                color = (0, 255, 0)
            else:
                # 未关联
                label_text = f"P{i+1}:?"
                color = (0, 255, 255)
            
            cv2.circle(display, (px, py), 6, color, -1)
            cv2.circle(display, (px, py), 10, (0, 0, 255), 1)
            cv2.putText(display, label_text, (px + 10, py - 10), self.font, 0.6, (255, 255, 255), 2)

        # 2. 绘制状态栏
        msg = ""
        count_click = len(self.clicked_points)
        count_idx = len(self.keypoint_indices)
        
        if count_click > count_idx:
            msg = f"请在控制台输入 P{count_click} 的对应标签..."
            color = (0, 255, 255)
        elif count_click < 4:
            msg = f"已标定 {count_click} 个点，至少需要 4 个。"
            color = (100, 100, 255)
        else:
            msg = f"已标定 {count_click} 个点。点击窗口并按 's' 保存，'r' 重置。"
            color = (0, 255, 0)

        display = self.put_chinese_text(display, msg, (20, 15), 24, color)
        
        self.display_image = display
        cv2.imshow(self.window_name, self.display_image)

    def select_keypoint_id(self, point_seq):
        """控制台交互：选择球场ID (修复死循环版)"""
        print("\n" + "=" * 50)
        print(f"🔵 正在为第 {point_seq} 个点 (P{point_seq}) 选择球场位置")
        print("请查看球场图，输入对应的数字标签（例如：输入 '1' 代表 '01' 点，输入 '20' 代表 '20' 点）")
        print("可用标签: " + ", ".join(self.config.labels))
        print("=" * 50)

        # 建立映射表
        label_map = {lbl: idx for idx, lbl in enumerate(self.config.labels)}

        while True:
            # 强制刷新缓冲区，确保提示显示
            import sys
            sys.stdout.flush()

            user_input = input(f"👉 请输入 P{point_seq} 的标签 [输入 q 撤销此点]: ").strip()

            # 1. 处理退出/撤销
            if user_input.lower() == 'q':
                print("-> 用户取消当前点")
                return None

            # 2. 尝试直接匹配 (例如用户输入 "01")
            if user_input in label_map:
                print(f"✅ 已关联: P{point_seq} -> 标签 {user_input}")
                return label_map[user_input]

            # 3. 尝试自动补零匹配 (例如用户输入 "1" -> 自动转为 "01")
            padded_input = user_input.zfill(2)  # "1" -> "01", "20" -> "20"
            if padded_input in label_map:
                print(f"✅ 已关联: P{point_seq} -> 标签 {padded_input}")
                return label_map[padded_input]

            # 4. 如果都不匹配，提示错误并继续循环
            print(f"❌ 输入 '{user_input}' (或 '{padded_input}') 不是有效的标签。")
            print(f"   请确保输入的是以下列表中的一个: {self.config.labels}")
            # 这里不return，继续下一次 while 循环，直到输入正确

    def annotate(self, frame: np.ndarray) -> Tuple[Optional[List[int]], Optional[List[Tuple[int, int]]]]:
        """主标定循环"""
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("输入的 frame 必须是有效的 numpy array")

        self.image = frame.copy()
        self.display_image = self.image.copy()
        self.clicked_points = []
        self.keypoint_indices = []

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        h, w = frame.shape[:2]
        display_w = min(1280, w)
        display_h = int(display_w * (h / w))
        cv2.resizeWindow(self.window_name, display_w, display_h)

        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        print("\n" + "-" * 50)
        print("🚀 标定开始！")
        print("1. 在图片窗口中【点击】关键点")
        print("2. 在控制台【输入】对应的标签 (如 1, 19, 20)")
        print("-" * 50)

        while True:
            self.update_display()
            # 缩短延时提高响应速度
            key = cv2.waitKey(30) & 0xFF

            # --- 关键逻辑修改 ---
            # 只有当点击的点数比已确认的ID多时，才进入输入模式
            # 且一次只处理一个，处理完后再继续循环
            if len(self.clicked_points) > len(self.keypoint_indices):
                # 暂时移除鼠标回调，防止在输入时误触
                cv2.setMouseCallback(self.window_name, lambda *args: None)

                # 调用输入函数
                idx = self.select_keypoint_id(len(self.clicked_points))
                
                print("✨ 标签录入完成，请点击图片窗口以激活焦点，然后继续操作。")

                # 恢复鼠标回调
                cv2.setMouseCallback(self.window_name, self.mouse_callback)

                if idx is not None:
                    self.keypoint_indices.append(idx)
                else:
                    # 用户选择了撤销 (q)
                    self.clicked_points.pop()
                    self.update_display()  # 立即刷新去除该点
                continue

            # 快捷键 (支持大小写)
            if key == ord('s') or key == ord('S'):
                if len(self.keypoint_indices) < 4:
                    print(f"⚠️ 点数不足 ({len(self.keypoint_indices)}/4)，无法保存。")
                else:
                    break
            elif key == ord('r') or key == ord('R'):
                self.clicked_points.clear()
                self.keypoint_indices.clear()
                self.display_image = self.image.copy()
                print("🔄 已重置所有标定")
            elif key == ord('d') or key == ord('D'):
                if self.clicked_points:
                    self.clicked_points.pop()
                    if len(self.keypoint_indices) > len(self.clicked_points):
                        self.keypoint_indices.pop()
                    print("⬅️ 已删除最后一个点")
            elif key == 27 or key == ord('q') or key == ord('Q'):  # ESC
                cv2.destroyAllWindows()
                return None, None

        cv2.destroyAllWindows()
        return self.keypoint_indices, self.clicked_points

    def save_results(self, output_json_path: str, source_identifier: str = "video_frame"):
        """
        保存结果
        
        Args:
            output_json_path: JSON文件的保存全路径
            source_identifier: 标识当前帧的来源（如 "cam01", "frame_105"），写入json以备查
        """
        if not self.keypoint_indices:
            return

        json_path_obj = Path(output_json_path)
        
        # 构造数据
        data = {
            "source": source_identifier,
            "config_type": self.config.__class__.__name__,
            "annotations": []
        }
        
        vis_img = self.image.copy()
        
        for i, (kp_idx, (x, y)) in enumerate(zip(self.keypoint_indices, self.clicked_points)):
            label = self.config.labels[kp_idx]
            phy_coord = self.config.vertices[kp_idx]
            
            data["annotations"].append({
                "id": i,
                "label": label,
                "img_coord": [int(x), int(y)],
                "world_coord": [int(phy_coord[0]), int(phy_coord[1])],
                "vertex_index": int(kp_idx)
            })
            
            # 绘制保存用的可视化图
            cv2.circle(vis_img, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(vis_img, label, (x+10, y-5), self.font, 0.6, (0, 255, 255), 2)
            
        # 保存JSON
        try:
            with open(json_path_obj, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"JSON已保存: {json_path_obj}")
        except Exception as e:
            print(f"保存JSON失败: {e}")

        # 保存图片 (文件名基于JSON文件名生成: name.json -> name_annotated.png)
        try:
            vis_path = json_path_obj.parent / f"{json_path_obj.stem}_annotated.png"
            # 使用 imwritex 替代 cv2.imwrite
            imwritex(str(vis_path), vis_img)
            print(f"可视化图已保存: {vis_path}")
        except Exception as e:
            print(f"保存图片失败: {e}")


class MarkerAnnotator:
    """
    通用标志物标定器
    
    用于标定多组标志物（例如：起始点、障碍杆、锥桶等）。
    支持通过 'n' 键切换/新建组。
    """
    
    def __init__(self):
        self.image = None
        self.display_image = None
        
        # 数据结构: List[List[Tuple[x, y]]]
        # marker_groups[0] 是第1组，marker_groups[1] 是第2组...
        self.marker_groups = [[]] 
        
        # 当前组索引
        self.current_group_idx = 0
        
        self.window_name = "Marker Annotation Tool"
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        
        # 预定义一些颜色 (BGR)，循环使用
        self.colors = [
            (0, 255, 0),    # 绿 (Group 1)
            (0, 0, 255),    # 红 (Group 2)
            (255, 0, 0),    # 蓝 (Group 3)
            (0, 255, 255),  # 黄 (Group 4)
            (255, 0, 255),  # 紫 (Group 5)
            (255, 255, 0)   # 青 (Group 6)
        ]

    def put_chinese_text(self, img, text, position, font_size=24, color=(255, 255, 255)):
        """复用中文绘制函数"""
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        try:
            import platform
            system = platform.system()
            if system == 'Windows':
                font = ImageFont.truetype("msyh.ttc", font_size)
            elif system == 'Darwin':
                font = ImageFont.truetype("/System/Library/Fonts/PingFang.ttc", font_size)
            else:
                font = ImageFont.truetype("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf", font_size)
        except:
            font = ImageFont.load_default()
        
        rgb_color = (color[2], color[1], color[0])
        draw.text(position, text, font=font, fill=rgb_color)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

    def mouse_callback(self, event, x, y, flags, param):
        """鼠标点击回调"""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 确保当前组列表存在
            while len(self.marker_groups) <= self.current_group_idx:
                self.marker_groups.append([])
                
            self.marker_groups[self.current_group_idx].append((x, y))
            self.update_display()

    def update_display(self):
        """刷新显示"""
        if self.image is None: return
        
        display = self.image.copy()
        
        # 1. 绘制所有已标定的点
        for group_id, points in enumerate(self.marker_groups):
            color = self.colors[group_id % len(self.colors)]
            
            for i, (px, py) in enumerate(points):
                cv2.circle(display, (px, py), 6, color, -1)
                # 标注 G1-1, G2-1 等
                label = f"G{group_id+1}-{i+1}"
                cv2.putText(display, label, (px + 10, py - 10), self.font, 0.5, color, 1)
            
            # 连接线 (可选，显示同一组的关系)
            if len(points) > 1:
                for i in range(len(points) - 1):
                    cv2.line(display, points[i], points[i+1], color, 1)

        # 2. 绘制状态提示
        curr_color = self.colors[self.current_group_idx % len(self.colors)]
        curr_count = len(self.marker_groups[self.current_group_idx])
        
        msg_lines = [
            f"当前: 第 {self.current_group_idx + 1} 组 (已标 {curr_count} 个)",
            "操作: [左键]添加点  [n]切换下一组  [d]撤销点  [s]保存退出"
        ]
        
        # 绘制多行文本
        y_pos = 20
        for line in msg_lines:
            display = self.put_chinese_text(display, line, (20, y_pos), 24, curr_color)
            y_pos += 35
        
        self.display_image = display
        cv2.imshow(self.window_name, self.display_image)

    def annotate(self, frame: np.ndarray) -> Dict[str, List[List[Tuple[int, int]]]]:
        """
        启动标定流程
        
        Returns:
            Dict: {
                "groups": [ [(x,y),...], [(x,y),...] ]
            }
        """
        self.image = frame.copy()
        self.display_image = self.image.copy()
        self.marker_groups = [[]]
        self.current_group_idx = 0
        
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        h, w = frame.shape[:2]
        display_w = min(1280, w)
        display_h = int(display_w * (h / w))
        cv2.resizeWindow(self.window_name, display_w, display_h)
        
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\n" + "=" * 50)
        print("🚩 通用标志物标定开始")
        print("1. 点击标记当前组的点")
        print("2. 按 'n' 完成当前组，开始标记下一组")
        print("3. 按 's' 保存所有数据并退出")
        print("=" * 50)

        while True:
            self.update_display()
            key = cv2.waitKey(30) & 0xFF
            
            # 切换/新建组
            if key == ord('n') or key == ord('N'):
                # 只有当当前组有点时，才允许切到下一组，或者允许空组跳过？
                # 策略：直接切到下一组，初始化为空列表
                self.current_group_idx += 1
                if len(self.marker_groups) <= self.current_group_idx:
                    self.marker_groups.append([])
                print(f"-> 切换至第 {self.current_group_idx + 1} 组")
            
            # 保存退出
            elif key == ord('s') or key == ord('S'):
                # 移除空组
                self.marker_groups = [g for g in self.marker_groups if g]
                print(f"-> 保存结果... 共 {len(self.marker_groups)} 组有效数据")
                break
                
            # 撤销上一个点
            elif key == ord('d') or key == ord('D'):
                if self.marker_groups and self.marker_groups[self.current_group_idx]:
                    self.marker_groups[self.current_group_idx].pop()
                    print("<- 删除当前组最后一个点")
                elif self.current_group_idx > 0 and not self.marker_groups[self.current_group_idx]:
                    # 如果当前组空了，按删除键是否回退到上一组？
                    # 策略：回退组索引
                    self.current_group_idx -= 1
                    print(f"<- 回退至第 {self.current_group_idx + 1} 组")
            
            # 退出
            elif key == 27 or key == ord('q') or key == ord('Q'):
                cv2.destroyAllWindows()
                return None

        cv2.destroyAllWindows()
        return {
            "groups": self.marker_groups
        }

    def save_results(self, output_json_path: str, source_identifier: str = "markers"):
        """保存标定结果到JSON"""
        data = {
            "source": source_identifier,
            "type": "marker_annotation",
            "groups": []
        }
        
        for g_idx, group in enumerate(self.marker_groups):
            group_data = {
                "group_id": g_idx,
                "points": []
            }
            for p_idx, pt in enumerate(group):
                group_data["points"].append({
                    "id": p_idx,
                    "img_coord": [int(pt[0]), int(pt[1])]
                })
            data["groups"].append(group_data)
            
        json_path = Path(output_json_path)
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ 结果已保存至: {json_path}")
            
            # 保存可视化图
            vis_img = self.display_image.copy()
            vis_path = json_path.parent / f"{json_path.stem}_annotated.png"
            imwritex(str(vis_path), vis_img)
            print(f"✅ 可视化图已保存至: {vis_path}")
            
        except Exception as e:
            print(f"❌ 保存失败: {e}")


# ==========================================
# 外部调用接口
# ==========================================

def load_keypoints(json_path: str) -> Tuple[List[int], List[Tuple[int, int]]]:
    """
    从JSON文件中加载关键点的indices和points
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        Tuple[List[int], List[Tuple[int, int]]]: 
            - indices: 顶点索引列表，例如 [19, 20, 21, 17, ...]
            - points: 图像坐标点列表，例如 [(1275, 1312), (1271, 773), ...]
    
    Raises:
        FileNotFoundError: JSON文件不存在
        ValueError: JSON格式错误或缺少必要字段
    """
    json_path = Path(json_path)
    
    if not json_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 检查必要字段
        if 'annotations' not in data:
            raise ValueError("JSON文件中缺少 'annotations' 字段")
        
        annotations = data['annotations']
        
        # 提取indices和points
        indices = []
        points = []
        
        for annotation in annotations:
            if 'vertex_index' not in annotation:
                raise ValueError("annotation中缺少 'vertex_index' 字段")
            if 'img_coord' not in annotation:
                raise ValueError("annotation中缺少 'img_coord' 字段")
            
            vertex_index = annotation['vertex_index']
            img_coord = annotation['img_coord']
            
            # 验证img_coord格式
            if not isinstance(img_coord, list) or len(img_coord) != 2:
                raise ValueError(f"img_coord格式错误，应为[x, y]格式: {img_coord}")
            
            indices.append(vertex_index)
            points.append((int(img_coord[0]), int(img_coord[1])))
        
        return indices, points
        
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON文件解析错误: {e}")
    except Exception as e:
        raise ValueError(f"加载关键点数据时发生错误: {e}")

def load_marker_keypoints(json_path: str) -> Dict[str, List[List[Tuple[int, int]]]]:
    """
    从JSON文件中加载标志物关键点
    返回格式: {'groups': [[(x,y),...], [(x,y),...], ...]}
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON文件不存在: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    groups_list = []
    # 按照 group_id 排序以确保顺序一致
    sorted_groups = sorted(data.get('groups', []), key=lambda x: x.get('group_id', 0))
    
    for group in sorted_groups:
        points = []
        # 提取该组所有点的 img_coord
        # 同样可以按照点 id 排序
        sorted_points = sorted(group.get('points', []), key=lambda x: x.get('id', 0))
        for point in sorted_points:
            if 'img_coord' in point:
                points.append(tuple(point['img_coord']))
        groups_list.append(points)
            
    return {'groups': groups_list}

def run_annotation(frame: np.ndarray, 
                   pitch_config: Any, 
                   output_json_path: str, 
                   frame_name: str = "frame"):
    """
    启动标定流程的封装函数
    
    Args:
        frame: 图像数据 (NumPy Array)
        pitch_config: 球场配置
        output_json_path: 结果保存路径 (.json)
        frame_name: 帧的名称/ID (用于记录)
    """
    annotator = KeypointAnnotator(pitch_config)
    try:
        indices, points = annotator.annotate(frame)
        if indices and len(indices) >= 4:
            annotator.save_results(output_json_path, frame_name)
            return indices, points
    except Exception as e:
        print(f"标定过程发生错误: {e}")
    return None, None


def run_marker_annotation(frame: np.ndarray, 
                          output_json_path: str, 
                          frame_name: str = "markers") -> Optional[Dict]:
    """
    启动通用标志物标定
    
    Args:
        frame: 图像数据
        output_json_path: 结果保存路径
        frame_name: 帧名称
    
    Returns:
        Dict: { "groups": [[(x,y),...], ...] }
    """
    annotator = MarkerAnnotator()
    try:
        results = annotator.annotate(frame)
        if results and results["groups"]:
            annotator.save_results(output_json_path, frame_name)
            return results
    except Exception as e:
        print(f"标志物标定错误: {e}")
    return None


def run_shot_train_annotation(frame: np.ndarray, 
                             output_json_path: str, 
                             frame_name: str = "shot_train"):
    """
    启动射门训练场景标定 (兼容模式)
    使用 MarkerAnnotator: Group 0 -> start_points, Group 1 -> obstacles
    """
    annotator = MarkerAnnotator()
    try:
        # 设置提示信息以模拟旧版 ShotTrain 体验 (可选)
        print("兼容模式: 第1组为起始点，第2组为障碍杆")
        
        results = annotator.annotate(frame)
        if results and results["groups"]:
            # 转换结果格式以保持兼容性
            groups = results["groups"]
            start_points = groups[0] if len(groups) > 0 else []
            obstacles = groups[1] if len(groups) > 1 else []
            
            compat_results = {
                "start_points": start_points,
                "obstacles": obstacles
            }
            
            # 保存时也使用兼容格式? 
            # MarkerAnnotator.save_results 保存的是通用格式。
            # 为了完全兼容，这里手动保存旧格式
            data = {
                "source": frame_name,
                "type": "shot_train_annotation",
                "start_points": [{"id": i, "img_coord": [int(p[0]), int(p[1])]} for i, p in enumerate(start_points)],
                "obstacles": [{"id": i, "img_coord": [int(p[0]), int(p[1])]} for i, p in enumerate(obstacles)]
            }
            
            json_path = Path(output_json_path)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            print(f"✅ (兼容模式) 结果已保存至: {json_path}")
            
            # 保存可视化图 (直接使用 annotator 的显示图)
            vis_path = json_path.parent / f"{json_path.stem}_annotated.png"
            imwritex(str(vis_path), annotator.display_image)
            print(f"✅ 可视化图已保存至: {vis_path}")

            return compat_results
            
    except Exception as e:
        print(f"射门训练标定错误: {e}")
    return None
