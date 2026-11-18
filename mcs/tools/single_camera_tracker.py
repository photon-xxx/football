"""
单相机追踪器 (Single Camera Tracker)

本模块是基于BoT-SORT的单相机追踪器，用于单个相机视野内的球员追踪。
适用于单相机场景，不涉及跨相机追踪。

对于多相机系统的全局追踪，请使用新的 global_pitch_tracker.py 模块。
"""

import numpy as np
import sys
import cv2
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any, TYPE_CHECKING
import math
import json
import copy
import shutil
from collections import Counter
import config

# 类型检查时导入，避免循环导入
if TYPE_CHECKING:
    from camera import Detection

try:
    from ocr import recognize_track, check_has_number, batch_recognize_tracks, batch_check_has_number
except ImportError as e:
    print(f"[SingleCameraTracker] Warning: Failed to import ocr module: {e}")
    recognize_track = None
    check_has_number = None
    batch_recognize_tracks = None
    batch_check_has_number = None

# ==========================================
# BoT-SORT 环境配置
# ==========================================
# 尝试将 BoT-SORT 路径加入 sys.path 以导入其模块
if config.BOTSORT_ROOT.exists() and str(config.BOTSORT_ROOT) not in sys.path:
    sys.path.append(str(config.BOTSORT_ROOT))
    print(f"[SingleCameraTracker] Added BoT-SORT to sys.path: {config.BOTSORT_ROOT}")

try:
    # 尝试导入 BoTSORT
    from tracker.bot_sort import BoTSORT
except ImportError as e:
    print(f"[SingleCameraTracker] Critical Error: Failed to import BoTSORT: {e}")
    # Debug info
    print(f"Current sys.path: {sys.path}")
    if config.BOTSORT_ROOT.exists():
        print(f"BoT-SORT directory exists at {config.BOTSORT_ROOT}")
        print(f"Contents: {[p.name for p in config.BOTSORT_ROOT.glob('*')]}")
        tracker_dir = config.BOTSORT_ROOT / "tracker"
        if tracker_dir.exists():
             print(f"tracker subdir contents: {[p.name for p in tracker_dir.glob('*')]}")
        else:
             print(f"tracker subdir NOT found in {config.BOTSORT_ROOT}")
    else:
        print(f"BoT-SORT directory NOT found at {config.BOTSORT_ROOT}")
        
    print("请确保 'BoT-SORT' 文件夹在项目根目录下，并且解决了 'tracker' 包名冲突问题。")
    BoTSORT = None


@dataclass
class BoTSORTArgs:
    """BoT-SORT 所需的参数配置"""
    # Tracking
    track_high_thresh: float = 0.5  # 高分检测阈值
    track_low_thresh: float = 0.1   # 低分检测阈值
    new_track_thresh: float = 0.6   # 新轨迹初始化阈值
    track_buffer: float = 10.0      # 轨迹丢失保留时间（秒）
    match_thresh: float = 0.8       # 匹配阈值
    
    # ReID
    with_reid: bool = True        # 是否启用 ReID (如果显存允许建议开启)
    # 默认路径需根据实际情况调整
    fast_reid_config: str = str(config.FAST_REID_CONFIG)
    fast_reid_weights: str = str(config.FAST_REID_WEIGHTS)
    device: str = "cuda"
    proximity_thresh: float = 0.5
    appearance_thresh: float = 0.2
    
    # CMC (Camera Motion Compensation)
    # 如果相机移动剧烈，建议开启 'sparseOptFlow'；如果相机固定或追求速度，设为 'none'
    cmc_method: str = "none" # 'orb', 'sift', 'ecc', 'sparseOptFlow', 'none'
    name: str = "MOT17"
    ablation: bool = False
    
    # Other
    mot20: bool = False


class JerseyNumberManager:
    """
    球衣号码管理：管理单个轨迹对象的图像截取、存储和合成。
    用于后续的号码识别或数据集构建。
    """
    def __init__(self, track_id: int):
        """
        Args:
            track_id: 轨迹ID
        """
        self.track_id = track_id
        self.raw_crops = []  # 存储原始 crop 图片 (未缩放)
        self.last_capture_frame = -999  # 上一次截取的帧号
        # Phase 2 采样间隔 
        self.capture_interval = 10  
        # Phase 1 采样间隔 (每30秒/900帧)
        self.first_glance_interval = 60 
        
        # 限制密集采样的最大图片数量
        self.max_crops = 1000 # [修改] 增大限制，确保 Pass 3 能采集完整的区间数据，后续导出时再均匀采样
        
        # OCR 提交数量限制
        self.ocr_target_count = 100 
        
        # First Glance 有效区间上限
        self.max_positives = 50
        self.last_first_glance_frame = -999

        self.saved_crops_count = 0
        
        # 定义 OCR 临时输入目录
        self.ocr_root = config.OCR_INPUT_DIR
        self.tmp_root = self.ocr_root / "tmp"
        self.first_root = self.ocr_root / "first"
        
        # 清理旧数据，确保 1-to-1 对应
        self._clean_dir(self.tmp_root / f"track{self.track_id}")
        self._clean_dir(self.first_root / f"track{self.track_id}")
        self._clean_dir(self.ocr_root / f"track{self.track_id}")

        self.ocr_results = [] # 存储 OCR 原始结果 (List[str])
        self.final_number = None # 最终投票结果
        self.confidence = 0.0 # 投票置信度

    def _clean_dir(self, path: Path):
        if path.exists():
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"[SingleCameraTracker] Warning: Failed to clean dir {path}: {e}")

    def update(self, track, frame: np.ndarray, frame_id: int):
        """
        更新状态，决定是否截取当前帧
        Args:
            track: STrack 对象，包含 .tlwh 属性
            frame: 原始视频帧
            frame_id: 当前帧号
        """
        # 0. 空间遮挡检查
        if hasattr(track, 'iou') and track.iou > 0:
            return

        # 1. 基础采样频率检查 (Phase 2)
        if frame_id - self.last_capture_frame < self.capture_interval:
            return

        # 2. 获取检测框坐标
        x1, y1, w, h = track.tlwh
        
        # 3. 坐标边界检查
        img_h, img_w = frame.shape[:2]
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(img_w, x1 + w))
        y2 = int(min(img_h, y1 + h))
        
        if x2 <= x1 or y2 <= y1:
            return

        # 4. 截取原始图像
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return

        # 5. 存储原始 Crop (用于最后合成大图)
        self.raw_crops.append(crop)
        
        # 6. 保存 Phase 2 密集采样图
        # [修改] 初始追踪阶段暂停保存密集采样图，仅在 Refine 阶段根据 First Glance 结果回溯采样
        # track_dir = self.tmp_root / f"track{self.track_id}"
        # track_dir.mkdir(parents=True, exist_ok=True)
        # crop_path = track_dir / f"crop_{frame_id:06d}.jpg"
        # cv2.imwrite(str(crop_path), crop)
        
        self.last_capture_frame = frame_id

        # 7. 保存 Phase 1 稀疏采样图 (First Glance)
        # 策略：如果距离上次 First Glance 超过间隔，或者这是第一张图
        if (self.last_first_glance_frame == -999) or \
           (frame_id - self.last_first_glance_frame >= self.first_glance_interval):
            
            first_dir = self.first_root / f"track{self.track_id}"
            first_dir.mkdir(parents=True, exist_ok=True)
            first_path = first_dir / f"check_{frame_id:06d}.jpg"
            cv2.imwrite(str(first_path), crop)
            
            self.last_first_glance_frame = frame_id

    def save_dense_crop(self, frame, frame_id, track):
        """[新增] 回溯采样时手动保存密集 Crop 到 tmp 目录"""
        # 0. 数量上限检查
        if self.saved_crops_count >= self.max_crops:
            return

        # 1. 坐标边界检查
        x1, y1, w, h = track.tlwh
        img_h, img_w = frame.shape[:2]
        x1 = int(max(0, x1))
        y1 = int(max(0, y1))
        x2 = int(min(img_w, x1 + w))
        y2 = int(min(img_h, y1 + h))
        
        if x2 <= x1 or y2 <= y1:
            return

        # 2. 截取原始图像
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return
            
        # 3. 存储
        track_dir = self.tmp_root / f"track{self.track_id}"
        track_dir.mkdir(parents=True, exist_ok=True)
        crop_path = track_dir / f"crop_{frame_id:06d}.jpg"
        cv2.imwrite(str(crop_path), crop)
        self.saved_crops_count += 1
        
        # 4. 同时添加到 raw_crops 列表供合成图使用 (注意避免重复添加)
        # 由于我们是在第二遍遍历时添加，需要清空之前的 raw_crops 吗？
        # refine_track_video 会重置 manager，所以这里添加是安全的
        self.raw_crops.append(crop)

    def generate_composite_image(self) -> Optional[np.ndarray]:
        """
        将所有收集到的 crops 合成为一张大图
        使用自适应网格大小，并允许放大图片以减少黑边。
        
        Returns:
            composite: 合成后的图像，如果没有 crops 则返回 None
        """
        if not self.raw_crops:
            return None

        # 1. 计算自适应网格大小
        # 使用 90% 分位数来确定网格尺寸，避免极个别异常大框撑大整个网格
        # 同时允许小图放大以填充网格，减少黑边
        heights = [c.shape[0] for c in self.raw_crops]
        widths = [c.shape[1] for c in self.raw_crops]
        
        # 至少保证有一定大小
        grid_h = int(max(64, np.percentile(heights, 90)))
        grid_w = int(max(32, np.percentile(widths, 90)))

        count = len(self.raw_crops)
        # 计算合成图的行数和列数 (尽量方形)
        cols = int(math.ceil(math.sqrt(count)))
        rows = int(math.ceil(count / cols))
        
        comp_w = cols * grid_w
        comp_h = rows * grid_h
        
        # 增加 Header 区域显示结果
        header_h = 30
        full_h = comp_h + header_h
        
        composite = np.zeros((full_h, comp_w, 3), dtype=np.uint8)
        
        # 绘制 Header 信息
        header_text = f"Track {self.track_id}"
        if self.final_number:
            header_text += f" | Jersey: {self.final_number} (Conf: {self.confidence:.0%})"
        else:
            header_text += " | Jersey: N/A"
            
        cv2.putText(composite, header_text, (10, 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 计算 OCR 结果对应的索引 (复现 export_to_ocr_input 的采样逻辑)
        ocr_indices = set()
        ocr_target_count = self.ocr_target_count # 使用成员变量
        if count > ocr_target_count:
            indices = np.linspace(0, count - 1, ocr_target_count, dtype=int)
            # 必须与 export 逻辑保持完全一致: sorted(list(set(indices)))
            # 实际上 linspace 在 count > target_count 时生成的已经是 unique sorted
            ocr_indices = set(indices)
        else:
            ocr_indices = set(range(count))
            
        ocr_res_ptr = 0 # 指向 self.ocr_results 的指针
        
        for i, raw_crop in enumerate(self.raw_crops):
            r = i // cols
            c = i % cols
            
            # 处理 Crop 以适应动态计算的网格 (允许放大)
            processed_crop = self._fit_to_grid(raw_crop, grid_w, grid_h)
            
            y_start = header_h + r * grid_h # 注意加上 header_h 偏移
            x_start = c * grid_w
            
            composite[y_start:y_start+grid_h, x_start:x_start+grid_w] = processed_crop
            
            # 如果当前 crop 是被 OCR 采样过的，且有结果
            if i in ocr_indices and ocr_res_ptr < len(self.ocr_results):
                res_text = self.ocr_results[ocr_res_ptr]
                ocr_res_ptr += 1
                
                # 简短显示，如果是 Error 显示 Err
                display_text = "Err" if "Error" in res_text else res_text
                # 绘制在左下角 (黄色)
                cv2.putText(composite, display_text, (x_start + 2, y_start + grid_h - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
        return composite

    def save_composite_image(self, output_dir: str):
        """
        将合成的大图保存到指定目录
        """
        composite = self.generate_composite_image()
        if composite is None:
            return
            
        # 保存
        out_path = Path(output_dir) / f"track_{self.track_id}_crops.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), composite)

    def get_sampling_intervals(self) -> List[Tuple[int, int]]:
        """
        [DEPRECATED] 该方法包含实时 OCR 调用逻辑，已废弃。
        请使用 get_first_glance_candidates 和 compute_intervals_from_results 替代。
        """
        return []

    def get_first_glance_candidates(self) -> List[Path]:
        """
        获取需要进行 First Glance 检查的候选图片路径列表。
        采用 bisect_indices 逻辑优化检查顺序。
        """
        first_dir = self.first_root / f"track{self.track_id}"
        
        if not first_dir.exists():
            return []
            
        first_images = sorted(list(first_dir.glob("*.jpg")))
        if not first_images:
            return []
            
        # 生成二分/均匀分布的遍历顺序
        def bisect_indices(n):
            if n == 0: return []
            queue = [(0, n)] # [start, end)
            indices = []
            visited = set()
            
            while queue:
                start, end = queue.pop(0)
                if start >= end: continue
                
                mid = (start + end) // 2
                if mid not in visited:
                    indices.append(mid)
                    visited.add(mid)
                    
                queue.append((start, mid))
                queue.append((mid + 1, end))
            return indices
            
        search_indices = bisect_indices(len(first_images))
        
        # 返回按优化顺序排列的图片路径
        return [first_images[i] for i in search_indices]

    def compute_intervals_from_results(self, results: Dict[str, bool]) -> List[Tuple[int, int]]:
        """
        根据 Batch API 的结果计算采样区间。
        
        Args:
            results: {filename: has_number(bool)}
        """
        intervals = []
        positive_count = 0
        MAX_POSITIVES = self.max_positives
        
        # 遍历结果
        for filename, has_number in results.items():
            if positive_count >= MAX_POSITIVES:
                break
                
            if has_number:
                try:
                    # check_{frame_id}.jpg
                    frame_idx = int(filename.split('_')[1].split('.')[0])
                    
                    # 标记区间 [T - interval, T + interval]
                    start = max(0, frame_idx - self.capture_interval * 2) 
                    end = frame_idx + self.capture_interval * 2
                    intervals.append((start, end))
                    positive_count += 1
                except:
                    pass
                
        intervals.sort(key=lambda x: x[0])
        return intervals

    def export_to_ocr_input(self, new_id: int):
        """
        将 tmp 目录下的 crops 筛选后复制到 ocr_input/track{new_id}
        """
        old_path = self.tmp_root / f"track{self.track_id}"
        final_path = self.ocr_root / f"track{new_id}"
        
        if final_path.exists(): shutil.rmtree(final_path)
        final_path.mkdir(parents=True, exist_ok=True)
        
        if old_path.exists():
            try:
                # 1. 获取所有源文件并排序 (按帧号)
                src_files = sorted(list(old_path.glob("*.jpg")))
                
                if not src_files:
                    return new_id
                    
                # 2. 均匀采样
                target_count = self.ocr_target_count
                if len(src_files) > target_count:
                    import numpy as np
                    indices = np.linspace(0, len(src_files) - 1, target_count, dtype=int)
                    indices = sorted(list(set(indices))) 
                    files_to_copy = [src_files[i] for i in indices]
                else:
                    files_to_copy = src_files
                
                # 3. 复制选中的文件
                for f in files_to_copy:
                    shutil.copy2(str(f), str(final_path / f.name))
                    
            except OSError as e:
                print(f"[SingleCameraTracker] Error copying {old_path} to {final_path}: {e}")
                
        return new_id

    def get_voted_jersey_number(self, override_track_id: Optional[int] = None) -> Optional[str]:
        """
        两阶段识别逻辑：
        1. 检查 First Glance (稀疏采样) 是否有数字。
        2. 如果有，则识别所有密集采样的图片并投票。
        """
        if recognize_track is None:
            print("[SingleCameraTracker] Error: 'ocr' module not available.")
            return None

        # 如果 self.ocr_results 为空，尝试实时识别
        if not self.ocr_results:
            input_dir = self.ocr_root / f"track{self.track_id}"
            if input_dir.exists() and list(input_dir.glob("*.jpg")):
                print(f"[SingleCameraTracker] Track {self.track_id} falling back to real-time recognition...")
                raw_results = recognize_track(self.track_id)
                self.ocr_results = raw_results if raw_results else []
            else:
                return None
        
        if not self.ocr_results:
            return None
            
        # 2. 过滤结果
        valid_results = []
        import re
        for res in self.ocr_results:
            res = str(res).strip()
            if not res or res.lower() == "none" or res == "error":
                continue
            
            found_numbers = re.findall(r'\d+', res)
            
            if found_numbers:
                valid_results.extend(found_numbers)
                
        if not valid_results:
            return None
            
        # 3. 投票
        counts = Counter(valid_results)
        top_result = counts.most_common(1)
        
        if top_result:
            number, count = top_result[0]
            confidence = count / len(valid_results)
            
            self.final_number = number
            self.confidence = confidence
            
            return number
            
        return None


    def _fit_to_grid(self, crop: np.ndarray, target_w: int, target_h: int) -> np.ndarray:
        """
        将 crop 放入固定大小的网格中。
        策略：
        - 保持长宽比 (Aspect Fit)
        - 总是缩放 (包括放大) 以触及网格边界
        - 居中填充
        """
        h, w = crop.shape[:2]
        
        # 计算缩放比例
        scale = min(target_w / w, target_h / h)
        
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # 缩放图像
        if scale != 1.0:
            resized_crop = cv2.resize(crop, (new_w, new_h))
        else:
            resized_crop = crop
            
        # 计算居中位置
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        
        # 创建黑色背景画布
        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        # 贴图 (注意边界安全)
        x_end = min(x_offset + new_w, target_w)
        y_end = min(y_offset + new_h, target_h)
        
        crop_x_end = x_end - x_offset
        crop_y_end = y_end - y_offset
        
        canvas[y_offset:y_end, x_offset:x_end] = resized_crop[:crop_y_end, :crop_x_end]
        
        return canvas


class tracker_botsort:
    """单相机BoT-SORT追踪器"""
    def __init__(self, camera, args: Optional[BoTSORTArgs] = None):
        """
        初始化 BoT-SORT 追踪器适配器
        
        Args:
            camera: Camera 对象，用于坐标转换和范围筛选
            args: BoTSORTArgs 配置对象，如果为 None 则使用默认值
        """
        self.camera = camera
        self.args = args if args else BoTSORTArgs()
        
        if BoTSORT is None:
            raise ImportError(
                "BoT-SORT module not loaded correctly. \n"
                "Possible reasons:\n"
                "1. 'BoT-SORT' folder is missing or empty on the server.\n"
                "2. 'tracker' package conflict.\n"
                "3. Missing dependencies for BoT-SORT.\n"
                f"Checked path: {config.BOTSORT_ROOT}"
            )
            
        # 初始化 BoT-SORT
        # 获取视频实际帧率
        video_fps = self.camera.video_manager.get_fps()
        if video_fps <= 0:
            print(f"[SingleCameraTracker] ⚠️ 无法获取视频帧率，使用默认值 30 FPS")
            video_fps = 30
        else:
            print(f"[SingleCameraTracker] 检测到视频帧率: {video_fps:.2f} FPS")
        
        self.tracker = BoTSORT(self.args, frame_rate=video_fps)
        
        # # [Global Cleanup] 清理 OCR 缓存目录
        # ocr_root = config.OCR_INPUT_DIR
        # if ocr_root.exists():
        #     try:
        #         shutil.rmtree(ocr_root)
        #         print("[SingleCameraTracker] Cleaned up previous OCR cache.")
        #     except Exception as e:
        #         print(f"[SingleCameraTracker] Warning: Failed to clean OCR cache: {e}")
        # ocr_root.mkdir(parents=True, exist_ok=True)
        
        # # 管理所有轨迹的 crop manager
        # self.crop_managers: Dict[int, JerseyNumberManager] = {}
        
        print(f"[SingleCameraTracker] BoT-SORT initialized for Camera {camera.id}")
        if self.args.with_reid:
            print("[SingleCameraTracker] ReID module enabled.")

    def update(self, detections=None, frame: np.ndarray = None):
        """
        [内部方法] 处理单帧追踪
        
        Args:
            detections: List[Detection], 检测结果
                       - 如果为 None，使用 camera.detection.detections_filtered (默认)
                       - Detection对象包含预计算的ReID特征和球场坐标
            frame: 当前视频帧
                   - 如果为 None，使用 camera.detection.frame (默认)
            
        Returns:
            tracks: List[STrack], BoT-SORT 的追踪结果
        """
        # 如果未提供 detections，使用相机当前帧的检测结果
        if detections is None:
            if hasattr(self.camera, 'detection') and self.camera.detection is not None:
                detections = self.camera.detection.detections_filtered
                if detections is None:
                    detections = []
            else:
                raise ValueError(
                    "[SingleCameraTracker] detections 参数为 None，且 camera.detection 未初始化。"
                    "请先调用 camera.init_detection() 或显式传入 detections 参数。"
                )
        
        # 如果未提供 frame，使用相机检测时的帧
        if frame is None:
            if hasattr(self.camera, 'detection') and self.camera.detection is not None:
                frame = self.camera.detection.frame
                if frame is None:
                    raise ValueError(
                        "[SingleCameraTracker] frame 参数为 None，且 camera.detection.frame 未设置。"
                        "请先调用 camera.detection.detect() 或显式传入 frame 参数。"
                    )
            else:
                raise ValueError(
                    "[SingleCameraTracker] frame 参数为 None，且 camera.detection 未初始化。"
                )
        
        # 直接传递 Detection 对象列表给 BoT-SORT
        # BoT-SORT 会自动处理 Detection 对象，复用预计算的 ReID 特征和球场坐标
        tracks = self.tracker.update(detections, frame)

        # 更新 Crops
        self._update_crops(tracks, frame)
        
        return tracks

    def _update_pitch_positions(self, tracks):
        """
        计算并更新所有轨迹的二维球场坐标（传统方法，用于回退）
        """
        if not tracks:
            return

        pixel_points = []
        for t in tracks:
            x1, y1, w, h = t.tlwh
            foot_x = x1 + w / 2.0
            foot_y = y1 + h
            pixel_points.append([foot_x, foot_y])
        
        pixel_points = np.array(pixel_points, dtype=np.float32)
        pitch_points = self.camera.pixel_to_pitch(pixel_points)

        for i, t in enumerate(tracks):
            t.position = pitch_points[i]

    def _update_crops(self, tracks, frame):
        """更新每个活跃轨迹的截图"""
        for t in tracks:
            tid = t.track_id
            if tid not in self.crop_managers:
                self.crop_managers[tid] = JerseyNumberManager(tid)
            
            self.crop_managers[tid].update(t, frame, self.tracker.frame_id)

    def save_all_crops(self, output_dir: str):
        """在视频处理结束后调用，保存所有轨迹的合成图"""
        print(f"[SingleCameraTracker] Saving crop summaries to {output_dir}...")
        for tid, manager in self.crop_managers.items():
            manager.save_composite_image(output_dir)

    def track_video(self):
        """
        处理整个视频流：检测 -> 追踪 -> 筛选
        
        Yields:
            tuple: (frame_id, final_tracks, frame)
        """
        print(f"[SingleCameraTracker] 开始追踪视频流: {self.camera.video_path}")
        
        for frame_id, detections, frame in self.detect_video():
            raw_tracks = self.update(detections, frame)
            yield frame_id, raw_tracks, frame

    def refine_track_video(self, min_duration=30):
        """
        [高级方法] 执行追踪，并进行后处理：过滤短轨迹、重新分配ID
        
        该方法会进行多次视频遍历。
        
        Yields:
            tuple: (frame_id, filtered_tracks, frame)
        """
        print(f"[SingleCameraTracker] 开始精细化追踪 (过滤短轨迹 < {min_duration} 帧)...")
        
        # --- Initialization ---
        self.tracker = BoTSORT(self.args, frame_rate=30)
        self.crop_managers = {}
        
        raw_tracks_history = {}
        track_counts = {}
        
        # ======================================================
        # Pass 1: 初始追踪 & 稀疏采样
        # ======================================================
        print("\n=== Pass 1: Tracking & Sparse Sampling ===")
        for frame_id, tracks, _ in self.track_video():
            if frame_id % 30 == 0:
                 print(f"  Analyzing frame {frame_id}... Found tracks: {len(tracks)}", end="\r")

            frame_tracks = []
            for t in tracks:
                track_counts[t.track_id] = track_counts.get(t.track_id, 0) + 1
                frame_tracks.append(self._strip_track_for_storage(t))
            
            raw_tracks_history[frame_id] = frame_tracks
            
        # ======================================================
        # Pass 2: 分析与筛选
        # ======================================================
        print("\n\n=== Pass 2: Analysis & ID Remapping ===")
        
        valid_ids = {tid for tid, count in track_counts.items() if count >= min_duration}
        removed_count = len(track_counts) - len(valid_ids)
        print(f"[SingleCameraTracker] 轨迹筛选: 总数 {len(track_counts)} -> 保留 {len(valid_ids)} (移除 {removed_count})")
        
        ids_with_potential = []
        ids_others = []
        sampling_schedule = {} 
        
        print(f"[SingleCameraTracker] 正在收集 First Glance 图片数据...")
        all_first_glance_requests = []
        
        for old_id in sorted(list(valid_ids)):
            mgr = self.crop_managers.get(old_id)
            if not mgr: continue
            
            candidates = mgr.get_first_glance_candidates()
            for img_path in candidates:
                all_first_glance_requests.append((old_id, img_path))
                
        print(f"[SingleCameraTracker] 共收集到 {len(all_first_glance_requests)} 张图片用于初筛检查。")
        
        first_glance_results = {}
        
        if all_first_glance_requests and batch_check_has_number:
            print("[SingleCameraTracker] 提交 Batch 任务进行号码检查...")
            first_glance_results = batch_check_has_number(all_first_glance_requests)
        
        print(f"[SingleCameraTracker] 分析 Batch 结果并生成采样计划...")
        for old_id in sorted(list(valid_ids)):
            mgr = self.crop_managers.get(old_id)
            if not mgr: continue
            
            track_results = first_glance_results.get(old_id, {})
            
            if track_results:
                intervals = mgr.compute_intervals_from_results(track_results)
                if intervals:
                    ids_with_potential.append(old_id)
                    sampling_schedule[old_id] = intervals
                else:
                    ids_others.append(old_id)
            else:
                ids_others.append(old_id)
                     
        print(f"  Analysis Done.")
        
        # ID 重排
        id_map = {}
        current_new_id = 1
        
        for old_id in ids_with_potential:
            id_map[old_id] = current_new_id
            current_new_id += 1
            
        for old_id in ids_others:
            id_map[old_id] = current_new_id
            current_new_id += 1
            
        print(f"[SingleCameraTracker] ID 重排完成: {len(ids_with_potential)} 个轨迹被标记为有潜力 (ID 1-{len(ids_with_potential)})")

        # ======================================================
        # Pass 3: 回溯密集采样
        # ======================================================
        print("\n=== Pass 3: Dense Resampling ===")
        
        if sampling_schedule:
            print(f"[SingleCameraTracker] 需回溯采样的轨迹数: {len(sampling_schedule)}")
            
            frame_generator = self.camera.get_frame_generator()
            
            for frame_id, frame in enumerate(frame_generator):
                if frame is None: break
                
                if frame_id in raw_tracks_history:
                    raw_tracks = raw_tracks_history[frame_id]
                    
                    for t in raw_tracks:
                        old_id = t.track_id
                        if old_id in sampling_schedule:
                            in_interval = False
                            for start, end in sampling_schedule[old_id]:
                                if start <= frame_id <= end:
                                    in_interval = True
                                    break
                            
                            if in_interval:
                                self.crop_managers[old_id].save_dense_crop(frame, frame_id, t)
                                
            print("[SingleCameraTracker] 回溯采样完成.")
        else:
            print("[SingleCameraTracker] 无需回溯采样.")

        # ======================================================
        # Pass 4: 导出与识别
        # ======================================================
        print("\n=== Pass 4: Export to OCR & Recognition ===")
        
        print("[SingleCameraTracker] 导出数据到 ocr_input...")
        
        new_crop_managers = {}
        
        for old_id in valid_ids:
            new_id = id_map[old_id]
            manager = self.crop_managers[old_id]
            
            if old_id in ids_with_potential:
                manager.export_to_ocr_input(new_id)
            
            new_crop_managers[new_id] = manager
            
        self.crop_managers = new_crop_managers
        
        print("[SingleCameraTracker] 执行 OCR 识别...")
        track_numbers = {}
        
        group1_count = len(ids_with_potential)
        target_track_ids = list(range(1, group1_count + 1))

        if target_track_ids and batch_recognize_tracks:
            print(f"[SingleCameraTracker] 正在批量识别 {len(target_track_ids)} 个轨迹...")
            batch_results_map = batch_recognize_tracks(target_track_ids)
            
            for new_id in target_track_ids:
                manager = self.crop_managers.get(new_id)
                if not manager: continue
                
                raw_results = batch_results_map.get(new_id, [])
            
                if raw_results:
                    valid_nums = []
                    import re
                    for res in raw_results:
                        res = str(res).strip()
                        if res and res.lower() not in ["none", "error", ""]:
                            nums = re.findall(r'\d+', res)
                            if nums: valid_nums.extend(nums)
                    
                    if valid_nums:
                        counts = Counter(valid_nums)
                        top = counts.most_common(1)[0]
                        
                        number = top[0]
                        conf = top[1] / len(valid_nums)
                        
                        track_numbers[new_id] = number
                        manager.final_number = number
                        manager.confidence = conf
                        manager.ocr_results = raw_results
                        print(f"[SingleCameraTracker] Track {new_id} voted number: {number} (Conf: {conf:.0%})")
                    
        print(f"\n[SingleCameraTracker] 识别完成. 成功识别出 {len(track_numbers)} 个号码.")

        # ======================================================
        # Pass 5: 生成最终结果
        # ======================================================
        print("\n=== Pass 5: Generating Final Result Stream ===")
        
        frame_generator = self.camera.get_frame_generator()
        
        for frame_id, frame in enumerate(frame_generator):
            if frame is None: break
            
            final_tracks = []
            
            if frame_id in raw_tracks_history:
                raw_tracks = raw_tracks_history[frame_id]
                for t in raw_tracks:
                    if t.track_id in valid_ids:
                        new_id = id_map[t.track_id]
                        t.track_id = new_id
                        
                        if new_id in track_numbers:
                            t.jersey_number = track_numbers[new_id]
                        
                        final_tracks.append(t)
            
            yield frame_id, final_tracks, frame

    def _safe_rmtree(self, path: Path):
        if path.exists():
            try:
                shutil.rmtree(path)
            except Exception as e:
                print(f"[SingleCameraTracker] Warning: Failed to delete {path}: {e}")

    def _strip_track_for_storage(self, track):
        """创建 STrack 的轻量级副本"""
        new_t = type(track)(track.tlwh, track.score, feat=None)
        new_t.track_id = track.track_id
        new_t.jersey_number = getattr(track, 'jersey_number', None)
        new_t.position = getattr(track, 'position', None)
        new_t.iou = getattr(track, 'iou', 0.0)
        
        return new_t

    def detect_video(self):
        """
        生成器函数：遍历视频帧，逐帧执行检测和筛选
        
        Yields:
            tuple: (frame_id, detections, frame)
        """
        if self.camera.detection is None:
            raise RuntimeError(f"[SingleCameraTracker] Camera {self.camera.id} detection module not initialized.")

        frame_generator = self.camera.get_frame_generator()
        
        for frame_id, frame in enumerate(frame_generator):
            if frame is None:
                break
                
            self.camera.detection.detect(frame, conf=0.25)
            detections = self.camera.detection.get_filtered_detections()
            
            yield frame_id, detections, frame

    def draw_tracks(self, frame, tracks, alpha=0.4):
        """
        可视化追踪结果：在图像上绘制半透明边界框和清晰的ID标签
        """
        if not tracks:
            return frame
            
        vis_frame = frame.copy()
        overlay = frame.copy()
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), 
            (255, 255, 0), (255, 0, 255), (0, 255, 255),
            (128, 0, 0), (0, 128, 0), (0, 0, 128)
        ]
        
        for t in tracks:
            tlwh = t.tlwh
            x1, y1, w, h = tlwh
            x2, y2 = x1 + w, y1 + h
            
            box = np.array([x1, y1, x2, y2]).astype(int)
            color = colors[int(t.track_id) % len(colors)]
            
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, -1)
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 4)
            
            if getattr(t, 'jersey_number', None):
                label = f"No.{t.jersey_number}"
            else:
                label = f"ID:{t.track_id}"
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            
            cv2.rectangle(overlay, 
                          (box[0], box[1] - text_h - 6), 
                          (box[0] + text_w, box[1]), 
                          color, -1)
            
        cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0, vis_frame)
        
        for t in tracks:
            tlwh = t.tlwh
            x1, y1 = int(tlwh[0]), int(tlwh[1])
            if getattr(t, 'jersey_number', None):
                label = f"No.{t.jersey_number}"
            else:
                label = f"ID:{t.track_id}"
            
            cv2.putText(vis_frame, label, (x1, y1 - 4), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
             
        return vis_frame

    def save_tracking_video(self, output_path: str, alpha=0.4, refine=True):
        """
        [工具方法] 处理整个视频，绘制追踪结果并保存为 MP4 文件
        """
        from tools.frame_tool import VideoWriterStream
        
        print(f"[SingleCameraTracker] 正在生成可视化视频: {output_path}")
        
        output_path_obj = Path(output_path)
        crops_dir = output_path_obj.parent / f"{output_path_obj.stem}_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        
        if refine:
            track_generator = self.refine_track_video()
        else:
            track_generator = self.track_video()
        
        tracking_data_list = []

        with VideoWriterStream(output_path, fps=30.0) as writer:
            
            for frame_id, tracks, frame in track_generator:
                vis_frame = self.draw_tracks(frame, tracks, alpha=alpha)
                
                frame_info = {
                    "frame_id": int(frame_id),
                    "tracks": []
                }
                for t in tracks:
                    x, y, w, h = t.tlwh
                    
                    track_info = {
                        "track_id": int(t.track_id),
                        "bbox": [float(x), float(y), float(w), float(h)],
                        "score": float(t.score),
                        "jersey_number": getattr(t, 'jersey_number', None),
                        "position": getattr(t, 'position', None).tolist() if getattr(t, 'position', None) is not None else None
                    }
                    frame_info["tracks"].append(track_info)
                tracking_data_list.append(frame_info)

                writer.write(vis_frame)
                
                if frame_id % 30 == 0:
                     print(f"  Saving frame {frame_id}... Tracks: {len(tracks)}", end="\r")
        
        self.save_all_crops(str(crops_dir))

        video_path_obj = Path(output_path)
        store_dir = config.STORE_DIR
        store_dir.mkdir(parents=True, exist_ok=True)
        
        json_path = store_dir / f"{video_path_obj.stem}.json"
        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(tracking_data_list, f, indent=2)
            print(f"\n[SingleCameraTracker] Tracking data saved to {json_path}")
        except Exception as e:
            print(f"\n[SingleCameraTracker] Error saving JSON data: {e}")

        
        print(f"\n[SingleCameraTracker] 视频已保存至: {output_path}")
