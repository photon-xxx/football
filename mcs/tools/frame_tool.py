import cv2
import numpy as np
from pathlib import Path
from typing import Union, Tuple, List


def load_frame(image_path: Union[str, Path]) -> np.ndarray:
    """
    读取图片文件为frame（numpy数组）
    
    Args:
        image_path: 图片文件路径（支持str或Path对象）
    
    Returns:
        np.ndarray: 图像数组，形状为(H, W, 3)，BGR格式
                   如果读取失败返回None
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 无法读取图像
    """
    image_path = Path(image_path)
    
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在: {image_path}")
    
    frame = cv2.imread(str(image_path))
    
    if frame is None:
        raise ValueError(f"无法读取图像文件: {image_path}")
    
    return frame


def save_frame(frame: np.ndarray, output_path: Union[str, Path]) -> bool:
    """
    保存frame为图片文件
    
    Args:
        frame: 图像数组，形状为(H, W, 3)，BGR格式
        output_path: 输出图片文件路径（支持str或Path对象）
                    支持格式：.jpg, .png, .bmp等
    
    Returns:
        bool: 保存成功返回True，失败返回False
    
    Raises:
        ValueError: frame为空或格式不正确
    """
    if frame is None or frame.size == 0:
        raise ValueError("frame为空，无法保存")
    
    output_path = Path(output_path)
    
    # 确保输出目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 使用 imencode + numpy 写入，兼容含中文的路径（cv2.imwrite 在 Windows 上不支持非 ASCII 路径）
    ext = output_path.suffix.lower() or ".jpg"
    ret, buf = cv2.imencode(ext, frame)
    if not ret:
        raise ValueError(f"保存图像失败（编码错误）: {output_path}")
    buf.tofile(str(output_path))

    return True

def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames , original_fps

class VideoWriterStream:
    """
    流式视频写入器，支持上下文管理器。
    
    使用示例:
        with VideoWriterStream("output.mp4", fps=30, resolution=(1920, 1080)) as writer:
            for frame in frames:
                writer.write(frame)
    """
    def __init__(self, output_path: str, fps: float = 30.0, resolution: Tuple[int, int] = None, fourcc: str = 'mp4v'):
        self.output_path = str(output_path)
        self.fps = fps
        self.resolution = resolution # (width, height)
        self.fourcc_code = fourcc
        self.writer = None
        
        # 确保目录存在
        Path(self.output_path).parent.mkdir(parents=True, exist_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def write(self, frame: np.ndarray):
        """
        写入一帧。如果是第一帧且未指定分辨率，会自动初始化 Writer。
        """
        if frame is None:
            return

        if self.writer is None:
            # 如果初始化时没给分辨率，就用第一帧的分辨率
            if self.resolution is None:
                h, w = frame.shape[:2]
                self.resolution = (w, h)
            
            fourcc = cv2.VideoWriter_fourcc(*self.fourcc_code)
            self.writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, self.resolution)
            
            if not self.writer.isOpened():
                raise RuntimeError(f"无法创建视频文件: {self.output_path}")

        # 如果分辨率不匹配，尝试 resize (安全网)
        if (frame.shape[1], frame.shape[0]) != self.resolution:
             frame = cv2.resize(frame, self.resolution)

        self.writer.write(frame)

    def release(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None

def save_video(ouput_video_frames,output_video_path):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (ouput_video_frames[0].shape[1], ouput_video_frames[0].shape[0]))
    for frame in ouput_video_frames:
        out.write(frame)
    out.release()

def read_video_generator(video_path, target_indices=None):
    """
    生成器函数：读取视频，只 yield 指定索引的帧。

    Args:
        video_path (str): 视频路径
        target_indices (list or None):
            - 如果是 list/set: 只读取这些索引的帧 (例如 [0, 100, 200])
            - 如果是 None: 表现为普通的生成器，按顺序读取所有帧

    Yields:
        tuple: (当前帧索引 frame_idx, 图像数据 frame, 原始帧率 fps)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    original_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 情况 A: 如果指定了要读哪些帧
    if target_indices is not None:
        # 1. 去重并排序 (关键优化！)
        # 视频解码器依赖前后帧，按顺序读取比乱序跳转(先读100再读50)快得多且稳定
        sorted_indices = sorted(list(set(target_indices)))

        for idx in sorted_indices:
            if idx >= total_frames:
                print(f"⚠️ 警告: 请求的帧索引 {idx} 超出视频范围 ({total_frames})，已跳过。")
                continue

            # 2. 跳转到指定位置
            # 这是一个 Seek 操作，比从头读要快，但如果太频繁也会有开销
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)

            ret, frame = cap.read()
            if ret:
                # 成功读取，暂停函数并返回数据
                yield idx, frame, original_fps
            else:
                print(f"❌ 错误: 无法读取第 {idx} 帧")

    # 情况 B: 如果没指定，就按顺序全读 (流式)
    else:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            yield frame_idx, frame, original_fps
            frame_idx += 1

    # 循环结束后释放资源
    cap.release()


def create_combined_image_layout(
    images_data: List[Tuple[str, Union[np.ndarray, List[np.ndarray]]]],
    title: str,
    layout: str = 'horizontal',
    target_height: int = 800,
    label_position: str = 'top',
    label_height: int = 60,
    label_width: int = 150,
    header_height: int = 100,
    add_column_headers: bool = False,
    column_headers: List[str] = None
) -> np.ndarray:
    """
    通用的图片组合拼接工具
    
    功能：
    - 自动调整图片到统一高度
    - 添加标签（相机名称等）
    - 支持水平/网格布局
    - 添加总标题头部
    - 支持多图拼接（每个条目可以是单张图或多张图的列表）
    
    Args:
        images_data: 图片数据列表，格式为 [(label, image_or_images), ...]
                    - label: 标签文本（如相机名称）
                    - image_or_images: 单张图片（np.ndarray）或图片列表（List[np.ndarray]）
                                      如果是None，会创建占位图
        title: 总标题文本
        layout: 布局方式
                - 'horizontal': 水平拼接（所有图片排成一行）
                - 'grid': 网格布局（每个条目占一行）
        target_height: 统一的图片高度（像素）
        label_position: 标签位置
                       - 'top': 标签在图片上方
                       - 'left': 标签在图片左侧（仅 grid 布局）
                       - 'bottom': 标签在图片下方
        label_height: 标签高度（当 label_position='top'/'bottom' 时）
        label_width: 标签宽度（当 label_position='left' 时）
        header_height: 总标题头部高度
        add_column_headers: 是否添加列标题（仅 grid 布局）
        column_headers: 列标题列表（与每行的图片数对应）
        
    Returns:
        combined_image: 拼接后的图像
        
    Example:
        # 示例1: 水平拼接多个相机的检测图
        images_data = [
            ('camera1', img1),
            ('camera2', img2),
            ('camera3', img3)
        ]
        combined = create_combined_image_layout(
            images_data, 
            title="Multi-Camera Detection",
            layout='horizontal'
        )
        
        # 示例2: 网格布局，每个相机多张图
        images_data = [
            ('camera1', [coverage_img, keypoints_img, boundaries_img]),
            ('camera2', [coverage_img, keypoints_img, boundaries_img])
        ]
        combined = create_combined_image_layout(
            images_data,
            title="MCS System Overview",
            layout='grid',
            label_position='left',
            add_column_headers=True,
            column_headers=['Coverage', 'Keypoints', 'Boundaries']
        )
    """
    if not images_data:
        raise ValueError("images_data 不能为空")
    
    # 处理数据：统一格式为 [(label, [images])]
    processed_data = []
    for label, img_or_imgs in images_data:
        if img_or_imgs is None:
            processed_data.append((label, [None]))
        elif isinstance(img_or_imgs, list):
            processed_data.append((label, img_or_imgs))
        else:
            processed_data.append((label, [img_or_imgs]))
    
    # 根据布局生成拼接图
    if layout == 'horizontal':
        combined = _create_horizontal_layout(
            processed_data, target_height, label_position, 
            label_height, label_width
        )
    elif layout == 'grid':
        combined = _create_grid_layout(
            processed_data, target_height, label_position,
            label_height, label_width, add_column_headers, column_headers
        )
    else:
        raise ValueError(f"不支持的布局类型: {layout}")
    
    # 添加总标题头部
    final = _add_header(combined, title, header_height)
    
    return final


def _create_horizontal_layout(
    data: List[Tuple[str, List[np.ndarray]]],
    target_height: int,
    label_position: str,
    label_height: int,
    label_width: int
) -> np.ndarray:
    """水平布局：所有图片排成一行"""
    labeled_images = []
    
    for label, images in data:
        # 只取第一张图（水平布局每个条目只显示一张）
        img = images[0] if images else None
        
        # 调整尺寸
        if img is not None:
            h, w = img.shape[:2]
            scale = target_height / h
            new_w = int(w * scale)
            resized_img = cv2.resize(img, (new_w, target_height))
        else:
            # 占位图
            new_w = 400
            resized_img = np.ones((target_height, new_w, 3), dtype=np.uint8) * 200
            cv2.putText(resized_img, "No Image", (new_w//2 - 60, target_height//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
        
        # 添加标签
        labeled_img = _add_label_to_image(
            resized_img, label, label_position, label_height, label_width
        )
        labeled_images.append(labeled_img)
    
    # 水平拼接
    combined = np.hstack(labeled_images)
    return combined


def _create_grid_layout(
    data: List[Tuple[str, List[np.ndarray]]],
    target_height: int,
    label_position: str,
    label_height: int,
    label_width: int,
    add_column_headers: bool,
    column_headers: List[str]
) -> np.ndarray:
    """网格布局：每个条目占一行，行内可以有多张图"""
    rows = []
    
    for label, images in data:
        row_images = []
        
        # 处理该行的所有图片
        for img in images:
            if img is not None:
                h, w = img.shape[:2]
                scale = target_height / h
                new_w = int(w * scale)
                resized_img = cv2.resize(img, (new_w, target_height))
            else:
                # 占位图
                new_w = 400
                resized_img = np.ones((target_height, new_w, 3), dtype=np.uint8) * 200
                cv2.putText(resized_img, "No Image", (new_w//2 - 60, target_height//2),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            row_images.append(resized_img)
        
        # 如果需要左侧标签，创建标签图
        if label_position == 'left':
            label_img = np.ones((target_height, label_width, 3), dtype=np.uint8) * 240
            # 绘制标签文本（垂直居中）
            font_scale = 1.0
            thickness = 2
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                                   font_scale, thickness)
            text_x = (label_width - text_w) // 2
            text_y = (target_height + text_h) // 2
            cv2.putText(label_img, label, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (50, 50, 50), thickness)
            row_images.insert(0, label_img)
        
        # 拼接该行
        row_combined = np.hstack(row_images)
        rows.append(row_combined)
    
    # 垂直拼接所有行
    combined = np.vstack(rows)
    
    # 添加列标题（如果需要）
    if add_column_headers and column_headers:
        combined = _add_column_headers(combined, column_headers, label_width, 
                                       target_height, label_position == 'left')
    
    return combined


def _add_label_to_image(
    img: np.ndarray,
    label: str,
    position: str,
    label_height: int,
    label_width: int
) -> np.ndarray:
    """为单张图片添加标签"""
    h, w = img.shape[:2]
    
    if position == 'top':
        # 创建标签栏（顶部）
        label_bar = np.ones((label_height, w, 3), dtype=np.uint8) * 240
        # 绘制文本
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               font_scale, thickness)
        text_x = (w - text_w) // 2
        text_y = (label_height + text_h) // 2
        cv2.putText(label_bar, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (30, 30, 30), thickness)
        return np.vstack([label_bar, img])
    
    elif position == 'bottom':
        # 创建标签栏（底部）
        label_bar = np.ones((label_height, w, 3), dtype=np.uint8) * 240
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX,
                                               font_scale, thickness)
        text_x = (w - text_w) // 2
        text_y = (label_height + text_h) // 2
        cv2.putText(label_bar, label, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (30, 30, 30), thickness)
        return np.vstack([img, label_bar])
    
    else:
        return img


def _add_column_headers(
    img: np.ndarray,
    column_headers: List[str],
    label_width: int,
    target_height: int,
    has_left_label: bool
) -> np.ndarray:
    """在网格布局顶部添加列标题"""
    total_width = img.shape[1]
    header_height = 40
    header_bar = np.ones((header_height, total_width, 3), dtype=np.uint8) * 230
    
    # 计算每列的宽度（简单平均分配）
    if has_left_label:
        # 第一列是标签列
        num_cols = len(column_headers) + 1
        col_width = (total_width - label_width) // len(column_headers)
        x_offset = label_width
        
        # 标签列的标题
        cv2.putText(header_bar, "Camera", (label_width//2 - 30, header_height//2 + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
    else:
        col_width = total_width // len(column_headers)
        x_offset = 0
    
    # 绘制每列的标题
    for col_name in column_headers:
        text_x = x_offset + col_width // 2 - 40
        cv2.putText(header_bar, col_name, (text_x, header_height//2 + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 80), 2)
        x_offset += col_width
    
    return np.vstack([header_bar, img])


def _add_header(img: np.ndarray, title: str, header_height: int) -> np.ndarray:
    """添加总标题头部"""
    total_width = img.shape[1]
    header = np.ones((header_height, total_width, 3), dtype=np.uint8) * 250
    
    # 绘制标题
    (title_w, title_h), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)
    title_x = (total_width - title_w) // 2
    title_y = (header_height + title_h) // 2
    cv2.putText(header, title, (title_x, title_y),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (30, 30, 30), 3)
    
    # 拼接
    final = np.vstack([header, img])
    return final


def sample_frames_from_video(
    video_path: Union[str, Path],
    n: int = 10,
    output_dir: Union[str, Path] = None,
    seed: int = None
) -> List[Path]:
    """
    从视频中随机采样 n 帧并保存为图片。

    Args:
        video_path: 视频文件路径
        n: 采样帧数，默认 10
        output_dir: 输出目录，默认与视频文件同目录
        seed: 随机种子，用于复现结果；None 表示不固定

    Returns:
        List[Path]: 保存的图片路径列表
    """
    import random

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"视频文件不存在: {video_path}")

    if output_dir is None:
        output_dir = video_path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        raise ValueError(f"视频帧数无效: {total_frames}")

    actual_n = min(n, total_frames)
    rng = random.Random(seed)
    sampled_indices = sorted(rng.sample(range(total_frames), actual_n))

    stem = video_path.stem
    saved_paths = []

    for idx, frame, _ in read_video_generator(str(video_path), target_indices=sampled_indices):
        out_path = output_dir / f"{stem}_frame{idx:06d}.jpg"
        save_frame(frame, out_path)
        saved_paths.append(out_path)
        print(f"  已保存: {out_path.name}")

    return saved_paths


def sample_frames_synchronized(
    video_paths: List[Union[str, Path]],
    n: int = 5,
    output_dir: Union[str, Path] = None,
    seed: int = None
) -> List[List[Path]]:
    """
    对多个视频在相同的时间点同步采样 n 帧。

    采样逻辑：
    1. 读取每个视频的帧率和总帧数，计算各自时长（秒）。
    2. 取所有视频中最短的时长作为公共时间范围。
    3. 在公共时间范围内随机选取 n 个时间点（秒）。
    4. 对每个视频，将时间点换算为对应的帧索引并提取该帧。

    保存命名：{视频名}_sync{组号:02d}_t{时间点ms}ms.jpg

    Args:
        video_paths: 视频路径列表
        n: 同步采样的时刻数，默认 5
        output_dir: 输出目录；默认各自保存到视频所在目录
        seed: 随机种子

    Returns:
        按视频顺序的二维列表，每个子列表为该视频保存的 n 张图片路径
    """
    import random

    video_paths = [Path(p) for p in video_paths]
    for vp in video_paths:
        if not vp.exists():
            raise FileNotFoundError(f"视频文件不存在: {vp}")

    # 读取每个视频的元信息
    meta = []
    for vp in video_paths:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {vp}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        duration = total / fps if fps > 0 else 0
        meta.append({"fps": fps, "total": total, "duration": duration})
        print(f"  {vp.name}: {total} 帧, {fps:.2f} fps, 时长 {duration:.2f} s")

    # 公共时间范围（取最短视频时长）
    common_duration = min(m["duration"] for m in meta)
    print(f"\n公共时间范围: {common_duration:.2f} s，从中随机选取 {n} 个时刻\n")

    rng = random.Random(seed)
    # 随机选取 n 个时间点（秒），结果排序
    time_points = sorted(rng.uniform(0, common_duration) for _ in range(n))

    all_saved: List[List[Path]] = []

    for vp, m in zip(video_paths, meta):
        out_dir = Path(output_dir) if output_dir else vp.parent
        out_dir.mkdir(parents=True, exist_ok=True)

        # 将时间点换算为该视频的帧索引
        target_indices = [
            min(int(t * m["fps"]), m["total"] - 1) for t in time_points
        ]

        stem = vp.stem
        saved: List[Path] = []
        print(f"[{vp.name}]")

        for group_idx, (t, idx, (_, frame, _)) in enumerate(
            zip(time_points, target_indices,
                read_video_generator(str(vp), target_indices=target_indices))
        ):
            t_ms = int(t * 1000)
            out_path = out_dir / f"{stem}_sync{group_idx + 1:02d}_t{t_ms}ms.jpg"
            save_frame(frame, out_path)
            saved.append(out_path)
            print(f"  组{group_idx + 1:02d} t={t:.3f}s → 帧{idx:06d}  已保存: {out_path.name}")

        all_saved.append(saved)
        print()

    return all_saved


if __name__ == "__main__":
    video_dir = Path("D:/视频/cumtb/raw")
    video_files = sorted(video_dir.glob("*.mp4"))

    if not video_files:
        print(f"未找到视频文件: {video_dir}")
    else:
        print(f"找到 {len(video_files)} 个视频，开始同步采样...\n")
        try:
            sample_frames_synchronized(video_files, n=5)
        except Exception as e:
            print(f"错误: {e}")

