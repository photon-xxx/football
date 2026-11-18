from pathlib import Path
import os

# 获取项目根目录 (假设 config.py 位于项目根目录)
PROJECT_ROOT = Path(__file__).parent.absolute()

# 目录配置
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
OCR_INPUT_DIR = PROJECT_ROOT / "ocr_input"
STORE_DIR = PROJECT_ROOT / "store" / "botsort"
BOTSORT_ROOT = PROJECT_ROOT / "BoT-SORT"

# 确保必要的目录存在
MODELS_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
OCR_INPUT_DIR.mkdir(exist_ok=True)
STORE_DIR.mkdir(parents=True, exist_ok=True)

# 模型路径配置
YOLOV8_WEIGHTS = MODELS_DIR / "yolov8n.pt"

# BoT-SORT 相关配置
YOLOX_WEIGHTS = BOTSORT_ROOT / "pretrained" / "bytetrack_x_mot17.pth.tar"
YOLOX_EXP_FILE = BOTSORT_ROOT / "yolox" / "exps" / "example" / "mot" / "yolox_x_mix_det.py"

FAST_REID_CONFIG = BOTSORT_ROOT / "fast_reid" / "configs" / "MOT17" / "sbs_S50.yml"
FAST_REID_WEIGHTS = BOTSORT_ROOT / "pretrained" / "mot17_sbs_S50.pth"

# 默认配置
DEFAULT_VIDEO_OUTPUT = OUTPUT_DIR / "tracked_video.mp4"
DEFAULT_CALIBRATION_OUTPUT = "calibration.json"

def get_yolox_exp_file():
    """Helper to get string path for YOLOX exp file (needed by YOLOX internals)"""
    return str(YOLOX_EXP_FILE)

def check_paths():
    """检查关键路径是否存在"""
    paths = [
        ("BoT-SORT Root", BOTSORT_ROOT),
        ("Models Dir", MODELS_DIR),
        ("YOLOv8 Weights", YOLOV8_WEIGHTS),
    ]
    
    missing = []
    for name, p in paths:
        if not p.exists():
            missing.append(f"{name}: {p}")
    
    if missing:
        print("Warning: Some configured paths are missing:")
        for m in missing:
            print(f"  - {m}")
    else:
        print("All key paths verified.")



