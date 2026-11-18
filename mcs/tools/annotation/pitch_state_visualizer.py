"""
球场状态重建可视化
读取标注数据 + 相机单应矩阵，将像素坐标映射到球场坐标，
在球场俯视图上绘制所有追踪目标（球员和球）。
采样帧之间做线性插值，输出 30fps 平滑视频。
"""

import sys
import json
import bisect
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import supervision as sv
from soccer import QingYuanConfiguration
from pitch_annotation_tool import draw_pitch

class EventDetector:
    def __init__(self, fps):
        self.fps = fps
        self.current_possessor = None
        self.last_possessor = None
        self.events = []
        
        self.GOAL_1 = np.array([4350.0, 0.0])
        self.GOAL_2 = np.array([4350.0, 3600.0])
        self.POSSESSION_THRESHOLD = 250.0
        self.SHOT_THRESHOLD = 300.0
        
    def update(self, frame_idx, objects):
        b_coord = objects.get("ball")
        if b_coord is None:
            return
            
        b_coord = np.array(b_coord)
        
        closest_player = None
        min_dist = float('inf')
        
        for key, pos in objects.items():
            if key == "ball" or pos is None:
                continue
            
            p_coord = np.array(pos)
            dist = np.linalg.norm(p_coord - b_coord)
            if dist < min_dist:
                min_dist = dist
                closest_player = key
                
        # Possession logic
        if min_dist <= self.POSSESSION_THRESHOLD:
            if self.current_possessor != closest_player:
                if self.current_possessor is not None:
                    team_curr = self.current_possessor.split('_')[0]
                    team_new = closest_player.split('_')[0]
                    
                    if team_curr == team_new:
                        self.events.append({
                            "frame": frame_idx,
                            "type": "Pass",
                            "text": f"PASS: {self.current_possessor} -> {closest_player}"
                        })
                    else:
                        self.events.append({
                            "frame": frame_idx,
                            "type": "Tackle",
                            "text": f"TACKLE: {closest_player} dispossessed {self.current_possessor}"
                        })
                self.current_possessor = closest_player
                self.last_possessor = closest_player
                
        # Shot logic
        dist_g1 = np.linalg.norm(b_coord - self.GOAL_1)
        dist_g2 = np.linalg.norm(b_coord - self.GOAL_2)
        
        if dist_g1 < self.SHOT_THRESHOLD or dist_g2 < self.SHOT_THRESHOLD:
            is_new_shot = True
            if len(self.events) > 0 and self.events[-1]["type"] == "Shot":
                if frame_idx - self.events[-1]["frame"] < self.fps * 2:
                    is_new_shot = False
                    
            if is_new_shot and self.last_possessor is not None:
                self.events.append({
                    "frame": frame_idx,
                    "type": "Shot",
                    "text": f"SHOT: {self.last_possessor}"
                })
                self.current_possessor = None
                
    def get_recent_events(self, current_frame, duration_frames):
        recent = []
        for e in reversed(self.events):
            if current_frame - e["frame"] <= duration_frames:
                if current_frame >= e["frame"]:
                    recent.append(e)
            else:
                break
        return recent


# ==================== 硬编码配置 ====================
ANNOTATIONS_PATH = r"D:/intern/teenager_football/备份/store/data/annotations.json"
CAMERA_INFO_PATH = r"D:/intern/teenager_football/备份/store/qingyuantest_mcs/cameratest/camera_info.json"
OUTPUT_VIDEO_PATH = r"D:/intern/teenager_football/备份/output/pitch_reconstruction.mp4"

PITCH_CONFIG = QingYuanConfiguration()
PITCH_SCALE = 0.2
PITCH_PADDING = 50
OUTPUT_FPS = 30
# ====================================================

COLOR_T1 = (60, 60, 255)
COLOR_T1_BORDER = (30, 30, 180)
COLOR_T2 = (255, 140, 40)
COLOR_T2_BORDER = (180, 100, 20)
COLOR_BALL = (255, 255, 255)
COLOR_BALL_BORDER = (180, 180, 180)
COLOR_TEST = (0, 220, 0)
COLOR_TEST_BORDER = (0, 160, 0)

PLAYER_RADIUS = 12
BALL_RADIUS = 6
FONT = cv2.FONT_HERSHEY_SIMPLEX


def pixel_to_pitch(px, py, H):
    pt = np.array([px, py, 1.0], dtype=np.float64)
    result = H @ pt
    return result[0] / result[2], result[1] / result[2]


def build_keyframes(annotations, H):
    meta = annotations["meta"]
    t1_count = meta["t1_count"]
    t2_count = meta["t2_count"]

    test_count = meta.get("test_count", 0)

    obj_keys = [f"t1_o{i}" for i in range(1, t1_count + 1)]
    obj_keys += [f"t2_o{i}" for i in range(1, t2_count + 1)]
    obj_keys.append("ball")
    obj_keys += [f"test_o{i}" for i in range(1, test_count + 1)]

    keyframes = {}
    for frame_str, frame_data in annotations["frames"].items():
        frame_idx = int(frame_str)
        kf = {}

        for team in ["t1", "t2", "test"]:
            if team not in frame_data:
                continue
            for obj_id, coord in frame_data[team].items():
                key = f"{team}_{obj_id}"
                if coord is not None:
                    kf[key] = pixel_to_pitch(coord["x"], coord["y"], H)
                else:
                    kf[key] = None

        ball = frame_data.get("ball")
        if ball is not None:
            kf["ball"] = pixel_to_pitch(ball["x"], ball["y"], H)
        else:
            kf["ball"] = None

        keyframes[frame_idx] = kf

    return keyframes, obj_keys


def interpolate_frame(prev_kf, next_kf, t, obj_keys):
    result = {}
    for key in obj_keys:
        p = prev_kf.get(key)
        n = next_kf.get(key)

        if p is not None and n is not None:
            result[key] = (p[0] + t * (n[0] - p[0]),
                           p[1] + t * (n[1] - p[1]))
        elif p is not None:
            result[key] = p if t < 0.5 else None
        elif n is not None:
            result[key] = n if t >= 0.5 else None
        else:
            result[key] = None

    return result


def get_frame_objects(keyframes, sorted_indices, frame_idx, obj_keys):
    if frame_idx in keyframes:
        return keyframes[frame_idx]

    pos = bisect.bisect_right(sorted_indices, frame_idx)
    if pos == 0:
        return keyframes[sorted_indices[0]]
    if pos >= len(sorted_indices):
        return keyframes[sorted_indices[-1]]

    prev_idx = sorted_indices[pos - 1]
    next_idx = sorted_indices[pos]
    t = (frame_idx - prev_idx) / (next_idx - prev_idx)
    return interpolate_frame(keyframes[prev_idx], keyframes[next_idx], t, obj_keys)


def draw_objects(pitch_bg, objects, scale, padding):
    frame = pitch_bg.copy()

    for key, pos in objects.items():
        if pos is None:
            continue

        dx = int(pos[0] * scale) + padding
        dy = int(pos[1] * scale) + padding

        if key.startswith("t1"):
            color, border, radius = COLOR_T1, COLOR_T1_BORDER, PLAYER_RADIUS
            num = key.split("_")[1][1:]
        elif key.startswith("t2"):
            color, border, radius = COLOR_T2, COLOR_T2_BORDER, PLAYER_RADIUS
            num = key.split("_")[1][1:]
        elif key.startswith("test"):
            color, border, radius = COLOR_TEST, COLOR_TEST_BORDER, PLAYER_RADIUS
            num = key.split("_")[1][1:]
        else:
            color, border, radius = COLOR_BALL, COLOR_BALL_BORDER, BALL_RADIUS
            num = None

        cv2.circle(frame, (dx, dy), radius, color, -1, cv2.LINE_AA)
        cv2.circle(frame, (dx, dy), radius, border, 2, cv2.LINE_AA)

        if num is not None:
            (tw, th), _ = cv2.getTextSize(num, FONT, 0.45, 1)
            cv2.putText(frame, num,
                        (dx - tw // 2, dy + th // 2),
                        FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


def draw_hud(frame, frame_idx, fps, total_frames, has_test=False):
    h, w = frame.shape[:2]

    time_sec = frame_idx / fps
    minutes = int(time_sec // 60)
    seconds = time_sec % 60
    time_str = f"{minutes:02d}:{seconds:04.1f}"
    frame_str = f"F{frame_idx}"

    cv2.rectangle(frame, (4, 4), (170, 56), (0, 0, 0), -1)
    cv2.rectangle(frame, (4, 4), (170, 56), (80, 80, 80), 1)
    cv2.putText(frame, time_str, (10, 30), FONT, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, frame_str, (10, 50), FONT, 0.4,
                (160, 160, 160), 1, cv2.LINE_AA)

    lx, ly = 10, h - 18
    cv2.circle(frame, (lx, ly), 6, COLOR_T1, -1, cv2.LINE_AA)
    cv2.putText(frame, "T1", (lx + 10, ly + 5), FONT, 0.38,
                (220, 220, 220), 1, cv2.LINE_AA)
    cv2.circle(frame, (lx + 50, ly), 6, COLOR_T2, -1, cv2.LINE_AA)
    cv2.putText(frame, "T2", (lx + 60, ly + 5), FONT, 0.38,
                (220, 220, 220), 1, cv2.LINE_AA)
    cv2.circle(frame, (lx + 100, ly), 4, COLOR_BALL, -1, cv2.LINE_AA)
    cv2.putText(frame, "Ball", (lx + 108, ly + 5), FONT, 0.38,
                (220, 220, 220), 1, cv2.LINE_AA)
    if has_test:
        cv2.circle(frame, (lx + 160, ly), 6, COLOR_TEST, -1, cv2.LINE_AA)
        cv2.putText(frame, "Test", (lx + 170, ly + 5), FONT, 0.38,
                    (220, 220, 220), 1, cv2.LINE_AA)

    return frame


def draw_events(frame, events):
    if not events:
        return frame
        
    h, w = frame.shape[:2]
    
    # Display the most recent event at the top center
    event = events[0] # The first one in the reversed list is the most recent
    text = event["text"]
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x = (w - tw) // 2
    y = 80
    
    # Background box
    padding = 15
    cv2.rectangle(frame, (x - padding, y - th - padding), (x + tw + padding, y + padding), (0, 0, 0), -1)
    cv2.rectangle(frame, (x - padding, y - th - padding), (x + tw + padding, y + padding), (0, 255, 255), 2)
    
    # Text
    cv2.putText(frame, text, (x, y), font, font_scale, (0, 255, 255), thickness, cv2.LINE_AA)
    
    return frame

def main():
    with open(ANNOTATIONS_PATH, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    with open(CAMERA_INFO_PATH, "r", encoding="utf-8") as f:
        camera_info = json.load(f)

    H = np.array(camera_info["homography_matrix"], dtype=np.float64)
    meta = annotations["meta"]
    total_frames = meta["total_frames"]
    fps = meta["fps"]

    print(f"标注关键帧: {len(annotations['frames'])} 个")
    print(f"视频总帧数: {total_frames}  FPS: {fps:.1f}")

    keyframes, obj_keys = build_keyframes(annotations, H)
    sorted_indices = sorted(keyframes.keys())

    if not sorted_indices:
        print("无关键帧数据!")
        return

    has_test = any(k.startswith("test") for k in obj_keys)

    print(f"关键帧范围: {sorted_indices[0]} ~ {sorted_indices[-1]}")
    print(f"追踪目标: {obj_keys}")

    pitch_bg = draw_pitch(PITCH_CONFIG, scale=PITCH_SCALE, padding=PITCH_PADDING)

    Path(OUTPUT_VIDEO_PATH).parent.mkdir(parents=True, exist_ok=True)
    h, w = pitch_bg.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, OUTPUT_FPS, (w, h))

    if not writer.isOpened():
        print(f"无法创建视频文件: {OUTPUT_VIDEO_PATH}")
        return

    start = sorted_indices[0]
    end = sorted_indices[-1]
    total_output = end - start + 1

    print(f"输出: {OUTPUT_VIDEO_PATH}  ({w}x{h} @ {OUTPUT_FPS}fps)")
    print(f"生成 {total_output} 帧 ({total_output / OUTPUT_FPS:.1f} 秒)...\n")

    detector = EventDetector(fps=OUTPUT_FPS)

    for frame_idx in range(start, end + 1):
        objects = get_frame_objects(keyframes, sorted_indices, frame_idx, obj_keys)
        
        detector.update(frame_idx, objects)
        recent_events = detector.get_recent_events(frame_idx, OUTPUT_FPS * 2) # 2 seconds
        
        frame = draw_objects(pitch_bg, objects, PITCH_SCALE, PITCH_PADDING)
        draw_hud(frame, frame_idx, fps, total_frames, has_test)
        frame = draw_events(frame, recent_events)
        
        writer.write(frame)

        done = frame_idx - start + 1
        if done % 300 == 0 or done == total_output:
            pct = 100 * done / total_output
            print(f"  {done}/{total_output}  ({pct:.0f}%)")

    writer.release()
    print(f"\n完成! 已保存: {OUTPUT_VIDEO_PATH}")


if __name__ == "__main__":
    main()
