"""
球员/球 像素坐标标注工具 (v2)
每个球员标注两个点：foot (球场投影点) + head (头部位置)。
区分门将和球员，支持可配置门将数量。
标注下一帧时，显示当前目标在上一采样帧的标注位置供参考。
支持懒加载、流式写入、断点续标、回退、回滚。
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path

# ==================== 硬编码配置 ====================
VIDEO_PATH = r"D:/视频/wuchuan/wuchaun1_high_1m.mp4"
OUTPUT_PATH = r"D:/intern/teenager_football/MCS-main/store/data/wuchuan1_high_1m_annotations.json"
SAMPLING_INTERVAL = 15
DISPLAY_SCALE = 1.0
TEST_COUNT = 1
ROLLBACK_FROM = -1
# ====================================================

COLOR_T1 = (255, 255, 0)
COLOR_T2 = (0, 165, 255)
COLOR_BALL = (255, 255, 255)
COLOR_TEST = (0, 255, 0)
COLOR_NULL = (128, 128, 128)
COLOR_PREFILL = (160, 160, 160)
COLOR_STATUS_BG = (40, 40, 40)
COLOR_STATUS_TEXT = (255, 255, 255)
COLOR_CROSSHAIR = (0, 255, 0)
COLOR_HELP_TEXT = (180, 180, 180)

MARKER_RADIUS = 4
MARKER_DOT = 1
DIAMOND_SIZE = 4
ANNOTATION_ALPHA = 0.45
PREV_REF_ALPHA = 0.35
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _dim_color(color, factor=0.45):
    return tuple(int(c * factor) for c in color)


def _draw_diamond(img, center, size, color, thickness):
    cx, cy = center
    pts = np.array([
        [cx, cy - size],
        [cx + size, cy],
        [cx, cy + size],
        [cx - size, cy],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], True, color, thickness, cv2.LINE_AA)


def _draw_x_mark(img, center, size, color, thickness=2):
    cx, cy = center
    cv2.line(img, (cx - size, cy - size), (cx + size, cy + size),
             color, thickness, cv2.LINE_AA)
    cv2.line(img, (cx - size, cy + size), (cx + size, cy - size),
             color, thickness, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# AnnotationState
# ---------------------------------------------------------------------------

class AnnotationState:

    def __init__(self, t1_count, t2_count, t1_gk_count=1, t2_gk_count=1,
                 test_count=0):
        self.t1_count = t1_count
        self.t2_count = t2_count
        self.t1_gk_count = t1_gk_count
        self.t2_gk_count = t2_gk_count
        self.test_count = test_count

        self.targets = []
        self._build_targets()
        self.total = len(self.targets)

        self.annotations = []
        self.prefill_count = 0
        self.mouse_pos = (0, 0)

    def _build_targets(self):
        for i in range(1, self.t1_gk_count + 1):
            self.targets.append((f"t1_gk{i}", "foot"))
            self.targets.append((f"t1_gk{i}", "head"))
        for i in range(1, self.t1_count - self.t1_gk_count + 1):
            self.targets.append((f"t1_p{i}", "foot"))
            self.targets.append((f"t1_p{i}", "head"))

        for i in range(1, self.t2_gk_count + 1):
            self.targets.append((f"t2_gk{i}", "foot"))
            self.targets.append((f"t2_gk{i}", "head"))
        for i in range(1, self.t2_count - self.t2_gk_count + 1):
            self.targets.append((f"t2_p{i}", "foot"))
            self.targets.append((f"t2_p{i}", "head"))

        self.targets.append(("ball", "point"))

        for i in range(1, self.test_count + 1):
            self.targets.append((f"test_o{i}", "point"))

    def reset(self, prefill=None):
        if prefill is not None:
            self.annotations = list(prefill)
            self.prefill_count = len(prefill)
        else:
            self.annotations = []
            self.prefill_count = 0
        self.mouse_pos = (0, 0)

    @property
    def current_index(self):
        return len(self.annotations)

    @property
    def is_complete(self):
        return self.current_index >= self.total

    def current_target(self):
        if self.current_index >= self.total:
            return None, None
        return self.targets[self.current_index]

    def current_target_name(self):
        key, part = self.current_target()
        if key is None:
            return "DONE"
        display = key.replace("_", "-")
        if part == "point":
            return display
        return f"{display} [{part}]"

    def current_target_key(self):
        key, _ = self.current_target()
        return key

    def add(self, coord):
        if not self.is_complete:
            self.annotations.append(coord)

    def add_null(self):
        if self.is_complete:
            return
        key, part = self.targets[self.current_index]
        if part == "foot":
            self.annotations.append(None)
            self.annotations.append(None)
        else:
            self.annotations.append(None)

    def undo(self):
        if len(self.annotations) <= self.prefill_count:
            return
        self.annotations.pop()

    def target_color(self, index):
        if index >= len(self.targets):
            return COLOR_NULL
        key, _ = self.targets[index]
        if key == "ball":
            return COLOR_BALL
        if key.startswith("test_"):
            return COLOR_TEST
        if key.startswith("t1_"):
            return COLOR_T1
        if key.startswith("t2_"):
            return COLOR_T2
        return COLOR_NULL

    def target_display_label(self, index):
        if index >= len(self.targets):
            return "?"
        key, _ = self.targets[index]
        return key.replace("_", "-")

    def is_group_done(self, group_prefix):
        """Check if all targets in a group (e.g. 't1', 't2') are annotated."""
        last_idx = -1
        for i, (key, _) in enumerate(self.targets):
            if key.startswith(group_prefix + "_") or key == group_prefix:
                last_idx = i
        return last_idx >= 0 and self.current_index > last_idx

    def target_group(self, index):
        if index >= len(self.targets):
            return None
        key, _ = self.targets[index]
        if key.startswith("t1_"):
            return "t1"
        if key.startswith("t2_"):
            return "t2"
        if key == "ball":
            return "ball"
        if key.startswith("test_"):
            return "test"
        return None

    def to_dict(self):
        result = {
            "t1": {"goalkeepers": {}, "players": {}},
            "t2": {"goalkeepers": {}, "players": {}},
        }

        for i, (key, part) in enumerate(self.targets):
            ann = self.annotations[i] if i < len(self.annotations) else None

            if key == "ball":
                result["ball"] = ann
                continue
            if key.startswith("test_"):
                if "test" not in result:
                    result["test"] = {}
                obj_id = key.split("_", 1)[1]
                result["test"][obj_id] = ann
                continue

            team = key[:2]
            obj_id = key[3:]
            is_gk = obj_id.startswith("gk")
            sub_group = "goalkeepers" if is_gk else "players"

            if obj_id not in result[team][sub_group]:
                result[team][sub_group][obj_id] = {"foot": None, "head": None}
            result[team][sub_group][obj_id][part] = ann

        return result


# ---------------------------------------------------------------------------
# Frame data helpers
# ---------------------------------------------------------------------------

def _lookup_target_in_frame(frame_data, key):
    """Look up a target's annotation in the nested frame dict.
    For players returns {"foot": ..., "head": ...}, for ball/test returns coord or None.
    """
    if frame_data is None:
        return None
    if key == "ball":
        return frame_data.get("ball")
    if key.startswith("test_"):
        test = frame_data.get("test")
        if not isinstance(test, dict):
            return None
        return test.get(key.split("_", 1)[1])

    team = key[:2]
    obj_id = key[3:]
    is_gk = obj_id.startswith("gk")
    team_data = frame_data.get(team, {})

    if isinstance(team_data, dict):
        sub_group = "goalkeepers" if is_gk else "players"
        sub = team_data.get(sub_group, {})
        if isinstance(sub, dict):
            return sub.get(obj_id)
    return None


def save_json(data, path):
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def load_or_create_data(output_path, video_path, interval, fps, total_frames):
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        print(f"[续标] 已加载: {output_path}  (已有 {len(data.get('frames', {}))} 帧)")
        return data

    return {
        "meta": {
            "video": video_path,
            "sampling_interval": interval,
            "t1_count": None,
            "t2_count": None,
            "t1_gk_count": None,
            "t2_gk_count": None,
            "test_count": 0,
            "fps": fps,
            "total_frames": total_frames,
            "game_id": None,
            "period_id": None,
            "team1_name": None,
            "team2_name": None,
            "team1_ground": "home",
            "team2_ground": "away",
            "pitch_config": None,
        },
        "frames": {},
    }


def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total


def is_frame_complete(frame_data, test_count):
    if frame_data is None:
        return False
    if "t1" not in frame_data or "ball" not in frame_data:
        return False
    if test_count <= 0:
        return True
    test = frame_data.get("test", {})
    return len(test) >= test_count


def build_prefill(frame_data, targets):
    """Flatten existing frame data back into annotations list order (for test mode)."""
    prefill = []
    for key, part in targets:
        if key.startswith("test_"):
            break
        val = _lookup_target_in_frame(frame_data, key)
        if val is None:
            prefill.append(None)
        elif part == "point":
            prefill.append(val)
        else:
            prefill.append(val.get(part) if isinstance(val, dict) else None)
    return prefill


def get_prev_frame_data(data, all_indices, cursor):
    if cursor <= 0:
        return None
    prev_idx = all_indices[cursor - 1]
    return data["frames"].get(str(prev_idx))


def undo_frame(data, frame_idx, test_count):
    key = str(frame_idx)
    fd = data["frames"].get(key)
    if fd is None:
        return False
    if test_count > 0 and "test" in fd:
        del fd["test"]
    else:
        del data["frames"][key]
    return True


def read_frame(cap, video_path, frame_idx, seq_pos):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if ret:
        return cap, frame, seq_pos

    if seq_pos < 0 or seq_pos > frame_idx:
        cap.release()
        cap = cv2.VideoCapture(video_path)
        seq_pos = 0

    skip = frame_idx - seq_pos
    print(f"  [顺序读取] 从位置 {seq_pos} 前进 {skip} 帧到帧 {frame_idx} ...")
    for _ in range(skip):
        ret, _ = cap.read()
        if not ret:
            return cap, None, seq_pos
    ret, frame = cap.read()
    if ret:
        return cap, frame, frame_idx + 1
    return cap, None, seq_pos


def do_rollback(data, rollback_from, test_count):
    count = 0
    for key in list(data["frames"].keys()):
        if int(key) >= rollback_from:
            fd = data["frames"][key]
            if test_count > 0 and "test" in fd:
                del fd["test"]
                count += 1
            elif test_count <= 0:
                del data["frames"][key]
                count += 1
    return count


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_overlay(frame_display, state, frame_idx, frame_progress, total_sampled,
                 scale, prev_frame_data=None):
    display = frame_display.copy()
    h, w = display.shape[:2]

    bar_h = 48
    cv2.rectangle(display, (0, 0), (w, bar_h), COLOR_STATUS_BG, -1)

    target_name = state.current_target_name()
    status = f"Frame {frame_idx}  |  {frame_progress}/{total_sampled}  |  NOW: {target_name}"
    if state.is_complete:
        status += "  |  [Space/Enter] confirm"
    cv2.putText(display, status, (10, 32), FONT, 0.6, COLOR_STATUS_TEXT, 2,
                cv2.LINE_AA)

    help_txt = "[LClick]mark [RClick/n]null [z]undo [b]back [s]save [q]quit"
    cv2.putText(display, help_txt, (10, h - 12), FONT, 0.45, COLOR_HELP_TEXT,
                1, cv2.LINE_AA)

    # --- Previous frame reference for current target (semi-transparent) ---
    cur_key = state.current_target_key()
    if prev_frame_data is not None and cur_key is not None:
        prev_val = _lookup_target_in_frame(prev_frame_data, cur_key)
        base_color = state.target_color(state.current_index)
        dim = _dim_color(base_color)

        overlay_prev = display.copy()
        if isinstance(prev_val, dict) and "foot" in prev_val:
            foot = prev_val.get("foot")
            head = prev_val.get("head")
            if foot is not None:
                fx, fy = int(foot["x"] * scale), int(foot["y"] * scale)
                _draw_x_mark(overlay_prev, (fx, fy), 5, dim)
                cv2.putText(overlay_prev, "prev-foot", (fx + 7, fy - 3),
                            FONT, 0.3, dim, 1, cv2.LINE_AA)
            if head is not None:
                hx, hy = int(head["x"] * scale), int(head["y"] * scale)
                _draw_x_mark(overlay_prev, (hx, hy), 4, dim)
                cv2.putText(overlay_prev, "prev-head", (hx + 7, hy - 3),
                            FONT, 0.3, dim, 1, cv2.LINE_AA)
            if foot is not None and head is not None:
                cv2.line(overlay_prev, (fx, fy), (hx, hy), dim, 1,
                         cv2.LINE_AA)
        elif prev_val is not None and isinstance(prev_val, dict) and "x" in prev_val:
            px, py = int(prev_val["x"] * scale), int(prev_val["y"] * scale)
            _draw_x_mark(overlay_prev, (px, py), 5, dim)
            cv2.putText(overlay_prev, "prev", (px + 7, py - 3),
                        FONT, 0.3, dim, 1, cv2.LINE_AA)
        cv2.addWeighted(overlay_prev, PREV_REF_ALPHA, display,
                        1 - PREV_REF_ALPHA, 0, display)

    # --- Draw completed annotations (semi-transparent, hide finished teams) ---
    overlay_ann = display.copy()
    null_count = 0
    i = 0
    while i < len(state.annotations):
        if i >= len(state.targets):
            break
        key, part = state.targets[i]
        ann = state.annotations[i]
        is_prefill = i < state.prefill_count

        grp = state.target_group(i)
        if grp in ("t1", "t2") and state.is_group_done(grp):
            i += 2 if part == "foot" else 1
            continue

        color = state.target_color(i)
        draw_color = COLOR_PREFILL if is_prefill else color
        thickness = 1
        label = state.target_display_label(i)

        if part == "foot":
            head_ann = (state.annotations[i + 1]
                        if i + 1 < len(state.annotations) else None)

            if ann is None and head_ann is None:
                null_y = bar_h + 20 + null_count * 18
                nc = COLOR_PREFILL if is_prefill else COLOR_NULL
                cv2.putText(display, f"{label}: NULL", (10, null_y), FONT,
                            0.35, nc, 1, cv2.LINE_AA)
                null_count += 1
                i += 2
                continue

            if ann is not None:
                fx, fy = int(ann["x"] * scale), int(ann["y"] * scale)
                cv2.circle(overlay_ann, (fx, fy), MARKER_RADIUS, draw_color,
                           thickness, cv2.LINE_AA)
                cv2.circle(overlay_ann, (fx, fy), MARKER_DOT, draw_color, -1,
                           cv2.LINE_AA)
                cv2.putText(overlay_ann, label, (fx + 8, fy - 5), FONT, 0.32,
                            draw_color, 1, cv2.LINE_AA)

            if head_ann is not None:
                hx, hy = int(head_ann["x"] * scale), int(head_ann["y"] * scale)
                _draw_diamond(overlay_ann, (hx, hy), DIAMOND_SIZE, draw_color,
                              thickness)
                if ann is not None:
                    cv2.line(overlay_ann, (fx, fy), (hx, hy), draw_color, 1,
                             cv2.LINE_AA)

            i += 2

        elif part == "point":
            if ann is None:
                null_y = bar_h + 20 + null_count * 18
                nc = COLOR_PREFILL if is_prefill else COLOR_NULL
                cv2.putText(display, f"{label}: NULL", (10, null_y), FONT,
                            0.35, nc, 1, cv2.LINE_AA)
                null_count += 1
            else:
                dx, dy = int(ann["x"] * scale), int(ann["y"] * scale)
                cv2.circle(overlay_ann, (dx, dy), MARKER_RADIUS, draw_color,
                           thickness, cv2.LINE_AA)
                cv2.circle(overlay_ann, (dx, dy), MARKER_DOT, draw_color, -1,
                           cv2.LINE_AA)
                cv2.putText(overlay_ann, label, (dx + 8, dy - 5), FONT, 0.32,
                            draw_color, 1, cv2.LINE_AA)
            i += 1

        else:
            i += 1

    cv2.addWeighted(overlay_ann, ANNOTATION_ALPHA, display,
                    1 - ANNOTATION_ALPHA, 0, display)

    # --- Crosshair ---
    mx, my = state.mouse_pos
    cv2.line(display, (mx - 20, my), (mx + 20, my), COLOR_CROSSHAIR, 1,
             cv2.LINE_AA)
    cv2.line(display, (mx, my - 20), (mx, my + 20), COLOR_CROSSHAIR, 1,
             cv2.LINE_AA)

    return display


# ---------------------------------------------------------------------------
# Mouse callback
# ---------------------------------------------------------------------------

def mouse_callback(event, x, y, flags, param):
    state, scale = param
    state.mouse_pos = (x, y)
    if event == cv2.EVENT_LBUTTONDOWN and not state.is_complete:
        orig_x = int(x / scale) if scale != 1.0 else x
        orig_y = int(y / scale) if scale != 1.0 else y
        state.add({"x": orig_x, "y": orig_y})
    elif event == cv2.EVENT_RBUTTONDOWN and not state.is_complete:
        state.add_null()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    video_path = VIDEO_PATH
    output_path = OUTPUT_PATH
    interval = SAMPLING_INTERVAL
    scale = DISPLAY_SCALE
    test_count = TEST_COUNT
    rollback_from = ROLLBACK_FROM

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fps, total_frames = get_video_info(video_path)
    print(f"视频: {video_path}")
    print(f"FPS: {fps:.1f}  总帧数: {total_frames}")

    all_sample_indices = list(range(0, total_frames, interval))
    total_sampled = len(all_sample_indices)
    print(f"采样间隔: {interval}  采样帧数: {total_sampled}")

    data = load_or_create_data(output_path, video_path, interval, fps,
                               total_frames)

    if data["meta"]["t1_count"] is not None:
        t1_count = data["meta"]["t1_count"]
        t2_count = data["meta"]["t2_count"]
        t1_gk = data["meta"].get("t1_gk_count") or 1
        t2_gk = data["meta"].get("t2_gk_count") or 1
        print(f"从缓存读取: t1={t1_count}(gk={t1_gk}), t2={t2_count}(gk={t2_gk})")
    else:
        t1_count = int(input("请输入 t1 (队伍1) 总人数 (含门将): "))
        t2_count = int(input("请输入 t2 (队伍2) 总人数 (含门将): "))
        if t1_count <= 0 or t2_count <= 0:
            print("错误: 队伍人数必须大于 0")
            return
        t1_gk = int(input("t1 门将人数 (回车默认1): ").strip() or "1")
        t2_gk = int(input("t2 门将人数 (回车默认1): ").strip() or "1")
        if t1_gk > t1_count or t2_gk > t2_count:
            print("错误: 门将人数不能超过队伍总人数")
            return
        data["meta"]["t1_count"] = t1_count
        data["meta"]["t2_count"] = t2_count
        data["meta"]["t1_gk_count"] = t1_gk
        data["meta"]["t2_gk_count"] = t2_gk

    if data["meta"].get("game_id") is None:
        data["meta"]["game_id"] = (
            input("比赛标识 (game_id, 回车跳过): ").strip() or "match_001"
        )
        data["meta"]["period_id"] = int(
            input("半场编号 (1 或 2, 回车默认1): ").strip() or "1"
        )
        data["meta"]["team1_name"] = (
            input("队伍1名称 (回车默认 Team1): ").strip() or "Team1"
        )
        data["meta"]["team2_name"] = (
            input("队伍2名称 (回车默认 Team2): ").strip() or "Team2"
        )
        data["meta"]["team1_ground"] = "home"
        data["meta"]["team2_ground"] = "away"
        pitch_cfg = input(
            "球场配置类名 (如 WuChuanConfiguration, 回车默认 SoccerPitchConfiguration): "
        ).strip()
        data["meta"]["pitch_config"] = pitch_cfg or "SoccerPitchConfiguration"
        save_json(data, output_path)

    if test_count > 0:
        data["meta"]["test_count"] = test_count
        print(f"测试标注模式: test_count={test_count}")

    if rollback_from >= 0:
        rc = do_rollback(data, rollback_from, test_count)
        if rc > 0:
            save_json(data, output_path)
            print(f"[回滚] 已清除帧 >= {rollback_from} 的标注 ({rc} 帧)")

    state = AnnotationState(t1_count, t2_count, t1_gk, t2_gk, test_count)

    print(f"标注顺序 ({state.total} slots):")
    seen_keys = set()
    for key, part in state.targets:
        if key not in seen_keys:
            seen_keys.add(key)
            print(f"  {key.replace('_', '-')}", end="")
            if part != "point":
                print(" [foot+head]", end="")
            print()

    start_cursor = 0
    for i, idx in enumerate(all_sample_indices):
        fd = data["frames"].get(str(idx))
        if not is_frame_complete(fd, test_count):
            start_cursor = i
            break
    else:
        print("所有采样帧已标注完成！")
        return

    done_count = sum(
        1 for idx in all_sample_indices
        if is_frame_complete(data["frames"].get(str(idx)), test_count)
    )
    print(f"进度: {done_count}/{total_sampled}  "
          f"从帧 {all_sample_indices[start_cursor]} 继续\n")

    win_name = "Player Annotation Tool"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_name, mouse_callback, (state, scale))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    seq_pos = -1

    cursor = start_cursor
    while cursor < total_sampled:
        frame_idx = all_sample_indices[cursor]
        fd = data["frames"].get(str(frame_idx))

        if is_frame_complete(fd, test_count):
            cursor += 1
            continue

        cap, raw_frame, seq_pos = read_frame(cap, video_path, frame_idx,
                                             seq_pos)
        if raw_frame is None:
            print(f"  [跳过] 无法读取帧 {frame_idx}")
            cursor += 1
            continue

        frame = raw_frame.copy()

        if fd is not None and test_count > 0:
            prefill = build_prefill(fd, state.targets)
            state.reset(prefill=prefill)
        else:
            state.reset()

        done_count = sum(
            1 for idx in all_sample_indices
            if is_frame_complete(data["frames"].get(str(idx)), test_count)
        )
        frame_progress = done_count + 1

        if scale != 1.0:
            frame_display = cv2.resize(frame, None, fx=scale, fy=scale,
                                       interpolation=cv2.INTER_AREA)
        else:
            frame_display = frame.copy()

        prev_fd = get_prev_frame_data(data, all_sample_indices, cursor)
        action = "next"

        while True:
            display = draw_overlay(frame_display, state, frame_idx,
                                   frame_progress, total_sampled, scale,
                                   prev_frame_data=prev_fd)
            cv2.imshow(win_name, display)
            key = cv2.waitKey(30) & 0xFF

            if key == ord("q"):
                save_json(data, output_path)
                print(f"\n已保存并退出 ({len(data['frames'])} 帧)")
                cap.release()
                cv2.destroyAllWindows()
                return

            elif key == ord("s"):
                save_json(data, output_path)
                print(f"[保存] {len(data['frames'])} 帧已写入磁盘")

            elif key == ord("z") or key == 8:
                state.undo()

            elif key == ord("n"):
                if not state.is_complete:
                    state.add_null()

            elif key == ord("b"):
                if cursor > 0:
                    prev_frame_idx = all_sample_indices[cursor - 1]
                    if undo_frame(data, prev_frame_idx, test_count):
                        save_json(data, output_path)
                        print(f"  [回退] 撤销帧 {prev_frame_idx} 的标注")
                    action = "back"
                    break
                else:
                    print("  [回退] 已在第一帧，无法回退")

            elif key in (32, 13):
                if state.is_complete:
                    data["frames"][str(frame_idx)] = state.to_dict()
                    save_json(data, output_path)
                    done_count += 1
                    print(f"  帧 {frame_idx} 已保存  "
                          f"({done_count}/{total_sampled})")
                    action = "next"
                    break

        if action == "back":
            cursor = max(0, cursor - 1)
        else:
            cursor += 1

    cap.release()
    print(f"\n全部标注完成！共 {len(data['frames'])} 帧。")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
