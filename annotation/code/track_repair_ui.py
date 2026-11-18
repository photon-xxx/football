"""Interactive bbox/MOT trajectory repair UI for McByte outputs.

Locate the first bad frame by binary search, then swap two tracker IDs from
that frame to the end. Original MOT files are never overwritten.
"""

from __future__ import annotations

import argparse
import colorsys
import json
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2


WINDOW_NAME = "McByte track repair"


@dataclass
class MotRow:
    fields: list[str]

    @property
    def frame(self) -> int:
        return int(self.fields[0])

    @property
    def track_id(self) -> int:
        return int(self.fields[1])

    @track_id.setter
    def track_id(self, value: int) -> None:
        self.fields[1] = str(value)

    @property
    def xywh(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.fields[2:6])  # type: ignore[return-value]

    @property
    def confidence(self) -> float:
        return float(self.fields[6]) if len(self.fields) > 6 else 1.0

    def serialize(self) -> str:
        return ",".join(self.fields)


@dataclass
class SwapAction:
    start_frame: int
    first_id: int
    second_id: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "operation": "swap_id_suffix",
            "start_frame": self.start_frame,
            "first_id": self.first_id,
            "second_id": self.second_id,
        }


class MotTable:
    def __init__(self, rows: list[MotRow], source: Path | None = None):
        self.rows = rows
        self.source = source
        self.by_frame: dict[int, list[MotRow]] = {}
        self.frames_by_id: dict[int, list[int]] = {}
        self.reindex()

    @classmethod
    def load(cls, path: Path) -> "MotTable":
        rows: list[MotRow] = []
        with path.open("r", encoding="utf-8") as handle:
            for line_no, raw in enumerate(handle, 1):
                text = raw.strip()
                if not text:
                    continue
                fields = [part.strip() for part in text.split(",")]
                if len(fields) < 6:
                    raise ValueError(f"{path}:{line_no}: expected >=6 columns")
                try:
                    int(fields[0])
                    int(fields[1])
                    tuple(float(v) for v in fields[2:6])
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: invalid MOT row") from exc
                rows.append(MotRow(fields))
        if not rows:
            raise ValueError(f"empty MOT file: {path}")
        return cls(rows, source=path)

    def reindex(self) -> None:
        by_frame: dict[int, list[MotRow]] = defaultdict(list)
        frames_by_id: dict[int, set[int]] = defaultdict(set)
        for row in self.rows:
            by_frame[row.frame].append(row)
            frames_by_id[row.track_id].add(row.frame)
        self.by_frame = dict(by_frame)
        self.frames_by_id = {
            track_id: sorted(frames) for track_id, frames in frames_by_id.items()
        }

    @property
    def min_frame(self) -> int:
        return min(self.by_frame)

    @property
    def max_frame(self) -> int:
        return max(self.by_frame)

    def rows_at(self, frame: int) -> list[MotRow]:
        return self.by_frame.get(frame, [])

    def track_last_frame(self, track_id: int, at_or_after: int | None = None) -> int | None:
        frames = self.frames_by_id.get(track_id, [])
        if at_or_after is not None:
            frames = [frame for frame in frames if frame >= at_or_after]
        return frames[-1] if frames else None

    def swap_suffix(self, first_id: int, second_id: int, start_frame: int) -> SwapAction:
        if first_id == second_id:
            raise ValueError("cannot swap an ID with itself")
        changed_first = changed_second = False
        for row in self.rows:
            if row.frame < start_frame:
                continue
            if row.track_id == first_id:
                row.track_id = second_id
                changed_first = True
            elif row.track_id == second_id:
                row.track_id = first_id
                changed_second = True
        if not changed_first or not changed_second:
            self._swap_raw(first_id, second_id, start_frame)
            raise ValueError("both IDs must have rows at or after the repair frame")
        self.reindex()
        return SwapAction(start_frame, first_id, second_id)

    def _swap_raw(self, first_id: int, second_id: int, start_frame: int) -> None:
        for row in self.rows:
            if row.frame < start_frame:
                continue
            if row.track_id == first_id:
                row.track_id = second_id
            elif row.track_id == second_id:
                row.track_id = first_id

    def apply_action(self, action: SwapAction) -> None:
        """Apply or undo a swap; swapping is its own inverse."""
        self._swap_raw(action.first_id, action.second_id, action.start_frame)
        self.reindex()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        ordered = sorted(enumerate(self.rows), key=lambda item: (item[1].frame, item[0]))
        with path.open("w", encoding="utf-8", newline="\n") as handle:
            for _, row in ordered:
                handle.write(row.serialize() + "\n")


class RandomAccessVideo:
    def __init__(self, path: Path, cache_size: int = 64):
        self.path = path
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open video: {path}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        self.cache_size = cache_size
        self.cache: OrderedDict[int, object] = OrderedDict()
        self.next_index = 0

    def read(self, frame_number: int):
        """Read a one-based frame exactly, using seek plus a small LRU cache."""
        index = frame_number - 1
        if index < 0 or index >= self.frame_count:
            raise IndexError(frame_number)
        if index in self.cache:
            frame = self.cache.pop(index)
            self.cache[index] = frame
            return frame.copy()
        if self.next_index != index:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            raise RuntimeError(f"failed to decode frame {frame_number} from {self.path}")
        self.next_index = index + 1
        self.cache[index] = frame
        while len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return frame.copy()

    def close(self) -> None:
        self.cap.release()


def id_color(track_id: int) -> tuple[int, int, int]:
    hue = (track_id * 0.61803398875) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.78, 1.0)
    return int(blue * 255), int(green * 255), int(red * 255)


class TrackRepairUI:
    def __init__(self, video_path: Path, table: MotTable, output_path: Path, display_width: int):
        self.video = RandomAccessVideo(video_path)
        self.table = table
        self.output_path = output_path
        self.display_width = min(display_width, self.video.width)
        self.scale = self.display_width / self.video.width
        self.display_height = round(self.video.height * self.scale)
        self.max_frame = min(self.video.frame_count, max(table.max_frame, 1))
        self.current_frame = max(1, table.min_frame)
        self.selected_id: int | None = None
        self.good_frame: int | None = None
        self.bad_frame: int | None = None
        self.mode = "SELECT"
        self.message = "Click a bbox to select a trajectory"
        self.actions: list[SwapAction] = []
        self.saved_action_count = 0
        self.running = True
        self._trackbar_guard = False
        self._last_drawn_rows: list[tuple[MotRow, tuple[int, int, int, int]]] = []

    @property
    def dirty(self) -> bool:
        return len(self.actions) != self.saved_action_count

    def set_frame(self, frame: int) -> None:
        self.current_frame = max(1, min(int(frame), self.max_frame))
        if not self._trackbar_guard:
            self._trackbar_guard = True
            cv2.setTrackbarPos("frame", WINDOW_NAME, self.current_frame - 1)
            self._trackbar_guard = False

    def clear_selection(self) -> None:
        self.selected_id = None
        self.good_frame = None
        self.bad_frame = None
        self.mode = "SELECT"
        self.message = "Click a bbox to select a trajectory"

    def select_row(self, row: MotRow) -> None:
        self.selected_id = row.track_id
        self.good_frame = self.current_frame
        self.bad_frame = self.table.track_last_frame(row.track_id, self.current_frame)
        self.mode = "SELECTED"
        self.message = f"Selected ID {row.track_id}; N: inspect its last frame"

    def _jump_midpoint(self) -> None:
        assert self.good_frame is not None and self.bad_frame is not None
        if self.bad_frame - self.good_frame <= 1:
            self.set_frame(self.bad_frame)
            self.mode = "ERROR_FOUND"
            self.message = f"First bad frame = {self.bad_frame}; W then click correct bbox"
            return
        midpoint = (self.good_frame + self.bad_frame) // 2
        self.set_frame(midpoint)
        self.mode = "SEARCH"
        self.message = f"Binary [{self.good_frame}, {self.bad_frame}] at {midpoint}: N=correct, B=bad"

    def on_n(self) -> None:
        if self.selected_id is None:
            self.message = "Select a bbox first"
            return
        if self.mode in {"SELECTED", "REPAIRED"}:
            last = self.table.track_last_frame(self.selected_id, self.current_frame)
            if last is None:
                self.message = f"ID {self.selected_id} has no later rows"
                return
            self.good_frame = self.current_frame
            self.bad_frame = last
            self.set_frame(last)
            self.mode = "END_CHECK"
            self.message = "End check: N=all correct/next trajectory, B=bad/start binary search"
        elif self.mode == "END_CHECK":
            next_frame = min(self.current_frame + 1, self.max_frame)
            self.clear_selection()
            self.set_frame(next_frame)
            self.message = "Trajectory accepted; click the target bbox for the next trajectory"
        elif self.mode == "SEARCH":
            self.good_frame = self.current_frame
            self._jump_midpoint()
        elif self.mode == "ERROR_FOUND":
            self.message = "Press W, then click the correct target bbox"
        elif self.mode == "WAIT_SWAP":
            self.message = "Click the correct target bbox, or R to cancel"

    def on_b(self) -> None:
        if self.selected_id is None or self.good_frame is None:
            self.message = "Select a trajectory before binary search"
            return
        if self.current_frame <= self.good_frame:
            self.message = "Bad frame must be after the last known-good frame"
            return
        if self.mode not in {"END_CHECK", "SEARCH"}:
            self.message = "B is valid at the end check or during binary search"
            return
        self.bad_frame = self.current_frame
        self._jump_midpoint()

    def on_w(self) -> None:
        if self.selected_id is None:
            self.message = "Select the wrong trajectory first"
            return
        self.mode = "WAIT_SWAP"
        self.message = f"Repair at frame {self.current_frame}: click bbox that should become ID {self.selected_id}"

    def repair_with(self, correct_row: MotRow) -> None:
        assert self.selected_id is not None
        if correct_row.track_id == self.selected_id:
            self.message = "Clicked selected ID; choose the other/correct trajectory"
            return
        wrong_id = self.selected_id
        try:
            action = self.table.swap_suffix(wrong_id, correct_row.track_id, self.current_frame)
        except ValueError as exc:
            self.message = f"Repair rejected: {exc}"
            return
        self.actions.append(action)
        self.good_frame = self.current_frame
        self.bad_frame = None
        self.mode = "REPAIRED"
        self.message = (
            f"Swapped ID {wrong_id}<->{correct_row.track_id} from frame {self.current_frame}; "
            "N=inspect end, U=undo"
        )

    def undo(self) -> None:
        if not self.actions:
            self.message = "Nothing to undo"
            return
        action = self.actions.pop()
        self.table.apply_action(action)
        self.clear_selection()
        self.set_frame(action.start_frame)
        self.message = (
            f"Undid swap ID {action.first_id}<->{action.second_id} from frame {action.start_frame}"
        )

    def save(self) -> None:
        if self.table.source is not None and self.output_path.resolve() == self.table.source.resolve():
            self.message = "Refusing to overwrite source MOT; choose another --output"
            return
        self.table.save(self.output_path)
        log_path = self.output_path.with_suffix(self.output_path.suffix + ".repairs.json")
        payload = {
            "source_tracks": str(self.table.source) if self.table.source else None,
            "output_tracks": str(self.output_path),
            "repair_count": len(self.actions),
            "repairs": [action.as_dict() for action in self.actions],
        }
        log_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        self.saved_action_count = len(self.actions)
        self.message = f"Saved {len(self.actions)} repairs -> {self.output_path.name}"

    def _row_under_cursor(self, x: int, y: int) -> MotRow | None:
        hits: list[tuple[float, MotRow]] = []
        for row, (x1, y1, x2, y2) in self._last_drawn_rows:
            if x1 <= x <= x2 and y1 <= y <= y2:
                hits.append(((x2 - x1) * (y2 - y1), row))
        return min(hits, key=lambda item: item[0])[1] if hits else None

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        row = self._row_under_cursor(x, y)
        if row is None:
            self.message = "No bbox under cursor"
        elif self.mode == "WAIT_SWAP":
            self.repair_with(row)
        else:
            self.select_row(row)

    def on_trackbar(self, index: int) -> None:
        if not self._trackbar_guard:
            self.current_frame = max(1, min(index + 1, self.max_frame))

    def draw(self):
        frame = self.video.read(self.current_frame)
        canvas = cv2.resize(frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA)
        self._last_drawn_rows = []
        for row in self.table.rows_at(self.current_frame):
            x, y, width, height = row.xywh
            x1, y1 = round(x * self.scale), round(y * self.scale)
            x2, y2 = round((x + width) * self.scale), round((y + height) * self.scale)
            selected = row.track_id == self.selected_id
            color = (0, 255, 255) if selected else id_color(row.track_id)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 4 if selected else 2)
            cv2.putText(
                canvas, f"ID {row.track_id} {row.confidence:.2f}", (x1, max(16, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA,
            )
            self._last_drawn_rows.append((row, (x1, y1, x2, y2)))
        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_width, 76), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0, canvas)
        selected = "-" if self.selected_id is None else str(self.selected_id)
        dirty = "UNSAVED" if self.dirty else "saved"
        lines = (
            f"frame {self.current_frame}/{self.max_frame} | mode {self.mode} | selected {selected} | {dirty}",
            self.message[:150],
            "N correct/next | B bad | W repair | click bbox | A/D +/-1 | J/L +/-10 | U undo | S save | Q quit",
        )
        for index, text in enumerate(lines):
            cv2.putText(canvas, text, (12, 21 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55 if index != 2 else 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        return canvas

    def handle_key(self, key: int) -> None:
        if key in (ord("n"), ord("N")):
            self.on_n()
        elif key in (ord("b"), ord("B")):
            self.on_b()
        elif key in (ord("w"), ord("W")):
            self.on_w()
        elif key in (ord("a"), ord("A"), 2424832):
            self.set_frame(self.current_frame - 1)
        elif key in (ord("d"), ord("D"), 2555904):
            self.set_frame(self.current_frame + 1)
        elif key in (ord("j"), ord("J")):
            self.set_frame(self.current_frame - 10)
        elif key in (ord("l"), ord("L")):
            self.set_frame(self.current_frame + 10)
        elif key in (ord("u"), ord("U")):
            self.undo()
        elif key in (ord("s"), ord("S")):
            self.save()
        elif key in (ord("r"), ord("R")):
            self.clear_selection()
        elif key == ord("q") or key == 27:
            if self.dirty:
                self.message = "Unsaved repairs: S=save, uppercase Q=discard"
            else:
                self.running = False
        elif key == ord("Q"):
            self.running = False

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.display_width, self.display_height + 80)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
        cv2.createTrackbar("frame", WINDOW_NAME, self.current_frame - 1,
                           self.max_frame - 1, self.on_trackbar)
        try:
            while self.running:
                cv2.imshow(WINDOW_NAME, self.draw())
                key = cv2.waitKeyEx(20)
                if key != -1:
                    self.handle_key(key)
                if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    if self.dirty:
                        self.message = "S=save then Q=quit; uppercase Q discards"
                        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)
                    else:
                        break
        finally:
            self.video.close()
            cv2.destroyAllWindows()


def workspace_root() -> Path:
    """Locate the dataset workspace.

    ``MVANNOT_ROOT`` lets the tool run outside the original research workspace:
    it must point at a directory that contains ``experiment/``.
    """
    override = os.environ.get("MVANNOT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "experiment").is_dir() and (parent / "AGENTS.md").exists():
            return parent
    return here.parents[3]


def preset_paths(root: Path, view: str, tracker: str) -> tuple[Path, Path]:
    stem = {"view1": "cumtb_view1_undistorted", "view2": "cumtb_view2", "view3": "cumtb_view3"}[view]
    video = root / "experiment" / "mcbyte-five-video-eval" / "data" / "inputs" / f"{stem}.mp4"
    tracks = root / "experiment" / "mcbyte-gt-yoloxmot-rerun" / "outputs" / stem / f"{tracker}.txt"
    return video, tracks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=("view1", "view2", "view3"), default="view1")
    parser.add_argument("--tracker", choices=("mask_conditioned", "locked_iou"), default="mask_conditioned")
    parser.add_argument("--video", type=Path, help="override preset video")
    parser.add_argument("--tracks", type=Path, help="override preset MOT file")
    parser.add_argument("--output", type=Path, help="repaired MOT output")
    parser.add_argument("--display-width", type=int, default=1440)
    parser.add_argument("--headless-check", action="store_true", help="validate without GUI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = workspace_root()
    preset_video, preset_tracks = preset_paths(root, args.view, args.tracker)
    video_path = (args.video or preset_video).resolve()
    tracks_path = (args.tracks or preset_tracks).resolve()
    output_path = args.output or (root / "experiment" / "mcbyte-track-repair-ui" / "outputs" /
                                  f"{args.view}_{args.tracker}_repaired.txt")
    if not video_path.is_file():
        raise FileNotFoundError(video_path)
    if not tracks_path.is_file():
        raise FileNotFoundError(tracks_path)
    table = MotTable.load(tracks_path)
    video = RandomAccessVideo(video_path)
    metadata = {
        "view": args.view, "video": str(video_path), "tracks": str(tracks_path),
        "output": str(output_path.resolve()), "video_frames": video.frame_count,
        "mot_frame_range": [table.min_frame, table.max_frame], "mot_rows": len(table.rows),
        "unique_ids": len(table.frames_by_id), "fps": video.fps,
        "resolution": [video.width, video.height],
    }
    video.close()
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    if args.headless_check:
        return
    TrackRepairUI(video_path, table, output_path.resolve(), args.display_width).run()


if __name__ == "__main__":
    main()
