"""McByte repair UI with logical-player and archived-trajectory management."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import track_repair_ui as base
from run_track_repair_ui import GapAwareTrackRepairUI


@dataclass
class PlayerRecord:
    player_id: int
    name: str
    track_ids: list[int] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "track_ids": list(self.track_ids),
        }


class PlayerRegistry:
    def __init__(self):
        self.players: dict[int, PlayerRecord] = {}
        self.track_owner: dict[int, int] = {}
        self.next_player_id = 1

    @classmethod
    def load(cls, path: Path) -> "PlayerRegistry":
        registry = cls()
        if not path.is_file():
            return registry
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("players", []):
            player = PlayerRecord(
                player_id=int(item["player_id"]),
                name=str(item.get("name") or f"Player {item['player_id']}"),
                track_ids=[int(track_id) for track_id in item.get("track_ids", [])],
            )
            registry.players[player.player_id] = player
            for track_id in player.track_ids:
                if track_id in registry.track_owner:
                    raise ValueError(f"track ID {track_id} belongs to multiple players")
                registry.track_owner[track_id] = player.player_id
        registry.next_player_id = max(registry.players, default=0) + 1
        return registry

    def new_player(self) -> PlayerRecord:
        player_id = self.next_player_id
        self.next_player_id += 1
        player = PlayerRecord(player_id, f"Player {player_id}")
        self.players[player_id] = player
        return player

    def assign(self, player_id: int, track_id: int) -> None:
        owner = self.track_owner.get(track_id)
        if owner is not None and owner != player_id:
            raise ValueError(f"track ID {track_id} is archived by Player {owner}")
        player = self.players[player_id]
        if track_id not in player.track_ids:
            player.track_ids.append(track_id)
        self.track_owner[track_id] = player_id

    def unassign_last(self, player_id: int) -> int | None:
        player = self.players[player_id]
        if not player.track_ids:
            return None
        track_id = player.track_ids.pop()
        self.track_owner.pop(track_id, None)
        return track_id

    def snapshot(self) -> list[dict]:
        return [self.players[player_id].as_dict() for player_id in sorted(self.players)]

    def save(self, path: Path, source_tracks: Path | None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "source_tracks": str(source_tracks) if source_tracks else None,
            "players": self.snapshot(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


class PlayerTrackRepairUI(GapAwareTrackRepairUI):
    sidebar_width = 310
    button_height = 38
    player_row_height = 62

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.players_path = self.output_path.with_suffix(".players.json")
        self.registry = PlayerRegistry.load(self.players_path)
        self.saved_players = self.registry.snapshot()
        self.active_player_id: int | None = None
        self.assign_mode = not bool(self.registry.players)
        self.mode = "PLAYER_POOL" if self.assign_mode else "PLAYER_LIST"
        self.message = (
            "Create a player, then click an unarchived bbox"
            if not self.registry.players
            else "Click a player in the sidebar to show only that player's tracks"
        )
        self._sidebar_hits: list[tuple[tuple[int, int, int, int], str, int | None]] = []

    @property
    def dirty(self) -> bool:
        repairs_dirty = [action.as_dict() for action in self.actions] != self.saved_actions
        return repairs_dirty or self.registry.snapshot() != self.saved_players

    def new_player(self) -> None:
        player = self.registry.new_player()
        self.active_player_id = player.player_id
        self.assign_mode = True
        self.selected_id = None
        self.good_frame = self.bad_frame = None
        self.mode = "ASSIGN_TRACK"
        self.message = f"{player.name}: click an unarchived bbox to add its trajectory"

    def focus_player(self, player_id: int) -> None:
        self.active_player_id = player_id
        self.assign_mode = False
        self.selected_id = None
        self.good_frame = self.bad_frame = None
        self.mode = "PLAYER_FOCUS"
        player = self.registry.players[player_id]
        self.message = f"Showing only {player.name}: archived track IDs {player.track_ids or '-'}"

    def start_assign(self) -> None:
        if self.active_player_id is None:
            self.new_player()
            return
        self.assign_mode = True
        self.selected_id = None
        self.good_frame = self.bad_frame = None
        self.mode = "ASSIGN_TRACK"
        player = self.registry.players[self.active_player_id]
        self.message = f"{player.name}: click an unarchived bbox"

    def show_pool(self) -> None:
        self.active_player_id = None
        self.assign_mode = True
        self.selected_id = None
        self.good_frame = self.bad_frame = None
        self.mode = "PLAYER_POOL"
        self.message = "Unarchived trajectory pool; clicking a bbox creates a new player"

    def assign_track(self, row: base.MotRow) -> None:
        if row.track_id in self.registry.track_owner:
            owner = self.registry.track_owner[row.track_id]
            self.message = f"Track ID {row.track_id} is already archived by Player {owner}"
            return
        if self.active_player_id is None:
            player = self.registry.new_player()
            self.active_player_id = player.player_id
        self.registry.assign(self.active_player_id, row.track_id)
        self.assign_mode = False
        super().select_row(row)
        player = self.registry.players[self.active_player_id]
        self.message = (
            f"Archived track ID {row.track_id} to {player.name}; "
            "N=inspect its last frame"
        )

    def remove_last_track(self) -> None:
        if self.active_player_id is None:
            self.message = "Select a player first"
            return
        track_id = self.registry.unassign_last(self.active_player_id)
        if track_id is None:
            self.message = "This player has no archived trajectory"
            return
        if self.selected_id == track_id:
            self.selected_id = None
            self.good_frame = self.bad_frame = None
        self.mode = "PLAYER_FOCUS"
        self.message = f"Returned track ID {track_id} to the unarchived pool"

    def on_n(self) -> None:
        was_end_check = self.mode == "END_CHECK"
        super().on_n()
        if was_end_check and self.active_player_id is not None:
            self.assign_mode = True
            self.mode = "ASSIGN_TRACK"
            player = self.registry.players[self.active_player_id]
            self.message = f"{player.name}: accepted; click the next unarchived trajectory"

    def on_w(self) -> None:
        super().on_w()
        if self.mode == "WAIT_SWAP":
            self.message += " (only unarchived candidates are operable)"

    def save(self) -> None:
        super().save()
        # super().save refuses only when output equals the source; preserve that guard.
        if self.table.source is not None and self.output_path.resolve() == self.table.source.resolve():
            return
        self.registry.save(self.players_path, self.table.source)
        self.saved_players = self.registry.snapshot()
        self.message = (
            f"Saved MOT repairs and {len(self.registry.players)} players -> "
            f"{self.players_path.name}"
        )

    def _video_rows_for_mode(self):
        rows = self.table.rows_at(self.current_frame)
        owned_ids: set[int] = set()
        if self.active_player_id is not None:
            owned_ids = set(self.registry.players[self.active_player_id].track_ids)
        owned_rows = [row for row in rows if row.track_id in owned_ids]
        unarchived_rows = [row for row in rows if row.track_id not in self.registry.track_owner]
        if self.mode == "WAIT_SWAP":
            candidates = [row for row in unarchived_rows if row.track_id != self.selected_id]
            return owned_rows, candidates
        if self.assign_mode:
            return owned_rows, unarchived_rows
        return owned_rows, []

    def _draw_bbox(self, image, row, color, thickness, prefix=""):
        x, y, width, height = row.xywh
        rect = (
            round(x * self.scale),
            round(y * self.scale),
            round((x + width) * self.scale),
            round((y + height) * self.scale),
        )
        x1, y1, x2, y2 = rect
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            image,
            f"{prefix}T{row.track_id} {row.confidence:.2f}",
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )
        return rect

    @staticmethod
    def _button(panel, rect, text, active=False):
        x1, y1, x2, y2 = rect
        color = (80, 120, 55) if active else (62, 62, 68)
        cv2.rectangle(panel, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(panel, (x1, y1), (x2, y2), (115, 115, 120), 1)
        cv2.putText(panel, text, (x1 + 10, y1 + 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (245, 245, 245), 1, cv2.LINE_AA)

    def _draw_sidebar(self, height: int):
        panel = np.full((height, self.sidebar_width, 3), (32, 32, 36), dtype=np.uint8)
        self._sidebar_hits = []
        cv2.putText(panel, "PLAYERS", (14, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.68, (245, 245, 245), 2, cv2.LINE_AA)
        buttons = [
            ((10, 38, 300, 74), "NEW PLAYER  [P]", "new", None),
            ((10, 82, 300, 118), "ADD TRACK   [T]", "assign", None),
            ((10, 126, 300, 162), "UNARCHIVED  [V]", "pool", None),
            ((10, 170, 300, 206), "REMOVE LAST [X]", "remove", None),
        ]
        for rect, text, action, value in buttons:
            active = (action == "assign" and self.assign_mode and self.active_player_id is not None) or (
                action == "pool" and self.active_player_id is None
            )
            self._button(panel, rect, text, active)
            self._sidebar_hits.append((rect, action, value))
        y = 220
        for player_id in sorted(self.registry.players):
            player = self.registry.players[player_id]
            rect = (10, y, 300, min(y + self.player_row_height - 6, height - 2))
            if rect[1] >= height:
                break
            active = player_id == self.active_player_id
            fill = (92, 78, 36) if active else (47, 47, 52)
            cv2.rectangle(panel, rect[:2], rect[2:], fill, -1)
            cv2.rectangle(panel, rect[:2], rect[2:], (0, 220, 255) if active else (90, 90, 95), 2 if active else 1)
            cv2.putText(panel, f"{player.name}  ({len(player.track_ids)})", (20, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
            ids_text = ", ".join(f"T{track_id}" for track_id in player.track_ids)
            if len(ids_text) > 36:
                ids_text = ids_text[:33] + "..."
            cv2.putText(panel, ids_text or "no archived tracks", (20, y + 45),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 195), 1, cv2.LINE_AA)
            self._sidebar_hits.append((rect, "player", player_id))
            y += self.player_row_height
        return panel

    def draw(self):
        frame = self.video.read(self.current_frame)
        video_canvas = cv2.resize(
            frame, (self.display_width, self.display_height), interpolation=cv2.INTER_AREA
        )
        owned_rows, operable_rows = self._video_rows_for_mode()
        self._last_drawn_rows = []

        # Operable pool boxes are gray and translucent. Archived tracks from
        # other players are completely absent from this view.
        if operable_rows:
            layer = video_canvas.copy()
            for row in operable_rows:
                rect = self._draw_bbox(layer, row, (170, 170, 170), 2)
                self._last_drawn_rows.append((row, rect))
            cv2.addWeighted(layer, 0.42, video_canvas, 0.58, 0, video_canvas)

        if self.active_player_id is not None:
            prefix = f"P{self.active_player_id}/"
            for row in owned_rows:
                selected = row.track_id == self.selected_id
                self._draw_bbox(video_canvas, row, (0, 255, 255), 4 if selected else 3, prefix)

        overlay = video_canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_width, 76), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.68, video_canvas, 0.32, 0, video_canvas)
        player_text = "POOL" if self.active_player_id is None else f"Player {self.active_player_id}"
        dirty = "UNSAVED" if self.dirty else "saved"
        lines = (
            f"frame {self.current_frame}/{self.max_frame} | {player_text} | mode {self.mode} | {dirty}",
            self.message[:145],
            "P new | T add track | V pool | X remove last | N/B binary | W repair | S save",
        )
        for index, text in enumerate(lines):
            cv2.putText(video_canvas, text, (12, 21 + index * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.54 if index != 2 else 0.46,
                        (255, 255, 255), 1, cv2.LINE_AA)
        return np.concatenate([video_canvas, self._draw_sidebar(self.display_height)], axis=1)

    @staticmethod
    def _inside(x: int, y: int, rect) -> bool:
        x1, y1, x2, y2 = rect
        return x1 <= x <= x2 and y1 <= y <= y2

    def on_mouse(self, event: int, x: int, y: int, _flags: int, _param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x >= self.display_width:
            sidebar_x = x - self.display_width
            for rect, action, value in self._sidebar_hits:
                if not self._inside(sidebar_x, y, rect):
                    continue
                if action == "new":
                    self.new_player()
                elif action == "assign":
                    self.start_assign()
                elif action == "pool":
                    self.show_pool()
                elif action == "remove":
                    self.remove_last_track()
                elif action == "player" and value is not None:
                    self.focus_player(value)
                return
            return
        row = self._row_under_cursor(x, y)
        if row is None:
            self.message = "No operable bbox here; archived boxes are view-only"
        elif self.mode == "WAIT_SWAP":
            self.repair_with(row)
        elif self.assign_mode:
            self.assign_track(row)
        else:
            self.message = "Archived trajectory is view-only; use ADD TRACK for candidates"

    def handle_key(self, key: int) -> None:
        if key in (ord("p"), ord("P")):
            self.new_player()
        elif key in (ord("t"), ord("T")):
            self.start_assign()
        elif key in (ord("v"), ord("V")):
            self.show_pool()
        elif key in (ord("x"), ord("X")):
            self.remove_last_track()
        elif key in (ord("r"), ord("R")):
            if self.active_player_id is None:
                self.show_pool()
            else:
                self.focus_player(self.active_player_id)
        else:
            super().handle_key(key)

    def run(self) -> None:
        cv2.namedWindow(base.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            base.WINDOW_NAME,
            self.display_width + self.sidebar_width,
            self.display_height + 80,
        )
        cv2.setMouseCallback(base.WINDOW_NAME, self.on_mouse)
        cv2.createTrackbar(
            "frame", base.WINDOW_NAME, self.current_frame - 1,
            self.max_frame - 1, self.on_trackbar,
        )
        try:
            while self.running:
                cv2.imshow(base.WINDOW_NAME, self.draw())
                key = cv2.waitKeyEx(20)
                if key != -1:
                    self.handle_key(key)
                if cv2.getWindowProperty(base.WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                    if self.dirty:
                        self.message = "S=save, uppercase Q=discard"
                        cv2.namedWindow(base.WINDOW_NAME, cv2.WINDOW_NORMAL)
                        cv2.setMouseCallback(base.WINDOW_NAME, self.on_mouse)
                    else:
                        break
        finally:
            self.video.close()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    base.TrackRepairUI = PlayerTrackRepairUI
    base.main()

