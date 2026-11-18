"""Three-view McByte repair session with shared logical players.

The three camera views keep independent MOT rows and repair histories, while
logical players are shared.  A tracker ID is unique only inside one view.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import track_repair_ui as base
from run_track_repair_ui_players import PlayerTrackRepairUI


VIEWS = ("view1", "view2", "view3")


@dataclass
class SessionPlayerRecord:
    player_id: int
    name: str
    tracks_by_view: dict[str, list[int]] = field(default_factory=dict)
    active_view: str = "view1"

    @property
    def track_ids(self) -> list[int]:
        """Compatibility facade used by the single-view player UI."""
        return self.tracks_by_view.setdefault(self.active_view, [])

    def as_dict(self) -> dict:
        return {
            "player_id": self.player_id,
            "name": self.name,
            "tracks_by_view": {
                view: list(self.tracks_by_view.get(view, [])) for view in VIEWS
            },
        }


class SessionPlayerRegistry:
    """Global players plus a separate track-ID namespace for every view."""

    schema_version = 2

    def __init__(self, active_view: str = "view1"):
        self.active_view = active_view
        self.players: dict[int, SessionPlayerRecord] = {}
        self.owners_by_view: dict[str, dict[int, int]] = {view: {} for view in VIEWS}
        self.next_player_id = 1

    @property
    def track_owner(self) -> dict[int, int]:
        return self.owners_by_view[self.active_view]

    def set_view(self, view: str) -> None:
        if view not in VIEWS:
            raise ValueError(view)
        self.active_view = view
        for player in self.players.values():
            player.active_view = view

    @classmethod
    def load(
        cls,
        session_path: Path,
        legacy_path: Path | None = None,
        legacy_view: str = "view1",
    ) -> tuple["SessionPlayerRegistry", bool]:
        """Load a session, or migrate the old per-view players file in memory."""
        path = session_path if session_path.is_file() else legacy_path
        registry = cls(legacy_view)
        if path is None or not path.is_file():
            return registry, False
        payload = json.loads(path.read_text(encoding="utf-8"))
        migrated = path != session_path or int(payload.get("schema_version", 1)) < 2
        for item in payload.get("players", []):
            if "tracks_by_view" in item:
                tracks = {
                    view: [int(track_id) for track_id in item["tracks_by_view"].get(view, [])]
                    for view in VIEWS
                }
            else:
                tracks = {view: [] for view in VIEWS}
                tracks[legacy_view] = [int(track_id) for track_id in item.get("track_ids", [])]
            player_id = int(item["player_id"])
            player = SessionPlayerRecord(
                player_id=player_id,
                name=str(item.get("name") or f"Player {player_id}"),
                tracks_by_view=tracks,
                active_view=legacy_view,
            )
            registry.players[player_id] = player
            for view, track_ids in tracks.items():
                for track_id in track_ids:
                    owner = registry.owners_by_view[view].get(track_id)
                    if owner is not None:
                        raise ValueError(
                            f"{view} track ID {track_id} belongs to Players {owner} and {player_id}"
                        )
                    registry.owners_by_view[view][track_id] = player_id
        registry.next_player_id = max(registry.players, default=0) + 1
        registry.set_view(legacy_view)
        return registry, migrated

    def new_player(self) -> SessionPlayerRecord:
        player_id = self.next_player_id
        self.next_player_id += 1
        player = SessionPlayerRecord(
            player_id,
            f"Player {player_id}",
            {view: [] for view in VIEWS},
            self.active_view,
        )
        self.players[player_id] = player
        return player

    def assign(self, player_id: int, track_id: int) -> None:
        owner = self.track_owner.get(track_id)
        if owner is not None and owner != player_id:
            raise ValueError(
                f"{self.active_view} track ID {track_id} is archived by Player {owner}"
            )
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

    def save(self, path: Path, source_tracks: dict[str, str] | Path | None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        sources = source_tracks if isinstance(source_tracks, dict) else {}
        payload = {
            "schema_version": self.schema_version,
            "session": "cumtb",
            "views": list(VIEWS),
            "source_tracks": sources,
            "players": self.snapshot(),
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@dataclass
class ViewState:
    view: str
    video_path: Path
    source_tracks: Path
    output_path: Path
    table: base.MotTable
    actions: list[base.SwapAction]
    saved_actions: list[dict]
    current_frame: int
    selected_id: int | None = None
    good_frame: int | None = None
    bad_frame: int | None = None
    mode: str = "PLAYER_LIST"
    message: str = ""


def _action_from_dict(item: dict) -> base.SwapAction:
    if item.get("operation") != "swap_id_suffix":
        raise ValueError(f"unsupported repair operation: {item.get('operation')}")
    return base.SwapAction(
        start_frame=int(item["start_frame"]),
        first_id=int(item["first_id"]),
        second_id=int(item["second_id"]),
    )


def load_view_state(root: Path, view: str, tracker: str) -> ViewState:
    video_path, source_tracks = base.preset_paths(root, view, tracker)
    output_path = (
        root
        / "experiment"
        / "mcbyte-track-repair-ui"
        / "outputs"
        / f"{view}_{tracker}_repaired.txt"
    )
    repair_path = output_path.with_suffix(output_path.suffix + ".repairs.json")
    actions: list[base.SwapAction] = []
    if repair_path.is_file():
        table = base.MotTable.load(source_tracks)
        payload = json.loads(repair_path.read_text(encoding="utf-8"))
        actions = [_action_from_dict(item) for item in payload.get("repairs", [])]
        for action in actions:
            table.apply_action(action)
    elif output_path.is_file():
        # Preserve an output made by an older version even when its audit log is absent.
        table = base.MotTable.load(output_path)
        table.source = source_tracks
    else:
        table = base.MotTable.load(source_tracks)
    return ViewState(
        view=view,
        video_path=video_path,
        source_tracks=source_tracks,
        output_path=output_path,
        table=table,
        actions=actions,
        saved_actions=[action.as_dict().copy() for action in actions],
        current_frame=max(1, table.min_frame),
    )


class SessionTrackRepairUI(PlayerTrackRepairUI):
    """Player UI that switches three views without changing the player list."""

    view_button_top = 38
    list_top = 264
    list_bottom_margin = 12
    scroll_width = 10

    def __init__(
        self,
        states: dict[str, ViewState],
        initial_view: str,
        registry: SessionPlayerRegistry,
        session_path: Path,
        display_width: int,
        migrated_legacy: bool = False,
    ):
        state = states[initial_view]
        super().__init__(
            state.video_path,
            state.table,
            state.output_path,
            display_width,
        )
        self.states = states
        self.current_view = initial_view
        self.registry = registry
        self.registry.set_view(initial_view)
        self.players_path = session_path
        self.saved_players = self.registry.snapshot()
        self.actions = state.actions
        self.saved_actions = state.saved_actions
        self.current_frame = state.current_frame
        self.active_player_id = None
        self.assign_mode = not bool(self.registry.players)
        self.mode = "PLAYER_POOL" if self.assign_mode else "PLAYER_LIST"
        self.sidebar_scroll = 0
        self._scrollbar_track: tuple[int, int, int, int] | None = None
        self._scrollbar_thumb: tuple[int, int, int, int] | None = None
        self._scroll_drag_offset: int | None = None
        if migrated_legacy:
            self.message = (
                "Imported existing view1 players into this session; press S to save migration"
            )
        elif self.registry.players:
            self.message = "Shared player session loaded; select a player, then match this view"

    def _capture_current_state(self) -> None:
        state = self.states[self.current_view]
        state.table = self.table
        state.actions = self.actions
        state.saved_actions = self.saved_actions
        state.current_frame = self.current_frame
        state.selected_id = self.selected_id
        state.good_frame = self.good_frame
        state.bad_frame = self.bad_frame
        state.mode = self.mode
        state.message = self.message

    @property
    def dirty(self) -> bool:
        if not hasattr(self, "states"):
            return super().dirty
        self._capture_current_state()
        repairs_dirty = any(
            [action.as_dict() for action in state.actions] != state.saved_actions
            for state in self.states.values()
        )
        return repairs_dirty or self.registry.snapshot() != self.saved_players

    def _sync_trackbar(self) -> None:
        try:
            cv2.setTrackbarMax("frame", base.WINDOW_NAME, max(1, self.max_frame - 1))
            cv2.setTrackbarPos("frame", base.WINDOW_NAME, self.current_frame - 1)
        except cv2.error:
            # Headless tests switch views before an OpenCV window exists.
            pass

    def switch_view(self, view: str) -> None:
        if view == self.current_view:
            self.message = f"Already showing {view}"
            return
        if view not in self.states:
            raise ValueError(view)
        self._capture_current_state()
        active_player = self.active_player_id
        assign_mode = self.assign_mode
        self.video.close()
        state = self.states[view]
        self.current_view = view
        self.registry.set_view(view)
        self.video = base.RandomAccessVideo(state.video_path)
        self.table = state.table
        self.output_path = state.output_path
        self.actions = state.actions
        self.saved_actions = state.saved_actions
        self.display_width = min(self.display_width, self.video.width)
        self.scale = self.display_width / self.video.width
        self.display_height = round(self.video.height * self.scale)
        self.max_frame = min(self.video.frame_count, max(self.table.max_frame, 1))
        self.current_frame = max(1, min(state.current_frame, self.max_frame))
        self.active_player_id = active_player
        self.assign_mode = assign_mode
        self.selected_id = None
        self.good_frame = self.bad_frame = None
        if active_player is None:
            self.mode = "PLAYER_POOL"
            self.message = f"{view}: unarchived trajectory pool"
        elif assign_mode:
            self.mode = "ASSIGN_TRACK"
            self.message = f"{view}: click a trajectory for Player {active_player}"
        else:
            self.mode = "PLAYER_FOCUS"
            self.message = f"{view}: showing Player {active_player} only"
        self.sidebar_scroll = 0
        self._sync_trackbar()

    def _write_view(self, state: ViewState) -> None:
        state.table.save(state.output_path)
        repair_path = state.output_path.with_suffix(state.output_path.suffix + ".repairs.json")
        payload = {
            "source_tracks": str(state.source_tracks),
            "output_tracks": str(state.output_path),
            "repair_count": len(state.actions),
            "repairs": [action.as_dict() for action in state.actions],
        }
        repair_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        state.saved_actions = [action.as_dict().copy() for action in state.actions]

    def save(self) -> None:
        self._capture_current_state()
        saved_views: list[str] = []
        for state in self.states.values():
            is_current = state.view == self.current_view
            is_dirty = [action.as_dict() for action in state.actions] != state.saved_actions
            if is_current or is_dirty:
                self._write_view(state)
                saved_views.append(state.view)
        sources = {view: str(state.source_tracks) for view, state in self.states.items()}
        self.registry.save(self.players_path, sources)
        self.saved_players = self.registry.snapshot()
        self.saved_actions = self.states[self.current_view].saved_actions
        self.message = (
            f"Saved shared session ({len(self.registry.players)} players) and "
            f"{', '.join(saved_views)}"
        )

    def _scroll_limits(self, height: int) -> tuple[int, int, int]:
        viewport = max(1, height - self.list_top - self.list_bottom_margin)
        content = len(self.registry.players) * self.player_row_height
        return viewport, content, max(0, content - viewport)

    def _set_scroll(self, value: int, height: int) -> None:
        _, _, maximum = self._scroll_limits(height)
        self.sidebar_scroll = max(0, min(int(value), maximum))

    def _ensure_player_visible(self, player_id: int) -> None:
        ordered = sorted(self.registry.players)
        if player_id not in ordered:
            return
        index = ordered.index(player_id)
        top = index * self.player_row_height
        bottom = top + self.player_row_height
        viewport, _, _ = self._scroll_limits(self.display_height)
        if top < self.sidebar_scroll:
            self._set_scroll(top, self.display_height)
        elif bottom > self.sidebar_scroll + viewport:
            self._set_scroll(bottom - viewport, self.display_height)

    def new_player(self) -> None:
        super().new_player()
        if self.active_player_id is not None:
            self._ensure_player_visible(self.active_player_id)

    def focus_player(self, player_id: int) -> None:
        super().focus_player(player_id)
        self._ensure_player_visible(player_id)

    def _draw_sidebar(self, height: int):
        panel = np.full((height, self.sidebar_width, 3), (32, 32, 36), dtype=np.uint8)
        self._sidebar_hits = []
        cv2.putText(
            panel,
            f"PLAYERS | {self.current_view.upper()}",
            (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (245, 245, 245),
            2,
            cv2.LINE_AA,
        )
        segment_width = 94
        for index, view in enumerate(VIEWS):
            x1 = 10 + index * (segment_width + 5)
            rect = (x1, 38, x1 + segment_width, 74)
            self._button(panel, rect, view.upper(), view == self.current_view)
            self._sidebar_hits.append((rect, "view", view))
        buttons = [
            ((10, 82, 300, 118), "NEW PLAYER  [P]", "new", None),
            ((10, 126, 300, 162), "ADD TRACK   [T]", "assign", None),
            ((10, 170, 300, 206), "UNARCHIVED  [V]", "pool", None),
            ((10, 214, 300, 250), "REMOVE LAST [X]", "remove", None),
        ]
        for rect, text, action, value in buttons:
            active = (action == "assign" and self.assign_mode and self.active_player_id is not None) or (
                action == "pool" and self.active_player_id is None
            )
            self._button(panel, rect, text, active)
            self._sidebar_hits.append((rect, action, value))

        viewport, content, maximum = self._scroll_limits(height)
        clip_bottom = height - self.list_bottom_margin
        y = self.list_top - self.sidebar_scroll
        for player_id in sorted(self.registry.players):
            player = self.registry.players[player_id]
            row_top, row_bottom = y, y + self.player_row_height - 6
            y += self.player_row_height
            if row_bottom < self.list_top or row_top >= clip_bottom:
                continue
            draw_top = max(row_top, self.list_top)
            draw_bottom = min(row_bottom, clip_bottom - 1)
            rect = (10, draw_top, 290, draw_bottom)
            active = player_id == self.active_player_id
            fill = (92, 78, 36) if active else (47, 47, 52)
            cv2.rectangle(panel, rect[:2], rect[2:], fill, -1)
            cv2.rectangle(
                panel,
                rect[:2],
                rect[2:],
                (0, 220, 255) if active else (90, 90, 95),
                2 if active else 1,
            )
            text_y = row_top + 22
            if text_y >= self.list_top + 12:
                total = sum(len(ids) for ids in player.tracks_by_view.values())
                cv2.putText(
                    panel,
                    f"{player.name}  ({self.current_view}:{len(player.track_ids)} all:{total})",
                    (20, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            ids_text = ", ".join(f"T{track_id}" for track_id in player.track_ids)
            if len(ids_text) > 34:
                ids_text = ids_text[:31] + "..."
            second_y = row_top + 45
            if self.list_top + 10 <= second_y < clip_bottom:
                cv2.putText(
                    panel,
                    ids_text or f"no {self.current_view} tracks",
                    (20, second_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (190, 190, 195),
                    1,
                    cv2.LINE_AA,
                )
            self._sidebar_hits.append((rect, "player", player_id))

        track = (298, self.list_top, 307, clip_bottom)
        self._scrollbar_track = track
        cv2.rectangle(panel, track[:2], track[2:], (55, 55, 61), -1)
        if content > viewport:
            track_height = track[3] - track[1]
            thumb_height = max(36, round(track_height * viewport / content))
            travel = track_height - thumb_height
            thumb_top = track[1] + round(travel * self.sidebar_scroll / maximum)
            thumb = (track[0], thumb_top, track[2], thumb_top + thumb_height)
            cv2.rectangle(panel, thumb[:2], thumb[2:], (145, 145, 153), -1)
            self._scrollbar_thumb = thumb
        else:
            self._scrollbar_thumb = track
            cv2.rectangle(panel, track[:2], track[2:], (100, 100, 108), -1)
        return panel

    def draw(self):
        canvas = super().draw()
        view_text = self.current_view.upper()
        cv2.putText(
            canvas,
            view_text,
            (self.display_width - 90, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        return canvas

    def _scroll_from_pointer(self, y: int) -> None:
        if self._scrollbar_track is None or self._scrollbar_thumb is None:
            return
        viewport, content, maximum = self._scroll_limits(self.display_height)
        if content <= viewport:
            return
        track_top, track_bottom = self._scrollbar_track[1], self._scrollbar_track[3]
        thumb_height = self._scrollbar_thumb[3] - self._scrollbar_thumb[1]
        travel = max(1, track_bottom - track_top - thumb_height)
        thumb_top = y - (self._scroll_drag_offset or thumb_height // 2)
        ratio = (thumb_top - track_top) / travel
        self._set_scroll(round(max(0.0, min(1.0, ratio)) * maximum), self.display_height)

    def on_mouse(self, event: int, x: int, y: int, flags: int, param) -> None:
        if event == cv2.EVENT_MOUSEWHEEL and x >= self.display_width:
            delta = cv2.getMouseWheelDelta(flags)
            steps = -1 if delta > 0 else 1
            self._set_scroll(
                self.sidebar_scroll + steps * self.player_row_height * 3,
                self.display_height,
            )
            return
        if x >= self.display_width:
            sidebar_x = x - self.display_width
            if event == cv2.EVENT_LBUTTONUP:
                self._scroll_drag_offset = None
                return
            if event == cv2.EVENT_MOUSEMOVE and self._scroll_drag_offset is not None:
                self._scroll_from_pointer(y)
                return
            if event == cv2.EVENT_LBUTTONDOWN:
                if self._scrollbar_thumb and self._inside(sidebar_x, y, self._scrollbar_thumb):
                    self._scroll_drag_offset = y - self._scrollbar_thumb[1]
                    return
                if self._scrollbar_track and self._inside(sidebar_x, y, self._scrollbar_track):
                    self._scroll_drag_offset = None
                    self._scroll_from_pointer(y)
                    return
                for rect, action, value in self._sidebar_hits:
                    if self._inside(sidebar_x, y, rect) and action == "view":
                        self.switch_view(str(value))
                        return
        super().on_mouse(event, x, y, flags, param)

    def handle_key(self, key: int) -> None:
        if key in (ord("1"), ord("2"), ord("3")):
            self.switch_view(f"view{chr(key)}")
        else:
            super().handle_key(key)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--view", choices=VIEWS, default="view1", help="initial view")
    parser.add_argument(
        "--tracker", choices=("mask_conditioned", "locked_iou"), default="mask_conditioned"
    )
    parser.add_argument("--display-width", type=int, default=1440)
    parser.add_argument("--session", type=Path, help="shared session JSON override")
    parser.add_argument("--headless-check", action="store_true")
    return parser.parse_args()


def build_ui(args: argparse.Namespace) -> SessionTrackRepairUI:
    root = base.workspace_root()
    states = {view: load_view_state(root, view, args.tracker) for view in VIEWS}
    for state in states.values():
        if not state.video_path.is_file():
            raise FileNotFoundError(state.video_path)
        if not state.source_tracks.is_file():
            raise FileNotFoundError(state.source_tracks)
    outputs = root / "experiment" / "mcbyte-track-repair-ui" / "outputs"
    session_path = (args.session or outputs / f"cumtb_{args.tracker}.session.json").resolve()
    legacy_path = outputs / f"view1_{args.tracker}_repaired.players.json"
    registry, migrated = SessionPlayerRegistry.load(session_path, legacy_path, "view1")
    registry.set_view(args.view)
    return SessionTrackRepairUI(
        states,
        args.view,
        registry,
        session_path,
        args.display_width,
        migrated_legacy=migrated,
    )


def main() -> None:
    args = parse_args()
    ui = build_ui(args)
    metadata = {
        "session": str(ui.players_path),
        "initial_view": ui.current_view,
        "players": len(ui.registry.players),
        "views": {
            view: {
                "video": str(state.video_path),
                "tracks": str(state.source_tracks),
                "output": str(state.output_path),
                "mot_rows": len(state.table.rows),
                "unique_ids": len(state.table.frames_by_id),
            }
            for view, state in ui.states.items()
        },
    }
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    if args.headless_check:
        ui.video.close()
        return
    ui.run()


if __name__ == "__main__":
    main()
