"""Canonical launcher with gap-aware binary search and robust dirty tracking."""

from __future__ import annotations

import json

import track_repair_ui as base


class GapAwareTrackRepairUI(base.TrackRepairUI):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_actions: list[dict[str, int | str]] = []

    @property
    def dirty(self) -> bool:
        return [action.as_dict() for action in self.actions] != self.saved_actions

    def _jump_midpoint(self) -> None:
        assert self.selected_id is not None
        assert self.good_frame is not None and self.bad_frame is not None
        candidates = [
            frame
            for frame in self.table.frames_by_id.get(self.selected_id, [])
            if self.good_frame < frame < self.bad_frame
        ]
        if not candidates:
            self.set_frame(self.bad_frame)
            self.mode = "ERROR_FOUND"
            self.message = (
                f"First observable bad frame = {self.bad_frame}; "
                "W then click correct bbox"
            )
            return
        target = (self.good_frame + self.bad_frame) / 2
        midpoint = min(candidates, key=lambda frame: (abs(frame - target), frame))
        self.set_frame(midpoint)
        self.mode = "SEARCH"
        self.message = (
            f"Gap-aware binary [{self.good_frame}, {self.bad_frame}] at {midpoint}: "
            "N=correct, B=bad"
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
        self.saved_actions = [action.as_dict().copy() for action in self.actions]
        self.message = f"Saved {len(self.actions)} repairs -> {self.output_path.name}"


if __name__ == "__main__":
    base.TrackRepairUI = GapAwareTrackRepairUI
    base.main()

