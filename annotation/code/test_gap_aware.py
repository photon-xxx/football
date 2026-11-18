"""Headless test for binary search when a trajectory has missing frames."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_repair_ui import MotRow, MotTable
from run_track_repair_ui import GapAwareTrackRepairUI


def row(frame: int) -> MotRow:
    return MotRow([str(frame), "7", "0", "0", "10", "20", "1", "-1", "-1", "-1"])


ui = object.__new__(GapAwareTrackRepairUI)
ui.table = MotTable([row(frame) for frame in (1, 2, 8, 9, 20)])
ui.selected_id = 7
ui.good_frame = 2
ui.bad_frame = 20
ui.current_frame = 2
ui.mode = "SEARCH"
ui.message = ""
ui.set_frame = lambda frame: setattr(ui, "current_frame", frame)
ui._jump_midpoint()
assert ui.current_frame in (8, 9)
ui.good_frame = 9
ui.bad_frame = 20
ui._jump_midpoint()
assert ui.current_frame == 20
assert ui.mode == "ERROR_FOUND"
print("gap-aware binary search tests passed")
