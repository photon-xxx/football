"""Render a real frame before and after trajectory selection."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_repair_ui import MotTable, preset_paths, workspace_root
from run_track_repair_ui_focus import FocusTrackRepairUI


root = workspace_root()
video, tracks = preset_paths(root, "view1", "mask_conditioned")
ui = FocusTrackRepairUI(video, MotTable.load(tracks), root / "experiment" / "mcbyte-track-repair-ui" / "outputs" / "unused.txt", 960)
try:
    before = ui.draw()
    rows = ui.table.rows_at(ui.current_frame)
    assert len(rows) >= 2
    ui.selected_id = rows[0].track_id
    after = ui.draw()
    assert before.shape == after.shape == (540, 960, 3)
    assert (before != after).any()
    assert any(row.track_id != ui.selected_id for row in rows)
    print("focus rendering test passed")
finally:
    ui.video.close()

