"""Headless tests for player archive state and real-frame rendering."""

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_repair_ui import MotTable, preset_paths, workspace_root
from run_track_repair_ui_players import PlayerRegistry, PlayerTrackRepairUI


registry = PlayerRegistry()
p1 = registry.new_player()
p2 = registry.new_player()
registry.assign(p1.player_id, 10)
registry.assign(p1.player_id, 11)
try:
    registry.assign(p2.player_id, 10)
except ValueError:
    pass
else:
    raise AssertionError("one track must not belong to two players")
assert registry.unassign_last(p1.player_id) == 11

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "players.json"
    registry.save(path, None)
    restored = PlayerRegistry.load(path)
    assert restored.snapshot() == registry.snapshot()

root = workspace_root()
video, tracks = preset_paths(root, "view1", "mask_conditioned")
with tempfile.TemporaryDirectory() as tmp:
    ui = PlayerTrackRepairUI(
        video,
        MotTable.load(tracks),
        Path(tmp) / "repaired.txt",
        960,
    )
    try:
        first = ui.draw()
        assert first.shape == (540, 1270, 3)
        rows = ui.table.rows_at(ui.current_frame)
        ui.new_player()
        ui.assign_track(rows[0])
        focused = ui.draw()
        assert focused.shape == first.shape
        assert ui.registry.track_owner[rows[0].track_id] == ui.active_player_id
        assert ui._last_drawn_rows == []  # archived bbox is visible but not operable
        ui.start_assign()
        ui.draw()
        assert ui._last_drawn_rows
        assert all(row.track_id not in ui.registry.track_owner for row, _ in ui._last_drawn_rows)
        print("player manager and focus rendering tests passed")
    finally:
        ui.video.close()

