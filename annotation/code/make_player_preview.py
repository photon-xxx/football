"""Generate a real-data preview of the player sidebar UI."""

from pathlib import Path
import sys

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_repair_ui import MotTable, preset_paths, workspace_root
from run_track_repair_ui_players import PlayerTrackRepairUI


root = workspace_root()
video, tracks = preset_paths(root, "view1", "mask_conditioned")
output = root / "experiment" / "mcbyte-track-repair-ui" / "outputs" / "player_sidebar_preview.jpg"
ui = PlayerTrackRepairUI(video, MotTable.load(tracks), output.with_suffix(".txt"), 1280)
try:
    ui.current_frame = 360
    rows = ui.table.rows_at(ui.current_frame)
    player = ui.registry.new_player()
    ui.active_player_id = player.player_id
    for row in rows[:2]:
        if row.track_id not in ui.registry.track_owner:
            ui.registry.assign(player.player_id, row.track_id)
    ui.focus_player(player.player_id)
    image = ui.draw()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), image):
        raise RuntimeError(f"failed to write {output}")
    print(output)
finally:
    ui.video.close()
