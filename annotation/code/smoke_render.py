"""Decode and draw representative frames without opening a GUI window."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_repair_ui import MotTable, TrackRepairUI, preset_paths, workspace_root


root = workspace_root()
for view in ("view1", "view2", "view3"):
    video, tracks = preset_paths(root, view, "mask_conditioned")
    ui = TrackRepairUI(video, MotTable.load(tracks), root / "experiment" / "mcbyte-track-repair-ui" / "outputs" / "unused.txt", 960)
    try:
        for frame_number in (1, 720, 1440):
            ui.current_frame = frame_number
            image = ui.draw()
            assert image.shape == (540, 960, 3), (view, frame_number, image.shape)
        print(f"{view}: frames 1/720/1440 decoded and rendered")
    finally:
        ui.video.close()

