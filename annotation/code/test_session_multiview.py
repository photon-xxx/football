"""Headless regression checks for the shared-player three-view session."""

from argparse import Namespace
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

import track_repair_ui as base
import run_track_repair_ui_session as implementation
from run_track_repair_ui_multiview import SessionTrackRepairUI


root = base.workspace_root()
outputs = root / "experiment" / "mcbyte-track-repair-ui" / "outputs"
legacy = outputs / "view1_mask_conditioned_repaired.players.json"
states = {
    view: implementation.load_view_state(root, view, "mask_conditioned")
    for view in implementation.VIEWS
}

with tempfile.TemporaryDirectory() as tmp:
    session_path = Path(tmp) / "cumtb.session.json"
    registry, migrated = implementation.SessionPlayerRegistry.load(
        session_path, legacy, "view1"
    )
    assert migrated
    assert len(registry.players) == 26
    assert registry.owners_by_view["view1"][0] == 1
    assert registry.owners_by_view["view2"] == {}

    ui = SessionTrackRepairUI(
        states,
        "view1",
        registry,
        session_path,
        960,
        migrated_legacy=migrated,
    )
    try:
        assert ui.dirty  # migration cannot be silently discarded
        first = ui.draw()
        assert first.shape == (540, 1270, 3)

        viewport, content, maximum = ui._scroll_limits(ui.display_height)
        assert content > viewport and maximum > 0
        ui._set_scroll(maximum, ui.display_height)
        ui.draw()
        visible_players = {
            value for _, action, value in ui._sidebar_hits if action == "player"
        }
        assert 26 in visible_players
        assert 1 not in visible_players

        # ID namespaces are independent: view2/T0 may belong to another player.
        ui.switch_view("view2")
        assert ui.current_view == "view2"
        assert ui.registry.active_view == "view2"
        ui.registry.assign(2, 0)
        assert ui.registry.owners_by_view["view2"][0] == 2
        assert ui.registry.owners_by_view["view1"][0] == 1
        second = ui.draw()
        assert second.shape == first.shape

        ui.switch_view("view3")
        third = ui.draw()
        assert third.shape == first.shape

        sources = {view: str(state.source_tracks) for view, state in states.items()}
        registry.save(session_path, sources)
        restored, remigrated = implementation.SessionPlayerRegistry.load(session_path)
        assert not remigrated
        assert restored.snapshot() == registry.snapshot()
        print("shared session, three-view switching, and sidebar scrolling tests passed")
    finally:
        ui.video.close()
