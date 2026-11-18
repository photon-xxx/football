"""Canonical launcher for the shared-player, three-view CumTB session UI."""

from __future__ import annotations

import run_track_repair_ui_session as implementation


class SessionTrackRepairUI(implementation.SessionTrackRepairUI):
    """Keep a legacy-to-session migration dirty until it is explicitly saved."""

    def __init__(self, *args, migrated_legacy: bool = False, **kwargs):
        self.migration_pending = migrated_legacy
        super().__init__(*args, migrated_legacy=migrated_legacy, **kwargs)

    @property
    def dirty(self) -> bool:
        return super().dirty or self.migration_pending

    def save(self) -> None:
        super().save()
        self.migration_pending = False


def main() -> None:
    implementation.SessionTrackRepairUI = SessionTrackRepairUI
    implementation.main()


if __name__ == "__main__":
    main()
