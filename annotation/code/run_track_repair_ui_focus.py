"""Launcher that visually focuses the selected trajectory.

The selected track remains yellow. All other bboxes and labels are rendered as
semi-transparent gray while a trajectory is selected.
"""

from __future__ import annotations

import cv2

import track_repair_ui as base
from run_track_repair_ui import GapAwareTrackRepairUI


class FocusTrackRepairUI(GapAwareTrackRepairUI):
    muted_color = (170, 170, 170)
    muted_alpha = 0.38

    @staticmethod
    def _draw_box(layer, row, rect, color, thickness):
        x1, y1, x2, y2 = rect
        cv2.rectangle(layer, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            layer,
            f"ID {row.track_id} {row.confidence:.2f}",
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            1,
            cv2.LINE_AA,
        )

    def draw(self):
        frame = self.video.read(self.current_frame)
        canvas = cv2.resize(
            frame,
            (self.display_width, self.display_height),
            interpolation=cv2.INTER_AREA,
        )
        rows_with_rects = []
        for row in self.table.rows_at(self.current_frame):
            x, y, width, height = row.xywh
            rect = (
                round(x * self.scale),
                round(y * self.scale),
                round((x + width) * self.scale),
                round((y + height) * self.scale),
            )
            rows_with_rects.append((row, rect))
        self._last_drawn_rows = rows_with_rects

        # Draw muted tracks together on one overlay so overlapping gray boxes do
        # not repeatedly darken the image.
        muted_layer = canvas.copy()
        has_muted = False
        for row, rect in rows_with_rects:
            if self.selected_id is not None and row.track_id != self.selected_id:
                self._draw_box(muted_layer, row, rect, self.muted_color, 2)
                has_muted = True
        if has_muted:
            cv2.addWeighted(
                muted_layer,
                self.muted_alpha,
                canvas,
                1.0 - self.muted_alpha,
                0,
                canvas,
            )

        # With no active selection, retain per-ID colors. Draw the selected
        # trajectory last so it stays visually dominant over gray overlaps.
        if self.selected_id is None:
            for row, rect in rows_with_rects:
                self._draw_box(canvas, row, rect, base.id_color(row.track_id), 2)
        else:
            for row, rect in rows_with_rects:
                if row.track_id == self.selected_id:
                    self._draw_box(canvas, row, rect, (0, 255, 255), 4)

        overlay = canvas.copy()
        cv2.rectangle(overlay, (0, 0), (self.display_width, 76), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.68, canvas, 0.32, 0, canvas)
        selected = "-" if self.selected_id is None else str(self.selected_id)
        dirty = "UNSAVED" if self.dirty else "saved"
        lines = (
            f"frame {self.current_frame}/{self.max_frame} | mode {self.mode} | selected {selected} | {dirty}",
            self.message[:150],
            "N correct/next | B bad | W repair | click bbox | A/D +/-1 | J/L +/-10 | U undo | S save | Q quit",
        )
        for index, text in enumerate(lines):
            cv2.putText(
                canvas,
                text,
                (12, 21 + index * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55 if index != 2 else 0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        return canvas


if __name__ == "__main__":
    base.TrackRepairUI = FocusTrackRepairUI
    base.main()

