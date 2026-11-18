"""Headless tests for core ID-swap behavior."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from track_repair_ui import MotRow, MotTable


def row(frame: int, track_id: int, x: int) -> MotRow:
    return MotRow([str(frame), str(track_id), str(x), "0", "10", "20", "0.9", "-1", "-1", "-1"])


def test_swap_suffix_and_undo() -> None:
    table = MotTable([
        row(1, 10, 10), row(1, 20, 20),
        row(2, 10, 20), row(2, 20, 10),
        row(3, 10, 20), row(3, 20, 10),
    ])
    action = table.swap_suffix(10, 20, 2)
    assert [r.track_id for r in table.rows_at(1)] == [10, 20]
    assert [(r.track_id, r.xywh[0]) for r in table.rows_at(2)] == [(20, 20.0), (10, 10.0)]
    table.apply_action(action)
    assert [(r.track_id, r.xywh[0]) for r in table.rows_at(2)] == [(10, 20.0), (20, 10.0)]


def test_swap_requires_both_ids() -> None:
    table = MotTable([row(1, 1, 10), row(2, 1, 10)])
    original = [r.serialize() for r in table.rows]
    try:
        table.swap_suffix(1, 2, 2)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    assert [r.serialize() for r in table.rows] == original


if __name__ == "__main__":
    test_swap_suffix_and_undo()
    test_swap_requires_both_ids()
    print("core tests passed")
