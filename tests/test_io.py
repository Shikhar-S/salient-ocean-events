"""Selection-table round-trip and alias-tolerant reading."""

import pytest

from src.core.types import Detections, Event
from src.io.selection_table import read_selection_table, write_selection_table


def test_round_trip(tmp_path):
    dets = [
        Detections(
            file_id="rec1",
            events=[Event(0.0, 1.0, 0.9), Event(2.5, 3.25, 0.4)],
            detector="stub",
        ),
        Detections(
            file_id="rec2",
            events=[Event(10.0, 11.5, 0.7)],
            detector="stub",
        ),
    ]
    path = tmp_path / "preds.tsv"
    write_selection_table(str(path), dets)
    back = read_selection_table(str(path))

    assert set(back) == {"rec1", "rec2"}
    assert len(back["rec1"]) == 2
    e = back["rec1"][0]
    assert e.onset_s == pytest.approx(0.0)
    assert e.offset_s == pytest.approx(1.0)
    assert e.score == pytest.approx(0.9)
    assert back["rec2"][0].offset_s == pytest.approx(11.5)


def test_reads_aliased_columns(tmp_path):
    # A foreign GT table using 'onset'/'offset'/'filename' and no score column.
    path = tmp_path / "gt.tsv"
    path.write_text(
        "filename\tonset\toffset\n"
        "recA\t0.10\t0.80\n"
        "recA\t1.00\t1.50\n"
    )
    gt = read_selection_table(str(path))
    assert list(gt) == ["recA"]
    assert len(gt["recA"]) == 2
    assert gt["recA"][0].score == 1.0  # default when absent
    assert gt["recA"][1].onset_s == pytest.approx(1.0)
