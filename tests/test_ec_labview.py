from pathlib import Path

import numpy as np
import pytest

from proespm.ec.ec_labview import _read_labview_data

testdata = Path(__file__).parent / "testdata" / "ec_labview"
cv_file = testdata / "CV_251010_001.csv"


def _data_lines(filepath: Path) -> list[str]:
    """Normalized, non-empty lines of a LabView file, header included."""

    text = (
        filepath.read_bytes()
        .replace(b"\r\n", b"\n")
        .replace(b"\r", b"\n")
        .decode("latin-1")
    )

    return [line for line in text.split("\n") if line.strip()]


def test_intact_cv_file():
    data, messages = _read_labview_data(cv_file)

    assert data.shape[1] == 9
    assert messages == []
    assert np.isfinite(data).all()


def test_truncated_row_is_dropped(tmp_path: Path):
    """An aborted acquisition leaves an incomplete final line."""

    lines = _data_lines(cv_file)
    lines[100] = "\t".join(lines[100].split("\t")[:8])

    broken = tmp_path / "CV_truncated.csv"
    _ = broken.write_text("\n".join(lines) + "\n")

    data, messages = _read_labview_data(broken)

    assert data.shape == (len(lines) - 2, 9)
    assert len(messages) == 1
    assert "line 101" in messages[0]


def test_empty_cell_becomes_nan(tmp_path: Path):
    """An empty cell must not silently shift the columns."""

    lines = _data_lines(cv_file)
    fields = lines[100].split("\t")
    fields[3] = ""
    lines[100] = "\t".join(fields)

    broken = tmp_path / "CV_empty_cell.csv"
    _ = broken.write_text("\n".join(lines) + "\n")

    data, messages = _read_labview_data(broken)

    assert data.shape == (len(lines) - 1, 9)
    assert np.isnan(data[99, 3])
    assert np.count_nonzero(np.isnan(data)) == 1
    assert len(messages) == 1


def test_header_only_file_raises(tmp_path: Path):
    empty = tmp_path / "CV_header_only.csv"
    _ = empty.write_text(_data_lines(cv_file)[0] + "\n")

    with pytest.raises(ValueError):
        _ = _read_labview_data(empty)
