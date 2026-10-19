from pathlib import Path

import pytest

from photo_sorter.progress import announce, progress_disabled, track, track_bytes


def test_track_yields_items_and_uses_file_names(tmp_path: Path) -> None:
    files = [tmp_path / "a.jpg", tmp_path / "b.jpg"]
    assert list(track(files, desc="sort")) == files


def test_track_bytes_yields_files(tmp_path: Path) -> None:
    path = tmp_path / "shot.jpg"
    path.write_bytes(b"abc")
    assert list(track_bytes([path], desc="backup")) == [path]


def test_progress_disabled_in_tests() -> None:
    assert progress_disabled() is True


def test_announce_is_quiet_when_disabled(capsys: pytest.CaptureFixture[str]) -> None:
    announce("backup", 1, 5)
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == ""
