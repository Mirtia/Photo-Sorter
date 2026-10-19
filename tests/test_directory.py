from pathlib import Path

import pytest

from photo_sorter.directory import Directory
from photo_sorter.exceptions import InvalidInputError


def test_from_path_lists_files_and_nested_directories(tmp_path: Path) -> None:
    nested = tmp_path / "album" / "day1"
    nested.mkdir(parents=True)
    (tmp_path / "root.jpg").write_bytes(b"root")
    (nested / "shot.jpg").write_bytes(b"shot")

    tree = Directory.from_path(tmp_path)

    assert tree.name == tmp_path.name
    assert [path.name for path in tree.files] == ["root.jpg"]
    assert tree.subdirectories[0].name == "album"
    assert [path.name for path in tree.flatten_files()] == ["root.jpg", "shot.jpg"]


def test_from_path_rejects_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError):
        Directory.from_path(tmp_path / "missing")
