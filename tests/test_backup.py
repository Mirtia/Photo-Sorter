import tarfile
import zipfile
from pathlib import Path

import pytest

from photo_sorter.backup import backup_input_directory_to_archive
from photo_sorter.exceptions import InvalidInputError


def test_backup_creates_tar_gz(tmp_path: Path) -> None:
    source = tmp_path / "photos"
    nested = source / "album"
    nested.mkdir(parents=True)
    (nested / "shot.jpg").write_bytes(b"jpeg-bytes")
    archive = tmp_path / "backup.tar.gz"

    result = backup_input_directory_to_archive(source, archive)

    assert result == archive
    assert archive.is_file()
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert any(name.endswith("shot.jpg") for name in names)


def test_backup_creates_zip(tmp_path: Path) -> None:
    source = tmp_path / "photos"
    source.mkdir()
    (source / "shot.jpg").write_bytes(b"jpeg-bytes")
    archive = tmp_path / "backup.zip"

    backup_input_directory_to_archive(source, archive)

    with zipfile.ZipFile(archive) as zipped:
        assert any(name.endswith("shot.jpg") for name in zipped.namelist())


def test_backup_rejects_missing_input(tmp_path: Path) -> None:
    with pytest.raises(InvalidInputError):
        backup_input_directory_to_archive(tmp_path / "missing", tmp_path / "out.tar.gz")
