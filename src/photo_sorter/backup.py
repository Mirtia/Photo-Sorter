"""Phase 0: backup the original directory into an archive."""

from __future__ import annotations

import tarfile
import zipfile
from pathlib import Path

from photo_sorter.exceptions import InvalidInputError
from photo_sorter.media import as_path
from photo_sorter.progress import track_bytes


def backup_input_directory_to_archive(
    input_directory: Path | str,
    backup_archive: Path | str,
) -> Path:
    """Compress ``input_directory`` into ``backup_archive``.

    Format is chosen from the destination suffix: ``.zip`` writes a zip file,
    everything else (including ``.tar.gz`` / ``.tgz``) writes a gzip tar.
    """
    source = as_path(input_directory)
    if not source.is_dir():
        raise InvalidInputError(source)

    archive = as_path(backup_archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    suffixes = [part.lower() for part in archive.suffixes]

    if suffixes[-1:] == [".zip"]:
        _write_zip(source, archive)
    else:
        _write_tar_gz(source, archive)
    return archive


def _files_under(source: Path) -> list[Path]:
    return sorted(path for path in source.rglob("*") if path.is_file())


def _write_tar_gz(source: Path, archive: Path) -> None:
    files = _files_under(source)
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname=source.name, recursive=False)
        for directory in sorted(path for path in source.rglob("*") if path.is_dir()):
            tar.add(directory, arcname=str(Path(source.name) / directory.relative_to(source)), recursive=False)
        for path in track_bytes(files, desc="backup"):
            tar.add(path, arcname=str(Path(source.name) / path.relative_to(source)), recursive=False)


def _write_zip(source: Path, archive: Path) -> None:
    files = _files_under(source)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        zipped.write(source, arcname=source.name)
        for directory in sorted(path for path in source.rglob("*") if path.is_dir()):
            zipped.write(directory, arcname=str(directory.relative_to(source.parent)))
        for path in track_bytes(files, desc="backup"):
            zipped.write(path, arcname=str(path.relative_to(source.parent)))
