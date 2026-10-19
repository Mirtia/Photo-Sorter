"""Directory representation for file system navigation."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from photo_sorter.exceptions import InvalidInputError


@dataclass
class Directory:
    """A snapshot of a directory tree.

    ``files`` holds direct children that are files. Nested folders are stored
    in ``subdirectories``, so a walk of the tree is a flat list of files plus
    a recursive list of child :class:`Directory` objects.
    """

    path: Path
    name: str
    files: list[Path]
    subdirectories: list[Directory]

    @classmethod
    def from_path(cls, path: Path | str, *, recursive: bool = True) -> Directory:
        """Build a :class:`Directory` snapshot from an existing folder."""
        directory = Path(path).expanduser()
        if not directory.is_dir():
            raise InvalidInputError(directory)

        files: list[Path] = []
        subdirectories: list[Directory] = []
        for child in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if child.is_file():
                files.append(child)
            elif child.is_dir():
                if recursive:
                    subdirectories.append(cls.from_path(child, recursive=True))
                else:
                    subdirectories.append(cls(path=child, name=child.name, files=[], subdirectories=[]))
        return cls(path=directory, name=directory.name, files=files, subdirectories=subdirectories)

    def iter_files(self, *, recursive: bool = True) -> Iterator[Path]:
        """Yield files in this directory, optionally walking subdirectories."""
        yield from self.files
        if recursive:
            for subdirectory in self.subdirectories:
                yield from subdirectory.iter_files(recursive=True)

    def flatten_files(self) -> list[Path]:
        """Return every file under this directory as a flat list."""
        return list(self.iter_files(recursive=True))
