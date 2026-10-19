"""Phase 1: flatten, convert PNGs, and bin media by season and year."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

from photo_sorter.directory import Directory
from photo_sorter.exceptions import InvalidInputError
from photo_sorter.media import (
    as_path,
    capture_datetime,
    convert_png_to_jpg,
    is_image,
    is_png,
    is_video,
    season_bin_name,
    unique_destination,
)
from photo_sorter.progress import track

logger = logging.getLogger(__name__)

PICTURES_DIR = "pictures"
VIDEOS_DIR = "videos"


def sort_input_directory(input_directory: Path | str, output_directory: Path | str) -> Path:
    """Convert PNGs, flatten nested folders, and bin files by season and year."""
    source = as_path(input_directory)
    if not source.is_dir():
        raise InvalidInputError(source)

    destination = as_path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    tree = Directory.from_path(source)
    for file_path in track(tree.flatten_files(), desc="sort"):
        if is_image(file_path):
            _place_picture(file_path, destination)
        elif is_video(file_path):
            _place_video(file_path, destination)
        else:
            logger.debug("Skipping unsupported file: %s", file_path)
    return destination


def _bin_directories(output_directory: Path, captured_name: str) -> tuple[Path, Path]:
    bin_dir = output_directory / captured_name
    pictures = bin_dir / PICTURES_DIR
    videos = bin_dir / VIDEOS_DIR
    pictures.mkdir(parents=True, exist_ok=True)
    videos.mkdir(parents=True, exist_ok=True)
    return pictures, videos


def _place_picture(file_path: Path, output_directory: Path) -> None:
    bin_name = season_bin_name(capture_datetime(file_path))
    pictures, _ = _bin_directories(output_directory, bin_name)
    if is_png(file_path):
        target = unique_destination(pictures, f"{file_path.stem}.jpg")
        convert_png_to_jpg(file_path, target)
        return
    target = unique_destination(pictures, file_path.name)
    shutil.copy2(file_path, target)


def _place_video(file_path: Path, output_directory: Path) -> None:
    bin_name = season_bin_name(capture_datetime(file_path))
    _, videos = _bin_directories(output_directory, bin_name)
    target = unique_destination(videos, file_path.name)
    shutil.copy2(file_path, target)
