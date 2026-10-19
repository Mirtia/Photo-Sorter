from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from photo_sorter.media import (
    capture_datetime,
    convert_png_to_jpg,
    is_image,
    is_png,
    is_video,
    md5_file,
    season_bin_name,
    season_for_month,
    unique_destination,
)
from tests.helpers import save_rgb, set_mtime


def test_season_for_month() -> None:
    assert season_for_month(1) == "winter"
    assert season_for_month(4) == "spring"
    assert season_for_month(7) == "summer"
    assert season_for_month(10) == "autumn"
    with pytest.raises(ValueError, match="Month must be"):
        season_for_month(0)


def test_season_bin_name() -> None:
    assert season_bin_name(datetime(2025, 4, 12)) == "spring_2025"
    assert season_bin_name(datetime(2025, 12, 1)) == "winter_2025"


def test_media_type_helpers() -> None:
    assert is_image(Path("a.PNG"))
    assert is_png(Path("a.PNG"))
    assert is_video(Path("clip.MP4"))
    assert not is_image(Path("notes.txt"))


def test_unique_destination_adds_numeric_suffix(tmp_path: Path) -> None:
    first = unique_destination(tmp_path, "img.jpg")
    first.write_bytes(b"one")
    second = unique_destination(tmp_path, "img.jpg")
    second.write_bytes(b"two")
    third = unique_destination(tmp_path, "img.jpg")

    assert first.name == "img.jpg"
    assert second.name == "img_1.jpg"
    assert third.name == "img_2.jpg"


def test_capture_datetime_falls_back_to_mtime(tmp_path: Path) -> None:
    path = save_rgb(tmp_path / "shot.jpg", (10, 20, 30))
    expected = datetime(2024, 7, 4, 12, 0, 0)
    set_mtime(path, expected)

    taken = capture_datetime(path)

    assert taken.year == 2024
    assert taken.month == 7


def test_capture_datetime_prefers_exif(tmp_path: Path) -> None:
    path = tmp_path / "dated.jpg"
    image = Image.new("RGB", (16, 16), (1, 2, 3))
    exif = image.getexif()
    exif[306] = "2025:03:18 09:15:00"
    image.save(path, format="JPEG", exif=exif)

    taken = capture_datetime(path)

    assert taken == datetime(2025, 3, 18, 9, 15, 0)


def test_convert_png_to_jpg_flattens_transparency(tmp_path: Path) -> None:
    source = tmp_path / "logo.png"
    Image.new("RGBA", (12, 12), (0, 255, 0, 0)).save(source)
    destination = tmp_path / "logo.jpg"

    convert_png_to_jpg(source, destination)

    assert destination.is_file()
    with Image.open(destination) as converted:
        assert converted.format == "JPEG"
        assert converted.mode == "RGB"


def test_md5_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"abc")
    assert md5_file(path) == md5_file(path)
    assert len(md5_file(path)) == 32
