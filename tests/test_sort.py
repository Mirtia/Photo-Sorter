from datetime import datetime
from pathlib import Path

from PIL import Image

from photo_sorter.sort import sort_input_directory
from tests.helpers import save_rgb, set_mtime


def test_sort_converts_png_flattens_and_bins_by_season(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    nested = source / "trip" / "day1"
    nested.mkdir(parents=True)
    png = nested / "view.png"
    Image.new("RGB", (20, 20), (0, 128, 255)).save(png)
    set_mtime(png, datetime(2025, 4, 2, 10, 0, 0))

    jpg = source / "portrait.jpg"
    save_rgb(jpg, (255, 0, 0))
    set_mtime(jpg, datetime(2025, 7, 20, 10, 0, 0))

    video = nested / "clip.mp4"
    video.write_bytes(b"fake-mp4")
    set_mtime(video, datetime(2025, 4, 3, 10, 0, 0))

    (source / "notes.txt").write_text("ignore me", encoding="utf-8")

    output = tmp_path / "sorted"
    sort_input_directory(source, output)

    spring_pictures = list((output / "spring_2025" / "pictures").iterdir())
    spring_videos = list((output / "spring_2025" / "videos").iterdir())
    summer_pictures = list((output / "summer_2025" / "pictures").iterdir())

    assert len(spring_pictures) == 1
    assert spring_pictures[0].suffix == ".jpg"
    assert spring_pictures[0].name == "view.jpg"
    assert [path.name for path in spring_videos] == ["clip.mp4"]
    assert [path.name for path in summer_pictures] == ["portrait.jpg"]
    assert (output / "spring_2025" / "videos").is_dir()
    assert (output / "summer_2025" / "videos").is_dir()


def test_sort_adds_suffix_on_name_collision(tmp_path: Path) -> None:
    source = tmp_path / "incoming"
    left = source / "a"
    right = source / "b"
    left.mkdir(parents=True)
    right.mkdir(parents=True)
    captured = datetime(2025, 1, 15, 8, 0, 0)
    for folder in (left, right):
        path = folder / "img.jpg"
        save_rgb(path, (10, 10, 10))
        set_mtime(path, captured)

    output = tmp_path / "sorted"
    sort_input_directory(source, output)

    names = sorted(path.name for path in (output / "winter_2025" / "pictures").iterdir())
    assert names == ["img.jpg", "img_1.jpg"]
