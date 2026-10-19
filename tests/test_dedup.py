from pathlib import Path

import pytest

from photo_sorter import manifest
from photo_sorter.dedup import deduplicate_sorted_directory, group_similar_pictures
from photo_sorter.sort import sort_input_directory
from tests.helpers import save_rgb


def _season_dir(root: Path) -> Path:
    return next(path for path in sorted(root.iterdir()) if path.is_dir())


def _sorted_season(tmp_path: Path) -> Path:
    incoming = tmp_path / "incoming"
    incoming.mkdir()
    save_rgb(incoming / "dup_a.jpg", (200, 10, 10), size=(40, 40))
    save_rgb(incoming / "dup_b.jpg", (200, 10, 10), size=(40, 40))
    save_rgb(incoming / "other.jpg", (10, 200, 10), size=(64, 64))
    (incoming / "clip.mp4").write_bytes(b"video")
    sorted_dir = tmp_path / "sorted"
    sort_input_directory(incoming, sorted_dir)
    return sorted_dir


def test_group_similar_pictures_clusters_near_duplicates(tmp_path: Path) -> None:
    save_rgb(tmp_path / "a.jpg", (200, 10, 10), size=(32, 32))
    save_rgb(tmp_path / "b.jpg", (200, 10, 10), size=(32, 32))
    save_rgb(tmp_path / "c.jpg", (10, 10, 200), size=(32, 32))

    groups = group_similar_pictures(list(tmp_path.glob("*.jpg")))
    sizes = sorted(len(group) for group in groups)
    assert sizes == [1, 2]


def test_dedup_keeps_largest_and_discards_the_rest(tmp_path: Path) -> None:
    sorted_dir = _sorted_season(tmp_path)
    output = tmp_path / "deduped"
    discard = tmp_path / "discard"

    deduplicate_sorted_directory(sorted_dir, output, discard, temp_directory=tmp_path / "work")

    pictures = list((_season_dir(output) / "pictures").iterdir())
    discarded = list(next(discard.iterdir()).iterdir()) if any(discard.iterdir()) else []
    videos = list((_season_dir(output) / "videos").iterdir())

    assert len(pictures) == 2
    assert len(discarded) == 1
    assert [path.name for path in videos] == ["clip.mp4"]


def test_dedup_writes_lineage_manifest(tmp_path: Path) -> None:
    sorted_dir = _sorted_season(tmp_path)
    output = tmp_path / "deduped"
    discard = tmp_path / "discard"

    deduplicate_sorted_directory(sorted_dir, output, discard, temp_directory=tmp_path / "work")

    records = manifest.read_records(output)
    assert len(records) == 3
    assert all(record["phase"] == "dedup" for record in records)
    kept = [record for record in records if record["kept"]]
    dropped = [record for record in records if not record["kept"]]
    assert len(kept) == 2
    assert len(dropped) == 1
    dup_group = {record["group_id"] for record in records if record["original_name"].startswith("dup_")}
    assert len(dup_group) == 1
    for record in records:
        assert record["phash"]
        assert Path(record["output_path"]).is_file()

    lineage = manifest.load_dedup_lineage(output)
    assert len(lineage) == 2


def test_dedup_manual_selector_can_keep_all(tmp_path: Path) -> None:
    sorted_dir = _sorted_season(tmp_path)
    output = tmp_path / "deduped"
    discard = tmp_path / "discard"

    deduplicate_sorted_directory(
        sorted_dir,
        output,
        discard,
        enable_manual_selection=True,
        selector=lambda group: list(group),
        temp_directory=tmp_path / "work",
    )

    pictures = list((_season_dir(output) / "pictures").iterdir())
    assert len(pictures) == 3
    assert not any(discard.rglob("*.jpg"))


def test_dedup_manual_selection_uses_gui_picker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from photo_sorter import dedup as dedup_mod

    sorted_dir = _sorted_season(tmp_path)
    output = tmp_path / "deduped"
    discard = tmp_path / "discard"
    seen: list[int] = []

    def fake_pick(group: list[Path]) -> list[Path]:
        seen.append(len(group))
        return [max(group, key=lambda path: path.stat().st_size)]

    monkeypatch.setattr(dedup_mod, "gui_enabled", lambda: True)
    monkeypatch.setattr(dedup_mod, "pick_duplicates", fake_pick)

    deduplicate_sorted_directory(
        sorted_dir,
        output,
        discard,
        enable_manual_selection=True,
        temp_directory=tmp_path / "work",
    )

    assert 2 in seen
    pictures = list((_season_dir(output) / "pictures").iterdir())
    assert len(pictures) == 2
