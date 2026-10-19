from pathlib import Path

from photo_sorter.picker import default_selection, gui_enabled, selected_paths


def test_gui_disabled_in_tests() -> None:
    assert gui_enabled() is False


def test_default_selection_checks_the_largest_file(tmp_path: Path) -> None:
    small = tmp_path / "small.jpg"
    large = tmp_path / "large.jpg"
    small.write_bytes(b"aa")
    large.write_bytes(b"bbbbbb")
    assert default_selection([small, large]) == [False, True]
    assert selected_paths([small, large], [False, False]) == [large]
    assert selected_paths([small, large], [True, False]) == [small]
