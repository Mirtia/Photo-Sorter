from pathlib import Path

import pytest

from photo_sorter.detect import DetectedObject, detect_objects, heuristic_objects, season_from_path, use_object_detector
from photo_sorter.media import load_image
from tests.helpers import save_rgb


def test_heuristic_objects_prefer_green_as_vegetation(tmp_path: Path) -> None:
    path = save_rgb(tmp_path / "field.jpg", (20, 180, 40), size=(80, 80))
    image = load_image(path)
    assert image is not None
    objects = heuristic_objects(image)
    assert objects
    assert objects[0].name in {"vegetation", "outdoor"}
    assert len(objects) <= 3


def test_detect_objects_uses_injected_detector(tmp_path: Path) -> None:
    path = save_rgb(tmp_path / "any.jpg", (10, 10, 10), size=(32, 32))
    image = load_image(path)
    assert image is not None

    def fake_detector(_image: object) -> list[DetectedObject]:
        return [
            DetectedObject(name="boat", confidence=0.91, bbox=(0.1, 0.2, 0.8, 0.9)),
            DetectedObject(name="person", confidence=0.7),
        ]

    use_object_detector(fake_detector)
    try:
        found = detect_objects(image)
    finally:
        use_object_detector(None)

    assert [item.name for item in found] == ["boat", "person"]
    assert found[0].bbox == (0.1, 0.2, 0.8, 0.9)


def test_detect_objects_skips_yolo_when_rfdetr_loaded(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from photo_sorter import detect

    path = save_rgb(tmp_path / "empty.jpg", (10, 10, 10), size=(32, 32))
    image = load_image(path)
    assert image is not None
    yolo_calls: list[int] = []

    monkeypatch.setattr(detect, "_rfdetr_model", lambda: object())
    monkeypatch.setattr(detect, "_rfdetr_objects", lambda _image, _limit: [])
    monkeypatch.setattr(detect, "_yolo_objects", lambda _image, _limit: yolo_calls.append(1) or [DetectedObject(name="boat", confidence=0.9)])
    found = detect.detect_objects(image)
    assert yolo_calls == []
    assert found
    assert found[0].name in {"vegetation", "outdoor", "indoor", "sky", "water", "snow", "sunset"}


def test_season_from_path() -> None:
    assert season_from_path(("work", "enhanced", "spring_2025", "pictures", "a.jpg")) == "spring_2025"
    assert season_from_path(("random", "folder")) is None
