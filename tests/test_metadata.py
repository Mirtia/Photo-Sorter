import json
from datetime import datetime
from pathlib import Path

import pytest

from photo_sorter.detect import DetectedObject
from photo_sorter.media import load_image
from photo_sorter.metadata import (
    apply_face_fields,
    build_description,
    count_objects,
    detect_topics,
    generate_metadata_directory,
    load_face_samples,
    orientation_label,
    refresh_face_metadata,
)
from photo_sorter.sort import PICTURES_DIR
from tests.helpers import save_rgb, set_mtime


def test_detect_topics_returns_at_most_three(tmp_path: Path) -> None:
    path = save_rgb(tmp_path / "blue.jpg", (30, 90, 220), size=(80, 80))
    image = load_image(path)
    assert image is not None
    topics = detect_topics(image)
    assert len(topics) <= 3
    assert all("name" in topic and "confidence" in topic for topic in topics)


def test_generate_metadata_writes_json_per_image(tmp_path: Path) -> None:
    season = tmp_path / "enhanced" / "spring_2025" / PICTURES_DIR
    season.mkdir(parents=True)
    photo = save_rgb(season / "aabbccddeeff00112233445566778899.jpg", (20, 180, 40), size=(64, 64))
    set_mtime(photo, datetime(2025, 4, 9, 14, 0, 0))

    samples = tmp_path / "samples" / "Jane_Doe"
    samples.mkdir(parents=True)
    save_rgb(samples / "face_1.jpg", (200, 180, 160), size=(48, 48))

    output = tmp_path / "metadata"
    generate_metadata_directory(tmp_path / "enhanced", output, tmp_path / "samples")

    files = list(output.glob("*_metadata.json"))
    assert len(files) == 1
    document = json.loads(files[0].read_text(encoding="utf-8"))
    assert document["filename"].endswith(".jpg")
    assert document["hash"] == "aabbccddeeff00112233445566778899"
    assert "date_taken" in document
    assert document["topics"]
    assert "objects" in document
    assert document["scene"] or document["object_labels"]
    assert "embedding" in document
    assert "people" in document
    assert document["resolution"] == {"width": 64, "height": 64}
    assert files[0].name == "aabbccddeeff00112233445566778899_metadata.json"
    assert document["description"]
    assert "taken in spring 2025" in document["description"]
    assert document["orientation"] == "square"
    assert "object_counts" in document
    assert "face_count" in document


def test_load_face_samples_skips_photos_without_faces(tmp_path: Path) -> None:
    save_rgb(tmp_path / "Myrto.jpg", (200, 180, 160), size=(48, 48))
    assert load_face_samples(tmp_path) == {}


def test_load_face_samples_uses_detected_embeddings(tmp_path: Path) -> None:
    import numpy as np

    from photo_sorter.faces import use_face_embedder

    save_rgb(tmp_path / "Myrto.jpg", (200, 180, 160), size=(48, 48))
    vector = np.array([1.0, 0.0], dtype=np.float32)
    use_face_embedder(lambda _image: [vector])
    try:
        samples = load_face_samples(tmp_path)
    finally:
        use_face_embedder(None)
    assert "Myrto" in samples
    np.testing.assert_allclose(samples["Myrto"][0], vector)


def test_count_objects_and_description_use_natural_language() -> None:
    assert count_objects(["person", "person", "car"]) == {"person": 2, "car": 1}
    assert orientation_label(1920, 1080) == "landscape"
    assert orientation_label(800, 1200) == "portrait"
    text = build_description(
        objects=[
            DetectedObject(name="person", confidence=0.9),
            DetectedObject(name="person", confidence=0.8),
            DetectedObject(name="cell phone", confidence=0.6),
        ],
        scene=[DetectedObject(name="indoor", confidence=0.4)],
        people=["Myrto"],
        face_count=2,
        orientation="portrait",
        lighting="daylight",
        caption=None,
        taken_phrase="taken in winter 2022",
    )
    assert text.startswith("An indoor photo showing two people and a cell phone")
    assert "featuring Myrto" in text
    assert text.endswith("taken in winter 2022.")


def test_description_uses_clip_caption_when_present() -> None:
    text = build_description(
        objects=[DetectedObject(name="umbrella", confidence=0.7)],
        scene=[DetectedObject(name="outdoor", confidence=0.5)],
        people=[],
        face_count=0,
        orientation="landscape",
        lighting="bright",
        caption="a beach scene",
        taken_phrase="taken in summer 2022",
    )
    assert text == "A beach scene showing an umbrella, taken in summer 2022."


def test_rank_scene_captions_picks_matching_prompt() -> None:
    from photo_sorter.embed import rank_scene_captions, use_embedders

    def fake_text(text: str) -> list[float]:
        return [1.0, 0.0] if "beach" in text else [0.0, 1.0]

    use_embedders(text_embedder=fake_text)
    try:
        ranked = rank_scene_captions([1.0, 0.0], limit=1)
    finally:
        use_embedders()
    assert ranked
    assert ranked[0][0] == "a beach scene"


def test_apply_face_fields_keeps_objects_and_rebuilds_description() -> None:
    document = {
        "object_labels": ["person", "cell phone"],
        "objects": [{"name": "person", "confidence": 0.9}, {"name": "cell phone", "confidence": 0.6}],
        "scene": [{"name": "indoor", "confidence": 0.4}],
        "orientation": "portrait",
        "lighting": "daylight",
        "season": "winter_2022",
        "embedding": [0.1, 0.2],
        "people": [],
        "face_count": 0,
        "tags": ["person", "cell phone", "indoor", "portrait", "daylight", "winter 2022"],
        "description": "An indoor photo showing a person and a cell phone, taken in winter 2022.",
    }
    apply_face_fields(document, ["Myrto"], 1)
    assert document["people"] == ["Myrto"]
    assert document["face_count"] == 1
    assert document["embedding"] == [0.1, 0.2]
    assert document["objects"][0]["name"] == "person"
    assert "featuring Myrto" in document["description"]
    assert document["tags"][-4:] == ["Myrto", "portrait", "daylight", "winter 2022"]


def test_refresh_face_metadata_patches_json_without_rewriting_embeddings(tmp_path: Path) -> None:
    import numpy as np

    from photo_sorter.faces import use_face_embedder

    pictures = tmp_path / "enhanced" / "spring_2025" / PICTURES_DIR
    pictures.mkdir(parents=True)
    photo = save_rgb(pictures / "aabbccddeeff00112233445566778899.jpg", (20, 180, 40), size=(64, 64))
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    document = {
        "filename": photo.name,
        "hash": "aabbccddeeff00112233445566778899",
        "path": str(photo),
        "season": "spring_2025",
        "object_labels": ["boat"],
        "objects": [{"name": "boat", "confidence": 0.8}],
        "scene": [{"name": "outdoor", "confidence": 0.5}],
        "orientation": "square",
        "lighting": "daylight",
        "people": [],
        "face_count": 0,
        "tags": ["boat", "outdoor", "square", "daylight", "spring 2025"],
        "description": "An outdoor photo showing a boat, taken in spring 2025.",
        "embedding": [0.42, 0.11],
    }
    target = metadata / "aabbccddeeff00112233445566778899_metadata.json"
    target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    save_rgb(tmp_path / "samples" / "Myrto.jpg", (200, 180, 160), size=(48, 48))
    vector = np.array([1.0, 0.0], dtype=np.float32)
    use_face_embedder(lambda _image: [vector])
    try:
        refresh_face_metadata(tmp_path / "enhanced", metadata, tmp_path / "samples")
    finally:
        use_face_embedder(None)
    updated = json.loads(target.read_text(encoding="utf-8"))
    assert updated["people"] == ["Myrto"]
    assert updated["embedding"] == [0.42, 0.11]
    assert updated["objects"] == [{"name": "boat", "confidence": 0.8}]
    assert "featuring Myrto" in updated["description"]
    assert "Myrto" in updated["tags"]


def test_refresh_face_metadata_requires_existing_json(tmp_path: Path) -> None:
    from photo_sorter.exceptions import InvalidOptionError

    (tmp_path / "enhanced").mkdir()
    (tmp_path / "metadata").mkdir()
    (tmp_path / "samples").mkdir()
    with pytest.raises(InvalidOptionError, match="faces-only"):
        refresh_face_metadata(tmp_path / "enhanced", tmp_path / "metadata", tmp_path / "samples")
