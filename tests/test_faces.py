from __future__ import annotations

import numpy as np
import pytest

from photo_sorter.constants import FACE_MATCH_THRESHOLD
from photo_sorter.faces import (
    count_and_people,
    default_match_threshold,
    detect_faces,
    match_embeddings,
    match_people,
    reference_embeddings,
    scale_yunet_rows,
    use_face_embedder,
)
from photo_sorter.media import ImageArray


def _solid() -> ImageArray:
    return np.broadcast_to(np.array([40, 80, 160], dtype=np.uint8), (64, 64, 3)).copy()


def test_match_people_does_not_tag_when_no_face_is_found() -> None:
    gallery = {"Myrto": [np.array([1.0, 0.0], dtype=np.float32)]}
    assert match_people(_solid(), gallery) == []


def test_match_people_tags_only_a_matching_reference_face() -> None:
    probe = np.array([1.0, 0.0], dtype=np.float32)
    other = np.array([0.0, 1.0], dtype=np.float32)
    use_face_embedder(lambda _image: [probe])
    try:
        assert match_people(_solid(), {"Myrto": [probe]}) == ["Myrto"]
        assert match_people(_solid(), {"Alex": [other]}) == []
    finally:
        use_face_embedder(None)


def test_match_embeddings_uses_best_exemplar_per_identity() -> None:
    probe = np.array([1.0, 0.0], dtype=np.float32)
    off_pose = np.array([0.0, 1.0], dtype=np.float32)
    # Myrto has one bad reference and one good one; the good one should win.
    samples = {"Myrto": [off_pose, probe], "Alex": [off_pose]}
    assert match_embeddings([probe], samples) == ["Myrto"]
    assert match_embeddings([off_pose], {"Alex": [probe]}) == []


def test_default_match_threshold_is_sface_scale_without_arcface() -> None:
    # ML is disabled in the test suite, so the classic SFace threshold applies.
    assert default_match_threshold() == FACE_MATCH_THRESHOLD


def test_count_and_people_uses_override_embeddings() -> None:
    probe = np.array([1.0, 0.0], dtype=np.float32)
    use_face_embedder(lambda _image: [probe, probe])
    try:
        count, people = count_and_people(_solid(), {"Myrto": [probe]})
    finally:
        use_face_embedder(None)
    assert count == 2
    assert people == ["Myrto"]


def test_match_embeddings_skips_empty_exemplar_lists() -> None:
    probe = np.array([1.0, 0.0], dtype=np.float32)
    assert match_embeddings([probe], {"Ghost": [], "Myrto": [probe]}) == ["Myrto"]


def test_reference_embeddings_keeps_largest_face_when_several_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    from photo_sorter.faces import FaceObservation

    small = np.array([0.0, 1.0], dtype=np.float32)
    large = np.array([1.0, 0.0], dtype=np.float32)

    def fake_analyze(_image: ImageArray, *, haar_fallback: bool = False) -> list[FaceObservation]:
        assert haar_fallback is True
        return [
            FaceObservation(bbox=(0, 0, 10, 10), embedding=small),
            FaceObservation(bbox=(0, 0, 80, 80), embedding=large),
        ]

    monkeypatch.setattr("photo_sorter.faces.analyze", fake_analyze)
    vectors = reference_embeddings(_solid())
    assert len(vectors) == 1
    np.testing.assert_allclose(vectors[0], large)


def test_detect_faces_ignores_haar_false_positives(monkeypatch: pytest.MonkeyPatch) -> None:
    from photo_sorter import faces

    monkeypatch.setattr(faces, "_yunet_faces", lambda _image: None)
    monkeypatch.setattr(faces, "_haar_boxes", lambda _image: [(2, 2, 40, 40)])
    assert detect_faces(_solid()) == []
    assert detect_faces(_solid(), haar_fallback=True)


def test_scale_yunet_rows_maps_resized_boxes_back() -> None:
    rows = np.array(
        [[10.0, 20.0, 30.0, 40.0, 12.0, 22.0, 28.0, 22.0, 20.0, 30.0, 14.0, 38.0, 26.0, 38.0, 0.9]],
        dtype=np.float32,
    )
    scaled = scale_yunet_rows(rows, 0.5)
    np.testing.assert_allclose(scaled[0, :4], [20.0, 40.0, 60.0, 80.0])
    np.testing.assert_allclose(scaled[0, 4:6], [24.0, 44.0])
    np.testing.assert_allclose(scaled[0, -1], 0.9)
