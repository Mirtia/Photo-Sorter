from datetime import datetime
from pathlib import Path

import pytest
from PIL import Image

from photo_sorter.exceptions import InvalidOptionError
from photo_sorter.pipeline import PipelineConfig, parse_from_phase, run_pipeline
from tests.helpers import save_rgb, set_mtime


def test_run_pipeline_produces_all_phase_outputs(tmp_path: Path) -> None:
    incoming = tmp_path / "incoming"
    nested = incoming / "album"
    nested.mkdir(parents=True)
    png = nested / "lake.png"
    Image.new("RGB", (24, 24), (20, 80, 200)).save(png)
    set_mtime(png, datetime(2025, 4, 1, 9, 0, 0))
    save_rgb(nested / "lake_copy.jpg", (20, 80, 200), size=(24, 24))
    set_mtime(nested / "lake_copy.jpg", datetime(2025, 4, 1, 9, 0, 0))
    (nested / "movie.mp4").write_bytes(b"mp4")
    set_mtime(nested / "movie.mp4", datetime(2025, 4, 1, 9, 0, 0))

    samples = tmp_path / "samples" / "Alex"
    samples.mkdir(parents=True)
    save_rgb(samples / "face_1.jpg", (210, 180, 150), size=(32, 32))

    results = run_pipeline(
        PipelineConfig(
            input_directory=incoming,
            archive=tmp_path / "backup.tar.gz",
            output_directory=tmp_path / "work",
            discard_directory=tmp_path / "work" / "discarded",
            face_samples_directory=tmp_path / "samples",
            apply_color_correction=True,
            apply_denoising="light",
        )
    )

    assert results["archive"].is_file()
    assert any(results["sorted"].rglob("*.jpg"))
    assert any(results["deduped"].rglob("*.jpg"))
    assert any(results["enhanced"].rglob("*.jpg"))
    assert list(results["metadata"].glob("*_metadata.json"))
    assert any(results["deduped"].rglob("*.mp4"))


def test_parse_from_phase_accepts_names_and_indices() -> None:
    assert parse_from_phase(None) == "backup"
    assert parse_from_phase("4") == "metadata"
    assert parse_from_phase("Enhance") == "enhance"


def test_parse_from_phase_rejects_unknown() -> None:
    with pytest.raises(InvalidOptionError, match="from-phase"):
        parse_from_phase("thumbnails")


def test_run_pipeline_from_metadata_skips_earlier_phases(tmp_path: Path) -> None:
    enhanced = tmp_path / "work" / "enhanced" / "spring_2025" / "pictures"
    enhanced.mkdir(parents=True)
    save_rgb(enhanced / "aabbccddeeff00112233445566778899.jpg", (20, 180, 40), size=(64, 64))
    set_mtime(enhanced / "aabbccddeeff00112233445566778899.jpg", datetime(2025, 4, 9, 14, 0, 0))
    save_rgb(tmp_path / "samples" / "Alex.jpg", (210, 180, 150), size=(32, 32))

    results = run_pipeline(
        PipelineConfig(
            input_directory=tmp_path / "missing",
            archive=tmp_path / "unused.tar.gz",
            output_directory=tmp_path / "work",
            discard_directory=tmp_path / "work" / "discarded",
            face_samples_directory=tmp_path / "samples",
            from_phase="metadata",
        )
    )

    assert not (tmp_path / "unused.tar.gz").exists()
    assert not (tmp_path / "work" / "sorted").exists()
    assert list(results["metadata"].glob("*_metadata.json"))


def test_run_pipeline_faces_only_skips_regenerating_metadata(tmp_path: Path) -> None:
    import json

    import numpy as np

    from photo_sorter.faces import use_face_embedder

    enhanced = tmp_path / "work" / "enhanced" / "spring_2025" / "pictures"
    enhanced.mkdir(parents=True)
    photo = save_rgb(enhanced / "aabbccddeeff00112233445566778899.jpg", (20, 180, 40), size=(64, 64))
    metadata = tmp_path / "work" / "metadata"
    metadata.mkdir(parents=True)
    payload = {
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
        "tags": ["boat"],
        "description": "An outdoor photo showing a boat.",
        "embedding": [0.5, 0.25],
    }
    target = metadata / "aabbccddeeff00112233445566778899_metadata.json"
    target.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    save_rgb(tmp_path / "samples" / "Myrto.jpg", (210, 180, 150), size=(32, 32))
    vector = np.array([1.0, 0.0], dtype=np.float32)
    use_face_embedder(lambda _image: [vector])
    try:
        run_pipeline(
            PipelineConfig(
                input_directory=tmp_path / "missing",
                archive=tmp_path / "unused.tar.gz",
                output_directory=tmp_path / "work",
                discard_directory=tmp_path / "work" / "discarded",
                face_samples_directory=tmp_path / "samples",
                from_phase="enhance",
                faces_only=True,
            )
        )
    finally:
        use_face_embedder(None)

    assert not (tmp_path / "unused.tar.gz").exists()
    updated = json.loads(target.read_text(encoding="utf-8"))
    assert updated["people"] == ["Myrto"]
    assert updated["embedding"] == [0.5, 0.25]
    assert updated["objects"] == [{"name": "boat", "confidence": 0.8}]
