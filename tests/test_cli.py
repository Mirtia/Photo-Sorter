from pathlib import Path

import pytest

from photo_sorter.cli import build_parser, main, parse_arguments
from photo_sorter.constants import DEFAULT_QUALITY_THRESHOLD
from photo_sorter.exceptions import InvalidOptionError


def test_parse_arguments_backup() -> None:
    args = parse_arguments(["backup", "--input", "/photos", "--archive", "backup.tar.gz"])
    assert args.command == "backup"
    assert args.input == "/photos"
    assert args.archive == "backup.tar.gz"


def test_parse_arguments_enhance_flags() -> None:
    args = parse_arguments([
        "enhance",
        "--input",
        "deduped",
        "--output",
        "enhanced",
        "--quality-threshold",
        "0.4",
        "--apply-color-correction",
        "--apply-denoising",
        "strong",
        "--apply-sharpening",
        "--apply-artifact-removal",
        "--apply-upscaling",
        "light",
    ])
    assert args.quality_threshold == 0.4
    assert args.apply_color_correction is True
    assert args.apply_denoising == "strong"
    assert args.apply_sharpening == "medium"
    assert args.apply_artifact_removal is True
    assert args.apply_upscaling == "light"


def test_parse_arguments_defaults_quality_threshold() -> None:
    args = parse_arguments(["enhance", "--input", "in", "--output", "out"])
    assert args.quality_threshold == DEFAULT_QUALITY_THRESHOLD
    assert args.apply_denoising is None


def test_parse_arguments_from_phase_metadata() -> None:
    args = parse_arguments([
        "run",
        "--from-phase",
        "metadata",
        "--output",
        "./work",
        "--face-samples",
        "./samples",
    ])
    assert args.from_phase == "metadata"
    assert args.input is None


def test_parse_arguments_from_phase_numeric() -> None:
    args = parse_arguments([
        "run",
        "--from-phase",
        "3",
        "--output",
        "./work",
        "--discard",
        "./work/discarded",
        "--face-samples",
        "./samples",
    ])
    assert args.from_phase == "enhance"


def test_parse_arguments_from_phase_invalid() -> None:
    with pytest.raises(InvalidOptionError, match="from-phase"):
        parse_arguments(["run", "--from-phase", "thumbnails", "--output", "./work"])


def test_parse_arguments_faces_only_implies_metadata() -> None:
    args = parse_arguments([
        "run",
        "--faces-only",
        "--output",
        "./work",
        "--face-samples",
        "./samples",
    ])
    assert args.faces_only is True
    assert args.from_phase == "metadata"


def test_parse_arguments_metadata_faces_only() -> None:
    args = parse_arguments([
        "metadata",
        "--faces-only",
        "--input",
        "./work/enhanced",
        "--output",
        "./work/metadata",
        "--face-samples",
        "./samples",
    ])
    assert args.command == "metadata"
    assert args.faces_only is True


def test_parse_arguments_search_local() -> None:
    args = parse_arguments([
        "search",
        "boat on the water",
        "--metadata",
        "output/metadata",
        "--person",
        "Dimitris",
        "--object",
        "boat",
    ])
    assert args.command == "search"
    assert args.query == "boat on the water"
    assert args.person == "Dimitris"
    assert args.object_label == "boat"


def test_help_lists_phases() -> None:
    help_text = build_parser().format_help()
    assert "backup" in help_text
    assert "sort" in help_text
    assert "dedup" in help_text
    assert "enhance" in help_text
    assert "metadata" in help_text
    assert "index" in help_text
    assert "search" in help_text
    assert "run" in help_text
    assert "pull-models" in help_text
    assert "--config" in help_text


def test_main_backup_roundtrip(tmp_path: Path) -> None:
    source = tmp_path / "photos"
    source.mkdir()
    (source / "shot.jpg").write_bytes(b"data")
    archive = tmp_path / "backup.tar.gz"

    main(["backup", "--input", str(source), "--archive", str(archive)])

    assert archive.is_file()


def test_main_missing_input_exits_nonzero(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exited:
        main(["sort", "--input", str(tmp_path / "missing"), "--output", str(tmp_path / "out")])
    assert exited.value.code == 1


def test_main_without_command_exits_2() -> None:
    with pytest.raises(SystemExit) as exited:
        main([])
    assert exited.value.code == 2


def test_parse_arguments_pull_models() -> None:
    args = parse_arguments(["pull-models", "--yolo-model", "yolov8s.pt"])
    assert args.command == "pull-models"
    assert args.yolo_model == "yolov8s.pt"


def test_parse_arguments_upscaling_ml() -> None:
    args = parse_arguments(["enhance", "--input", "in", "--output", "out", "--apply-upscaling", "ml"])
    assert args.apply_upscaling == "ml"


def test_main_pull_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    from photo_sorter.models import PulledModels

    monkeypatch.setattr(
        "photo_sorter.cli.pull_models",
        lambda yolo_name=None: PulledModels(
            yolo=tmp_path / "yolov8n.pt",
            clip="ViT-B-32 (openai)",
            upscale=tmp_path / "realesr-general-x4v3.pth",
            face_detector=tmp_path / "face_detection_yunet_2023mar.onnx",
            face_recognizer=tmp_path / "face_recognition_sface_2021dec.onnx",
        ),
    )
    main(["pull-models"])
    output = capsys.readouterr().out
    assert "YOLO:" in output
    assert "CLIP: ViT-B-32 (openai)" in output
    assert "upscale:" in output
    assert "face detector:" in output
    assert "face recognizer:" in output


def test_main_pull_models_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    from photo_sorter.exceptions import MissingDependencyError

    def boom(yolo_name: str | None = None) -> None:
        raise MissingDependencyError("ultralytics")

    monkeypatch.setattr("photo_sorter.cli.pull_models", boom)
    with pytest.raises(SystemExit) as exited:
        main(["pull-models"])
    assert exited.value.code == 1
