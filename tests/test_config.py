import os
from pathlib import Path

import pytest

from photo_sorter.cli import parse_arguments
from photo_sorter.config import flatten_config
from photo_sorter.constants import DEFAULT_QUALITY_THRESHOLD
from photo_sorter.exceptions import ConfigError, MissingOptionError


def test_flatten_nested_yaml_keys() -> None:
    flat = flatten_config({
        "paths": {
            "input": "/photos",
            "face-samples": "./samples",
            "enhanced": "./work/enhanced",
        },
        "dedup": {"enable_manual_selection": True},
        "enhance": {"apply_denoising": True, "quality_threshold": 0.4, "apply_upscaling": "ml"},
        "search": {"object": "boat", "es-url": "http://localhost:9200"},
        "ml": {"yolo_model": "yolov8s.pt"},
        "verbose": True,
    })
    assert flat["input"] == "/photos"
    assert flat["face_samples"] == "./samples"
    assert flat["images"] == "./work/enhanced"
    assert flat["enable_manual_selection"] is True
    assert flat["apply_denoising"] is True
    assert flat["apply_upscaling"] == "ml"
    assert flat["quality_threshold"] == 0.4
    assert flat["object_label"] == "boat"
    assert flat["es_url"] == "http://localhost:9200"
    assert flat["verbose"] is True
    assert flat["yolo_model"] == "yolov8s.pt"


def test_run_loads_options_from_config(tmp_path: Path) -> None:
    config = tmp_path / "photos.yaml"
    config.write_text(
        "\n".join([
            "paths:",
            "  input: /disk/photos",
            "  archive: ./backup.tar.gz",
            "  output: ./work",
            "  discard: ./work/discarded",
            "  face_samples: ./samples",
            "enhance:",
            "  apply_color_correction: true",
            "  apply_denoising: true",
            "  apply_sharpening: strong",
            "search:",
            "  es_url: http://localhost:9200",
            "  es_index: family",
            "",
        ]),
        encoding="utf-8",
    )

    args = parse_arguments(["run", "--config", str(config)])

    assert args.input == "/disk/photos"
    assert args.archive == "./backup.tar.gz"
    assert args.output == "./work"
    assert args.discard == "./work/discarded"
    assert args.face_samples == "./samples"
    assert args.apply_color_correction is True
    assert args.apply_denoising == "medium"
    assert args.apply_sharpening == "strong"
    assert args.es_url == "http://localhost:9200"
    assert args.es_index == "family"
    assert args.quality_threshold == DEFAULT_QUALITY_THRESHOLD


def test_cli_flag_overrides_config(tmp_path: Path) -> None:
    config = tmp_path / "photos.yaml"
    config.write_text("input: /from-yaml\narchive: ./a.tar.gz\n", encoding="utf-8")

    args = parse_arguments([
        "backup",
        "--config",
        str(config),
        "--input",
        "/from-cli",
    ])

    assert args.input == "/from-cli"
    assert args.archive == "./a.tar.gz"


def test_missing_required_option_after_config() -> None:
    with pytest.raises(MissingOptionError, match="input"):
        parse_arguments(["sort", "--output", "./out"])


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError):
        parse_arguments(["backup", "--config", str(tmp_path / "missing.yaml")])


def test_autoload_photo_sorter_yaml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHOTO_SORTER_DISABLE_AUTOCONFIG", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "photo-sorter.yaml").write_text(
        "input: /auto/photos\narchive: ./auto.tar.gz\n",
        encoding="utf-8",
    )

    args = parse_arguments(["backup"])

    assert args.input == "/auto/photos"
    assert args.archive == "./auto.tar.gz"


def test_yaml_ml_sets_yolo_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PHOTO_SORTER_YOLO_MODEL", raising=False)
    config = tmp_path / "photos.yaml"
    config.write_text("ml:\n  yolo_model: yolov8s.pt\n", encoding="utf-8")
    args = parse_arguments(["pull-models", "--config", str(config)])
    assert args.yolo_model == "yolov8s.pt"
    assert os.environ["PHOTO_SORTER_YOLO_MODEL"] == "yolov8s.pt"


def test_yaml_upscaling_ml(tmp_path: Path) -> None:
    config = tmp_path / "photos.yaml"
    config.write_text(
        "input: ./in\noutput: ./out\nenhance:\n  apply_upscaling: ml\n",
        encoding="utf-8",
    )
    args = parse_arguments(["enhance", "--config", str(config)])
    assert args.apply_upscaling == "ml"
