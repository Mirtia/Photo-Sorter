from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from photo_sorter.constants import (
    DEFAULT_FACE_DETECTOR_MODEL,
    DEFAULT_FACE_RECOGNIZER_MODEL,
    DEFAULT_UPSCALE_MODEL,
)
from photo_sorter.exceptions import MissingDependencyError, ModelDownloadError
from photo_sorter.models import (
    ensure_clip_weights,
    ensure_upscale_weights,
    ensure_yolo_weights,
    pull_models,
    resolve_yolo_weights,
    yolo_model_name,
)


def test_yolo_model_name_prefers_explicit_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTO_SORTER_YOLO_MODEL", "from-env.pt")
    assert yolo_model_name("custom.pt") == "custom.pt"
    assert yolo_model_name() == "from-env.pt"


def test_resolve_yolo_weights_prefers_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "models"
    cache.mkdir()
    weights = cache / "yolov8n.pt"
    weights.write_bytes(b"cached")
    monkeypatch.setenv("PHOTO_SORTER_MODEL_DIR", str(cache))
    monkeypatch.delenv("PHOTO_SORTER_YOLO_MODEL", raising=False)
    assert resolve_yolo_weights() == str(weights)


def test_ensure_yolo_weights_skips_download_when_cached(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "models"
    cache.mkdir()
    weights = cache / "yolov8n.pt"
    weights.write_bytes(b"cached")
    monkeypatch.setenv("PHOTO_SORTER_MODEL_DIR", str(cache))
    monkeypatch.delenv("PHOTO_SORTER_YOLO_MODEL", raising=False)
    called: list[str] = []
    fake = types.ModuleType("ultralytics")
    fake.YOLO = lambda name: called.append(name) or object()  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    assert ensure_yolo_weights() == weights
    assert called == []


def test_ensure_yolo_weights_downloads_into_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "models"
    monkeypatch.setenv("PHOTO_SORTER_MODEL_DIR", str(cache))
    monkeypatch.delenv("PHOTO_SORTER_YOLO_MODEL", raising=False)

    def fake_yolo(name: str) -> object:
        Path(name).write_bytes(b"weights")
        return object()

    fake = types.ModuleType("ultralytics")
    fake.YOLO = fake_yolo  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake)

    dest = ensure_yolo_weights()
    assert dest == cache / "yolov8n.pt"
    assert dest.read_bytes() == b"weights"


def test_ensure_yolo_weights_missing_extra(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PHOTO_SORTER_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("PHOTO_SORTER_YOLO_MODEL", raising=False)
    monkeypatch.setitem(sys.modules, "ultralytics", None)
    with pytest.raises(MissingDependencyError):
        ensure_yolo_weights()


def test_ensure_yolo_weights_download_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTO_SORTER_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.delenv("PHOTO_SORTER_YOLO_MODEL", raising=False)

    def boom(_name: str) -> object:
        raise RuntimeError("offline")

    fake = types.ModuleType("ultralytics")
    fake.YOLO = boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ultralytics", fake)
    with pytest.raises(ModelDownloadError):
        ensure_yolo_weights()


def test_ensure_clip_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("open_clip")
    fake.create_model_and_transforms = lambda *_args, **_kwargs: (object(), None, object())
    fake.get_tokenizer = lambda _name: object()
    monkeypatch.setitem(sys.modules, "open_clip", fake)
    assert ensure_clip_weights() == "ViT-B-32 (openai)"


def test_ensure_upscale_weights_uses_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "models"
    cache.mkdir()
    weights = cache / DEFAULT_UPSCALE_MODEL
    weights.write_bytes(b"sr")
    monkeypatch.setenv("PHOTO_SORTER_MODEL_DIR", str(cache))
    monkeypatch.delenv("PHOTO_SORTER_UPSCALE_MODEL", raising=False)
    assert ensure_upscale_weights() == weights


def test_ensure_upscale_weights_downloads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "models"
    monkeypatch.setenv("PHOTO_SORTER_MODEL_DIR", str(cache))
    monkeypatch.delenv("PHOTO_SORTER_UPSCALE_MODEL", raising=False)

    class FakeResponse:
        def __init__(self) -> None:
            self._payload = b"sr-weights"
            self._offset = 0
            self.headers = {"Content-Length": str(len(self._payload))}

        def read(self, size: int = -1) -> bytes:
            if self._offset >= len(self._payload):
                return b""
            chunk = self._payload[self._offset :] if size < 0 else self._payload[self._offset : self._offset + size]
            self._offset += len(chunk)
            return chunk

        def __enter__(self) -> FakeResponse:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    dest = ensure_upscale_weights()
    assert dest == cache / DEFAULT_UPSCALE_MODEL
    assert dest.read_bytes() == b"sr-weights"


def test_ensure_upscale_weights_rejects_http(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from photo_sorter.models import _download_https

    with pytest.raises(ModelDownloadError):
        _download_https("http://example.com/model.pth", tmp_path / "x.pth")


def test_pull_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cache = tmp_path / "models"
    cache.mkdir()
    (cache / "yolov8n.pt").write_bytes(b"cached")
    monkeypatch.setenv("PHOTO_SORTER_MODEL_DIR", str(cache))
    monkeypatch.delenv("PHOTO_SORTER_YOLO_MODEL", raising=False)
    fake = types.ModuleType("open_clip")
    fake.create_model_and_transforms = lambda *_args, **_kwargs: (object(), None, object())
    fake.get_tokenizer = lambda _name: object()
    monkeypatch.setitem(sys.modules, "open_clip", fake)
    (cache / DEFAULT_UPSCALE_MODEL).write_bytes(b"sr")
    (cache / DEFAULT_FACE_DETECTOR_MODEL).write_bytes(b"yunet")
    (cache / DEFAULT_FACE_RECOGNIZER_MODEL).write_bytes(b"sface")
    pulled = pull_models()
    assert pulled.yolo == cache / "yolov8n.pt"
    assert pulled.clip == "ViT-B-32 (openai)"
    assert pulled.upscale == cache / DEFAULT_UPSCALE_MODEL
    assert pulled.face_detector == cache / DEFAULT_FACE_DETECTOR_MODEL
    assert pulled.face_recognizer == cache / DEFAULT_FACE_RECOGNIZER_MODEL
