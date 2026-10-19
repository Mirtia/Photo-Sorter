from __future__ import annotations

from pathlib import Path

import pytest

from photo_sorter.arcface import _flatten_nested_pack
from photo_sorter.constants import ARCFACE_MODEL_NAME
from photo_sorter.models import insightface_root


def test_flatten_nested_antelopev2_pack(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOTO_SORTER_MODEL_DIR", str(tmp_path / "models"))
    root = insightface_root()
    nested = root / "models" / ARCFACE_MODEL_NAME / ARCFACE_MODEL_NAME
    nested.mkdir(parents=True)
    (nested / "scrfd_10g_bnkps.onnx").write_bytes(b"det")
    (nested / "glintr100.onnx").write_bytes(b"rec")
    _flatten_nested_pack(root)
    pack = root / "models" / ARCFACE_MODEL_NAME
    assert (pack / "scrfd_10g_bnkps.onnx").read_bytes() == b"det"
    assert (pack / "glintr100.onnx").read_bytes() == b"rec"
    assert not nested.exists()
