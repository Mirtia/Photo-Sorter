from __future__ import annotations

import pytest

from photo_sorter.coco import coco_label
from photo_sorter.device import force_cpu, onnx_providers, torch_device


def test_coco_label_known_and_unknown() -> None:
    assert coco_label(1) == "person"
    assert coco_label(88) == "teddy bear"
    assert coco_label(999) == "999"


def test_force_cpu_pins_torch_and_onnx(monkeypatch: pytest.MonkeyPatch) -> None:
    import photo_sorter.device as device

    monkeypatch.setenv("PHOTO_SORTER_FORCE_CPU", "1")
    device.cuda_available.cache_clear()
    device.torch_device.cache_clear()
    device.use_fp16.cache_clear()
    device.onnx_providers.cache_clear()
    device.insightface_ctx_id.cache_clear()
    try:
        assert force_cpu() is True
        assert torch_device() == "cpu"
        assert onnx_providers() == ("CPUExecutionProvider",)
    finally:
        device.cuda_available.cache_clear()
        device.torch_device.cache_clear()
        device.use_fp16.cache_clear()
        device.onnx_providers.cache_clear()
        device.insightface_ctx_id.cache_clear()
