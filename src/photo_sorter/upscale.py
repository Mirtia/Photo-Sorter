"""Image upscaling: Lanczos by level, or Real-ESRGAN when ``ml`` is selected."""

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import lru_cache
from typing import Any

import cv2
import numpy as np

from photo_sorter.detect import ml_enabled
from photo_sorter.media import ImageArray, as_uint8

logger = logging.getLogger(__name__)

_UPSCALE_TARGET = {"light": 720, "medium": 1080, "strong": 1440, "ml": 1440}
_MAX_UPSCALE = 4.0
_SR_TILE = 256
_SR_OVERLAP = 16

MlUpscaler = Callable[[ImageArray], ImageArray]
_upscaler_override: MlUpscaler | None = None


def use_ml_upscaler(upscaler: MlUpscaler | None) -> None:
    """Override the Real-ESRGAN upscaler (used in tests). Pass None to restore."""
    global _upscaler_override
    _upscaler_override = upscaler


def upscale_target(level: str) -> int:
    """Minimum side length this upscaling level aims for."""
    return _UPSCALE_TARGET.get(level, _UPSCALE_TARGET["strong"])


def upscale_factor(image: ImageArray, level: str) -> float:
    """Return the scale factor to apply (1.0 means leave the image unchanged)."""
    min_side = min(image.shape[:2])
    target = upscale_target(level)
    if min_side >= target:
        return 1.0
    return float(min(target / max(min_side, 1), _MAX_UPSCALE))


def _lanczos(image: ImageArray, scale: float) -> ImageArray:
    if scale <= 1.0:
        return image
    return as_uint8(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4))


def _resize_to_shape(image: ImageArray, height: int, width: int) -> ImageArray:
    if image.shape[0] == height and image.shape[1] == width:
        return image
    interpolation = cv2.INTER_AREA if image.shape[0] > height else cv2.INTER_LANCZOS4
    return as_uint8(cv2.resize(image, (width, height), interpolation=interpolation))


def _output_size(image: ImageArray, scale: float) -> tuple[int, int]:
    height, width = image.shape[:2]
    return max(round(height * scale), 1), max(round(width * scale), 1)


@lru_cache(maxsize=1)
def _sr_bundle() -> tuple[Any, Any] | None:
    if not ml_enabled():
        return None
    try:
        import torch
        from spandrel import ImageModelDescriptor, ModelLoader
    except ImportError:
        logger.info("spandrel/torch is not installed; using Lanczos upscaling")
        return None
    try:
        from photo_sorter.models import ensure_upscale_weights

        loaded = ModelLoader().load_from_file(str(ensure_upscale_weights()))
    except Exception:
        logger.warning("Could not load Real-ESRGAN weights; using Lanczos upscaling")
        return None
    if not isinstance(loaded, ImageModelDescriptor):
        return None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return loaded.to(device).eval(), device


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if length <= tile:
        return [0]
    step = max(tile - overlap, 1)
    starts = list(range(0, length - tile + 1, step))
    last = length - tile
    if starts[-1] != last:
        starts.append(last)
    return starts


def _sr_forward(model: Any, device: Any, rgb: ImageArray) -> ImageArray:
    import torch

    tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float().div_(255.0).unsqueeze(0)
    tensor = tensor.to(device)
    with torch.no_grad():
        output = model(tensor)
    array = output.squeeze(0).clamp(0, 1).mul(255).permute(1, 2, 0).cpu().numpy()
    return as_uint8(array)


def _sr_infer(model: Any, device: Any, bgr: ImageArray) -> ImageArray:
    rgb = as_uint8(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    height, width = rgb.shape[:2]
    scale = int(getattr(model, "scale", 4))
    if height <= _SR_TILE and width <= _SR_TILE:
        return as_uint8(cv2.cvtColor(_sr_forward(model, device, rgb), cv2.COLOR_RGB2BGR))

    out_h, out_w = height * scale, width * scale
    accumulated = np.zeros((out_h, out_w, 3), dtype=np.float32)
    weight = np.zeros((out_h, out_w, 1), dtype=np.float32)
    for top in _tile_starts(height, _SR_TILE, _SR_OVERLAP):
        for left in _tile_starts(width, _SR_TILE, _SR_OVERLAP):
            bottom = min(top + _SR_TILE, height)
            right = min(left + _SR_TILE, width)
            tile = rgb[top:bottom, left:right]
            super_tile = _sr_forward(model, device, tile).astype(np.float32)
            y0, x0 = top * scale, left * scale
            y1, x1 = y0 + super_tile.shape[0], x0 + super_tile.shape[1]
            accumulated[y0:y1, x0:x1] += super_tile
            weight[y0:y1, x0:x1] += 1.0
    blended = accumulated / np.maximum(weight, 1e-6)
    return as_uint8(cv2.cvtColor(as_uint8(blended), cv2.COLOR_RGB2BGR))


def _try_ml_upscale(image: ImageArray, scale: float) -> ImageArray | None:
    out_h, out_w = _output_size(image, scale)
    if _upscaler_override is not None:
        return _resize_to_shape(_upscaler_override(image), out_h, out_w)
    bundle = _sr_bundle()
    if bundle is None:
        return None
    model, device = bundle
    try:
        super_res = _sr_infer(model, device, image)
    except Exception:
        logger.warning("Real-ESRGAN failed; using Lanczos upscaling")
        return None
    return _resize_to_shape(super_res, out_h, out_w)


def upscale_image(image: ImageArray, level: str) -> ImageArray:
    """Enlarge ``image`` toward the target for ``level``. ``ml`` uses Real-ESRGAN."""
    scale = upscale_factor(image, level)
    if scale <= 1.0:
        return image
    if level == "ml":
        result = _try_ml_upscale(image, scale)
        if result is not None:
            return result
        logger.info("Falling back to Lanczos upscaling")
    return _lanczos(image, scale)
