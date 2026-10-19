"""Object detection: RF-DETR, then YOLO, then colour heuristics."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from photo_sorter.constants import (
    DEFAULT_RFDETR_MODEL,
    RFDETR_CONFIDENCE_THRESHOLD,
    SEASON_NAMES,
    TOP_OBJECT_COUNT,
    YOLO_CONFIDENCE_THRESHOLD,
)
from photo_sorter.media import ImageArray, as_uint8

logger = logging.getLogger(__name__)

Detector = Callable[[ImageArray], list["DetectedObject"]]
SCENE_LABELS = frozenset({"sky", "vegetation", "water", "indoor", "sunset", "snow", "outdoor"})


@dataclass(frozen=True)
class DetectedObject:
    """A labelled region in an image."""

    name: str
    confidence: float
    bbox: tuple[float, float, float, float] | None = None

    def to_dict(self) -> dict[str, object]:
        """JSON-serialisable form used in metadata and Elasticsearch."""
        payload: dict[str, object] = {
            "name": self.name,
            "confidence": round(self.confidence, 3),
        }
        if self.bbox is not None:
            payload["bbox"] = [round(value, 4) for value in self.bbox]
        return payload


def ml_enabled() -> bool:
    """Return False when PHOTO_SORTER_DISABLE_ML is set (tests, offline)."""
    flag = os.environ.get("PHOTO_SORTER_DISABLE_ML", "").strip().lower()
    return flag not in {"1", "true", "yes"}


def _ratio(mask: NDArray[np.bool_]) -> float:
    if mask.size == 0:
        return 0.0
    return float(mask.mean())


def heuristic_objects(image: ImageArray, limit: int = TOP_OBJECT_COUNT) -> list[DetectedObject]:
    """Guess scene-level labels from colour and layout (no neural net)."""
    height, _width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    upper = slice(0, max(height // 2, 1))
    lower = slice(max(height // 2, 1), height)
    sky = _ratio((hue[upper] >= 90) & (hue[upper] <= 130) & (saturation[upper] > 40) & (value[upper] > 80))
    vegetation = _ratio((hue >= 35) & (hue <= 85) & (saturation > 40) & (value > 40))
    water = _ratio((hue[lower] >= 90) & (hue[lower] <= 130) & (saturation[lower] > 50) & (value[lower] > 50))
    indoor = _ratio((value < 80) | ((saturation < 40) & (value < 180)))
    sunset = _ratio(((hue <= 20) | (hue >= 160)) & (saturation > 80) & (value > 80))
    snow = _ratio((saturation < 30) & (value > 200))
    candidates = [
        ("sky", sky),
        ("vegetation", vegetation),
        ("water", water),
        ("indoor", indoor * 0.6),
        ("sunset", sunset),
        ("snow", snow),
        ("outdoor", max(sky, vegetation, water) * 0.8),
    ]
    ranked = [
        DetectedObject(name=name, confidence=round(float(score), 3))
        for name, score in sorted(candidates, key=lambda item: item[1], reverse=True)
        if score > 0.05
    ]
    return ranked[:limit]


def _normalised_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    return (
        max(x1 / max(width, 1), 0.0),
        max(y1 / max(height, 1), 0.0),
        min(x2 / max(width, 1), 1.0),
        min(y2 / max(height, 1), 1.0),
    )


@lru_cache(maxsize=1)
def _yolo_model() -> Any | None:
    if not ml_enabled():
        return None
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.info("ultralytics is not installed; using colour heuristics for objects")
        return None
    from photo_sorter.models import ensure_yolo_weights, yolo_model_name

    try:
        return YOLO(str(ensure_yolo_weights()))
    except Exception:
        logger.warning("Could not load YOLO model %s; using colour heuristics", yolo_model_name())
        return None


def _yolo_objects(image: ImageArray, limit: int) -> list[DetectedObject]:
    model = _yolo_model()
    if model is None:
        return []
    rgb = as_uint8(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    height, width = image.shape[:2]
    predict = getattr(model, "predict", None)
    if predict is None:
        return []
    results = predict(rgb, verbose=False, conf=YOLO_CONFIDENCE_THRESHOLD)
    found: list[DetectedObject] = []
    for result in results:
        names = getattr(result, "names", {})
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            continue
        for box in boxes:
            class_id = int(box.cls[0])
            label = str(names.get(class_id, class_id))
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            found.append(
                DetectedObject(
                    name=label,
                    confidence=confidence,
                    bbox=_normalised_bbox(x1, y1, x2, y2, width, height),
                )
            )
    found.sort(key=lambda item: item.confidence, reverse=True)
    return found[:limit]


_RFDETR_FACTORIES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


def _rfdetr_model_name() -> str:
    configured = os.environ.get("PHOTO_SORTER_RFDETR_MODEL", "").strip().lower()
    if configured in _RFDETR_FACTORIES:
        return configured
    return DEFAULT_RFDETR_MODEL


@lru_cache(maxsize=1)
def _rfdetr_model() -> Any | None:
    if not ml_enabled():
        return None
    try:
        import rfdetr
    except ImportError:
        logger.info("rfdetr is not installed; falling back to YOLO/heuristics for objects")
        return None
    from photo_sorter.device import torch_device, use_fp16

    name = _rfdetr_model_name()
    try:
        model = getattr(rfdetr, _RFDETR_FACTORIES[name])(device=torch_device())
    except Exception:
        logger.warning("Could not load RF-DETR model %s; falling back to YOLO/heuristics", name)
        return None
    try:
        if use_fp16():
            import torch

            model.optimize_for_inference(dtype=torch.float16)
        else:
            model.optimize_for_inference()
    except Exception:
        logger.debug("RF-DETR inference optimization unavailable; using eager mode")
    logger.info("Object backend: RF-DETR %s on %s", name, torch_device())
    return model


def _rfdetr_objects(image: ImageArray, limit: int) -> list[DetectedObject]:
    model = _rfdetr_model()
    if model is None:
        return []
    from photo_sorter.coco import coco_label

    rgb = as_uint8(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    height, width = image.shape[:2]
    try:
        detections = model.predict(rgb, threshold=RFDETR_CONFIDENCE_THRESHOLD)
    except Exception:
        logger.debug("RF-DETR prediction failed", exc_info=True)
        return []
    boxes = getattr(detections, "xyxy", None)
    class_ids = getattr(detections, "class_id", None)
    confidences = getattr(detections, "confidence", None)
    if boxes is None or class_ids is None or confidences is None:
        return []
    found: list[DetectedObject] = []
    for (x1, y1, x2, y2), class_id, confidence in zip(boxes, class_ids, confidences):
        found.append(
            DetectedObject(
                name=coco_label(int(class_id)),
                confidence=float(confidence),
                bbox=_normalised_bbox(float(x1), float(y1), float(x2), float(y2), width, height),
            )
        )
    found.sort(key=lambda item: item.confidence, reverse=True)
    return found[:limit]


_detector_override: Detector | None = None


def use_object_detector(detector: Detector | None) -> None:
    """Override the detector (used in tests). Pass None to restore defaults."""
    global _detector_override
    _detector_override = detector


def detect_objects(image: ImageArray, limit: int = TOP_OBJECT_COUNT) -> list[DetectedObject]:
    """Return the top objects in ``image``.

    Prefers RF-DETR when it loaded, then YOLO, then colour/layout heuristics.
    An empty RF-DETR result is treated as "no objects" (a landscape is valid)
    and only heuristics fill scene-level tags — YOLO is not a second guess.
    """
    if _detector_override is not None:
        return _detector_override(image)[:limit]
    if _rfdetr_model() is not None:
        return _rfdetr_objects(image, limit) or heuristic_objects(image, limit)
    detected = _yolo_objects(image, limit)
    if detected:
        return detected
    return heuristic_objects(image, limit)


def season_from_path(path_parts: tuple[str, ...] | list[str]) -> str | None:
    """Return a ``spring_2025``-style bin name if one appears in the path."""
    for part in path_parts:
        season, separator, year = part.partition("_")
        if separator and season in SEASON_NAMES and year.isdigit():
            return part
    return None
