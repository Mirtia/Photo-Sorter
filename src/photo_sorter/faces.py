"""Detect faces and match them against reference photos of known people."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from photo_sorter import arcface
from photo_sorter.constants import (
    ARCFACE_MATCH_THRESHOLD,
    ARCFACE_MODEL_NAME,
    FACE_MATCH_THRESHOLD,
    SFACE_ALIGN_SIZE,
    YUNET_DETECT_MAX_SIDE,
    YUNET_NMS_THRESHOLD,
    YUNET_SCORE_THRESHOLD,
    YUNET_TOP_K,
)
from photo_sorter.detect import ml_enabled
from photo_sorter.media import ImageArray, as_uint8

logger = logging.getLogger(__name__)

Embedding = NDArray[np.float32]
FaceEmbedder = Callable[[ImageArray], list[Embedding]]


@dataclass(frozen=True)
class FaceObservation:
    """A single detected face: optional pixel box plus its embedding."""

    bbox: tuple[int, int, int, int] | None
    embedding: Embedding | None


_embedder_override: FaceEmbedder | None = None


def _use_arcface() -> bool:
    """True when the InsightFace ArcFace backend is available for this run."""
    return ml_enabled() and arcface.available()


def default_match_threshold() -> float:
    """Recognition threshold for the active backend (ArcFace vs SFace scale)."""
    return ARCFACE_MATCH_THRESHOLD if _use_arcface() else FACE_MATCH_THRESHOLD


def backend_name() -> str:
    """Short label for logs: which face detector/recognizer is active."""
    if _use_arcface():
        return f"InsightFace {ARCFACE_MODEL_NAME}"
    if ml_enabled():
        return "YuNet/SFace"
    return "disabled"


def use_face_embedder(embedder: FaceEmbedder | None) -> None:
    """Override face embedding (used in tests). Pass None to restore defaults."""
    global _embedder_override
    _embedder_override = embedder


def _normalise(vector: NDArray[np.floating[Any]]) -> Embedding:
    flat = np.asarray(vector, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(flat))
    if norm == 0:
        return flat
    return np.asarray(flat / norm, dtype=np.float32)


def _cosine(left: Embedding, right: Embedding) -> float:
    length = min(left.size, right.size)
    if length == 0:
        return 0.0
    return float(np.dot(left[:length], right[:length]))


@lru_cache(maxsize=1)
def _yunet() -> Any | None:
    if not ml_enabled():
        return None
    try:
        from photo_sorter.models import ensure_face_detector_weights

        path = ensure_face_detector_weights()
        return cv2.FaceDetectorYN.create(
            str(path),
            "",
            (320, 320),
            YUNET_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K,
        )
    except Exception:
        logger.warning("YuNet face detector is unavailable")
        return None


@lru_cache(maxsize=1)
def _sface() -> Any | None:
    if not ml_enabled():
        return None
    try:
        from photo_sorter.models import ensure_face_recognizer_weights

        path = ensure_face_recognizer_weights()
        return cv2.FaceRecognizerSF.create(str(path), "")
    except Exception:
        logger.warning("SFace recognizer is unavailable; people will not be tagged")
        return None


@lru_cache(maxsize=1)
def _haar() -> cv2.CascadeClassifier:
    haar_dir = cv2.data.haarcascades  # type: ignore[attr-defined]
    cascade_path = Path(haar_dir) / "haarcascade_frontalface_default.xml"
    classifier = cv2.CascadeClassifier(str(cascade_path))
    if classifier.empty():
        logger.warning("Haar face cascade could not be loaded from %s", cascade_path)
    return classifier


def _haar_boxes(image: ImageArray) -> list[tuple[int, int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = _haar().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return [(int(x), int(y), int(width), int(height)) for (x, y, width, height) in faces]


def _detection_view(image: ImageArray) -> tuple[ImageArray, float]:
    """Downscale large frames so YuNet sees a ~640px input.

    Native-resolution detection on multi-megapixel photos routinely misses
    obvious faces; a shorter side of 640 recovers them at high confidence.
    """
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= YUNET_DETECT_MAX_SIDE:
        return image, 1.0
    scale = YUNET_DETECT_MAX_SIDE / float(longest)
    new_width = max(round(width * scale), 1)
    new_height = max(round(height * scale), 1)
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return as_uint8(resized), scale


def scale_yunet_rows(faces: Any, scale: float) -> Any:
    """Map YuNet rows from a resized frame back onto the original image."""
    rows = np.asarray(faces, dtype=np.float32)
    if scale == 1.0 or rows.size == 0:
        return rows
    mapped = rows.copy()
    mapped[:, :14] /= scale
    return mapped


def _box_to_yunet_row(x: int, y: int, width: int, height: int) -> NDArray[np.float32]:
    """Approximate YuNet landmarks from a Haar box so SFace can alignCrop."""
    return np.asarray(
        [
            float(x),
            float(y),
            float(width),
            float(height),
            x + 0.3 * width,
            y + 0.37 * height,
            x + 0.7 * width,
            y + 0.37 * height,
            x + 0.5 * width,
            y + 0.55 * height,
            x + 0.35 * width,
            y + 0.78 * height,
            x + 0.65 * width,
            y + 0.78 * height,
            1.0,
        ],
        dtype=np.float32,
    )


def _crop_box(image: ImageArray, x: int, y: int, width: int, height: int) -> ImageArray | None:
    bound_h, bound_w = image.shape[:2]
    x1, y1 = max(x, 0), max(y, 0)
    x2, y2 = min(x + width, bound_w), min(y + height, bound_h)
    if x2 <= x1 or y2 <= y1:
        return None
    return as_uint8(image[y1:y2, x1:x2])


def _yunet_faces(image: ImageArray) -> Any | None:
    detector = _yunet()
    if detector is None:
        return None
    view, scale = _detection_view(image)
    height, width = view.shape[:2]
    detector.setInputSize((width, height))
    _retval, faces = detector.detect(view)
    if faces is None or len(faces) == 0:
        return None
    return scale_yunet_rows(faces, scale)


def _face_rows(image: ImageArray, *, haar_fallback: bool = False) -> Any | None:
    """YuNet rows only, unless ``haar_fallback`` (used for reference samples).

    Haar cascades fire on trees, water, and texture, so collection photos never
    use them. A tight sample crop may still try Haar if YuNet finds nothing.
    """
    faces = _yunet_faces(image)
    if faces is not None:
        return faces
    if not haar_fallback:
        return None
    boxes = _haar_boxes(image)
    if not boxes:
        return None
    return np.stack([_box_to_yunet_row(x, y, width, height) for (x, y, width, height) in boxes])


def _row_box(row: Any) -> tuple[int, int, int, int]:
    x, y, width, height = (round(float(value)) for value in row[:4])
    return x, y, width, height


def _classic_observations(image: ImageArray, *, haar_fallback: bool) -> list[FaceObservation]:
    rows = _face_rows(image, haar_fallback=haar_fallback)
    if rows is None:
        return []
    recognizer = _sface()
    observations: list[FaceObservation] = []
    for row in rows:
        box = _row_box(row)
        embedding = _embed_row(recognizer, image, row) if recognizer is not None else None
        observations.append(FaceObservation(bbox=box, embedding=embedding))
    return observations


def analyze(image: ImageArray, *, haar_fallback: bool = False) -> list[FaceObservation]:
    """Detect faces once and return boxes + embeddings via the active backend.

    ArcFace (InsightFace) is used when available; otherwise YuNet detection with
    SFace embeddings. ``haar_fallback`` only affects reference-sample loading,
    where a tight head crop may need whole-frame embedding.
    """
    if _embedder_override is not None:
        return [FaceObservation(bbox=None, embedding=vector) for vector in _embedder_override(image)]
    if _use_arcface():
        observations = [FaceObservation(bbox=face.bbox, embedding=face.embedding) for face in arcface.analyze(image)]
        if not observations and haar_fallback:
            fallback = arcface.embed_aligned_fallback(image)
            if fallback is not None:
                observations = [FaceObservation(bbox=None, embedding=fallback)]
        return observations
    return _classic_observations(image, haar_fallback=haar_fallback)


def detect_faces(image: ImageArray, *, haar_fallback: bool = False) -> list[ImageArray]:
    """Return BGR face crops. Empty when no face is found — never the full frame."""
    crops: list[ImageArray] = []
    for observation in analyze(image, haar_fallback=haar_fallback):
        if observation.bbox is None:
            continue
        crop = _crop_box(image, *observation.bbox)
        if crop is not None:
            crops.append(crop)
    return crops


def _embed_row(recognizer: Any, image: ImageArray, row: Any) -> Embedding | None:
    try:
        aligned = recognizer.alignCrop(image, row)
        feature = recognizer.feature(aligned)
        return _normalise(np.asarray(feature))
    except Exception:
        x, y, box_w, box_h = _row_box(row)
        crop = _crop_box(image, x, y, box_w, box_h)
        if crop is None:
            return None
        resized = cv2.resize(crop, (SFACE_ALIGN_SIZE, SFACE_ALIGN_SIZE), interpolation=cv2.INTER_AREA)
        try:
            feature = recognizer.feature(resized)
        except Exception:
            return None
        return _normalise(np.asarray(feature))


def face_embeddings(image: ImageArray, *, haar_fallback: bool = False) -> list[Embedding]:
    """Return face embeddings for each detected face, or [] if none / unavailable."""
    return [obs.embedding for obs in analyze(image, haar_fallback=haar_fallback) if obs.embedding is not None]


def match_embeddings(
    probes: list[Embedding],
    samples: dict[str, list[Embedding]],
    *,
    threshold: float | None = None,
) -> list[str]:
    """Return the names whose reference faces match any probe embedding.

    Each person keeps every reference exemplar (multiple photos per identity);
    a probe matches on the *best* cosine over that person's exemplars, so a
    single off-pose reference cannot sink a genuine match.
    """
    if not samples or not probes:
        return []
    cutoff = default_match_threshold() if threshold is None else threshold
    matched: list[str] = []
    for probe in probes:
        best_name = None
        best_score = cutoff
        for name, embeddings in samples.items():
            if not embeddings:
                continue
            score = max(_cosine(probe, sample) for sample in embeddings)
            if score > best_score:
                best_score = score
                best_name = name
        if best_name is not None and best_name not in matched:
            matched.append(best_name)
    return matched


def match_people(
    image: ImageArray,
    samples: dict[str, list[Embedding]],
    *,
    threshold: float | None = None,
) -> list[str]:
    """Tag a person only when a detected face matches a reference embedding."""
    return match_embeddings(face_embeddings(image), samples, threshold=threshold)


def count_and_people(
    image: ImageArray,
    samples: dict[str, list[Embedding]],
    *,
    threshold: float | None = None,
) -> tuple[int, list[str]]:
    """Detect faces once and return ``(face_count, matched_people)``."""
    observations = analyze(image)
    probes = [obs.embedding for obs in observations if obs.embedding is not None]
    return len(observations), match_embeddings(probes, samples, threshold=threshold)


def _bbox_area(observation: FaceObservation) -> int:
    if observation.bbox is None:
        return 0
    _x, _y, width, height = observation.bbox
    return max(width, 0) * max(height, 0)


def reference_embeddings(image: ImageArray) -> list[Embedding]:
    """Embeddings used to enroll a sample photo.

    Uses Haar / whole-crop fallback for tight head shots. If several faces are
    present, keep only the largest so a bystander is not enrolled under this
    person's name.
    """
    observations = [obs for obs in analyze(image, haar_fallback=True) if obs.embedding is not None]
    if len(observations) <= 1:
        return [obs.embedding for obs in observations if obs.embedding is not None]
    logger.warning("Sample photo has %s faces; enrolling the largest only", len(observations))
    best = max(observations, key=_bbox_area)
    return [best.embedding] if best.embedding is not None else []
