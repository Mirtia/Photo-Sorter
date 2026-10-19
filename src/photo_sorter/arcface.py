"""ArcFace face detection + recognition via InsightFace (antelopev2, R100).

This is the high-accuracy backend: RetinaFace for detection and an ArcFace R100
embedding (512-d) for recognition, both ONNX models running on the CUDA
execution provider when a GPU is available. Everything degrades gracefully —
if InsightFace or its weights are missing, :func:`available` returns False and
``faces.py`` falls back to the OpenCV YuNet/SFace path.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from photo_sorter.constants import ARCFACE_DET_SIZE, ARCFACE_MODEL_NAME, SFACE_ALIGN_SIZE
from photo_sorter.detect import ml_enabled
from photo_sorter.device import insightface_ctx_id, onnx_providers
from photo_sorter.media import ImageArray, as_uint8

logger = logging.getLogger(__name__)

Embedding = NDArray[np.float32]


@dataclass(frozen=True)
class ArcFaceResult:
    """One detected face: pixel box ``(x, y, w, h)`` plus a unit embedding."""

    bbox: tuple[int, int, int, int]
    det_score: float
    embedding: Embedding


def _normalise(vector: NDArray[np.floating[Any]]) -> Embedding:
    flat = np.asarray(vector, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(flat))
    if norm == 0.0:
        return flat
    return np.asarray(flat / norm, dtype=np.float32)


def _flatten_nested_pack(root: Path) -> None:
    """Repair InsightFace's ``antelopev2.zip`` nested-folder extraction.

    The official archive unpacks to ``<models>/antelopev2/antelopev2/*.onnx``,
    one level deeper than ``FaceAnalysis`` expects, so it reports no detection
    model. Move the weights up a level when we spot that layout.
    """
    pack = root / "models" / ARCFACE_MODEL_NAME
    nested = pack / ARCFACE_MODEL_NAME
    if not nested.is_dir():
        return
    for onnx in nested.glob("*.onnx"):
        target = pack / onnx.name
        if not target.exists():
            shutil.move(str(onnx), str(target))
    if not any(nested.iterdir()):
        nested.rmdir()


def _build_app(root: Path) -> Any:
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(
        name=ARCFACE_MODEL_NAME,
        root=str(root),
        providers=list(onnx_providers()),
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=insightface_ctx_id(), det_size=ARCFACE_DET_SIZE)
    return app


@lru_cache(maxsize=1)
def _app() -> Any | None:
    if not ml_enabled():
        return None
    try:
        import insightface  # noqa: F401
    except ImportError:
        logger.info("insightface is not installed; using YuNet/SFace for faces")
        return None
    try:
        from photo_sorter.models import insightface_root

        root = insightface_root()
        _flatten_nested_pack(root)
        try:
            app = _build_app(root)
        except AssertionError:
            # First run downloaded the pack but extracted it nested; repair + retry.
            _flatten_nested_pack(root)
            app = _build_app(root)
    except Exception:
        logger.warning(
            "InsightFace %s unavailable; falling back to YuNet/SFace",
            ARCFACE_MODEL_NAME,
            exc_info=True,
        )
        return None
    logger.info("Face backend: InsightFace %s (%s)", ARCFACE_MODEL_NAME, onnx_providers()[0])
    return app


def available() -> bool:
    """True when the InsightFace ArcFace backend loaded successfully."""
    return _app() is not None


def _embedding_of(face: Any) -> Embedding | None:
    embedding = getattr(face, "normed_embedding", None)
    if embedding is not None:
        return np.asarray(embedding, dtype=np.float32)
    raw = getattr(face, "embedding", None)
    if raw is not None:
        return _normalise(np.asarray(raw))
    return None


def analyze(image: ImageArray) -> list[ArcFaceResult]:
    """Detect faces and return their boxes + unit ArcFace embeddings."""
    app = _app()
    if app is None:
        return []
    try:
        faces = app.get(as_uint8(image))
    except Exception:
        logger.debug("InsightFace analyze failed", exc_info=True)
        return []
    results: list[ArcFaceResult] = []
    for face in faces:
        embedding = _embedding_of(face)
        if embedding is None:
            continue
        x1, y1, x2, y2 = (round(float(value)) for value in face.bbox[:4])
        bbox = (x1, y1, max(x2 - x1, 0), max(y2 - y1, 0))
        results.append(ArcFaceResult(bbox=bbox, det_score=float(getattr(face, "det_score", 0.0)), embedding=embedding))
    return results


_MAX_FALLBACK_SIDE = 640


def embed_aligned_fallback(image: ImageArray) -> Embedding | None:
    """Embed a pre-cropped face (a reference sample) that detection missed.

    Reference photos are often tight head crops where RetinaFace finds nothing;
    here we feed the whole frame straight to the recognition head. Full-size
    photos are skipped so a missed detection is not enrolled as a face.
    """
    height, width = image.shape[:2]
    if max(height, width) > _MAX_FALLBACK_SIDE:
        return None
    app = _app()
    if app is None:
        return None
    recognizer = getattr(app, "models", {}).get("recognition")
    if recognizer is None:
        return None
    try:
        crop = cv2.resize(as_uint8(image), (SFACE_ALIGN_SIZE, SFACE_ALIGN_SIZE), interpolation=cv2.INTER_AREA)
        feature = recognizer.get_feat(crop)
        return _normalise(np.asarray(feature))
    except Exception:
        logger.debug("InsightFace fallback embedding failed", exc_info=True)
        return None
