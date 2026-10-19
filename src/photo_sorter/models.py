"""Download and cache YOLO, CLIP, Real-ESRGAN, and face-recognition weights."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from photo_sorter.constants import (
    CLIP_MODEL_NAME,
    CLIP_PRETRAINED,
    DEFAULT_FACE_DETECTOR_MODEL,
    DEFAULT_FACE_RECOGNIZER_MODEL,
    DEFAULT_UPSCALE_MODEL,
    DEFAULT_YOLO_MODEL,
    FACE_DETECTOR_MODEL_URL,
    FACE_RECOGNIZER_MODEL_URL,
    UPSCALE_MODEL_URL,
)
from photo_sorter.exceptions import MissingDependencyError, ModelDownloadError
from photo_sorter.progress import counted

DEFAULT_MODEL_CACHE = Path.home() / ".cache" / "photo-sorter" / "models"


@dataclass(frozen=True)
class PulledModels:
    """Local paths / ids after weights have been fetched."""

    yolo: Path
    clip: str
    upscale: Path
    face_detector: Path
    face_recognizer: Path


def model_cache_dir() -> Path:
    """Directory where YOLO weights are stored (override with PHOTO_SORTER_MODEL_DIR)."""
    override = os.environ.get("PHOTO_SORTER_MODEL_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return DEFAULT_MODEL_CACHE


def insightface_root() -> Path:
    """Root passed to InsightFace; models land under ``<root>/models/<name>``."""
    return model_cache_dir().parent / "insightface"


def yolo_model_name(explicit: str | None = None) -> str:
    """Return the YOLO weight filename (env, CLI, or default)."""
    if explicit:
        return Path(explicit).name
    configured = os.environ.get("PHOTO_SORTER_YOLO_MODEL", "").strip()
    if configured:
        return Path(configured).name
    return DEFAULT_YOLO_MODEL


def resolve_yolo_weights(explicit: str | None = None) -> str:
    """Return an existing file path, or a name Ultralytics can download."""
    if explicit:
        path = Path(explicit).expanduser()
        if path.is_file():
            return str(path)
        cached = model_cache_dir() / path.name
        if cached.is_file():
            return str(cached)
        return path.name
    configured = os.environ.get("PHOTO_SORTER_YOLO_MODEL", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
    cached = model_cache_dir() / yolo_model_name(explicit)
    if cached.is_file():
        return str(cached)
    return yolo_model_name(explicit)


def ensure_yolo_weights(explicit: str | None = None) -> Path:
    """Download YOLO weights into the cache if they are not already there."""
    name = yolo_model_name(explicit)
    dest = model_cache_dir() / name
    if dest.is_file():
        return dest
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise MissingDependencyError("ultralytics") from exc
    dest.parent.mkdir(parents=True, exist_ok=True)
    previous = Path.cwd()
    try:
        os.chdir(dest.parent)
        YOLO(name)
    except Exception as exc:
        raise ModelDownloadError(name) from exc
    finally:
        os.chdir(previous)
    if not dest.is_file():
        raise ModelDownloadError(name)
    return dest


def ensure_clip_weights() -> str:
    """Download CLIP weights into the OpenCLIP / torch cache if needed."""
    try:
        import open_clip
    except ImportError as exc:
        raise MissingDependencyError("open-clip-torch") from exc
    label = f"{CLIP_MODEL_NAME} ({CLIP_PRETRAINED})"
    try:
        open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
        open_clip.get_tokenizer(CLIP_MODEL_NAME)
    except Exception as exc:
        raise ModelDownloadError(label) from exc
    return label


def _download_https(url: str, dest: Path) -> None:
    """Fetch ``url`` into ``dest`` (HTTPS only)."""
    if not url.startswith("https://"):
        raise ModelDownloadError(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    from urllib.request import Request, urlopen

    try:
        request = Request(url, headers={"User-Agent": "photo-sorter"})  # noqa: S310
        with urlopen(request, timeout=120) as response, partial.open("wb") as handle:  # noqa: S310
            total = int(response.headers.get("Content-Length") or 0)
            with counted(total or None, desc=dest.name, unit="B", unit_scale=True) as bar:
                while True:
                    chunk = response.read(256 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    bar.update(len(chunk))
        partial.replace(dest)
    except Exception as exc:
        if partial.exists():
            partial.unlink()
        raise ModelDownloadError(dest.name) from exc


def ensure_upscale_weights() -> Path:
    """Download Real-ESRGAN weights into the cache if they are not already there."""
    configured = os.environ.get("PHOTO_SORTER_UPSCALE_MODEL", "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return path
        dest = model_cache_dir() / path.name
        if dest.is_file():
            return dest
        raise ModelDownloadError(path.name)
    dest = model_cache_dir() / DEFAULT_UPSCALE_MODEL
    if dest.is_file():
        return dest
    _download_https(UPSCALE_MODEL_URL, dest)
    if not dest.is_file():
        raise ModelDownloadError(DEFAULT_UPSCALE_MODEL)
    return dest


def ensure_face_detector_weights() -> Path:
    """Download YuNet face-detection weights into the cache if needed."""
    dest = model_cache_dir() / DEFAULT_FACE_DETECTOR_MODEL
    if dest.is_file():
        return dest
    _download_https(FACE_DETECTOR_MODEL_URL, dest)
    if not dest.is_file():
        raise ModelDownloadError(DEFAULT_FACE_DETECTOR_MODEL)
    return dest


def ensure_face_recognizer_weights() -> Path:
    """Download SFace recognition weights into the cache if needed."""
    dest = model_cache_dir() / DEFAULT_FACE_RECOGNIZER_MODEL
    if dest.is_file():
        return dest
    _download_https(FACE_RECOGNIZER_MODEL_URL, dest)
    if not dest.is_file():
        raise ModelDownloadError(DEFAULT_FACE_RECOGNIZER_MODEL)
    return dest


def pull_models(yolo_name: str | None = None) -> PulledModels:
    """Fetch YOLO, CLIP, Real-ESRGAN, YuNet, and SFace weights onto disk."""
    return PulledModels(
        yolo=ensure_yolo_weights(yolo_name),
        clip=ensure_clip_weights(),
        upscale=ensure_upscale_weights(),
        face_detector=ensure_face_detector_weights(),
        face_recognizer=ensure_face_recognizer_weights(),
    )
