"""Phase 3: score image quality, enhance low-scoring pictures, rename to MD5."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from photo_sorter import manifest
from photo_sorter.constants import DEFAULT_QUALITY_THRESHOLD, ENHANCEMENT_LEVELS, UPSCALE_LEVELS
from photo_sorter.exceptions import InvalidInputError, InvalidOptionError
from photo_sorter.media import (
    ImageArray,
    as_path,
    as_uint8,
    is_image,
    is_video,
    load_image,
    md5_file,
    save_image,
    unique_destination,
)
from photo_sorter.progress import track
from photo_sorter.sort import PICTURES_DIR, VIDEOS_DIR
from photo_sorter.upscale import upscale_factor, upscale_image

logger = logging.getLogger(__name__)

_DENOISE_H = {"light": 3.0, "medium": 6.0, "strong": 10.0}
_SHARPEN_AMOUNT = {"light": 0.35, "medium": 0.7, "strong": 1.1}
_SHARPEN_RADIUS = {"light": 0.8, "medium": 1.0, "strong": 1.3}
_SHARPEN_THRESHOLD = {"light": 6.0, "medium": 4.0, "strong": 2.0}

_SHARPNESS_CUTOFF = 0.55
_NOISE_CUTOFF = 0.65
_ARTIFACT_CUTOFF = 0.65
_FLAT_GRADIENT_ENERGY = 40.0
_DEFOCUS_LAPLACIAN = 80.0
_MOTION_LAPLACIAN = 1500.0
_MOTION_ANISOTROPY_LIMIT = 0.50


@dataclass(frozen=True)
class QualityBreakdown:
    """Per-component quality scores in 0..1 (higher is better)."""

    overall: float
    resolution: float
    sharpness: float
    noise: float
    artifacts: float
    exposure: float


@dataclass
class EnhancementConfig:
    """Flags that control which enhancement techniques are applied."""

    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    apply_color_correction: bool = False
    apply_denoising: str | None = None
    apply_sharpening: str | None = None
    apply_artifact_removal: bool = False
    apply_upscaling: str | None = None

    def techniques_enabled(self) -> bool:
        """Return True if at least one enhancement technique is turned on."""
        return any((
            self.apply_color_correction,
            self.apply_denoising,
            self.apply_sharpening,
            self.apply_artifact_removal,
            self.apply_upscaling,
        ))


def _validate_level(name: str, value: str | None, allowed: tuple[str, ...] = ENHANCEMENT_LEVELS) -> str | None:
    if value is None:
        return None
    if value not in allowed:
        raise InvalidOptionError(name, value)
    return value


def _resolution_score(width: int, height: int) -> float:
    min_side = min(width, height)
    pixels = width * height
    side_score = min(min_side / 1080.0, 1.0)
    pixel_score = min(pixels / (1920 * 1080), 1.0)
    return float(0.6 * side_score + 0.4 * pixel_score)


def _sharpness_score(gray: ImageArray) -> float:
    variance = float(np.var(gray))
    if variance < 4.0:
        return 1.0
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    normalized = float(laplacian.var()) / (variance + 1e-6)
    return float(min(normalized / 1.8, 1.0))


def _anisotropy_from_gradients(gx: np.ndarray, gy: np.ndarray) -> float:
    """Return 0 for isotropic structure and 1 for strongly directional edges."""
    ixx = float(np.mean(gx * gx))
    iyy = float(np.mean(gy * gy))
    ixy = float(np.mean(gx * gy))
    trace = ixx + iyy
    if trace < 1e-6:
        return 0.0
    disc = max(trace * trace - 4.0 * (ixx * iyy - ixy * ixy), 0.0)
    delta = float(np.sqrt(disc))
    lambda_1 = 0.5 * (trace + delta)
    lambda_2 = 0.5 * (trace - delta)
    return float((lambda_1 - lambda_2) / (lambda_1 + lambda_2 + 1e-12))


def has_unfixable_blur(image: ImageArray) -> bool:
    """Return True for motion streak or defocus that enhancement cannot recover.

    Flat / low-detail frames are not treated as blur. A sharp picture with
    directional content (a fence, a horizon) keeps a high Laplacian variance
    and is kept.
    """
    gray = as_uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = float(np.mean(gx * gx + gy * gy))
    if energy < _FLAT_GRADIENT_ENERGY:
        return False
    if laplacian_var < _DEFOCUS_LAPLACIAN:
        return True
    return _anisotropy_from_gradients(gx, gy) > _MOTION_ANISOTROPY_LIMIT and laplacian_var < _MOTION_LAPLACIAN


def _noise_score(gray: ImageArray) -> float:
    """Return 1 for a clean image and 0 for a very noisy one."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    sigma = float(np.median(residual)) * 1.4826
    return float(max(1.0 - sigma / 12.0, 0.0))


def _exposure_score(gray: ImageArray) -> float:
    mean_l = float(np.mean(gray)) / 255.0
    centred = max(1.0 - abs(mean_l - 0.45) / 0.45, 0.0)
    clipped = float(np.mean((gray <= 5) | (gray >= 250)))
    return float(max(centred - clipped, 0.0))


def _mean_abs_diff(left: ImageArray, right: ImageArray) -> float:
    height = min(left.shape[0], right.shape[0])
    width = min(left.shape[1], right.shape[1])
    if height == 0 or width == 0:
        return 0.0
    delta = left[:height, :width].astype(np.float32) - right[:height, :width].astype(np.float32)
    return float(np.mean(np.abs(delta)))


def _blockiness_score(gray: ImageArray) -> float:
    """Estimate JPEG block-artifact absence (1 is clean, 0 is blocky)."""
    height, width = gray.shape[:2]
    if height < 16 or width < 16:
        return 1.0
    horizontal = _mean_abs_diff(gray[:, 8::8], gray[:, 7::8])
    vertical = _mean_abs_diff(gray[8::8, :], gray[7::8, :])
    return float(max(1.0 - (horizontal + vertical) / 40.0, 0.0))


def assess_quality(image: ImageArray) -> QualityBreakdown:
    """Score resolution, sharpness, noise, blockiness, and exposure in 0..1."""
    height, width = image.shape[:2]
    gray = as_uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    resolution = _resolution_score(width, height)
    sharpness = _sharpness_score(gray)
    noise = _noise_score(gray)
    artifacts = _blockiness_score(gray)
    exposure = _exposure_score(gray)
    overall = 0.25 * resolution + 0.25 * sharpness + 0.25 * noise + 0.15 * artifacts + 0.10 * exposure
    return QualityBreakdown(
        overall=float(overall),
        resolution=resolution,
        sharpness=sharpness,
        noise=noise,
        artifacts=artifacts,
        exposure=exposure,
    )


def quality_score(image: ImageArray) -> float:
    """Return the overall 0..1 quality score for ``image``."""
    return assess_quality(image).overall


def _gamma_lut(gamma: float) -> ImageArray:
    table = np.clip((np.arange(256) / 255.0) ** gamma * 255.0, 0, 255)
    return as_uint8(table)


def _auto_exposure(image: ImageArray) -> ImageArray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, axis_a, axis_b = cv2.split(lab)
    mean_l = float(np.mean(lightness)) / 255.0
    if 0.32 <= mean_l <= 0.62:
        return image
    gamma = float(np.log(0.45) / np.log(max(mean_l, 0.05)))
    gamma = min(max(gamma, 0.65), 1.55)
    adjusted = cv2.LUT(lightness, _gamma_lut(gamma))
    merged = cv2.merge((as_uint8(adjusted), axis_a, axis_b))
    return as_uint8(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR))


def _maybe_clahe(image: ImageArray) -> ImageArray:
    height, width = image.shape[:2]
    if min(height, width) < 16:
        return image
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness, axis_a, axis_b = cv2.split(lab)
    if float(np.std(lightness)) >= 42.0:
        return image
    tile = 8 if min(height, width) >= 32 else 4
    clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(tile, tile))
    merged = cv2.merge((clahe.apply(lightness), axis_a, axis_b))
    return as_uint8(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR))


def _correct_color(image: ImageArray) -> ImageArray:
    """Lift exposure and add contrast if the picture is flat. Colours are left as-is."""
    return _maybe_clahe(_auto_exposure(image))


def _denoise(image: ImageArray, level: str) -> ImageArray:
    if min(image.shape[:2]) < 24:
        return _remove_artifacts(image)
    strength = _DENOISE_H[level]
    return as_uint8(cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21))


def _sharpen(image: ImageArray, level: str) -> ImageArray:
    amount = _SHARPEN_AMOUNT[level]
    blurred = cv2.GaussianBlur(image, (0, 0), _SHARPEN_RADIUS[level])
    residual = image.astype(np.float32) - blurred.astype(np.float32)
    magnitude = np.max(np.abs(residual), axis=2, keepdims=True)
    mask = (magnitude >= _SHARPEN_THRESHOLD[level]).astype(np.float32)
    sharpened = image.astype(np.float32) + residual * amount * mask
    return as_uint8(np.clip(sharpened, 0, 255))


def _remove_artifacts(image: ImageArray) -> ImageArray:
    """Soften JPEG block edges without a heavy bilateral smear."""
    smoothed = cv2.GaussianBlur(image, (3, 3), 0)
    return as_uint8(cv2.addWeighted(image, 0.7, smoothed, 0.3, 0))


def _technique_needed(
    breakdown: QualityBreakdown,
    image: ImageArray,
    config: EnhancementConfig,
) -> bool:
    if config.apply_artifact_removal and breakdown.artifacts < _ARTIFACT_CUTOFF:
        return True
    if config.apply_denoising is not None and breakdown.noise < _NOISE_CUTOFF:
        return True
    if config.apply_color_correction and breakdown.exposure < 0.7:
        return True
    if config.apply_sharpening is not None and breakdown.sharpness < _SHARPNESS_CUTOFF:
        return True
    return config.apply_upscaling is not None and upscale_factor(image, config.apply_upscaling) > 1.0


def _should_process(
    breakdown: QualityBreakdown,
    image: ImageArray,
    config: EnhancementConfig,
) -> bool:
    if not config.techniques_enabled():
        return False
    if breakdown.overall < config.quality_threshold:
        return True
    return _technique_needed(breakdown, image, config)


def _apply_techniques(
    image: ImageArray,
    breakdown: QualityBreakdown,
    config: EnhancementConfig,
) -> ImageArray:
    result = image
    if config.apply_artifact_removal and breakdown.artifacts < _ARTIFACT_CUTOFF:
        result = _remove_artifacts(result)
    if config.apply_denoising is not None and breakdown.noise < _NOISE_CUTOFF:
        result = _denoise(result, config.apply_denoising)
    if config.apply_color_correction:
        result = _correct_color(result)
    if config.apply_upscaling is not None:
        result = upscale_image(result, config.apply_upscaling)
    if config.apply_sharpening is not None and assess_quality(result).sharpness < _SHARPNESS_CUTOFF:
        result = _sharpen(result, config.apply_sharpening)
    return result


def enhance_image(image: ImageArray, config: EnhancementConfig) -> ImageArray:
    """Apply only the enabled techniques that the image actually needs.

    Order is deblock → denoise → colour/exposure → upscale → sharpen, so
    sharpening runs at the final resolution and does not amplify noise.
    If the overall score drops and the size did not change, the original
    pixels are kept.
    """
    before = assess_quality(image)
    if not _should_process(before, image, config):
        return image
    processed = _apply_techniques(image, before, config)
    if processed.shape != image.shape:
        return processed
    after = assess_quality(processed)
    if after.overall + 0.01 < before.overall:
        logger.debug("Reverting enhancement (%.3f -> %.3f)", before.overall, after.overall)
        return image
    return processed


def _copy_videos(season_dir: Path, destination_season: Path) -> None:
    source_videos = season_dir / VIDEOS_DIR
    dest_videos = destination_season / VIDEOS_DIR
    dest_videos.mkdir(parents=True, exist_ok=True)
    if not source_videos.is_dir():
        return
    for path in source_videos.iterdir():
        if path.is_file() and is_video(path):
            shutil.copy2(path, unique_destination(dest_videos, path.name))


def _process_picture(
    path: Path,
    pictures_out: Path,
    settings: EnhancementConfig,
    discard_season: Path | None,
    season: str,
    lineage: dict[str, Any] | None,
) -> dict[str, Any] | None:
    image = load_image(path)
    if image is None:
        logger.warning("Skipping unreadable image: %s", path)
        return None
    if has_unfixable_blur(image):
        logger.info("%s discarded (motion or focus blur)", path.name)
        discard_path = None
        if discard_season is not None:
            discard_path = unique_destination(discard_season, path.name)
            shutil.copy2(path, discard_path)
        return manifest.enhance_record(
            season=season,
            source_path=path,
            lineage=lineage,
            discarded_blur=True,
            discard_path=discard_path,
        )
    before = assess_quality(image)
    processed = enhance_image(image, settings)
    enhanced = processed is not image
    if enhanced:
        logger.info("%s quality=%.2f enhanced", path.name, before.overall)
    temp_path = pictures_out / f".tmp_{path.name}.jpg"
    save_image(processed, temp_path)
    digest = md5_file(temp_path)
    final_path = unique_destination(pictures_out, f"{digest}.jpg")
    temp_path.replace(final_path)
    return manifest.enhance_record(
        season=season,
        source_path=path,
        lineage=lineage,
        enhanced_hash=digest,
        enhanced_path=final_path,
        enhanced=enhanced,
    )


def enhance_directory(
    input_directory: Path | str,
    output_directory: Path | str,
    config: EnhancementConfig | None = None,
    *,
    discard_directory: Path | str | None = None,
) -> Path:
    """Enhance low-quality pictures and rename every image to its MD5 hash.

    Pictures with motion or defocus blur are copied to ``discard_directory``
    (when given) and are not written to the enhanced tree.
    """
    source = as_path(input_directory)
    if not source.is_dir():
        raise InvalidInputError(source)

    settings = config or EnhancementConfig()
    settings.apply_denoising = _validate_level("apply_denoising", settings.apply_denoising)
    settings.apply_sharpening = _validate_level("apply_sharpening", settings.apply_sharpening)
    settings.apply_upscaling = _validate_level("apply_upscaling", settings.apply_upscaling, UPSCALE_LEVELS)

    destination = as_path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    discarded = as_path(discard_directory) if discard_directory is not None else None
    lineage = manifest.load_dedup_lineage(source)

    jobs: list[tuple[Path, Path, Path | None, str]] = []
    for season_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        dest_season = destination / season_dir.name
        pictures_out = dest_season / PICTURES_DIR
        pictures_out.mkdir(parents=True, exist_ok=True)
        _copy_videos(season_dir, dest_season)
        discard_season = discarded / season_dir.name if discarded is not None else None

        pictures_in = season_dir / PICTURES_DIR
        if not pictures_in.is_dir():
            continue
        for path in sorted(pictures_in.iterdir()):
            if path.is_file() and is_image(path):
                jobs.append((path, pictures_out, discard_season, season_dir.name))

    records: list[dict[str, Any]] = []
    for path, pictures_out, discard_season, season in track(jobs, desc="enhance"):
        origin = lineage.get(str(path.resolve()))
        record = _process_picture(path, pictures_out, settings, discard_season, season, origin)
        if record is not None:
            records.append(record)

    manifest.write_records(destination, records)
    return destination
