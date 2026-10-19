from pathlib import Path

import cv2
import numpy as np

from photo_sorter import manifest
from photo_sorter.enhance import (
    EnhancementConfig,
    assess_quality,
    enhance_directory,
    enhance_image,
    has_unfixable_blur,
    quality_score,
)
from photo_sorter.media import as_uint8, load_image, md5_file
from photo_sorter.sort import PICTURES_DIR, VIDEOS_DIR
from tests.helpers import save_checkerboard, save_rgb


def _bgr(color: tuple[int, int, int], size: tuple[int, int] = (80, 80)) -> np.ndarray:
    height, width = size[1], size[0]
    pixel = np.array(color[::-1], dtype=np.uint8)
    return np.broadcast_to(pixel, (height, width, 3)).copy()


def test_quality_score_prefers_large_smooth_images(tmp_path: Path) -> None:
    smooth = save_rgb(tmp_path / "smooth.jpg", (120, 130, 140), size=(400, 300))
    noisy = save_checkerboard(tmp_path / "noisy.jpg", size=(32, 32))

    smooth_image = load_image(smooth)
    noisy_image = load_image(noisy)
    assert smooth_image is not None
    assert noisy_image is not None
    smooth_quality = assess_quality(smooth_image)
    noisy_quality = assess_quality(noisy_image)
    assert smooth_quality.resolution > noisy_quality.resolution
    assert smooth_quality.noise > noisy_quality.noise
    assert quality_score(smooth_image) > quality_score(noisy_image)


def test_color_correction_lifts_a_dark_image() -> None:
    dark = _bgr((20, 20, 20), size=(96, 96))
    enhanced = enhance_image(
        dark,
        EnhancementConfig(quality_threshold=1.0, apply_color_correction=True),
    )
    assert float(enhanced.mean()) > float(dark.mean())


def test_color_correction_does_not_turn_red_photos_green() -> None:
    red = _bgr((200, 30, 30), size=(96, 96))
    enhanced = enhance_image(
        red,
        EnhancementConfig(quality_threshold=1.0, apply_color_correction=True),
    )
    assert float(enhanced[:, :, 2].mean()) > float(enhanced[:, :, 1].mean())


def test_upscale_enlarges_a_small_image() -> None:
    small = _bgr((120, 130, 140), size=(48, 40))
    enhanced = enhance_image(
        small,
        EnhancementConfig(quality_threshold=1.0, apply_upscaling="medium"),
    )
    assert enhanced.shape[0] > small.shape[0]
    assert enhanced.shape[1] > small.shape[1]


def test_ml_upscale_falls_back_to_lanczos_when_disabled() -> None:
    small = _bgr((120, 130, 140), size=(48, 40))
    enhanced = enhance_image(
        small,
        EnhancementConfig(quality_threshold=1.0, apply_upscaling="ml"),
    )
    assert enhanced.shape[0] == small.shape[0] * 4
    assert enhanced.shape[1] == small.shape[1] * 4


def test_ml_upscale_uses_injected_model() -> None:
    import cv2

    from photo_sorter.upscale import use_ml_upscaler

    small = _bgr((120, 130, 140), size=(32, 24))

    def fake_sr(image: np.ndarray) -> np.ndarray:
        return cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)

    use_ml_upscaler(fake_sr)
    try:
        enhanced = enhance_image(
            small,
            EnhancementConfig(quality_threshold=1.0, apply_upscaling="ml"),
        )
    finally:
        use_ml_upscaler(None)

    assert enhanced.shape[0] == small.shape[0] * 4
    assert enhanced.shape[1] == small.shape[1] * 4


def test_clean_high_res_image_is_left_alone() -> None:
    clean = _bgr((128, 128, 128), size=(1920, 1080))
    enhanced = enhance_image(
        clean,
        EnhancementConfig(
            quality_threshold=0.2,
            apply_denoising="strong",
            apply_sharpening="strong",
            apply_artifact_removal=True,
        ),
    )
    np.testing.assert_array_equal(enhanced, clean)


def test_enhance_renames_pictures_to_md5_and_copies_videos(tmp_path: Path) -> None:
    season = tmp_path / "input" / "spring_2025"
    pictures = season / PICTURES_DIR
    videos = season / VIDEOS_DIR
    pictures.mkdir(parents=True)
    videos.mkdir()
    original = save_checkerboard(pictures / "holiday.jpg")
    (videos / "clip.mp4").write_bytes(b"video")
    output = tmp_path / "enhanced"

    enhance_directory(
        tmp_path / "input",
        output,
        EnhancementConfig(
            quality_threshold=1.0,
            apply_color_correction=True,
            apply_denoising="light",
            apply_sharpening="light",
            apply_artifact_removal=True,
            apply_upscaling="light",
        ),
    )

    enhanced_pictures = list((output / "spring_2025" / PICTURES_DIR).glob("*.jpg"))
    assert len(enhanced_pictures) == 1
    assert enhanced_pictures[0].stem == md5_file(enhanced_pictures[0])
    assert enhanced_pictures[0].name != original.name
    assert (output / "spring_2025" / VIDEOS_DIR / "clip.mp4").is_file()


def test_enhance_manifest_links_hash_back_to_dedup_group(tmp_path: Path) -> None:
    season = tmp_path / "input" / "spring_2025"
    pictures = season / PICTURES_DIR
    pictures.mkdir(parents=True)
    source = save_checkerboard(pictures / "holiday.jpg")
    original = tmp_path / "sorted" / "spring_2025" / PICTURES_DIR / "IMG_0007.jpg"
    manifest.write_records(
        tmp_path / "input",
        [
            manifest.dedup_record(
                season="spring_2025",
                group_id="spring_2025:2",
                group_size=3,
                original_path=original,
                phash="0f0f",
                size_bytes=source.stat().st_size,
                kept=True,
                output_path=source,
            )
        ],
    )
    output = tmp_path / "enhanced"

    enhance_directory(tmp_path / "input", output, EnhancementConfig(quality_threshold=1.0, apply_sharpening="light"))

    enhanced_pictures = list((output / "spring_2025" / PICTURES_DIR).glob("*.jpg"))
    assert len(enhanced_pictures) == 1
    records = manifest.read_records(output)
    assert len(records) == 1
    record = records[0]
    assert record["phase"] == "enhance"
    assert record["enhanced_hash"] == enhanced_pictures[0].stem
    assert record["group_id"] == "spring_2025:2"
    assert record["original_path"] == str(original)
    assert record["original_name"] == "IMG_0007.jpg"
    assert record["discarded_blur"] is False


def test_enhance_manifest_records_blur_discards(tmp_path: Path) -> None:
    season = tmp_path / "input" / "summer_2024"
    pictures = season / PICTURES_DIR
    pictures.mkdir(parents=True)
    sharp = _bgr_checkerboard()
    cv2.imwrite(str(pictures / "keep.png"), sharp)
    kernel = np.zeros((1, 25), dtype=np.float32)
    kernel[0, :] = 1.0 / 25.0
    cv2.imwrite(str(pictures / "blur.png"), cv2.filter2D(sharp, -1, kernel))
    output = tmp_path / "enhanced"
    discard = tmp_path / "discarded"

    enhance_directory(
        tmp_path / "input",
        output,
        EnhancementConfig(quality_threshold=0.9, apply_sharpening="light"),
        discard_directory=discard,
    )

    records = manifest.read_records(output)
    blur = [record for record in records if record["discarded_blur"]]
    assert len(blur) == 1
    assert blur[0]["original_name"] == "blur.png"
    assert blur[0]["enhanced_hash"] is None
    assert Path(blur[0]["discard_path"]).is_file()


def _bgr_checkerboard(size: int = 256, tile: int = 16) -> np.ndarray:
    rows, cols = np.indices((size, size))
    mask = ((cols // tile) + (rows // tile)) % 2 == 0
    array = np.zeros((size, size, 3), dtype=np.uint8)
    array[mask] = 255
    array[~mask] = 40
    return array


def test_solid_color_is_not_treated_as_blur() -> None:
    assert has_unfixable_blur(_bgr((128, 128, 128), size=(120, 80))) is False


def test_sharp_checkerboard_is_not_treated_as_blur() -> None:
    assert has_unfixable_blur(_bgr_checkerboard()) is False


def test_defocus_and_motion_blur_are_unfixable() -> None:
    sharp = _bgr_checkerboard()
    defocused = as_uint8(cv2.GaussianBlur(sharp, (21, 21), 0))
    kernel = np.zeros((1, 25), dtype=np.float32)
    kernel[0, :] = 1.0 / 25.0
    streaked = as_uint8(cv2.filter2D(sharp, -1, kernel))
    assert has_unfixable_blur(defocused) is True
    assert has_unfixable_blur(streaked) is True


def test_blurry_pictures_are_discarded_not_enhanced(tmp_path: Path) -> None:
    season = tmp_path / "input" / "summer_2024"
    pictures = season / PICTURES_DIR
    pictures.mkdir(parents=True)
    sharp = _bgr_checkerboard()
    cv2.imwrite(str(pictures / "keep.png"), sharp)
    kernel = np.zeros((1, 25), dtype=np.float32)
    kernel[0, :] = 1.0 / 25.0
    cv2.imwrite(str(pictures / "blur.png"), cv2.filter2D(sharp, -1, kernel))
    output = tmp_path / "enhanced"
    discard = tmp_path / "discarded"

    enhance_directory(
        tmp_path / "input",
        output,
        EnhancementConfig(quality_threshold=0.9, apply_sharpening="light"),
        discard_directory=discard,
    )

    kept = list((output / "summer_2024" / PICTURES_DIR).glob("*.jpg"))
    dumped = list((discard / "summer_2024").iterdir())
    assert len(kept) == 1
    assert [path.name for path in dumped] == ["blur.png"]
