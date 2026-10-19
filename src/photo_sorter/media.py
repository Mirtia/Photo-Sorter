"""Media type helpers, capture dates, hashing, and PNG conversion."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import ExifTags, Image

from photo_sorter.constants import (
    IMAGE_EXTENSIONS,
    JPEG_QUALITY,
    PHASH_SIZE,
    PNG_EXTENSIONS,
    VIDEO_EXTENSIONS,
)

ImageArray = NDArray[np.uint8]
HashArray = NDArray[np.bool_]


def as_uint8(array: object) -> ImageArray:
    """Cast an OpenCV/NumPy array to uint8."""
    return np.asarray(array, dtype=np.uint8)


_DATETIME_TAGS = frozenset({"DateTimeOriginal", "DateTimeDigitized", "DateTime"})


def as_path(path: Path | str) -> Path:
    """Expand ``~`` in a filesystem path."""
    return Path(path).expanduser()


def unique_destination(directory: Path, filename: str) -> Path:
    """Return a path in ``directory`` that does not yet exist.

    The first file keeps its original name. Collisions get a numeric suffix
    (``img.jpg`` then ``img_1.jpg``, ``img_2.jpg``, ...).
    """
    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / filename
    if not destination.exists():
        return destination
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    index = 1
    while True:
        candidate = directory / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def is_image(path: Path) -> bool:
    """Return True if ``path`` looks like a still image."""
    return path.suffix.lower() in IMAGE_EXTENSIONS


def is_video(path: Path) -> bool:
    """Return True if ``path`` looks like a video file."""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_png(path: Path) -> bool:
    """Return True if ``path`` is a PNG image."""
    return path.suffix.lower() in PNG_EXTENSIONS


def season_for_month(month: int) -> str:
    """Return the meteorological season name for a month in 1..12.

    >>> season_for_month(4)
    'spring'
    >>> season_for_month(12)
    'winter'
    """
    if month in (12, 1, 2):
        return "winter"
    if month in (3, 4, 5):
        return "spring"
    if month in (6, 7, 8):
        return "summer"
    if month in (9, 10, 11):
        return "autumn"
    msg = f"Month must be in 1..12, got {month}"
    raise ValueError(msg)


def season_bin_name(captured_at: datetime) -> str:
    """Return a ``<season>_<year>`` bin name for a capture timestamp."""
    return f"{season_for_month(captured_at.month)}_{captured_at.year}"


def _exif_datetime(image: Image.Image) -> datetime | None:
    try:
        exif = image.getexif()
    except OSError:
        return None
    if not exif:
        return None
    tags = {ExifTags.TAGS.get(key, str(key)): value for key, value in exif.items()}
    for name in _DATETIME_TAGS:
        raw = tags.get(name)
        if not isinstance(raw, str):
            continue
        for fmt in ("%Y:%m:%d %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
    return None


def exif_datetime(path: Path) -> datetime | None:
    """Return DateTimeOriginal (or similar) from EXIF, or None if absent."""
    if not is_image(path):
        return None
    try:
        with Image.open(path) as image:
            return _exif_datetime(image)
    except OSError:
        return None


def capture_datetime(path: Path) -> datetime:
    """Return capture time from EXIF, falling back to file modification time."""
    taken = exif_datetime(path)
    if taken is not None:
        return taken
    return datetime.fromtimestamp(path.stat().st_mtime)


def convert_png_to_jpg(source: Path, destination: Path) -> None:
    """Convert a PNG to high-quality JPEG (4:4:4, quality 95)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        exif = image.info.get("exif")
        if image.mode in {"RGBA", "LA"} or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.split()[-1])
            rgb = background
        else:
            rgb = image.convert("RGB")
        if exif:
            rgb.save(
                destination,
                format="JPEG",
                quality=JPEG_QUALITY,
                subsampling=0,
                optimize=True,
                exif=exif,
            )
        else:
            rgb.save(
                destination,
                format="JPEG",
                quality=JPEG_QUALITY,
                subsampling=0,
                optimize=True,
            )


def load_image(path: Path) -> ImageArray | None:
    """Load an image as a BGR uint8 array, or None if it cannot be read."""
    loaded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if loaded is not None:
        return np.asarray(loaded, dtype=np.uint8)
    try:
        with Image.open(path) as image:
            rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
        return np.asarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), dtype=np.uint8)
    except OSError:
        return None


def save_image(image: ImageArray, destination: Path) -> None:
    """Write a BGR image as high-quality JPEG."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(
        destination,
        format="JPEG",
        quality=JPEG_QUALITY,
        subsampling=0,
        optimize=True,
    )


def perceptual_hash(image: ImageArray, hash_size: int = PHASH_SIZE) -> HashArray:
    """Compute a DCT perceptual hash plus a coarse colour hash."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(resized.astype(np.float32))
    dct_low = dct[:hash_size, :hash_size]
    structure = np.asarray(dct_low > np.median(dct_low), dtype=np.bool_).flatten()
    colour_small = cv2.resize(image, (4, 4), interpolation=cv2.INTER_AREA)
    colour = np.asarray(colour_small.flatten() > np.median(colour_small), dtype=np.bool_)
    return np.concatenate([structure, colour])


def hamming_distance(left: HashArray, right: HashArray) -> int:
    """Return the Hamming distance between two perceptual hashes."""
    return int(np.count_nonzero(left != right))


def phash_to_hex(hash_array: HashArray) -> str:
    """Serialise a perceptual-hash bit array to a compact hex string."""
    packed = np.packbits(np.asarray(hash_array, dtype=np.bool_))
    return packed.tobytes().hex()


def md5_file(path: Path) -> str:
    """Return the MD5 hex digest of a file's contents."""
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()
