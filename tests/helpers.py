"""Shared helpers for photo-sorter tests."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image


def save_rgb(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (48, 48)) -> Path:
    """Write a solid-color RGB image (JPEG unless the suffix is .png)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", size, color)
    if path.suffix.lower() == ".png":
        image.save(path, format="PNG")
    else:
        image.save(path, format="JPEG", quality=95)
    return path


def save_checkerboard(path: Path, size: tuple[int, int] = (48, 48)) -> Path:
    """Write a high-frequency JPEG useful as a 'noisy' low-quality picture."""
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    array[::2, ::2] = 255
    array[1::2, 1::2] = 128
    Image.fromarray(array, mode="RGB").save(path, format="JPEG", quality=40)
    return path


def set_mtime(path: Path, captured_at: datetime) -> None:
    """Set a file's modification time to a naive local datetime."""
    timestamp = captured_at.timestamp()
    os.utime(path, (timestamp, timestamp))
