"""Pytest configuration."""

from __future__ import annotations

import os

# Keep unit tests off YOLO/CLIP downloads even if the ml extra is installed.
os.environ["PHOTO_SORTER_DISABLE_ML"] = "1"
# Do not pick up a photo-sorter.yaml from the repo checkout during tests.
os.environ["PHOTO_SORTER_DISABLE_AUTOCONFIG"] = "1"
# Keep pytest output free of tqdm bars.
os.environ["PHOTO_SORTER_DISABLE_PROGRESS"] = "1"
# Never open a Tk duplicate-picker window during tests.
os.environ["PHOTO_SORTER_DISABLE_GUI"] = "1"
