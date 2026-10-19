"""Shared constants for the photo-sorter pipeline."""

from __future__ import annotations

IMAGE_EXTENSIONS = frozenset({
    ".bmp",
    ".gif",
    ".heic",
    ".heif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
})
VIDEO_EXTENSIONS = frozenset({
    ".3gp",
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".webm",
    ".wmv",
})
PNG_EXTENSIONS = frozenset({".png"})
JPEG_EXTENSIONS = frozenset({".jpg", ".jpeg"})

JPEG_QUALITY = 95
PHASH_SIZE = 8
PHASH_HAMMING_THRESHOLD = 10
DEFAULT_QUALITY_THRESHOLD = 0.75
PIPELINE_PHASES = ("backup", "sort", "dedup", "enhance", "metadata", "index")
FACE_MATCH_THRESHOLD = 0.40
# ArcFace (antelopev2/R100) cosine similarities run on a different scale than
# SFace; genuine pairs usually exceed ~0.5 and impostors sit below ~0.2. Start
# conservative and let users tune with --face-threshold on their own gallery.
ARCFACE_MODEL_NAME = "antelopev2"
ARCFACE_DET_SIZE = (640, 640)
ARCFACE_MATCH_THRESHOLD = 0.35
ARCFACE_EMBEDDING_DIMS = 512
YUNET_SCORE_THRESHOLD = 0.7
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000
YUNET_DETECT_MAX_SIDE = 640
SFACE_ALIGN_SIZE = 112
TOP_TOPIC_COUNT = 3
TOP_OBJECT_COUNT = 12
CLIP_CAPTION_MIN_SCORE = 0.2
DEFAULT_FACE_DETECTOR_MODEL = "face_detection_yunet_2023mar.onnx"
FACE_DETECTOR_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
DEFAULT_FACE_RECOGNIZER_MODEL = "face_recognition_sface_2021dec.onnx"
FACE_RECOGNIZER_MODEL_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
)
YOLO_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_YOLO_MODEL = "yolov8n.pt"
# RF-DETR (Apache-2.0) is the primary detector; override the size with
# PHOTO_SORTER_RFDETR_MODEL (nano|small|medium|base|large).
DEFAULT_RFDETR_MODEL = "medium"
RFDETR_CONFIDENCE_THRESHOLD = 0.3
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "openai"
CLIP_EMBEDDING_DIMS = 512
DEFAULT_SEARCH_INDEX = "photos"
DEFAULT_SEARCH_RESULTS = 10
DEFAULT_CONFIG_FILENAMES = ("photo-sorter.yaml", ".photo-sorter.yaml")

SEASON_NAMES = ("winter", "spring", "summer", "autumn")
ENHANCEMENT_LEVELS = ("light", "medium", "strong")
UPSCALE_LEVELS = ("light", "medium", "strong", "ml")
DEFAULT_UPSCALE_MODEL = "realesr-general-x4v3.pth"
UPSCALE_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
