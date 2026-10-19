"""CLIP image/text embeddings for semantic search."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from PIL import Image

from photo_sorter.constants import CLIP_MODEL_NAME, CLIP_PRETRAINED
from photo_sorter.detect import ml_enabled
from photo_sorter.device import torch_device, use_fp16
from photo_sorter.media import ImageArray, as_uint8

logger = logging.getLogger(__name__)

Embedder = Callable[[ImageArray], list[float] | None]
TextEmbedder = Callable[[str], list[float] | None]

# (short label used in captions, CLIP text prompt)
SCENE_CAPTION_PROMPTS: tuple[tuple[str, str], ...] = (
    ("a portrait", "a portrait photo of a person"),
    ("a group photo", "a group photo of several people together"),
    ("a selfie", "a close-up selfie taken with a phone"),
    ("a beach scene", "a photo of a beach, sand, and the sea"),
    ("a mountain landscape", "a photo of mountains and a wide landscape"),
    ("a city street", "a photo of a city street with buildings and traffic"),
    ("an indoor scene", "a photo taken indoors inside a room"),
    ("an outdoor scene", "a photo taken outdoors in nature"),
    ("a meal", "a photo of food, a plate, or a meal on a table"),
    ("a sunset", "a photo of a sunset or sunrise sky"),
    ("a snowy winter scene", "a photo of snow and winter weather"),
    ("a park or garden", "a photo of a park, garden, trees, or plants"),
    ("vehicles", "a photo of cars, bikes, or other vehicles"),
    ("a pet", "a photo of a dog or a cat"),
    ("a night scene", "a photo taken at night with artificial lights"),
    ("a document", "a photo of a document, screenshot, or printed text"),
    ("a celebration", "a photo of a party, concert, or public event"),
    ("a body of water", "a photo of a lake, river, pool, or the ocean"),
)


@lru_cache(maxsize=1)
def _clip_bundle() -> tuple[Any, Any, Any] | None:
    if not ml_enabled():
        return None
    try:
        import open_clip
    except ImportError:
        logger.info("open-clip-torch is not installed; skipping image embeddings")
        return None
    try:
        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL_NAME,
            pretrained=CLIP_PRETRAINED,
        )
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    except Exception:
        logger.warning("Could not load CLIP model %s", CLIP_MODEL_NAME)
        return None
    model.eval()
    try:
        model.to(torch_device())
        if use_fp16():
            model.half()
    except Exception:
        logger.debug("Keeping CLIP on CPU; GPU move failed")
    return model, preprocess, tokenizer


def _to_pil(image: ImageArray) -> Image.Image:
    rgb = as_uint8(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return Image.fromarray(rgb)


def _normalise(vector: np.ndarray) -> list[float]:
    flat = np.asarray(vector, dtype=np.float32).flatten()
    norm = float(np.linalg.norm(flat))
    if norm == 0:
        return [float(value) for value in flat]
    return [float(value) for value in (flat / norm)]


_image_override: Embedder | None = None
_text_override: TextEmbedder | None = None


def use_embedders(
    image_embedder: Embedder | None = None,
    text_embedder: TextEmbedder | None = None,
) -> None:
    """Override embedders (used in tests). Pass None to restore defaults."""
    global _image_override, _text_override
    _image_override = image_embedder
    _text_override = text_embedder
    _cached_scene_vectors.cache_clear()


def embed_image(image: ImageArray) -> list[float] | None:
    """Return a unit CLIP vector for ``image``, or None if CLIP is unavailable."""
    if _image_override is not None:
        return _image_override(image)
    bundle = _clip_bundle()
    if bundle is None:
        return None
    model, preprocess, _tokenizer = bundle
    try:
        import torch
    except ImportError:
        return None
    tensor = preprocess(_to_pil(image)).unsqueeze(0).to(torch_device())
    if use_fp16():
        tensor = tensor.half()
    with torch.no_grad():
        encoded = model.encode_image(tensor)
    return _normalise(encoded.float().cpu().numpy())


def embed_text(text: str) -> list[float] | None:
    """Return a unit CLIP vector for a search query, or None if unavailable."""
    if _text_override is not None:
        return _text_override(text)
    if not text.strip():
        return None
    bundle = _clip_bundle()
    if bundle is None:
        return None
    model, _preprocess, tokenizer = bundle
    try:
        import torch
    except ImportError:
        return None
    tokens = tokenizer([text]).to(torch_device())
    with torch.no_grad():
        encoded = model.encode_text(tokens)
    return _normalise(encoded.float().cpu().numpy())


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Cosine similarity of two already-normalised vectors."""
    length = min(len(left), len(right))
    if length == 0:
        return 0.0
    return float(np.dot(left[:length], right[:length]))


def _embed_prompt_pairs(prompts: Sequence[tuple[str, str]]) -> list[tuple[str, list[float]]]:
    pairs: list[tuple[str, list[float]]] = []
    for label, prompt in prompts:
        vector = embed_text(prompt)
        if vector:
            pairs.append((label, vector))
    return pairs


@lru_cache(maxsize=1)
def _cached_scene_vectors() -> tuple[tuple[str, tuple[float, ...]], ...]:
    return tuple((label, tuple(vector)) for label, vector in _embed_prompt_pairs(SCENE_CAPTION_PROMPTS))


def rank_scene_captions(
    embedding: Sequence[float],
    *,
    limit: int = 3,
) -> list[tuple[str, float]]:
    """Rank short scene labels against a CLIP image embedding."""
    if not embedding:
        return []
    if _text_override is not None:
        pairs = _embed_prompt_pairs(SCENE_CAPTION_PROMPTS)
    else:
        pairs = [(label, list(vector)) for label, vector in _cached_scene_vectors()]
    image_vector = list(embedding)
    scored = [(label, cosine_similarity(image_vector, vector)) for label, vector in pairs]
    scored.sort(key=lambda item: item[1], reverse=True)
    return scored[:limit]
