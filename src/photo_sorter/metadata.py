"""Phase 4: extract EXIF, detect topics, and match faces against samples."""

from __future__ import annotations

import json
import logging
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import ExifTags, Image

from photo_sorter.constants import TOP_OBJECT_COUNT, TOP_TOPIC_COUNT
from photo_sorter.detect import (
    SCENE_LABELS,
    DetectedObject,
    detect_objects,
    heuristic_objects,
    ml_enabled,
    season_from_path,
)
from photo_sorter.device import describe
from photo_sorter.embed import embed_image
from photo_sorter.exceptions import InvalidInputError, InvalidOptionError
from photo_sorter.faces import backend_name, count_and_people, reference_embeddings
from photo_sorter.media import ImageArray, as_path, as_uint8, capture_datetime, exif_datetime, is_image, load_image
from photo_sorter.progress import track

logger = logging.getLogger(__name__)

Embedding = NDArray[np.float32]


class Topic(TypedDict):
    name: str
    confidence: float


def _embeddings_from_image(path: Path) -> list[Embedding]:
    image = load_image(path)
    if image is None:
        return []
    return reference_embeddings(image)


def _add_sample(samples: dict[str, list[Embedding]], person: str, embeddings: list[Embedding]) -> None:
    if not embeddings:
        return
    samples.setdefault(person, []).extend(embeddings)


def load_face_samples(samples_directory: Path) -> dict[str, list[Embedding]]:
    """Load face embeddings keyed by person name.

    Accepts ``samples/<Name>/*.jpg`` and loose files such as ``samples/Myrto.jpg``.
    Only photos in which a face can be detected are used as references.
    """
    samples: dict[str, list[Embedding]] = {}
    if not samples_directory.is_dir():
        return samples
    for child in sorted(samples_directory.iterdir()):
        if child.is_dir():
            person = child.name.replace("_", " ")
            embeddings: list[Embedding] = []
            for path in sorted(child.iterdir()):
                if path.is_file() and is_image(path):
                    embeddings.extend(_embeddings_from_image(path))
            _add_sample(samples, person, embeddings)
        elif child.is_file() and is_image(child):
            person = child.stem.replace("_", " ")
            _add_sample(samples, person, _embeddings_from_image(child))
    if samples:
        logger.info(
            "Loaded face samples for %s",
            ", ".join(f"{name} ({len(vectors)})" for name, vectors in sorted(samples.items())),
        )
    else:
        logger.warning("No faces found in reference photos under %s", samples_directory)
    return samples


_SMALL_COUNTS = (
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
)
_IRREGULAR_PLURALS = {
    "person": "people",
    "man": "men",
    "woman": "women",
    "child": "children",
    "sheep": "sheep",
    "knife": "knives",
    "mouse": "mice",
    "bus": "buses",
    "bench": "benches",
    "sandwich": "sandwiches",
    "toothbrush": "toothbrushes",
    "skis": "skis",
    "scissors": "scissors",
}
_SCENE_OPENINGS = {
    "indoor": "An indoor photo",
    "outdoor": "An outdoor photo",
    "sunset": "A sunset photo",
    "snow": "A snowy photo",
    "water": "A photo by the water",
    "vegetation": "A photo of greenery",
    "sky": "A photo under the sky",
}


def detect_topics(image: ImageArray) -> list[Topic]:
    """Return the top objects/topics for ``image`` (YOLO or colour fallback)."""
    return [Topic(name=item.name, confidence=item.confidence) for item in detect_objects(image, limit=TOP_TOPIC_COUNT)]


def _gps_decimal(values: Any, ref: str) -> float | None:
    try:
        degrees, minutes, seconds = values
        decimal = float(degrees) + float(minutes) / 60.0 + float(seconds) / 3600.0
    except (TypeError, ValueError):
        return None
    if ref in {"S", "W"}:
        decimal = -decimal
    return decimal


def extract_location(path: Path) -> dict[str, float] | None:
    """Return GPS latitude/longitude from EXIF when present."""
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return None
            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo) if hasattr(ExifTags, "IFD") else None
            if not gps_ifd:
                gps_ifd = exif.get(ExifTags.Base.GPSInfo) if hasattr(ExifTags, "Base") else None
            if not isinstance(gps_ifd, dict):
                return None
            names = {ExifTags.GPSTAGS.get(key, key): value for key, value in gps_ifd.items()}
            lat = _gps_decimal(names.get("GPSLatitude"), str(names.get("GPSLatitudeRef", "N")))
            lon = _gps_decimal(names.get("GPSLongitude"), str(names.get("GPSLongitudeRef", "E")))
            if lat is None or lon is None:
                return None
            return {"latitude": lat, "longitude": lon}
    except (OSError, ValueError, KeyError):
        return None


def extract_camera(path: Path) -> dict[str, str] | None:
    """Return camera make/model from EXIF when present."""
    try:
        with Image.open(path) as image:
            exif = image.getexif()
            if not exif:
                return None
            tags = {str(ExifTags.TAGS.get(key, key)): value for key, value in exif.items()}
    except (OSError, ValueError, KeyError):
        return None
    make = str(tags.get("Make") or "").strip()
    model = str(tags.get("Model") or "").strip()
    if not make and not model:
        return None
    camera: dict[str, str] = {}
    if make:
        camera["make"] = make
    if model:
        camera["model"] = model
    return camera


def _image_hash_from_name(path: Path) -> str:
    return path.stem


def orientation_label(width: int, height: int) -> str:
    """Return landscape, portrait, or square from pixel size."""
    if width > height * 1.15:
        return "landscape"
    if height > width * 1.15:
        return "portrait"
    return "square"


def lighting_label(image: ImageArray) -> str:
    """Coarse lighting from mean brightness: night, dim, or bright."""
    gray = as_uint8(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    mean_l = float(np.mean(gray))
    if mean_l < 45:
        return "night"
    if mean_l < 85:
        return "dim"
    if mean_l > 190:
        return "bright"
    return "daylight"


def time_of_day_label(taken: datetime) -> str:
    hour = taken.hour
    if hour < 5 or hour >= 21:
        return "night"
    if hour < 12:
        return "morning"
    if hour < 17:
        return "afternoon"
    return "evening"


def _article(noun: str) -> str:
    first = noun.strip().split()[0] if noun.strip() else ""
    if first[:1].lower() in {"a", "e", "i", "o", "u"}:
        return "an"
    return "a"


def _plural(name: str, count: int) -> str:
    if count == 1:
        return name
    if name in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[name]
    if name.endswith(("s", "x", "z", "ch", "sh")):
        return f"{name}es"
    if len(name) > 1 and name.endswith("y") and name[-2] not in "aeiou":
        return f"{name[:-1]}ies"
    return f"{name}s"


def count_objects(names: list[str]) -> dict[str, int]:
    """Return object name counts, preserving first-seen order."""
    counts = Counter(names)
    return {name: counts[name] for name in dict.fromkeys(names)}


def _count_phrase(name: str, count: int) -> str:
    noun = _plural(name, count)
    if count == 1:
        return f"{_article(name)} {noun}"
    if 2 <= count <= 12:
        return f"{_SMALL_COUNTS[count - 1]} {noun}"
    return f"{count} {noun}"


def _join_phrases(parts: list[str]) -> str:
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return f"{', '.join(parts[:-1])}, and {parts[-1]}"


def _season_words(season: str | None) -> str | None:
    if not season:
        return None
    return season.replace("_", " ")


def _opening(
    caption: str | None,
    scene: list[DetectedObject],
    orientation: str,
) -> str:
    if caption:
        return caption[:1].upper() + caption[1:]
    for item in scene:
        opening = _SCENE_OPENINGS.get(item.name)
        if opening:
            return opening
    if orientation == "portrait":
        return "A portrait photo"
    if orientation == "landscape":
        return "A landscape photo"
    return "A photo"


def build_description(
    *,
    objects: list[DetectedObject],
    scene: list[DetectedObject],
    people: list[str],
    face_count: int,
    orientation: str,
    lighting: str,
    caption: str | None,
    taken_phrase: str | None,
) -> str:
    """Build a searchable natural-language caption from detected facts.

    ``face_count`` is the number of detected face boxes. It is not a count of
    people in the scene (Haar texture hits are ignored on collection photos).
    """
    opening = _opening(caption, scene, orientation)
    counted = [_count_phrase(name, count) for name, count in count_objects([item.name for item in objects]).items()]
    clauses = [opening]
    if counted:
        clauses[0] = f"{opening} showing {_join_phrases(counted)}"
    elif scene and caption is None:
        scene_names = [item.name for item in scene[:2]]
        clauses[0] = f"{opening} with {_join_phrases(scene_names)}"
    extras: list[str] = []
    if people:
        extras.append(f"featuring {_join_phrases(list(people))}")
    elif face_count == 1:
        extras.append("with one visible face")
    elif face_count > 1:
        extras.append(f"with {face_count} visible faces")
    if taken_phrase:
        extras.append(taken_phrase)
    elif lighting in {"night", "dim"}:
        extras.append(f"in {lighting} light")
    sentence = clauses[0]
    if extras:
        sentence = f"{sentence}, {', '.join(extras)}"
    if not sentence.endswith("."):
        sentence += "."
    return sentence


def _taken_phrase(season: str | None, captured: datetime | None) -> str | None:
    bits: list[str] = []
    words = _season_words(season)
    if words:
        bits.append(words)
    if captured is not None:
        bits.append(f"the {time_of_day_label(captured)}")
    if not bits:
        return None
    if len(bits) == 1:
        return f"taken in {bits[0]}"
    return f"taken in {bits[0]} in {bits[1]}"


def _unique_topics(
    objects: list[DetectedObject],
    scene: list[DetectedObject],
) -> list[Topic]:
    topics: list[Topic] = []
    seen: set[str] = set()
    for item in [*objects, *scene]:
        if item.name in seen:
            continue
        seen.add(item.name)
        topics.append(Topic(name=item.name, confidence=float(item.confidence)))
        if len(topics) >= TOP_TOPIC_COUNT:
            break
    return topics


def build_metadata(
    path: Path,
    image: ImageArray,
    samples: dict[str, list[Embedding]],
) -> dict[str, Any]:
    """Assemble the metadata document for a single enhanced image."""
    height, width = image.shape[:2]
    detected = detect_objects(image, limit=TOP_OBJECT_COUNT)
    objects = [item for item in detected if item.name not in SCENE_LABELS]
    scene = [item for item in heuristic_objects(image, limit=6) if item.confidence >= 0.12]
    face_count, people = count_and_people(image, samples)
    labels = list(dict.fromkeys(item.name for item in objects))
    scene_names = list(dict.fromkeys(item.name for item in scene))
    taken = capture_datetime(path)
    captured = exif_datetime(path)
    season = season_from_path(path.parts)
    orientation = orientation_label(width, height)
    lighting = lighting_label(image)
    embedding = embed_image(image)
    tags = list(dict.fromkeys([*labels, *scene_names, *people, orientation, lighting]))
    season_words = _season_words(season)
    if season_words:
        tags.append(season_words)
    return {
        "filename": path.name,
        "hash": _image_hash_from_name(path),
        "path": str(path),
        "season": season,
        "date_taken": taken.isoformat(timespec="seconds"),
        "time_of_day": time_of_day_label(captured) if captured else None,
        "location": extract_location(path),
        "camera": extract_camera(path),
        "orientation": orientation,
        "lighting": lighting,
        "scene": [item.to_dict() for item in scene],
        "objects": [item.to_dict() for item in objects],
        "object_labels": labels,
        "object_counts": count_objects([item.name for item in objects]),
        "topics": _unique_topics(objects, scene),
        "people": people,
        "face_count": face_count,
        "tags": tags,
        "description": build_description(
            objects=objects,
            scene=scene,
            people=people,
            face_count=face_count,
            orientation=orientation,
            lighting=lighting,
            caption=None,
            taken_phrase=_taken_phrase(season, captured),
        ),
        "resolution": {"width": int(width), "height": int(height)},
        "embedding": embedding,
    }


def _log_backends() -> None:
    if not ml_enabled():
        return
    logger.info("Compute device: %s", describe())
    logger.info("Face backend: %s", backend_name())


def generate_metadata_directory(
    input_directory: Path | str,
    output_directory: Path | str,
    face_samples_directory: Path | str,
) -> Path:
    """Write ``<hash>_metadata.json`` for every picture under ``input_directory``."""
    source = as_path(input_directory)
    if not source.is_dir():
        raise InvalidInputError(source)

    samples_dir = as_path(face_samples_directory)
    if not samples_dir.is_dir():
        raise InvalidInputError(samples_dir)

    destination = as_path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    _log_backends()
    samples = load_face_samples(samples_dir)
    written: set[str] = set()

    image_paths = sorted(path for path in source.rglob("*") if path.is_file() and is_image(path))
    for path in track(image_paths, desc="metadata"):
        image = load_image(path)
        if image is None:
            logger.warning("Skipping unreadable image: %s", path)
            continue
        document = build_metadata(path, image, samples)
        key = str(document["hash"])
        if key in written:
            continue
        written.add(key)
        target = destination / f"{key}_metadata.json"
        target.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    return destination


def _detected_from_payload(items: object) -> list[DetectedObject]:
    objects: list[DetectedObject] = []
    if not isinstance(items, list):
        return objects
    for item in items:
        if not isinstance(item, dict) or not item.get("name"):
            continue
        try:
            confidence = float(item.get("confidence") or 0.0)
        except (TypeError, ValueError):
            confidence = 0.0
        objects.append(DetectedObject(name=str(item["name"]), confidence=confidence))
    return objects


def _taken_phrase_from_document(document: dict[str, Any]) -> str | None:
    season = document.get("season")
    words = _season_words(str(season) if season else None)
    time_of_day = document.get("time_of_day")
    bits: list[str] = []
    if words:
        bits.append(words)
    if isinstance(time_of_day, str) and time_of_day.strip():
        bits.append(f"the {time_of_day.strip()}")
    if not bits:
        return None
    if len(bits) == 1:
        return f"taken in {bits[0]}"
    return f"taken in {bits[0]} in {bits[1]}"


def _tags_with_people(document: dict[str, Any], people: list[str]) -> list[str]:
    labels = [str(name) for name in document.get("object_labels") or [] if name]
    scene_names: list[str] = []
    for item in document.get("scene") or []:
        if isinstance(item, dict) and item.get("name"):
            scene_names.append(str(item["name"]))
    parts = [
        *labels,
        *scene_names,
        *people,
        str(document.get("orientation") or ""),
        str(document.get("lighting") or ""),
        _season_words(str(document["season"])) if document.get("season") else None,
    ]
    return list(dict.fromkeys(part for part in parts if part))


def apply_face_fields(
    document: dict[str, Any],
    people: list[str],
    face_count: int,
) -> dict[str, Any]:
    """Patch people, face_count, tags, and description; leave other fields intact."""
    document["people"] = people
    document["face_count"] = face_count
    document["tags"] = _tags_with_people(document, people)
    document["description"] = build_description(
        objects=_detected_from_payload(document.get("objects")),
        scene=_detected_from_payload(document.get("scene")),
        people=people,
        face_count=face_count,
        orientation=str(document.get("orientation") or "square"),
        lighting=str(document.get("lighting") or "daylight"),
        caption=None,
        taken_phrase=_taken_phrase_from_document(document),
    )
    return document


def _image_path_for_document(document: dict[str, Any], images_directory: Path) -> Path | None:
    raw = document.get("path")
    if raw:
        path = Path(str(raw))
        if path.is_file():
            return path
    names: list[str] = []
    filename = document.get("filename")
    if filename:
        names.append(str(filename))
    digest = document.get("hash")
    if digest:
        names.extend([f"{digest}.jpg", f"{digest}.jpeg"])
    for name in names:
        hits = sorted(path for path in images_directory.rglob(name) if path.is_file())
        if hits:
            return hits[0]
    return None


def refresh_face_metadata(
    images_directory: Path | str,
    metadata_directory: Path | str,
    face_samples_directory: Path | str,
) -> Path:
    """Re-run face detection/recognition on existing JSON. Does not touch objects or CLIP."""
    source = as_path(images_directory)
    if not source.is_dir():
        raise InvalidInputError(source)
    destination = as_path(metadata_directory)
    if not destination.is_dir():
        raise InvalidInputError(destination)
    samples_dir = as_path(face_samples_directory)
    if not samples_dir.is_dir():
        raise InvalidInputError(samples_dir)

    documents = sorted(destination.glob("*_metadata.json"))
    if not documents:
        raise InvalidOptionError("faces-only", f"no metadata JSON in {destination}")

    samples = load_face_samples(samples_dir)
    _log_backends()
    for path in track(documents, desc="faces"):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping unreadable metadata: %s", path)
            continue
        if not isinstance(document, dict):
            logger.warning("Skipping unreadable metadata: %s", path)
            continue
        image_path = _image_path_for_document(document, source)
        if image_path is None:
            logger.warning("Skipping metadata with missing image: %s", path)
            continue
        image = load_image(image_path)
        if image is None:
            logger.warning("Skipping unreadable image: %s", image_path)
            continue
        face_count, people = count_and_people(image, samples)
        apply_face_fields(document, people, face_count)
        path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return destination
