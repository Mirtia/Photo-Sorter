"""Load pipeline options from a YAML config file."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from photo_sorter.constants import (
    DEFAULT_CONFIG_FILENAMES,
    DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_SEARCH_INDEX,
    DEFAULT_SEARCH_RESULTS,
    ENHANCEMENT_LEVELS,
    UPSCALE_LEVELS,
)
from photo_sorter.exceptions import ConfigError, MissingOptionError

OPTION_FLAGS: dict[str, tuple[str, ...]] = {
    "verbose": ("-v", "--verbose"),
    "input": ("--input",),
    "archive": ("--archive",),
    "output": ("--output",),
    "discard": ("--discard",),
    "face_samples": ("--face-samples",),
    "enable_manual_selection": ("--enable-manual-selection",),
    "quality_threshold": ("--quality-threshold",),
    "apply_color_correction": ("--apply-color-correction",),
    "apply_denoising": ("--apply-denoising",),
    "apply_sharpening": ("--apply-sharpening",),
    "apply_artifact_removal": ("--apply-artifact-removal",),
    "apply_upscaling": ("--apply-upscaling",),
    "images": ("--images",),
    "metadata": ("--metadata",),
    "es_url": ("--es-url",),
    "es_index": ("--es-index",),
    "person": ("--person",),
    "object_label": ("--object",),
    "date_from": ("--from",),
    "date_to": ("--to",),
    "limit": ("--limit",),
    "query": (),
    "from_phase": ("--from-phase",),
    "faces_only": ("--faces-only",),
    "yolo_model": ("--yolo-model",),
}

PATH_ALIASES = {
    "input": "input",
    "archive": "archive",
    "output": "output",
    "discard": "discard",
    "face_samples": "face_samples",
    "face-samples": "face_samples",
    "images": "images",
    "enhanced": "images",
    "metadata": "metadata",
}

ML_ALIASES = {
    "yolo_model": "yolo_model",
    "yolo-model": "yolo_model",
}

SEARCH_ALIASES = {
    "es_url": "es_url",
    "es-url": "es_url",
    "es_index": "es_index",
    "es-index": "es_index",
    "query": "query",
    "person": "person",
    "object": "object_label",
    "object_label": "object_label",
    "from": "date_from",
    "to": "date_to",
    "limit": "limit",
}

LEVEL_DESTS = frozenset({"apply_denoising", "apply_sharpening", "apply_upscaling"})
BOOL_DESTS = frozenset({
    "verbose",
    "enable_manual_selection",
    "apply_color_correction",
    "apply_artifact_removal",
    "faces_only",
})

REQUIRED_BY_COMMAND: dict[str, tuple[str, ...]] = {
    "backup": ("input", "archive"),
    "sort": ("input", "output"),
    "dedup": ("input", "output", "discard"),
    "enhance": ("input", "output"),
    "metadata": ("input", "output", "face_samples"),
    "index": ("metadata", "es_url"),
    "run": ("input", "archive", "output", "discard", "face_samples"),
}

FILL_DEFAULTS: dict[str, Any] = {
    "quality_threshold": DEFAULT_QUALITY_THRESHOLD,
    "es_index": DEFAULT_SEARCH_INDEX,
    "limit": DEFAULT_SEARCH_RESULTS,
    "from_phase": "backup",
}


def cli_has_option(argv: Sequence[str], flags: Sequence[str]) -> bool:
    """Return True if any of ``flags`` appears on the command line."""
    for argument in argv:
        for flag in flags:
            if argument == flag or argument.startswith(f"{flag}="):
                return True
    return False


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off", ""}:
        return False
    raise ConfigError(value)


def _as_level(value: object, allowed: tuple[str, ...] = ENHANCEMENT_LEVELS) -> str | None:
    if value is None or value is False:
        return None
    if value is True:
        return "medium"
    text = str(value).strip().lower()
    if text in {"true", "yes", "on"}:
        return "medium"
    if text in {"false", "no", "off", "null", "none", ""}:
        return None
    if text not in allowed:
        raise ConfigError(value)
    return text


def coerce_value(dest: str, value: object) -> object:
    """Normalise a YAML value to the type argparse would produce."""
    if dest in BOOL_DESTS:
        return _as_bool(value)
    if dest == "apply_upscaling":
        return _as_level(value, UPSCALE_LEVELS)
    if dest in LEVEL_DESTS:
        return _as_level(value)
    if dest == "quality_threshold":
        try:
            return float(str(value))
        except ValueError as exc:
            raise ConfigError(value) from exc
    if dest == "limit":
        try:
            return int(str(value))
        except ValueError as exc:
            raise ConfigError(value) from exc
    if dest == "from_phase":
        return str(value).strip().lower()
    if value is None:
        return None
    return value


def _merge_section(raw: dict[str, Any], aliases: dict[str, str], dest: dict[str, Any]) -> None:
    for key, value in raw.items():
        mapped = aliases.get(str(key))
        if mapped is not None:
            dest[mapped] = value


def _merge_optional_section(
    raw: dict[str, Any],
    key: str,
    aliases: dict[str, str],
    dest: dict[str, Any],
) -> None:
    section = raw.get(key)
    if isinstance(section, dict):
        _merge_section(section, aliases, dest)


def flatten_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Map nested YAML keys onto argparse destination names."""
    flat: dict[str, Any] = {}
    _merge_optional_section(raw, "paths", PATH_ALIASES, flat)
    dedup = raw.get("dedup")
    if isinstance(dedup, dict) and "enable_manual_selection" in dedup:
        flat["enable_manual_selection"] = dedup["enable_manual_selection"]
    enhance = raw.get("enhance")
    if isinstance(enhance, dict):
        for key, value in enhance.items():
            dest = str(key).replace("-", "_")
            if dest in OPTION_FLAGS:
                flat[dest] = value
    _merge_optional_section(raw, "search", SEARCH_ALIASES, flat)
    _merge_optional_section(raw, "ml", ML_ALIASES, flat)
    for key, value in raw.items():
        if key in {"paths", "dedup", "enhance", "search", "ml"}:
            continue
        dest = PATH_ALIASES.get(str(key), str(key).replace("-", "_"))
        if dest in OPTION_FLAGS:
            flat[dest] = value
    return flat


def load_yaml_file(path: Path) -> dict[str, Any]:
    """Read a YAML mapping from ``path``."""
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(path) from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(path)
    return loaded


def resolve_config_path(explicit: str | None) -> Path | None:
    """Return the config file to load, or None if none is present."""
    if explicit:
        path = Path(explicit).expanduser()
        if not path.is_file():
            raise ConfigError(path)
        return path
    flag = os.environ.get("PHOTO_SORTER_DISABLE_AUTOCONFIG", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return None
    for name in DEFAULT_CONFIG_FILENAMES:
        candidate = Path(name)
        if candidate.is_file():
            return candidate
    return None


def load_user_config(explicit: str | None) -> dict[str, Any]:
    """Load and flatten YAML from ``explicit`` or a default filename in cwd."""
    path = resolve_config_path(explicit)
    if path is None:
        return {}
    return flatten_config(load_yaml_file(path))


def apply_config(args: argparse.Namespace, config: dict[str, Any], argv: Sequence[str]) -> None:
    """Fill unset argparse fields from YAML. Explicit CLI flags always win."""
    for dest, value in config.items():
        if dest not in OPTION_FLAGS or not hasattr(args, dest):
            continue
        flags = OPTION_FLAGS[dest]
        if flags and cli_has_option(argv, flags):
            continue
        if dest == "query" and args.query:
            continue
        setattr(args, dest, coerce_value(dest, value))
    for dest, default in FILL_DEFAULTS.items():
        if hasattr(args, dest) and getattr(args, dest) is None:
            setattr(args, dest, default)
    yolo_model = getattr(args, "yolo_model", None) or config.get("yolo_model")
    if yolo_model:
        os.environ["PHOTO_SORTER_YOLO_MODEL"] = str(yolo_model)


def require_command_options(args: argparse.Namespace) -> None:
    """Error if a command is missing options after CLI and YAML are merged."""
    if args.command == "run":
        from photo_sorter.pipeline import parse_from_phase, required_run_options

        if getattr(args, "faces_only", False):
            args.from_phase = "metadata"
        args.from_phase = parse_from_phase(getattr(args, "from_phase", None))
        required = required_run_options(args.from_phase)
    else:
        required = REQUIRED_BY_COMMAND.get(args.command or "", ())
    for name in required:
        if not getattr(args, name, None):
            raise MissingOptionError(name)
