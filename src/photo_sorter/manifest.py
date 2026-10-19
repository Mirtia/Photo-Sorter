"""Lineage manifests that trace enhanced hashes and discards back to originals.

Phase 2 collapses each similar group down to the kept representative(s), and
Phase 3 renames every surviving picture to the MD5 of its *enhanced* bytes.
Neither step is reversible on its own, so each writes a ``manifest.jsonl`` into
its output directory:

* ``deduped/manifest.jsonl`` — one row per source picture: its ``group_id``,
  perceptual hash, whether it was kept, and where it landed (deduped tree or
  discard bin).
* ``enhanced/manifest.jsonl`` — one row per surviving picture: the enhanced
  hash (or a blur-discard note) linked back to the original path and group.

Together they let you go ``enhanced_hash -> original_path -> group_id`` and find
the near-duplicates that were dropped in favour of a kept shot.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from photo_sorter.media import as_path

logger = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.jsonl"


def manifest_path(directory: Path | str) -> Path:
    """Return the manifest file path inside ``directory``."""
    return as_path(directory) / MANIFEST_NAME


def write_records(directory: Path | str, records: Iterable[dict[str, Any]]) -> Path:
    """Write ``records`` as JSON lines, replacing any existing manifest."""
    target = as_path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / MANIFEST_NAME
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    return path


def read_records(source: Path | str) -> list[dict[str, Any]]:
    """Read JSON-line records from a manifest file or its directory.

    Returns an empty list when the manifest is missing or malformed.
    """
    path = as_path(source)
    if path.is_dir():
        path = path / MANIFEST_NAME
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed manifest line in %s", path)
            continue
        if isinstance(record, dict):
            records.append(record)
    return records


def dedup_record(
    *,
    season: str,
    group_id: str,
    group_size: int,
    original_path: Path,
    phash: str,
    size_bytes: int,
    kept: bool,
    output_path: Path,
) -> dict[str, Any]:
    """Build one Phase 2 lineage row for a source picture."""
    return {
        "phase": "dedup",
        "season": season,
        "group_id": group_id,
        "group_size": group_size,
        "original_path": str(original_path),
        "original_name": original_path.name,
        "phash": phash,
        "size_bytes": size_bytes,
        "kept": kept,
        "output_path": str(output_path),
    }


def load_dedup_lineage(source: Path | str) -> dict[str, dict[str, Any]]:
    """Map each kept deduped picture (resolved path) to its origin and group."""
    lineage: dict[str, dict[str, Any]] = {}
    for record in read_records(source):
        if record.get("phase") != "dedup" or not record.get("kept"):
            continue
        output_path = record.get("output_path")
        if not output_path:
            continue
        key = str(as_path(output_path).resolve())
        lineage[key] = {
            "original_path": record.get("original_path"),
            "original_name": record.get("original_name"),
            "group_id": record.get("group_id"),
        }
    return lineage


def enhance_record(
    *,
    season: str,
    source_path: Path,
    lineage: dict[str, Any] | None,
    enhanced_hash: str | None = None,
    enhanced_path: Path | None = None,
    enhanced: bool = False,
    discarded_blur: bool = False,
    discard_path: Path | None = None,
) -> dict[str, Any]:
    """Build one Phase 3 lineage row for a surviving picture."""
    origin = lineage or {}
    return {
        "phase": "enhance",
        "season": season,
        "group_id": origin.get("group_id"),
        "source_path": str(source_path),
        "original_path": origin.get("original_path") or str(source_path),
        "original_name": origin.get("original_name") or source_path.name,
        "enhanced_hash": enhanced_hash,
        "enhanced_path": str(enhanced_path) if enhanced_path is not None else None,
        "enhanced": enhanced,
        "discarded_blur": discarded_blur,
        "discard_path": str(discard_path) if discard_path is not None else None,
    }
