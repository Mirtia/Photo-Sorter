"""Phase 2: group similar pictures and drop duplicates."""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path

from photo_sorter import manifest
from photo_sorter.constants import PHASH_HAMMING_THRESHOLD
from photo_sorter.exceptions import InvalidInputError
from photo_sorter.media import (
    HashArray,
    as_path,
    hamming_distance,
    is_image,
    load_image,
    perceptual_hash,
    phash_to_hex,
    unique_destination,
)
from photo_sorter.picker import gui_enabled, pick_duplicates
from photo_sorter.progress import counted, track
from photo_sorter.sort import PICTURES_DIR, VIDEOS_DIR

logger = logging.getLogger(__name__)

Selector = Callable[[Sequence[Path]], Sequence[Path]]


def _parent_union(parents: list[int], index: int) -> int:
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


def _hash_pictures(pictures: Sequence[Path]) -> tuple[list[Path], list[HashArray]]:
    existing = [path for path in pictures if path.is_file()]
    if not existing:
        return [], []
    ordered = sorted(existing, key=lambda path: (path.stat().st_size, path.name))
    hashes: list[HashArray] = []
    valid: list[Path] = []
    for path in track(ordered, desc="dedup hash"):
        image = load_image(path)
        if image is None:
            logger.warning("Could not read image for hashing: %s", path)
            continue
        hashes.append(perceptual_hash(image))
        valid.append(path)
    return valid, hashes


def _group_indices(hashes: Sequence[HashArray], hamming_threshold: int) -> list[list[int]]:
    count = len(hashes)
    parents = list(range(count))
    for i in range(count):
        for j in range(i + 1, count):
            if hamming_distance(hashes[i], hashes[j]) <= hamming_threshold:
                left = _parent_union(parents, i)
                right = _parent_union(parents, j)
                if left != right:
                    parents[right] = left
    grouped: dict[int, list[int]] = {}
    for index in range(count):
        grouped.setdefault(_parent_union(parents, index), []).append(index)
    return list(grouped.values())


def group_similar_pictures(
    pictures: Sequence[Path],
    *,
    hamming_threshold: int = PHASH_HAMMING_THRESHOLD,
) -> list[list[Path]]:
    """Group pictures by file size order and perceptual-hash similarity."""
    valid, hashes = _hash_pictures(pictures)
    if not valid:
        return []
    return [[valid[index] for index in group] for group in _group_indices(hashes, hamming_threshold)]


def _keep_largest(group: Sequence[Path]) -> list[Path]:
    return [max(group, key=lambda path: (path.stat().st_size, path.name))]


def _prompt_selection(group: Sequence[Path]) -> list[Path]:
    print("Similar pictures:")
    for index, path in enumerate(group, start=1):
        size_kb = path.stat().st_size / 1024
        print(f"  [{index}] {path.name} ({size_kb:.1f} KiB) — {path}")
    raw = input("Keep which numbers (comma-separated), 'all', or empty for largest: ").strip()
    if not raw:
        return _keep_largest(group)
    if raw.lower() == "all":
        return list(group)
    kept: list[Path] = []
    for token in raw.split(","):
        token = token.strip()
        if not token.isdigit():
            continue
        choice = int(token)
        if 1 <= choice <= len(group):
            kept.append(group[choice - 1])
    return kept or _keep_largest(group)


def _manual_selection(group: Sequence[Path]) -> list[Path]:
    if len(group) <= 1:
        return list(group)
    if gui_enabled():
        try:
            chosen = pick_duplicates(group)
        except Exception:
            logger.exception("Duplicate picker failed; falling back to a text prompt")
        else:
            return chosen
    return _prompt_selection(group)


def _materialize_groups(
    groups: Sequence[Sequence[Path]],
    temp_root: Path,
) -> list[list[tuple[Path, Path]]]:
    """Copy each group into ``temp_root/<index>`` and keep the (copy, original) link."""
    materialized: list[list[tuple[Path, Path]]] = []
    for group_id, group in enumerate(groups):
        group_dir = temp_root / str(group_id)
        group_dir.mkdir(parents=True, exist_ok=True)
        pairs: list[tuple[Path, Path]] = []
        for path in group:
            target = unique_destination(group_dir, path.name)
            shutil.copy2(path, target)
            pairs.append((target, path))
        materialized.append(pairs)
    return materialized


def _copy_tree_videos(season_dir: Path, destination_season: Path) -> None:
    source_videos = season_dir / VIDEOS_DIR
    dest_videos = destination_season / VIDEOS_DIR
    dest_videos.mkdir(parents=True, exist_ok=True)
    if not source_videos.is_dir():
        return
    for path in source_videos.iterdir():
        if path.is_file():
            shutil.copy2(path, unique_destination(dest_videos, path.name))


def _season_dirs(sorted_directory: Path) -> list[Path]:
    return sorted(path for path in sorted_directory.iterdir() if path.is_dir())


def deduplicate_sorted_directory(
    input_directory: Path | str,
    output_directory: Path | str,
    discard_directory: Path | str,
    *,
    enable_manual_selection: bool = False,
    selector: Selector | None = None,
    temp_directory: Path | str | None = None,
    hamming_threshold: int = PHASH_HAMMING_THRESHOLD,
) -> Path:
    """Keep the best picture in each similar group; copy discards aside."""
    source = as_path(input_directory)
    if not source.is_dir():
        raise InvalidInputError(source)

    destination = as_path(output_directory)
    discarded = as_path(discard_directory)
    destination.mkdir(parents=True, exist_ok=True)
    discarded.mkdir(parents=True, exist_ok=True)

    choose = selector
    if choose is None:
        choose = _manual_selection if enable_manual_selection else _keep_largest

    temp_root = as_path(temp_directory) if temp_directory is not None else Path(tempfile.gettempdir())
    records: list[dict[str, object]] = []

    for season_dir in _season_dirs(source):
        dest_season = destination / season_dir.name
        pictures_out = dest_season / PICTURES_DIR
        pictures_out.mkdir(parents=True, exist_ok=True)
        _copy_tree_videos(season_dir, dest_season)

        pictures_in = season_dir / PICTURES_DIR
        if not pictures_in.is_dir():
            continue
        pictures = [path for path in pictures_in.iterdir() if path.is_file() and is_image(path)]
        valid, hashes = _hash_pictures(pictures)
        if not valid:
            continue
        phash_hex = {valid[index]: phash_to_hex(hashes[index]) for index in range(len(valid))}
        groups = [[valid[index] for index in group] for group in _group_indices(hashes, hamming_threshold)]

        work_root = temp_root / season_dir.name
        if work_root.exists():
            shutil.rmtree(work_root)
        work_root.mkdir(parents=True, exist_ok=True)
        try:
            staged_groups = _materialize_groups(groups, work_root)
            decisions = [
                (pairs, {copy.resolve() for copy in choose([copy for copy, _ in pairs])}) for pairs in staged_groups
            ]
            total = sum(len(pairs) for pairs, _keep in decisions)
            with counted(total, desc=f"dedup {season_dir.name}") as bar:
                for group_index, (pairs, keep) in enumerate(decisions):
                    group_id = f"{season_dir.name}:{group_index}"
                    for copy, original in pairs:
                        bar.set_postfix_str(original.name, refresh=True)
                        kept = copy.resolve() in keep
                        if kept:
                            output_path = unique_destination(pictures_out, original.name)
                        else:
                            output_path = unique_destination(discarded / season_dir.name, original.name)
                        shutil.copy2(copy, output_path)
                        records.append(
                            manifest.dedup_record(
                                season=season_dir.name,
                                group_id=group_id,
                                group_size=len(pairs),
                                original_path=original,
                                phash=phash_hex[original],
                                size_bytes=original.stat().st_size,
                                kept=kept,
                                output_path=output_path,
                            )
                        )
                        bar.update(1)
        finally:
            shutil.rmtree(work_root, ignore_errors=True)

    manifest.write_records(destination, records)
    return destination
