"""Terminal progress bars and stage banners for the CLI."""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TypeVar

from tqdm import tqdm

T = TypeVar("T")


def progress_disabled() -> bool:
    """Return True when bars should stay quiet (tests, PHOTO_SORTER_DISABLE_PROGRESS)."""
    flag = os.environ.get("PHOTO_SORTER_DISABLE_PROGRESS", "").strip().lower()
    return flag in {"1", "true", "yes"}


def announce(name: str, current: int | None = None, total: int | None = None) -> None:
    """Print the current pipeline stage to stderr."""
    if progress_disabled():
        return
    title = f"Phase {current}/{total} · {name}" if current is not None and total is not None else name
    print(f"\n==> {title}", file=sys.stderr, flush=True)


def _label(item: object) -> str:
    if isinstance(item, Path):
        return item.name
    if isinstance(item, tuple) and item and isinstance(item[0], Path):
        return item[0].name
    return ""


def track(items: Iterable[T], *, desc: str, unit: str = "file") -> Iterator[T]:
    """Iterate ``items`` with a tqdm bar that shows the current file name."""
    sequence: Sequence[T] = (
        items if isinstance(items, Sequence) and not isinstance(items, (str, bytes)) else list(items)
    )
    bar = tqdm(
        sequence,
        desc=desc,
        unit=unit,
        disable=progress_disabled(),
        dynamic_ncols=True,
        file=sys.stderr,
    )
    for item in bar:
        label = _label(item)
        if label:
            bar.set_postfix_str(label, refresh=True)
        yield item


def track_bytes(files: Sequence[Path], *, desc: str) -> Iterator[Path]:
    """Iterate files while showing a byte-scale bar (used for backup)."""
    total = sum(path.stat().st_size for path in files if path.is_file())
    bar = tqdm(
        total=total,
        desc=desc,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        disable=progress_disabled(),
        dynamic_ncols=True,
        file=sys.stderr,
    )
    try:
        for path in files:
            bar.set_postfix_str(path.name, refresh=True)
            yield path
            if path.is_file():
                bar.update(path.stat().st_size)
    finally:
        bar.close()


@contextmanager
def counted(
    total: int | None,
    *,
    desc: str,
    unit: str = "file",
    unit_scale: bool = False,
) -> Iterator[Any]:
    """A manual tqdm bar for loops that are not a single iterable of files."""
    bar = tqdm(
        total=total,
        desc=desc,
        unit=unit,
        unit_scale=unit_scale,
        unit_divisor=1024,
        disable=progress_disabled(),
        dynamic_ncols=True,
        file=sys.stderr,
    )
    try:
        yield bar
    finally:
        bar.close()
