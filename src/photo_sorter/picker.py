"""Lightweight Tk window for choosing which similar pictures to keep."""

from __future__ import annotations

import importlib.util
import os
import tkinter as tk
from collections.abc import Sequence
from pathlib import Path

from PIL import Image, ImageTk

_PREVIEW_MAX = (920, 640)


def gui_enabled() -> bool:
    """Return True when a duplicate-picker window may be shown."""
    flag = os.environ.get("PHOTO_SORTER_DISABLE_GUI", "").strip().lower()
    if flag in {"1", "true", "yes"}:
        return False
    if os.name != "nt":
        display = os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        if not display:
            return False
    return importlib.util.find_spec("tkinter") is not None


def default_selection(paths: Sequence[Path]) -> list[bool]:
    """Pre-check the largest file in ``paths`` (name as a tie-breaker)."""
    if not paths:
        return []
    best = max(range(len(paths)), key=lambda index: (paths[index].stat().st_size, paths[index].name))
    return [index == best for index in range(len(paths))]


def selected_paths(paths: Sequence[Path], selected: Sequence[bool]) -> list[Path]:
    """Return checked paths, or the largest file if nothing is checked."""
    kept = [path for path, flag in zip(paths, selected, strict=True) if flag]
    if kept:
        return kept
    return [max(paths, key=lambda path: (path.stat().st_size, path.name))]


def _load_preview(path: Path) -> Image.Image:
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        image.thumbnail(_PREVIEW_MAX, Image.Resampling.LANCZOS)
        return image.copy()


class _DuplicatePicker:
    def __init__(self, paths: Sequence[Path]) -> None:
        self.paths = list(paths)
        self.selected = default_selection(self.paths)
        self.index = 0
        self.confirmed = False
        self._photo: ImageTk.PhotoImage | None = None

        self.root = tk.Tk()
        self.root.title("Similar pictures")
        self.root.minsize(640, 520)
        self.root.geometry("980x780")

        self.caption = tk.Label(self.root, font=("sans-serif", 12))
        self.caption.pack(pady=(12, 4))

        self.preview = tk.Label(self.root, background="#1c1c1c")
        self.preview.pack(fill=tk.BOTH, expand=True, padx=12, pady=8)

        self.keep_var = tk.BooleanVar(value=self.selected[0])
        self.keep = tk.Checkbutton(
            self.root,
            text="Keep this picture",
            variable=self.keep_var,
            command=self._on_toggle,
            font=("sans-serif", 12),
        )
        self.keep.pack(pady=4)

        hint = tk.Label(
            self.root,
            text="← / → browse   Space check/uncheck   Enter confirm   Esc keep largest",
            font=("sans-serif", 10),
        )
        hint.pack(pady=(0, 8))

        buttons = tk.Frame(self.root)
        buttons.pack(pady=(0, 16))
        tk.Button(buttons, text="← Left", width=10, command=self._prev).pack(side=tk.LEFT, padx=6)
        tk.Button(buttons, text="Right →", width=10, command=self._next).pack(side=tk.LEFT, padx=6)
        tk.Button(buttons, text="OK", width=10, command=self._ok).pack(side=tk.LEFT, padx=6)

        self.root.bind("<Left>", lambda _event: self._prev())
        self.root.bind("<Right>", lambda _event: self._next())
        self.root.bind("<space>", self._on_space)
        self.root.bind("<Return>", lambda _event: self._ok())
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.root.protocol("WM_DELETE_WINDOW", self.root.destroy)
        self._show()

    def _on_space(self, event: object) -> str | None:
        widget = getattr(event, "widget", None)
        if isinstance(widget, tk.Checkbutton):
            return None
        self.keep.toggle()
        return "break"

    def _on_toggle(self) -> None:
        self.selected[self.index] = bool(self.keep_var.get())

    def _prev(self) -> None:
        self.index = (self.index - 1) % len(self.paths)
        self._show()

    def _next(self) -> None:
        self.index = (self.index + 1) % len(self.paths)
        self._show()

    def _ok(self) -> None:
        self.confirmed = True
        self.root.destroy()

    def _show(self) -> None:
        path = self.paths[self.index]
        size_kb = path.stat().st_size / 1024
        self.caption.config(
            text=f"{self.index + 1} / {len(self.paths)}  ·  {path.name}  ({size_kb:.1f} KiB)",
        )
        self.keep_var.set(self.selected[self.index])
        try:
            preview = _load_preview(path)
            self._photo = ImageTk.PhotoImage(preview)
            self.preview.config(image=self._photo, text="")
        except (OSError, ValueError) as exc:
            self._photo = None
            self.preview.config(image="", text=f"Could not load preview\n{path}\n{exc}")

    def run(self) -> list[Path]:
        self.root.mainloop()
        if not self.confirmed:
            return selected_paths(self.paths, default_selection(self.paths))
        return selected_paths(self.paths, self.selected)


def pick_duplicates(paths: Sequence[Path]) -> list[Path]:
    """Open a window to pick keepers. Closing the window keeps the largest file."""
    if len(paths) <= 1:
        return list(paths)
    picker = _DuplicatePicker(paths)
    return picker.run()
