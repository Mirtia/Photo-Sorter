"""Run every pipeline phase in order."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from photo_sorter.backup import backup_input_directory_to_archive
from photo_sorter.constants import DEFAULT_QUALITY_THRESHOLD, DEFAULT_SEARCH_INDEX, PIPELINE_PHASES
from photo_sorter.dedup import Selector, deduplicate_sorted_directory
from photo_sorter.enhance import EnhancementConfig, enhance_directory
from photo_sorter.exceptions import InvalidInputError, InvalidOptionError, MissingOptionError
from photo_sorter.media import as_path
from photo_sorter.metadata import generate_metadata_directory, refresh_face_metadata
from photo_sorter.progress import announce
from photo_sorter.search import index_to_elasticsearch, load_metadata_documents
from photo_sorter.sort import sort_input_directory


@dataclass
class PipelineConfig:
    """Filesystem locations and flags for a full pipeline run."""

    input_directory: Path | str
    archive: Path | str
    output_directory: Path | str
    discard_directory: Path | str
    face_samples_directory: Path | str
    enable_manual_selection: bool = False
    selector: Selector | None = None
    quality_threshold: float = DEFAULT_QUALITY_THRESHOLD
    apply_color_correction: bool = False
    apply_denoising: str | None = None
    apply_sharpening: str | None = None
    apply_artifact_removal: bool = False
    apply_upscaling: str | None = None
    sorted_directory: Path | None = None
    deduped_directory: Path | None = None
    enhanced_directory: Path | None = None
    metadata_directory: Path | None = None
    elasticsearch_url: str | None = None
    elasticsearch_index: str = DEFAULT_SEARCH_INDEX
    from_phase: str = "backup"
    faces_only: bool = False


def parse_from_phase(value: object | None) -> str:
    """Map a phase name or 0-5 index onto a canonical pipeline stage."""
    if value is None or value == "":
        return "backup"
    text = str(value).strip().lower()
    aliases = {name: name for name in PIPELINE_PHASES}
    for index, name in enumerate(PIPELINE_PHASES):
        aliases[str(index)] = name
    if text not in aliases:
        raise InvalidOptionError("from-phase", value)
    return aliases[text]


def required_run_options(from_phase: str) -> tuple[str, ...]:
    """Return argparse destinations needed to resume from ``from_phase``."""
    start = PIPELINE_PHASES.index(from_phase)
    needed: list[str] = ["output"]
    if start <= PIPELINE_PHASES.index("sort"):
        needed.append("input")
    if start == PIPELINE_PHASES.index("backup"):
        needed.append("archive")
    if start <= PIPELINE_PHASES.index("enhance"):
        needed.append("discard")
    if start <= PIPELINE_PHASES.index("metadata"):
        needed.append("face_samples")
    if start == PIPELINE_PHASES.index("index"):
        needed.append("es_url")
    return tuple(needed)


def _require_directory(path: Path) -> Path:
    if not path.is_dir():
        raise InvalidInputError(path)
    return path


def _phase_total(config: PipelineConfig) -> int:
    return 6 if config.elasticsearch_url or config.from_phase == "index" else 5


def _includes(start: int, name: str) -> bool:
    return start <= PIPELINE_PHASES.index(name)


def _workspace(config: PipelineConfig) -> dict[str, Path]:
    output = as_path(config.output_directory)
    return {
        "sorted": config.sorted_directory or output / "sorted",
        "deduped": config.deduped_directory or output / "deduped",
        "enhanced": config.enhanced_directory or output / "enhanced",
        "metadata": config.metadata_directory or output / "metadata",
        "discard": as_path(config.discard_directory),
        "archive": as_path(config.archive),
    }


_RESUME_SOURCE = {
    "dedup": "sorted",
    "enhance": "deduped",
    "metadata": "enhanced",
    "index": "metadata",
}


def _prepare_resume(config: PipelineConfig) -> tuple[int, dict[str, Path]]:
    from_phase = parse_from_phase(config.from_phase)
    if config.faces_only:
        from_phase = "metadata"
    config.from_phase = from_phase
    if from_phase == "index" and not config.elasticsearch_url:
        raise MissingOptionError("es_url")
    dirs = _workspace(config)
    source_key = _RESUME_SOURCE.get(from_phase)
    if source_key is not None:
        _require_directory(dirs[source_key])
    if config.faces_only:
        _require_directory(dirs["metadata"])
    return PIPELINE_PHASES.index(from_phase), dirs


def run_pipeline(config: PipelineConfig) -> dict[str, Path]:
    """Backup, sort, de-duplicate, enhance, and generate metadata.

    ``from_phase`` skips earlier stages so a later step can be re-run against
    existing ``output/sorted``, ``deduped``, ``enhanced``, or ``metadata``.
    ``faces_only`` re-runs face detection/recognition on existing JSON and skips
    object detection and CLIP.
    """
    start, dirs = _prepare_resume(config)
    phases = _phase_total(config)
    archive = dirs["archive"]

    if _includes(start, "backup"):
        announce("backup", 1, phases)
        archive = backup_input_directory_to_archive(config.input_directory, config.archive)
        dirs["archive"] = archive
    if _includes(start, "sort"):
        announce("sort", 2, phases)
        sort_input_directory(config.input_directory, dirs["sorted"])
    if _includes(start, "dedup"):
        announce("dedup", 3, phases)
        deduplicate_sorted_directory(
            dirs["sorted"],
            dirs["deduped"],
            config.discard_directory,
            enable_manual_selection=config.enable_manual_selection,
            selector=config.selector,
        )
    if _includes(start, "enhance"):
        announce("enhance", 4, phases)
        enhance_directory(
            dirs["deduped"],
            dirs["enhanced"],
            EnhancementConfig(
                quality_threshold=config.quality_threshold,
                apply_color_correction=config.apply_color_correction,
                apply_denoising=config.apply_denoising,
                apply_sharpening=config.apply_sharpening,
                apply_artifact_removal=config.apply_artifact_removal,
                apply_upscaling=config.apply_upscaling,
            ),
            discard_directory=config.discard_directory,
        )
    if _includes(start, "metadata"):
        announce("faces" if config.faces_only else "metadata", 5, phases)
        if config.faces_only:
            refresh_face_metadata(dirs["enhanced"], dirs["metadata"], config.face_samples_directory)
        else:
            generate_metadata_directory(dirs["enhanced"], dirs["metadata"], config.face_samples_directory)
    if config.elasticsearch_url and _includes(start, "index"):
        announce("index", 6, phases)
        documents = load_metadata_documents(dirs["metadata"], dirs["enhanced"])
        index_to_elasticsearch(documents, config.elasticsearch_url, config.elasticsearch_index)
    return {key: dirs[key] for key in ("archive", "sorted", "deduped", "enhanced", "metadata", "discard")}
