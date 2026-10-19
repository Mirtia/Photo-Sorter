"""Command-line interface for the photo-sorter pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from photo_sorter.backup import backup_input_directory_to_archive
from photo_sorter.config import apply_config, load_user_config, require_command_options
from photo_sorter.constants import (
    DEFAULT_QUALITY_THRESHOLD,
    DEFAULT_SEARCH_INDEX,
    DEFAULT_SEARCH_RESULTS,
    ENHANCEMENT_LEVELS,
    UPSCALE_LEVELS,
)
from photo_sorter.dedup import deduplicate_sorted_directory
from photo_sorter.enhance import EnhancementConfig, enhance_directory
from photo_sorter.exceptions import InvalidOptionError, PhotoSorterError
from photo_sorter.metadata import generate_metadata_directory, refresh_face_metadata
from photo_sorter.models import pull_models
from photo_sorter.pipeline import PipelineConfig, run_pipeline
from photo_sorter.progress import announce
from photo_sorter.search import (
    SearchHit,
    SearchQuery,
    index_to_elasticsearch,
    load_metadata_documents,
    search_elasticsearch,
    search_metadata_directory,
)
from photo_sorter.sort import sort_input_directory


def _add_enhancement_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=None,
        help=f"Enhance pictures whose quality score is below this value (default: {DEFAULT_QUALITY_THRESHOLD}). "
        "Pictures with motion or focus blur are discarded instead of enhanced.",
    )
    parser.add_argument(
        "--apply-color-correction",
        action="store_true",
        help="Adjust brightness and contrast on low-quality pictures. Does not rebalance colours.",
    )
    parser.add_argument(
        "--apply-denoising",
        nargs="?",
        const="medium",
        choices=ENHANCEMENT_LEVELS,
        default=None,
        help="Remove noise. Optional level: light, medium (default if flag is set), strong.",
    )
    parser.add_argument(
        "--apply-sharpening",
        nargs="?",
        const="medium",
        choices=ENHANCEMENT_LEVELS,
        default=None,
        help="Sharpen low-quality pictures. Optional level: light, medium, strong.",
    )
    parser.add_argument(
        "--apply-artifact-removal",
        action="store_true",
        help="Reduce compression artifacts on low-quality pictures.",
    )
    parser.add_argument(
        "--apply-upscaling",
        nargs="?",
        const="medium",
        choices=UPSCALE_LEVELS,
        default=None,
        help="Upscale small pictures. Optional level: light, medium, strong, or ml (Real-ESRGAN).",
    )


def build_parser() -> argparse.ArgumentParser:
    """Create the top-level parser and phase subcommands."""
    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument(
        "-c",
        "--config",
        default=argparse.SUPPRESS,
        metavar="FILE",
        help="YAML config file. If omitted, photo-sorter.yaml in the current directory is used when present.",
    )
    shared.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Print debug logging for each phase.",
    )
    parser = argparse.ArgumentParser(
        prog="photo-sorter",
        description="Sort, de-duplicate, enhance, and tag a personal photo collection.",
        parents=[shared],
    )
    subparsers = parser.add_subparsers(dest="command")

    backup = subparsers.add_parser("backup", parents=[shared], help="Phase 0: compress the original directory.")
    backup.add_argument("--input", help="Original photo directory to archive.")
    backup.add_argument("--archive", help="Destination archive (e.g. backup.tar.gz).")

    sort = subparsers.add_parser("sort", parents=[shared], help="Phase 1: convert PNGs, flatten, and bin by season.")
    sort.add_argument("--input", help="Original photo directory.")
    sort.add_argument("--output", help="Sorted output directory (e.g. output/sorted/).")

    dedup = subparsers.add_parser(
        "dedup",
        parents=[shared],
        help="Phase 2: group similar pictures and drop duplicates.",
    )
    dedup.add_argument("--input", help="Sorted directory from phase 1.")
    dedup.add_argument("--output", help="De-duplicated output directory.")
    dedup.add_argument("--discard", help="Directory for discarded duplicates.")
    dedup.add_argument(
        "--enable-manual-selection",
        action="store_true",
        help="Open a window to pick which picture(s) to keep in each similar group.",
    )

    enhance = subparsers.add_parser(
        "enhance",
        parents=[shared],
        help="Phase 3: enhance low-quality pictures and hash names.",
    )
    enhance.add_argument("--input", help="De-duplicated directory from phase 2.")
    enhance.add_argument("--output", help="Enhanced output directory.")
    enhance.add_argument(
        "--discard",
        help="Directory for pictures discarded as motion or focus blur.",
    )
    _add_enhancement_flags(enhance)

    metadata = subparsers.add_parser("metadata", parents=[shared], help="Phase 4: write per-image metadata JSON.")
    metadata.add_argument("--input", help="Enhanced directory from phase 3.")
    metadata.add_argument("--output", help="Metadata output directory.")
    metadata.add_argument(
        "--face-samples",
        help="Directory of known faces (samples/<Name>/*.jpg or samples/Name.jpg).",
    )
    metadata.add_argument(
        "--faces-only",
        action="store_true",
        help="Re-run YuNet/SFace on existing metadata JSON. Leaves objects, CLIP embeddings, and EXIF unchanged.",
    )

    index_cmd = subparsers.add_parser(
        "index",
        parents=[shared],
        help="Phase 5: index metadata into Elasticsearch for semantic search.",
    )
    index_cmd.add_argument("--metadata", help="Metadata directory from phase 4.")
    index_cmd.add_argument(
        "--images",
        default=None,
        help="Enhanced pictures directory (used to resolve file paths).",
    )
    index_cmd.add_argument(
        "--es-url",
        help="Elasticsearch URL (e.g. http://localhost:9200).",
    )
    index_cmd.add_argument(
        "--es-index",
        default=None,
        help=f"Index name (default: {DEFAULT_SEARCH_INDEX}).",
    )

    search_cmd = subparsers.add_parser(
        "search",
        parents=[shared],
        help="Search photos by text, person, or detected object.",
    )
    search_cmd.add_argument("query", nargs="?", default="", help="Free-text query (CLIP + keywords).")
    search_cmd.add_argument(
        "--metadata",
        default=None,
        help="Search JSON metadata on disk (no Elasticsearch required).",
    )
    search_cmd.add_argument("--images", default=None, help="Enhanced pictures directory (local search).")
    search_cmd.add_argument("--es-url", default=None, help="Elasticsearch URL.")
    search_cmd.add_argument(
        "--es-index",
        default=None,
        help=f"Index name (default: {DEFAULT_SEARCH_INDEX}).",
    )
    search_cmd.add_argument("--person", default=None, help="Filter to a known person name.")
    search_cmd.add_argument("--object", dest="object_label", default=None, help="Filter to a detected object label.")
    search_cmd.add_argument("--from", dest="date_from", default=None, help="Earliest date_taken (ISO).")
    search_cmd.add_argument("--to", dest="date_to", default=None, help="Latest date_taken (ISO).")
    search_cmd.add_argument(
        "--limit",
        type=int,
        default=None,
        help=f"Maximum hits to return (default: {DEFAULT_SEARCH_RESULTS}).",
    )

    run = subparsers.add_parser("run", parents=[shared], help="Run phases 0-4 in order (optionally index).")
    run.add_argument("--input", help="Original photo directory.")
    run.add_argument("--archive", help="Backup archive path (e.g. backup.tar.gz).")
    run.add_argument("--output", help="Workspace for sorted/deduped/enhanced/metadata.")
    run.add_argument("--discard", help="Directory for discarded duplicates and unfixable blur.")
    run.add_argument("--face-samples", help="Directory of known face samples.")
    run.add_argument(
        "--from-phase",
        default=None,
        metavar="PHASE",
        help="Resume from this stage: backup, sort, dedup, enhance, metadata, index (or 0-5). "
        "Earlier phases are skipped. Default: backup (full run).",
    )
    run.add_argument(
        "--faces-only",
        action="store_true",
        help="Only re-run face detection/recognition on existing metadata (implies --from-phase metadata).",
    )
    run.add_argument(
        "--enable-manual-selection",
        action="store_true",
        help="Open a window to pick which picture(s) to keep in each similar group.",
    )
    _add_enhancement_flags(run)
    run.add_argument(
        "--es-url",
        default=None,
        help="If set, index metadata into Elasticsearch after phase 4.",
    )
    run.add_argument(
        "--es-index",
        default=None,
        help=f"Index name used with --es-url (default: {DEFAULT_SEARCH_INDEX}).",
    )

    pull = subparsers.add_parser(
        "pull-models",
        parents=[shared],
        help="Download YOLO, CLIP, Real-ESRGAN, and face (YuNet/SFace) weights into the local cache.",
    )
    pull.add_argument(
        "--yolo-model",
        default=None,
        help="YOLO weights to fetch (default: yolov8n.pt). Also set via PHOTO_SORTER_YOLO_MODEL or ml.yolo_model in YAML.",
    )
    return parser


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments, then fill gaps from a YAML config file."""
    argv_list = list(argv) if argv is not None else sys.argv[1:]
    args = build_parser().parse_args(argv_list)
    apply_config(args, load_user_config(getattr(args, "config", None)), argv_list)
    require_command_options(args)
    return args


def _enhancement_from_args(args: argparse.Namespace) -> EnhancementConfig:
    return EnhancementConfig(
        quality_threshold=args.quality_threshold,
        apply_color_correction=args.apply_color_correction,
        apply_denoising=args.apply_denoising,
        apply_sharpening=args.apply_sharpening,
        apply_artifact_removal=args.apply_artifact_removal,
        apply_upscaling=args.apply_upscaling,
    )


def _search_query_from_args(args: argparse.Namespace) -> SearchQuery:
    return SearchQuery(
        text=args.query or "",
        person=args.person,
        object_label=args.object_label,
        date_from=args.date_from,
        date_to=args.date_to,
        limit=args.limit,
    )


def _print_hits(hits: list[SearchHit]) -> None:
    if not hits:
        print("No matches.")
        return
    for index, hit in enumerate(hits, start=1):
        where = hit.path or hit.hash
        extras = []
        if hit.people:
            extras.append("people=" + ", ".join(hit.people))
        if hit.object_labels:
            extras.append("objects=" + ", ".join(hit.object_labels))
        suffix = f" ({'; '.join(extras)})" if extras else ""
        print(f"{index}. {where}  score={hit.score:.3f}{suffix}")
        if hit.description:
            print(f"   {hit.description}")


def _run_backup(args: argparse.Namespace) -> None:
    announce("backup")
    backup_input_directory_to_archive(args.input, args.archive)


def _run_sort(args: argparse.Namespace) -> None:
    announce("sort")
    sort_input_directory(args.input, args.output)


def _run_dedup(args: argparse.Namespace) -> None:
    announce("dedup")
    deduplicate_sorted_directory(
        args.input,
        args.output,
        args.discard,
        enable_manual_selection=args.enable_manual_selection,
    )


def _run_enhance(args: argparse.Namespace) -> None:
    announce("enhance")
    enhance_directory(
        args.input,
        args.output,
        _enhancement_from_args(args),
        discard_directory=args.discard,
    )


def _run_metadata(args: argparse.Namespace) -> None:
    announce("faces" if args.faces_only else "metadata")
    if args.faces_only:
        refresh_face_metadata(args.input, args.output, args.face_samples)
    else:
        generate_metadata_directory(args.input, args.output, args.face_samples)


def _run_index(args: argparse.Namespace) -> None:
    announce("index")
    documents = load_metadata_documents(args.metadata, args.images)
    count = index_to_elasticsearch(documents, args.es_url, args.es_index)
    print(f"Indexed {count} photos into {args.es_index}.")


def _run_search(args: argparse.Namespace) -> None:
    query = _search_query_from_args(args)
    if args.es_url:
        hits = search_elasticsearch(query, args.es_url, args.es_index)
    elif args.metadata:
        hits = search_metadata_directory(args.metadata, query, args.images)
    else:
        raise InvalidOptionError("search", "provide --metadata or --es-url")
    _print_hits(hits)


def _run_pull_models(args: argparse.Namespace) -> None:
    announce("pull-models")
    pulled = pull_models(yolo_name=args.yolo_model)
    print(f"YOLO: {pulled.yolo}")
    print(f"CLIP: {pulled.clip}")
    print(f"upscale: {pulled.upscale}")
    print(f"face detector: {pulled.face_detector}")
    print(f"face recognizer: {pulled.face_recognizer}")


def _run_all(args: argparse.Namespace) -> None:
    output = args.output
    run_pipeline(
        PipelineConfig(
            input_directory=args.input or output,
            archive=args.archive or f"{output}/backup.tar.gz",
            output_directory=output,
            discard_directory=args.discard or f"{output}/discarded",
            face_samples_directory=args.face_samples or f"{output}/samples",
            enable_manual_selection=args.enable_manual_selection,
            quality_threshold=args.quality_threshold,
            apply_color_correction=args.apply_color_correction,
            apply_denoising=args.apply_denoising,
            apply_sharpening=args.apply_sharpening,
            apply_artifact_removal=args.apply_artifact_removal,
            apply_upscaling=args.apply_upscaling,
            elasticsearch_url=args.es_url,
            elasticsearch_index=args.es_index,
            from_phase=args.from_phase,
            faces_only=args.faces_only,
        )
    )


def _dispatch(args: argparse.Namespace) -> None:
    handlers = {
        "backup": _run_backup,
        "sort": _run_sort,
        "dedup": _run_dedup,
        "enhance": _run_enhance,
        "metadata": _run_metadata,
        "index": _run_index,
        "search": _run_search,
        "run": _run_all,
        "pull-models": _run_pull_models,
    }
    handler = handlers.get(args.command or "")
    if handler is None:
        build_parser().print_help()
        raise SystemExit(2)
    handler(args)


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the ``photo-sorter`` console script."""
    args = parse_arguments(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        _dispatch(args)
    except PhotoSorterError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":  # pragma: no cover
    main()
