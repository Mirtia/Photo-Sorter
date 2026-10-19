"""Exceptions raised by photo-sorter."""


class PhotoSorterError(Exception):
    """Base error for photo-sorter failures."""


class InvalidInputError(PhotoSorterError):
    """Raised when a required path or argument is invalid."""

    def __init__(self, path: object) -> None:
        super().__init__(f"Not a directory: {path}")


class MissingDependencyError(PhotoSorterError):
    """Raised when an optional extra (YOLO, CLIP, Real-ESRGAN, Elasticsearch) is missing."""

    def __init__(self, package: str) -> None:
        super().__init__(f"Missing optional dependency: {package}")


class ModelDownloadError(PhotoSorterError):
    """Raised when YOLO, CLIP, or Real-ESRGAN weights cannot be downloaded or loaded."""

    def __init__(self, name: object) -> None:
        super().__init__(f"Could not download model: {name}")


class UnreachableSearchError(PhotoSorterError):
    """Raised when Elasticsearch cannot be reached."""

    def __init__(self, url: object) -> None:
        super().__init__(f"Elasticsearch is not reachable: {url}")


class MissingOptionError(PhotoSorterError):
    """Raised when a required CLI/config option was not provided."""

    def __init__(self, name: str) -> None:
        super().__init__(f"Missing required option: {name}")


class ConfigError(PhotoSorterError):
    """Raised when a YAML config file cannot be read or parsed."""

    def __init__(self, reason: object) -> None:
        super().__init__(f"Config error: {reason}")


class InvalidOptionError(PhotoSorterError):
    """Raised when a CLI/config option is not supported."""

    def __init__(self, name: str, value: object) -> None:
        super().__init__(f"Invalid {name}: {value}")
