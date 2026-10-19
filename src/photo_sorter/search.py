"""Index photo metadata and run keyword / semantic search."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from photo_sorter.constants import DEFAULT_SEARCH_INDEX, DEFAULT_SEARCH_RESULTS
from photo_sorter.detect import season_from_path
from photo_sorter.embed import cosine_similarity, embed_text
from photo_sorter.exceptions import InvalidInputError, MissingDependencyError, UnreachableSearchError
from photo_sorter.media import as_path
from photo_sorter.progress import track

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SearchHit:
    """One ranked search result."""

    hash: str
    score: float
    path: str | None
    description: str
    people: list[str]
    object_labels: list[str]
    date_taken: str | None


@dataclass(frozen=True)
class SearchQuery:
    """Filters and free-text applied to the photo index."""

    text: str = ""
    person: str | None = None
    object_label: str | None = None
    date_from: str | None = None
    date_to: str | None = None
    limit: int = DEFAULT_SEARCH_RESULTS


def _object_labels(document: dict[str, Any]) -> list[str]:
    labels = document.get("object_labels")
    if isinstance(labels, list) and labels:
        return [str(label) for label in labels]
    objects = document.get("objects") or document.get("topics") or []
    names: list[str] = []
    if isinstance(objects, list):
        for item in objects:
            if isinstance(item, dict) and item.get("name"):
                names.append(str(item["name"]))
    return names


def _geo_point(location: object) -> dict[str, float] | None:
    if not isinstance(location, dict):
        return None
    lat = location.get("latitude")
    lon = location.get("longitude")
    if lat is None or lon is None:
        return None
    return {"lat": float(lat), "lon": float(lon)}


def load_metadata_documents(
    metadata_directory: Path | str,
    images_directory: Path | str | None = None,
) -> list[dict[str, Any]]:
    """Load ``*_metadata.json`` files into Elasticsearch-shaped documents."""
    source = as_path(metadata_directory)
    if not source.is_dir():
        raise InvalidInputError(source)
    images_root = as_path(images_directory) if images_directory is not None else None
    documents: list[dict[str, Any]] = []
    for path in sorted(source.glob("*_metadata.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Skipping invalid metadata JSON: %s", path)
            continue
        if not isinstance(raw, dict):
            continue
        photo_hash = str(raw.get("hash") or path.name.replace("_metadata.json", ""))
        image_path = raw.get("path")
        if images_root is not None and not image_path:
            matches = list(images_root.rglob(f"{photo_hash}.jpg"))
            image_path = str(matches[0]) if matches else None
        season = raw.get("season")
        if not season and image_path:
            season = season_from_path(Path(str(image_path)).parts)
        labels = _object_labels(raw)
        embedding = raw.get("embedding")
        if not isinstance(embedding, list):
            embedding = None
        documents.append({
            "hash": photo_hash,
            "path": image_path,
            "filename": raw.get("filename"),
            "date_taken": raw.get("date_taken"),
            "people": list(raw.get("people") or []),
            "objects": raw.get("objects") or raw.get("topics") or [],
            "object_labels": labels,
            "tags": list(raw.get("tags") or []),
            "description": raw.get("description") or "",
            "caption": raw.get("caption"),
            "scene": raw.get("scene") or [],
            "face_count": raw.get("face_count"),
            "orientation": raw.get("orientation"),
            "lighting": raw.get("lighting"),
            "location": _geo_point(raw.get("location")),
            "season": season,
            "resolution": raw.get("resolution"),
            "embedding": embedding,
        })
    return documents


def _tokens(text: str) -> set[str]:
    return {token for token in text.lower().replace(",", " ").split() if token}


def _passes_filters(document: dict[str, Any], query: SearchQuery) -> bool:
    if query.person:
        people = {str(name).lower() for name in document.get("people") or []}
        if query.person.lower() not in people:
            return False
    if query.object_label:
        labels = {str(name).lower() for name in document.get("object_labels") or []}
        if query.object_label.lower() not in labels:
            return False
    taken = document.get("date_taken")
    if isinstance(taken, str):
        if query.date_from and taken < query.date_from:
            return False
        if query.date_to and taken > query.date_to:
            return False
    return True


def _text_score(document: dict[str, Any], query_text: str) -> float:
    tokens = _tokens(query_text)
    if not tokens:
        return 0.0
    haystack = " ".join([
        str(document.get("description") or ""),
        str(document.get("caption") or ""),
        " ".join(str(tag) for tag in document.get("tags") or []),
        " ".join(str(label) for label in document.get("object_labels") or []),
        " ".join(str(name) for name in document.get("people") or []),
        " ".join(str(item.get("name", "")) for item in document.get("scene") or [] if isinstance(item, dict)),
    ]).lower()
    return sum(1 for token in tokens if token in haystack) / len(tokens)


def _hit_from_document(document: dict[str, Any], score: float) -> SearchHit:
    return SearchHit(
        hash=str(document.get("hash") or ""),
        score=round(float(score), 4),
        path=str(document["path"]) if document.get("path") else None,
        description=str(document.get("description") or ""),
        people=[str(name) for name in document.get("people") or []],
        object_labels=[str(name) for name in document.get("object_labels") or []],
        date_taken=str(document["date_taken"]) if document.get("date_taken") else None,
    )


def search_documents(documents: list[dict[str, Any]], query: SearchQuery) -> list[SearchHit]:
    """Rank in-memory documents with keyword overlap and optional CLIP cosine."""
    query_vector = embed_text(query.text) if query.text.strip() else None
    scored: list[SearchHit] = []
    for document in documents:
        if not _passes_filters(document, query):
            continue
        text_score = _text_score(document, query.text)
        vector = document.get("embedding")
        vector_score = 0.0
        if query_vector and isinstance(vector, list) and vector:
            vector_score = cosine_similarity(query_vector, [float(value) for value in vector])
        score = 0.35 * text_score + 0.65 * max(vector_score, 0.0) if query_vector and vector_score else text_score
        if not query.text.strip():
            score = 1.0
        if score <= 0 and query.text.strip():
            continue
        scored.append(_hit_from_document(document, score))
    scored.sort(key=lambda hit: hit.score, reverse=True)
    return scored[: query.limit]


def search_metadata_directory(
    metadata_directory: Path | str,
    query: SearchQuery,
    images_directory: Path | str | None = None,
) -> list[SearchHit]:
    """Search JSON metadata on disk (no Elasticsearch required)."""
    documents = load_metadata_documents(metadata_directory, images_directory)
    return search_documents(documents, query)


def _index_mappings(dims: int | None) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "hash": {"type": "keyword"},
        "path": {"type": "keyword"},
        "filename": {"type": "keyword"},
        "date_taken": {"type": "date", "format": "strict_date_optional_time||epoch_millis"},
        "people": {"type": "keyword"},
        "object_labels": {"type": "keyword"},
        "objects": {
            "type": "nested",
            "properties": {
                "name": {"type": "keyword"},
                "confidence": {"type": "float"},
                "bbox": {"type": "float"},
            },
        },
        "tags": {"type": "keyword"},
        "description": {"type": "text"},
        "caption": {"type": "text"},
        "scene": {
            "type": "nested",
            "properties": {
                "name": {"type": "keyword"},
                "confidence": {"type": "float"},
            },
        },
        "face_count": {"type": "integer"},
        "orientation": {"type": "keyword"},
        "lighting": {"type": "keyword"},
        "location": {"type": "geo_point"},
        "season": {"type": "keyword"},
        "resolution": {
            "properties": {
                "width": {"type": "integer"},
                "height": {"type": "integer"},
            }
        },
    }
    if dims:
        properties["embedding"] = {
            "type": "dense_vector",
            "dims": dims,
            "index": True,
            "similarity": "cosine",
        }
    return {"properties": properties}


def _es_client(url: str) -> Any:
    try:
        from elasticsearch import Elasticsearch
    except ImportError as exc:
        raise MissingDependencyError("elasticsearch") from exc
    client = Elasticsearch(url)
    if not client.ping():
        raise UnreachableSearchError(url)
    return client


def _ensure_index(client: Any, index_name: str, dims: int | None) -> None:
    if client.indices.exists(index=index_name):
        return
    client.indices.create(index=index_name, mappings=_index_mappings(dims))


def _source_for_elasticsearch(document: dict[str, Any]) -> dict[str, Any]:
    payload = dict(document)
    if payload.get("embedding") is None:
        payload.pop("embedding", None)
    return payload


def index_to_elasticsearch(
    documents: list[dict[str, Any]],
    elasticsearch_url: str,
    index_name: str = DEFAULT_SEARCH_INDEX,
    *,
    client: Any | None = None,
) -> int:
    """Create/update the photos index and upsert one document per hash."""
    es_client = client or _es_client(elasticsearch_url)
    dims = next(
        (len(doc["embedding"]) for doc in documents if isinstance(doc.get("embedding"), list) and doc["embedding"]),
        None,
    )
    _ensure_index(es_client, index_name, dims)
    count = 0
    for document in track(documents, desc="index", unit="doc"):
        es_client.index(
            index=index_name,
            id=str(document["hash"]),
            document=_source_for_elasticsearch(document),
        )
        count += 1
    if hasattr(es_client, "indices") and hasattr(es_client.indices, "refresh"):
        es_client.indices.refresh(index=index_name)
    return count


def _filter_clauses(query: SearchQuery) -> list[dict[str, Any]]:
    filters: list[dict[str, Any]] = []
    if query.person:
        filters.append({"term": {"people": query.person}})
    if query.object_label:
        filters.append({"term": {"object_labels": query.object_label}})
    if query.date_from or query.date_to:
        range_body: dict[str, str] = {}
        if query.date_from:
            range_body["gte"] = query.date_from
        if query.date_to:
            range_body["lte"] = query.date_to
        filters.append({"range": {"date_taken": range_body}})
    return filters


def search_elasticsearch(
    query: SearchQuery,
    elasticsearch_url: str,
    index_name: str = DEFAULT_SEARCH_INDEX,
    *,
    client: Any | None = None,
) -> list[SearchHit]:
    """Run a keyword or kNN query against Elasticsearch."""
    es_client = client or _es_client(elasticsearch_url)
    filters = _filter_clauses(query)
    query_vector = embed_text(query.text) if query.text.strip() else None
    request: dict[str, Any]
    if query_vector:
        knn: dict[str, Any] = {
            "field": "embedding",
            "query_vector": query_vector,
            "k": query.limit,
            "num_candidates": max(query.limit * 5, 50),
        }
        if filters:
            knn["filter"] = {"bool": {"filter": filters}}
        request = {"knn": knn, "size": query.limit}
    else:
        must: list[dict[str, Any]] = []
        if query.text.strip():
            must.append({
                "multi_match": {
                    "query": query.text,
                    "fields": ["description", "caption", "tags", "object_labels", "people"],
                }
            })
        bool_query: dict[str, Any] = {}
        if must:
            bool_query["must"] = must
        if filters:
            bool_query["filter"] = filters
        request = {"query": {"bool": bool_query} if bool_query else {"match_all": {}}, "size": query.limit}
    response = es_client.search(index=index_name, **request)
    hits = []
    for item in response.get("hits", {}).get("hits", []):
        source = item.get("_source") or {}
        hits.append(_hit_from_document(source, float(item.get("_score") or 0.0)))
    return hits
