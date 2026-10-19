import json
from pathlib import Path

from photo_sorter.search import (
    SearchQuery,
    index_to_elasticsearch,
    load_metadata_documents,
    search_documents,
    search_elasticsearch,
    search_metadata_directory,
)


class _FakeIndices:
    def __init__(self) -> None:
        self.created: dict[str, object] | None = None
        self.refreshed: str | None = None

    def exists(self, index: str) -> bool:
        return self.created is not None

    def create(self, index: str, mappings: object) -> None:
        self.created = {"index": index, "mappings": mappings}

    def refresh(self, index: str) -> None:
        self.refreshed = index


class _FakeElasticsearch:
    def __init__(self) -> None:
        self.indices = _FakeIndices()
        self.docs: dict[str, dict[str, object]] = {}
        self.last_search: dict[str, object] | None = None

    def ping(self) -> bool:
        return True

    def index(self, index: str, document: dict[str, object], **kwargs: object) -> None:
        self.docs[str(kwargs.get("id"))] = document

    def search(self, index: str, **request: object) -> dict[str, object]:
        self.last_search = {"index": index, **request}
        hits = [{"_score": 1.2, "_source": doc} for doc in self.docs.values()]
        return {"hits": {"hits": hits}}


def _write_metadata(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_and_search_metadata_by_object_and_person(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    _write_metadata(
        meta / "aaa_metadata.json",
        {
            "hash": "aaa",
            "path": str(tmp_path / "enhanced" / "spring_2025" / "pictures" / "aaa.jpg"),
            "date_taken": "2025-04-12T10:00:00",
            "people": ["Dimitris"],
            "objects": [{"name": "boat", "confidence": 0.8}],
            "object_labels": ["boat"],
            "tags": ["boat", "Dimitris"],
            "description": "A photo featuring boat featuring Dimitris",
            "location": {"latitude": 35.3, "longitude": 25.1},
        },
    )
    _write_metadata(
        meta / "bbb_metadata.json",
        {
            "hash": "bbb",
            "people": ["Alex"],
            "objects": [{"name": "dog", "confidence": 0.7}],
            "object_labels": ["dog"],
            "tags": ["dog"],
            "description": "A photo featuring dog",
            "date_taken": "2024-01-01T00:00:00",
        },
    )

    documents = load_metadata_documents(meta)
    assert documents[0]["location"] == {"lat": 35.3, "lon": 25.1}
    assert documents[0]["season"] == "spring_2025"

    hits = search_metadata_directory(
        meta,
        SearchQuery(text="boat", person="Dimitris", object_label="boat"),
    )
    assert [hit.hash for hit in hits] == ["aaa"]
    assert hits[0].object_labels == ["boat"]

    none = search_documents(documents, SearchQuery(text="boat", person="Alex"))
    assert none == []


def test_index_and_search_elasticsearch_with_fake_client(tmp_path: Path) -> None:
    meta = tmp_path / "metadata"
    meta.mkdir()
    _write_metadata(
        meta / "ccc_metadata.json",
        {
            "hash": "ccc",
            "description": "A photo featuring bicycle",
            "object_labels": ["bicycle"],
            "objects": [{"name": "bicycle", "confidence": 0.9}],
            "people": [],
            "embedding": [0.1, 0.2, 0.3],
        },
    )
    documents = load_metadata_documents(meta)
    client = _FakeElasticsearch()

    count = index_to_elasticsearch(documents, "http://localhost:9200", client=client)
    assert count == 1
    assert "ccc" in client.docs
    assert "embedding" in client.docs["ccc"]
    assert client.indices.created is not None

    hits = search_elasticsearch(
        SearchQuery(text="bicycle"),
        "http://localhost:9200",
        client=client,
    )
    assert hits[0].hash == "ccc"
    assert client.last_search is not None
