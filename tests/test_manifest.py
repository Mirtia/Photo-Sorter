from pathlib import Path

from photo_sorter import manifest


def test_write_and_read_roundtrip(tmp_path: Path) -> None:
    records = [{"phase": "dedup", "group_id": "spring_2025:0"}, {"phase": "enhance", "enhanced_hash": "abc"}]
    path = manifest.write_records(tmp_path, records)
    assert path == tmp_path / manifest.MANIFEST_NAME
    assert manifest.read_records(tmp_path) == records
    assert manifest.read_records(path) == records


def test_read_missing_manifest_returns_empty(tmp_path: Path) -> None:
    assert manifest.read_records(tmp_path) == []
    assert manifest.read_records(tmp_path / "manifest.jsonl") == []


def test_read_skips_malformed_lines(tmp_path: Path) -> None:
    path = tmp_path / manifest.MANIFEST_NAME
    path.write_text('{"phase": "dedup"}\nnot-json\n\n{"phase": "enhance"}\n', encoding="utf-8")
    records = manifest.read_records(tmp_path)
    assert [record["phase"] for record in records] == ["dedup", "enhance"]


def test_write_records_replaces_previous_manifest(tmp_path: Path) -> None:
    manifest.write_records(tmp_path, [{"phase": "dedup", "n": 1}])
    manifest.write_records(tmp_path, [{"phase": "dedup", "n": 2}])
    records = manifest.read_records(tmp_path)
    assert len(records) == 1
    assert records[0]["n"] == 2


def test_load_dedup_lineage_keeps_only_kept_rows(tmp_path: Path) -> None:
    kept = tmp_path / "deduped" / "spring_2025" / "pictures" / "IMG_1.jpg"
    kept.parent.mkdir(parents=True)
    kept.write_bytes(b"x")
    dropped = tmp_path / "discarded" / "spring_2025" / "IMG_2.jpg"
    manifest.write_records(
        tmp_path / "deduped",
        [
            manifest.dedup_record(
                season="spring_2025",
                group_id="spring_2025:0",
                group_size=2,
                original_path=tmp_path / "sorted" / "spring_2025" / "pictures" / "IMG_1.jpg",
                phash="00ff",
                size_bytes=10,
                kept=True,
                output_path=kept,
            ),
            manifest.dedup_record(
                season="spring_2025",
                group_id="spring_2025:0",
                group_size=2,
                original_path=tmp_path / "sorted" / "spring_2025" / "pictures" / "IMG_2.jpg",
                phash="00fe",
                size_bytes=8,
                kept=False,
                output_path=dropped,
            ),
        ],
    )
    lineage = manifest.load_dedup_lineage(tmp_path / "deduped")
    assert list(lineage) == [str(kept.resolve())]
    assert lineage[str(kept.resolve())]["group_id"] == "spring_2025:0"
    assert lineage[str(kept.resolve())]["original_name"] == "IMG_1.jpg"


def test_enhance_record_falls_back_to_source_when_no_lineage(tmp_path: Path) -> None:
    source = tmp_path / "deduped" / "spring_2025" / "pictures" / "IMG_9.jpg"
    record = manifest.enhance_record(
        season="spring_2025",
        source_path=source,
        lineage=None,
        enhanced_hash="deadbeef",
        enhanced_path=tmp_path / "enhanced" / "spring_2025" / "pictures" / "deadbeef.jpg",
        enhanced=True,
    )
    assert record["original_path"] == str(source)
    assert record["original_name"] == "IMG_9.jpg"
    assert record["group_id"] is None
    assert record["enhanced_hash"] == "deadbeef"
    assert record["discarded_blur"] is False
