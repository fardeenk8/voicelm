from pathlib import Path

import pytest

from voicelm.domain.models import Chunk, PageSpan, Source
from voicelm.ingestion.chunking import chunk_document
from voicelm.ingestion.loader import hash_content
from voicelm.storage.sqlite import (
    DuplicatePathError,
    IngestMeta,
    SqliteMetadataStore,
)

META = IngestMeta(embedding_model="nomic-embed-text", chunk_max_chars=1000, chunk_overlap=150)


def make_source(
    path: Path,
    text: str,
    source_id: str = "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
    pages: tuple[PageSpan, ...] = (),
) -> Source:
    return Source(
        id=source_id,
        path=path,
        title=path.stem,
        text=text,
        content_hash=hash_content(text),
        pages=pages,
    )


def make_chunks(source: Source) -> list[Chunk]:
    return chunk_document(source)


def open_store(tmp_path: Path) -> SqliteMetadataStore:
    return SqliteMetadataStore(tmp_path / "voicelm.db")


def test_round_trip_preserves_source_and_chunks(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    source = make_source(tmp_path / "notes.md", "Alpha paragraph.\n\nBeta paragraph.")
    chunks = make_chunks(source)

    store.save(source, chunks, META)
    record = store.get(source.id)

    assert record is not None
    assert record.source.id == source.id
    assert record.source.text == source.text
    assert record.source.content_hash == source.content_hash
    assert record.source.path == (tmp_path / "notes.md").resolve()
    assert record.embedding_model == META.embedding_model
    assert store.get_chunks(source.id) == chunks


def test_lookup_by_path_resolves_the_file(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    path = tmp_path / "notes.md"
    source = make_source(path, "body")
    store.save(source, make_chunks(source), META)

    found = store.get_by_path(path)

    assert found is not None
    assert found.source.id == source.id
    assert found.source.path == path.resolve()


def test_unknown_id_returns_none(tmp_path: Path) -> None:
    store = open_store(tmp_path)

    assert store.get("missing") is None
    assert store.get_by_path(tmp_path / "nope.md") is None


def test_saving_again_replaces_chunks(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    path = tmp_path / "notes.md"
    original = make_source(path, "short.")
    store.save(original, make_chunks(original), META)

    updated = make_source(path, "A much longer document.\n\nWith two paragraphs.", original.id)
    store.save(updated, make_chunks(updated), META)

    record = store.get(original.id)
    chunks = store.get_chunks(original.id)

    assert record is not None
    assert record.source.text == updated.text
    assert chunks == make_chunks(updated)
    assert all(chunk.source_id == original.id for chunk in chunks)


def test_same_path_different_id_is_rejected(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    path = tmp_path / "notes.md"
    first = make_source(path, "body", "11111111-1111-1111-1111-111111111111")
    second = make_source(path, "other", "22222222-2222-2222-2222-222222222222")
    store.save(first, make_chunks(first), META)

    with pytest.raises(DuplicatePathError):
        store.save(second, make_chunks(second), META)


def test_two_paths_with_identical_text_are_two_sources(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    first = make_source(tmp_path / "a.md", "same body", "11111111-1111-1111-1111-111111111111")
    second = make_source(tmp_path / "b.md", "same body", "22222222-2222-2222-2222-222222222222")
    store.save(first, make_chunks(first), META)
    store.save(second, make_chunks(second), META)

    assert first.content_hash == second.content_hash
    assert len(store.list_sources()) == 2


def test_delete_removes_chunks_via_cascade(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    source = make_source(tmp_path / "notes.md", "Alpha.\n\nBeta.")
    store.save(source, make_chunks(source), META)

    assert store.delete(source.id) is True
    assert store.get(source.id) is None
    assert store.get_chunks(source.id) == []


def test_page_spans_round_trip(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    text = "Page one.\n\nPage two."
    pages = (
        PageSpan(number=1, start_char=0, end_char=9),
        PageSpan(number=2, start_char=11, end_char=20),
    )
    source = make_source(tmp_path / "paper.pdf", text, pages=pages)

    store.save(source, make_chunks(source), META)
    record = store.get(source.id)

    assert record is not None
    assert record.source.pages == pages
    for span, expected in zip(record.source.pages, ["Page one.", "Page two."], strict=True):
        assert record.source.text[span.start_char : span.end_char] == expected


def test_text_files_persist_with_no_pages(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    source = make_source(tmp_path / "notes.md", "just text")
    store.save(source, make_chunks(source), META)

    record = store.get(source.id)

    assert record is not None
    assert record.source.pages == ()


def test_delete_removes_pages_via_cascade(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    pages = (PageSpan(number=1, start_char=0, end_char=4),)
    source = make_source(tmp_path / "paper.pdf", "body", pages=pages)
    store.save(source, make_chunks(source), META)

    store.delete(source.id)

    leftover = store._conn.execute("SELECT count(*) FROM pages").fetchone()[0]
    assert leftover == 0


def test_delete_unknown_id_returns_false(tmp_path: Path) -> None:
    store = open_store(tmp_path)

    assert store.delete("missing") is False


def test_chunk_from_another_source_is_rejected(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    source = make_source(tmp_path / "notes.md", "body")
    foreign = Chunk(
        id="other:0",
        source_id="other",
        text="nope",
        start_char=0,
        end_char=4,
        ordinal=0,
    )

    with pytest.raises(ValueError, match="belongs to"):
        store.save(source, [foreign], META)


def test_get_chunks_by_ids_preserves_caller_order(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    source = make_source(
        tmp_path / "notes.md", "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    )
    chunks = make_chunks(source)
    store.save(source, chunks, META)
    requested = [chunks[-1].id, chunks[0].id]

    loaded = store.get_chunks_by_ids(requested)

    assert [chunk.id for chunk in loaded] == requested


def test_get_chunks_by_ids_raises_on_a_missing_id(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    source = make_source(tmp_path / "notes.md", "body")
    chunks = make_chunks(source)
    store.save(source, chunks, META)

    with pytest.raises(KeyError, match="not in catalog"):
        store.get_chunks_by_ids([chunks[0].id, "ghost"])


def test_offsets_still_slice_after_a_round_trip(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    source = make_source(tmp_path / "notes.md", "Alpha paragraph.\n\nBeta paragraph.")
    store.save(source, make_chunks(source), META)

    loaded = store.get(source.id)
    assert loaded is not None
    for chunk in store.get_chunks(source.id):
        assert loaded.source.text[chunk.start_char : chunk.end_char] == chunk.text


def test_list_sources_newest_first(tmp_path: Path) -> None:
    store = open_store(tmp_path)
    older = make_source(tmp_path / "old.md", "old", "11111111-1111-1111-1111-111111111111")
    newer = make_source(tmp_path / "new.md", "new", "22222222-2222-2222-2222-222222222222")
    store.save(
        older,
        make_chunks(older),
        IngestMeta("nomic-embed-text", 1000, 150, "2026-01-01T00:00:00+00:00"),
    )
    store.save(
        newer,
        make_chunks(newer),
        IngestMeta("nomic-embed-text", 1000, 150, "2026-06-01T00:00:00+00:00"),
    )

    listed = store.list_sources()

    assert [record.source.title for record in listed] == ["new", "old"]


def test_survives_reopening_the_file(tmp_path: Path) -> None:
    db = tmp_path / "voicelm.db"
    source = make_source(tmp_path / "notes.md", "persisted body")
    first = SqliteMetadataStore(db)
    first.save(source, make_chunks(source), META)
    first.close()

    second = SqliteMetadataStore(db)
    record = second.get(source.id)

    assert record is not None
    assert record.source.text == "persisted body"
    second.close()
