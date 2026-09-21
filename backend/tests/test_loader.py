from pathlib import Path

import pytest
from pdf_fixtures import make_pdf

from voicelm.ingestion.loader import UndecodableFile, UnsupportedFileType, load_source
from voicelm.ingestion.pdf import CorruptPdf, NoTextLayer


def write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


@pytest.mark.parametrize("name", ["notes.txt", "notes.md", "notes.markdown"])
def test_loads_supported_text_formats(tmp_path: Path, name: str) -> None:
    path = write(tmp_path, name, "Hello.")

    assert load_source(path).text == "Hello."


def test_title_comes_from_the_filename(tmp_path: Path) -> None:
    path = write(tmp_path, "meeting-notes.md", "Body.")

    assert load_source(path).title == "meeting-notes"


def test_text_is_cleaned_on_load(tmp_path: Path) -> None:
    path = write(tmp_path, "notes.txt", "line   \r\n\r\n\r\n\r\nnext")

    assert load_source(path).text == "line\n\nnext"


def test_identical_content_yields_the_same_hash(tmp_path: Path) -> None:
    first = write(tmp_path, "a.txt", "same body")
    second = write(tmp_path, "b.txt", "same body")

    assert load_source(first).content_hash == load_source(second).content_hash


def test_identical_content_still_gets_distinct_ids(tmp_path: Path) -> None:
    # Identity is the file path, not the text (ADR-0019). Two copies are two sources.
    first = write(tmp_path, "a.txt", "same body")
    second = write(tmp_path, "b.txt", "same body")

    assert load_source(first).id != load_source(second).id


def test_different_content_yields_a_different_hash(tmp_path: Path) -> None:
    first = write(tmp_path, "a.txt", "one body")
    second = write(tmp_path, "b.txt", "another body")

    assert load_source(first).content_hash != load_source(second).content_hash


def test_id_is_a_uuid(tmp_path: Path) -> None:
    source = load_source(write(tmp_path, "a.txt", "body"))

    assert len(source.id) == 36
    assert source.id.count("-") == 4


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    path = write(tmp_path, "slides.pptx", "content")

    with pytest.raises(UnsupportedFileType):
        load_source(path)


def test_extension_check_is_case_insensitive(tmp_path: Path) -> None:
    path = write(tmp_path, "NOTES.TXT", "content")

    assert load_source(path).text == "content"


def test_non_utf8_file_fails_loudly(tmp_path: Path) -> None:
    path = tmp_path / "legacy.txt"
    path.write_bytes(b"caf\xe9")  # é in cp1252, invalid UTF-8

    with pytest.raises(UndecodableFile):
        load_source(path)


def test_empty_file_loads_as_empty_text(tmp_path: Path) -> None:
    path = write(tmp_path, "empty.txt", "")

    assert load_source(path).text == ""


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_source(tmp_path / "nope.txt")


# --- PDFs ------------------------------------------------------------------------------


def write_pdf(tmp_path: Path, name: str, pages: list[str]) -> Path:
    path = tmp_path / name
    path.write_bytes(make_pdf(pages))
    return path


def test_loads_a_pdf(tmp_path: Path) -> None:
    source = load_source(write_pdf(tmp_path, "bio.pdf", ["Mitochondria make ATP."]))

    assert "Mitochondria make ATP." in source.text
    assert source.title == "bio"


def test_a_pdf_carries_a_page_span_per_page(tmp_path: Path) -> None:
    path = write_pdf(tmp_path, "bio.pdf", ["Page one text.", "Page two text."])

    source = load_source(path)

    assert [span.number for span in source.pages] == [1, 2]
    for span, expected in zip(source.pages, ["Page one text.", "Page two text."], strict=True):
        assert source.text[span.start_char : span.end_char] == expected


def test_text_formats_carry_no_pages(tmp_path: Path) -> None:
    """A Markdown file genuinely has no pages, so it claims none."""
    assert load_source(write(tmp_path, "notes.md", "Body.")).pages == ()


def test_pdf_extension_check_is_case_insensitive(tmp_path: Path) -> None:
    path = write_pdf(tmp_path, "REPORT.PDF", ["Quarterly results."])

    assert "Quarterly results." in load_source(path).text


def test_a_scanned_pdf_is_refused_rather_than_ingested_empty(tmp_path: Path) -> None:
    path = write_pdf(tmp_path, "scan.pdf", ["", ""])

    with pytest.raises(NoTextLayer):
        load_source(path)


def test_a_pdf_that_is_not_a_pdf_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"%PDF-1.4 and then nonsense")

    with pytest.raises(CorruptPdf):
        load_source(path)


def test_pdf_content_hash_follows_the_text(tmp_path: Path) -> None:
    same = write_pdf(tmp_path, "a.pdf", ["Identical body."])
    other = write_pdf(tmp_path, "b.pdf", ["Different body."])

    assert load_source(same).content_hash != load_source(other).content_hash
