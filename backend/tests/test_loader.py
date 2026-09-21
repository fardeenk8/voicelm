from pathlib import Path

import pytest

from voicelm.ingestion.loader import UndecodableFile, UnsupportedFileType, load_source


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


def test_identical_content_yields_the_same_id(tmp_path: Path) -> None:
    first = write(tmp_path, "a.txt", "same body")
    second = write(tmp_path, "b.txt", "same body")

    assert load_source(first).id == load_source(second).id


def test_different_content_yields_a_different_id(tmp_path: Path) -> None:
    first = write(tmp_path, "a.txt", "one body")
    second = write(tmp_path, "b.txt", "another body")

    assert load_source(first).id != load_source(second).id


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
