from voicelm.ingestion.cleaning import clean_text


def test_normalises_windows_and_mac_line_endings() -> None:
    assert clean_text("a\r\nb\rc") == "a\nb\nc"


def test_strips_trailing_whitespace_on_each_line() -> None:
    assert clean_text("alpha   \nbeta\t\n") == "alpha\nbeta"


def test_collapses_runs_of_blank_lines_to_one() -> None:
    assert clean_text("first\n\n\n\n\nsecond") == "first\n\nsecond"


def test_keeps_a_single_paragraph_break() -> None:
    # Chunking splits on paragraph breaks, so cleaning must not destroy them.
    assert clean_text("first\n\nsecond") == "first\n\nsecond"


def test_removes_invisible_characters() -> None:
    assert clean_text("\ufeffhead\u200bway\u00ad") == "headway"


def test_normalises_equivalent_unicode_spellings() -> None:
    composed = "caf\u00e9"  # é as one code point
    decomposed = "cafe\u0301"  # e followed by a combining acute accent

    assert composed != decomposed
    assert clean_text(composed) == clean_text(decomposed)


def test_preserves_indentation_inside_a_line() -> None:
    # Deliberate: indentation is meaningful in Markdown and we quote cleaned text back
    # to the user in citations.
    assert clean_text("text:\n\n    indented code\n") == "text:\n\n    indented code"


def test_strips_leading_and_trailing_whitespace_of_the_document() -> None:
    assert clean_text("\n\n  body  \n\n") == "body"


def test_empty_input_stays_empty() -> None:
    assert clean_text("") == ""
    assert clean_text("   \n\n  ") == ""
