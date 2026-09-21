"""Normalising raw document text before chunking.

Cleaning is deliberately conservative: it removes noise that would waste chunk budget or
split identical text into different embeddings, but it does not restructure the document.

In particular we do *not* collapse runs of spaces inside a line. Indentation carries
meaning in Markdown (code blocks, nested lists), and because we quote cleaned text back
to the user in citations, mangling it would show up in the product.
"""

import re
import unicodedata

# Characters that render as nothing but change the bytes we embed: zero-width space,
# zero-width non-joiner/joiner, byte-order mark, and soft hyphen.
_INVISIBLE_CHARS = dict.fromkeys(map(ord, "\u200b\u200c\u200d\ufeff\u00ad"), None)

_TRAILING_WHITESPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_BLANK_LINE_RUN = re.compile(r"\n{3,}")


def clean_text(raw: str) -> str:
    """Return a normalised version of `raw`.

    Order matters here, so each step can assume the previous one has run.
    """
    # Unicode normalisation first: "é" can be one code point or "e" plus a combining
    # accent. They look identical but are different bytes, and would otherwise produce
    # two different embeddings for the same word.
    text = unicodedata.normalize("NFC", raw)

    # Windows and classic-Mac line endings become "\n" so later rules only see one form.
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    text = text.translate(_INVISIBLE_CHARS)
    text = _TRAILING_WHITESPACE.sub("", text)

    # Collapse any run of blank lines to exactly one. This preserves paragraph breaks,
    # which chunking relies on as split points, while dropping decorative gaps.
    text = _BLANK_LINE_RUN.sub("\n\n", text)

    return text.strip()
