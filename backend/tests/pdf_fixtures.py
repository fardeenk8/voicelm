"""Builds small but genuinely valid PDF files for tests.

Writing these by hand rather than adding a PDF-authoring dependency keeps the test suite
light, and the structure below is a fair look at why extracting text from a PDF is
reconstruction rather than reading: a page's content stream positions glyphs, and the
only thing marking a line break is a `T*` operator moving the cursor down.

A PDF is a set of numbered objects, then a cross-reference table giving the *byte offset*
of each object, then a trailer pointing at the catalog. The byte offsets are why this
helper assembles the file in one pass and records positions as it goes.
"""

from collections.abc import Sequence

FONT_SIZE = 12
LINE_HEIGHT = 14
TEXT_ORIGIN = (72, 720)  # one inch in from the left, one inch down from the top


def _escape(line: str) -> str:
    """Escape the three characters that terminate or continue a PDF literal string."""
    return line.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(text: str) -> bytes:
    """Lay out `text` one source line per PDF line, top down.

    `BT`/`ET` bracket a text object, `Tf` selects the font, `TL` sets the leading (line
    height) that `T*` then uses to step down a line, and `Tj` draws a string.
    """
    x, y = TEXT_ORIGIN
    lines = ["BT", f"/F1 {FONT_SIZE} Tf", f"{LINE_HEIGHT} TL", f"{x} {y} Td"]
    for index, line in enumerate(text.split("\n")):
        if index:
            lines.append("T*")
        lines.append(f"({_escape(line)}) Tj")
    lines.append("ET")
    return "\n".join(lines).encode("latin-1")


def make_pdf(pages: Sequence[str]) -> bytes:
    """Return the bytes of a PDF whose pages contain `pages`, one string per page.

    Pass an empty string for a page with no text at all, which is what a scanned page
    looks like to an extractor.
    """
    if not pages:
        raise ValueError("a PDF needs at least one page")

    # Object numbers are assigned up front because objects reference each other and a
    # reference has to name a number that does not exist yet.
    catalog, page_tree, font = 1, 2, 3
    first = 4
    content_numbers = [first + 2 * index for index in range(len(pages))]
    page_numbers = [number + 1 for number in content_numbers]

    bodies: list[bytes] = [
        f"<< /Type /Catalog /Pages {page_tree} 0 R >>".encode(),
        (
            f"<< /Type /Pages /Count {len(pages)} /Kids "
            f"[{' '.join(f'{number} 0 R' for number in page_numbers)}] >>"
        ).encode(),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]

    # Each page contributes two objects, its content stream then the page itself, which is
    # why the numbers above step in twos.
    for text, content_number in zip(pages, content_numbers, strict=True):
        stream = _content_stream(text) if text else b""
        bodies.append(
            b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream)  # noqa: UP031
        )
        bodies.append(
            (
                f"<< /Type /Page /Parent {page_tree} 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_number} 0 R "
                f"/Resources << /Font << /F1 {font} 0 R >> >> >>"
            ).encode()
        )

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for number, body in enumerate(bodies, start=catalog):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"

    # The cross-reference table. Entry 0 is always the mandatory free-object head, and
    # every entry is exactly 20 bytes wide — the format is positional, not whitespace
    # delimited, which is why the padding below matters.
    xref_offset = len(out)
    out += b"xref\n0 %d\n" % (len(bodies) + 1)
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset

    out += b"trailer\n<< /Size %d /Root %d 0 R >>\n" % (len(bodies) + 1, catalog)
    out += b"startxref\n%d\n%%%%EOF\n" % xref_offset
    return bytes(out)
