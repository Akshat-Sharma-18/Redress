"""Turn an uploaded file into one canonical text.

The word *canonical* is doing real work here. Three separate mechanisms index
into a document by character offset: chunk boundaries (`Chunk.char_start`),
citation spans resolved in `app.api.schemas`, and the denial-letter anchor
that gives the citation beam its origin. If extraction and display disagreed
about a single character, every one of those offsets would point somewhere
slightly wrong, and the failure would look like a UI bug rather than an
extraction bug.

So normalisation happens exactly once, here, and what this module returns is
the *only* version of the document the rest of the system ever sees. The
uploaded bytes are not kept as a second source of truth.

PDF text extraction is lossy in ways that matter for verbatim citation
matching. Two are corrected because they are unambiguous mechanical damage
(CRLF line endings, non-breaking spaces standing in for ordinary ones); a
third, line-break hyphenation, is corrected only under a conservative rule
described at `_DEHYPHENATE`. Nothing else is touched, because the strictness
of verbatim matching is a safety property and cleaning text more aggressively
would quietly erode it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

#: Extensions accepted at upload. Deliberately short: every additional format
#: is another extractor whose failure modes have to be understood before a
#: verdict can rest on its output.
SUPPORTED_SUFFIXES = (".pdf", ".txt", ".md", ".text")


class ExtractionError(ValueError):
    """Raised when a file cannot be turned into text worth auditing."""


@dataclass
class ExtractedDocument:
    """Canonical text plus the provenance needed to cite locations in it."""

    text: str
    filename: str
    #: Character offset in `text` where each page begins. Empty for formats
    #: with no pagination. Used to synthesise "p. 4" locators for prose that
    #: carries no section numbering of its own.
    page_starts: list[int] = field(default_factory=list)

    def page_of(self, offset: int) -> int | None:
        """1-based page containing `offset`, or None if unpaginated."""
        if not self.page_starts:
            return None
        page = 0
        for index, start in enumerate(self.page_starts):
            if start <= offset:
                page = index
            else:
                break
        return page + 1


#: Joins a word broken across a line by a hyphen: "emer-\ngency" -> "emergency".
#:
#: Restricted to lowercase-to-lowercase because that is the case where the
#: hyphen is almost certainly typographic rather than lexical. A capital on
#: either side suggests a real compound ("Blue-Cross", "co-Insurance" at a
#: sentence start) and is left alone. Getting this wrong in the permissive
#: direction would corrupt the text a citation must match verbatim.
_DEHYPHENATE = re.compile(r"([a-z])-\n([a-z])")

#: Three or more blank lines collapse to one blank line. PDF extractors emit
#: runs of these around page furniture; they carry no meaning and they inflate
#: every offset downstream of them.
_BLANK_RUN = re.compile(r"\n{3,}")

#: Trailing spaces before a newline. Invisible, and they make an otherwise
#: exact quote match fail.
_TRAILING_WS = re.compile(r"[ \t]+\n")


def normalise(raw: str) -> str:
    """Canonicalise extracted text. Idempotent."""
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    # NBSP, narrow NBSP, and the zero-width space PDF extractors like to
    # sprinkle into justified text.
    text = text.replace(" ", " ").replace(" ", " ").replace("​", "")
    text = _DEHYPHENATE.sub(r"\1\2", text)
    text = _TRAILING_WS.sub("\n", text)
    text = _BLANK_RUN.sub("\n\n", text)
    return text.strip()


def extract(data: bytes, filename: str) -> ExtractedDocument:
    """Extract canonical text from uploaded bytes.

    Dispatches on extension rather than sniffing content: the upload endpoint
    already restricts what it accepts, and a file whose extension lies about
    its contents should fail loudly at the parser instead of being quietly
    reinterpreted.
    """
    if len(data) > MAX_UPLOAD_BYTES:
        raise ExtractionError(
            f"{filename} is {len(data) // 1024 // 1024} MB; "
            f"the limit is {MAX_UPLOAD_BYTES // 1024 // 1024} MB"
        )
    if not data:
        raise ExtractionError(f"{filename} is empty")

    lowered = filename.lower()
    if lowered.endswith(".pdf"):
        document = _extract_pdf(data, filename)
    elif lowered.endswith((".txt", ".md", ".text")):
        document = _extract_text(data, filename)
    else:
        raise ExtractionError(
            f"cannot read {filename}: supported formats are "
            f"{', '.join(SUPPORTED_SUFFIXES)}"
        )

    if not document.text.strip():
        raise ExtractionError(
            f"no text could be read from {filename}. If it is a scanned "
            f"document it needs OCR first — this system does not guess at "
            f"the contents of an image."
        )
    return document


def _extract_text(data: bytes, filename: str) -> ExtractedDocument:
    try:
        raw = data.decode("utf-8")
    except UnicodeDecodeError:
        # Latin-1 decodes any byte sequence, so this cannot fail — it is the
        # honest fallback for a file that is text but not UTF-8.
        raw = data.decode("latin-1")
    return ExtractedDocument(text=normalise(raw), filename=filename)


def _extract_pdf(data: bytes, filename: str) -> ExtractedDocument:
    import io

    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise ExtractionError(f"cannot read {filename} as a PDF: {exc}") from exc

    if reader.is_encrypted:
        # An empty user password is common on "protected" policy documents and
        # decrypts silently; a real password is the user's to supply, and this
        # system will not attempt to work around one.
        try:
            reader.decrypt("")
        except Exception as exc:  # noqa: BLE001 - pypdf raises several types
            raise ExtractionError(
                f"{filename} is password-protected; remove the password and "
                f"upload it again"
            ) from exc

    # Pages are normalised individually and then joined, so that a page break
    # is always a paragraph break in the canonical text. Normalising the join
    # afterwards would let the last line of one page run into the first line
    # of the next and produce a sentence that exists in no document.
    # The cursor is advanced by exactly what is appended, rather than joining
    # afterwards and inferring offsets: an empty page (scanned insert, blank
    # verso) would otherwise shift every subsequent page's recorded start.
    parts: list[str] = []
    page_starts: list[int] = []
    cursor = 0
    for index, page in enumerate(reader.pages):
        try:
            page_text = normalise(page.extract_text() or "")
        except Exception:  # noqa: BLE001 - a damaged page should not lose the rest
            page_text = ""
        if index:
            parts.append("\n\n")
            cursor += 2
        page_starts.append(cursor)
        parts.append(page_text)
        cursor += len(page_text)

    # rstrip only. Stripping the front would invalidate every offset above.
    return ExtractedDocument(
        text="".join(parts).rstrip(),
        filename=filename,
        page_starts=page_starts,
    )
