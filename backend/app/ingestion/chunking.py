"""Split a canonical document into retrievable, citable chunks.

The golden set is hand-chunked: a human decided that Section 7.2(a) and
7.2(b) are two chunks, which is exactly why the emergency-carveback case is
solvable — the carve-back is retrievable independently of the exclusion it
modifies. Uploaded documents get no such help, so this module has to
reconstruct that structure from the text, and how well it does so sets a
ceiling on what retrieval can possibly find.

That leads to the one rule everything here follows: **split on the document's
own section boundaries, never on a fixed window.** A clause cut in half by a
character budget is a clause that can be retrieved without the sentence that
qualifies it, which is precisely the failure this system exists to catch.
Fixed-size windowing is used only as a fallback for prose that has no
numbering at all, and only at paragraph boundaries.

Invariant maintained throughout: ``chunk.text == document[char_start:char_end]``.
Citation spans, evidence highlighting, and the beam's origin anchor all
resolve against the document using those offsets, so a chunk whose text has
been trimmed or rewritten relative to its own slice would silently break all
three. Whitespace is stripped by *moving the offsets*, never by editing text.
"""

from __future__ import annotations

import re

from app.core.models import Chunk, SourceKind
from app.ingestion.extract import ExtractedDocument

#: Upper bound on a chunk, in characters. Sized against the 8192-token context
#: the Ollama backend runs with: six chunks per corpus at this size leaves
#: room for the denial letter, the system prompt, and the response.
MAX_CHUNK_CHARS = 1600

#: A section shorter than this is treated as a heading that got separated from
#: its body ("ARTICLE IV" alone on a line) and is merged into what follows.
#: Retrieving a bare heading wastes one of six evidence slots on nothing.
MIN_CHUNK_CHARS = 80

#: Section headings, anchored to line starts. Ordered most specific first so
#: "Section 7.2(a)" wins over the bare-numeric pattern.
#:
#: These cover the numbering conventions in US policy documents and insurance
#: codes. A document numbered some other way still ingests — it falls through
#: to paragraph chunking with page-based locators, and loses only the ability
#: to cite by section name.
_HEADING = re.compile(
    r"""^[ \t]*(
          (?:ARTICLE|Article)\s+(?:[IVXLC]+|\d+)\b[^\n]{0,80}
        | (?:SECTION|Section|Sec\.)\s*\d+(?:\.\d+)*(?:\s*\([a-z0-9]+\))*
        | §+\s*\d+(?:\.\d+)*(?:\s*\([a-z0-9]+\))*
        | \d+\.\d+(?:\.\d+)*(?:\s*\([a-z0-9]+\))*
    )""",
    re.VERBOSE | re.MULTILINE,
)

#: Paragraph break used when a section has to be subdivided.
_PARAGRAPH = re.compile(r"\n\s*\n")

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def _slug(text: str) -> str:
    return _SLUG_STRIP.sub("-", text.lower()).strip("-")


def _trim(text: str, start: int) -> tuple[str, int, int]:
    """Strip surrounding whitespace by advancing offsets, not by editing.

    Returns the trimmed text with the start/end offsets that still satisfy
    ``trimmed == document[start:end]``.
    """
    lead = len(text) - len(text.lstrip())
    trimmed = text.strip()
    return trimmed, start + lead, start + lead + len(trimmed)


def _sections(text: str) -> list[tuple[str | None, int, int]]:
    """Split into (locator, start, end) triples on detected headings.

    Text before the first heading becomes a leading section with no locator —
    on a policy that is the cover page and definitions preamble, which is
    ordinary evidence and must not be dropped.
    """
    matches = list(_HEADING.finditer(text))
    if not matches:
        return [(None, 0, len(text))]

    spans: list[tuple[str | None, int, int]] = []
    if matches[0].start() > 0:
        spans.append((None, 0, matches[0].start()))

    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        locator = " ".join(match.group(1).split())
        spans.append((locator, match.start(), end))
    return spans


def _subdivide(text: str, start: int) -> list[tuple[int, int]]:
    """Break an oversized section at paragraph boundaries.

    Paragraphs are accumulated greedily up to the budget. A single paragraph
    larger than the budget is emitted whole rather than cut: an over-long
    chunk costs context, while a chunk severed mid-sentence costs meaning,
    and only one of those is recoverable downstream.
    """
    if len(text) <= MAX_CHUNK_CHARS:
        return [(start, start + len(text))]

    breaks = [0]
    for match in _PARAGRAPH.finditer(text):
        breaks.append(match.end())
    breaks.append(len(text))

    spans: list[tuple[int, int]] = []
    window_start = 0
    for index in range(1, len(breaks)):
        window_end = breaks[index]
        if window_end - window_start < MAX_CHUNK_CHARS:
            continue
        # Prefer the previous boundary; use this one only when a single
        # paragraph already exceeds the budget on its own.
        cut = breaks[index - 1] if breaks[index - 1] > window_start else window_end
        spans.append((start + window_start, start + cut))
        window_start = cut
    if window_start < len(text):
        spans.append((start + window_start, start + len(text)))
    return spans


def chunk_document(
    document: ExtractedDocument,
    *,
    source_kind: SourceKind,
    document_id: str,
) -> list[Chunk]:
    """Chunk one extracted document into citable units."""
    text = document.text
    raw_sections = _sections(text)

    # Absorb runt sections into a neighbour. A heading separated from its body
    # ("ARTICLE IV" alone on a line) has no citable content, and retrieving one
    # would spend an evidence slot on nothing.
    #
    # Runts merge *backwards* by default, which deliberately leaves the
    # following section's own locator intact — "Section 7.2 Exclusions" as a
    # bare heading is less useful to cite than the "7.2(a)" that follows it. A
    # runt at the very start of the document has nothing behind it, so it
    # merges forwards instead and lends its locator to what follows.
    merged: list[tuple[str | None, int, int]] = []
    for locator, start, end in raw_sections:
        if merged and (end - start) < MIN_CHUNK_CHARS:
            previous_locator, previous_start, _ = merged[-1]
            merged[-1] = (previous_locator, previous_start, end)
        elif merged and (merged[-1][2] - merged[-1][1]) < MIN_CHUNK_CHARS:
            keep_locator = merged[-1][0] or locator
            merged[-1] = (keep_locator, merged[-1][1], end)
        else:
            merged.append((locator, start, end))

    chunks: list[Chunk] = []
    used_ids: set[str] = set()
    for locator, start, end in merged:
        for span_start, span_end in _subdivide(text[start:end], start):
            body, offset_start, offset_end = _trim(
                text[span_start:span_end], span_start
            )
            if not body:
                continue

            page = document.page_of(offset_start)
            resolved = locator
            if resolved is None:
                resolved = f"p. {page}" if page else None

            base = _slug(resolved) if resolved else f"{document_id}-c"
            chunk_id = base
            suffix = 1
            while chunk_id in used_ids:
                suffix += 1
                chunk_id = f"{base}-{suffix}"
            used_ids.add(chunk_id)

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=body,
                    source_kind=source_kind,
                    document_id=document_id,
                    locator=resolved,
                    char_start=offset_start,
                    char_end=offset_end,
                    metadata={"filename": document.filename}
                    | ({"page": page} if page else {}),
                )
            )
    return chunks
