"""Ingestion tests.

The load-bearing test here is `test_offsets_index_back_into_the_document`.
Three separate features resolve character offsets against the document —
evidence highlighting, citation spans, the beam's origin anchor — and all
three fail silently and identically if a chunk's text drifts from its own
slice. Everything else in this file is a detail; that one is the contract.
"""

from __future__ import annotations

import pytest

from app.core.models import SourceKind
from app.ingestion import (
    ExtractionError,
    chunk_document,
    extract,
    normalise,
)
from app.ingestion.chunking import MAX_CHUNK_CHARS

POLICY = """CERTIFICATE OF COVERAGE
Acme Health Plan

This document describes your benefits.

Section 7.2 Exclusions

7.2(a) This plan does not provide benefits for services rendered in a
hospital emergency department. The charges are the member's responsibility.

7.2(b) Notwithstanding 7.2(a), the plan covers emergency services required
to evaluate or stabilize an emergency medical condition, as defined in
Section 2.1 of this Certificate.

Section 4.3 Cosmetic Surgery

This plan does not provide benefits for cosmetic surgery. Cosmetic surgery
means a procedure performed principally to improve appearance and not to
restore bodily function.
"""


def _chunks(text: str, kind: SourceKind = SourceKind.POLICY):
    document = extract(text.encode("utf-8"), "policy.txt")
    return document, chunk_document(document, source_kind=kind, document_id="doc-1")


def test_offsets_index_back_into_the_document():
    document, chunks = _chunks(POLICY)
    assert chunks
    for chunk in chunks:
        assert chunk.text == document.text[chunk.char_start : chunk.char_end]


def test_adjacent_subsections_are_separately_retrievable():
    """The carve-back case depends on this.

    If 7.2(a) and 7.2(b) land in one chunk, a retriever cannot return the
    exclusion without the clause that limits it — which sounds safe, but the
    reverse is what matters: a chunker that split them badly could surface
    the exclusion alone and confirm the denial.
    """
    _, chunks = _chunks(POLICY)
    locators = [c.locator for c in chunks]
    assert "7.2(a)" in locators
    assert "7.2(b)" in locators

    carveback = next(c for c in chunks if c.locator == "7.2(b)")
    assert "stabilize an emergency medical condition" in carveback.text
    assert "does not provide benefits" not in carveback.text


def test_preamble_before_the_first_heading_is_kept():
    """Cover pages and definitions are evidence, not chrome."""
    _, chunks = _chunks(POLICY)
    assert any("describes your benefits" in c.text for c in chunks)


def test_chunk_ids_are_unique():
    text = "Section 1.1 A\n\n" + ("body one. " * 30)
    text += "\n\nSection 1.1 A\n\n" + ("body two. " * 30)
    _, chunks = _chunks(text)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))


def test_oversized_section_splits_on_paragraph_boundaries():
    paragraph = "This clause is long. " * 40  # ~840 chars
    text = f"Section 9.1 Long\n\n{paragraph}\n\n{paragraph}\n\n{paragraph}"
    document, chunks = _chunks(text)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.text == document.text[chunk.char_start : chunk.char_end]
    # A split may exceed the budget only when a single paragraph does.
    assert all(
        len(c.text) <= MAX_CHUNK_CHARS or c.text.count("\n\n") == 0 for c in chunks
    )


def test_unnumbered_prose_still_chunks():
    """A denial letter has no section numbering and must still ingest."""
    letter = "NOTICE OF ADVERSE BENEFIT DETERMINATION\n\nWe have denied your claim."
    document, chunks = _chunks(letter, SourceKind.DENIAL)
    assert len(chunks) == 1
    assert chunks[0].source_kind is SourceKind.DENIAL
    assert chunks[0].text == document.text


def test_normalise_is_idempotent():
    messy = "line one   \r\n\r\n\r\n\r\nline two here"
    once = normalise(messy)
    assert normalise(once) == once
    assert "\r" not in once
    assert " " not in once


def test_dehyphenation_joins_only_lowercase_pairs():
    assert "emergency" in normalise("emer-\ngency care")
    # A real compound spanning a line break keeps its hyphen.
    assert "Blue-\nCross" in normalise("Blue-\nCross plan")


def test_empty_and_unsupported_files_are_rejected():
    with pytest.raises(ExtractionError, match="empty"):
        extract(b"", "policy.pdf")
    with pytest.raises(ExtractionError, match="supported formats"):
        extract(b"data", "policy.docx")


def test_whitespace_only_file_is_rejected_not_silently_accepted():
    """A scanned PDF extracts to nothing; saying so beats auditing an empty
    document and reporting insufficient evidence."""
    with pytest.raises(ExtractionError):
        extract(b"   \n\n  \n", "scan.txt")
