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


def test_short_clause_keeps_its_own_locator():
    """Regression: a brief carve-back must stay independently citable.

    Merging sections by total length swallowed this clause into the exclusion
    it limits, costing it both its locator and its ability to be retrieved on
    its own. The shortest clauses are disproportionately the decisive ones —
    a one-sentence "notwithstanding" is the whole appeal.
    """
    text = (
        "Section 7.2 Exclusions\n\n"
        "7.2(a) This plan does not provide benefits for services rendered in a\n"
        "hospital emergency department, and the charges are the member's own\n"
        "responsibility in every case.\n\n"
        "7.2(b) Notwithstanding 7.2(a), emergency care is covered.\n"
    )
    _, chunks = _chunks(text)
    locators = [c.locator for c in chunks]
    assert "7.2(b)" in locators

    carveback = next(c for c in chunks if c.locator == "7.2(b)")
    assert "emergency care is covered" in carveback.text
    assert "does not provide benefits" not in carveback.text


def test_bare_heading_does_not_become_its_own_chunk():
    """"ARTICLE IV" alone is not evidence; it must not occupy an evidence slot."""
    text = (
        "ARTICLE IV\n\n"
        "Section 4.3 Cosmetic Surgery\n\n"
        "This plan does not provide benefits for cosmetic surgery, meaning a\n"
        "procedure performed principally to improve appearance.\n"
    )
    _, chunks = _chunks(text)
    assert not any(c.text.strip() == "ARTICLE IV" for c in chunks)
    # The more specific locator survives the merge.
    assert "Section 4.3" in [c.locator for c in chunks]
    assert any("cosmetic surgery" in c.text for c in chunks)


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


# --- PDF ---------------------------------------------------------------
#
# PDF is the format users actually upload, and it is the one whose text
# extraction is lossy. These tests run a real PDF through pypdf rather than
# trusting that the .txt path generalises.

reportlab = pytest.importorskip(
    "reportlab", reason="reportlab generates the PDF fixtures"
)


def _pdf(pages: list[list[str]]) -> bytes:
    """Render pages of lines to a PDF, the way a policy document arrives."""
    import io

    from reportlab.lib.pagesizes import LETTER
    from reportlab.pdfgen import canvas

    buffer = io.BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=LETTER)
    for lines in pages:
        y = 720
        for line in lines:
            pdf.drawString(72, y, line)
            y -= 16
        pdf.showPage()
    pdf.save()
    return buffer.getvalue()


def test_pdf_extraction_preserves_section_structure():
    data = _pdf(
        [
            [
                "CERTIFICATE OF COVERAGE",
                "",
                "Section 7.2 Exclusions",
                "",
                "7.2(a) This plan does not provide benefits for services",
                "rendered in a hospital emergency department.",
                "",
                "7.2(b) Notwithstanding 7.2(a), the plan covers emergency",
                "services required to stabilize an emergency condition.",
            ]
        ]
    )
    document = extract(data, "policy.pdf")
    chunks = chunk_document(
        document, source_kind=SourceKind.POLICY, document_id="doc-pdf"
    )

    locators = [c.locator for c in chunks]
    assert "7.2(a)" in locators
    assert "7.2(b)" in locators
    for chunk in chunks:
        assert chunk.text == document.text[chunk.char_start : chunk.char_end]


def test_pdf_page_numbers_survive_as_locators():
    """Prose with no section numbering still has to cite as something."""
    document = extract(
        _pdf([["Page one prose without numbering."], ["Page two prose."]]),
        "letter.pdf",
    )
    assert len(document.page_starts) == 2
    assert document.page_of(0) == 1
    # The second page's own offset must resolve to page 2, which is the whole
    # point of tracking starts rather than inferring them after joining.
    assert document.page_of(document.page_starts[1]) == 2

    chunks = chunk_document(
        document, source_kind=SourceKind.DENIAL, document_id="doc-pdf-2"
    )
    assert any(c.locator == "p. 1" for c in chunks)


def test_pdf_offsets_are_valid_across_a_page_break():
    document = extract(
        _pdf([["Section 1.1 First", "Body of the first section here."],
              ["Section 2.1 Second", "Body of the second section here."]]),
        "two-page.pdf",
    )
    chunks = chunk_document(
        document, source_kind=SourceKind.POLICY, document_id="doc-pdf-3"
    )
    assert len(chunks) >= 2
    for chunk in chunks:
        assert chunk.text == document.text[chunk.char_start : chunk.char_end]
