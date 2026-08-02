"""Uploaded documents in, retrievable chunks out.

This package is the boundary between "a person's PDF" and everything the
retrieval stack already knows how to handle. Nothing downstream of here can
tell whether a chunk came from an uploaded policy or a hand-written golden
fixture, which is the property that lets the eval harness keep measuring the
same system that users are actually running.
"""

from app.ingestion.chunking import MAX_CHUNK_CHARS, chunk_document
from app.ingestion.extract import (
    MAX_UPLOAD_BYTES,
    SUPPORTED_SUFFIXES,
    ExtractedDocument,
    ExtractionError,
    extract,
    normalise,
)

__all__ = [
    "MAX_CHUNK_CHARS",
    "MAX_UPLOAD_BYTES",
    "SUPPORTED_SUFFIXES",
    "ExtractedDocument",
    "ExtractionError",
    "chunk_document",
    "extract",
    "normalise",
]
