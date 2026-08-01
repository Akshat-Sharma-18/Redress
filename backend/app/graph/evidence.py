"""Turning graph findings and version diffs into citable evidence.

Both the pattern query and the regulation registry produce *derived* facts —
counts over complaint records, a diff between statute versions — rather than
retrieved text. To reach the reconciliation agent they have to become chunks,
because the agent is constrained to cite chunk ids and a fact it cannot cite
is a fact it must ignore.

The derived text is generated deterministically from the underlying records,
so the "verbatim quote" contract still holds: the agent quotes a sentence
this module wrote, and this module wrote it from the graph. No model is in
this loop.

Framing matters as much as content here. A regulator overturning a similar
denial is *contextual* evidence — it is not a ruling about this claim, and
an agent that treats it as one would produce exactly the overconfident
verdict the confidence gate exists to prevent. The generated text says so
explicitly, and the prompt-visible caveat is part of the chunk rather than
something the system prompt has to remember.
"""

from __future__ import annotations

from datetime import date

from app.core.models import Chunk, ScoredChunk, SourceKind
from app.graph.models import PatternFinding
from app.graph.store import GraphStore
from app.retrieval.temporal import RegulationRegistry, VersionDiff

_PATTERN_CAVEAT = (
    "This is regulatory context about prior, separate complaints. It does "
    "not establish that the present denial is improper, and it cannot on "
    "its own support a finding of 'contradicted'."
)

_NO_RULING_CAVEAT = (
    "No prior complaint reached a ruling, so this record says nothing about "
    "how a regulator would view the present denial."
)


def pattern_chunk(finding: PatternFinding) -> Chunk:
    """Render one PatternFinding as a citable precedent chunk."""
    lines = [finding.summary()]

    if finding.ruled == 0:
        lines.append(_NO_RULING_CAVEAT)
    else:
        lines.append(_PATTERN_CAVEAT)

    if finding.settled or finding.withdrawn:
        lines.append(
            f"A further {finding.settled} complaint(s) settled and "
            f"{finding.withdrawn} were withdrawn without a ruling."
        )
    if finding.statute_ids:
        lines.append(
            "Statutes applied in those complaints: "
            + ", ".join(finding.statute_ids)
            + "."
        )

    return Chunk(
        id=f"pattern:{finding.insurer_id}:{finding.reason_code}",
        text=" ".join(lines),
        source_kind=SourceKind.PRECEDENT,
        document_id=f"doi-complaints:{finding.insurer_id}",
        locator=(
            f"DOI complaint record — {finding.insurer_label}, "
            f"reason {finding.reason_code}"
        ),
        metadata={
            "overturned": finding.overturned,
            "upheld": finding.upheld,
            "ruled": finding.ruled,
            "total": finding.total,
            "complaint_ids": finding.complaint_ids,
            "statute_ids": finding.statute_ids,
            "derived": True,
        },
    )


def version_diff_chunk(diff: VersionDiff) -> Chunk:
    """Render a statute's amendment history as a citable chunk.

    Only emitted when the statute actually changed. An unchanged statute
    needs no note, and generating one would pad every evidence pack with a
    sentence the agent has to read past.
    """
    return Chunk(
        id=f"version-diff:{diff.statute_id}@{diff.as_of.isoformat()}",
        text=(
            f"{diff.summary()} Text in force on the denial date: "
            f"{diff.applicable.text}"
        ),
        source_kind=SourceKind.STATUTE,
        document_id=diff.statute_id,
        locator=f"{diff.applicable.citation} (as of {diff.as_of.isoformat()})",
        effective_from=diff.applicable.effective_from,
        effective_to=diff.applicable.effective_to,
        metadata={
            "amended_since": diff.amended_since,
            "applicable_version": diff.applicable.version_id,
            "current_version": diff.current.version_id,
            "derived": True,
        },
    )


class GraphEnricher:
    """Adds graph- and version-derived evidence to a sub-claim's evidence set.

    Sits alongside the four retrieval pipelines rather than inside one: its
    inputs are the *case* (which insurer, which reason code, which denial
    date), not the sub-claim text, so it produces the same supplementary
    evidence for every sub-claim in a case.
    """

    def __init__(
        self,
        store: GraphStore | None = None,
        registry: RegulationRegistry | None = None,
        score: float = 0.0,
    ):
        self.store = store
        self.registry = registry
        # Derived evidence carries score 0.0 by default: it was not retrieved
        # by relevance and should not displace retrieved text in the ranking.
        self.score = score

    def enrich(
        self,
        *,
        insurer_id: str | None,
        reason_code: str | None,
        denial_date: date | None,
        statute_ids: list[str] | None = None,
    ) -> list[ScoredChunk]:
        chunks: list[Chunk] = []

        if self.store is not None and insurer_id:
            for finding in self.store.insurer_pattern(insurer_id, reason_code):
                # A pattern with no adjudicated complaints is not a pattern.
                # Emitting it would put a suggestive-looking precedent chunk
                # in front of the agent that says nothing.
                if finding.ruled == 0 and finding.total < 2:
                    continue
                chunks.append(pattern_chunk(finding))

        if self.registry is not None and denial_date is not None:
            for statute_id in statute_ids or []:
                diff = self.registry.diff_since(statute_id, denial_date)
                if diff is not None and diff.amended_since:
                    chunks.append(version_diff_chunk(diff))

        return [
            ScoredChunk(
                chunk=c, score=self.score, provenance={"graph": self.score}
            )
            for c in chunks
        ]
