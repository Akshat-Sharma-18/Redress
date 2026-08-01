"""Regulation version registry.

Chunk-level `effective_from`/`effective_to` filtering (see `hybrid.py`) keeps
the wrong version of a statute out of the evidence. This registry answers the
questions that filtering alone cannot:

  - which version of statute X was in force on date D?
  - what changed between the version that applied and the current one?

The second question is what makes an appeal letter useful. "Your denial
conflicts with s 1371.4" is weak if the insurer can reply that the section
was amended; "as of your denial date, s 1371.4 required prior authorization,
and the 2023 amendment removing that requirement postdates your claim" is a
statement about the record that survives being checked.

Versions are half-open intervals `[effective_from, effective_to]` with
inclusive ends, matching how statutes are actually published (an amendment
"effective January 1" replaces the prior text from that date).
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class RegulationVersion:
    """One published version of a statute section."""

    statute_id: str          # stable across versions, e.g. "ca-ins-1371.4"
    version_id: str          # unique per version, e.g. "ca-ins-1371.4@2019"
    citation: str            # human-facing, e.g. "Cal. Health & Safety s 1371.4"
    text: str
    effective_from: date
    effective_to: date | None = None   # None = still in force

    def covers(self, when: date) -> bool:
        if when < self.effective_from:
            return False
        return self.effective_to is None or when <= self.effective_to


@dataclass
class VersionDiff:
    """What changed between the version in force and the current one."""

    statute_id: str
    as_of: date
    applicable: RegulationVersion
    current: RegulationVersion
    intervening: list[RegulationVersion] = field(default_factory=list)

    @property
    def amended_since(self) -> bool:
        return self.applicable.version_id != self.current.version_id

    def summary(self) -> str:
        if not self.amended_since:
            return (
                f"{self.applicable.citation} has not been amended since "
                f"{self.as_of.isoformat()}; the current text applied."
            )
        n = len(self.intervening) + 1
        return (
            f"{self.applicable.citation} was amended {n} time(s) after "
            f"{self.as_of.isoformat()}. The version in force on that date "
            f"(effective {self.applicable.effective_from.isoformat()}) governs "
            f"this claim, not the current text (effective "
            f"{self.current.effective_from.isoformat()})."
        )


class OverlappingVersions(ValueError):
    """Two versions of one statute claim the same date.

    Raised at registration rather than resolved by preference. An overlap is
    an ingestion bug, and silently picking one version would make the system
    cite law it cannot justify — the exact failure this layer exists to
    prevent.
    """


class RegulationRegistry:
    """Version history per statute, queryable by date."""

    def __init__(self) -> None:
        self._by_statute: dict[str, list[RegulationVersion]] = {}
        self._starts: dict[str, list[date]] = {}

    def register(self, version: RegulationVersion) -> None:
        self.register_all([version])

    def register_all(self, versions: list[RegulationVersion]) -> None:
        touched: set[str] = set()
        for v in versions:
            if v.effective_to is not None and v.effective_to < v.effective_from:
                raise ValueError(
                    f"{v.version_id}: effective_to precedes effective_from"
                )
            self._by_statute.setdefault(v.statute_id, []).append(v)
            touched.add(v.statute_id)

        for statute_id in touched:
            chain = sorted(
                self._by_statute[statute_id], key=lambda v: v.effective_from
            )
            self._validate(statute_id, chain)
            self._by_statute[statute_id] = chain
            self._starts[statute_id] = [v.effective_from for v in chain]

    @staticmethod
    def _validate(statute_id: str, chain: list[RegulationVersion]) -> None:
        for earlier, later in zip(chain, chain[1:]):
            if earlier.effective_to is None:
                raise OverlappingVersions(
                    f"{statute_id}: {earlier.version_id} is open-ended but "
                    f"{later.version_id} starts on "
                    f"{later.effective_to and later.effective_from}"
                )
            if earlier.effective_to >= later.effective_from:
                raise OverlappingVersions(
                    f"{statute_id}: {earlier.version_id} (ends "
                    f"{earlier.effective_to.isoformat()}) overlaps "
                    f"{later.version_id} (starts "
                    f"{later.effective_from.isoformat()})"
                )

    def versions(self, statute_id: str) -> list[RegulationVersion]:
        return list(self._by_statute.get(statute_id, []))

    def as_of(self, statute_id: str, when: date) -> RegulationVersion | None:
        """The version in force on `when`, or None if the statute did not
        yet exist (or a gap in the published record covers that date)."""
        chain = self._by_statute.get(statute_id)
        if not chain:
            return None
        # Rightmost version whose effective_from <= when.
        idx = bisect_right(self._starts[statute_id], when) - 1
        if idx < 0:
            return None
        candidate = chain[idx]
        return candidate if candidate.covers(when) else None

    def current(self, statute_id: str) -> RegulationVersion | None:
        chain = self._by_statute.get(statute_id)
        return chain[-1] if chain else None

    def diff_since(self, statute_id: str, when: date) -> VersionDiff | None:
        """How the statute has changed since `when`.

        Returns None when no version was in force on that date — an absent
        answer, not an empty one, because "the statute did not exist yet" and
        "the statute is unchanged" are opposite conclusions for an appeal.
        """
        applicable = self.as_of(statute_id, when)
        current = self.current(statute_id)
        if applicable is None or current is None:
            return None

        chain = self._by_statute[statute_id]
        start = chain.index(applicable)
        end = chain.index(current)
        return VersionDiff(
            statute_id=statute_id,
            as_of=when,
            applicable=applicable,
            current=current,
            intervening=chain[start + 1 : end],
        )
