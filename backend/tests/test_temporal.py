"""Regulation version registry tests.

The property under test throughout: a denial is judged against the law as it
stood on its date, and the system can say precisely how that differs from
current law.
"""

from __future__ import annotations

from datetime import date

import pytest

from app.graph.evidence import version_diff_chunk
from app.retrieval.temporal import (
    OverlappingVersions,
    RegulationRegistry,
    RegulationVersion,
)

V2019 = RegulationVersion(
    statute_id="ca-1371.4",
    version_id="ca-1371.4@2019",
    citation="Cal. Health & Safety s 1371.4",
    text="An insurer shall reimburse emergency services, subject to prior authorization.",
    effective_from=date(2019, 1, 1),
    effective_to=date(2022, 12, 31),
)
V2023 = RegulationVersion(
    statute_id="ca-1371.4",
    version_id="ca-1371.4@2023",
    citation="Cal. Health & Safety s 1371.4",
    text="An insurer shall reimburse emergency services and may not require prior authorization.",
    effective_from=date(2023, 1, 1),
)


@pytest.fixture
def registry() -> RegulationRegistry:
    r = RegulationRegistry()
    # Registered out of order on purpose — the registry sorts the chain.
    r.register_all([V2023, V2019])
    return r


class TestVersionResolution:
    def test_resolves_version_in_force(self, registry):
        assert registry.as_of("ca-1371.4", date(2021, 6, 1)).version_id == "ca-1371.4@2019"
        assert registry.as_of("ca-1371.4", date(2024, 1, 1)).version_id == "ca-1371.4@2023"

    def test_boundary_dates_are_inclusive(self, registry):
        """An amendment 'effective January 1' governs from that day."""
        assert registry.as_of("ca-1371.4", date(2022, 12, 31)).version_id == "ca-1371.4@2019"
        assert registry.as_of("ca-1371.4", date(2023, 1, 1)).version_id == "ca-1371.4@2023"

    def test_before_first_version_is_none(self, registry):
        """The statute did not exist yet — an absent answer, not the earliest."""
        assert registry.as_of("ca-1371.4", date(2015, 1, 1)) is None

    def test_unknown_statute_is_none(self, registry):
        assert registry.as_of("tx-9999", date(2021, 1, 1)) is None

    def test_gap_in_published_record_is_none(self):
        """A repealed-then-reenacted statute has a date with no law in force.

        Returning the prior version here would cite repealed law as governing.
        """
        r = RegulationRegistry()
        r.register_all([
            RegulationVersion(
                statute_id="s", version_id="s@a", citation="s", text="old",
                effective_from=date(2015, 1, 1), effective_to=date(2016, 12, 31),
            ),
            RegulationVersion(
                statute_id="s", version_id="s@b", citation="s", text="new",
                effective_from=date(2020, 1, 1),
            ),
        ])
        assert r.as_of("s", date(2018, 6, 1)) is None


class TestValidation:
    def test_overlapping_versions_rejected(self):
        """An overlap is an ingestion bug; resolving it silently would make
        the system cite law it cannot justify."""
        r = RegulationRegistry()
        with pytest.raises(OverlappingVersions):
            r.register_all([
                V2019,
                RegulationVersion(
                    statute_id="ca-1371.4", version_id="ca-1371.4@2022",
                    citation="c", text="t",
                    effective_from=date(2022, 6, 1),  # inside V2019's window
                ),
            ])

    def test_open_ended_version_followed_by_another_rejected(self):
        r = RegulationRegistry()
        with pytest.raises(OverlappingVersions):
            r.register_all([
                RegulationVersion(
                    statute_id="s", version_id="s@a", citation="c", text="t",
                    effective_from=date(2019, 1, 1),  # no effective_to
                ),
                RegulationVersion(
                    statute_id="s", version_id="s@b", citation="c", text="t",
                    effective_from=date(2023, 1, 1),
                ),
            ])

    def test_inverted_window_rejected(self):
        r = RegulationRegistry()
        with pytest.raises(ValueError, match="precedes"):
            r.register(
                RegulationVersion(
                    statute_id="s", version_id="s@a", citation="c", text="t",
                    effective_from=date(2023, 1, 1),
                    effective_to=date(2019, 1, 1),
                )
            )


class TestDiff:
    def test_detects_amendment_after_denial_date(self, registry):
        diff = registry.diff_since("ca-1371.4", date(2021, 6, 1))

        assert diff.amended_since
        assert diff.applicable.version_id == "ca-1371.4@2019"
        assert diff.current.version_id == "ca-1371.4@2023"
        assert "governs this claim, not the current text" in diff.summary()

    def test_no_amendment_when_current_version_applied(self, registry):
        diff = registry.diff_since("ca-1371.4", date(2024, 6, 1))
        assert not diff.amended_since
        assert "has not been amended" in diff.summary()

    def test_counts_intervening_versions(self):
        r = RegulationRegistry()
        r.register_all([
            RegulationVersion(statute_id="s", version_id="s@1", citation="c", text="a",
                              effective_from=date(2018, 1, 1), effective_to=date(2019, 12, 31)),
            RegulationVersion(statute_id="s", version_id="s@2", citation="c", text="b",
                              effective_from=date(2020, 1, 1), effective_to=date(2021, 12, 31)),
            RegulationVersion(statute_id="s", version_id="s@3", citation="c", text="c",
                              effective_from=date(2022, 1, 1)),
        ])
        diff = r.diff_since("s", date(2018, 6, 1))
        assert [v.version_id for v in diff.intervening] == ["s@2"]
        assert "amended 2 time(s)" in diff.summary()

    def test_diff_before_statute_existed_is_none(self, registry):
        """'Did not exist yet' and 'unchanged' are opposite conclusions."""
        assert registry.diff_since("ca-1371.4", date(2015, 1, 1)) is None


class TestVersionDiffChunk:
    def test_chunk_carries_applicable_text_and_window(self, registry):
        diff = registry.diff_since("ca-1371.4", date(2021, 6, 1))
        chunk = version_diff_chunk(diff)

        assert "subject to prior authorization" in chunk.text  # the 2019 text
        assert "may not require prior authorization" not in chunk.text
        assert chunk.effective_from == date(2019, 1, 1)
        assert chunk.applies_on(date(2021, 6, 1))
        assert not chunk.applies_on(date(2024, 1, 1))
