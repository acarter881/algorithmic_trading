"""Unit tests for the Arena leaderboard monitor."""

from __future__ import annotations

import datetime
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from autotrader.config.models import ArenaMonitorConfig
from autotrader.signals.arena_monitor import ArenaMonitor, ArenaMonitorFailureThresholdError
from autotrader.signals.arena_types import (
    LeaderboardEntry,
    LeaderboardSnapshot,
    PairwiseAggregate,
)
from autotrader.signals.base import SignalUrgency

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ── Helper Factories ──────────────────────────────────────────────────


def _entry(
    name: str = "TestModel",
    org: str = "TestOrg",
    rank: int = 1,
    rank_ub: int = 1,
    score: float = 1300.0,
    votes: int = 10000,
    release_date: str = "",
) -> LeaderboardEntry:
    return LeaderboardEntry(
        model_name=name,
        organization=org,
        rank=rank,
        rank_ub=rank_ub,
        rank_lb=rank,
        score=score,
        votes=votes,
        release_date=release_date,
    )


def _snapshot(entries: list[LeaderboardEntry], url: str = "https://test.com") -> LeaderboardSnapshot:
    return LeaderboardSnapshot(
        entries=entries,
        source_url=url,
        captured_at=datetime.datetime(2026, 2, 26, 12, 0, 0),
    )


# ── LeaderboardSnapshot ──────────────────────────────────────────────


class TestLeaderboardSnapshot:
    def test_top_model(self) -> None:
        snap = _snapshot(
            [
                _entry("A", rank_ub=3),
                _entry("B", rank_ub=1),
                _entry("C", rank_ub=2),
            ]
        )
        assert snap.top_model == "B"

    def test_top_model_empty(self) -> None:
        snap = _snapshot([])
        assert snap.top_model is None

    def test_top_org(self) -> None:
        snap = _snapshot(
            [
                _entry("A", org="OrgA", rank_ub=2),
                _entry("B", org="OrgB", rank_ub=1),
            ]
        )
        assert snap.top_org == "OrgB"

    def test_top_model_tiebreak_score(self) -> None:
        snap = _snapshot(
            [
                _entry("A", rank_ub=1, score=1400.0),
                _entry("B", rank_ub=1, score=1410.0),
            ]
        )
        assert snap.top_model == "B"

    def test_top_model_tiebreak_votes(self) -> None:
        snap = _snapshot(
            [
                _entry("A", rank_ub=1, score=1400.0, votes=10100),
                _entry("B", rank_ub=1, score=1400.0, votes=10200),
            ]
        )
        assert snap.top_model == "B"

    def test_top_model_tiebreak_release_date(self) -> None:
        snap = _snapshot(
            [
                _entry("A", rank_ub=1, score=1400.0, votes=10100, release_date="2025-05-01"),
                _entry("B", rank_ub=1, score=1400.0, votes=10100, release_date="2025-04-01"),
            ]
        )
        assert snap.top_model == "B"

    def test_top_org_tiebreak_consistent_with_settlement_rules(self) -> None:
        snap = _snapshot(
            [
                _entry("A", org="OrgA", rank_ub=1, score=1500.0, votes=11000, release_date="2025-03-10"),
                _entry("B", org="OrgB", rank_ub=1, score=1500.0, votes=11000, release_date="2025-03-01"),
            ]
        )
        assert snap.top_model == "B"
        assert snap.top_org == "OrgB"

    def test_by_model_name(self) -> None:
        snap = _snapshot([_entry("X"), _entry("Y")])
        lookup = snap.by_model_name()
        assert "X" in lookup
        assert "Y" in lookup
        assert lookup["X"].model_name == "X"


# ── Diffing ───────────────────────────────────────────────────────────


class TestDiffSnapshots:
    def setup_method(self) -> None:
        self.monitor = ArenaMonitor()

    def test_no_changes(self) -> None:
        entries = [_entry("A"), _entry("B")]
        prev = _snapshot(entries)
        curr = _snapshot(entries)
        diff = self.monitor.diff_snapshots(prev, curr)
        assert diff.rank_changes == []
        assert diff.score_changes == []
        assert diff.new_entries == []
        assert diff.removed_entries == []
        assert diff.leader_changed is False

    def test_rank_change(self) -> None:
        prev = _snapshot([_entry("A", rank=1, rank_ub=1), _entry("B", rank=2, rank_ub=2)])
        curr = _snapshot([_entry("A", rank=2, rank_ub=2), _entry("B", rank=1, rank_ub=1)])
        diff = self.monitor.diff_snapshots(prev, curr)
        assert len(diff.rank_changes) == 2
        a_change = next(rc for rc in diff.rank_changes if rc.model_name == "A")
        assert a_change.old_rank_ub == 1
        assert a_change.new_rank_ub == 2

    def test_new_model(self) -> None:
        prev = _snapshot([_entry("A")])
        curr = _snapshot([_entry("A"), _entry("B", rank=2, rank_ub=2)])
        diff = self.monitor.diff_snapshots(prev, curr)
        assert len(diff.new_entries) == 1
        assert diff.new_entries[0].model_name == "B"

    def test_removed_model(self) -> None:
        prev = _snapshot([_entry("A"), _entry("B")])
        curr = _snapshot([_entry("A")])
        diff = self.monitor.diff_snapshots(prev, curr)
        assert len(diff.removed_entries) == 1
        assert diff.removed_entries[0].model_name == "B"

    def test_score_change(self) -> None:
        prev = _snapshot([_entry("A", score=1300.0)])
        curr = _snapshot([_entry("A", score=1310.0)])
        diff = self.monitor.diff_snapshots(prev, curr)
        assert len(diff.score_changes) == 1
        assert diff.score_changes[0].score_delta == pytest.approx(10.0)

    def test_score_change_below_threshold(self) -> None:
        prev = _snapshot([_entry("A", score=1300.0)])
        curr = _snapshot([_entry("A", score=1301.0)])
        diff = self.monitor.diff_snapshots(prev, curr)
        # Below default threshold of 3.0
        assert len(diff.score_changes) == 0

    def test_leader_change(self) -> None:
        prev = _snapshot([_entry("A", rank_ub=1), _entry("B", rank_ub=2)])
        curr = _snapshot([_entry("A", rank_ub=2), _entry("B", rank_ub=1)])
        diff = self.monitor.diff_snapshots(prev, curr)
        assert diff.leader_changed is True
        assert diff.new_leader == "B"
        assert diff.previous_leader == "A"

    def test_no_leader_change_same_leader(self) -> None:
        prev = _snapshot([_entry("A", rank_ub=1), _entry("B", rank_ub=2)])
        curr = _snapshot([_entry("A", rank_ub=1), _entry("B", rank_ub=3)])
        diff = self.monitor.diff_snapshots(prev, curr)
        assert diff.leader_changed is False


# ── Signal Generation ─────────────────────────────────────────────────


class TestSignalGeneration:
    def setup_method(self) -> None:
        self.monitor = ArenaMonitor()

    def test_no_signals_on_first_poll(self) -> None:
        """First poll has no previous snapshot, so no diff signals."""
        snap = _snapshot([_entry("A")])
        signals = self.monitor._generate_signals(snap)
        assert signals == []

    def test_ranking_change_signal(self) -> None:
        self.monitor._previous_snapshot = _snapshot(
            [
                _entry("A", rank=1, rank_ub=1),
                _entry("B", rank=2, rank_ub=2),
            ]
        )
        curr = _snapshot(
            [
                _entry("A", rank=2, rank_ub=3),
                _entry("B", rank=1, rank_ub=1),
            ]
        )
        signals = self.monitor._generate_signals(curr)
        ranking_signals = [s for s in signals if s.event_type == "ranking_change"]
        assert len(ranking_signals) == 2

    def test_new_leader_signal(self) -> None:
        self.monitor._previous_snapshot = _snapshot(
            [
                _entry("A", rank_ub=1),
                _entry("B", rank_ub=2),
            ]
        )
        curr = _snapshot(
            [
                _entry("A", rank_ub=2),
                _entry("B", rank_ub=1),
            ]
        )
        signals = self.monitor._generate_signals(curr)
        leader_signals = [s for s in signals if s.event_type == "new_leader"]
        assert len(leader_signals) == 1
        assert leader_signals[0].urgency == SignalUrgency.HIGH
        assert leader_signals[0].data["new_leader"] == "B"
        assert leader_signals[0].data["previous_leader"] == "A"

    def test_score_shift_signal(self) -> None:
        self.monitor._previous_snapshot = _snapshot([_entry("A", score=1300.0)])
        curr = _snapshot([_entry("A", score=1310.0)])
        signals = self.monitor._generate_signals(curr)
        score_signals = [s for s in signals if s.event_type == "score_shift"]
        assert len(score_signals) == 1
        assert score_signals[0].data["score_delta"] == pytest.approx(10.0)

    def test_new_model_signal(self) -> None:
        self.monitor._previous_snapshot = _snapshot([_entry("A")])
        curr = _snapshot([_entry("A"), _entry("B", rank=2, rank_ub=5)])
        signals = self.monitor._generate_signals(curr)
        new_signals = [s for s in signals if s.event_type == "new_model"]
        assert len(new_signals) == 1
        assert new_signals[0].data["model_name"] == "B"

    def test_new_model_rank1_high_urgency(self) -> None:
        self.monitor._previous_snapshot = _snapshot([_entry("A", rank_ub=2)])
        curr = _snapshot([_entry("A", rank_ub=2), _entry("B", rank_ub=1)])
        signals = self.monitor._generate_signals(curr)
        new_signals = [s for s in signals if s.event_type == "new_model"]
        assert len(new_signals) == 1
        assert new_signals[0].urgency == SignalUrgency.HIGH

    def test_removed_model_signal(self) -> None:
        self.monitor._previous_snapshot = _snapshot([_entry("A"), _entry("B")])
        curr = _snapshot([_entry("A")])
        signals = self.monitor._generate_signals(curr)
        removed_signals = [s for s in signals if s.event_type == "model_removed"]
        assert len(removed_signals) == 1
        assert removed_signals[0].data["model_name"] == "B"

    def test_signals_have_target_series(self) -> None:
        self.monitor._previous_snapshot = _snapshot([_entry("A", rank_ub=1)])
        curr = _snapshot([_entry("A", rank_ub=2)])
        signals = self.monitor._generate_signals(curr)
        for signal in signals:
            assert "KXTOPMODEL" in signal.relevant_series
            assert "KXLLM1" in signal.relevant_series

    def test_signals_have_timestamps(self) -> None:
        self.monitor._previous_snapshot = _snapshot([_entry("A", score=1300)])
        curr = _snapshot([_entry("A", score=1310)])
        signals = self.monitor._generate_signals(curr)
        for signal in signals:
            assert isinstance(signal.timestamp, datetime.datetime)
            assert signal.source == "arena_monitor"


# ── ArenaMonitor Interface ────────────────────────────────────────────


class TestArenaMonitorInterface:
    def test_name(self) -> None:
        monitor = ArenaMonitor()
        assert monitor.name == "arena_monitor"

    def test_poll_interval(self) -> None:
        config = ArenaMonitorConfig(poll_interval_seconds=60)
        monitor = ArenaMonitor(config=config)
        assert monitor.poll_interval_seconds == 60

    def test_default_poll_interval(self) -> None:
        monitor = ArenaMonitor()
        assert monitor.poll_interval_seconds == 30

    def test_consecutive_failures_tracking(self) -> None:
        monitor = ArenaMonitor()
        assert monitor.consecutive_failures == 0

    def test_previous_snapshot_property(self) -> None:
        monitor = ArenaMonitor()
        assert monitor.previous_snapshot is None
        snap = _snapshot([_entry("A")])
        monitor.previous_snapshot = snap
        assert monitor.previous_snapshot is snap


# ── HTTP Fetching ─────────────────────────────────────────────────────


class TestFetching:
    @pytest.mark.asyncio
    async def test_initialize_creates_client(self) -> None:
        monitor = ArenaMonitor()
        await monitor.initialize()
        assert monitor._http_client is not None
        await monitor.teardown()

    @pytest.mark.asyncio
    async def test_teardown_closes_client(self) -> None:
        monitor = ArenaMonitor()
        await monitor.initialize()
        await monitor.teardown()
        assert monitor._http_client is None

    @pytest.mark.asyncio
    async def test_fetch_success(self) -> None:
        json_data = (FIXTURES / "arena_leaderboard.json").read_text()
        monitor = ArenaMonitor()
        await monitor.initialize()

        mock_response = MagicMock()
        mock_response.text = json_data
        mock_response.raise_for_status = MagicMock()

        with patch.object(monitor._http_client, "get", new_callable=AsyncMock, return_value=mock_response):
            snapshot = await monitor._fetch_leaderboard()

        assert snapshot is not None
        assert len(snapshot.entries) == 8
        assert monitor.consecutive_failures == 0
        await monitor.teardown()

    @pytest.mark.asyncio
    async def test_fetch_fallback_on_primary_failure(self) -> None:
        json_data = (FIXTURES / "arena_leaderboard.json").read_text()
        monitor = ArenaMonitor()
        await monitor.initialize()

        good_response = MagicMock()
        good_response.text = json_data
        good_response.raise_for_status = MagicMock()

        async def mock_get(url: str) -> MagicMock:
            if "github" in url:
                raise httpx.RequestError("Connection refused")
            return good_response

        with patch.object(monitor._http_client, "get", side_effect=mock_get):
            snapshot = await monitor._fetch_leaderboard()

        assert snapshot is not None
        assert snapshot.source_url == "https://arena.ai/leaderboard/text/overall-no-style-control"
        await monitor.teardown()

    @pytest.mark.asyncio
    async def test_fetch_all_fail(self) -> None:
        monitor = ArenaMonitor()
        await monitor.initialize()

        async def mock_get(url: str) -> MagicMock:
            raise httpx.RequestError("Connection refused")

        with patch.object(monitor._http_client, "get", side_effect=mock_get):
            snapshot = await monitor._fetch_leaderboard()

        assert snapshot is None
        assert monitor.consecutive_failures == 1
        await monitor.teardown()

    @pytest.mark.asyncio
    async def test_fetch_raises_when_failure_threshold_reached(self) -> None:
        config = ArenaMonitorConfig(max_consecutive_failures=2)
        monitor = ArenaMonitor(config=config)
        await monitor.initialize()

        async def mock_get(url: str) -> MagicMock:
            raise httpx.RequestError("Connection refused")

        with patch.object(monitor._http_client, "get", side_effect=mock_get):
            snapshot = await monitor._fetch_leaderboard()
            assert snapshot is None

            with pytest.raises(ArenaMonitorFailureThresholdError) as exc:
                await monitor._fetch_leaderboard()

        assert monitor.consecutive_failures == 2
        assert exc.value.consecutive_failures == 2
        assert exc.value.max_consecutive_failures == 2
        assert exc.value.urls_attempted == [config.primary_url, *config.fallback_urls]
        await monitor.teardown()

    @pytest.mark.asyncio
    async def test_poll_returns_signals(self) -> None:
        json_data = (FIXTURES / "arena_leaderboard.json").read_text()
        monitor = ArenaMonitor()
        await monitor.initialize()

        mock_response = MagicMock()
        mock_response.text = json_data
        mock_response.raise_for_status = MagicMock()

        with patch.object(monitor._http_client, "get", new_callable=AsyncMock, return_value=mock_response):
            # First poll: no signals (initial snapshot)
            signals1 = await monitor.poll()
            assert signals1 == []
            assert monitor.previous_snapshot is not None

            # Modify data for second poll
            data = json.loads(json_data)
            data[0]["rank_ub"] = 2  # Claude drops from rank_ub 1 to 2
            data[1]["rank_ub"] = 1  # GPT-5 rises to rank_ub 1
            mock_response.text = json.dumps(data)

            # Second poll: should detect changes
            signals2 = await monitor.poll()
            assert len(signals2) > 0

        await monitor.teardown()

    @pytest.mark.asyncio
    async def test_poll_returns_empty_on_fetch_failure(self) -> None:
        monitor = ArenaMonitor()
        await monitor.initialize()

        async def mock_get(url: str) -> MagicMock:
            raise httpx.RequestError("Network error")

        with patch.object(monitor._http_client, "get", side_effect=mock_get):
            signals = await monitor.poll()

        assert signals == []
        await monitor.teardown()


# ── Snapshot Serialization ────────────────────────────────────────────


class TestSnapshotSerialization:
    def test_snapshot_to_dict(self) -> None:
        monitor = ArenaMonitor()
        snap = _snapshot(
            [
                _entry("Claude", org="Anthropic", rank=1, rank_ub=1, score=1350, votes=25000),
            ]
        )
        data = monitor.snapshot_to_dict(snap)
        assert "entries" in data
        assert len(data["entries"]) == 1
        entry = data["entries"][0]
        assert entry["model_name"] == "Claude"
        assert entry["organization"] == "Anthropic"
        assert entry["rank"] == 1
        assert entry["rank_ub"] == 1
        assert entry["score"] == 1350.0
        assert entry["votes"] == 25000
        assert entry["is_preliminary"] is False

    def test_snapshot_to_dict_empty(self) -> None:
        monitor = ArenaMonitor()
        snap = _snapshot([])
        data = monitor.snapshot_to_dict(snap)
        assert data == {"entries": [], "pairwise": {}}


class TestPairwiseSignals:
    def test_generates_pairwise_shift_signal(self) -> None:
        monitor = ArenaMonitor()
        previous = LeaderboardSnapshot(
            entries=[LeaderboardEntry(model_name="A", rank=1, rank_ub=1, score=1500, votes=1000)],
            pairwise={
                "A": PairwiseAggregate(model_name="A", total_pairwise_battles=1000, average_pairwise_win_rate=0.51)
            },
            source_url="x",
        )
        current = LeaderboardSnapshot(
            entries=[LeaderboardEntry(model_name="A", rank=1, rank_ub=1, score=1501, votes=1100)],
            pairwise={
                "A": PairwiseAggregate(model_name="A", total_pairwise_battles=1300, average_pairwise_win_rate=0.54)
            },
            source_url="x",
        )
        monitor.previous_snapshot = previous
        signals = monitor._generate_signals(current)
        assert any(s.event_type == "pairwise_shift" for s in signals)
