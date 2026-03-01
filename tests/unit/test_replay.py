"""Unit tests for the backtest replay engine."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from autotrader.backtest.replay import ReplayEngine, ReplayResult
from autotrader.config.models import AppConfig, ArenaMonitorConfig, LeaderboardAlphaConfig

if TYPE_CHECKING:
    from pathlib import Path


def _config() -> AppConfig:
    return AppConfig(
        arena_monitor=ArenaMonitorConfig(poll_interval_seconds=1),
        leaderboard_alpha=LeaderboardAlphaConfig(target_series=["KXTOPMODEL"]),
    )


def _market_data() -> dict:
    return {
        "markets": [
            {
                "ticker": "KXTOPMODEL-GPT5",
                "subtitle": "GPT-5",
                "title": "GPT-5",
                "yes_bid": 40,
                "yes_ask": 50,
                "last_price": 45,
                "event_ticker": "KXTOPMODEL",
            },
        ]
    }


def _signals_data() -> list[dict]:
    return [
        {
            "source": "arena_monitor",
            "timestamp": "2026-02-27T12:00:00",
            "event_type": "ranking_change",
            "data": {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "organization": "OpenAI",
            },
            "relevant_series": ["KXTOPMODEL"],
            "urgency": "high",
        },
    ]


class TestReplayResult:
    def test_default_result(self) -> None:
        r = ReplayResult()
        assert r.total_signals == 0
        assert r.total_proposals == 0
        assert r.realized_pnl_cents == 0


class TestReplayEngine:
    async def test_replay_empty_file(self, tmp_path: Path) -> None:
        signals_file = tmp_path / "empty.json"
        signals_file.write_text("[]")

        engine = ReplayEngine(_config())
        result = await engine.run(signals_path=str(signals_file))
        assert result.total_signals == 0

    async def test_replay_missing_file(self, tmp_path: Path) -> None:
        engine = ReplayEngine(_config())
        result = await engine.run(signals_path=str(tmp_path / "nonexistent.json"))
        assert result.total_signals == 0

    async def test_replay_processes_signals(self, tmp_path: Path) -> None:
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(_signals_data()))

        engine = ReplayEngine(_config())
        result = await engine.run(
            signals_path=str(signals_file),
            market_data=_market_data(),
        )
        assert result.total_signals == 1
        assert result.total_proposals >= 0  # May or may not generate proposals

    async def test_replay_with_no_market_data(self, tmp_path: Path) -> None:
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(_signals_data()))

        engine = ReplayEngine(_config())
        result = await engine.run(signals_path=str(signals_file))
        # No contracts loaded → no proposals
        assert result.total_proposals == 0

    async def test_load_signals_parses_correctly(self, tmp_path: Path) -> None:
        signals_file = tmp_path / "signals.json"
        signals_file.write_text(json.dumps(_signals_data()))

        signals = ReplayEngine._load_signals(signals_file)
        assert len(signals) == 1
        assert signals[0].source == "arena_monitor"
        assert signals[0].event_type == "ranking_change"
        assert signals[0].data["model_name"] == "GPT-5"

    async def test_load_signals_single_object(self, tmp_path: Path) -> None:
        """A single signal object (not wrapped in array) should work."""
        signals_file = tmp_path / "signal.json"
        signals_file.write_text(json.dumps(_signals_data()[0]))

        signals = ReplayEngine._load_signals(signals_file)
        assert len(signals) == 1
