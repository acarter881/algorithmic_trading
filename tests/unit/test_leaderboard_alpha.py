"""Unit tests for the Leaderboard Alpha trading strategy."""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING

import pytest

from autotrader.config.models import LeaderboardAlphaConfig
from autotrader.signals.arena_types import LeaderboardEntry
from autotrader.signals.base import Signal, SignalUrgency
from autotrader.strategies.base import OrderUrgency
from autotrader.strategies.leaderboard_alpha import LeaderboardAlphaStrategy, rank_to_probability

if TYPE_CHECKING:
    from autotrader.utils.fees import FeeCalculator


# ── Helpers ──────────────────────────────────────────────────────────────


def _config(**overrides: object) -> LeaderboardAlphaConfig:
    defaults: dict[str, object] = {
        "min_edge_after_fees_cents": 3,
        "elo_shift_threshold": 3,
        "rank_spread_change_threshold": 2,
        "max_position_per_contract": 100,
        "max_position_per_event": 250,
        "preliminary_model_discount": 0.3,
        "fuzzy_match_threshold": 0.8,
        "target_series": ["KXTOPMODEL"],
        "model_name_overrides": {},
    }
    defaults.update(overrides)
    return LeaderboardAlphaConfig(**defaults)  # type: ignore[arg-type]


def _strategy(
    cfg: LeaderboardAlphaConfig | None = None,
    fee_calc: FeeCalculator | None = None,
) -> LeaderboardAlphaStrategy:
    return LeaderboardAlphaStrategy(config=cfg or _config(), fee_calculator=fee_calc)


def _entry(
    name: str = "GPT-5",
    rank_ub: int = 1,
    rank: int = 1,
    score: float = 1350.0,
    votes: int = 10000,
    is_preliminary: bool = False,
    organization: str = "OpenAI",
    release_date: str = "",
) -> LeaderboardEntry:
    return LeaderboardEntry(
        model_name=name,
        organization=organization,
        rank=rank,
        rank_ub=rank_ub,
        score=score,
        votes=votes,
        is_preliminary=is_preliminary,
        release_date=release_date,
    )


def _signal(
    event_type: str,
    data: dict[str, object],
    source: str = "arena_monitor",
    urgency: SignalUrgency = SignalUrgency.MEDIUM,
) -> Signal:
    return Signal(
        source=source,
        timestamp=datetime.datetime(2026, 2, 26, 12, 0, 0),
        event_type=event_type,
        data=data,
        relevant_series=["KXTOPMODEL"],
        urgency=urgency,
    )


@pytest.fixture
def org_market_data() -> dict[str, object]:
    return {
        "markets": [
            {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 55, "yes_ask": 58, "last_price": 56},
            {"ticker": "KXTOPMODEL-GEMINI3", "subtitle": "Gemini 3", "yes_bid": 18, "yes_ask": 22, "last_price": 20},
            {"ticker": "KXLLM1-OPENAI", "subtitle": "OpenAI", "yes_bid": 42, "yes_ask": 46, "last_price": 44},
            {"ticker": "KXLLM1-GOOGLE", "subtitle": "Google", "yes_bid": 30, "yes_ask": 34, "last_price": 32},
        ]
    }


MARKET_DATA = {
    "markets": [
        {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 55, "yes_ask": 58, "last_price": 56},
        {"ticker": "KXTOPMODEL-GEMINI3", "subtitle": "Gemini 3", "yes_bid": 18, "yes_ask": 22, "last_price": 20},
        {"ticker": "KXTOPMODEL-CLAUDE5", "subtitle": "Claude 5", "yes_bid": 8, "yes_ask": 12, "last_price": 10},
        {"ticker": "KXTOPMODEL-LLAMA5", "subtitle": "Llama 5", "yes_bid": 3, "yes_ask": 5, "last_price": 4},
    ]
}


# ── rank_to_probability ─────────────────────────────────────────────────


class TestRankToProbability:
    def test_rank_zero_returns_zero(self) -> None:
        assert rank_to_probability(0) == 0.0

    def test_negative_rank_returns_zero(self) -> None:
        assert rank_to_probability(-1) == 0.0

    def test_rank_1_highest(self) -> None:
        assert rank_to_probability(1) == 0.65

    def test_rank_2(self) -> None:
        assert rank_to_probability(2) == 0.22

    def test_rank_3(self) -> None:
        assert rank_to_probability(3) == 0.10

    def test_rank_4(self) -> None:
        assert rank_to_probability(4) == 0.06

    def test_rank_5(self) -> None:
        assert rank_to_probability(5) == 0.04

    def test_monotonically_decreasing(self) -> None:
        probs = [rank_to_probability(r) for r in range(1, 20)]
        for i in range(len(probs) - 1):
            assert probs[i] >= probs[i + 1]

    def test_high_rank_floor(self) -> None:
        assert rank_to_probability(50) >= 0.01

    def test_rank_6_below_rank_5(self) -> None:
        assert rank_to_probability(6) < rank_to_probability(5)


# ── Strategy Properties ─────────────────────────────────────────────────


class TestStrategyProperties:
    def test_name(self) -> None:
        s = _strategy()
        assert s.name == "leaderboard_alpha"

    def test_enabled_by_default(self) -> None:
        s = _strategy()
        assert s.enabled is True

    def test_target_series(self) -> None:
        s = _strategy(_config(target_series=["KXTOPMODEL", "KXLLM1"]))
        assert s.target_series == ["KXTOPMODEL", "KXLLM1"]


# ── Initialization ──────────────────────────────────────────────────────


class TestInitialize:
    async def test_loads_markets(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        assert len(s.contracts) == 4
        assert "KXTOPMODEL-GPT5" in s.contracts

    async def test_market_prices_set(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        c = s.contracts["KXTOPMODEL-GPT5"]
        assert c.yes_bid == 55
        assert c.yes_ask == 58
        assert c.last_price == 56

    async def test_restores_positions(self) -> None:
        s = _strategy()
        state = {"positions": {"KXTOPMODEL-GPT5": 10}}
        await s.initialize(MARKET_DATA, state)
        assert s.contracts["KXTOPMODEL-GPT5"].position == 10

    async def test_ignores_unknown_position_tickers(self) -> None:
        s = _strategy()
        state = {"positions": {"UNKNOWN-TICKER": 5}}
        await s.initialize(MARKET_DATA, state)
        assert "UNKNOWN-TICKER" not in s.contracts

    async def test_none_market_data(self) -> None:
        s = _strategy()
        await s.initialize(None, None)
        assert len(s.contracts) == 0

    async def test_subtitle_used_for_model_name(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        c = s.contracts["KXTOPMODEL-GPT5"]
        assert c.model_name == "GPT-5"

    async def test_falls_back_to_title(self) -> None:
        s = _strategy()
        data = {"markets": [{"ticker": "T1", "title": "FallbackTitle", "yes_bid": 10, "yes_ask": 12, "last_price": 11}]}
        await s.initialize(data, None)
        assert s.contracts["T1"].model_name == "FallbackTitle"


# ── Fair Value ───────────────────────────────────────────────────────────


class TestFairValue:
    def test_rank1_fair_value(self) -> None:
        s = _strategy()
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})
        fv = s.estimate_fair_value("GPT-5")
        assert fv == 65

    def test_rank2_fair_value(self) -> None:
        s = _strategy()
        s.set_rankings({"GPT-5": _entry(rank_ub=2)})
        fv = s.estimate_fair_value("GPT-5")
        assert fv == 22

    def test_rank3_fair_value(self) -> None:
        s = _strategy()
        s.set_rankings({"GPT-5": _entry(rank_ub=3)})
        fv = s.estimate_fair_value("GPT-5")
        assert fv == 10

    def test_unknown_model_returns_none(self) -> None:
        s = _strategy()
        assert s.estimate_fair_value("UnknownModel") is None

    def test_preliminary_discount(self) -> None:
        s = _strategy(_config(preliminary_model_discount=0.5))
        s.set_rankings({"GPT-5": _entry(rank_ub=1, is_preliminary=True)})
        fv = s.estimate_fair_value("GPT-5")
        # 0.65 * 0.5 * 100 = 32.5 → 32
        assert fv == 32

    def test_fair_value_clamped_min(self) -> None:
        s = _strategy()
        s.set_rankings({"GPT-5": _entry(rank_ub=100)})
        fv = s.estimate_fair_value("GPT-5")
        assert fv is not None
        assert fv >= 1

    def test_fair_value_clamped_max(self) -> None:
        s = _strategy()
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})
        fv = s.estimate_fair_value("GPT-5")
        assert fv is not None
        assert fv <= 99


# ── Signal: Ignore Non-Arena ─────────────────────────────────────────────


class TestIgnoreNonArena:
    async def test_ignores_non_arena_source(self) -> None:
        s = _strategy()
        signal = _signal("ranking_change", {"model_name": "GPT-5"}, source="other_source")
        result = await s.on_signal(signal)
        assert result == []

    async def test_ignores_unknown_event_type(self) -> None:
        s = _strategy()
        signal = _signal("unknown_event", {"model_name": "GPT-5"})
        result = await s.on_signal(signal)
        assert result == []


# ── Signal: ranking_change ───────────────────────────────────────────────


class TestRankingChange:
    async def _setup(self, market_price: int = 22) -> LeaderboardAlphaStrategy:
        """Set up strategy with GPT-5 tracked at rank 2, priced at market_price."""
        s = _strategy()
        data = {
            "markets": [
                {
                    "ticker": "KXTOPMODEL-GPT5",
                    "subtitle": "GPT-5",
                    "yes_bid": market_price - 2,
                    "yes_ask": market_price,
                    "last_price": market_price - 1,
                },
            ]
        }
        await s.initialize(data, None)
        s.set_rankings({"GPT-5": _entry(rank_ub=3, rank=3, score=1300.0)})
        return s

    async def test_rank_improvement_generates_buy(self) -> None:
        # GPT-5 goes from rank_ub=3 to rank_ub=1 → fv=65, market=22 → edge=43 → buy
        s = await self._setup(market_price=22)
        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        orders = await s.on_signal(signal)
        assert len(orders) == 1
        assert orders[0].side == "yes"
        assert orders[0].ticker == "KXTOPMODEL-GPT5"
        assert orders[0].urgency == OrderUrgency.AGGRESSIVE  # rank_ub=1

    async def test_rank_degradation_with_position_generates_sell(self) -> None:
        # GPT-5 goes from rank_ub=1 to rank_ub=5 → fv=4, market=58 → overpriced, sell
        s = _strategy()
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 55, "yes_ask": 58, "last_price": 56},
            ]
        }
        await s.initialize(data, {"positions": {"KXTOPMODEL-GPT5": 10}})
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 1,
                "new_rank_ub": 5,
                "old_rank": 1,
                "new_rank": 5,
                "old_score": 1350.0,
                "new_score": 1280.0,
            },
        )
        orders = await s.on_signal(signal)
        assert len(orders) == 1
        assert orders[0].side == "no"
        assert orders[0].quantity <= 10

    async def test_rank_degradation_without_position_no_order(self) -> None:
        # GPT-5 goes from rank_ub=1 to rank_ub=5 → overpriced, but no position → no sell
        s = _strategy()
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 55, "yes_ask": 58, "last_price": 56},
            ]
        }
        await s.initialize(data, None)
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 1,
                "new_rank_ub": 5,
                "old_rank": 1,
                "new_rank": 5,
                "old_score": 1350.0,
                "new_score": 1280.0,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []

    async def test_no_order_when_edge_insufficient(self) -> None:
        # GPT-5 rank stays at 2 → fv=22, market=21 → edge=1 < min(3) → no order
        s = _strategy()
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 19, "yes_ask": 21, "last_price": 20},
            ]
        }
        await s.initialize(data, None)
        s.set_rankings({"GPT-5": _entry(rank_ub=3)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 2,
                "old_rank": 3,
                "new_rank": 2,
                "old_score": 1300.0,
                "new_score": 1310.0,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []

    async def test_equal_edge_does_not_generate_sell(self) -> None:
        s = _strategy()
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 20, "yes_ask": 22, "last_price": 21},
            ]
        }
        await s.initialize(data, {"positions": {"KXTOPMODEL-GPT5": 5}})
        s.set_rankings({"GPT-5": _entry(rank_ub=2)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 2,
                "new_rank_ub": 2,
                "old_rank": 2,
                "new_rank": 2,
            },
        )

        orders = await s.on_signal(signal)
        assert orders == []

    async def test_unmatched_model_no_order(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        s.set_rankings({"CompletelyUnknownModel": _entry(name="CompletelyUnknownModel", rank_ub=5)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "CompletelyUnknownModel",
                "old_rank_ub": 5,
                "new_rank_ub": 1,
                "old_rank": 5,
                "new_rank": 1,
                "old_score": 1200.0,
                "new_score": 1350.0,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []

    async def test_urgency_aggressive_when_involves_rank1(self) -> None:
        s = await self._setup(market_price=10)
        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        orders = await s.on_signal(signal)
        assert len(orders) == 1
        assert orders[0].urgency == OrderUrgency.AGGRESSIVE

    async def test_urgency_adaptive_when_large_rank_change(self) -> None:
        # rank_ub 5→3, delta=2 >= threshold=2 → adaptive
        s = await self._setup(market_price=5)
        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 5,
                "new_rank_ub": 3,
                "old_rank": 5,
                "new_rank": 3,
                "old_score": 1250.0,
                "new_score": 1300.0,
            },
        )
        orders = await s.on_signal(signal)
        assert len(orders) == 1
        assert orders[0].urgency == OrderUrgency.ADAPTIVE

    async def test_urgency_passive_for_small_rank_change(self) -> None:
        # rank_ub 4→3, delta=1 < threshold=2 → passive
        s = await self._setup(market_price=3)
        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 4,
                "new_rank_ub": 3,
                "old_rank": 4,
                "new_rank": 3,
                "old_score": 1280.0,
                "new_score": 1300.0,
            },
        )
        orders = await s.on_signal(signal)
        assert len(orders) == 1
        assert orders[0].urgency == OrderUrgency.PASSIVE

    async def test_updates_internal_rankings(self) -> None:
        s = await self._setup()
        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        await s.on_signal(signal)
        assert s.rankings["GPT-5"].rank_ub == 1
        assert s.rankings["GPT-5"].score == 1350.0


# ── Signal: new_leader ───────────────────────────────────────────────────


class TestNewLeader:
    async def _setup(self) -> LeaderboardAlphaStrategy:
        s = _strategy()
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 55, "yes_ask": 58, "last_price": 56},
                {
                    "ticker": "KXTOPMODEL-GEMINI3",
                    "subtitle": "Gemini 3",
                    "yes_bid": 18,
                    "yes_ask": 20,
                    "last_price": 19,
                },
            ]
        }
        await s.initialize(data, {"positions": {"KXTOPMODEL-GPT5": 5}})
        s.set_rankings(
            {
                "GPT-5": _entry(name="GPT-5", rank_ub=1, score=1350.0),
                "Gemini 3": _entry(name="Gemini 3", rank_ub=2, score=1330.0, organization="Google"),
            }
        )
        return s

    async def test_buys_new_leader(self) -> None:
        s = await self._setup()
        signal = _signal(
            "new_leader",
            {
                "new_leader": "Gemini 3",
                "previous_leader": "GPT-5",
            },
            urgency=SignalUrgency.HIGH,
        )

        orders = await s.on_signal(signal)
        buy_orders = [o for o in orders if o.side == "yes"]
        assert len(buy_orders) == 1
        assert buy_orders[0].ticker == "KXTOPMODEL-GEMINI3"
        assert buy_orders[0].urgency == OrderUrgency.AGGRESSIVE

    async def test_sells_previous_leader_if_long(self) -> None:
        s = await self._setup()
        signal = _signal(
            "new_leader",
            {
                "new_leader": "Gemini 3",
                "previous_leader": "GPT-5",
            },
        )

        orders = await s.on_signal(signal)
        sell_orders = [o for o in orders if o.side == "no"]
        assert len(sell_orders) == 1
        assert sell_orders[0].ticker == "KXTOPMODEL-GPT5"
        assert sell_orders[0].quantity <= 5

    async def test_no_sell_if_not_long_previous_leader(self) -> None:
        s = await self._setup()
        # Remove position
        s._contracts["KXTOPMODEL-GPT5"].position = 0

        signal = _signal(
            "new_leader",
            {
                "new_leader": "Gemini 3",
                "previous_leader": "GPT-5",
            },
        )

        orders = await s.on_signal(signal)
        sell_orders = [o for o in orders if o.side == "no"]
        assert len(sell_orders) == 0

    async def test_no_buy_when_market_price_too_high(self) -> None:
        s = _strategy()
        data = {
            "markets": [
                {
                    "ticker": "KXTOPMODEL-GEMINI3",
                    "subtitle": "Gemini 3",
                    "yes_bid": 70,
                    "yes_ask": 75,
                    "last_price": 72,
                },
            ]
        }
        await s.initialize(data, None)

        signal = _signal(
            "new_leader",
            {
                "new_leader": "Gemini 3",
                "previous_leader": "GPT-5",
            },
        )

        orders = await s.on_signal(signal)
        buy_orders = [o for o in orders if o.side == "yes"]
        assert len(buy_orders) == 0

    async def test_uses_new_top_org_payload_for_kxllm1(self, org_market_data: dict[str, object]) -> None:
        s = _strategy(_config(target_series=["KXTOPMODEL", "KXLLM1"]))
        await s.initialize(org_market_data, {"positions": {"KXLLM1-OPENAI": 2}})
        s.set_rankings(
            {
                "GPT-5": _entry(name="GPT-5", rank_ub=1, organization="OpenAI"),
                "Gemini 3": _entry(name="Gemini 3", rank_ub=2, organization="Google"),
            }
        )

        signal = _signal(
            "new_leader",
            {
                "new_leader": "Gemini 3",
                "previous_leader": "GPT-5",
                "new_top_org": "Google",
            },
        )

        orders = await s.on_signal(signal)
        assert any(o.ticker == "KXLLM1-GOOGLE" and o.side == "yes" for o in orders)
        assert any(o.ticker == "KXLLM1-OPENAI" and o.side == "no" for o in orders)

    async def test_tracks_new_leader_in_rankings(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        signal = _signal(
            "new_leader",
            {
                "new_leader": "GPT-5",
                "previous_leader": "OldModel",
            },
        )
        await s.on_signal(signal)
        assert "GPT-5" in s.rankings
        assert s.rankings["GPT-5"].rank_ub == 1


# ── Signal: score_shift ──────────────────────────────────────────────────


class TestScoreShift:
    async def _setup(self, market_price: int = 15) -> LeaderboardAlphaStrategy:
        s = _strategy()
        data = {
            "markets": [
                {
                    "ticker": "KXTOPMODEL-GPT5",
                    "subtitle": "GPT-5",
                    "yes_bid": market_price - 2,
                    "yes_ask": market_price,
                    "last_price": market_price - 1,
                },
            ]
        }
        await s.initialize(data, None)
        s.set_rankings({"GPT-5": _entry(rank_ub=2, score=1310.0)})
        return s

    async def test_positive_shift_with_edge_generates_order(self) -> None:
        s = await self._setup(market_price=10)
        signal = _signal(
            "score_shift",
            {
                "model_name": "GPT-5",
                "old_score": 1310.0,
                "new_score": 1320.0,
                "score_delta": 10.0,
            },
        )
        orders = await s.on_signal(signal)
        assert len(orders) == 1
        assert orders[0].side == "yes"
        assert orders[0].urgency == OrderUrgency.ADAPTIVE

    async def test_negative_shift_no_order(self) -> None:
        s = await self._setup()
        signal = _signal(
            "score_shift",
            {
                "model_name": "GPT-5",
                "old_score": 1310.0,
                "new_score": 1300.0,
                "score_delta": -10.0,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []

    async def test_small_shift_ignored(self) -> None:
        # delta=2 < threshold=3
        s = await self._setup(market_price=10)
        signal = _signal(
            "score_shift",
            {
                "model_name": "GPT-5",
                "old_score": 1310.0,
                "new_score": 1312.0,
                "score_delta": 2.0,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []

    async def test_updates_score_in_rankings(self) -> None:
        s = await self._setup()
        signal = _signal(
            "score_shift",
            {
                "model_name": "GPT-5",
                "old_score": 1310.0,
                "new_score": 1320.0,
                "score_delta": 10.0,
            },
        )
        await s.on_signal(signal)
        assert s.rankings["GPT-5"].score == 1320.0


# ── Signal: new_model ────────────────────────────────────────────────────


class TestNewModel:
    async def test_competitive_new_model_generates_order(self) -> None:
        s = _strategy()
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 3, "yes_ask": 5, "last_price": 4},
            ]
        }
        await s.initialize(data, None)

        signal = _signal(
            "new_model",
            {
                "model_name": "GPT-5",
                "organization": "OpenAI",
                "rank": 2,
                "rank_ub": 2,
                "score": 1340.0,
                "votes": 10000,
                "is_preliminary": False,
            },
        )
        orders = await s.on_signal(signal)
        assert len(orders) == 1
        assert orders[0].side == "yes"

    async def test_non_competitive_model_ignored(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)

        signal = _signal(
            "new_model",
            {
                "model_name": "GPT-5",
                "rank": 10,
                "rank_ub": 10,
                "score": 1200.0,
                "votes": 500,
                "is_preliminary": False,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []

    async def test_zero_rank_ub_ignored(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)

        signal = _signal(
            "new_model",
            {
                "model_name": "GPT-5",
                "rank_ub": 0,
                "score": 1300.0,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []

    async def test_adds_to_rankings(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)

        signal = _signal(
            "new_model",
            {
                "model_name": "GPT-5",
                "organization": "OpenAI",
                "rank": 1,
                "rank_ub": 1,
                "score": 1360.0,
                "votes": 15000,
                "is_preliminary": False,
            },
        )
        await s.on_signal(signal)
        assert "GPT-5" in s.rankings
        assert s.rankings["GPT-5"].organization == "OpenAI"
        assert s.rankings["GPT-5"].votes == 15000

    async def test_preliminary_model_at_rank1(self) -> None:
        s = _strategy(_config(preliminary_model_discount=0.5))
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-NEWMODEL", "subtitle": "NewModel", "yes_bid": 3, "yes_ask": 5, "last_price": 4},
            ]
        }
        await s.initialize(data, None)

        signal = _signal(
            "new_model",
            {
                "model_name": "NewModel",
                "rank": 1,
                "rank_ub": 1,
                "score": 1380.0,
                "votes": 200,
                "is_preliminary": True,
            },
        )
        orders = await s.on_signal(signal)
        # fv = round(0.65 * 0.5 * 100) = 32, market = 5, edge = 27 → should trade
        assert len(orders) == 1
        assert s.rankings["NewModel"].is_preliminary is True


# ── Market Updates ───────────────────────────────────────────────────────


class TestMarketUpdate:
    async def test_updates_prices(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)

        await s.on_market_update({"ticker": "KXTOPMODEL-GPT5", "yes_bid": 60, "yes_ask": 63, "last_price": 61})

        c = s.contracts["KXTOPMODEL-GPT5"]
        assert c.yes_bid == 60
        assert c.yes_ask == 63
        assert c.last_price == 61

    async def test_partial_update(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)

        await s.on_market_update({"ticker": "KXTOPMODEL-GPT5", "yes_bid": 60})

        c = s.contracts["KXTOPMODEL-GPT5"]
        assert c.yes_bid == 60
        assert c.yes_ask == 58  # unchanged

    async def test_unknown_ticker_ignored(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        result = await s.on_market_update({"ticker": "UNKNOWN", "yes_bid": 99})
        assert result == []

    async def test_non_dict_ignored(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        result = await s.on_market_update("not a dict")
        assert result == []

    async def test_returns_empty_list(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        result = await s.on_market_update({"ticker": "KXTOPMODEL-GPT5", "yes_bid": 60})
        assert result == []


# ── Fills / Position Tracking ────────────────────────────────────────────


class TestFillTracking:
    async def test_buy_yes_increases_position(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        await s.on_fill({"ticker": "KXTOPMODEL-GPT5", "side": "yes", "action": "buy", "count": 5})
        assert s.contracts["KXTOPMODEL-GPT5"].position == 5

    async def test_buy_no_decreases_position(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        await s.on_fill({"ticker": "KXTOPMODEL-GPT5", "side": "no", "action": "buy", "count": 3})
        assert s.contracts["KXTOPMODEL-GPT5"].position == -3

    async def test_sell_yes_decreases_position(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, {"positions": {"KXTOPMODEL-GPT5": 10}})
        await s.on_fill({"ticker": "KXTOPMODEL-GPT5", "side": "yes", "action": "sell", "count": 4})
        assert s.contracts["KXTOPMODEL-GPT5"].position == 6

    async def test_sell_no_increases_position(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, {"positions": {"KXTOPMODEL-GPT5": -5}})
        await s.on_fill({"ticker": "KXTOPMODEL-GPT5", "side": "no", "action": "sell", "count": 3})
        assert s.contracts["KXTOPMODEL-GPT5"].position == -2

    async def test_unknown_ticker_ignored(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        await s.on_fill({"ticker": "UNKNOWN", "side": "yes", "action": "buy", "count": 5})
        # No crash, no effect

    async def test_non_dict_ignored(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        await s.on_fill("not a dict")
        # No crash


# ── Position Limits ──────────────────────────────────────────────────────


class TestPositionLimits:
    async def test_respects_max_position(self) -> None:
        s = _strategy(_config(max_position_per_contract=5))
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 3, "yes_ask": 5, "last_price": 4},
            ]
        }
        await s.initialize(data, {"positions": {"KXTOPMODEL-GPT5": 3}})
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        orders = await s.on_signal(signal)
        assert len(orders) == 1
        assert orders[0].quantity <= 2  # max 5 - current 3

    async def test_no_order_at_max_position(self) -> None:
        s = _strategy(_config(max_position_per_contract=5))
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 3, "yes_ask": 5, "last_price": 4},
            ]
        }
        await s.initialize(data, {"positions": {"KXTOPMODEL-GPT5": 5}})
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []


# ── Ticker Resolution ────────────────────────────────────────────────────


class TestTickerResolution:
    async def test_exact_match(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        await s.on_signal(signal)
        assert s.model_ticker_map.get("GPT-5") == "KXTOPMODEL-GPT5"

    async def test_fuzzy_match(self) -> None:
        s = _strategy(_config(fuzzy_match_threshold=0.7))
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 3, "yes_ask": 5, "last_price": 4},
            ]
        }
        await s.initialize(data, None)
        s.set_rankings({"gpt-5": _entry(name="gpt-5", rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "gpt-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        orders = await s.on_signal(signal)
        assert len(orders) == 1

    async def test_override_match(self) -> None:
        s = _strategy(_config(model_name_overrides={"arena-gpt5": "GPT-5"}))
        await s.initialize(MARKET_DATA, None)
        s.set_rankings({"arena-gpt5": _entry(name="arena-gpt5", rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "arena-gpt5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        await s.on_signal(signal)
        assert s.model_ticker_map.get("arena-gpt5") == "KXTOPMODEL-GPT5"

    async def test_cached_resolution(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        await s.on_signal(signal)
        # Second call should use cache
        await s.on_signal(signal)
        assert s.model_ticker_map.get("GPT-5") == "KXTOPMODEL-GPT5"

    async def test_empty_model_name(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        signal = _signal(
            "ranking_change",
            {
                "model_name": "",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []


# ── State Serialization ──────────────────────────────────────────────────


class TestGetState:
    async def test_serializes_rankings(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        s.set_rankings({"GPT-5": _entry(rank_ub=1, score=1350.0, organization="OpenAI")})

        state = s.get_state()
        assert "GPT-5" in state["rankings"]
        assert state["rankings"]["GPT-5"]["rank_ub"] == 1
        assert state["rankings"]["GPT-5"]["organization"] == "OpenAI"

    async def test_serializes_positions(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, {"positions": {"KXTOPMODEL-GPT5": 10}})

        state = s.get_state()
        assert state["positions"]["KXTOPMODEL-GPT5"] == 10

    async def test_excludes_zero_positions(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)

        state = s.get_state()
        assert len(state["positions"]) == 0

    async def test_serializes_ticker_map(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        await s.on_signal(signal)

        state = s.get_state()
        assert state["model_ticker_map"]["GPT-5"] == "KXTOPMODEL-GPT5"


# ── Edge Cases ───────────────────────────────────────────────────────────


class TestEdgeCases:
    async def test_market_price_at_100_no_order(self) -> None:
        s = _strategy()
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 98, "yes_ask": 100, "last_price": 99},
            ]
        }
        await s.initialize(data, None)
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []

    async def test_market_price_at_0_no_order(self) -> None:
        s = _strategy()
        data = {
            "markets": [
                {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 0, "yes_ask": 0, "last_price": 0},
            ]
        }
        await s.initialize(data, None)
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []

    async def test_teardown(self) -> None:
        s = _strategy()
        await s.teardown()  # Should not raise

    async def test_no_contracts_loaded(self) -> None:
        s = _strategy()
        await s.initialize(None, None)
        s.set_rankings({"GPT-5": _entry(rank_ub=1)})

        signal = _signal(
            "ranking_change",
            {
                "model_name": "GPT-5",
                "old_rank_ub": 3,
                "new_rank_ub": 1,
                "old_rank": 3,
                "new_rank": 1,
                "old_score": 1300.0,
                "new_score": 1350.0,
            },
        )
        orders = await s.on_signal(signal)
        assert orders == []

    async def test_model_removed_signal_ignored(self) -> None:
        s = _strategy()
        await s.initialize(MARKET_DATA, None)
        signal = _signal("model_removed", {"model_name": "GPT-5", "last_rank_ub": 3})
        orders = await s.on_signal(signal)
        assert orders == []


class TestPairwiseShift:
    async def test_pairwise_shift_can_generate_buy(self) -> None:
        s = _strategy()
        await s.initialize(
            {
                "markets": [
                    {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 40, "yes_ask": 42, "last_price": 41},
                ]
            },
            None,
        )
        s.set_rankings({"GPT-5": _entry(rank_ub=1, rank=1, score=1500)})
        signal = _signal(
            "pairwise_shift",
            {
                "model_name": "GPT-5",
                "new_average_pairwise_win_rate": 0.58,
                "new_total_pairwise_battles": 5000,
            },
        )
        orders = await s.on_signal(signal)
        assert len(orders) == 1
        assert orders[0].side == "yes"


class TestSettlementBonusGuard:
    def test_no_winner_bonus_with_incomplete_tiebreak_inputs(self) -> None:
        s = _strategy()
        s.set_rankings(
            {
                "A": _entry(rank_ub=1, score=1500, votes=0),
                "B": _entry(rank_ub=1, score=1500, votes=0),
            }
        )
        fv = s.estimate_fair_value("A")
        assert fv == 65

    def test_complete_tiebreak_inputs_when_votes_and_release_dates_present(self) -> None:
        s = _strategy()
        s.set_rankings(
            {
                "A": _entry(name="A", rank_ub=1, score=1500, votes=1000, release_date="2025-01-01"),
                "B": _entry(name="B", rank_ub=1, score=1500, votes=1200, release_date="2025-02-01"),
            }
        )
        assert s._has_complete_tiebreak_inputs() is True


class TestOrgContracts:
    async def test_kxllm1_mapping_uses_organization_names(self, org_market_data: dict[str, object]) -> None:
        s = _strategy(_config(target_series=["KXTOPMODEL", "KXLLM1"]))
        await s.initialize(org_market_data, None)
        s.set_rankings(
            {
                "GPT-5": _entry(name="GPT-5", rank_ub=1, organization="OpenAI"),
                "Gemini 3": _entry(name="Gemini 3", rank_ub=2, organization="Google"),
            }
        )

        orders = await s.on_signal(
            _signal(
                "ranking_change",
                {
                    "model_name": "GPT-5",
                    "old_rank_ub": 2,
                    "new_rank_ub": 1,
                    "old_rank": 2,
                    "new_rank": 1,
                },
            )
        )

        assert s.model_ticker_map["GPT-5"] == "KXTOPMODEL-GPT5"
        assert s._org_ticker_map["OpenAI"] == "KXLLM1-OPENAI"
        assert {o.ticker for o in orders} >= {"KXTOPMODEL-GPT5", "KXLLM1-OPENAI"}

    def test_org_fair_value_uses_top_org_resolution(self) -> None:
        s = _strategy()
        s.set_rankings(
            {
                "OpenAI A": _entry(name="OpenAI A", rank_ub=1, score=1400, votes=10000, organization="OpenAI"),
                "OpenAI B": _entry(name="OpenAI B", rank_ub=2, score=1390, votes=9000, organization="OpenAI"),
                "Gemini 3": _entry(name="Gemini 3", rank_ub=1, score=1399, votes=10000, organization="Google"),
            }
        )

        # OpenAI wins tiebreak via score and receives winner bonus.
        assert s.estimate_org_fair_value("OpenAI") == 70
        assert s.estimate_org_fair_value("Google") == 65

    async def test_org_leader_change_buy_sell_logic(self, org_market_data: dict[str, object]) -> None:
        s = _strategy(_config(target_series=["KXTOPMODEL", "KXLLM1"]))
        await s.initialize(
            org_market_data,
            {"positions": {"KXTOPMODEL-GPT5": 4, "KXLLM1-OPENAI": 3}},
        )
        s.set_rankings(
            {
                "GPT-5": _entry(name="GPT-5", rank_ub=1, organization="OpenAI"),
                "Gemini 3": _entry(name="Gemini 3", rank_ub=2, organization="Google"),
            }
        )

        orders = await s.on_signal(
            _signal(
                "new_leader",
                {
                    "new_leader": "Gemini 3",
                    "previous_leader": "GPT-5",
                },
            )
        )

        assert any(o.ticker == "KXLLM1-GOOGLE" and o.side == "yes" for o in orders)
        assert any(o.ticker == "KXLLM1-OPENAI" and o.side == "no" for o in orders)


class TestOrgStateMaintenance:
    async def test_ranking_change_updates_organization_from_signal(self) -> None:
        s = _strategy(_config(target_series=["KXTOPMODEL", "KXLLM1"]))
        await s.initialize(
            {
                "markets": [
                    {"ticker": "KXTOPMODEL-GPT5", "subtitle": "GPT-5", "yes_bid": 10, "yes_ask": 12, "last_price": 11},
                    {"ticker": "KXLLM1-OPENAI", "subtitle": "OpenAI", "yes_bid": 10, "yes_ask": 12, "last_price": 11},
                ]
            },
            None,
        )

        await s.on_signal(
            _signal(
                "ranking_change",
                {
                    "model_name": "GPT-5",
                    "organization": "OpenAI",
                    "old_rank_ub": 2,
                    "new_rank_ub": 1,
                    "old_rank": 2,
                    "new_rank": 1,
                },
            )
        )

        assert s.rankings["GPT-5"].organization == "OpenAI"
        assert s.estimate_org_fair_value("OpenAI") is not None

    async def test_initialize_restores_org_ticker_map_state(self, org_market_data: dict[str, object]) -> None:
        s = _strategy()
        await s.initialize(org_market_data, {"org_ticker_map": {"OpenAI": "KXLLM1-OPENAI"}})

        assert s._org_ticker_map["OpenAI"] == "KXLLM1-OPENAI"
