"""Leaderboard Alpha trading strategy.

Reacts to LMSYS Chatbot Arena leaderboard changes to trade Kalshi
prediction market contracts (KXTOPMODEL, KXLLM1).

Edge source: faster reaction to rank/score changes than manual traders,
combined with quantitative fair-value estimation and fee-aware filtering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from autotrader.signals.arena_types import LeaderboardEntry, PairwiseAggregate
from autotrader.signals.settlement import resolve_top_model, resolve_top_org
from autotrader.strategies.base import OrderUrgency, ProposedOrder, Strategy
from autotrader.utils.fees import FeeCalculator
from autotrader.utils.matching import fuzzy_match

if TYPE_CHECKING:
    from autotrader.config.models import LeaderboardAlphaConfig
    from autotrader.signals.base import Signal

logger = structlog.get_logger("autotrader.strategies.leaderboard_alpha")


@dataclass
class ContractView:
    """Strategy's internal view of a tradeable contract."""

    ticker: str
    model_name: str  # Name extracted from Kalshi contract title/subtitle
    series: str = ""
    yes_bid: int = 0
    yes_ask: int = 0
    last_price: int = 0
    position: int = 0  # Positive = long YES


# Rank(UB) → P(model is #1) lookup.  Based on typical Arena volatility:
# rank_ub=1 already holds #1 but could slip; rank_ub=2 is close contender, etc.
_RANK_PROB: dict[int, float] = {1: 0.65, 2: 0.22, 3: 0.10, 4: 0.06, 5: 0.04}


def rank_to_probability(rank_ub: int) -> float:
    """Estimate P(model finishes #1) from its current rank_ub."""
    if rank_ub <= 0:
        return 0.0
    if rank_ub in _RANK_PROB:
        return _RANK_PROB[rank_ub]
    # Exponential decay for rank_ub > 5
    return max(0.01, 0.04 * math.exp(-0.3 * (rank_ub - 5)))


class LeaderboardAlphaStrategy(Strategy):
    """Trades Kalshi AI leaderboard contracts based on Arena ranking changes.

    Signal types handled:
    - ``ranking_change`` — model rank_ub moved → recalculate fair value
    - ``new_leader``     — #1 model changed → aggressive buy/sell
    - ``score_shift``    — Elo score shifted → adaptive order
    - ``new_model``      — new model appeared → evaluate if competitive
    """

    def __init__(
        self,
        config: LeaderboardAlphaConfig,
        fee_calculator: FeeCalculator | None = None,
    ) -> None:
        self._config = config
        self._fee_calc = fee_calculator or FeeCalculator()
        self._enabled = True

        # Arena model name → LeaderboardEntry
        self._rankings: dict[str, LeaderboardEntry] = {}
        # Arena model name → Kalshi ticker
        self._model_ticker_map: dict[str, str] = {}
        # Kalshi ticker → ContractView
        self._contracts: dict[str, ContractView] = {}
        # Kalshi ticker → model name (from contract title/subtitle)
        self._ticker_model_names: dict[str, str] = {}
        # Arena model name → pairwise aggregate metrics
        self._pairwise: dict[str, PairwiseAggregate] = {}
        # Arena organization → best leaderboard entry for that org
        self._org_rankings: dict[str, LeaderboardEntry] = {}
        # Arena organization → Kalshi ticker
        self._org_ticker_map: dict[str, str] = {}

    # ── Strategy interface ────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "leaderboard_alpha"

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def target_series(self) -> list[str]:
        return self._config.target_series

    async def initialize(self, market_data: Any, state: Any) -> None:
        """Initialize with market data and optionally persisted state.

        ``market_data`` expected shape::

            {"markets": [{"ticker": str, "subtitle": str, "title": str,
                          "yes_bid": int, "yes_ask": int, "last_price": int}, ...]}

        ``state`` expected shape (optional)::

            {"positions": {"TICKER": int, ...}}
        """
        if isinstance(market_data, dict):
            for m in market_data.get("markets", []):
                ticker = m.get("ticker", "")
                model_name = m.get("subtitle", m.get("title", ""))
                self._ticker_model_names[ticker] = model_name
                self._contracts[ticker] = ContractView(
                    ticker=ticker,
                    model_name=model_name,
                    series=self._ticker_series(ticker),
                    yes_bid=m.get("yes_bid", 0),
                    yes_ask=m.get("yes_ask", 0),
                    last_price=m.get("last_price", 0),
                )

        if isinstance(state, dict):
            for ticker, pos in state.get("positions", {}).items():
                if ticker in self._contracts:
                    self._contracts[ticker].position = pos
            for model_name, pair in state.get("pairwise", {}).items():
                if isinstance(pair, dict):
                    self._pairwise[model_name] = PairwiseAggregate(
                        model_name=model_name,
                        total_pairwise_battles=int(pair.get("total_pairwise_battles", 0)),
                        average_pairwise_win_rate=float(pair.get("average_pairwise_win_rate", 0.0)),
                    )
        self._rebuild_org_rankings()

        logger.info("strategy_initialized", strategy=self.name, contracts=len(self._contracts))

    async def on_signal(self, signal: Signal) -> list[ProposedOrder]:
        """Process an arena signal and generate trade proposals."""
        if signal.source != "arena_monitor":
            return []

        handler = {
            "ranking_change": self._handle_ranking_change,
            "new_leader": self._handle_new_leader,
            "score_shift": self._handle_score_shift,
            "new_model": self._handle_new_model,
            "pairwise_shift": self._handle_pairwise_shift,
        }.get(signal.event_type)

        if handler is None:
            return []

        proposals = handler(signal)
        if proposals:
            logger.info(
                "proposals_generated",
                strategy=self.name,
                signal_type=signal.event_type,
                count=len(proposals),
                tickers=[p.ticker for p in proposals],
            )
        return proposals

    async def on_market_update(self, data: Any) -> list[ProposedOrder]:
        """Update market prices.  This strategy is signal-driven, so no
        orders are generated from market updates alone."""
        if isinstance(data, dict) and data.get("ticker") in self._contracts:
            c = self._contracts[data["ticker"]]
            for key in ("yes_bid", "yes_ask", "last_price"):
                if key in data:
                    setattr(c, key, data[key])
        return []

    async def on_fill(self, fill_data: Any) -> None:
        """Update position tracking after a fill."""
        if not isinstance(fill_data, dict):
            return
        ticker = fill_data.get("ticker", "")
        if ticker not in self._contracts:
            return
        count = fill_data.get("count", 0)
        side = fill_data.get("side", "")
        action = fill_data.get("action", "buy")
        # Positive delta = adding YES exposure
        delta = count if side == "yes" else -count
        if action == "sell":
            delta = -delta
        self._contracts[ticker].position += delta

    def get_state(self) -> dict[str, Any]:
        """Return serializable state for persistence."""
        return {
            "rankings": {
                name: {
                    "model_name": e.model_name,
                    "organization": e.organization,
                    "rank": e.rank,
                    "rank_ub": e.rank_ub,
                    "score": e.score,
                    "votes": e.votes,
                    "is_preliminary": e.is_preliminary,
                }
                for name, e in self._rankings.items()
            },
            "positions": {t: c.position for t, c in self._contracts.items() if c.position != 0},
            "model_ticker_map": dict(self._model_ticker_map),
            "org_ticker_map": dict(self._org_ticker_map),
            "pairwise": {
                m: {
                    "total_pairwise_battles": p.total_pairwise_battles,
                    "average_pairwise_win_rate": p.average_pairwise_win_rate,
                }
                for m, p in self._pairwise.items()
            },
        }

    async def teardown(self) -> None:
        logger.info("strategy_teardown", strategy=self.name)

    # ── Fair value ────────────────────────────────────────────────────

    def estimate_fair_value(self, model_name: str) -> int | None:
        """Estimate fair value (cents) for a model's KXTOPMODEL contract.

        Returns ``None`` if the model is not tracked in rankings.
        """
        entry = self._rankings.get(model_name)
        if entry is None:
            return None
        prob = rank_to_probability(entry.rank_ub)
        if self._has_complete_tiebreak_inputs():
            winner = resolve_top_model(list(self._rankings.values()))
            if winner and winner.model_name == model_name:
                prob = min(0.95, prob + 0.05)
        pairwise = self._pairwise.get(model_name)
        if pairwise and pairwise.total_pairwise_battles > 0:
            pairwise_edge = pairwise.average_pairwise_win_rate - 0.5
            prob = max(0.0, min(1.0, prob + 0.15 * pairwise_edge))
        if entry.is_preliminary:
            prob *= 1.0 - self._config.preliminary_model_discount
        return max(1, min(99, round(prob * 100)))

    def estimate_org_fair_value(self, organization: str) -> int | None:
        """Estimate fair value (cents) for an organization's KXLLM1 contract."""
        entry = self._org_rankings.get(organization)
        if entry is None:
            return None
        prob = rank_to_probability(entry.rank_ub)
        if self._has_complete_tiebreak_inputs():
            winner = resolve_top_org(list(self._rankings.values()))
            if winner and winner == organization:
                prob = min(0.95, prob + 0.05)
        if entry.is_preliminary:
            prob *= 1.0 - self._config.preliminary_model_discount
        return max(1, min(99, round(prob * 100)))

    # ── Signal handlers ───────────────────────────────────────────────

    def _handle_ranking_change(self, signal: Signal) -> list[ProposedOrder]:
        data = signal.data
        model_name = data.get("model_name", "")
        old_rank_ub = data.get("old_rank_ub", 0)
        new_rank_ub = data.get("new_rank_ub", 0)

        self._update_ranking(model_name, data)

        model_ticker = self._resolve_ticker(model_name, series="KXTOPMODEL")
        org_name = self._rankings.get(model_name, LeaderboardEntry(model_name=model_name)).organization
        org_ticker = self._resolve_ticker(org_name, series="KXLLM1") if org_name else None

        targets: list[tuple[str, int]] = []
        if model_ticker:
            fair_value = self.estimate_fair_value(model_name)
            if fair_value is not None:
                targets.append((model_ticker, fair_value))
        if org_ticker:
            org_fair_value = self.estimate_org_fair_value(org_name)
            if org_fair_value is not None:
                targets.append((org_ticker, org_fair_value))
        if not targets:
            return []

        rank_delta = abs(old_rank_ub - new_rank_ub)
        urgency = (
            OrderUrgency.AGGRESSIVE
            if new_rank_ub == 1 or old_rank_ub == 1
            else OrderUrgency.ADAPTIVE
            if rank_delta >= self._config.rank_spread_change_threshold
            else OrderUrgency.PASSIVE
        )

        orders: list[ProposedOrder] = []
        for ticker, fair_value in targets:
            contract = self._contracts.get(ticker)
            if contract is None:
                continue
            market_price = contract.yes_ask or contract.last_price
            if market_price <= 0 or market_price >= 100:
                continue
            edge = fair_value - market_price
            if edge > 0:
                orders.extend(
                    self._propose_buy(
                        ticker,
                        contract,
                        fair_value,
                        market_price,
                        urgency,
                        f"rank_ub {old_rank_ub}->{new_rank_ub}, fv={fair_value}c mkt={market_price}c",
                    )
                )
            elif contract.position > 0:
                sell_price = contract.yes_bid or contract.last_price
                if 0 < sell_price < 100:
                    sell_qty = min(contract.position, int(self._config.max_position_per_contract))
                    orders.append(
                        ProposedOrder(
                            strategy=self.name,
                            ticker=ticker,
                            side="no",
                            price_cents=100 - sell_price,
                            quantity=sell_qty,
                            urgency=urgency,
                            rationale=(f"rank_ub {old_rank_ub}->{new_rank_ub}, overpriced by {-edge}c, reducing position"),
                        )
                    )
        return orders

    def _handle_new_leader(self, signal: Signal) -> list[ProposedOrder]:
        data = signal.data
        new_leader = data.get("new_leader", "")
        previous_leader = data.get("previous_leader", "")
        orders: list[ProposedOrder] = []

        # Ensure we track the new leader at rank_ub=1
        if new_leader and new_leader not in self._rankings:
            self._rankings[new_leader] = LeaderboardEntry(model_name=new_leader, rank_ub=1)

        # Buy new leader model on KXTOPMODEL.
        new_ticker = self._resolve_ticker(new_leader, series="KXTOPMODEL")
        if new_ticker:
            contract = self._contracts.get(new_ticker)
            if contract:
                fair_value = max(1, min(99, round(rank_to_probability(1) * 100)))
                market_price = contract.yes_ask or contract.last_price
                if 0 < market_price < 100:
                    orders.extend(
                        self._propose_buy(
                            new_ticker,
                            contract,
                            fair_value,
                            market_price,
                            OrderUrgency.AGGRESSIVE,
                            f"new #1: {new_leader}, fv={fair_value}c mkt={market_price}c",
                        )
                    )

        # Sell previous leader if we hold YES
        prev_ticker = self._resolve_ticker(previous_leader, series="KXTOPMODEL")
        if prev_ticker:
            prev_contract = self._contracts.get(prev_ticker)
            if prev_contract and prev_contract.position > 0:
                sell_price = prev_contract.yes_bid or prev_contract.last_price
                if 0 < sell_price < 100:
                    sell_qty = min(prev_contract.position, int(self._config.max_position_per_contract))
                    orders.append(
                        ProposedOrder(
                            strategy=self.name,
                            ticker=prev_ticker,
                            side="no",
                            price_cents=100 - sell_price,
                            quantity=sell_qty,
                            urgency=OrderUrgency.AGGRESSIVE,
                            rationale=f"lost #1 to {new_leader}, reducing position",
                        )
                    )
        # Also trade organization leader on KXLLM1 contracts.
        new_org = data.get("new_organization") or self._rankings.get(new_leader, LeaderboardEntry(model_name="")).organization
        prev_org = data.get("previous_organization") or self._rankings.get(previous_leader, LeaderboardEntry(model_name="")).organization

        if isinstance(new_org, str) and new_org:
            org_ticker = self._resolve_ticker(new_org, series="KXLLM1")
            org_contract = self._contracts.get(org_ticker) if org_ticker else None
            org_fv = self.estimate_org_fair_value(new_org)
            if org_ticker and org_contract and org_fv is not None:
                market_price = org_contract.yes_ask or org_contract.last_price
                if 0 < market_price < 100:
                    orders.extend(
                        self._propose_buy(
                            org_ticker,
                            org_contract,
                            org_fv,
                            market_price,
                            OrderUrgency.AGGRESSIVE,
                            f"new #1 org: {new_org}, fv={org_fv}c mkt={market_price}c",
                        )
                    )
        if isinstance(prev_org, str) and prev_org:
            prev_org_ticker = self._resolve_ticker(prev_org, series="KXLLM1")
            prev_org_contract = self._contracts.get(prev_org_ticker) if prev_org_ticker else None
            if prev_org_ticker and prev_org_contract and prev_org_contract.position > 0:
                sell_price = prev_org_contract.yes_bid or prev_org_contract.last_price
                if 0 < sell_price < 100:
                    sell_qty = min(prev_org_contract.position, int(self._config.max_position_per_contract))
                    orders.append(
                        ProposedOrder(
                            strategy=self.name,
                            ticker=prev_org_ticker,
                            side="no",
                            price_cents=100 - sell_price,
                            quantity=sell_qty,
                            urgency=OrderUrgency.AGGRESSIVE,
                            rationale=f"org lost #1 to {new_org}, reducing position",
                        )
                    )
        return orders

    def _handle_score_shift(self, signal: Signal) -> list[ProposedOrder]:
        data = signal.data
        model_name = data.get("model_name", "")
        score_delta = data.get("score_delta", 0.0)

        self._update_ranking(model_name, data)

        if abs(score_delta) < self._config.elo_shift_threshold:
            return []
        # Only buy on positive momentum
        if score_delta <= 0:
            return []

        orders: list[ProposedOrder] = []
        model_ticker = self._resolve_ticker(model_name, series="KXTOPMODEL")
        if model_ticker:
            fair_value = self.estimate_fair_value(model_name)
            contract = self._contracts.get(model_ticker)
            if fair_value is not None and contract is not None:
                market_price = contract.yes_ask or contract.last_price
                if 0 < market_price < 100:
                    orders.extend(
                        self._propose_buy(
                            model_ticker,
                            contract,
                            fair_value,
                            market_price,
                            OrderUrgency.ADAPTIVE,
                            f"score +{score_delta:.1f}, fv={fair_value}c mkt={market_price}c",
                        )
                    )

        org_name = self._rankings.get(model_name, LeaderboardEntry(model_name=model_name)).organization
        org_ticker = self._resolve_ticker(org_name, series="KXLLM1") if org_name else None
        if org_ticker:
            org_fv = self.estimate_org_fair_value(org_name)
            org_contract = self._contracts.get(org_ticker)
            if org_fv is not None and org_contract is not None:
                market_price = org_contract.yes_ask or org_contract.last_price
                if 0 < market_price < 100:
                    orders.extend(
                        self._propose_buy(
                            org_ticker,
                            org_contract,
                            org_fv,
                            market_price,
                            OrderUrgency.ADAPTIVE,
                            f"score +{score_delta:.1f} for {org_name}, fv={org_fv}c mkt={market_price}c",
                        )
                    )
        return orders

    def _handle_new_model(self, signal: Signal) -> list[ProposedOrder]:
        data = signal.data
        model_name = data.get("model_name", "")
        rank_ub = data.get("rank_ub", 0)

        self._rankings[model_name] = LeaderboardEntry(
            model_name=model_name,
            organization=data.get("organization", ""),
            rank=data.get("rank", 0),
            rank_ub=rank_ub,
            score=data.get("score", 0.0),
            votes=data.get("votes", 0),
            is_preliminary=data.get("is_preliminary", False),
            release_date=data.get("release_date", ""),
        )
        self._rebuild_org_rankings()

        # Only trade if competitively ranked
        if rank_ub > 5 or rank_ub <= 0:
            return []

        ticker = self._resolve_ticker(model_name, series="KXTOPMODEL")
        if ticker is None:
            return []

        fair_value = self.estimate_fair_value(model_name)
        contract = self._contracts.get(ticker)
        if fair_value is None or contract is None:
            return []

        market_price = contract.yes_ask or contract.last_price
        if market_price <= 0 or market_price >= 100:
            return []

        urgency = OrderUrgency.ADAPTIVE if rank_ub <= 2 else OrderUrgency.PASSIVE
        return self._propose_buy(
            ticker,
            contract,
            fair_value,
            market_price,
            urgency,
            f"new model {model_name} rank_ub={rank_ub}, fv={fair_value}c mkt={market_price}c",
        )

    def _handle_pairwise_shift(self, signal: Signal) -> list[ProposedOrder]:
        data = signal.data
        model_name = data.get("model_name", "")
        new_avg = float(data.get("new_average_pairwise_win_rate", 0.0))
        new_battles = int(data.get("new_total_pairwise_battles", 0))
        self._pairwise[model_name] = PairwiseAggregate(
            model_name=model_name,
            total_pairwise_battles=new_battles,
            average_pairwise_win_rate=new_avg,
        )
        if new_battles < 2000 or new_avg <= 0.52:
            return []

        ticker = self._resolve_ticker(model_name, series="KXTOPMODEL")
        if ticker is None:
            return []
        fair_value = self.estimate_fair_value(model_name)
        contract = self._contracts.get(ticker)
        if fair_value is None or contract is None:
            return []
        market_price = contract.yes_ask or contract.last_price
        if market_price <= 0 or market_price >= 100:
            return []
        return self._propose_buy(
            ticker,
            contract,
            fair_value,
            market_price,
            OrderUrgency.ADAPTIVE,
            f"pairwise win={new_avg:.3f} battles={new_battles}, fv={fair_value}c mkt={market_price}c",
        )

    # ── Helpers ───────────────────────────────────────────────────────

    def _has_complete_tiebreak_inputs(self) -> bool:
        """Return True when settlement tie-break fields are present for all tracked models."""
        if len(self._rankings) < 2:
            return False
        return all(e.votes > 0 and bool(e.release_date) for e in self._rankings.values())

    def _update_ranking(self, model_name: str, data: dict[str, Any]) -> None:
        """Update internal ranking from signal data."""
        old = self._rankings.get(model_name)
        self._rankings[model_name] = LeaderboardEntry(
            model_name=model_name,
            organization=old.organization if old else "",
            rank=data.get("new_rank", data.get("rank", old.rank if old else 0)),
            rank_ub=data.get("new_rank_ub", data.get("rank_ub", old.rank_ub if old else 0)),
            rank_lb=old.rank_lb if old else 0,
            score=data.get("new_score", data.get("score", old.score if old else 0.0)),
            ci_lower=old.ci_lower if old else 0.0,
            ci_upper=old.ci_upper if old else 0.0,
            votes=data.get("votes", old.votes if old else 0),
            is_preliminary=data.get("is_preliminary", old.is_preliminary if old else False),
            release_date=data.get("release_date", old.release_date if old else ""),
        )
        self._rebuild_org_rankings()

    def _resolve_ticker(self, model_name: str, series: str = "KXTOPMODEL") -> str | None:
        """Resolve an Arena model name to a Kalshi ticker via fuzzy matching."""
        if not model_name:
            return None
        ticker_map = self._org_ticker_map if series == "KXLLM1" else self._model_ticker_map
        if model_name in ticker_map:
            return ticker_map[model_name]
        if not self._ticker_model_names:
            return None

        scoped = {t: n for t, n in self._ticker_model_names.items() if self._ticker_series(t) == series}
        if not scoped:
            return None

        candidates = list(scoped.values())
        match = fuzzy_match(
            model_name,
            candidates,
            threshold=self._config.fuzzy_match_threshold,
            overrides=self._config.model_name_overrides,
        )
        if match is None:
            return None

        for ticker, name in scoped.items():
            if name == match.matched:
                ticker_map[model_name] = ticker
                logger.info(
                    "model_ticker_mapped",
                    model=model_name,
                    ticker=ticker,
                    series=series,
                    score=match.score,
                )
                return ticker
        return None

    def _rebuild_org_rankings(self) -> None:
        """Build org-level standings from tracked model-level rankings."""
        by_org: dict[str, list[LeaderboardEntry]] = {}
        for entry in self._rankings.values():
            if not entry.organization:
                continue
            by_org.setdefault(entry.organization, []).append(entry)

        self._org_rankings = {
            org: min(
                entries,
                key=lambda e: (
                    e.rank_ub if e.rank_ub > 0 else 10_000,
                    -e.score,
                    -e.votes,
                    e.release_date or "9999-12-31",
                    e.model_name,
                ),
            )
            for org, entries in by_org.items()
        }

    def _ticker_series(self, ticker: str) -> str:
        return ticker.split("-", 1)[0] if ticker else ""

    def _propose_buy(
        self,
        ticker: str,
        contract: ContractView,
        fair_value: int,
        market_price: int,
        urgency: OrderUrgency,
        rationale: str,
    ) -> list[ProposedOrder]:
        """Create a buy-YES proposal if edge exceeds fees + min threshold."""
        edge = fair_value - market_price
        if edge <= 0:
            return []

        quantity = self._compute_quantity(contract, edge)
        if quantity <= 0:
            return []

        # Ensure edge covers fees
        min_edge = self._fee_calc.min_edge_for_profit(market_price, quantity)
        required = max(min_edge, self._config.min_edge_after_fees_cents)
        if edge < required:
            return []

        return [
            ProposedOrder(
                strategy=self.name,
                ticker=ticker,
                side="yes",
                price_cents=market_price,
                quantity=quantity,
                urgency=urgency,
                rationale=rationale,
            )
        ]

    def _compute_quantity(self, contract: ContractView, edge: int) -> int:
        """Compute order size respecting position limits.

        Scales linearly with edge: 1 contract at min edge, +1 per extra cent.
        """
        remaining = int(self._config.max_position_per_contract) - abs(contract.position)
        if remaining <= 0:
            return 0
        base = 1 + max(0, edge - self._config.min_edge_after_fees_cents)
        return min(base, remaining)

    # ── Public accessors (monitoring / testing) ───────────────────────

    @property
    def rankings(self) -> dict[str, LeaderboardEntry]:
        return dict(self._rankings)

    @property
    def contracts(self) -> dict[str, ContractView]:
        return dict(self._contracts)

    @property
    def model_ticker_map(self) -> dict[str, str]:
        return dict(self._model_ticker_map)

    def set_rankings(self, rankings: dict[str, LeaderboardEntry]) -> None:
        """Set the leaderboard rankings (for initialization / testing)."""
        self._rankings = dict(rankings)
        self._rebuild_org_rankings()
