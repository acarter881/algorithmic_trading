"""Leaderboard Alpha trading strategy.

Reacts to LMSYS Chatbot Arena leaderboard changes to trade Kalshi
prediction market contracts (KXTOPMODEL, KXLLM1).

Edge source: faster reaction to rank/score changes than manual traders,
combined with quantitative fair-value estimation and fee-aware filtering.
"""

from __future__ import annotations

import math
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import structlog

from autotrader.signals.arena_types import LeaderboardEntry, PairwiseAggregate
from autotrader.signals.settlement import resolve_top_model, resolve_top_org
from autotrader.strategies.base import OrderUrgency, ProposedOrder, Strategy
from autotrader.utils.fees import FeeCalculator
from autotrader.utils.matching import fuzzy_match, normalize_org_name

if TYPE_CHECKING:
    from autotrader.config.models import LeaderboardAlphaConfig
    from autotrader.signals.base import Signal

logger = structlog.get_logger("autotrader.strategies.leaderboard_alpha")

# Regex patterns for extracting model/org names from Kalshi contract titles.
# Titles follow patterns like "Claude Opus 4.6 will be the #1 AI model"
# or "Anthropic will have the #1 AI model".
_TITLE_SUFFIX_RE = re.compile(
    r"\s+will (?:be|have) the #1 AI model$",
    re.IGNORECASE,
)

# "Model:: Org" format used by some Kalshi subtitles (e.g. "Claude:: Anthropic").
# We extract just the first part (the model or org display name).
_SUBTITLE_DOUBLE_COLON_RE = re.compile(r"^(.+?)::\s*(.+)$")


def _extract_contract_name(subtitle: str, title: str) -> str:
    """Extract a clean model/org display name from Kalshi contract fields.

    Handles:
    - Normal subtitles (``"Claude Opus 4.6"``) — returned as-is.
    - ``"Model:: Org"`` subtitles — extracts the first component.
    - Empty subtitles — falls back to title with boilerplate stripped.
    """
    raw = (subtitle or "").strip()

    # Handle "Model:: Org" format — take the first part
    if raw:
        m = _SUBTITLE_DOUBLE_COLON_RE.match(raw)
        if m:
            return m.group(1).strip()
        return raw

    # Subtitle is empty — fall back to title, stripping boilerplate suffix.
    raw_title = (title or "").strip()
    if raw_title:
        cleaned = _TITLE_SUFFIX_RE.sub("", raw_title).strip()
        if cleaned:
            return cleaned
        return raw_title

    return ""


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
        # Kalshi ticker → Kalshi event ticker
        self._ticker_event_map: dict[str, str] = {}
        # Arena model name → pairwise aggregate metrics
        self._pairwise: dict[str, PairwiseAggregate] = {}
        # Arena organization → best leaderboard entry for that org
        self._org_rankings: dict[str, LeaderboardEntry] = {}
        # Arena organization → Kalshi ticker
        self._org_ticker_map: dict[str, str] = {}
        # Mispricing cooldowns: ticker → monotonic timestamp of last proposal
        self._mispricing_cooldowns: dict[str, float] = {}

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
                model_name = _extract_contract_name(
                    subtitle=m.get("subtitle", ""),
                    title=m.get("title", ""),
                )
                self._ticker_model_names[ticker] = model_name
                raw_event_ticker = m.get("event_ticker")
                event_ticker = (
                    raw_event_ticker
                    if isinstance(raw_event_ticker, str) and raw_event_ticker
                    else self._fallback_event_ticker(ticker)
                )
                self._ticker_event_map[ticker] = event_ticker
                self._contracts[ticker] = ContractView(
                    ticker=ticker,
                    model_name=model_name,
                    series=self._ticker_series(ticker),
                    yes_bid=m.get("yes_bid", 0),
                    yes_ask=m.get("yes_ask", 0),
                    last_price=m.get("last_price", 0),
                )
            logger.info(
                "contract_names_resolved",
                count=len(self._ticker_model_names),
                names={t: n for t, n in list(self._ticker_model_names.items())[:10]},
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
            for org_name, ticker in state.get("org_ticker_map", {}).items():
                if isinstance(org_name, str) and isinstance(ticker, str):
                    self._org_ticker_map[org_name] = ticker
            for ticker, event_ticker in state.get("ticker_event_map", {}).items():
                if isinstance(ticker, str) and isinstance(event_ticker, str) and event_ticker:
                    self._ticker_event_map[ticker] = event_ticker
        self._rebuild_org_rankings()

        logger.info(
            "strategy_initialized",
            strategy=self.name,
            contracts=len(self._contracts),
            contract_names={t: n for t, n in self._ticker_model_names.items()},
        )

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
        """Update market prices and check for mispricings."""
        if isinstance(data, dict) and data.get("ticker") in self._contracts:
            c = self._contracts[data["ticker"]]
            for key in ("yes_bid", "yes_ask", "last_price"):
                value = data.get(key)
                if isinstance(value, int) and value > 0:
                    setattr(c, key, value)
            event_ticker = data.get("event_ticker")
            if isinstance(event_ticker, str) and event_ticker:
                self._ticker_event_map[data["ticker"]] = event_ticker
            return self._check_mispricing(c)
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
            "ticker_event_map": dict(self._ticker_event_map),
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

    # ── Ranking seeding ──────────────────────────────────────────────

    def seed_rankings(
        self,
        entries: list[LeaderboardEntry],
        pairwise: dict[str, PairwiseAggregate] | None = None,
    ) -> None:
        """Populate rankings from a full leaderboard snapshot.

        Called once by the trading loop after the first successful Arena
        fetch so the strategy has fair-value estimates before any diff
        signals arrive.  Eagerly resolves tickers for top-ranked models.
        """
        for entry in entries:
            self._rankings[entry.model_name] = entry
        if pairwise:
            self._pairwise.update(pairwise)
        self._rebuild_org_rankings()
        self._apply_config_overrides()

        # Eagerly resolve tickers for competitive models.
        # Sort by rank_ub and take the top N rather than filtering by a
        # hardcoded threshold — the Arena's rank_ub is a confidence-
        # interval bound (e.g. rank 1 with spread 1-13 → rank_ub=13),
        # not a display rank, so a fixed cutoff silently drops everyone.
        top_n = self._config.mapping_validation_top_n
        eligible = sorted(
            (e for e in entries if e.rank_ub > 0),
            key=lambda e: e.rank_ub,
        )[:top_n]
        if not eligible:
            logger.warning(
                "seed_rankings_no_eligible_models",
                strategy=self.name,
                total_entries=len(entries),
                sample_rank_ubs=[e.rank_ub for e in entries[:10]],
                hint="No entries with rank_ub > 0; check arena parser output",
            )
        for entry in eligible:
            self._resolve_ticker(entry.model_name, series="KXTOPMODEL")
            if entry.organization:
                self._resolve_ticker(entry.organization, series="KXLLM1")

        logger.info(
            "rankings_seeded",
            strategy=self.name,
            model_count=len(entries),
            eligible_for_mapping=len(eligible),
            mapped_models=len(self._model_ticker_map),
            mapped_orgs=len(self._org_ticker_map),
        )

    def validate_mappings(self) -> dict[str, Any]:
        """Validate that top-N Arena models/orgs map to Kalshi contracts.

        Attempts resolution for every top-N entry (not just those already
        cached by ``seed_rankings``, which only covers the top 10).

        Returns a report dict with:
        - ``mapped_models``: list of ``{model, ticker, series}`` dicts
        - ``unmapped_models``: list of ``{model, rank_ub, series, reason}`` dicts
        - ``mapped_orgs``: list of ``{org, ticker, series}`` dicts
        - ``unmapped_orgs``: list of ``{org, best_model_rank_ub, series, reason}`` dicts

        The trading loop calls this after ``seed_rankings`` and sends a
        Discord alert if any top-N entries are unmapped.
        """
        top_n = self._config.mapping_validation_top_n
        ranked = sorted(
            (e for e in self._rankings.values() if e.rank_ub > 0),
            key=lambda e: e.rank_ub,
        )[:top_n]

        mapped_models: list[dict[str, Any]] = []
        unmapped_models: list[dict[str, Any]] = []
        mapped_orgs: list[dict[str, Any]] = []
        unmapped_orgs: list[dict[str, Any]] = []
        seen_orgs: set[str] = set()

        for entry in ranked:
            # --- KXTOPMODEL: model → contract ---
            # Attempt resolution (returns cached result if already resolved)
            ticker = self._resolve_ticker(entry.model_name, series="KXTOPMODEL")
            if ticker:
                mapped_models.append({"model": entry.model_name, "ticker": ticker, "series": "KXTOPMODEL"})
            else:
                unmapped_models.append(
                    {
                        "model": entry.model_name,
                        "rank_ub": entry.rank_ub,
                        "series": "KXTOPMODEL",
                        "reason": "no Kalshi contract matched",
                    }
                )

            # --- KXLLM1: org → contract ---
            org = entry.organization
            if not org or org in seen_orgs:
                continue
            seen_orgs.add(org)
            org_ticker = self._resolve_ticker(org, series="KXLLM1")
            if org_ticker:
                mapped_orgs.append({"org": org, "ticker": org_ticker, "series": "KXLLM1"})
            else:
                unmapped_orgs.append(
                    {
                        "org": org,
                        "best_model_rank_ub": entry.rank_ub,
                        "series": "KXLLM1",
                        "reason": "no Kalshi contract matched",
                    }
                )

        report = {
            "top_n": top_n,
            "mapped_models": mapped_models,
            "unmapped_models": unmapped_models,
            "mapped_orgs": mapped_orgs,
            "unmapped_orgs": unmapped_orgs,
        }
        logger.info(
            "mapping_validation",
            strategy=self.name,
            top_n=top_n,
            models_mapped=len(mapped_models),
            models_unmapped=len(unmapped_models),
            orgs_mapped=len(mapped_orgs),
            orgs_unmapped=len(unmapped_orgs),
        )
        if unmapped_models or unmapped_orgs:
            logger.warning(
                "mapping_validation_gaps",
                strategy=self.name,
                unmapped_models=[m["model"] for m in unmapped_models],
                unmapped_orgs=[o["org"] for o in unmapped_orgs],
            )
        self._print_mapping_report(report)
        return report

    def _print_mapping_report(self, report: dict[str, Any]) -> None:
        """Print a human-readable mapping report to stdout.

        Includes copy-paste YAML for any unmapped entries so the operator
        can quickly add manual overrides to the config file.
        """
        mapped_models = report["mapped_models"]
        unmapped_models = report["unmapped_models"]
        mapped_orgs = report["mapped_orgs"]
        unmapped_orgs = report["unmapped_orgs"]
        top_n = report["top_n"]

        lines = [
            "",
            "=" * 64,
            f"  MAPPING REPORT  (top {top_n} Arena entries)",
            "=" * 64,
            "",
            f"  KXTOPMODEL  {len(mapped_models)} mapped, {len(unmapped_models)} unmapped",
            f"  KXLLM1      {len(mapped_orgs)} mapped, {len(unmapped_orgs)} unmapped",
            "",
        ]

        if mapped_models:
            lines.append("  Mapped models:")
            for m in mapped_models:
                lines.append(f"    {m['model']!r:40s} -> {m['ticker']}")

        if mapped_orgs:
            lines.append("  Mapped orgs:")
            for o in mapped_orgs:
                lines.append(f"    {o['org']!r:40s} -> {o['ticker']}")

        if unmapped_models or unmapped_orgs:
            lines.append("")
            lines.append("-" * 64)
            lines.append("  UNMAPPED — add to config/strategies/leaderboard_alpha.yaml:")
            lines.append("-" * 64)

        if unmapped_models:
            # Find best-guess ticker for each unmapped model
            lines.append("")
            lines.append("  model_overrides:")
            for m in unmapped_models:
                guess = self._best_ticker_guess(m["model"], "KXTOPMODEL")
                comment = f"  # rank_ub={m['rank_ub']}, best guess"
                if guess:
                    lines.append(f'    "{m["model"]}": "{guess}"{comment}')
                else:
                    lines.append(f'    "{m["model"]}": "KXTOPMODEL-???"{comment}')

        if unmapped_orgs:
            lines.append("")
            lines.append("  org_overrides:")
            for o in unmapped_orgs:
                guess = self._best_ticker_guess(o["org"], "KXLLM1")
                comment = "  # best guess"
                if guess:
                    lines.append(f'    "{o["org"]}": "{guess}"{comment}')
                else:
                    lines.append(f'    "{o["org"]}": "KXLLM1-???"{comment}')

        if unmapped_models or unmapped_orgs:
            lines.append("")
            lines.append("  Available tickers for reference:")
            for series in ("KXTOPMODEL", "KXLLM1"):
                tickers = sorted(t for t in self._ticker_model_names if self._ticker_series(t) == series)
                if tickers:
                    lines.append(f"    {series}:")
                    for t in tickers:
                        name = self._ticker_model_names[t]
                        mapped_flag = (
                            "*"
                            if t in set(m["ticker"] for m in mapped_models) | set(o["ticker"] for o in mapped_orgs)
                            else " "
                        )
                        lines.append(f"      {mapped_flag} {t:45s} ({name})")

        lines.append("")
        lines.append("=" * 64)
        lines.append("")

        print("\n".join(lines))

    def _best_ticker_guess(self, name: str, series: str) -> str | None:
        """Return the best-scoring ticker for *name* even if below threshold."""
        from thefuzz import fuzz as _fuzz

        from autotrader.utils.matching import normalize_model_name

        scoped = {t: n for t, n in self._ticker_model_names.items() if self._ticker_series(t) == series}
        if not scoped:
            return None

        # For org names, also try the stripped variant
        queries = [name]
        if series == "KXLLM1":
            stripped = normalize_org_name(name)
            if stripped != name:
                queries.append(stripped)

        best_ticker = None
        best_score = 0.0
        for q in queries:
            normalized = normalize_model_name(q)
            for ticker, candidate in scoped.items():
                score = _fuzz.token_sort_ratio(normalized, normalize_model_name(candidate)) / 100.0
                if score > best_score:
                    best_score = score
                    best_ticker = ticker
        return best_ticker

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
            elif edge < 0 and contract.position > 0:
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
                            rationale=(
                                f"rank_ub {old_rank_ub}->{new_rank_ub}, overpriced by {-edge}c, reducing position"
                            ),
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
        new_org = (
            data.get("new_organization")
            or data.get("new_top_org")
            or self._rankings.get(new_leader, LeaderboardEntry(model_name="")).organization
        )
        prev_org = (
            data.get("previous_organization")
            or self._rankings.get(previous_leader, LeaderboardEntry(model_name="")).organization
        )

        if isinstance(new_org, str) and new_org:
            org_ticker = self._resolve_ticker(new_org, series="KXLLM1")
            org_contract = self._contracts.get(org_ticker) if org_ticker else None
            rank1_org_fv = max(1, min(99, round(rank_to_probability(1) * 100)))
            estimated_org_fv = self.estimate_org_fair_value(new_org)
            org_fv = max(rank1_org_fv, estimated_org_fv or 0)
            if org_ticker and org_contract and org_fv > 0:
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

    # ── Mispricing detection ─────────────────────────────────────────

    def _check_mispricing(self, contract: ContractView) -> list[ProposedOrder]:
        """Check if a contract is mispriced relative to fair value.

        Called on every market data update.  Returns buy or sell proposals
        when the edge exceeds ``mispricing_min_edge_cents`` and the ticker
        is not on cooldown.
        """
        if not self._config.mispricing_detection_enabled:
            return []
        if not self._rankings:
            return []

        ticker = contract.ticker
        series = contract.series

        # Reverse-lookup: find the Arena name for this ticker
        fair_value: int | None = None
        if series == "KXTOPMODEL":
            arena_name = self._reverse_lookup_model(ticker)
            if arena_name is not None:
                fair_value = self.estimate_fair_value(arena_name)
        elif series == "KXLLM1":
            arena_name = self._reverse_lookup_org(ticker)
            if arena_name is not None:
                fair_value = self.estimate_org_fair_value(arena_name)
        else:
            return []

        if fair_value is None:
            return []

        # Cooldown check
        now = time.monotonic()
        last = self._mispricing_cooldowns.get(ticker)
        if last is not None and now - last < self._config.mispricing_cooldown_seconds:
            return []

        market_price = contract.yes_ask or contract.last_price
        if market_price <= 0 or market_price >= 100:
            return []

        edge = fair_value - market_price
        orders: list[ProposedOrder] = []

        if edge >= self._config.mispricing_min_edge_cents:
            # Underpriced — buy YES
            orders = self._propose_buy(
                ticker,
                contract,
                fair_value,
                market_price,
                OrderUrgency.AGGRESSIVE,
                f"mispricing: fv={fair_value}c mkt={market_price}c edge={edge}c",
            )
        elif edge <= -self._config.mispricing_min_edge_cents and contract.position > 0:
            # Overpriced and we hold YES — sell to reduce position
            sell_price = contract.yes_bid or contract.last_price
            if 0 < sell_price < 100:
                sell_qty = min(contract.position, int(self._config.max_position_per_contract))
                orders = [
                    ProposedOrder(
                        strategy=self.name,
                        ticker=ticker,
                        side="no",
                        price_cents=100 - sell_price,
                        quantity=sell_qty,
                        urgency=OrderUrgency.AGGRESSIVE,
                        rationale=f"mispricing: overpriced by {-edge}c, fv={fair_value}c mkt={market_price}c, reducing position",
                    )
                ]

        if orders:
            self._mispricing_cooldowns[ticker] = now
            logger.info(
                "mispricing_detected",
                strategy=self.name,
                ticker=ticker,
                fair_value=fair_value,
                market_price=market_price,
                edge=edge,
                side="buy" if edge > 0 else "sell",
            )

        return orders

    def _reverse_lookup_model(self, ticker: str) -> str | None:
        """Find the Arena model name mapped to a Kalshi ticker."""
        for arena_name, mapped_ticker in self._model_ticker_map.items():
            if mapped_ticker == ticker:
                return arena_name
        return None

    def _reverse_lookup_org(self, ticker: str) -> str | None:
        """Find the Arena organization name mapped to a Kalshi ticker."""
        for org_name, mapped_ticker in self._org_ticker_map.items():
            if mapped_ticker == ticker:
                return org_name
        return None

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
            organization=data.get("organization", old.organization if old else ""),
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

    def _apply_config_overrides(self) -> None:
        """Pre-populate ticker maps from config overrides.

        Called once during ``seed_rankings`` so that manual overrides take
        effect before any fuzzy-matching runs.  Unknown tickers (not
        present in loaded contracts) are logged and skipped.
        """
        known = set(self._ticker_model_names)
        for arena_name, ticker in self._config.model_overrides.items():
            if ticker not in known:
                logger.warning(
                    "model_override_unknown_ticker",
                    model=arena_name,
                    ticker=ticker,
                    known_ticker_count=len(known),
                )
                continue
            self._model_ticker_map[arena_name] = ticker
            logger.info("model_override_applied", model=arena_name, ticker=ticker)
        for arena_name, ticker in self._config.org_overrides.items():
            if ticker not in known:
                logger.warning(
                    "org_override_unknown_ticker",
                    org=arena_name,
                    ticker=ticker,
                    known_ticker_count=len(known),
                )
                continue
            self._org_ticker_map[arena_name] = ticker
            logger.info("org_override_applied", org=arena_name, ticker=ticker)

    def _resolve_ticker(self, model_name: str, series: str | None = None) -> str | None:
        """Resolve an Arena name to a Kalshi ticker via fuzzy matching."""
        if not model_name:
            return None

        if series == "KXLLM1":
            if model_name in self._org_ticker_map:
                return self._org_ticker_map[model_name]
        elif series == "KXTOPMODEL":
            if model_name in self._model_ticker_map:
                return self._model_ticker_map[model_name]
        else:
            if model_name in self._model_ticker_map:
                return self._model_ticker_map[model_name]
            if model_name in self._org_ticker_map:
                return self._org_ticker_map[model_name]

        if not self._ticker_model_names:
            logger.debug("resolve_ticker_no_contracts", model=model_name, series=series)
            return None

        if series is None:
            scoped = dict(self._ticker_model_names)
        else:
            scoped = {t: n for t, n in self._ticker_model_names.items() if self._ticker_series(t) == series}
        if not scoped:
            logger.debug(
                "resolve_ticker_no_series_contracts",
                model=model_name,
                series=series,
                available_series=list({self._ticker_series(t) for t in self._ticker_model_names}),
            )
            return None

        candidates = list(scoped.values())

        # Build list of query variants to try.  For org names (KXLLM1),
        # also try stripping common suffixes ("Google DeepMind" → "Google")
        # since Kalshi subtitles often use the shorter company name.
        queries = [model_name]
        if series == "KXLLM1":
            stripped = normalize_org_name(model_name)
            if stripped != model_name:
                queries.append(stripped)

        match = None
        for q in queries:
            match = fuzzy_match(
                q,
                candidates,
                threshold=self._config.fuzzy_match_threshold,
            )
            if match is not None:
                break

        if match is None:
            # Log the best score even when below threshold, so operators
            # can see near-misses and add overrides.
            from thefuzz import fuzz as _fuzz

            from autotrader.utils.matching import normalize_model_name

            normalized = normalize_model_name(model_name)
            scored = sorted(
                ((c, _fuzz.token_sort_ratio(normalized, normalize_model_name(c)) / 100.0) for c in candidates),
                key=lambda x: x[1],
                reverse=True,
            )
            top = scored[:3]
            logger.info(
                "resolve_ticker_no_match",
                model=model_name,
                series=series,
                threshold=self._config.fuzzy_match_threshold,
                best_candidates=[{"name": n, "score": round(s, 3)} for n, s in top],
            )
            return None

        for ticker, name in scoped.items():
            if name == match.matched:
                cache = self._org_ticker_map if self._ticker_series(ticker) == "KXLLM1" else self._model_ticker_map
                cache[model_name] = ticker
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

    def resolve_event_ticker(self, ticker: str) -> str:
        """Resolve the event ticker for a contract ticker."""
        return self._ticker_event_map.get(ticker, self._fallback_event_ticker(ticker))

    def _fallback_event_ticker(self, ticker: str) -> str:
        """Fallback event derivation for payloads missing event_ticker."""
        return ticker.rsplit("-", 1)[0] if "-" in ticker else ticker

    def set_rankings(self, rankings: dict[str, LeaderboardEntry]) -> None:
        """Set the leaderboard rankings (for initialization / testing)."""
        self._rankings = dict(rankings)
        self._rebuild_org_rankings()
