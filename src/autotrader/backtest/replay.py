"""Replay engine for backtesting strategies against historical signals.

Usage::

    engine = ReplayEngine(config)
    results = await engine.run(signals_path="signals.json", market_data={...})
"""

from __future__ import annotations

import datetime
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

from autotrader.execution.engine import ExecutionEngine, ExecutionMode
from autotrader.risk.manager import PortfolioSnapshot, PositionInfo, RiskManager
from autotrader.signals.base import Signal, SignalUrgency
from autotrader.strategies.leaderboard_alpha import LeaderboardAlphaStrategy
from autotrader.utils.fees import FeeCalculator

logger = structlog.get_logger("autotrader.backtest.replay")


@dataclass
class ReplayResult:
    """Summary of a replay backtest run."""

    total_signals: int = 0
    total_proposals: int = 0
    total_approved: int = 0
    total_rejected: int = 0
    total_fills: int = 0
    realized_pnl_cents: int = 0
    total_fees_cents: int = 0
    positions: dict[str, int] = field(default_factory=dict)
    trades: list[dict[str, Any]] = field(default_factory=list)


class ReplayEngine:
    """Replays historical signals through the strategy pipeline.

    Simulates: signal → strategy → risk → paper execution, tracking
    P&L and positions throughout.
    """

    def __init__(self, config: Any) -> None:
        self._config = config
        self._fee_calc = FeeCalculator()

    async def run(
        self,
        signals_path: str | Path,
        market_data: dict[str, Any] | None = None,
    ) -> ReplayResult:
        """Run the replay and return results."""
        signals = self._load_signals(Path(signals_path))
        if not signals:
            logger.warning("replay_no_signals", path=str(signals_path))
            return ReplayResult()

        # Initialize strategy
        strategy = LeaderboardAlphaStrategy(
            config=self._config.leaderboard_alpha,
            fee_calculator=self._fee_calc,
        )
        if market_data is None:
            market_data = {"markets": []}
        await strategy.initialize(market_data, {})

        # Risk manager
        risk = RiskManager(config=self._config.risk)

        # Paper execution engine
        engine = ExecutionEngine(mode=ExecutionMode.PAPER, fee_calculator=self._fee_calc)

        # Wire fill callbacks so strategy positions update on each fill.
        def _on_fill(fill_data: dict[str, Any]) -> None:
            ticker = fill_data.get("ticker", "")
            if ticker not in strategy.contracts:
                return
            count = fill_data.get("count", 0)
            side = fill_data.get("side", "")
            action = fill_data.get("action", "buy")
            delta = count if side == "yes" else -count
            if action == "sell":
                delta = -delta
            strategy.contracts[ticker].position += delta

        engine.on_fill(_on_fill)

        result = ReplayResult(total_signals=len(signals))

        for signal in signals:
            proposals = await strategy.on_signal(signal)
            result.total_proposals += len(proposals)

            if not proposals:
                continue

            # Build snapshot for risk checks
            positions = [
                PositionInfo(
                    ticker=t,
                    event_ticker=t.rsplit("-", 1)[0] if "-" in t else t,
                    quantity=c.position,
                    avg_cost_cents=float(c.last_price or c.yes_ask),
                )
                for t, c in strategy.contracts.items()
                if c.position != 0
            ]
            snapshot = PortfolioSnapshot(
                balance_cents=100_000,
                positions=positions,
            )
            risk.update_portfolio(snapshot)

            for proposal in proposals:
                decision = risk.evaluate(proposal)
                if decision.approved:
                    result.total_approved += 1
                    exec_results = await engine.submit_batch([proposal])
                    for er in exec_results:
                        if er.success:
                            result.total_fills += 1
                            fee = self._fee_calc.taker_fee(er.order.price_cents, er.order.quantity).total_fee_cents
                            result.total_fees_cents += fee
                            result.trades.append(
                                {
                                    "ticker": er.order.ticker,
                                    "side": er.order.side,
                                    "price_cents": er.order.price_cents,
                                    "quantity": er.order.quantity,
                                    "fee_cents": fee,
                                }
                            )
                else:
                    result.total_rejected += 1

        # Final positions
        result.positions = {t: c.position for t, c in strategy.contracts.items() if c.position != 0}

        # Compute realized P&L from trades
        result.realized_pnl_cents = self._compute_realized_pnl(result.trades)

        logger.info(
            "replay_complete",
            signals=result.total_signals,
            proposals=result.total_proposals,
            approved=result.total_approved,
            rejected=result.total_rejected,
            fills=result.total_fills,
            realized_pnl=result.realized_pnl_cents,
            fees=result.total_fees_cents,
        )

        return result

    @staticmethod
    def _load_signals(path: Path) -> list[Signal]:
        """Load signals from a JSON file."""
        if not path.exists():
            return []
        with open(path) as f:
            raw = json.load(f)

        if not isinstance(raw, list):
            raw = [raw]

        signals: list[Signal] = []
        for entry in raw:
            ts = entry.get("timestamp")
            if isinstance(ts, str):
                ts = datetime.datetime.fromisoformat(ts)
            elif ts is None:
                ts = datetime.datetime(2026, 1, 1)

            urgency_str = entry.get("urgency", "low")
            try:
                urgency = SignalUrgency(urgency_str)
            except ValueError:
                urgency = SignalUrgency.LOW

            signals.append(
                Signal(
                    source=entry.get("source", "replay"),
                    timestamp=ts,
                    event_type=entry.get("event_type", ""),
                    data=entry.get("data", {}),
                    relevant_series=entry.get("relevant_series", []),
                    urgency=urgency,
                )
            )
        return signals

    @staticmethod
    def _compute_realized_pnl(trades: list[dict[str, Any]]) -> int:
        """Estimate realized P&L from paper trades (simplified)."""
        # For paper trades the fill price = proposed price, and settlement
        # isn't known. Return negative fees as a conservative estimate.
        total_fees = sum(t.get("fee_cents", 0) for t in trades)
        return -total_fees
