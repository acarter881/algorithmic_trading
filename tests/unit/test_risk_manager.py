"""Unit tests for the risk manager."""

from __future__ import annotations

from autotrader.config.models import RiskConfig, RiskGlobalConfig, RiskStrategyConfig
from autotrader.risk.manager import PortfolioSnapshot, PositionInfo, RiskManager, RiskVerdict
from autotrader.strategies.base import OrderUrgency, ProposedOrder

# ── Helpers ──────────────────────────────────────────────────────────────


def _config(
    max_portfolio_exposure_pct: float = 0.60,
    max_daily_loss_pct: float = 0.05,
    kill_switch: bool = False,
    strategy_limits: dict[str, RiskStrategyConfig] | None = None,
) -> RiskConfig:
    return RiskConfig(
        global_config=RiskGlobalConfig(
            max_portfolio_exposure_pct=max_portfolio_exposure_pct,
            max_daily_loss_pct=max_daily_loss_pct,
            kill_switch_enabled=kill_switch,
        ),
        per_strategy=strategy_limits or {},
    )


DEFAULT_STRATEGY_LIMITS = {
    "leaderboard_alpha": RiskStrategyConfig(
        max_position_per_contract=100,
        max_position_per_event=250,
        max_strategy_loss=200,
        min_edge_multiplier=2.5,
    )
}


def _order(
    strategy: str = "leaderboard_alpha",
    ticker: str = "KXTOPMODEL-GPT5",
    side: str = "yes",
    price_cents: int = 50,
    quantity: int = 10,
    urgency: OrderUrgency = OrderUrgency.PASSIVE,
) -> ProposedOrder:
    return ProposedOrder(
        strategy=strategy,
        ticker=ticker,
        side=side,
        price_cents=price_cents,
        quantity=quantity,
        urgency=urgency,
        rationale="test order",
    )


def _portfolio(
    balance_cents: int = 100_000,
    positions: list[PositionInfo] | None = None,
    daily_realized: dict[str, int] | None = None,
    daily_unrealized: dict[str, int] | None = None,
) -> PortfolioSnapshot:
    return PortfolioSnapshot(
        balance_cents=balance_cents,
        positions=positions or [],
        daily_realized_pnl_cents=daily_realized or {},
        daily_unrealized_pnl_cents=daily_unrealized or {},
    )


def _manager(
    cfg: RiskConfig | None = None,
    portfolio: PortfolioSnapshot | None = None,
) -> RiskManager:
    rm = RiskManager(cfg or _config(strategy_limits=DEFAULT_STRATEGY_LIMITS))
    if portfolio:
        rm.update_portfolio(portfolio)
    return rm


# ── Basic Approval ───────────────────────────────────────────────────────


class TestBasicApproval:
    def test_simple_order_approved(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order())
        assert decision.approved
        assert decision.verdict == RiskVerdict.APPROVED

    def test_approved_has_all_check_results(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order())
        check_names = {r.check_name for r in decision.results}
        assert "kill_switch" in check_names
        assert "price_sanity" in check_names
        assert "position_per_contract" in check_names
        assert "position_per_event" in check_names
        assert "daily_loss" in check_names
        assert "portfolio_exposure" in check_names

    def test_no_strategy_config_passes_all_checks(self) -> None:
        rm = _manager(cfg=_config(strategy_limits={}), portfolio=_portfolio())
        decision = rm.evaluate(_order(strategy="unknown_strategy"))
        assert decision.approved


# ── Kill Switch ──────────────────────────────────────────────────────────


class TestKillSwitch:
    def test_kill_switch_from_config(self) -> None:
        rm = _manager(cfg=_config(kill_switch=True, strategy_limits=DEFAULT_STRATEGY_LIMITS))
        decision = rm.evaluate(_order())
        assert not decision.approved
        assert any(r.check_name == "kill_switch" for r in decision.results if r.verdict == RiskVerdict.REJECTED)

    def test_activate_kill_switch(self) -> None:
        rm = _manager(portfolio=_portfolio())
        assert rm.evaluate(_order()).approved

        rm.activate_kill_switch("test reason")
        assert rm.kill_switch_active
        assert not rm.evaluate(_order()).approved

    def test_deactivate_kill_switch(self) -> None:
        rm = _manager(cfg=_config(kill_switch=True, strategy_limits=DEFAULT_STRATEGY_LIMITS), portfolio=_portfolio())
        assert not rm.evaluate(_order()).approved

        rm.deactivate_kill_switch()
        assert not rm.kill_switch_active
        assert rm.evaluate(_order()).approved


# ── Price Sanity ─────────────────────────────────────────────────────────


class TestPriceSanity:
    def test_price_zero_rejected(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order(price_cents=0))
        assert not decision.approved
        assert "price_sanity" in [r.check_name for r in decision.results if r.verdict == RiskVerdict.REJECTED]

    def test_price_100_rejected(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order(price_cents=100))
        assert not decision.approved

    def test_price_negative_rejected(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order(price_cents=-5))
        assert not decision.approved

    def test_price_1_approved(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order(price_cents=1))
        assert decision.approved

    def test_price_99_approved(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order(price_cents=99))
        assert decision.approved

    def test_quantity_zero_rejected(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order(quantity=0))
        assert not decision.approved

    def test_quantity_negative_rejected(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order(quantity=-1))
        assert not decision.approved


# ── Position Per Contract ────────────────────────────────────────────────


class TestPositionPerContract:
    def test_within_limit_approved(self) -> None:
        positions = [PositionInfo(ticker="KXTOPMODEL-GPT5", event_ticker="KXTOPMODEL", quantity=80)]
        rm = _manager(portfolio=_portfolio(positions=positions))
        decision = rm.evaluate(_order(quantity=10))
        assert decision.approved

    def test_at_limit_rejected(self) -> None:
        positions = [PositionInfo(ticker="KXTOPMODEL-GPT5", event_ticker="KXTOPMODEL", quantity=95)]
        rm = _manager(portfolio=_portfolio(positions=positions))
        decision = rm.evaluate(_order(quantity=10))
        assert not decision.approved
        reasons = decision.rejection_reasons
        assert any("per-contract" in r for r in reasons)

    def test_no_existing_position_approved(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order(quantity=50))
        assert decision.approved

    def test_exactly_at_max(self) -> None:
        positions = [PositionInfo(ticker="KXTOPMODEL-GPT5", event_ticker="KXTOPMODEL", quantity=90)]
        rm = _manager(portfolio=_portfolio(positions=positions))
        # 90 + 10 = 100, max is 100 → should be approved
        decision = rm.evaluate(_order(quantity=10))
        assert decision.approved

    def test_one_over_max_rejected(self) -> None:
        positions = [PositionInfo(ticker="KXTOPMODEL-GPT5", event_ticker="KXTOPMODEL", quantity=91)]
        rm = _manager(portfolio=_portfolio(positions=positions))
        # 91 + 10 = 101 > 100 → rejected
        decision = rm.evaluate(_order(quantity=10))
        assert not decision.approved


# ── Position Per Event ───────────────────────────────────────────────────


class TestPositionPerEvent:
    def test_within_event_limit_approved(self) -> None:
        positions = [
            PositionInfo(ticker="KXTOPMODEL-GPT5", event_ticker="KXTOPMODEL-EV1", quantity=100),
            PositionInfo(ticker="KXTOPMODEL-GEMINI3", event_ticker="KXTOPMODEL-EV1", quantity=100),
        ]
        rm = _manager(portfolio=_portfolio(positions=positions))
        # New order for a different contract in the same event
        order = _order(ticker="KXTOPMODEL-CLAUDE5", quantity=40)
        # Ticker not in positions → event_ticker defaults to ticker itself
        # So this order's event is "KXTOPMODEL-CLAUDE5", not "KXTOPMODEL-EV1"
        decision = rm.evaluate(order)
        assert decision.approved

    def test_exceeds_event_limit_rejected(self) -> None:
        positions = [
            PositionInfo(ticker="KXTOPMODEL-GPT5", event_ticker="KXTOPMODEL-EV1", quantity=150),
            PositionInfo(ticker="KXTOPMODEL-GEMINI3", event_ticker="KXTOPMODEL-EV1", quantity=80),
        ]
        rm = _manager(portfolio=_portfolio(positions=positions))
        # Order on a ticker that's already in the same event
        order = _order(ticker="KXTOPMODEL-GPT5", quantity=30)
        # event total = 150 + 80 = 230, + 30 = 260 > 250 → rejected
        decision = rm.evaluate(order)
        assert not decision.approved
        reasons = decision.rejection_reasons
        assert any("per-event" in r for r in reasons)

    def test_exactly_at_event_max_approved(self) -> None:
        # Spread event exposure across contracts to avoid per-contract limit
        positions = [
            PositionInfo(ticker="KXTOPMODEL-GPT5", event_ticker="KXTOPMODEL-EV1", quantity=50),
            PositionInfo(ticker="KXTOPMODEL-GEMINI3", event_ticker="KXTOPMODEL-EV1", quantity=100),
            PositionInfo(ticker="KXTOPMODEL-CLAUDE5", event_ticker="KXTOPMODEL-EV1", quantity=50),
        ]
        rm = _manager(portfolio=_portfolio(positions=positions))
        order = _order(ticker="KXTOPMODEL-GPT5", quantity=50)
        # event total = 50+100+50 = 200, + 50 = 250 = max → approved
        # per-contract for GPT5 = 50 + 50 = 100 = max → approved
        decision = rm.evaluate(order)
        assert decision.approved


# ── Daily Loss ───────────────────────────────────────────────────────────


class TestDailyLoss:
    def test_no_loss_approved(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order())
        assert decision.approved

    def test_profitable_day_approved(self) -> None:
        rm = _manager(portfolio=_portfolio(daily_realized={"leaderboard_alpha": 5000}))
        decision = rm.evaluate(_order())
        assert decision.approved

    def test_loss_within_limit_approved(self) -> None:
        # max_strategy_loss = 200 ($200), loss = $150 → still OK
        rm = _manager(portfolio=_portfolio(daily_realized={"leaderboard_alpha": -15000}))
        decision = rm.evaluate(_order())
        assert decision.approved

    def test_loss_at_limit_rejected(self) -> None:
        # max_strategy_loss = 200 ($200), loss = $200 → rejected
        rm = _manager(portfolio=_portfolio(daily_realized={"leaderboard_alpha": -20000}))
        decision = rm.evaluate(_order())
        assert not decision.approved
        reasons = decision.rejection_reasons
        assert any("daily loss" in r.lower() for r in reasons)

    def test_combined_realized_unrealized_loss(self) -> None:
        # realized = -$100, unrealized = -$110 → total = -$210 > $200 limit
        rm = _manager(
            portfolio=_portfolio(
                daily_realized={"leaderboard_alpha": -10000},
                daily_unrealized={"leaderboard_alpha": -11000},
            )
        )
        decision = rm.evaluate(_order())
        assert not decision.approved

    def test_different_strategy_unaffected(self) -> None:
        # Loss is on a different strategy
        rm = _manager(portfolio=_portfolio(daily_realized={"other_strategy": -50000}))
        decision = rm.evaluate(_order(strategy="leaderboard_alpha"))
        assert decision.approved


# ── Portfolio Exposure ───────────────────────────────────────────────────


class TestPortfolioExposure:
    def test_within_exposure_limit_approved(self) -> None:
        # Balance = $1000, exposure = $500 (50%), order adds $50 → 55% < 60%
        positions = [PositionInfo(ticker="KXTOPMODEL-GPT5", event_ticker="EV1", quantity=10, avg_cost_cents=50)]
        rm = _manager(portfolio=_portfolio(balance_cents=100_000, positions=positions))
        decision = rm.evaluate(_order(price_cents=50, quantity=10))
        assert decision.approved

    def test_exceeds_exposure_limit_rejected(self) -> None:
        # Balance = 100000c ($1000), existing = 1000*50 = 50000c
        # Order adds 50*300 = 15000c → total 65000/100000 = 65% > 60%
        positions = [PositionInfo(ticker="KXTOPMODEL-GPT5", event_ticker="EV1", quantity=1000, avg_cost_cents=50)]
        rm = _manager(portfolio=_portfolio(balance_cents=100_000, positions=positions))
        decision = rm.evaluate(_order(price_cents=50, quantity=300))
        assert not decision.approved

    def test_zero_balance_approved(self) -> None:
        rm = _manager(portfolio=_portfolio(balance_cents=0))
        decision = rm.evaluate(_order())
        assert decision.approved

    def test_no_existing_positions(self) -> None:
        rm = _manager(portfolio=_portfolio(balance_cents=100_000))
        # order cost = 50 * 10 = 500c / 100000c = 0.5% << 60%
        decision = rm.evaluate(_order(price_cents=50, quantity=10))
        assert decision.approved


# ── Batch Evaluation ─────────────────────────────────────────────────────


class TestBatchEvaluation:
    def test_batch_returns_one_decision_per_order(self) -> None:
        rm = _manager(portfolio=_portfolio())
        orders = [_order(quantity=5), _order(quantity=10), _order(quantity=15)]
        decisions = rm.evaluate_batch(orders)
        assert len(decisions) == 3

    def test_batch_mixed_results(self) -> None:
        rm = _manager(portfolio=_portfolio())
        orders = [_order(price_cents=50), _order(price_cents=0)]  # 2nd has bad price
        decisions = rm.evaluate_batch(orders)
        assert decisions[0].approved
        assert not decisions[1].approved


# ── RiskDecision Helpers ─────────────────────────────────────────────────


class TestRiskDecision:
    def test_rejection_reasons_collected(self) -> None:
        rm = _manager(cfg=_config(kill_switch=True, strategy_limits=DEFAULT_STRATEGY_LIMITS))
        decision = rm.evaluate(_order(price_cents=0))
        # Both kill switch and price sanity should reject
        assert len(decision.rejection_reasons) >= 2

    def test_approved_no_rejection_reasons(self) -> None:
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order())
        assert decision.rejection_reasons == []


# ── Serialize Order ──────────────────────────────────────────────────────


class TestSerializeOrder:
    def test_serializes_all_fields(self) -> None:
        rm = _manager()
        order = _order(strategy="test_strat", ticker="T1", side="yes", price_cents=42, quantity=7)
        result = rm.serialize_order(order)
        assert result["strategy"] == "test_strat"
        assert result["ticker"] == "T1"
        assert result["side"] == "yes"
        assert result["price_cents"] == 42
        assert result["quantity"] == 7
        assert "urgency" in result
        assert "rationale" in result


# ── Portfolio Update ─────────────────────────────────────────────────────


class TestPortfolioUpdate:
    def test_update_replaces_portfolio(self) -> None:
        rm = _manager()
        p1 = _portfolio(balance_cents=1000)
        p2 = _portfolio(balance_cents=2000)
        rm.update_portfolio(p1)
        assert rm.portfolio.balance_cents == 1000
        rm.update_portfolio(p2)
        assert rm.portfolio.balance_cents == 2000


# ── Multiple Rejections ──────────────────────────────────────────────────


class TestMultipleRejections:
    def test_multiple_checks_can_fail(self) -> None:
        # Kill switch on + bad price → at least 2 rejections
        rm = _manager(cfg=_config(kill_switch=True, strategy_limits=DEFAULT_STRATEGY_LIMITS))
        decision = rm.evaluate(_order(price_cents=-1))
        rejections = [r for r in decision.results if r.verdict == RiskVerdict.REJECTED]
        assert len(rejections) >= 2
        check_names = {r.check_name for r in rejections}
        assert "kill_switch" in check_names
        assert "price_sanity" in check_names

    def test_single_rejection_blocks_order(self) -> None:
        # Only price sanity fails
        rm = _manager(portfolio=_portfolio())
        decision = rm.evaluate(_order(price_cents=100))
        assert not decision.approved
        rejections = [r for r in decision.results if r.verdict == RiskVerdict.REJECTED]
        assert len(rejections) == 1
        assert rejections[0].check_name == "price_sanity"
