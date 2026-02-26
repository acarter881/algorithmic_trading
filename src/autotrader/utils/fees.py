"""Kalshi fee calculation utilities.

Fee formula (taker): fee = 0.07 * P * (1 - P)
where P is the contract price in dollars (0.01 to 0.99).

Total fee per trade is rounded UP to the nearest cent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import ROUND_CEILING, Decimal


@dataclass(frozen=True)
class FeeResult:
    """Result of a fee calculation."""

    fee_per_contract_cents: int  # Fee per contract in cents (rounded up)
    total_fee_cents: int  # Total fee for the trade in cents (rounded up)
    fee_as_pct_of_price: float  # Fee as a percentage of the contract price
    effective_cost_cents: int  # Price + fee per contract


class FeeCalculator:
    """Calculate Kalshi trading fees.

    Standard taker fee formula: fee = 0.07 * P * (1 - P)
    where P is the contract price in dollars.

    The total fee for a trade is the per-contract fee * quantity,
    with the total rounded UP to the nearest cent.
    """

    TAKER_MULTIPLIER = 0.07
    DEFAULT_MAKER_MULTIPLIER = 0.0  # Many markets have zero maker fees

    def __init__(self, maker_multiplier: float = 0.0) -> None:
        """Initialize with optional maker fee multiplier.

        Args:
            maker_multiplier: Maker fee multiplier (0.0 for no maker fees).
        """
        self.maker_multiplier = maker_multiplier

    def _raw_fee_dollars(self, price_cents: int, multiplier: float) -> Decimal:
        """Compute the raw (unrounded) fee per contract in dollars.

        Uses Decimal for exact arithmetic to avoid floating-point rounding issues.

        Args:
            price_cents: Contract price in cents (1-99).
            multiplier: Fee multiplier (0.07 for taker standard).

        Returns:
            Raw fee in dollars (not yet rounded).
        """
        if price_cents < 1 or price_cents > 99:
            raise ValueError(f"Price must be 1-99 cents, got {price_cents}")
        p = Decimal(price_cents) / Decimal(100)
        m = Decimal(str(multiplier))
        return m * p * (1 - p)

    def taker_fee(self, price_cents: int, quantity: int) -> FeeResult:
        """Calculate taker fee for a trade.

        Args:
            price_cents: Contract price in cents (1-99).
            quantity: Number of contracts.

        Returns:
            FeeResult with per-contract and total fees.
        """
        return self._compute(price_cents, quantity, self.TAKER_MULTIPLIER)

    def maker_fee(self, price_cents: int, quantity: int) -> FeeResult:
        """Calculate maker fee for a trade.

        Args:
            price_cents: Contract price in cents (1-99).
            quantity: Number of contracts.

        Returns:
            FeeResult with per-contract and total fees.
        """
        return self._compute(price_cents, quantity, self.maker_multiplier)

    def _compute(self, price_cents: int, quantity: int, multiplier: float) -> FeeResult:
        """Core fee computation.

        Per Kalshi's fee model: the total fee for the entire trade is computed
        as (raw_fee_per_contract * quantity) and then rounded UP to the nearest cent.
        """
        if quantity < 1:
            raise ValueError(f"Quantity must be >= 1, got {quantity}")

        if multiplier == 0.0:
            return FeeResult(
                fee_per_contract_cents=0,
                total_fee_cents=0,
                fee_as_pct_of_price=0.0,
                effective_cost_cents=price_cents,
            )

        raw_per_contract = self._raw_fee_dollars(price_cents, multiplier)
        # Total fee = per_contract * quantity, rounded UP to nearest cent
        total_raw_cents = raw_per_contract * quantity * 100
        total_fee_cents = int(total_raw_cents.to_integral_value(rounding=ROUND_CEILING))

        # Per-contract fee (rounded up for display/comparison)
        per_contract_raw_cents = raw_per_contract * 100
        per_contract_cents = int(per_contract_raw_cents.to_integral_value(rounding=ROUND_CEILING))

        fee_pct = (per_contract_cents / price_cents * 100) if price_cents > 0 else 0.0

        return FeeResult(
            fee_per_contract_cents=per_contract_cents,
            total_fee_cents=total_fee_cents,
            fee_as_pct_of_price=round(fee_pct, 2),
            effective_cost_cents=price_cents + per_contract_cents,
        )

    def expected_profit_after_fees(
        self,
        fair_value_cents: int,
        market_price_cents: int,
        quantity: int,
        is_taker: bool = True,
    ) -> int:
        """Calculate expected profit after fees in cents.

        Buying YES when fair_value > market_price:
            profit = (fair_value - market_price) * quantity - fees

        Args:
            fair_value_cents: Estimated true value in cents.
            market_price_cents: Current market price in cents.
            quantity: Number of contracts.
            is_taker: Whether this is a taker trade.

        Returns:
            Expected profit in cents (can be negative).
        """
        edge_cents = fair_value_cents - market_price_cents
        gross_profit_cents = edge_cents * quantity

        if is_taker:
            fee_result = self.taker_fee(market_price_cents, quantity)
        else:
            fee_result = self.maker_fee(market_price_cents, quantity)

        return gross_profit_cents - fee_result.total_fee_cents

    def min_edge_for_profit(self, price_cents: int, quantity: int, is_taker: bool = True) -> int:
        """Calculate minimum edge in cents needed for a profitable trade.

        Args:
            price_cents: Contract price in cents.
            quantity: Number of contracts.
            is_taker: Whether this is a taker trade.

        Returns:
            Minimum edge per contract in cents (rounded up).
        """
        fee_result = self.taker_fee(price_cents, quantity) if is_taker else self.maker_fee(price_cents, quantity)

        # Need total_fee / quantity edge per contract, rounded up
        return math.ceil(fee_result.total_fee_cents / quantity)
