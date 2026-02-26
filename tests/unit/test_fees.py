"""Unit tests for the FeeCalculator."""

import pytest

from autotrader.utils.fees import FeeCalculator


@pytest.fixture
def calc() -> FeeCalculator:
    return FeeCalculator()


class TestTakerFees:
    """Test taker fee calculations against known expected values from the spec."""

    def test_fee_at_5_cents(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(5, 1)
        # 0.07 * 0.05 * 0.95 = 0.003325 → ceil to 1 cent
        assert result.fee_per_contract_cents == 1
        assert result.total_fee_cents == 1

    def test_fee_at_10_cents(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(10, 1)
        # 0.07 * 0.10 * 0.90 = 0.0063 → ceil to 1 cent
        assert result.fee_per_contract_cents == 1
        assert result.total_fee_cents == 1

    def test_fee_at_25_cents(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(25, 1)
        # 0.07 * 0.25 * 0.75 = 0.013125 → ceil to 2 cents
        assert result.fee_per_contract_cents == 2
        assert result.total_fee_cents == 2

    def test_fee_at_50_cents(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(50, 1)
        # 0.07 * 0.50 * 0.50 = 0.0175 → ceil to 2 cents
        assert result.fee_per_contract_cents == 2
        assert result.total_fee_cents == 2

    def test_fee_at_75_cents(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(75, 1)
        # 0.07 * 0.75 * 0.25 = 0.013125 → ceil to 2 cents
        assert result.fee_per_contract_cents == 2
        assert result.total_fee_cents == 2

    def test_fee_at_90_cents(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(90, 1)
        # 0.07 * 0.90 * 0.10 = 0.0063 → ceil to 1 cent
        assert result.fee_per_contract_cents == 1
        assert result.total_fee_cents == 1

    def test_fee_at_95_cents(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(95, 1)
        # 0.07 * 0.95 * 0.05 = 0.003325 → ceil to 1 cent
        assert result.fee_per_contract_cents == 1
        assert result.total_fee_cents == 1

    def test_fee_symmetric(self, calc: FeeCalculator) -> None:
        """Fee formula is symmetric: fee(P) == fee(1-P)."""
        for p in range(1, 50):
            result_low = calc.taker_fee(p, 1)
            result_high = calc.taker_fee(100 - p, 1)
            assert result_low.fee_per_contract_cents == result_high.fee_per_contract_cents

    def test_fee_max_at_50_cents(self, calc: FeeCalculator) -> None:
        """Maximum fee occurs at 50 cents."""
        raw_50 = 0.07 * 0.50 * 0.50
        for p in range(1, 100):
            raw_p = 0.07 * (p / 100.0) * (1 - p / 100.0)
            assert raw_p <= raw_50 + 1e-10


class TestMultiContractFees:
    """Test fee calculations with multiple contracts — total rounding behavior."""

    def test_single_vs_batch_rounding(self, calc: FeeCalculator) -> None:
        """Total fee for N contracts is ceil(raw * N * 100), not N * ceil(raw * 100).

        This means batch orders can have lower total fees than sum of singles.
        """
        # At 5 cents: raw = 0.003325 per contract
        single = calc.taker_fee(5, 1)
        batch_10 = calc.taker_fee(5, 10)

        # 10 singles = 10 * 1 cent = 10 cents
        assert single.total_fee_cents * 10 == 10
        # Batch of 10 = ceil(0.003325 * 10 * 100) = ceil(3.325) = 4 cents
        assert batch_10.total_fee_cents == 4

    def test_batch_5_at_10_cents(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(10, 5)
        # 0.07 * 0.10 * 0.90 * 5 = 0.0315 → ceil(3.15) = 4 cents
        assert result.total_fee_cents == 4

    def test_batch_100_at_50_cents(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(50, 100)
        # 0.07 * 0.50 * 0.50 * 100 = 1.75 → ceil(175) = 175 cents
        assert result.total_fee_cents == 175

    def test_batch_3_at_1_cent(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(1, 3)
        # 0.07 * 0.01 * 0.99 * 3 = 0.002079 → ceil(0.2079) = 1 cent
        assert result.total_fee_cents == 1

    def test_single_at_1_cent_extreme(self, calc: FeeCalculator) -> None:
        """At 1 cent, fee rounds up to 1 cent = 100% of price."""
        result = calc.taker_fee(1, 1)
        # 0.07 * 0.01 * 0.99 = 0.000693 → ceil to 1 cent
        assert result.fee_per_contract_cents == 1
        assert result.total_fee_cents == 1

    def test_single_at_99_cents_extreme(self, calc: FeeCalculator) -> None:
        """At 99 cents, fee rounds up to 1 cent."""
        result = calc.taker_fee(99, 1)
        assert result.fee_per_contract_cents == 1
        assert result.total_fee_cents == 1


class TestMakerFees:
    """Test maker fee calculations."""

    def test_zero_maker_fee_default(self, calc: FeeCalculator) -> None:
        """Default maker multiplier is 0, so fees are always 0."""
        result = calc.maker_fee(50, 10)
        assert result.fee_per_contract_cents == 0
        assert result.total_fee_cents == 0

    def test_custom_maker_multiplier(self) -> None:
        """With a custom maker multiplier, fees should be proportionally lower."""
        calc = FeeCalculator(maker_multiplier=0.03)
        result = calc.maker_fee(50, 1)
        # 0.03 * 0.50 * 0.50 = 0.0075 → ceil to 1 cent
        assert result.fee_per_contract_cents == 1
        assert result.total_fee_cents == 1


class TestFeeResult:
    """Test FeeResult properties."""

    def test_effective_cost(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(50, 1)
        assert result.effective_cost_cents == 50 + result.fee_per_contract_cents

    def test_fee_pct_at_low_price(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(1, 1)
        # 1 cent fee on 1 cent price = 100%
        assert result.fee_as_pct_of_price == 100.0

    def test_fee_pct_at_high_price(self, calc: FeeCalculator) -> None:
        result = calc.taker_fee(90, 1)
        # 1 cent fee on 90 cent price ≈ 1.11%
        assert result.fee_as_pct_of_price == pytest.approx(1.11, abs=0.01)


class TestEdgeCases:
    """Test edge cases and input validation."""

    def test_price_below_1(self, calc: FeeCalculator) -> None:
        with pytest.raises(ValueError, match="Price must be 1-99"):
            calc.taker_fee(0, 1)

    def test_price_above_99(self, calc: FeeCalculator) -> None:
        with pytest.raises(ValueError, match="Price must be 1-99"):
            calc.taker_fee(100, 1)

    def test_quantity_zero(self, calc: FeeCalculator) -> None:
        with pytest.raises(ValueError, match="Quantity must be >= 1"):
            calc.taker_fee(50, 0)

    def test_negative_price(self, calc: FeeCalculator) -> None:
        with pytest.raises(ValueError):
            calc.taker_fee(-5, 1)

    def test_negative_quantity(self, calc: FeeCalculator) -> None:
        with pytest.raises(ValueError):
            calc.taker_fee(50, -1)


class TestExpectedProfit:
    """Test expected profit calculations."""

    def test_profitable_trade(self, calc: FeeCalculator) -> None:
        # Fair value 60, market 50, buy 10 contracts
        # Gross profit: (60-50)*10 = 100 cents
        # Fee: ceil(0.07*0.50*0.50*10*100) = ceil(17.5) = 18 cents
        profit = calc.expected_profit_after_fees(60, 50, 10, is_taker=True)
        assert profit == 100 - 18

    def test_unprofitable_after_fees(self, calc: FeeCalculator) -> None:
        # Fair value 51, market 50, buy 1 contract
        # Gross: 1 cent, fee: 2 cents → net loss
        profit = calc.expected_profit_after_fees(51, 50, 1, is_taker=True)
        assert profit < 0

    def test_maker_profitable(self) -> None:
        calc = FeeCalculator(maker_multiplier=0.0)
        # Fair value 55, market 50, 10 contracts, maker (no fees)
        profit = calc.expected_profit_after_fees(55, 50, 10, is_taker=False)
        assert profit == 50  # Pure edge, no fees


class TestMinEdge:
    """Test minimum edge calculation."""

    def test_min_edge_at_50_cents_single(self, calc: FeeCalculator) -> None:
        edge = calc.min_edge_for_profit(50, 1, is_taker=True)
        # Fee at 50 cents = 2 cents for 1 contract
        assert edge == 2

    def test_min_edge_at_50_cents_batch(self, calc: FeeCalculator) -> None:
        edge = calc.min_edge_for_profit(50, 100, is_taker=True)
        # Fee for 100 at 50 = 175 cents total → 175/100 = 1.75 → ceil = 2
        assert edge == 2

    def test_min_edge_maker_no_fees(self, calc: FeeCalculator) -> None:
        edge = calc.min_edge_for_profit(50, 1, is_taker=False)
        assert edge == 0
