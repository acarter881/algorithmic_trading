"""Unit tests for preflight helpers."""

from autotrader.main import _collect_series_market_counts


class _FakeKalshiClient:
    def __init__(self, markets_by_series: dict[str, list[object]]) -> None:
        self._markets_by_series = markets_by_series
        self.calls: list[tuple[str, str]] = []

    def get_markets(self, *, series_ticker: str, status: str) -> tuple[list[object], None]:
        self.calls.append((series_ticker, status))
        return self._markets_by_series[series_ticker], None


def test_collect_series_market_counts_counts_open_markets_per_series() -> None:
    client = _FakeKalshiClient(
        {
            "KXTOPMODEL": [object(), object()],
            "KXLLM1": [object()],
        }
    )

    counts = _collect_series_market_counts(client, ["KXTOPMODEL", "KXLLM1"])

    assert counts == {"KXTOPMODEL": 2, "KXLLM1": 1}
    assert client.calls == [("KXTOPMODEL", "open"), ("KXLLM1", "open")]


def test_collect_series_market_counts_allows_zero_count_series() -> None:
    client = _FakeKalshiClient(
        {
            "KXTOPMODEL": [],
            "KXLLM1": [object()],
        }
    )

    counts = _collect_series_market_counts(client, ["KXTOPMODEL", "KXLLM1"])

    assert counts["KXTOPMODEL"] == 0
    assert counts["KXLLM1"] == 1
