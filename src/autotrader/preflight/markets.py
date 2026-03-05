"""Preflight checks related to Kalshi market availability."""

from __future__ import annotations

from typing import Any


def validate_target_series_markets(client: Any, target_series: list[str]) -> tuple[dict[str, int], list[str]]:
    """Return open-market counts for each target series and missing series list."""
    series_market_counts: dict[str, int] = {}
    for series in target_series:
        total = 0
        cursor: str | None = None
        while True:
            markets, cursor = client.get_markets(series_ticker=series, status="open", limit=200, cursor=cursor)
            total += len(markets)
            if not cursor or not markets:
                break
        series_market_counts[series] = total

    missing_series = [series for series, count in series_market_counts.items() if count < 1]
    return series_market_counts, missing_series
