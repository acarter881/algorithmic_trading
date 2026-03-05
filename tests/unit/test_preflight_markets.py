"""Unit tests for preflight market validation helpers."""

from unittest.mock import MagicMock

from autotrader.preflight.markets import validate_target_series_markets


def test_validate_target_series_markets_counts_all_pages() -> None:
    client = MagicMock()
    client.get_markets.side_effect = [
        ([{"ticker": "A"}], "next"),
        ([{"ticker": "B"}], None),
        ([{"ticker": "C"}], None),
    ]

    counts, missing = validate_target_series_markets(client, ["KXTOPMODEL", "KXLLM1"])

    assert counts == {"KXTOPMODEL": 2, "KXLLM1": 1}
    assert missing == []


def test_validate_target_series_markets_flags_missing_series() -> None:
    client = MagicMock()
    client.get_markets.side_effect = [
        ([], None),
        ([{"ticker": "C"}], None),
    ]

    counts, missing = validate_target_series_markets(client, ["KXTOPMODEL", "KXLLM1"])

    assert counts == {"KXTOPMODEL": 0, "KXLLM1": 1}
    assert missing == ["KXTOPMODEL"]
