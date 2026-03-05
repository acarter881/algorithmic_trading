"""Sanity-check test: every Kalshi contract should map to an Arena model.

Loads a realistic snapshot of KXTOPMODEL + KXLLM1 contracts alongside a
representative Arena leaderboard and verifies that *every* contract
(excluding "Other" catch-all contracts) resolves to a model or
organization.  A mapping summary is printed so reviewers can see the
program's understanding of the market at a glance.

If this test fails, the fix is one of:
1. Adjust ``normalize_org_name`` if a new org-suffix pattern appears.
2. Lower ``fuzzy_match_threshold`` (not recommended — risks false matches).
3. Verify that the Arena leaderboard names haven't diverged significantly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotrader.config.models import LeaderboardAlphaConfig
from autotrader.signals.arena_types import LeaderboardEntry
from autotrader.strategies.leaderboard_alpha import LeaderboardAlphaStrategy

FIXTURES = Path(__file__).parent.parent / "fixtures"

# ── Realistic Arena Leaderboard ──────────────────────────────────────
#
# Model names use the conventions the Arena uses — hyphens, spaces, etc.
# Organization names include suffixes the Arena appends ("DeepMind", "AI", "Labs").

ARENA_ENTRIES: list[dict[str, object]] = [
    # --- Anthropic ---
    {"name": "Claude Opus 4.6", "org": "Anthropic", "rank_ub": 1, "score": 1506.0, "votes": 35000},
    {"name": "Claude Opus 4.6 Thinking", "org": "Anthropic", "rank_ub": 2, "score": 1498.0, "votes": 28000},
    {"name": "Claude Sonnet 4.6", "org": "Anthropic", "rank_ub": 8, "score": 1445.0, "votes": 22000},
    # --- Google ---
    {"name": "Gemini-3-Pro", "org": "Google DeepMind", "rank_ub": 3, "score": 1492.0, "votes": 30000},
    {"name": "Gemini-3.1-Pro", "org": "Google DeepMind", "rank_ub": 5, "score": 1478.0, "votes": 18000},
    {"name": "Gemini 2.5 Pro", "org": "Google DeepMind", "rank_ub": 10, "score": 1430.0, "votes": 15000},
    # --- xAI ---
    {"name": "Grok-3", "org": "xAI", "rank_ub": 7, "score": 1458.0, "votes": 12000},
    {"name": "Grok 4.1 Thinking", "org": "xAI", "rank_ub": 4, "score": 1482.0, "votes": 10000},
    # --- OpenAI ---
    {"name": "GPT-5", "org": "OpenAI", "rank_ub": 6, "score": 1464.0, "votes": 32000},
    {"name": "GPT-5.1", "org": "OpenAI", "rank_ub": 9, "score": 1438.0, "votes": 25000},
    {"name": "GPT-5.3-Codex", "org": "OpenAI", "rank_ub": 11, "score": 1425.0, "votes": 15000},
    {"name": "GPT-4o", "org": "OpenAI", "rank_ub": 14, "score": 1395.0, "votes": 40000},
    # --- Meta ---
    {"name": "Llama-4-405B", "org": "Meta AI", "rank_ub": 12, "score": 1418.0, "votes": 14000},
    {"name": "Llama-4-Maverick", "org": "Meta AI", "rank_ub": 13, "score": 1405.0, "votes": 11000},
    # --- DeepSeek ---
    {"name": "DeepSeek-R1", "org": "DeepSeek Research", "rank_ub": 15, "score": 1388.0, "votes": 20000},
    {"name": "DeepSeek-V3", "org": "DeepSeek Research", "rank_ub": 17, "score": 1370.0, "votes": 18000},
    # --- Alibaba (Qwen) ---
    {"name": "Qwen3 Max", "org": "Alibaba", "rank_ub": 16, "score": 1375.0, "votes": 9000},
    # --- Zhipu (GLM) ---
    {"name": "GLM-4.7", "org": "Zhipu AI", "rank_ub": 18, "score": 1365.0, "votes": 7000},
    # --- Mistral ---
    {"name": "Mistral Large", "org": "Mistral AI", "rank_ub": 19, "score": 1355.0, "votes": 8000},
    # --- Moonshot (Kimi) ---
    {"name": "Kimi K2.5", "org": "Moonshot AI", "rank_ub": 20, "score": 1348.0, "votes": 5000},
    # --- Microsoft ---
    {"name": "Phi-4", "org": "Microsoft", "rank_ub": 22, "score": 1330.0, "votes": 6000},
    # --- Cohere ---
    {"name": "Command R+", "org": "Cohere Labs", "rank_ub": 23, "score": 1320.0, "votes": 4000},
    # --- Amazon ---
    {"name": "Amazon Nova Pro", "org": "Amazon", "rank_ub": 25, "score": 1300.0, "votes": 3000},
    # --- AI21 ---
    {"name": "Jamba 1.5 Large", "org": "AI21 Labs", "rank_ub": 26, "score": 1290.0, "votes": 2000},
]


def _build_entries() -> list[LeaderboardEntry]:
    return [
        LeaderboardEntry(
            model_name=str(e["name"]),
            organization=str(e["org"]),
            rank_ub=int(e["rank_ub"]),  # type: ignore[arg-type]
            rank=int(e["rank_ub"]),  # type: ignore[arg-type]
            score=float(e["score"]),  # type: ignore[arg-type]
            votes=int(e["votes"]),  # type: ignore[arg-type]
        )
        for e in ARENA_ENTRIES
    ]


def _load_markets() -> dict[str, object]:
    with open(FIXTURES / "kalshi_markets_snapshot.json") as f:
        return json.load(f)


class TestContractMapping:
    """Every Kalshi contract (excluding 'Other') must map to an Arena entry."""

    @pytest.fixture(autouse=True)
    async def _setup(self) -> None:
        cfg = LeaderboardAlphaConfig(
            target_series=["KXTOPMODEL", "KXLLM1"],
        )
        self.strategy = LeaderboardAlphaStrategy(config=cfg)
        market_data = _load_markets()
        await self.strategy.initialize(market_data, None)
        self.entries = _build_entries()
        self.strategy.seed_rankings(self.entries)

        # seed_rankings only eagerly resolves the top 10.  For a full
        # mapping sanity check, resolve every Arena entry explicitly.
        for entry in self.entries:
            self.strategy._resolve_ticker(entry.model_name, series="KXTOPMODEL")
            if entry.organization:
                self.strategy._resolve_ticker(entry.organization, series="KXLLM1")

    def test_all_model_contracts_mapped(self) -> None:
        """Every KXTOPMODEL contract resolves to an Arena model."""
        unmapped = []
        for ticker, contract in self.strategy.contracts.items():
            if not ticker.startswith("KXTOPMODEL"):
                continue
            if contract.model_name == "Other":
                continue
            if ticker not in self.strategy.model_ticker_map.values():
                unmapped.append(f"  {ticker}  subtitle={contract.model_name!r}")

        if unmapped:
            pytest.fail("KXTOPMODEL contracts not mapped to any Arena model:\n" + "\n".join(unmapped))

    def test_all_org_contracts_mapped(self) -> None:
        """Every KXLLM1 contract resolves to an Arena organization."""
        unmapped = []
        for ticker, contract in self.strategy.contracts.items():
            if not ticker.startswith("KXLLM1"):
                continue
            if contract.model_name == "Other":
                continue
            if ticker not in self.strategy._org_ticker_map.values():
                unmapped.append(f"  {ticker}  subtitle={contract.model_name!r}")

        if unmapped:
            pytest.fail("KXLLM1 contracts not mapped to any Arena org:\n" + "\n".join(unmapped))

    def test_mapping_summary(self) -> None:
        """Print a full mapping summary for review.

        This test always passes — its purpose is to make the mapping
        visible in test output (run with ``-s`` or ``--tb=short``).
        """
        lines = ["\n=== Contract ↔ Arena Mapping Summary ===\n"]

        lines.append("── KXTOPMODEL (model → contract) ──")
        model_map = self.strategy.model_ticker_map
        for arena_name, ticker in sorted(model_map.items(), key=lambda x: x[1]):
            contract = self.strategy.contracts.get(ticker)
            subtitle = contract.model_name if contract else "?"
            lines.append(f"  Arena: {arena_name!r:40s} → {ticker}  (subtitle: {subtitle!r})")

        # Show unmapped KXTOPMODEL contracts
        mapped_tickers = set(model_map.values())
        for ticker, contract in sorted(self.strategy.contracts.items()):
            if ticker.startswith("KXTOPMODEL") and ticker not in mapped_tickers:
                lines.append(f"  UNMAPPED contract: {ticker}  (subtitle: {contract.model_name!r})")

        lines.append("\n── KXLLM1 (organization → contract) ──")
        org_map = self.strategy._org_ticker_map
        for arena_org, ticker in sorted(org_map.items(), key=lambda x: x[1]):
            contract = self.strategy.contracts.get(ticker)
            subtitle = contract.model_name if contract else "?"
            lines.append(f"  Arena: {arena_org!r:40s} → {ticker}  (subtitle: {subtitle!r})")

        # Show unmapped KXLLM1 contracts
        mapped_org_tickers = set(org_map.values())
        for ticker, contract in sorted(self.strategy.contracts.items()):
            if ticker.startswith("KXLLM1") and ticker not in mapped_org_tickers:
                lines.append(f"  UNMAPPED contract: {ticker}  (subtitle: {contract.model_name!r})")

        lines.append(f"\nTotals: {len(model_map)} model mappings, {len(org_map)} org mappings")

        total_contracts = len(self.strategy.contracts)
        other_count = sum(1 for c in self.strategy.contracts.values() if c.model_name == "Other")
        mapped_count = len(mapped_tickers | mapped_org_tickers)
        lines.append(
            f"Coverage: {mapped_count}/{total_contracts - other_count} contracts mapped "
            f"({other_count} 'Other' contracts excluded)"
        )
        lines.append("")

        print("\n".join(lines))

    def test_no_duplicate_mappings(self) -> None:
        """Two different Arena models should not map to the same contract."""
        seen: dict[str, str] = {}
        for arena_name, ticker in self.strategy.model_ticker_map.items():
            if ticker in seen:
                pytest.fail(f"Duplicate mapping: both {seen[ticker]!r} and {arena_name!r} map to {ticker}")
            seen[ticker] = arena_name

    def test_contract_count_matches_fixture(self) -> None:
        """Verify we loaded the expected number of contracts."""
        market_data = _load_markets()
        expected = len(market_data["markets"])  # type: ignore[arg-type]
        actual = len(self.strategy.contracts)
        assert actual == expected, f"Expected {expected} contracts, got {actual}"


@pytest.mark.asyncio
async def test_duplicate_model_name_prefers_configured_event() -> None:
    cfg = LeaderboardAlphaConfig(
        target_series=["KXTOPMODEL"],
        preferred_event_tickers=["KXTOPMODEL-26MAR"],
        mapping_event_selection="any",
    )
    strategy = LeaderboardAlphaStrategy(config=cfg)
    await strategy.initialize(
        {
            "markets": [
                {
                    "ticker": "KXTOPMODEL-20APR-GPT5",
                    "event_ticker": "KXTOPMODEL-20APR",
                    "subtitle": "GPT-5",
                    "title": "GPT-5 will be the #1 AI model",
                    "yes_bid": 40,
                    "yes_ask": 45,
                    "last_price": 43,
                },
                {
                    "ticker": "KXTOPMODEL-26MAR-GPT5",
                    "event_ticker": "KXTOPMODEL-26MAR",
                    "subtitle": "GPT-5",
                    "title": "GPT-5 will be the #1 AI model",
                    "yes_bid": 41,
                    "yes_ask": 46,
                    "last_price": 44,
                },
            ]
        },
        None,
    )

    resolved = strategy._resolve_ticker("GPT-5", series="KXTOPMODEL")
    assert resolved == "KXTOPMODEL-26MAR-GPT5"


@pytest.mark.asyncio
async def test_duplicate_org_name_uses_nearest_expiration_then_ticker() -> None:
    cfg = LeaderboardAlphaConfig(
        target_series=["KXLLM1"],
        mapping_event_selection="nearest_expiration",
    )
    strategy = LeaderboardAlphaStrategy(config=cfg)
    await strategy.initialize(
        {
            "markets": [
                {
                    "ticker": "KXLLM1-31DEC-OPENAI",
                    "event_ticker": "KXLLM1-31DEC",
                    "subtitle": "OpenAI",
                    "title": "OpenAI will have the #1 AI model",
                    "close_time": "2030-12-31T00:00:00Z",
                    "yes_bid": 20,
                    "yes_ask": 25,
                    "last_price": 22,
                },
                {
                    "ticker": "KXLLM1-26MAR-OPENAI",
                    "event_ticker": "KXLLM1-26MAR",
                    "subtitle": "OpenAI",
                    "title": "OpenAI will have the #1 AI model",
                    "close_time": "2030-03-26T00:00:00Z",
                    "yes_bid": 21,
                    "yes_ask": 26,
                    "last_price": 23,
                },
            ]
        },
        None,
    )

    resolved = strategy._resolve_ticker("OpenAI", series="KXLLM1")
    assert resolved == "KXLLM1-26MAR-OPENAI"


@pytest.mark.asyncio
async def test_duplicate_name_lexicographic_tiebreaker_is_stable() -> None:
    cfg = LeaderboardAlphaConfig(
        target_series=["KXTOPMODEL"],
        mapping_event_selection="any",
    )
    strategy = LeaderboardAlphaStrategy(config=cfg)
    await strategy.initialize(
        {
            "markets": [
                {
                    "ticker": "KXTOPMODEL-26MAR-GPT5B",
                    "event_ticker": "KXTOPMODEL-26MAR",
                    "subtitle": "GPT-5",
                    "title": "GPT-5 will be the #1 AI model",
                    "close_time": "2030-03-26T00:00:00Z",
                    "yes_bid": 41,
                    "yes_ask": 46,
                    "last_price": 44,
                },
                {
                    "ticker": "KXTOPMODEL-26MAR-GPT5A",
                    "event_ticker": "KXTOPMODEL-26MAR",
                    "subtitle": "GPT-5",
                    "title": "GPT-5 will be the #1 AI model",
                    "close_time": "2030-03-26T00:00:00Z",
                    "yes_bid": 41,
                    "yes_ask": 46,
                    "last_price": 44,
                },
            ]
        },
        None,
    )

    first = strategy._resolve_ticker("GPT-5", series="KXTOPMODEL")
    strategy._model_ticker_map.clear()
    second = strategy._resolve_ticker("GPT-5", series="KXTOPMODEL")
    assert first == "KXTOPMODEL-26MAR-GPT5A"
    assert second == first


@pytest.mark.asyncio
async def test_duplicate_name_parses_event_ticker_yy_suffix_without_close_time() -> None:
    cfg = LeaderboardAlphaConfig(
        target_series=["KXTOPMODEL"],
        mapping_event_selection="nearest_expiration",
    )
    strategy = LeaderboardAlphaStrategy(config=cfg)
    await strategy.initialize(
        {
            "markets": [
                {
                    "ticker": "KXTOPMODEL-26FEB28-GPT5",
                    "event_ticker": "KXTOPMODEL-26FEB28",
                    "subtitle": "GPT-5",
                    "title": "GPT-5 will be the #1 AI model",
                    "yes_bid": 41,
                    "yes_ask": 46,
                    "last_price": 44,
                },
                {
                    "ticker": "KXTOPMODEL-26MAR07-GPT5",
                    "event_ticker": "KXTOPMODEL-26MAR07",
                    "subtitle": "GPT-5",
                    "title": "GPT-5 will be the #1 AI model",
                    "yes_bid": 42,
                    "yes_ask": 47,
                    "last_price": 45,
                },
            ]
        },
        None,
    )

    resolved = strategy._resolve_ticker("GPT-5", series="KXTOPMODEL")
    assert resolved == "KXTOPMODEL-26FEB28-GPT5"
