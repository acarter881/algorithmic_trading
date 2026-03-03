"""Sanity-check test: every Kalshi contract should map to an Arena model.

Loads a realistic snapshot of KXTOPMODEL + KXLLM1 contracts alongside a
representative Arena leaderboard and verifies that *every* contract
(excluding "Other" catch-all contracts) resolves to a model or
organization.  A mapping summary is printed so reviewers can see the
program's understanding of the market at a glance.

If this test fails, the fix is one of:
1. Add the missing name to ``OVERRIDES`` (model_name_overrides in config).
2. Adjust ``normalize_org_name`` if a new org-suffix pattern appears.
3. Lower ``fuzzy_match_threshold`` (not recommended — risks false matches).
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
# Model names use the *exact* conventions the Arena uses — hyphens,
# date suffixes, "Exp-" prefixes, etc.  Organization names include the
# suffixes the Arena appends ("DeepMind", "AI", "Labs").

ARENA_ENTRIES: list[dict[str, object]] = [
    # --- Anthropic ---
    {"name": "Claude Opus 4.6", "org": "Anthropic", "rank_ub": 1, "score": 1506.0, "votes": 35000},
    {"name": "Claude Opus 4.6 Thinking", "org": "Anthropic", "rank_ub": 2, "score": 1498.0, "votes": 28000},
    {"name": "Claude Sonnet 4.6", "org": "Anthropic", "rank_ub": 8, "score": 1445.0, "votes": 22000},
    # --- Google ---
    {"name": "Gemini-3-Pro", "org": "Google DeepMind", "rank_ub": 3, "score": 1492.0, "votes": 30000},
    {"name": "Gemini-3.1-Pro", "org": "Google DeepMind", "rank_ub": 5, "score": 1478.0, "votes": 18000},
    {"name": "Gemini-2.5-Pro-Exp-03-25", "org": "Google DeepMind", "rank_ub": 10, "score": 1430.0, "votes": 15000},
    # --- xAI ---
    {"name": "Grok-3", "org": "xAI", "rank_ub": 7, "score": 1458.0, "votes": 12000},
    {"name": "Grok-4.1-Thinking", "org": "xAI", "rank_ub": 4, "score": 1482.0, "votes": 10000},
    # --- OpenAI ---
    {"name": "GPT-5", "org": "OpenAI", "rank_ub": 6, "score": 1464.0, "votes": 32000},
    {"name": "GPT-5.1-2025-12-01", "org": "OpenAI", "rank_ub": 9, "score": 1438.0, "votes": 25000},
    {"name": "GPT-5.3-Codex", "org": "OpenAI", "rank_ub": 11, "score": 1425.0, "votes": 15000},
    {"name": "GPT-4o-2025-03-01", "org": "OpenAI", "rank_ub": 14, "score": 1395.0, "votes": 40000},
    # --- Meta ---
    {"name": "Llama-4-405B", "org": "Meta AI", "rank_ub": 12, "score": 1418.0, "votes": 14000},
    {"name": "Llama-4-Maverick", "org": "Meta AI", "rank_ub": 13, "score": 1405.0, "votes": 11000},
    # --- DeepSeek ---
    {"name": "DeepSeek-R1", "org": "DeepSeek Research", "rank_ub": 15, "score": 1388.0, "votes": 20000},
    {"name": "DeepSeek-V3", "org": "DeepSeek Research", "rank_ub": 17, "score": 1370.0, "votes": 18000},
    # --- Alibaba (Qwen) ---
    {"name": "Qwen3-Max-Preview", "org": "Alibaba", "rank_ub": 16, "score": 1375.0, "votes": 9000},
    # --- Zhipu (GLM) ---
    {"name": "GLM-4.7", "org": "Zhipu AI", "rank_ub": 18, "score": 1365.0, "votes": 7000},
    # --- Mistral ---
    {"name": "Mistral-Large-2", "org": "Mistral AI", "rank_ub": 19, "score": 1355.0, "votes": 8000},
    # --- Moonshot (Kimi) ---
    {"name": "Kimi-K2.5", "org": "Moonshot AI", "rank_ub": 20, "score": 1348.0, "votes": 5000},
    # --- Microsoft ---
    {"name": "Phi-4", "org": "Microsoft", "rank_ub": 22, "score": 1330.0, "votes": 6000},
    # --- Cohere ---
    {"name": "Command-R+", "org": "Cohere Labs", "rank_ub": 23, "score": 1320.0, "votes": 4000},
    # --- Amazon ---
    {"name": "Amazon Nova Pro", "org": "Amazon", "rank_ub": 25, "score": 1300.0, "votes": 3000},
    # --- AI21 ---
    {"name": "Jamba-1.5-Large", "org": "AI21 Labs", "rank_ub": 26, "score": 1290.0, "votes": 2000},
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


# Overrides needed for Arena names that diverge too much for fuzzy
# matching alone (date-suffixed releases, "Exp-" tags, etc.).
OVERRIDES: dict[str, str] = {
    "GPT-5.1-2025-12-01": "GPT-5.1",
    "GPT-4o-2025-03-01": "GPT-4o",
    "Gemini-2.5-Pro-Exp-03-25": "Gemini 2.5 Pro",
    "Qwen3-Max-Preview": "Qwen3 Max",
    "Grok-4.1-Thinking": "Grok 4.1 Thinking",
    "Mistral-Large-2": "Mistral Large",
    "Kimi-K2.5": "Kimi K2.5",
    "Command-R+": "Command R+",
    "Jamba-1.5-Large": "Jamba 1.5 Large",
}


class TestContractMapping:
    """Every Kalshi contract (excluding 'Other') must map to an Arena entry."""

    @pytest.fixture(autouse=True)
    async def _setup(self) -> None:
        cfg = LeaderboardAlphaConfig(
            target_series=["KXTOPMODEL", "KXLLM1"],
            model_name_overrides=OVERRIDES,
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
