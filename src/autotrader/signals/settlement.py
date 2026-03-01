"""Settlement helpers that mirror Kalshi TOPMODEL/LLM1 tie-break rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from autotrader.signals.arena_types import LeaderboardEntry


@dataclass(frozen=True)
class SettlementWinner:
    """Resolved winner according to Rank(UB) tie-break cascade."""

    model_name: str
    organization: str


def resolve_top_model(entries: list[LeaderboardEntry]) -> SettlementWinner | None:
    """Resolve top model by Rank(UB), Arena Score, votes, release date."""
    if not entries:
        return None
    ranked = sorted(
        entries,
        key=lambda e: (
            e.rank_ub if e.rank_ub > 0 else 10_000,
            -e.score,
            -e.votes,
            e.release_date or "9999-12-31",
            e.model_name,
        ),
    )
    best = ranked[0]
    return SettlementWinner(model_name=best.model_name, organization=best.organization)


def resolve_top_org(entries: list[LeaderboardEntry]) -> str | None:
    """Resolve winning organization from the winning model."""
    winner = resolve_top_model(entries)
    if winner is None:
        return None
    return winner.organization or None
