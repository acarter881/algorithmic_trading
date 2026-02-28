from autotrader.signals.arena_types import LeaderboardEntry
from autotrader.signals.settlement import resolve_top_model, resolve_top_org


def test_resolve_top_model_uses_tie_break_chain() -> None:
    entries = [
        LeaderboardEntry(
            model_name="A", organization="OrgA", rank_ub=1, score=1500, votes=10_000, release_date="2025-02-01"
        ),
        LeaderboardEntry(
            model_name="B", organization="OrgB", rank_ub=1, score=1500, votes=12_000, release_date="2025-03-01"
        ),
    ]
    winner = resolve_top_model(entries)
    assert winner is not None
    assert winner.model_name == "B"


def test_resolve_top_org() -> None:
    entries = [LeaderboardEntry(model_name="A", organization="OrgA", rank_ub=1, score=1500, votes=10)]
    assert resolve_top_org(entries) == "OrgA"
