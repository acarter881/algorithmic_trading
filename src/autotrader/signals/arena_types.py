"""Data types for Arena leaderboard monitoring."""

from __future__ import annotations

import datetime  # noqa: TCH003
from dataclasses import dataclass, field

from autotrader.signals.settlement import resolve_top_model, resolve_top_org


@dataclass(frozen=True)
class LeaderboardEntry:
    """A single model entry from the Arena leaderboard.

    Fields correspond to columns on the LMSYS Chatbot Arena leaderboard
    (Style Control OFF view). The key resolution metric for KXTOPMODEL
    is ``rank_ub`` — the upper bound of the bootstrap confidence interval
    on rank.  A ``rank_ub`` of 1 means the model is the Arena #1.
    """

    model_name: str
    organization: str = ""
    rank: int = 0  # Display rank
    rank_ub: int = 0  # Rank upper bound (bootstrap CI) — resolution metric
    rank_lb: int = 0  # Rank lower bound (bootstrap CI)
    score: float = 0.0  # Arena score (Elo-like rating)
    ci_lower: float = 0.0  # Score confidence interval lower bound
    ci_upper: float = 0.0  # Score confidence interval upper bound
    votes: int = 0  # Number of battles / votes
    is_preliminary: bool = False  # Below vote threshold
    release_date: str = ""  # Optional model release date string used for tie-breaks


@dataclass(frozen=True)
class PairwiseAggregate:
    """Aggregated pairwise chart metrics for one model.

    These are derived from Arena pairwise plots (battle-count matrix and
    model-vs-model win-rate matrix) and can be used as additional alpha
    features near rank/tie boundaries.
    """

    model_name: str
    total_pairwise_battles: int = 0
    average_pairwise_win_rate: float = 0.0


@dataclass(frozen=True)
class RankChange:
    """A detected change in a model's ranking between snapshots."""

    model_name: str
    old_rank_ub: int
    new_rank_ub: int
    old_rank: int
    new_rank: int
    old_score: float
    new_score: float


@dataclass(frozen=True)
class ScoreChange:
    """A significant shift in a model's Arena score between snapshots."""

    model_name: str
    old_score: float
    new_score: float
    score_delta: float


@dataclass(frozen=True)
class PairwiseChange:
    """A detected shift in pairwise aggregate metrics between snapshots."""

    model_name: str
    old_average_pairwise_win_rate: float
    new_average_pairwise_win_rate: float
    old_total_pairwise_battles: int
    new_total_pairwise_battles: int


@dataclass
class LeaderboardSnapshot:
    """Complete leaderboard state at a point in time."""

    entries: list[LeaderboardEntry] = field(default_factory=list)
    pairwise: dict[str, PairwiseAggregate] = field(default_factory=dict)
    source_url: str = ""
    captured_at: datetime.datetime = field(default_factory=datetime.datetime.utcnow)

    @property
    def top_model(self) -> str | None:
        """Model resolved by settlement tie-break rules."""
        winner = resolve_top_model(self.entries)
        if winner is None:
            return None
        return winner.model_name

    @property
    def top_org(self) -> str | None:
        """Organization resolved by settlement tie-break rules."""
        return resolve_top_org(self.entries)

    def by_model_name(self) -> dict[str, LeaderboardEntry]:
        """Index entries by model name for O(1) lookup."""
        return {e.model_name: e for e in self.entries}


@dataclass(frozen=True)
class LeaderboardDiff:
    """Diff between two consecutive leaderboard snapshots."""

    rank_changes: list[RankChange] = field(default_factory=list)
    score_changes: list[ScoreChange] = field(default_factory=list)
    pairwise_changes: list[PairwiseChange] = field(default_factory=list)
    new_entries: list[LeaderboardEntry] = field(default_factory=list)
    removed_entries: list[LeaderboardEntry] = field(default_factory=list)
    leader_changed: bool = False
    new_leader: str = ""
    previous_leader: str = ""
