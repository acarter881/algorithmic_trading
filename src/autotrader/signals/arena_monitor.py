"""Arena Leaderboard Monitor — SignalSource implementation.

Periodically fetches the LMSYS Chatbot Arena leaderboard, diffs against
the previous snapshot, and emits trading signals when rankings change.

Signal types emitted:
- ``ranking_change``  — a model's Rank(UB) moved
- ``new_leader``      — the #1 model changed
- ``score_shift``     — a model's Arena score shifted significantly
- ``new_model``       — a previously unseen model appeared
- ``model_removed``   — a model disappeared from the leaderboard
"""

from __future__ import annotations

import datetime

import httpx
import structlog

from autotrader.config.models import ArenaMonitorConfig
from autotrader.signals.arena_parser import extract_pairwise_aggregates, parse_leaderboard
from autotrader.signals.arena_types import (
    LeaderboardDiff,
    LeaderboardSnapshot,
    PairwiseChange,
    RankChange,
    ScoreChange,
)
from autotrader.signals.base import Signal, SignalSource, SignalUrgency
from autotrader.signals.settlement import resolve_top_model

logger: structlog.stdlib.BoundLogger = structlog.get_logger("autotrader.signals.arena_monitor")

# Score delta (Elo points) that triggers a ``score_shift`` signal
DEFAULT_SCORE_SHIFT_THRESHOLD = 3.0

# Target Kalshi series for leaderboard signals
TARGET_SERIES = ["KXTOPMODEL", "KXLLM1"]


class ArenaMonitor(SignalSource):
    """Monitors the LMSYS Chatbot Arena leaderboard for ranking changes.

    Implements the ``SignalSource`` plugin interface.  Each ``poll()`` call
    fetches the current leaderboard, compares it to the previous snapshot,
    and returns a list of ``Signal`` objects for any meaningful changes.
    """

    def __init__(
        self,
        config: ArenaMonitorConfig | None = None,
        score_shift_threshold: float = DEFAULT_SCORE_SHIFT_THRESHOLD,
    ) -> None:
        self._config = config or ArenaMonitorConfig()
        self._score_shift_threshold = score_shift_threshold
        self._previous_snapshot: LeaderboardSnapshot | None = None
        self._consecutive_failures = 0
        self._http_client: httpx.AsyncClient | None = None

    # ── SignalSource Interface ────────────────────────────────────────

    @property
    def name(self) -> str:
        return "arena_monitor"

    @property
    def poll_interval_seconds(self) -> int:
        return self._config.poll_interval_seconds

    async def initialize(self) -> None:
        """Create the HTTP client for fetching the leaderboard."""
        self._http_client = httpx.AsyncClient(
            timeout=self._config.request_timeout_seconds,
            follow_redirects=True,
            headers={
                "User-Agent": "KalshiAutotrader/0.1 (Arena Leaderboard Monitor)",
                "Accept": "text/html,application/json",
            },
        )
        logger.info(
            "arena_monitor_initialized",
            primary_url=self._config.primary_url,
            fallback_count=len(self._config.fallback_urls),
        )

    async def poll(self) -> list[Signal]:
        """Fetch the leaderboard, diff, and return signals."""
        snapshot = await self._fetch_leaderboard()
        if snapshot is None:
            return []

        signals = self._generate_signals(snapshot)
        self._previous_snapshot = snapshot
        return signals

    async def teardown(self) -> None:
        """Close the HTTP client."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("arena_monitor_teardown")

    # ── Fetching ──────────────────────────────────────────────────────

    async def _fetch_leaderboard(self) -> LeaderboardSnapshot | None:
        """Fetch and parse the leaderboard from primary URL with fallbacks.

        Returns ``None`` if all URLs fail.
        """
        urls = [self._config.primary_url, *self._config.fallback_urls]

        for url in urls:
            snapshot = await self._try_fetch(url)
            if snapshot is not None:
                self._consecutive_failures = 0
                return snapshot

        self._consecutive_failures += 1
        logger.warning(
            "arena_fetch_all_failed",
            consecutive_failures=self._consecutive_failures,
            max_failures=self._config.max_consecutive_failures,
        )
        return None

    async def _try_fetch(self, url: str) -> LeaderboardSnapshot | None:
        """Attempt to fetch and parse a single URL."""
        if not self._http_client:
            logger.error("arena_http_client_not_initialized")
            return None

        try:
            response = await self._http_client.get(url)
            response.raise_for_status()
            content = response.text

            entries = parse_leaderboard(content)
            if not entries:
                logger.warning("arena_no_entries_parsed", url=url)
                return None

            pairwise = extract_pairwise_aggregates(content)

            snapshot = LeaderboardSnapshot(
                entries=entries,
                pairwise=pairwise,
                source_url=url,
                captured_at=datetime.datetime.utcnow(),
            )
            logger.info(
                "arena_leaderboard_fetched",
                url=url,
                model_count=len(entries),
                top_model=snapshot.top_model,
            )
            return snapshot

        except httpx.HTTPStatusError as e:
            logger.warning("arena_http_error", url=url, status=e.response.status_code)
        except httpx.RequestError as e:
            logger.warning("arena_request_error", url=url, error=str(e))
        except Exception:
            logger.exception("arena_unexpected_error", url=url)

        return None

    # ── Diffing ───────────────────────────────────────────────────────

    def diff_snapshots(
        self,
        previous: LeaderboardSnapshot,
        current: LeaderboardSnapshot,
    ) -> LeaderboardDiff:
        """Compute the diff between two leaderboard snapshots."""
        prev_by_name = previous.by_model_name()
        curr_by_name = current.by_model_name()

        prev_names = set(prev_by_name.keys())
        curr_names = set(curr_by_name.keys())

        # New and removed models
        new_entries = [curr_by_name[n] for n in sorted(curr_names - prev_names)]
        removed_entries = [prev_by_name[n] for n in sorted(prev_names - curr_names)]

        # Rank and score changes for models present in both
        rank_changes: list[RankChange] = []
        score_changes: list[ScoreChange] = []
        pairwise_changes: list[PairwiseChange] = []

        for model_name in sorted(prev_names & curr_names):
            prev_entry = prev_by_name[model_name]
            curr_entry = curr_by_name[model_name]

            # Rank change
            if prev_entry.rank_ub != curr_entry.rank_ub or prev_entry.rank != curr_entry.rank:
                rank_changes.append(
                    RankChange(
                        model_name=model_name,
                        old_rank_ub=prev_entry.rank_ub,
                        new_rank_ub=curr_entry.rank_ub,
                        old_rank=prev_entry.rank,
                        new_rank=curr_entry.rank,
                        old_score=prev_entry.score,
                        new_score=curr_entry.score,
                    )
                )

            # Score change
            score_delta = curr_entry.score - prev_entry.score
            if abs(score_delta) >= self._score_shift_threshold:
                score_changes.append(
                    ScoreChange(
                        model_name=model_name,
                        old_score=prev_entry.score,
                        new_score=curr_entry.score,
                        score_delta=score_delta,
                    )
                )

            prev_pair = previous.pairwise.get(model_name)
            curr_pair = current.pairwise.get(model_name)
            if prev_pair and curr_pair:
                win_delta = curr_pair.average_pairwise_win_rate - prev_pair.average_pairwise_win_rate
                if abs(win_delta) >= 0.01 or curr_pair.total_pairwise_battles != prev_pair.total_pairwise_battles:
                    pairwise_changes.append(
                        PairwiseChange(
                            model_name=model_name,
                            old_average_pairwise_win_rate=prev_pair.average_pairwise_win_rate,
                            new_average_pairwise_win_rate=curr_pair.average_pairwise_win_rate,
                            old_total_pairwise_battles=prev_pair.total_pairwise_battles,
                            new_total_pairwise_battles=curr_pair.total_pairwise_battles,
                        )
                    )

        # Leader change
        prev_winner = resolve_top_model(previous.entries)
        curr_winner = resolve_top_model(current.entries)
        prev_leader = prev_winner.model_name if prev_winner else ""
        curr_leader = curr_winner.model_name if curr_winner else ""
        leader_changed = prev_leader != curr_leader and prev_leader != "" and curr_leader != ""

        return LeaderboardDiff(
            rank_changes=rank_changes,
            score_changes=score_changes,
            pairwise_changes=pairwise_changes,
            new_entries=new_entries,
            removed_entries=removed_entries,
            leader_changed=leader_changed,
            new_leader=curr_leader if leader_changed else "",
            previous_leader=prev_leader if leader_changed else "",
        )

    # ── Signal Generation ─────────────────────────────────────────────

    def _generate_signals(self, current: LeaderboardSnapshot) -> list[Signal]:
        """Generate trading signals from the current snapshot.

        On the first poll (no previous snapshot), no diff signals are
        generated — only an initial snapshot signal.
        """
        now = datetime.datetime.utcnow()
        signals: list[Signal] = []

        if self._previous_snapshot is None:
            logger.info("arena_initial_snapshot", model_count=len(current.entries))
            return signals

        diff = self.diff_snapshots(self._previous_snapshot, current)

        # Leader change — highest urgency
        if diff.leader_changed:
            signals.append(
                Signal(
                    source=self.name,
                    timestamp=now,
                    event_type="new_leader",
                    data={
                        "new_leader": diff.new_leader,
                        "previous_leader": diff.previous_leader,
                        "source_url": current.source_url,
                        "new_top_org": resolve_top_model(current.entries).organization if resolve_top_model(current.entries) else "",
                    },
                    relevant_series=TARGET_SERIES,
                    urgency=SignalUrgency.HIGH,
                )
            )

        # Rank changes
        for rc in diff.rank_changes:
            urgency = SignalUrgency.HIGH if rc.new_rank_ub == 1 or rc.old_rank_ub == 1 else SignalUrgency.MEDIUM
            signals.append(
                Signal(
                    source=self.name,
                    timestamp=now,
                    event_type="ranking_change",
                    data={
                        "model_name": rc.model_name,
                        "old_rank_ub": rc.old_rank_ub,
                        "new_rank_ub": rc.new_rank_ub,
                        "old_rank": rc.old_rank,
                        "new_rank": rc.new_rank,
                        "old_score": rc.old_score,
                        "new_score": rc.new_score,
                    },
                    relevant_series=TARGET_SERIES,
                    urgency=urgency,
                )
            )

        # Score shifts
        for sc in diff.score_changes:
            signals.append(
                Signal(
                    source=self.name,
                    timestamp=now,
                    event_type="score_shift",
                    data={
                        "model_name": sc.model_name,
                        "old_score": sc.old_score,
                        "new_score": sc.new_score,
                        "score_delta": sc.score_delta,
                    },
                    relevant_series=TARGET_SERIES,
                    urgency=SignalUrgency.MEDIUM,
                )
            )

        # Pairwise shifts
        for pc in diff.pairwise_changes:
            signals.append(
                Signal(
                    source=self.name,
                    timestamp=now,
                    event_type="pairwise_shift",
                    data={
                        "model_name": pc.model_name,
                        "old_average_pairwise_win_rate": pc.old_average_pairwise_win_rate,
                        "new_average_pairwise_win_rate": pc.new_average_pairwise_win_rate,
                        "old_total_pairwise_battles": pc.old_total_pairwise_battles,
                        "new_total_pairwise_battles": pc.new_total_pairwise_battles,
                    },
                    relevant_series=TARGET_SERIES,
                    urgency=SignalUrgency.MEDIUM,
                )
            )

        # New models
        for entry in diff.new_entries:
            urgency = SignalUrgency.HIGH if entry.rank_ub == 1 else SignalUrgency.LOW
            signals.append(
                Signal(
                    source=self.name,
                    timestamp=now,
                    event_type="new_model",
                    data={
                        "model_name": entry.model_name,
                        "organization": entry.organization,
                        "rank": entry.rank,
                        "rank_ub": entry.rank_ub,
                        "score": entry.score,
                        "votes": entry.votes,
                        "is_preliminary": entry.is_preliminary,
                    },
                    relevant_series=TARGET_SERIES,
                    urgency=urgency,
                )
            )

        # Removed models
        for entry in diff.removed_entries:
            signals.append(
                Signal(
                    source=self.name,
                    timestamp=now,
                    event_type="model_removed",
                    data={
                        "model_name": entry.model_name,
                        "last_rank_ub": entry.rank_ub,
                        "last_score": entry.score,
                    },
                    relevant_series=TARGET_SERIES,
                    urgency=SignalUrgency.LOW,
                )
            )

        if signals:
            logger.info(
                "arena_signals_generated",
                count=len(signals),
                types=[s.event_type for s in signals],
            )

        return signals

    # ── Snapshot Serialization ────────────────────────────────────────

    def snapshot_to_dict(self, snapshot: LeaderboardSnapshot) -> dict[str, list[dict[str, object]]]:
        """Serialize a snapshot for database storage (LeaderboardSnapshot.snapshot_data)."""
        return {
            "entries": [
                {
                    "model_name": e.model_name,
                    "organization": e.organization,
                    "rank": e.rank,
                    "rank_ub": e.rank_ub,
                    "rank_lb": e.rank_lb,
                    "score": e.score,
                    "ci_lower": e.ci_lower,
                    "ci_upper": e.ci_upper,
                    "votes": e.votes,
                    "is_preliminary": e.is_preliminary,
                }
                for e in snapshot.entries
            ],
            "pairwise": {
                name: {
                    "total_pairwise_battles": p.total_pairwise_battles,
                    "average_pairwise_win_rate": p.average_pairwise_win_rate,
                }
                for name, p in snapshot.pairwise.items()
            },
        }

    @property
    def previous_snapshot(self) -> LeaderboardSnapshot | None:
        """Access the last fetched snapshot (for external persistence)."""
        return self._previous_snapshot

    @previous_snapshot.setter
    def previous_snapshot(self, snapshot: LeaderboardSnapshot | None) -> None:
        self._previous_snapshot = snapshot

    @property
    def consecutive_failures(self) -> int:
        """Number of consecutive fetch failures."""
        return self._consecutive_failures
