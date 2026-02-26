"""Model name fuzzy matching utilities for mapping Arena leaderboard names to Kalshi tickers."""

from __future__ import annotations

from dataclasses import dataclass

from thefuzz import fuzz


@dataclass(frozen=True)
class MatchResult:
    """Result of a fuzzy match attempt."""

    query: str
    matched: str
    score: float  # 0.0 to 1.0
    is_override: bool  # True if matched via manual override table


def normalize_model_name(name: str) -> str:
    """Normalize a model name for matching.

    Lowercases, strips whitespace, normalizes separators.
    """
    return name.lower().strip().replace(" ", "-").replace("_", "-")


def fuzzy_match(
    query: str,
    candidates: list[str],
    threshold: float = 0.8,
    overrides: dict[str, str] | None = None,
) -> MatchResult | None:
    """Match a model name against a list of candidates.

    Checks manual overrides first, then falls back to fuzzy matching.

    Args:
        query: The model name to match (e.g., from Arena leaderboard).
        candidates: List of candidate names (e.g., from Kalshi tickers).
        threshold: Minimum similarity score (0.0-1.0) to accept a match.
        overrides: Manual override mapping (arena_name -> kalshi_name).

    Returns:
        MatchResult if a match is found, None otherwise.
    """
    normalized_query = normalize_model_name(query)

    # Check overrides first
    if overrides:
        for override_key, override_value in overrides.items():
            if normalize_model_name(override_key) == normalized_query:
                # Verify the override target exists in candidates
                normalized_candidates = {normalize_model_name(c): c for c in candidates}
                normalized_override = normalize_model_name(override_value)
                if normalized_override in normalized_candidates:
                    return MatchResult(
                        query=query,
                        matched=normalized_candidates[normalized_override],
                        score=1.0,
                        is_override=True,
                    )

    # Try exact match first
    normalized_candidates = {normalize_model_name(c): c for c in candidates}
    if normalized_query in normalized_candidates:
        return MatchResult(
            query=query,
            matched=normalized_candidates[normalized_query],
            score=1.0,
            is_override=False,
        )

    # Fuzzy match
    best_score = 0.0
    best_candidate = ""
    for candidate in candidates:
        # Use token_sort_ratio which handles reordering well
        score = fuzz.token_sort_ratio(normalized_query, normalize_model_name(candidate)) / 100.0
        if score > best_score:
            best_score = score
            best_candidate = candidate

    if best_score >= threshold:
        return MatchResult(
            query=query,
            matched=best_candidate,
            score=best_score,
            is_override=False,
        )

    return None
