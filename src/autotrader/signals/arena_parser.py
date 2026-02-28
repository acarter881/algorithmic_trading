"""Parser for the LMSYS Chatbot Arena leaderboard.

Supports multiple extraction strategies to handle the Arena site's
data format, which may change over time:

1. CSV — if the response is CSV-formatted data
2. JSON API response — structured JSON from an API endpoint
3. ``__NEXT_DATA__`` — JSON blob embedded by Next.js in the page HTML
4. HTML table fallback — scrape the rendered ``<table>`` element

The parser normalises all formats into a list of ``LeaderboardEntry`` objects.

Note: The lmarena.ai site renders via JavaScript (Next.js), so the HTML
table strategy only works if the page is server-side rendered.
"""

from __future__ import annotations

import csv
import io
import json
import re
from typing import Any

import structlog
from bs4 import BeautifulSoup, Tag

from autotrader.signals.arena_types import LeaderboardEntry, PairwiseAggregate

logger: structlog.stdlib.BoundLogger = structlog.get_logger("autotrader.signals.arena_parser")

# Minimum votes before a model is considered "established" (non-preliminary)
PRELIMINARY_VOTE_THRESHOLD = 500

# Known field name mappings for JSON data (Arena API field → our field)
_SCORE_KEYS = ("arena_score", "score", "elo", "rating", "elo_rating")
_RANK_KEYS = ("rank",)
_RANK_UB_KEYS = ("rank_ub", "rank_upper", "upper_bound", "rank_ci_upper")
_RANK_LB_KEYS = ("rank_lb", "rank_lower", "lower_bound", "rank_ci_lower")
_CI_LOWER_KEYS = ("ci_lower", "lower_ci", "ci_low", "rating_q025")
_CI_UPPER_KEYS = ("ci_upper", "upper_ci", "ci_high", "rating_q975")
_VOTES_KEYS = ("votes", "num_battles", "total_votes", "num_votes")
_ORG_KEYS = ("organization", "org", "provider", "developer")
_NAME_KEYS = ("model_name", "model", "name", "full_name", "key")
_RELEASE_KEYS = ("release_date", "released_at", "release_time", "launch_date")


def _get_first(data: dict[str, Any], keys: tuple[str, ...], default: Any = None) -> Any:
    """Return the value for the first matching key found in *data*."""
    for key in keys:
        if key in data:
            return data[key]
    return default


def parse_json_entries(data: list[dict[str, Any]]) -> list[LeaderboardEntry]:
    """Parse a list of JSON model objects into ``LeaderboardEntry`` instances.

    Handles varying key names across different Arena API versions.
    """
    entries: list[LeaderboardEntry] = []
    for row in data:
        name = str(_get_first(row, _NAME_KEYS, ""))
        if not name:
            continue

        votes = _safe_int(_get_first(row, _VOTES_KEYS, 0))
        score = _safe_float(_get_first(row, _SCORE_KEYS, 0.0))
        rank = _safe_int(_get_first(row, _RANK_KEYS, 0))
        rank_ub = _safe_int(_get_first(row, _RANK_UB_KEYS, rank))
        rank_lb = _safe_int(_get_first(row, _RANK_LB_KEYS, rank))

        entries.append(
            LeaderboardEntry(
                model_name=name,
                organization=str(_get_first(row, _ORG_KEYS, "")),
                rank=rank,
                rank_ub=rank_ub,
                rank_lb=rank_lb,
                score=score,
                ci_lower=_safe_float(_get_first(row, _CI_LOWER_KEYS, 0.0)),
                ci_upper=_safe_float(_get_first(row, _CI_UPPER_KEYS, 0.0)),
                votes=votes,
                is_preliminary=votes < PRELIMINARY_VOTE_THRESHOLD,
                release_date=str(_get_first(row, _RELEASE_KEYS, "")),
            )
        )

    logger.debug("parsed_json_entries", count=len(entries))
    return entries


def extract_next_data(html: str) -> list[dict[str, Any]] | None:
    """Extract leaderboard data from a Next.js ``__NEXT_DATA__`` script tag.

    Returns ``None`` if the tag is absent or does not contain leaderboard data.
    """
    soup = BeautifulSoup(html, "html.parser")
    script_tag = soup.find("script", id="__NEXT_DATA__")
    if not isinstance(script_tag, Tag) or not script_tag.string:
        return None

    try:
        payload = json.loads(script_tag.string)
    except json.JSONDecodeError:
        logger.warning("next_data_invalid_json")
        return None

    # Walk common paths where leaderboard data may live
    return _find_leaderboard_array(payload)


def _find_leaderboard_array(obj: Any, depth: int = 0) -> list[dict[str, Any]] | None:
    """Recursively search a JSON tree for an array that looks like leaderboard data."""
    if depth > 8:
        return None

    if isinstance(obj, list) and len(obj) > 5 and all(isinstance(item, dict) for item in obj[:3]):
        # Heuristic: an array of dicts with a "model_name" or "model" key
        sample = obj[0]
        if any(k in sample for k in ("model_name", "model", "name", "key", "full_name")):
            return obj

    if isinstance(obj, dict):
        # Check well-known Next.js paths first
        for path in ("props.pageProps.leaderboardData", "props.pageProps.data", "props.pageProps.models"):
            result = _walk_path(obj, path)
            if isinstance(result, list) and len(result) > 0:
                return result
        # Fallback: recurse into all values
        for value in obj.values():
            found = _find_leaderboard_array(value, depth + 1)
            if found is not None:
                return found

    return None


def _walk_path(obj: dict[str, Any], dotted_path: str) -> Any:
    """Walk a dotted path like ``props.pageProps.data`` into nested dicts."""
    current: Any = obj
    for key in dotted_path.split("."):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return None
    return current


def parse_html_table(html: str) -> list[LeaderboardEntry]:
    """Parse leaderboard entries from an HTML table in the page.

    Falls back to table-based extraction when JSON extraction fails.
    Handles various column naming conventions.
    """
    soup = BeautifulSoup(html, "html.parser")
    tables = soup.find_all("table")
    if not tables:
        logger.warning("no_html_tables_found")
        return []

    # Pick the largest table (most likely the leaderboard)
    table = max(tables, key=lambda t: len(t.find_all("tr")))

    # Extract header columns
    headers = _extract_headers(table)
    if not headers:
        logger.warning("no_table_headers_found")
        return []

    # Map header text to column indices
    col_map = _build_column_map(headers)
    if "name" not in col_map:
        logger.warning("no_model_name_column", headers=headers)
        return []

    # Parse rows
    entries: list[LeaderboardEntry] = []
    rows = table.find_all("tr")
    for row in rows[1:]:  # Skip header row
        cells = row.find_all(["td", "th"])
        if len(cells) < 2:
            continue

        cell_texts = [_clean_cell(c) for c in cells]
        name = cell_texts[col_map["name"]] if col_map["name"] < len(cell_texts) else ""
        if not name:
            continue

        rank = _safe_int(cell_texts[col_map["rank"]]) if "rank" in col_map else 0
        rank_ub, rank_lb = _parse_rank_range(
            cell_texts[col_map.get("rank", col_map.get("rank_ub", -1))]
            if "rank" in col_map or "rank_ub" in col_map
            else ""
        )
        if rank_ub == 0:
            rank_ub = rank
        if rank_lb == 0:
            rank_lb = rank

        score = _safe_float(cell_texts[col_map["score"]]) if "score" in col_map else 0.0
        ci_lower, ci_upper = _parse_ci(cell_texts[col_map["ci"]]) if "ci" in col_map else (0.0, 0.0)
        votes = _safe_int(cell_texts[col_map["votes"]]) if "votes" in col_map else 0
        org = cell_texts[col_map["org"]] if "org" in col_map else ""

        entries.append(
            LeaderboardEntry(
                model_name=name,
                organization=org,
                rank=rank,
                rank_ub=rank_ub,
                rank_lb=rank_lb,
                score=score,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                votes=votes,
                is_preliminary=votes < PRELIMINARY_VOTE_THRESHOLD if votes > 0 else False,
                release_date="",
            )
        )

    logger.debug("parsed_html_table", count=len(entries))
    return entries


def parse_csv(content: str) -> list[LeaderboardEntry]:
    """Parse leaderboard entries from CSV content.

    Supports CSV with columns like::

        rank,rank_stylectrl,model,arena_score,95_pct_ci,votes,organization,...

    The ``rank_stylectrl`` column corresponds to Rank(UB) — the upper bound
    of the bootstrap confidence interval on rank (with Style Control OFF,
    this equals the ``final_ranking`` field in the source pickle data).
    """
    entries: list[LeaderboardEntry] = []

    reader = csv.DictReader(io.StringIO(content))
    if not reader.fieldnames:
        return []

    # Normalise header names (lowercase, strip)
    field_map = {f.lower().strip(): f for f in reader.fieldnames}

    for row in reader:
        # Resolve model name
        name = ""
        for key in ("model", "model_name", "name"):
            if key in field_map:
                name = row.get(field_map[key], "").strip()
                if name:
                    break
        if not name:
            continue

        # Resolve rank fields — rank_stylectrl is the Rank(UB) we need
        rank = _safe_int(row.get(field_map.get("rank", "rank"), ""))
        rank_ub = rank  # default to display rank
        for key in ("rank_stylectrl", "rank_ub", "final_ranking"):
            if key in field_map:
                val = _safe_int(row.get(field_map[key], ""))
                if val > 0:
                    rank_ub = val
                    break

        # Resolve score
        score = 0.0
        for key in ("arena_score", "score", "elo", "rating"):
            if key in field_map:
                score = _safe_float(row.get(field_map[key], ""))
                if score > 0:
                    break

        # Parse CI from "95_pct_ci" or similar (format: "+5/-5")
        ci_lower, ci_upper = 0.0, 0.0
        for key in ("95_pct_ci", "95% ci", "ci"):
            if key in field_map:
                ci_lower, ci_upper = _parse_ci(row.get(field_map[key], ""))
                break

        # Votes
        votes = 0
        for key in ("votes", "num_battles"):
            if key in field_map:
                votes = _safe_int(row.get(field_map[key], ""))
                if votes > 0:
                    break

        # Organization
        org = ""
        for key in ("organization", "org", "provider"):
            if key in field_map:
                org = row.get(field_map[key], "").strip()
                if org:
                    break

        entries.append(
            LeaderboardEntry(
                model_name=name,
                organization=org,
                rank=rank,
                rank_ub=rank_ub,
                rank_lb=rank,  # CSV doesn't have separate lower bound
                score=score,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                votes=votes,
                is_preliminary=votes < PRELIMINARY_VOTE_THRESHOLD if votes > 0 else False,
                release_date="",
            )
        )

    logger.debug("parsed_csv_entries", count=len(entries))
    return entries


def parse_leaderboard(html_or_json: str) -> list[LeaderboardEntry]:
    """Parse leaderboard data from CSV, JSON, or HTML content.

    Tries strategies in order: CSV, JSON, ``__NEXT_DATA__``, HTML table.
    """
    # Strategy 1: CSV
    if _looks_like_csv(html_or_json):
        entries = parse_csv(html_or_json)
        if entries:
            return entries

    # Strategy 2: Raw JSON response
    try:
        data = json.loads(html_or_json)
        if isinstance(data, list):
            entries = parse_json_entries(data)
            if entries:
                return entries
        if isinstance(data, dict):
            # Look for data nested inside a wrapper
            for key in ("data", "leaderboard", "models", "results", "rows"):
                if key in data and isinstance(data[key], list):
                    entries = parse_json_entries(data[key])
                    if entries:
                        return entries
    except (json.JSONDecodeError, ValueError):
        pass  # Not JSON, try HTML strategies

    # Strategy 3: __NEXT_DATA__ embedded JSON
    next_data = extract_next_data(html_or_json)
    if next_data:
        entries = parse_json_entries(next_data)
        if entries:
            return entries

    # Strategy 4: HTML table
    return parse_html_table(html_or_json)


def extract_pairwise_aggregates(payload: Any) -> dict[str, PairwiseAggregate]:
    """Extract pairwise-derived aggregates from Arena payloads.

    The Arena site can publish chart data in varying shapes. This helper uses
    conservative heuristics and returns an empty map when no compatible data is
    found.
    """
    if isinstance(payload, str):
        with_payload: Any = None
        try:
            with_payload = json.loads(payload)
        except (json.JSONDecodeError, ValueError):
            next_data = extract_next_data(payload)
            if next_data:
                with_payload = {"leaderboardData": next_data}
        payload = with_payload if with_payload is not None else payload

    data = _find_pairwise_data(payload)
    if not data:
        return {}

    labels = [str(x) for x in data.get("labels", []) if str(x)]
    battle = data.get("battle_matrix", [])
    win = data.get("win_matrix", [])
    if not labels or len(battle) != len(labels) or len(win) != len(labels):
        return {}

    aggregates: dict[str, PairwiseAggregate] = {}
    for i, model in enumerate(labels):
        battle_row = battle[i] if isinstance(battle[i], list) else []
        win_row = win[i] if isinstance(win[i], list) else []
        total_battles = 0
        win_weighted_sum = 0.0

        for j, battles in enumerate(battle_row):
            if i == j:
                continue
            battle_count = _safe_int(battles)
            win_rate = _safe_float(win_row[j] if j < len(win_row) else 0.0)
            if battle_count <= 0:
                continue
            total_battles += battle_count
            win_weighted_sum += win_rate * battle_count

        avg = (win_weighted_sum / total_battles) if total_battles > 0 else 0.0
        aggregates[model] = PairwiseAggregate(
            model_name=model,
            total_pairwise_battles=total_battles,
            average_pairwise_win_rate=avg,
        )

    return aggregates


def _find_pairwise_data(obj: Any, depth: int = 0) -> dict[str, Any] | None:
    """Find chart payload containing model labels + battle/win matrices."""
    if depth > 10:
        return None
    if isinstance(obj, dict):
        labels = obj.get("labels")
        battle = obj.get("battle_matrix") or obj.get("battleCountMatrix")
        win = obj.get("win_matrix") or obj.get("winFractionMatrix")
        if isinstance(labels, list) and isinstance(battle, list) and isinstance(win, list):
            return {"labels": labels, "battle_matrix": battle, "win_matrix": win}
        for value in obj.values():
            found = _find_pairwise_data(value, depth + 1)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_pairwise_data(value, depth + 1)
            if found is not None:
                return found
    return None


def _looks_like_csv(content: str) -> bool:
    """Heuristic: does the content look like CSV data?"""
    if not content or content.lstrip().startswith(("{", "[", "<", "!")):
        return False
    first_line = content.split("\n", 1)[0].lower()
    return "," in first_line and any(keyword in first_line for keyword in ("rank", "model", "score", "elo", "arena"))


# ── Parsing Helpers ───────────────────────────────────────────────────


def _extract_headers(table: Tag) -> list[str]:
    """Extract normalised header text from a table."""
    thead = table.find("thead")
    if thead and isinstance(thead, Tag):
        header_row = thead.find("tr")
        if isinstance(header_row, Tag):
            return [_clean_cell(th).lower() for th in header_row.find_all(["th", "td"])]

    # Fallback: first row of the table
    first_row = table.find("tr")
    if first_row:
        return [_clean_cell(th).lower() for th in first_row.find_all(["th", "td"])]  # type: ignore[union-attr]
    return []


_COLUMN_ALIASES: dict[str, list[str]] = {
    "name": ["model", "model name", "name", "model_name", "full_name"],
    "rank": ["rank", "#", "rank (ub)", "rank(ub)", "ranking"],
    "rank_ub": ["rank (ub)", "rank(ub)", "rank_ub", "upper bound"],
    "score": ["arena score", "score", "elo", "rating", "elo rating", "arena elo"],
    "ci": ["95% ci", "ci", "confidence interval", "95% confidence interval", "ci (95%)"],
    "votes": ["votes", "battles", "num battles", "num_battles", "total votes"],
    "org": ["organization", "org", "provider", "developer", "company"],
}


def _build_column_map(headers: list[str]) -> dict[str, int]:
    """Map semantic column names to header indices."""
    col_map: dict[str, int] = {}
    for semantic, aliases in _COLUMN_ALIASES.items():
        for i, header in enumerate(headers):
            if header in aliases or any(alias in header for alias in aliases):
                col_map[semantic] = i
                break
    return col_map


def _clean_cell(cell: Tag) -> str:
    """Extract clean text from a table cell, stripping whitespace and special chars."""
    text = cell.get_text(strip=True)
    # Remove common decorative characters
    text = text.replace("\u200b", "").replace("\xa0", " ").strip()
    return text


def _parse_rank_range(text: str) -> tuple[int, int]:
    """Parse a rank that may be a range like '1-3' or a single number.

    Returns (rank_ub, rank_lb).  For a range '1-3': rank_lb=1, rank_ub=3.
    """
    if not text:
        return 0, 0
    text = text.strip()
    m = re.match(r"(\d+)\s*[-–—]\s*(\d+)", text)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return max(a, b), min(a, b)  # ub, lb
    try:
        val = int(text)
        return val, val
    except ValueError:
        return 0, 0


def _parse_ci(text: str) -> tuple[float, float]:
    """Parse a confidence interval string like '+5/-4' or '±5' or '1280-1290'.

    Returns (ci_lower, ci_upper) as absolute values or deltas.
    """
    if not text:
        return 0.0, 0.0
    text = text.strip()

    # Format: +X/-Y
    m = re.match(r"[+]?\s*(\d+(?:\.\d+)?)\s*/\s*[-]?\s*(\d+(?:\.\d+)?)", text)
    if m:
        return -float(m.group(2)), float(m.group(1))

    # Format: ±X
    m = re.match(r"[±]\s*(\d+(?:\.\d+)?)", text)
    if m:
        val = float(m.group(1))
        return -val, val

    # Format: X-Y (absolute range)
    m = re.match(r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)", text)
    if m:
        return float(m.group(1)), float(m.group(2))

    return 0.0, 0.0


def _safe_int(value: Any) -> int:
    """Safely convert a value to int."""
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        # Remove commas, whitespace
        cleaned = value.replace(",", "").replace(" ", "").strip()
        try:
            return int(float(cleaned)) if cleaned else 0
        except (ValueError, OverflowError):
            return 0
    return 0


def _safe_float(value: Any) -> float:
    """Safely convert a value to float."""
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.replace(",", "").replace(" ", "").strip()
        try:
            return float(cleaned) if cleaned else 0.0
        except (ValueError, OverflowError):
            return 0.0
    return 0.0
