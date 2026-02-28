"""Unit tests for the Arena leaderboard parser."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autotrader.signals.arena_parser import (
    _looks_like_csv,
    _parse_ci,
    _parse_rank_range,
    _safe_float,
    _safe_int,
    extract_next_data,
    extract_pairwise_aggregates,
    parse_csv,
    parse_html_table,
    parse_json_entries,
    parse_leaderboard,
)

FIXTURES = Path(__file__).parent.parent / "fixtures"


# ── Fixture Loading ───────────────────────────────────────────────────


@pytest.fixture
def html_leaderboard() -> str:
    return (FIXTURES / "arena_leaderboard.html").read_text()


@pytest.fixture
def json_leaderboard() -> str:
    return (FIXTURES / "arena_leaderboard.json").read_text()


@pytest.fixture
def json_entries() -> list[dict[str, object]]:
    return json.loads((FIXTURES / "arena_leaderboard.json").read_text())


@pytest.fixture
def next_data_html() -> str:
    return (FIXTURES / "arena_next_data.html").read_text()


@pytest.fixture
def csv_leaderboard() -> str:
    return (FIXTURES / "arena_leaderboard.csv").read_text()


# ── CSV Parsing ───────────────────────────────────────────────────────


class TestParseCsv:
    def test_basic_csv_parsing(self, csv_leaderboard: str) -> None:
        entries = parse_csv(csv_leaderboard)
        assert len(entries) == 8
        assert entries[0].model_name == "Claude Opus 4.6"
        assert entries[0].organization == "Anthropic"
        assert entries[0].score == 1350.0
        assert entries[0].votes == 25000

    def test_rank_stylectrl_is_rank_ub(self, csv_leaderboard: str) -> None:
        """rank_stylectrl column should be used as rank_ub."""
        entries = parse_csv(csv_leaderboard)
        # Claude: rank=1, rank_stylectrl=1
        assert entries[0].rank == 1
        assert entries[0].rank_ub == 1
        # Gemini: rank=3, rank_stylectrl=4
        gemini = next(e for e in entries if e.model_name == "Gemini 3.0 Ultra")
        assert gemini.rank == 3
        assert gemini.rank_ub == 4

    def test_ci_from_csv(self, csv_leaderboard: str) -> None:
        entries = parse_csv(csv_leaderboard)
        # Claude: +3/-4
        assert entries[0].ci_upper == 3.0
        assert entries[0].ci_lower == -4.0

    def test_preliminary_detection(self, csv_leaderboard: str) -> None:
        entries = parse_csv(csv_leaderboard)
        preview = next(e for e in entries if e.model_name == "NewModel-Preview")
        assert preview.is_preliminary is True
        assert entries[0].is_preliminary is False

    def test_empty_csv(self) -> None:
        assert parse_csv("") == []

    def test_header_only_csv(self) -> None:
        assert parse_csv("rank,model,score\n") == []

    def test_minimal_csv(self) -> None:
        csv_data = "rank,rank_stylectrl,model,arena_score,votes,organization\n1,1,TestModel,1300,5000,TestOrg\n"
        entries = parse_csv(csv_data)
        assert len(entries) == 1
        assert entries[0].model_name == "TestModel"
        assert entries[0].rank_ub == 1
        assert entries[0].score == 1300.0
        assert entries[0].votes == 5000
        assert entries[0].organization == "TestOrg"


class TestLooksLikeCsv:
    def test_csv_header(self) -> None:
        assert _looks_like_csv("rank,model,score\n1,Claude,1350\n") is True

    def test_json_not_csv(self) -> None:
        assert _looks_like_csv('[{"model": "Claude"}]') is False

    def test_html_not_csv(self) -> None:
        assert _looks_like_csv("<html><body>") is False

    def test_empty_not_csv(self) -> None:
        assert _looks_like_csv("") is False

    def test_plain_text_without_keywords(self) -> None:
        assert _looks_like_csv("hello,world,foo") is False


# ── JSON Parsing ──────────────────────────────────────────────────────


class TestParseJsonEntries:
    def test_basic_parsing(self, json_entries: list[dict[str, object]]) -> None:
        entries = parse_json_entries(json_entries)
        assert len(entries) == 8
        assert entries[0].model_name == "Claude Opus 4.6"
        assert entries[0].organization == "Anthropic"
        assert entries[0].rank == 1
        assert entries[0].rank_ub == 1
        assert entries[0].score == 1350.0

    def test_preliminary_detection(self, json_entries: list[dict[str, object]]) -> None:
        entries = parse_json_entries(json_entries)
        # NewModel-Preview has only 200 votes (< 500 threshold)
        preview = next(e for e in entries if e.model_name == "NewModel-Preview")
        assert preview.is_preliminary is True
        # Claude Opus has 25000 votes
        claude = next(e for e in entries if e.model_name == "Claude Opus 4.6")
        assert claude.is_preliminary is False

    def test_rank_bounds(self, json_entries: list[dict[str, object]]) -> None:
        entries = parse_json_entries(json_entries)
        gemini = next(e for e in entries if e.model_name == "Gemini 3.0 Ultra")
        assert gemini.rank_ub == 4
        assert gemini.rank_lb == 2

    def test_confidence_intervals(self, json_entries: list[dict[str, object]]) -> None:
        entries = parse_json_entries(json_entries)
        gpt = next(e for e in entries if e.model_name == "GPT-5")
        assert gpt.ci_lower == 1337.0
        assert gpt.ci_upper == 1344.0

    def test_empty_list(self) -> None:
        assert parse_json_entries([]) == []

    def test_skip_entries_without_name(self) -> None:
        data = [{"rank": 1, "score": 1000}]
        assert parse_json_entries(data) == []

    def test_alternative_field_names(self) -> None:
        """Should handle different key naming conventions."""
        data = [
            {
                "model": "TestModel",
                "org": "TestOrg",
                "rating": 1300,
                "num_battles": 5000,
                "rank": 1,
                "rank_ub": 1,
            }
        ]
        entries = parse_json_entries(data)
        assert len(entries) == 1
        assert entries[0].model_name == "TestModel"
        assert entries[0].organization == "TestOrg"
        assert entries[0].score == 1300.0
        assert entries[0].votes == 5000


# ── __NEXT_DATA__ Extraction ──────────────────────────────────────────


class TestExtractNextData:
    def test_extracts_from_next_data(self, next_data_html: str) -> None:
        data = extract_next_data(next_data_html)
        assert data is not None
        assert len(data) == 3
        assert data[0]["model_name"] == "Claude Opus 4.6"

    def test_returns_none_without_script_tag(self) -> None:
        html = "<html><body>No data here</body></html>"
        assert extract_next_data(html) is None

    def test_returns_none_for_invalid_json(self) -> None:
        html = '<script id="__NEXT_DATA__">not valid json</script>'
        assert extract_next_data(html) is None

    def test_returns_none_for_unrelated_data(self) -> None:
        html = '<script id="__NEXT_DATA__">{"props": {"pageProps": {"unrelated": true}}}</script>'
        assert extract_next_data(html) is None


# ── HTML Table Parsing ────────────────────────────────────────────────


class TestParseHtmlTable:
    def test_basic_table_parsing(self, html_leaderboard: str) -> None:
        entries = parse_html_table(html_leaderboard)
        assert len(entries) == 8
        assert entries[0].model_name == "Claude Opus 4.6"
        assert entries[0].organization == "Anthropic"
        assert entries[0].score == 1350.0
        assert entries[0].votes == 25000

    def test_rank_parsing(self, html_leaderboard: str) -> None:
        entries = parse_html_table(html_leaderboard)
        assert entries[0].rank == 1
        assert entries[4].rank == 5

    def test_preliminary_by_votes(self, html_leaderboard: str) -> None:
        entries = parse_html_table(html_leaderboard)
        preview = next(e for e in entries if e.model_name == "NewModel-Preview")
        assert preview.is_preliminary is True
        assert preview.votes == 200

    def test_no_tables(self) -> None:
        html = "<html><body>No tables</body></html>"
        assert parse_html_table(html) == []

    def test_empty_table(self) -> None:
        html = "<table><thead><tr><th>Rank</th><th>Model</th></tr></thead></table>"
        assert parse_html_table(html) == []

    def test_table_with_rank_range(self) -> None:
        html = """
        <table>
            <thead><tr><th>Rank</th><th>Model</th><th>Arena Score</th></tr></thead>
            <tbody>
                <tr><td>1-3</td><td>SomeModel</td><td>1300</td></tr>
            </tbody>
        </table>
        """
        entries = parse_html_table(html)
        assert len(entries) == 1
        assert entries[0].rank_ub == 3
        assert entries[0].rank_lb == 1

    def test_ci_parsing_in_table(self, html_leaderboard: str) -> None:
        entries = parse_html_table(html_leaderboard)
        # Claude Opus: +3/-4
        assert entries[0].ci_upper == 3.0
        assert entries[0].ci_lower == -4.0


# ── Unified parse_leaderboard ─────────────────────────────────────────


class TestParseLeaderboard:
    def test_parses_csv(self, csv_leaderboard: str) -> None:
        entries = parse_leaderboard(csv_leaderboard)
        assert len(entries) == 8
        assert entries[0].model_name == "Claude Opus 4.6"
        assert entries[0].rank_ub == 1

    def test_parses_json(self, json_leaderboard: str) -> None:
        entries = parse_leaderboard(json_leaderboard)
        assert len(entries) == 8
        assert entries[0].model_name == "Claude Opus 4.6"

    def test_parses_next_data(self, next_data_html: str) -> None:
        entries = parse_leaderboard(next_data_html)
        assert len(entries) == 3

    def test_parses_html_table(self, html_leaderboard: str) -> None:
        entries = parse_leaderboard(html_leaderboard)
        assert len(entries) == 8

    def test_json_wrapped_in_dict(self) -> None:
        data = json.dumps({"data": [{"model_name": "Test", "rank": 1, "arena_score": 1000, "votes": 5000}]})
        entries = parse_leaderboard(data)
        assert len(entries) == 1

    def test_empty_content(self) -> None:
        assert parse_leaderboard("") == []

    def test_garbage_content(self) -> None:
        assert parse_leaderboard("this is not data at all") == []


# ── Parsing Helpers ───────────────────────────────────────────────────


class TestParseRankRange:
    def test_single_number(self) -> None:
        assert _parse_rank_range("1") == (1, 1)

    def test_range_dash(self) -> None:
        assert _parse_rank_range("1-3") == (3, 1)

    def test_range_en_dash(self) -> None:
        assert _parse_rank_range("2\u20134") == (4, 2)

    def test_range_with_spaces(self) -> None:
        assert _parse_rank_range("1 - 5") == (5, 1)

    def test_empty(self) -> None:
        assert _parse_rank_range("") == (0, 0)

    def test_non_numeric(self) -> None:
        assert _parse_rank_range("abc") == (0, 0)


class TestParseCi:
    def test_plus_minus_format(self) -> None:
        assert _parse_ci("+5/-4") == (-4.0, 5.0)

    def test_plus_minus_symbol(self) -> None:
        assert _parse_ci("±5") == (-5.0, 5.0)

    def test_range_format(self) -> None:
        assert _parse_ci("1280-1290") == (1280.0, 1290.0)

    def test_empty(self) -> None:
        assert _parse_ci("") == (0.0, 0.0)


class TestSafeInt:
    def test_int(self) -> None:
        assert _safe_int(42) == 42

    def test_float(self) -> None:
        assert _safe_int(3.7) == 3

    def test_string(self) -> None:
        assert _safe_int("100") == 100

    def test_string_with_commas(self) -> None:
        assert _safe_int("1,000") == 1000

    def test_empty_string(self) -> None:
        assert _safe_int("") == 0

    def test_none(self) -> None:
        assert _safe_int(None) == 0


class TestSafeFloat:
    def test_float(self) -> None:
        assert _safe_float(3.14) == 3.14

    def test_int(self) -> None:
        assert _safe_float(42) == 42.0

    def test_string(self) -> None:
        assert _safe_float("1350.5") == 1350.5

    def test_empty_string(self) -> None:
        assert _safe_float("") == 0.0

    def test_none(self) -> None:
        assert _safe_float(None) == 0.0


class TestPairwiseExtraction:
    def test_extract_pairwise_from_next_payload_html(self) -> None:
        html = """
        <html><body><script id="__NEXT_DATA__" type="application/json">
        {
          "props": {
            "pageProps": {
              "charts": {
                "pairwise": {
                  "labels": ["A", "B"],
                  "battle_matrix": [[0, 200], [200, 0]],
                  "win_matrix": [[0.5, 0.6], [0.4, 0.5]]
                }
              }
            }
          }
        }
        </script></body></html>
        """
        agg = extract_pairwise_aggregates(html)
        assert agg["A"].total_pairwise_battles == 200
        assert agg["A"].average_pairwise_win_rate == pytest.approx(0.6)

    def test_extract_pairwise_aggregates(self) -> None:
        payload = {
            "charts": {
                "pairwise": {
                    "labels": ["A", "B", "C"],
                    "battle_matrix": [
                        [0, 100, 50],
                        [100, 0, 25],
                        [50, 25, 0],
                    ],
                    "win_matrix": [
                        [0.5, 0.55, 0.60],
                        [0.45, 0.5, 0.52],
                        [0.40, 0.48, 0.5],
                    ],
                }
            }
        }
        agg = extract_pairwise_aggregates(payload)
        assert set(agg.keys()) == {"A", "B", "C"}
        assert agg["A"].total_pairwise_battles == 150
        assert agg["A"].average_pairwise_win_rate == pytest.approx((0.55 * 100 + 0.60 * 50) / 150)

    def test_release_date_field_is_parsed(self) -> None:
        data = [
            {
                "model_name": "M1",
                "rank": 1,
                "rank_ub": 1,
                "arena_score": 1500,
                "votes": 1000,
                "release_date": "2025-01-01",
            }
        ]
        entries = parse_json_entries(data)
        assert entries[0].release_date == "2025-01-01"
