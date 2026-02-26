"""Unit tests for model name fuzzy matching."""

from autotrader.utils.matching import fuzzy_match, normalize_model_name


class TestNormalize:
    def test_lowercase(self) -> None:
        assert normalize_model_name("Claude-Opus-4-6") == "claude-opus-4-6"

    def test_strip_whitespace(self) -> None:
        assert normalize_model_name("  gpt-5  ") == "gpt-5"

    def test_underscores_to_hyphens(self) -> None:
        assert normalize_model_name("gpt_5_turbo") == "gpt-5-turbo"

    def test_spaces_to_hyphens(self) -> None:
        assert normalize_model_name("gemini 3 pro") == "gemini-3-pro"


class TestExactMatch:
    def test_exact_match(self) -> None:
        candidates = ["claude-opus-4-6", "gemini-3-pro", "gpt-5"]
        result = fuzzy_match("claude-opus-4-6", candidates)
        assert result is not None
        assert result.matched == "claude-opus-4-6"
        assert result.score == 1.0
        assert not result.is_override

    def test_case_insensitive_exact(self) -> None:
        candidates = ["claude-opus-4-6", "gemini-3-pro"]
        result = fuzzy_match("Claude-Opus-4-6", candidates)
        assert result is not None
        assert result.matched == "claude-opus-4-6"
        assert result.score == 1.0


class TestOverrides:
    def test_override_match(self) -> None:
        candidates = ["gemini-3.1-pro", "gemini-3-pro"]
        overrides = {"gemini-3.1-pro-preview": "gemini-3.1-pro"}
        result = fuzzy_match("gemini-3.1-pro-preview", candidates, overrides=overrides)
        assert result is not None
        assert result.matched == "gemini-3.1-pro"
        assert result.is_override

    def test_override_target_not_in_candidates(self) -> None:
        candidates = ["gemini-3-pro"]
        # Override target not found in candidates, falls back to fuzzy
        result = fuzzy_match(
            "gemini-3.1-pro-preview",
            candidates,
            threshold=0.5,
            overrides={"gemini-3.1-pro-preview": "nonexistent-model"},
        )
        assert result is not None
        assert not result.is_override


class TestFuzzyMatch:
    def test_close_match(self) -> None:
        candidates = ["claude-opus-4-6", "gemini-3-pro", "gpt-5.2"]
        result = fuzzy_match("claude-opus-4.6", candidates, threshold=0.8)
        assert result is not None
        assert result.matched == "claude-opus-4-6"
        assert result.score >= 0.8

    def test_no_match_below_threshold(self) -> None:
        candidates = ["completely-different-model"]
        result = fuzzy_match("claude-opus-4-6", candidates, threshold=0.8)
        assert result is None

    def test_best_match_selected(self) -> None:
        candidates = ["gpt-5", "gpt-5.2", "gpt-5.2-turbo"]
        result = fuzzy_match("gpt-5.2", candidates, threshold=0.8)
        assert result is not None
        assert result.matched == "gpt-5.2"


class TestKnownMismatches:
    """Test matching for known Arena ↔ Kalshi naming discrepancies."""

    def test_thinking_model_variant(self) -> None:
        candidates = ["claude-opus-4-6-thinking", "claude-opus-4-6"]
        result = fuzzy_match("claude-opus-4-6-thinking", candidates)
        assert result is not None
        assert result.matched == "claude-opus-4-6-thinking"

    def test_preview_suffix(self) -> None:
        candidates = ["gemini-3.1-pro"]
        overrides = {"gemini-3.1-pro-preview": "gemini-3.1-pro"}
        result = fuzzy_match("gemini-3.1-pro-preview", candidates, overrides=overrides)
        assert result is not None
        assert result.matched == "gemini-3.1-pro"

    def test_beta_suffix(self) -> None:
        candidates = ["grok-4.20", "grok-4.20-beta1"]
        result = fuzzy_match("grok-4.20-beta1", candidates)
        assert result is not None
        assert result.matched == "grok-4.20-beta1"
