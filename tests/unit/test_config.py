"""Unit tests for configuration loading and validation."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from autotrader.config.loader import _apply_env_overrides, _deep_merge, load_config
from autotrader.config.models import AppConfig, ExecutionMode, LoggingConfig


class TestDeepMerge:
    def test_flat_merge(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        assert _deep_merge(base, override) == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"x": {"a": 1, "b": 2}}
        override = {"x": {"b": 3, "c": 4}}
        assert _deep_merge(base, override) == {"x": {"a": 1, "b": 3, "c": 4}}

    def test_override_replaces_non_dict(self) -> None:
        base = {"x": {"a": 1}}
        override = {"x": "flat"}
        assert _deep_merge(base, override) == {"x": "flat"}

    def test_empty_override(self) -> None:
        base = {"a": 1}
        assert _deep_merge(base, {}) == {"a": 1}


class TestEnvOverrides:
    def test_simple_override(self) -> None:
        data: dict = {"kalshi": {"execution_mode": "paper"}}
        with patch.dict(os.environ, {"AUTOTRADER__KALSHI__EXECUTION_MODE": "live"}):
            result = _apply_env_overrides(data)
        assert result["kalshi"]["execution_mode"] == "live"

    def test_creates_nested_keys(self) -> None:
        data: dict = {}
        with patch.dict(os.environ, {"AUTOTRADER__LOGGING__LEVEL": "DEBUG"}):
            result = _apply_env_overrides(data)
        assert result["logging"]["level"] == "DEBUG"

    def test_ignores_non_prefixed(self) -> None:
        data: dict = {"a": 1}
        with patch.dict(os.environ, {"OTHER_VAR": "value"}, clear=False):
            result = _apply_env_overrides(data)
        assert "other_var" not in result

    def test_legacy_alias_execution_mode(self) -> None:
        data: dict = {"kalshi": {"execution_mode": "paper"}}
        with patch.dict(
            os.environ,
            {"EXECUTION_MODE": "live"},
            clear=True,
        ):
            result = _apply_env_overrides(data)
        assert result["kalshi"]["execution_mode"] == "live"

    def test_namespaced_keys_take_precedence_over_legacy_aliases(self) -> None:
        data: dict = {"kalshi": {"execution_mode": "paper"}}
        with patch.dict(
            os.environ,
            {
                "EXECUTION_MODE": "live",
                "AUTOTRADER__KALSHI__EXECUTION_MODE": "paper",
            },
            clear=True,
        ):
            result = _apply_env_overrides(data)
        assert result["kalshi"]["execution_mode"] == "paper"


class TestAppConfig:
    def test_defaults(self) -> None:
        config = AppConfig()
        assert config.kalshi.execution_mode == ExecutionMode.PAPER
        assert config.risk.global_config.max_portfolio_exposure_pct == 0.60
        assert config.logging.level == "INFO"

    def test_logging_level_validation(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            LoggingConfig(level="INVALID")

    def test_from_dict(self) -> None:
        data = {
            "kalshi": {"execution_mode": "live"},
            "logging": {"level": "DEBUG"},
        }
        config = AppConfig.from_dict(data)
        assert config.kalshi.execution_mode == ExecutionMode.LIVE
        assert config.logging.level == "DEBUG"


class TestLoadConfig:
    def test_load_from_empty_dir(self, tmp_path: Path) -> None:
        """Loading from an empty config dir should return defaults."""
        config = load_config(config_dir=tmp_path)
        assert config.kalshi.execution_mode == ExecutionMode.PAPER

    def test_load_with_base_yaml(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(
            yaml.dump(
                {
                    "kalshi": {"execution_mode": "paper"},
                    "logging": {"level": "DEBUG"},
                }
            )
        )
        config = load_config(config_dir=tmp_path)
        assert config.logging.level == "DEBUG"

    def test_paper_overlay(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"logging": {"level": "INFO"}}))
        paper = tmp_path / "paper.yaml"
        paper.write_text(yaml.dump({"logging": {"level": "DEBUG"}}))

        config = load_config(config_dir=tmp_path, execution_mode="paper")
        assert config.logging.level == "DEBUG"

    def test_live_overlay(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"kalshi": {"execution_mode": "paper"}, "logging": {"level": "DEBUG"}}))
        live = tmp_path / "live.yaml"
        live.write_text(yaml.dump({"kalshi": {"execution_mode": "live"}, "logging": {"level": "INFO"}}))

        config = load_config(config_dir=tmp_path, execution_mode="live")
        assert config.kalshi.execution_mode == ExecutionMode.LIVE
        assert config.logging.level == "INFO"

    def test_environment_variable_overrides_live_overlay(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"kalshi": {"execution_mode": "live"}, "logging": {"level": "INFO"}}))
        live = tmp_path / "live.yaml"
        live.write_text(yaml.dump({"logging": {"level": "WARNING"}}))

        with patch.dict(
            os.environ,
            {
                "AUTOTRADER__KALSHI__EXECUTION_MODE": "paper",
                "AUTOTRADER__LOGGING__LEVEL": "DEBUG",
            },
            clear=False,
        ):
            config = load_config(config_dir=tmp_path, execution_mode="live")

        assert config.kalshi.execution_mode == ExecutionMode.PAPER
        assert config.logging.level == "DEBUG"

    def test_legacy_execution_mode_selects_live_overlay(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(yaml.dump({"logging": {"level": "DEBUG"}}))
        live = tmp_path / "live.yaml"
        live.write_text(yaml.dump({"logging": {"level": "WARNING"}}))

        with patch.dict(os.environ, {"EXECUTION_MODE": "live"}, clear=True):
            config = load_config(config_dir=tmp_path)

        assert config.logging.level == "WARNING"

    def test_risk_yaml(self, tmp_path: Path) -> None:
        risk = tmp_path / "risk.yaml"
        risk.write_text(
            yaml.dump(
                {
                    "global": {"max_portfolio_exposure_pct": 0.40},
                }
            )
        )
        config = load_config(config_dir=tmp_path)
        assert config.risk.global_config.max_portfolio_exposure_pct == 0.40

    def test_strategy_yaml(self, tmp_path: Path) -> None:
        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()
        la = strategies_dir / "leaderboard_alpha.yaml"
        la.write_text(
            yaml.dump(
                {
                    "leaderboard_alpha": {
                        "min_edge_after_fees_cents": 5,
                        "model_ticker_overrides": {"GPT-5": "KXTOPMODEL-GPT5"},
                    }
                }
            )
        )
        config = load_config(config_dir=tmp_path)
        assert config.leaderboard_alpha.min_edge_after_fees_cents == 5
        assert config.leaderboard_alpha.model_ticker_overrides["GPT-5"] == "KXTOPMODEL-GPT5"

    def test_strategy_yaml_null_override_maps(self, tmp_path: Path) -> None:
        strategies_dir = tmp_path / "strategies"
        strategies_dir.mkdir()
        la = strategies_dir / "leaderboard_alpha.yaml"
        la.write_text(
            yaml.dump(
                {
                    "leaderboard_alpha": {
                        "model_ticker_overrides": None,
                        "org_ticker_overrides": None,
                    }
                }
            )
        )

        config = load_config(config_dir=tmp_path)

        assert config.leaderboard_alpha.model_ticker_overrides == {}
        assert config.leaderboard_alpha.org_ticker_overrides == {}

    def test_signal_source_yaml_with_top_level_key(self, tmp_path: Path) -> None:
        signal_sources_dir = tmp_path / "signal_sources"
        signal_sources_dir.mkdir()
        arena_monitor = signal_sources_dir / "arena-monitor.yaml"
        arena_monitor.write_text(
            yaml.dump(
                {
                    "arena_monitor": {
                        "poll_interval_seconds": 45,
                        "signal_rank_cutoff": 12,
                    }
                }
            )
        )

        config = load_config(config_dir=tmp_path)

        assert config.arena_monitor.poll_interval_seconds == 45
        assert config.arena_monitor.signal_rank_cutoff == 12

    def test_signal_source_yaml_with_flat_fields(self, tmp_path: Path) -> None:
        signal_sources_dir = tmp_path / "signal_sources"
        signal_sources_dir.mkdir()
        arena_monitor = signal_sources_dir / "arena-monitor.yaml"
        arena_monitor.write_text(
            yaml.dump(
                {
                    "poll_interval_seconds": 60,
                    "request_timeout_seconds": 8.0,
                }
            )
        )

        config = load_config(config_dir=tmp_path)

        assert config.arena_monitor.poll_interval_seconds == 60
        assert config.arena_monitor.request_timeout_seconds == 8.0
