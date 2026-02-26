"""Unit tests for configuration loading and validation."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from autotrader.config.loader import _apply_env_overrides, _deep_merge, load_config
from autotrader.config.models import AppConfig, Environment, LoggingConfig


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
        data: dict = {"kalshi": {"environment": "demo"}}
        with patch.dict(os.environ, {"AUTOTRADER__KALSHI__ENVIRONMENT": "production"}):
            result = _apply_env_overrides(data)
        assert result["kalshi"]["environment"] == "production"

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


class TestAppConfig:
    def test_defaults(self) -> None:
        config = AppConfig()
        assert config.kalshi.environment == Environment.DEMO
        assert config.risk.global_config.max_portfolio_exposure_pct == 0.60
        assert config.logging.level == "INFO"

    def test_logging_level_validation(self) -> None:
        with pytest.raises(ValueError, match="Invalid log level"):
            LoggingConfig(level="INVALID")

    def test_from_dict(self) -> None:
        data = {
            "kalshi": {"environment": "production"},
            "logging": {"level": "DEBUG"},
        }
        config = AppConfig.from_dict(data)
        assert config.kalshi.environment == Environment.PRODUCTION
        assert config.logging.level == "DEBUG"


class TestLoadConfig:
    def test_load_from_empty_dir(self, tmp_path: Path) -> None:
        """Loading from an empty config dir should return defaults."""
        config = load_config(config_dir=tmp_path)
        assert config.kalshi.environment == Environment.DEMO

    def test_load_with_base_yaml(self, tmp_path: Path) -> None:
        base = tmp_path / "base.yaml"
        base.write_text(
            yaml.dump(
                {
                    "kalshi": {"environment": "demo"},
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

        config = load_config(config_dir=tmp_path, environment="demo")
        assert config.logging.level == "DEBUG"

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
                    }
                }
            )
        )
        config = load_config(config_dir=tmp_path)
        assert config.leaderboard_alpha.min_edge_after_fees_cents == 5
