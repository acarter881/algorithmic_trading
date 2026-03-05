"""Configuration loading from YAML files with environment variable overrides."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from autotrader.config.models import AppConfig

_LEGACY_ENV_ALIASES = {
    "EXECUTION_MODE": "AUTOTRADER__KALSHI__EXECUTION_MODE",
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge override dict into base dict. Override values win."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file, returning empty dict if not found."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = yaml.safe_load(f)
    return data if isinstance(data, dict) else {}


def _apply_env_overrides(data: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides.

    Convention: AUTOTRADER__SECTION__KEY=value
    Double underscore separates nesting levels.
    Example: AUTOTRADER__KALSHI__EXECUTION_MODE=paper
    """
    prefix = "AUTOTRADER__"
    env_vars = dict(os.environ)

    # Backward compatibility for legacy flat environment variable names.
    for legacy_key, namespaced_key in _LEGACY_ENV_ALIASES.items():
        if namespaced_key not in env_vars and legacy_key in env_vars:
            env_vars[namespaced_key] = env_vars[legacy_key]

    for key, value in env_vars.items():
        if not key.startswith(prefix):
            continue
        parts = key[len(prefix) :].lower().split("__")
        current = data
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = value
    return data


def _normalize_leaderboard_alpha_config(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize leaderboard-alpha mapping fields loaded from YAML/env."""
    la = data.get("leaderboard_alpha")
    if not isinstance(la, dict):
        return data

    for key in ("model_ticker_overrides", "org_ticker_overrides", "model_aliases", "org_aliases"):
        if la.get(key) is None:
            la[key] = {}
    return data


def load_config(
    config_dir: Path | str = "config",
    execution_mode: str | None = None,
) -> AppConfig:
    """Load configuration from YAML files with environment overrides.

    Loading order (later values override earlier):
    1. config/base.yaml
    2. config/paper.yaml or config/live.yaml (selected by execution mode)
    3. config/strategies/*.yaml (merged under strategy keys)
    4. config/risk.yaml
    5. config/signal_sources/*.yaml
    6. Environment variables (AUTOTRADER__SECTION__KEY)

    The optional ``execution_mode`` argument (or ``AUTOTRADER__KALSHI__EXECUTION_MODE``)
    selects which overlay file is loaded in step 2.  Final environment
    variable overrides from step 6 still take precedence over every YAML file.
    """
    config_dir = Path(config_dir)

    # Determine execution mode for overlay selection
    env_vars = dict(os.environ)
    if "AUTOTRADER__KALSHI__EXECUTION_MODE" not in env_vars and "EXECUTION_MODE" in env_vars:
        env_vars["AUTOTRADER__KALSHI__EXECUTION_MODE"] = env_vars["EXECUTION_MODE"]
    mode = execution_mode or env_vars.get("AUTOTRADER__KALSHI__EXECUTION_MODE", "paper")

    # Layer 1: base config
    data = _load_yaml(config_dir / "base.yaml")

    # Layer 2: execution mode overlay
    overlay_file = "paper.yaml" if mode == "paper" else "live.yaml"
    overlay_data = _load_yaml(config_dir / overlay_file)
    data = _deep_merge(data, overlay_data)

    # Layer 3: strategy configs
    strategies_dir = config_dir / "strategies"
    if strategies_dir.exists():
        for yaml_file in sorted(strategies_dir.glob("*.yaml")):
            strategy_data = _load_yaml(yaml_file)
            stem = yaml_file.stem.replace("-", "_")
            if stem in strategy_data:
                data = _deep_merge(data, {stem: strategy_data[stem]})
            else:
                data = _deep_merge(data, {stem: strategy_data})

    # Layer 4: risk config
    risk_data = _load_yaml(config_dir / "risk.yaml")
    if risk_data:
        data = _deep_merge(data, {"risk": risk_data})

    # Layer 5: signal source configs
    signals_dir = config_dir / "signal_sources"
    if signals_dir.exists():
        for yaml_file in sorted(signals_dir.glob("*.yaml")):
            signal_data = _load_yaml(yaml_file)
            stem = yaml_file.stem.replace("-", "_")
            if stem in signal_data:
                data = _deep_merge(data, {stem: signal_data[stem]})
            else:
                data = _deep_merge(data, {stem: signal_data})

    # Layer 6: env var overrides
    data = _apply_env_overrides(data)
    data = _normalize_leaderboard_alpha_config(data)

    return AppConfig.from_dict(data)
