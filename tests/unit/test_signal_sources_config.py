"""Tests for signal source config loading variants."""

from pathlib import Path

import yaml

from autotrader.config.loader import load_config


def test_signal_source_yaml_with_wrapped_key(tmp_path: Path) -> None:
    signals_dir = tmp_path / "signal_sources"
    signals_dir.mkdir()
    arena = signals_dir / "arena_monitor.yaml"
    arena.write_text(yaml.dump({"arena_monitor": {"poll_interval_seconds": 12}}))

    config = load_config(config_dir=tmp_path)
    assert config.arena_monitor.poll_interval_seconds == 12


def test_signal_source_yaml_with_flat_payload(tmp_path: Path) -> None:
    signals_dir = tmp_path / "signal_sources"
    signals_dir.mkdir()
    arena = signals_dir / "arena_monitor.yaml"
    arena.write_text(yaml.dump({"poll_interval_seconds": 17}))

    config = load_config(config_dir=tmp_path)
    assert config.arena_monitor.poll_interval_seconds == 17
